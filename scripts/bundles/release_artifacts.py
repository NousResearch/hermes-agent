"""Record, stage and promote the exact native release artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import subprocess
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from scripts.releases.stable import read_admitted_candidate, accepted_smoke_results, validate_candidates


def sha256_file(file: Path) -> str:
    with file.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stamp_matches(stamp: dict, tag: str, commit: str, *, channel_request: dict | None = None) -> None:
    if channel_request is not None and channel_request.get("receiverCandidate"):
        if (stamp.get("channelBuild") is not None or stamp.get("source") != "build"
                or stamp.get("receiverProtocol") != 1 or stamp.get("displayVersion") != channel_request["version"]
                or stamp.get("commit") != commit or stamp.get("tag") != channel_request["releaseTag"]
                or stamp.get("baseVersion") != channel_request["sourceVersion"]):
            raise ValueError("Receiver candidate must use ordinary stable update ownership")
        return
    if channel_request is not None:
        if (stamp.get("source") != "channel-build" or stamp.get("channelBuild") != channel_request
                or stamp.get("commit") != channel_request["commit"] or stamp.get("tag")):
            raise ValueError("Built package provenance does not match the channel request")
        return
    if (stamp.get("commit") != commit or stamp.get("tag") != tag
            or stamp.get("baseVersion") != tag.removeprefix("v")):
        raise ValueError("Built package provenance does not match the release")


def single(items):
    items = list(items)
    if len(items) != 1:
        raise ValueError(f"Expected exactly one artifact, found {items}")
    return items[0]


def desktop_application(manifest: ET.Element, channel_request: dict | None) -> ET.Element:
    # CLI aliases are separate hidden Applications, not extra desktop entries.
    applications = manifest.findall("{*}Applications/{*}Application")
    visible = [app for app in applications if (
        (visual := app.find("{*}VisualElements")) is None or visual.get("AppListEntry") != "none")]
    application = single(visible)
    if channel_request is not None:
        expected = channel_request["identity"]["appNamePascal"]
        if application.get("Id") != expected or sum(app.get("Id") == expected for app in applications) != 1:
            raise ValueError("Desktop application ID differs from channel request")
    return application


def record(platform: str, arch: str, root: Path, tag: str, commit: str, out: Path,
           *, channel_request: dict | None = None) -> None:
    """Read identities from the built packages, never from the workflow matrix."""
    row: dict = {"platform": platform, "arch": arch, "tag": tag, "commit": commit}
    if platform == "windows":
        package = single(p for p in root.glob(f"*-win-{arch}.msix") if not p.name.startswith("Store-"))
        if channel_request is None and not package.name.endswith(f"-{tag[1:]}-win-{arch}.msix"):
            raise ValueError("Windows artifact filename differs from release tag")
        with zipfile.ZipFile(package) as archive:
            manifest = ET.fromstring(archive.read("AppxManifest.xml"))
            identity = manifest.find("{*}Identity")
            application = desktop_application(manifest, channel_request)
            stamp_name = single(n for n in archive.namelist() if n.replace("\\", "/").endswith("/resources/install-stamp.json"))
            stamp = json.loads(archive.read(stamp_name))
            stamp_matches(stamp, tag, commit, channel_request=channel_request)
        if identity.attrib["ProcessorArchitecture"].lower() != arch:
            raise ValueError("MSIX architecture differs from release target")
        row.update(identity=identity.attrib["Name"], publisher=identity.attrib["Publisher"],
                   applicationId=application.attrib["Id"], version=identity.attrib["Version"])
        version_sidecar = root / f"version-info-{arch}.json"
        if version_sidecar.is_file():
            row["executableVersion"] = json.loads(
                version_sidecar.read_text(encoding="utf-8-sig"),
            )["productVersion"]
        elif os.name == "nt":
            executable_name = (stamp.get("identity") or {}).get("windowsExecutableName")
            executables = list(root.rglob(f"{executable_name}.exe")) if executable_name else []
            executable = single(executables)
            row["executableVersion"] = subprocess.check_output([
                "powershell", "-NoProfile", "-Command",
                "(Get-Item -LiteralPath $args[0]).VersionInfo.ProductVersion", str(executable),
            ], text=True, encoding="utf-8", stdin=subprocess.DEVNULL).strip()
        if stamp.get("receiverProtocol") == 1:
            row["receiverProtocol"] = 1
    elif platform == "macos":
        package = single(root.glob(f"*-mac-{arch}.zip"))
        if not package.name.endswith(f"-{tag[1:]}-mac-{arch}.zip"):
            raise ValueError("macOS artifact filename differs from release tag")
        app = single(root.glob("mac*/*.app"))
        subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True)
        signature = subprocess.run(["codesign", "-dv", "--verbose=4", str(app)], check=True, capture_output=True, text=True, encoding="utf-8")
        team = re.search(r"^TeamIdentifier=([A-Z0-9]{10})$", signature.stderr, re.M)
        if not team:
            raise ValueError("Signed app has no Developer ID team")
        with zipfile.ZipFile(package) as archive:
            info = plistlib.loads(archive.read(single(n for n in archive.namelist() if re.fullmatch(r"[^/]+\.app/Contents/Info.plist", n))))
            stamp = json.loads(archive.read(single(n for n in archive.namelist() if re.fullmatch(r"[^/]+\.app/Contents/Resources/install-stamp.json", n))))
        stamp_matches(stamp, tag, commit, channel_request=channel_request)
        if stamp.get("receiverProtocol") == 1:
            row["receiverProtocol"] = 1
        row.update(identity=info["CFBundleIdentifier"], teamId=team.group(1), version=info["CFBundleShortVersionString"], filename=package.name)
        if row["version"] != (channel_request["version"] if channel_request else tag[1:]):
            raise ValueError("App version differs from release tag")
    elif platform == "termux":
        package = single((root / "deb").glob("*.deb"))
        fields = subprocess.check_output(["dpkg-deb", "--field", str(package), "Package", "Version", "Architecture"], text=True, encoding="utf-8")
        parsed = dict(line.split(": ", 1) for line in fields.splitlines())
        row.update(identity=parsed["Package"], version=parsed["Version"], filename=package.relative_to(root).as_posix())
        if parsed["Version"] != f"{tag[1:]}-1":
            raise ValueError("Termux artifact version differs from release tag")
        if parsed["Architecture"] != "aarch64":
            raise ValueError("Wrong Termux package architecture")
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(["dpkg-deb", "--extract", str(package), temp], check=True)
            stamps = list(Path(temp).rglob("install-stamp.json"))
            if not any((data := json.loads(p.read_text(encoding="utf-8-sig"))).get("commit") == commit and data.get("tag") == tag for p in stamps):
                raise ValueError("Termux package has no matching provenance")
    else:
        raise ValueError("Unknown platform")
    if channel_request is not None:
        expected = channel_request["identity"]["msixAppIdWithOrg" if platform == "windows" else "appId"]
        version = channel_request["windowsVersion" if platform == "windows" else "version"]
        if row["identity"] != expected or row["version"] != version:
            raise ValueError("Native package identity/version differs from channel request")
        row["request"] = channel_request
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def validate_windows_bundle(bundle: Path, windows: list[dict]) -> None:
    if sorted(row["arch"] for row in windows) != ["arm64", "x64"]:
        raise ValueError("Windows metadata must include both architectures")
    for field in ("identity", "publisher", "version", "applicationId"):
        if not windows[0].get(field) or windows[0][field] != windows[1].get(field):
            raise ValueError(f"Windows packages disagree on {field}")
    with zipfile.ZipFile(bundle) as archive:
        manifest = ET.fromstring(archive.read("AppxMetadata/AppxBundleManifest.xml"))
    identity = manifest.find("{*}Identity")
    if identity is None or any(identity.get(attr) != windows[0][field] for attr, field in
                               (("Name", "identity"), ("Publisher", "publisher"), ("Version", "version"))):
        raise ValueError("Universal bundle identity does not match its packages")
    if sorted(p.get("Architecture", "") for p in manifest.findall("{*}Packages/{*}Package")
              if p.get("Type") == "application") != ["arm64", "x64"]:
        raise ValueError("Universal bundle must cover both architectures")


def assemble(root: Path, tag: str, commit: str, public_base: str, out: Path,
             *, smoke_results: dict, release_epoch: int, archive: str) -> dict:
    """Bind the native metadata to files already staged by their build jobs.

    `tag` stays the plain payload identity; `archive` is the attempt ref the
    manifest and every artifact URL are keyed under."""
    from scripts.releases.handoff import receipt_name, validate_receipt
    from scripts.releases.r2 import put, staging_key_for

    smoke_results = accepted_smoke_results(smoke_results)
    expected = ("win32-x64", "win32-arm64", "darwin-x64", "darwin-arm64", "termux", "windows-universal")
    by_name = {}
    for name in expected:
        file = root / receipt_name(name)
        if not file.is_file():
            raise ValueError(f"Missing candidate handoff: {name}")
        receipt = json.loads(file.read_text(encoding="utf-8-sig"))
        # Handoffs are staged under the archive ref; their bound tag is the
        # archive, not the plain payload tag.
        for row in validate_receipt(receipt, archive, commit, name):
            prior = by_name.get(row["path"])
            if prior is not None and prior != row:
                raise ValueError("Candidate handoffs disagree on file receipts")
            by_name[row["path"]] = row
    rows = []
    for name, receipt in sorted(by_name.items()):
        if name.startswith("metadata-") or name.endswith(".msixbundle"):
            file = root / name
            if not file.is_file() or sha256_file(file) != receipt["sha256"]:
                raise ValueError(f"Candidate local digest mismatch: {name}")
            if name.startswith("metadata-"):
                rows.append(json.loads(file.read_text(encoding="utf-8-sig")))
    universal_name = single(name for name in by_name if name.endswith(".msixbundle") and not name.startswith("Store-"))
    single(name for name in by_name if name.endswith(".msixbundle") and name.startswith("Store-"))
    windows = [r for r in rows if r["platform"] == "windows"]
    validate_windows_bundle(root / universal_name, windows)
    files = [{**row, "url": f"{public_base.rstrip('/')}/{staging_key_for(archive, name)}"}
             for name, row in sorted(by_name.items()) if not name.startswith("metadata-")]
    by_name = {item["path"]: item for item in files}
    packages = []
    for row in rows:
        stamp_matches(row, tag, commit)
        filename = universal_name if row["platform"] == "windows" else row["filename"]
        item = by_name[filename]
        packages.append({k: v for k, v in {**row, "artifact": {"url": item["url"], "sha256": item["sha256"]}}.items() if k != "filename"})
    result = {"schema": 2, "tag": tag, "commit": commit, "releaseEpoch": release_epoch,
              "archive": archive,
              "packages": packages, "files": files, "smoke_results": smoke_results}
    validate_candidates(result, tag, commit, public_base, release_epoch, archive=archive)
    if not any(row["platform"] == "termux" for row in packages):
        raise ValueError("Missing Termux candidate")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    put(tag=archive, key="release-candidates.json", file=out, immutable=True)
    return result


def materialize(manifest: dict, root: Path, *, public_base: str, store_only: bool = False) -> None:
    validate_candidates(manifest, manifest["tag"], manifest["commit"], public_base,
                        archive=manifest["archive"])
    files = manifest.get("files", [])
    if not files or len({item["path"] for item in files}) != len(files):
        raise ValueError("Missing or duplicate candidate file receipts")
    by_url = {item["url"]: item["sha256"] for item in files}
    if any(by_url.get(row["artifact"]["url"]) != row["artifact"]["sha256"] for row in manifest["packages"]):
        raise ValueError("Package receipts differ from candidate file receipts")
    selected = [item for item in files if item["path"].startswith("Store-") and item["path"].endswith(".msixbundle")] if store_only else files
    if store_only and len(selected) != 1:
        raise ValueError("Expected one Store candidate")
    for item in selected:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or any(c in item["path"] for c in "\\:%?#") or item["path"].startswith("/") or "//" in item["path"]:
            raise ValueError("Invalid artifact path")
        expected = f"{public_base.rstrip('/')}/releases/tag/{manifest['archive']}/{relative.as_posix()}"
        if item["url"] != expected or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("Invalid artifact URL or digest")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with urllib.request.urlopen(item["url"], timeout=600) as response, target.open("wb") as file:
            if not response.geturl().startswith("https://"):
                raise ValueError("Artifact redirected outside HTTPS")
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                file.write(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Artifact digest mismatch: {relative}")


def publish(manifest: dict, root: Path, public_base: str) -> None:
    """Verify versioned uploads and stage APT content-addressed files, not indexes."""
    from scripts.releases.r2 import put

    materialize(manifest, root, public_base=public_base)
    for file in (root / "apt").rglob("*"):
        rel = file.relative_to(root / "apt").as_posix()
        if file.is_file() and (rel.startswith("pool/") or "/by-hash/" in rel):
            put(tag=manifest["tag"], key=f"releases/termux/stable/{rel}", file=file, key_is_full=True, immutable=True)


def write_appinstaller(out: Path, *, identity: str, publisher: str, version: str,
                       self_uri: str, artifact_uri: str, update_policy: str = "automatic") -> None:
    """Serialize verified package facts; callers own channel/acceptance policy."""
    if not all((identity, publisher, version, self_uri, artifact_uri)):
        raise ValueError("Explicit identity, publisher, version, self URI and artifact URI are required")
    ns = "http://schemas.microsoft.com/appx/appinstaller/2017/2"
    ET.register_namespace("", ns)
    descriptor = ET.Element(f"{{{ns}}}AppInstaller", {"Uri": self_uri, "Version": version})
    ET.SubElement(descriptor, f"{{{ns}}}MainBundle", {
        "Name": identity, "Publisher": publisher, "Version": version, "Uri": artifact_uri,
    })
    if update_policy not in ("automatic", "pinned"):
        raise ValueError("Unknown App Installer update policy")
    if update_policy == "automatic":
        settings = ET.SubElement(descriptor, f"{{{ns}}}UpdateSettings")
        ET.SubElement(settings, f"{{{ns}}}OnLaunch", {"HoursBetweenUpdateChecks": "12"})
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(descriptor).write(out, encoding="utf-8", xml_declaration=True)


def publish_canary_appinstaller(root: Path, *, tag: str, variant: str, bundle: Path,
                               identity: str, publisher: str, version: str, public_base: str) -> None:
    """Native SDK work is complete; verify its identity before writing a feed."""
    from scripts.releases.r2 import put
    from hermes_cli.update_channel import is_canary_tag

    if not is_canary_tag(tag):
        raise ValueError("Only canary feeds publish directly; stable requires accepted candidates")
    if variant not in ("bundled", "light"):
        raise ValueError("Store and commit builds have no App Installer feed")
    if not public_base:
        raise ValueError("Public feed base URL is required")
    with zipfile.ZipFile(bundle) as archive:
        native = ET.fromstring(archive.read("AppxMetadata/AppxBundleManifest.xml")).find("{*}Identity")
    for attr, expected in (("Name", identity), ("Publisher", publisher), ("Version", version)):
        if native is None or native.get(attr) != expected:
            raise ValueError(f"Universal bundle {attr} does not match its expected identity")
    directory = f"releases/win32/{'light/' if variant == 'light' else ''}canary"
    base = public_base.rstrip("/")
    descriptor = root / "canary.appinstaller"
    write_appinstaller(descriptor, identity=identity, publisher=publisher, version=version,
                       self_uri=f"{base}/{directory}/{descriptor.name}",
                       artifact_uri=f"{base}/{directory}/{bundle.name}")
    # A failed upload or HEAD verification leaves the previous pointer intact.
    put(tag=tag, key=f"{directory}/{bundle.name}", file=bundle, key_is_full=True, immutable=True)
    put(tag=tag, key=f"{directory}/{descriptor.name}", file=descriptor, key_is_full=True)


def promote(manifest: dict, root: Path, public_base: str) -> None:
    from scripts.releases.r2 import put, finalize

    materialize(manifest, root, public_base=public_base)
    windows = next(r for r in manifest["packages"] if r["platform"] == "windows")
    uri = f"{public_base.rstrip('/')}/releases/win32/stable/stable.appinstaller"
    appinstaller = root / "stable.appinstaller"
    write_appinstaller(appinstaller, identity=windows["identity"], publisher=windows["publisher"],
                       version=windows["version"], self_uri=uri, artifact_uri=windows["artifact"]["url"])
    apt = root / "apt"
    indexes = sorted(p for p in apt.rglob("*") if p.is_file() and not p.relative_to(apt).as_posix().startswith("pool/") and "/by-hash/" not in p.relative_to(apt).as_posix())
    indexes.sort(key=lambda p: p.name == "InRelease")
    if not any(p.name == "InRelease" for p in indexes):
        raise ValueError("Missing signed APT index")
    # Every package and index must exist before the first channel write.
    # The merged feed points at the attempt archive; the version stays plain.
    finalize(tag=manifest["tag"], dir=root, archive=manifest["archive"])
    def publish_pointer(key, file):
        put(tag=manifest["tag"], key=key, file=file, key_is_full=True)
        digest = hashlib.sha256()
        with urllib.request.urlopen(f"{public_base.rstrip('/')}/{key}", timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != sha256_file(file):
            raise ValueError(f"Channel read-back differs: {key}")

    publish_pointer("releases/win32/stable/stable.appinstaller", appinstaller)
    for file in indexes:
        publish_pointer(f"releases/termux/stable/{file.relative_to(apt).as_posix()}", file)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["record", "assemble", "publish", "promote", "materialize", "appinstaller", "publish-appinstaller"])
    parser.add_argument("--platform", choices=["windows", "macos", "termux"])
    parser.add_argument("--arch")
    parser.add_argument("--channel-request", type=Path)
    parser.add_argument("--update-policy", choices=["automatic", "pinned"], default="automatic")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tag", default=os.environ.get("RELEASE_TAG"))
    parser.add_argument("--archive", default=os.environ.get("HERMES_ARCHIVE_TAG"),
                        help="Attempt ref the release archive is keyed under when the payload tag is plain vX.Y.Z")
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--public-base", default=os.environ.get("CLOUDFLARE_R2_PUBLIC_URL"))
    parser.add_argument("--release-epoch", type=int, default=os.environ.get("HERMES_RELEASE_EPOCH"))
    parser.add_argument("--store-only", action="store_true")
    for name in ("identity", "publisher", "version", "self-uri", "artifact-uri"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--variant", choices=["bundled", "light"])
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    if args.command == "publish-appinstaller":
        if not args.bundle:
            parser.error("publish-appinstaller requires --bundle")
        publish_canary_appinstaller(args.root, tag=args.tag, variant=args.variant, bundle=args.bundle,
                                   identity=args.identity, publisher=args.publisher, version=args.version,
                                   public_base=args.public_base)
        return
    if args.command == "appinstaller":
        if not args.out:
            parser.error("appinstaller requires --out")
        write_appinstaller(args.out, identity=args.identity, publisher=args.publisher,
                           version=args.version, self_uri=args.self_uri, artifact_uri=args.artifact_uri,
                           update_policy=args.update_policy)
        return
    if args.command == "record":
        request = json.loads(args.channel_request.read_text(encoding="utf-8-sig")) if args.channel_request else None
        record(args.platform, args.arch, args.root, args.tag, args.commit, args.out, channel_request=request)
    elif args.command == "assemble":
        if args.release_epoch is None:
            parser.error("assemble requires the admitted --release-epoch")
        if not args.archive:
            parser.error("assemble requires the admitted --archive ref")
        assemble(args.root, args.tag, args.commit, args.public_base, args.out,
                 smoke_results=json.loads(os.environ.get("RELEASE_NEEDS", "{}")),
                 release_epoch=args.release_epoch, archive=args.archive)
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as file:
                file.write(f"manifest-url={args.public_base.rstrip('/')}/releases/tag/{args.archive}/release-candidates.json\nmanifest-sha256={sha256_file(args.out)}\n")
    else:
        if not args.archive:
            parser.error(f"{args.command} requires the admitted --archive ref")
        manifest = read_admitted_candidate(args.tag, args.commit, args.public_base,
                                          os.environ.get("CANDIDATE_MANIFEST_SHA256", ""),
                                          archive=args.archive)
        if args.command == "materialize":
            materialize(manifest, args.root, public_base=args.public_base, store_only=args.store_only)
        else:
            {"publish": publish, "promote": promote}[args.command](manifest, args.root, args.public_base)


if __name__ == "__main__":
    main()
