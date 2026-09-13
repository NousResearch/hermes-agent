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

from scripts.releases.stable import read_manifest, require_stable_identity, validate_candidates


def sha256_file(file: Path) -> str:
    with file.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stamp_matches(stamp: dict, tag: str, commit: str) -> None:
    if stamp.get("commit") != commit or stamp.get("tag") != tag:
        raise ValueError("Built package provenance does not match the release")


def single(items):
    items = list(items)
    if len(items) != 1:
        raise ValueError(f"Expected exactly one artifact, found {items}")
    return items[0]


def record(platform: str, arch: str, root: Path, tag: str, commit: str, out: Path) -> None:
    """Read identities from the built packages, never from the workflow matrix."""
    row = {"platform": platform, "arch": arch, "tag": tag, "commit": commit}
    if platform == "windows":
        package = single(p for p in root.glob(f"*-win-{arch}.msix") if not p.name.startswith("Store-"))
        with zipfile.ZipFile(package) as archive:
            manifest = ET.fromstring(archive.read("AppxManifest.xml"))
            identity = manifest.find("{*}Identity")
            application = single(manifest.findall("{*}Applications/{*}Application"))
            stamp_name = single(n for n in archive.namelist() if n.replace("\\", "/").endswith("/resources/install-stamp.json"))
            stamp_matches(json.loads(archive.read(stamp_name)), tag, commit)
        if identity.attrib["ProcessorArchitecture"].lower() != arch:
            raise ValueError("MSIX architecture differs from release target")
        row.update(identity=identity.attrib["Name"], publisher=identity.attrib["Publisher"],
                   applicationId=application.attrib["Id"], version=identity.attrib["Version"])
    elif platform == "macos":
        package = single(root.glob(f"*-mac-{arch}.zip"))
        app = single(root.glob("mac*/*.app"))
        subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True)
        signature = subprocess.run(["codesign", "-dv", "--verbose=4", str(app)], check=True, capture_output=True, text=True, encoding="utf-8")
        team = re.search(r"^TeamIdentifier=([A-Z0-9]{10})$", signature.stderr, re.M)
        if not team:
            raise ValueError("Signed app has no Developer ID team")
        with zipfile.ZipFile(package) as archive:
            info = plistlib.loads(archive.read(single(n for n in archive.namelist() if re.fullmatch(r"[^/]+\.app/Contents/Info.plist", n))))
            stamp = json.loads(archive.read(single(n for n in archive.namelist() if re.fullmatch(r"[^/]+\.app/Contents/Resources/install-stamp.json", n))))
        stamp_matches(stamp, tag, commit)
        row.update(identity=info["CFBundleIdentifier"], teamId=team.group(1), version=info["CFBundleShortVersionString"], filename=package.name)
        if row["version"] != tag[1:]:
            raise ValueError("App version differs from release tag")
    elif platform == "termux":
        package = single((root / "deb").glob("*.deb"))
        fields = subprocess.check_output(["dpkg-deb", "--field", str(package), "Package", "Version", "Architecture"], text=True, encoding="utf-8")
        parsed = dict(line.split(": ", 1) for line in fields.splitlines())
        row.update(identity=parsed["Package"], version=parsed["Version"], filename=package.relative_to(root).as_posix())
        if parsed["Architecture"] != "aarch64":
            raise ValueError("Wrong Termux package architecture")
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(["dpkg-deb", "--extract", str(package), temp], check=True)
            stamps = list(Path(temp).rglob("install-stamp.json"))
            if not any((data := json.loads(p.read_text(encoding="utf-8"))).get("commit") == commit and data.get("tag") == tag for p in stamps):
                raise ValueError("Termux package has no matching provenance")
    else:
        raise ValueError("Unknown platform")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def assemble(root: Path, tag: str, commit: str, public_base: str, out: Path) -> dict:
    """Bind the native metadata to files already staged by their build jobs."""
    from scripts.releases.handoff import receipt_name, validate_receipt
    from scripts.releases.r2 import put, staging_key_for

    expected = ("win32-x64", "win32-arm64", "darwin-x64", "darwin-arm64", "termux", "windows-universal")
    by_name = {}
    for name in expected:
        file = root / receipt_name(name)
        if not file.is_file():
            raise ValueError(f"Missing candidate handoff: {name}")
        receipt = json.loads(file.read_text(encoding="utf-8"))
        for row in validate_receipt(receipt, tag, commit, name):
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
                rows.append(json.loads(file.read_text(encoding="utf-8")))
    universal_name = single(name for name in by_name if name.endswith(".msixbundle") and not name.startswith("Store-"))
    single(name for name in by_name if name.endswith(".msixbundle") and name.startswith("Store-"))
    windows = [r for r in rows if r["platform"] == "windows"]
    if sorted(r["arch"] for r in windows) != ["arm64", "x64"]:
        raise ValueError("Windows metadata must include both architectures")
    for field in ("identity", "publisher", "version", "applicationId"):
        if len({r[field] for r in windows}) != 1:
            raise ValueError(f"Windows packages disagree on {field}")
    with zipfile.ZipFile(root / universal_name) as archive:
        manifest = ET.fromstring(archive.read("AppxMetadata/AppxBundleManifest.xml"))
        identity = manifest.find("{*}Identity")
        for attr, field in (("Name", "identity"), ("Publisher", "publisher"), ("Version", "version")):
            if identity.attrib[attr] != windows[0][field]:
                raise ValueError("Universal bundle identity does not match its packages")
    files = [{**row, "url": f"{public_base.rstrip('/')}/{staging_key_for(tag, name)}"}
             for name, row in sorted(by_name.items()) if not name.startswith("metadata-")]
    by_name = {item["path"]: item for item in files}
    packages = []
    for row in rows:
        stamp_matches(row, tag, commit)
        filename = universal_name if row["platform"] == "windows" else row["filename"]
        item = by_name[filename]
        packages.append({k: v for k, v in {**row, "artifact": {"url": item["url"], "sha256": item["sha256"]}}.items() if k != "filename"})
    result = {"schema": 1, "tag": tag, "commit": commit, "packages": packages, "files": files}
    validate_candidates(result, tag, commit, public_base)
    if not any(row["platform"] == "termux" for row in packages):
        raise ValueError("Missing Termux candidate")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    put(tag=tag, key="release-candidates.json", file=out, immutable=True)
    return result


def materialize(manifest: dict, root: Path, *, public_base: str, store_only: bool = False) -> None:
    validate_candidates(manifest, manifest["tag"], manifest["commit"], public_base)
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
        expected = f"{public_base.rstrip('/')}/releases/tag/{manifest['tag']}/{relative.as_posix()}"
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


def promote(manifest: dict, root: Path, public_base: str) -> None:
    from scripts.releases.r2 import put, finalize

    materialize(manifest, root, public_base=public_base)
    windows = next(r for r in manifest["packages"] if r["platform"] == "windows")
    uri = f"{public_base.rstrip('/')}/releases/win32/stable/stable.appinstaller"
    ns = "http://schemas.microsoft.com/appx/appinstaller/2017/2"
    ET.register_namespace("", ns)
    descriptor = ET.Element(f"{{{ns}}}AppInstaller", {"Uri": uri, "Version": windows["version"]})
    ET.SubElement(descriptor, f"{{{ns}}}MainBundle", {"Name": windows["identity"], "Publisher": windows["publisher"], "Version": windows["version"], "Uri": windows["artifact"]["url"]})
    settings = ET.SubElement(descriptor, f"{{{ns}}}UpdateSettings")
    ET.SubElement(settings, f"{{{ns}}}OnLaunch", {"HoursBetweenUpdateChecks": "12"})
    appinstaller = root / "stable.appinstaller"
    ET.ElementTree(descriptor).write(appinstaller, encoding="utf-8", xml_declaration=True)
    apt = root / "apt"
    indexes = sorted(p for p in apt.rglob("*") if p.is_file() and not p.relative_to(apt).as_posix().startswith("pool/") and "/by-hash/" not in p.relative_to(apt).as_posix())
    indexes.sort(key=lambda p: p.name == "InRelease")
    if not any(p.name == "InRelease" for p in indexes):
        raise ValueError("Missing signed APT index")
    # Every package and index must exist before the first channel write.
    finalize(tag=manifest["tag"], dir=root)
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
    parser.add_argument("command", choices=["record", "assemble", "publish", "promote", "materialize"])
    parser.add_argument("--platform", choices=["windows", "macos", "termux"])
    parser.add_argument("--arch")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tag", default=os.environ.get("RELEASE_TAG"))
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--public-base", default=os.environ.get("CLOUDFLARE_R2_PUBLIC_URL"))
    parser.add_argument("--store-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "record":
        record(args.platform, args.arch, args.root, args.tag, args.commit, args.out)
    elif args.command == "assemble":
        assemble(args.root, args.tag, args.commit, args.public_base, args.out)
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as file:
                file.write(f"manifest-url={args.public_base.rstrip('/')}/releases/tag/{args.tag}/release-candidates.json\nmanifest-sha256={sha256_file(args.out)}\n")
    else:
        expected_digest = os.environ.get("CANDIDATE_MANIFEST_SHA256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", expected_digest):
            raise ValueError("Pinned candidate manifest digest is required")
        require_stable_identity(args.tag, args.commit, f"refs/tags/{args.tag}")
        manifest = read_manifest(
            f"{args.public_base.rstrip('/')}/releases/tag/{args.tag}/release-candidates.json",
            expected_digest, expected_origin=args.public_base,
        )
        validate_candidates(manifest, args.tag, args.commit, args.public_base)
        if args.command == "materialize":
            materialize(manifest, args.root, public_base=args.public_base, store_only=args.store_only)
        else:
            {"publish": publish, "promote": promote}[args.command](manifest, args.root, args.public_base)


if __name__ == "__main__":
    main()
