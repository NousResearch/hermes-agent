"""Record, stage and promote the exact native release artifacts."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Iterable
from urllib.parse import unquote

from scripts.releases import lanes as lane_set
from scripts.releases.stable import read_admitted_candidate, accepted_smoke_results, validate_candidates
from scripts.termux.deb_version import deb_version_for_tag

if TYPE_CHECKING:
    from scripts.releases.r2 import CopySource


def release_version(tag: str) -> str:
    """The version a stable or canary tag publishes; packages are named after it."""
    from hermes_cli.update_channel import release_version as version_for

    version = version_for(tag)
    if version is None:
        raise ValueError(f"Not a release tag: {tag!r}")
    return version


def release_channel(tag: str) -> str:
    """The channel an accepted candidate publishes to: ``canary`` for a canary tag, else ``stable``.

    The tag decides, never a caller flag, so a stable candidate can never write a canary feed or
    the reverse (scripts/termux/deb_version.py and scripts/releases/r2.py key on it the same way)."""
    from hermes_cli.update_channel import is_canary_tag, is_commit_tag

    if is_commit_tag(tag):
        raise ValueError(f"Commit builds publish to no release channel: {tag!r}")
    return "canary" if is_canary_tag(tag) else "stable"


def sha256_file(file: Path) -> str:
    with file.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha512_base64(file: Path) -> str:
    with file.open("rb") as handle:
        return base64.b64encode(hashlib.file_digest(handle, "sha512").digest()).decode("ascii")


def _verify_blockmap_size(artifact: Path) -> None:
    blockmap = artifact.with_name(artifact.name + ".blockmap")
    if not blockmap.is_file():
        raise ValueError(f"Missing DMG blockmap: {blockmap.name}")
    try:
        data = json.loads(gzip.decompress(blockmap.read_bytes()))
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid DMG blockmap: {blockmap.name}") from exc
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise ValueError(f"Invalid DMG blockmap structure: {blockmap.name}")
    sizes = files[0].get("sizes")
    if not isinstance(sizes, list) or any(type(item) is not int or item < 0 for item in sizes):
        raise ValueError(f"Invalid DMG blockmap chunk sizes: {blockmap.name}")
    actual_size = artifact.stat().st_size
    if sum(sizes) != actual_size:
        raise ValueError(
            f"DMG blockmap covers {sum(sizes)} bytes, but {artifact.name} is {actual_size} bytes"
        )


def _require_feed_entry(feed_name: str, entries: list[dict], artifact: Path) -> None:
    matches = [entry for entry in entries if entry.get("url") == artifact.name]
    if len(matches) != 1:
        raise ValueError(
            f"macOS feed {feed_name} must list exactly one entry for {artifact.name}, found {len(matches)}"
        )
    entry = matches[0]
    if entry.get("size") != artifact.stat().st_size or entry.get("sha512") != _sha512_base64(artifact):
        raise ValueError(f"macOS feed {feed_name} disagrees with {artifact.name} on disk")


def _verify_mac_feed_dmg(root: Path, package: Path, dmg: Path, *, expect_feed: bool) -> None:
    feeds = sorted(root.glob("*-mac.yml"))
    if not feeds:
        if expect_feed:
            raise ValueError(f"Missing macOS update feed for {dmg.name}")
        return
    from scripts.releases.darwin import parse_mac_feed

    for feed_path in feeds:
        parsed = parse_mac_feed(feed_path.read_text(encoding="utf-8-sig"))
        _require_feed_entry(feed_path.name, parsed["files"], package)
        _require_feed_entry(feed_path.name, parsed["files"], dmg)


def _verify_macos_dmg(
    root: Path,
    arch: str,
    package: Path,
    expected_team: str,
    *,
    expect_feed: bool,
) -> None:
    dmg = single(root.glob(f"*-mac-{arch}.dmg"))
    expected_name = package.name.removesuffix(".zip") + ".dmg"
    if dmg.name != expected_name:
        raise ValueError("macOS DMG filename differs from requested version")
    subprocess.run(["codesign", "--verify", "--verbose=2", str(dmg)], check=True, timeout=60)
    signature = subprocess.run(
        ["codesign", "-dv", "--verbose=4", str(dmg)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    dmg_team = re.search(r"^TeamIdentifier=([A-Z0-9]{10})$", signature.stderr, re.MULTILINE)
    if not dmg_team or dmg_team.group(1) != expected_team:
        raise ValueError("Signed DMG team differs from signed app")
    _verify_blockmap_size(dmg)
    _verify_mac_feed_dmg(root, package, dmg, expect_feed=expect_feed)


def stamp_matches(stamp: dict, tag: str, commit: str, *, channel_request: dict | None = None,
                  legacy_commit_build: bool = False) -> None:
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
    from hermes_cli.update_channel import is_commit_tag, tag_core

    if legacy_commit_build:
        if not is_commit_tag(tag):
            raise ValueError("Legacy commit build mode is only allowed for commit claims")
        if (stamp.get("source") != "commit-build" or stamp.get("commit") != commit
                or stamp.get("tag") is not None or stamp.get("baseVersion") != tag_core(tag)):
            raise ValueError("Built legacy commit package provenance does not match the claim")
        return
    if (stamp.get("commit") != commit or stamp.get("tag") != tag
            or stamp.get("baseVersion") != release_version(tag)):
        raise ValueError("Built package provenance does not match the release")
    # A commit build's packages refuse every update (source commit-build); a release's never do.
    if (stamp.get("source") == "commit-build") != is_commit_tag(tag):
        raise ValueError("Built package update ownership does not match the release kind")


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


def require_owner_write(archive: zipfile.ZipFile) -> None:
    """Squirrel.Mac's ShipIt clears quarantine on every file and directory of the
    update before swapping it in, and aborts the whole install when one is
    read-only (#138659). Symlinks and entries without a Unix mode are skipped."""
    locked = [info.filename for info in archive.infolist()
              if (mode := info.external_attr >> 16) and not stat.S_ISLNK(mode) and not mode & stat.S_IWUSR]
    if locked:
        raise ValueError(f"{len(locked)} update ZIP entries lack owner write; ShipIt cannot install it ({locked[:3]})")


SQUIRREL_ZIP_LIMIT = 1 << 30


def require_squirrel_downloadable(package: Path) -> None:
    """Squirrel.Mac before Electron 44.7 (v0.22.0 ships 40) reads the whole update ZIP into one buffer;
    from 1 GiB its growth asks Electron's allocator for 2 GiB, which refuses, and the app dies
    mid-download (electron/electron#55089)."""
    size = package.stat().st_size
    if size >= SQUIRREL_ZIP_LIMIT:
        raise ValueError(f"update ZIP is {size:,} bytes; Squirrel.Mac before Electron 44.7 crashes "
                         f"downloading {SQUIRREL_ZIP_LIMIT:,} bytes (1 GiB) or more")
    print(f"macOS update ZIP: {size:,} bytes, {(SQUIRREL_ZIP_LIMIT - size) / 1e6:.0f} MB under the 1 GiB limit")


def _probe_windows_product_version(executable: Path) -> str:
    # -Command joins trailing argv into the script text, so $args stays empty.
    return subprocess.check_output([
        "powershell", "-NoProfile", "-Command",
        "(Get-Item -LiteralPath $env:RECORD_EXECUTABLE).VersionInfo.ProductVersion",
    ], env={**os.environ, "RECORD_EXECUTABLE": str(executable)},
        text=True, encoding="utf-8", stdin=subprocess.DEVNULL, timeout=30).strip()


def _packaged_windows_executable_version(
    archive: zipfile.ZipFile,
    executable_relpath: str,
    *,
    probe=None,
) -> str:
    """Extract the manifest's declared executable from the MSIX itself and probe its version.

    Reading ``*-unpacked/`` on disk is unsafe when a later variant build (such as the
    Store build on a canary) overwrites the unpacked directory with a different quad.
    """
    if not executable_relpath:
        raise ValueError("MSIX application manifest has no Executable attribute")
    target = PureWindowsPath(executable_relpath).as_posix().lstrip("/")
    # MakeAppx percent-encodes part names: "Hermes Agent.exe" is stored as Hermes%20Agent.exe.
    entry = single(
        name for name in archive.namelist()
        if unquote(name.replace("\\", "/")).lstrip("/").casefold() == target.casefold()
    )
    probe = probe or _probe_windows_product_version
    with tempfile.TemporaryDirectory() as temp:
        extracted = Path(temp) / PureWindowsPath(entry).name
        with archive.open(entry) as src, extracted.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return probe(extracted)


def _record_macos(
    root: Path,
    arch: str,
    tag: str,
    commit: str,
    *,
    expected_version: str | None,
    channel_request: dict | None,
    legacy_commit_build: bool,
) -> dict:
    version = (
        channel_request["version"]
        if channel_request is not None
        else expected_version if legacy_commit_build else release_version(tag)
    )
    package = single(root.glob(f"*-mac-{arch}.zip"))
    if not package.name.endswith(f"-{version}-mac-{arch}.zip"):
        raise ValueError("macOS artifact filename differs from requested version")
    require_squirrel_downloadable(package)
    app = single(root.glob("mac*/*.app"))
    subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True, timeout=60)
    signature = subprocess.run(
        ["codesign", "-dv", "--verbose=4", str(app)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    team = re.search(r"^TeamIdentifier=([A-Z0-9]{10})$", signature.stderr, re.MULTILINE)
    if not team:
        raise ValueError("Signed app has no Developer ID team")
    with zipfile.ZipFile(package) as archive:
        require_owner_write(archive)
        info = plistlib.loads(archive.read(single(
            n for n in archive.namelist() if re.fullmatch(r"[^/]+\.app/Contents/Info\.plist", n)
        )))
        stamp = json.loads(archive.read(single(
            n for n in archive.namelist() if re.fullmatch(r"[^/]+\.app/Contents/Resources/install-stamp\.json", n)
        )))
    stamp_matches(stamp, tag, commit, channel_request=channel_request,
                  legacy_commit_build=legacy_commit_build)
    from hermes_cli.update_channel import is_commit_tag

    is_commit = channel_request is None and is_commit_tag(tag)
    if not is_commit or (not legacy_commit_build and stamp.get("signedDmg") is True):
        _verify_macos_dmg(
            root,
            arch,
            package,
            team.group(1),
            expect_feed=channel_request is None and not is_commit,
        )
    fields: dict = {
        "identity": info["CFBundleIdentifier"],
        "teamId": team.group(1),
        "version": info["CFBundleShortVersionString"],
        "filename": package.name,
    }
    if stamp.get("receiverProtocol") == 1:
        fields["receiverProtocol"] = 1
    if fields["version"] != version:
        raise ValueError("App version differs from requested version")
    return fields


def _record_termux(root: Path, tag: str, commit: str, *, legacy_commit_build: bool) -> dict:
    from hermes_cli.update_channel import is_commit_tag, tag_core

    package = single((root / "deb").glob("*.deb"))
    fields = subprocess.check_output(
        ["dpkg-deb", "--field", str(package), "Package", "Version", "Architecture"],
        text=True, encoding="utf-8", timeout=60,
    )
    parsed = dict(line.split(": ", 1) for line in fields.splitlines())
    # A legacy claim builds through the tagless --commit path, which versions X.Y.Z+commit<sha12>.
    expected_version = (f"{tag_core(tag)}+commit{commit[:12]}" if legacy_commit_build
                        else deb_version_for_tag(tag))
    if parsed["Version"] != expected_version:
        raise ValueError("Termux artifact version differs from release tag")
    if parsed["Architecture"] != "aarch64":
        raise ValueError("Wrong Termux package architecture")
    expected_tag = None if legacy_commit_build else tag
    with tempfile.TemporaryDirectory() as temp:
        subprocess.run(["dpkg-deb", "--extract", str(package), temp], check=True, timeout=60)
        stamps = list(Path(temp).rglob("install-stamp.json"))
        if not any((data := json.loads(p.read_text(encoding="utf-8-sig"))).get("commit") == commit
                   and data.get("tag") == expected_tag
                   and (data.get("source") == "commit-build") == is_commit_tag(tag)
                   for p in stamps):
            raise ValueError("Termux package has no matching provenance")
    return {"identity": parsed["Package"], "version": parsed["Version"],
            "filename": package.relative_to(root).as_posix()}


def record(platform: str, arch: str, root: Path, tag: str, commit: str, out: Path,
           *, channel_request: dict | None = None, executable_version_probe=None,
           legacy_commit_build: bool = False) -> None:
    """Read identities from the built packages, never from the workflow matrix."""
    from hermes_cli.update_channel import is_commit_tag, tag_core

    if legacy_commit_build and (channel_request is not None or not is_commit_tag(tag)):
        raise ValueError("Legacy commit build mode is only allowed for commit claims")
    row: dict = {"platform": platform, "arch": arch, "tag": tag, "commit": commit}
    expected_version = tag_core(tag) if legacy_commit_build else None
    if platform == "windows":
        package = single(p for p in root.glob(f"*-win-{arch}.msix") if not p.name.startswith("Store-"))
        if channel_request is None:
            expected_file_version = expected_version if legacy_commit_build else release_version(tag)
            if not package.name.endswith(f"-{expected_file_version}-win-{arch}.msix"):
                raise ValueError("Windows artifact filename differs from release tag")
        version_sidecar = root / f"version-info-{arch}.json"
        with zipfile.ZipFile(package) as archive:
            manifest = ET.fromstring(archive.read("AppxManifest.xml"))
            identity = manifest.find("{*}Identity")
            application = desktop_application(manifest, channel_request)
            stamp_name = single(n for n in archive.namelist() if n.replace("\\", "/").endswith("/resources/install-stamp.json"))
            stamp = json.loads(archive.read(stamp_name))
            stamp_matches(stamp, tag, commit, channel_request=channel_request,
                          legacy_commit_build=legacy_commit_build)
            if identity.attrib["ProcessorArchitecture"].lower() != arch:
                raise ValueError("MSIX architecture differs from release target")
            row.update(identity=identity.attrib["Name"], publisher=identity.attrib["Publisher"],
                       applicationId=application.attrib["Id"], version=identity.attrib["Version"])
            if version_sidecar.is_file():
                row["executableVersion"] = json.loads(
                    version_sidecar.read_text(encoding="utf-8-sig"),
                )["productVersion"]
            elif executable_version_probe is not None or os.name == "nt":
                # The stamp carries no executable name; the package's own manifest does.
                row["executableVersion"] = _packaged_windows_executable_version(
                    archive,
                    application.attrib.get("Executable", ""),
                    probe=executable_version_probe,
                )
        if stamp.get("receiverProtocol") == 1:
            row["receiverProtocol"] = 1
    elif platform == "macos":
        row.update(_record_macos(
            root, arch, tag, commit,
            expected_version=expected_version,
            channel_request=channel_request,
            legacy_commit_build=legacy_commit_build,
        ))
    elif platform == "termux":
        row.update(_record_termux(root, tag, commit, legacy_commit_build=legacy_commit_build))
    else:
        raise ValueError("Unknown platform")
    if legacy_commit_build:
        row["legacyCommitBuild"] = True
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


ALL_HANDOFFS = ("win32-x64", "win32-arm64", "darwin-arm64", "termux", "windows-universal")


def assemble(root: Path, tag: str, commit: str, public_base: str, out: Path,
             *, smoke_results: dict, release_epoch: int, archive: str,
             lanes: Iterable[str]) -> dict:
    """Bind the native metadata to files already staged by their build jobs.

    `tag` stays the plain payload identity; `archive` is the attempt ref the
    manifest and every artifact URL are keyed under. `lanes` are the claim's:
    a handoff group a lane did not build must have no target in the manifest.
    The Windows groups build together, so the universal bundle (`windows`) and
    the Store bundle (`windows-store`) are each present exactly when their lane is."""
    from scripts.releases.handoff import receipt_name, validate_receipt
    from scripts.releases.r2 import put, staging_key_for

    lanes = frozenset(lanes)
    expected = lane_set.handoffs(lanes, ALL_HANDOFFS)
    if not expected:
        raise ValueError("Invalid candidate handoff set: the lanes built no package")
    absent = frozenset(lane_set.TARGET_LANE) - lane_set.targets(lanes)
    smoke_results = accepted_smoke_results(smoke_results, absent=absent)
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
    # Package bytes stay bound by their receipt digests; publish() downloads and
    # checks them, and the universal bundle's identity, before anything goes live.
    rows = []
    for name, receipt in sorted(by_name.items()):
        if name.startswith("metadata-"):
            file = root / name
            if not file.is_file() or sha256_file(file) != receipt["sha256"]:
                raise ValueError(f"Candidate local digest mismatch: {name}")
            rows.append(json.loads(file.read_text(encoding="utf-8-sig")))
    universal_name = None
    if "windows" in lanes:
        universal_name = single(name for name in by_name if name.endswith(".msixbundle") and not name.startswith("Store-"))
    elif any(name.endswith(".msixbundle") and not name.startswith("Store-") for name in by_name):
        raise ValueError("Candidate carries a signed Windows bundle the claim did not build")
    stores = [name for name in by_name if name.endswith(".msixbundle") and name.startswith("Store-")]
    if len(stores) != ("windows-store" in lanes):
        raise ValueError("Candidate Store bundle differs from the claim's windows-store lane")
    files = [{**row, "url": f"{public_base.rstrip('/')}/{staging_key_for(archive, name)}"}
             for name, row in sorted(by_name.items()) if not name.startswith("metadata-")]
    by_name = {item["path"]: item for item in files}
    packages = []
    for row in rows:
        # record() verified each package's own stamp; a row is metadata and carries no baseVersion.
        if row.get("commit") != commit or row.get("tag") != tag:
            raise ValueError("Built package provenance does not match the release")
        filename = universal_name if row["platform"] == "windows" else row["filename"]
        item = by_name[filename]
        packages.append({k: v for k, v in {**row, "artifact": {"url": item["url"], "sha256": item["sha256"]}}.items() if k != "filename"})
    result = {"schema": 2, "tag": tag, "commit": commit, "releaseEpoch": release_epoch,
              "archive": archive,
              "packages": packages, "files": files, "smoke_results": smoke_results}
    validate_candidates(result, tag, commit, public_base, release_epoch, archive=archive)
    present = {f"{row['platform']}/{row['arch']}" for row in packages}
    if present != set(lane_set.TARGET_LANE) - absent:
        raise ValueError("Candidate package rows differ from the built handoffs")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    put(tag=archive, key="release-candidates.json", file=out, immutable=True)
    return result


# Package payloads: receipt-bound by sha256, never needed on disk to write feeds.
PACKAGE_SUFFIXES = (".dmg", ".zip", ".blockmap", ".msix", ".msixbundle", ".deb")


def materialize(manifest: dict, root: Path, *, public_base: str, store_only: bool = False,
                metadata_only: bool = False, only: Iterable[str] | None = None) -> None:
    validate_candidates(manifest, manifest["tag"], manifest["commit"], public_base,
                        archive=manifest["archive"])
    files = manifest.get("files", [])
    if not files or len({item["path"] for item in files}) != len(files):
        raise ValueError("Missing or duplicate candidate file receipts")
    by_url = {item["url"]: item["sha256"] for item in files}
    if any(by_url.get(row["artifact"]["url"]) != row["artifact"]["sha256"] for row in manifest["packages"]):
        raise ValueError("Package receipts differ from candidate file receipts")
    selected = [item for item in files if item["path"].startswith("Store-") and item["path"].endswith(".msixbundle")] if store_only else files
    if metadata_only:
        selected = [item for item in selected if not item["path"].endswith(PACKAGE_SUFFIXES)]
    if only is not None:
        wanted = set(only)
        selected = [item for item in selected if item["path"] in wanted]
        if {item["path"] for item in selected} != wanted:
            raise ValueError("Requested candidate files are not in the manifest")
    if store_only and len(selected) != 1:
        raise ValueError("Expected one Store candidate")
    from scripts.releases.progress import log

    total = sum(item.get("size", 0) for item in selected)
    log(f"Materializing {len(selected)} candidate files ({total / 1e9:.1f} GB) into {root}")
    for number, item in enumerate(selected, start=1):
        log(f"  [{number}/{len(selected)}] {item['path']} ({item.get('size', 0) / 1e6:.0f} MB)")
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

    channel = release_channel(manifest["tag"])
    materialize(manifest, root, public_base=public_base)
    windows = [row for row in manifest["packages"] if row["platform"] == "windows"]
    if windows:
        paths = {item["url"]: item["path"] for item in manifest["files"]}
        (universal,) = {paths[row["artifact"]["url"]] for row in windows}
        validate_windows_bundle(root / universal, windows)
    for file in (root / "apt").rglob("*"):
        rel = file.relative_to(root / "apt").as_posix()
        if file.is_file() and (rel.startswith("pool/") or "/by-hash/" in rel):
            put(tag=manifest["tag"], key=f"releases/termux/{channel}/{rel}", file=file, key_is_full=True, immutable=True)


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


STABLE_DARWIN_DMG_CONTENT_TYPE = "application/x-apple-diskimage"
STABLE_DARWIN_DMG_DISPOSITION = 'attachment; filename="HermesBundled-mac-arm64.dmg"'


def _stable_darwin_dmg_item(manifest: dict, public_base: str) -> tuple[str, dict]:
    from scripts.releases import r2

    macos = single(row for row in manifest["packages"] if row["platform"] == "macos" and row["arch"] == "arm64")
    files = manifest.get("files", [])
    by_url = {item["url"]: item for item in files}
    zip_item = by_url.get(macos["artifact"]["url"])
    if zip_item is None or not str(zip_item.get("path", "")).endswith(".zip"):
        raise ValueError("Missing receipt-bound macOS ZIP in candidate manifest")
    dmg_name = zip_item["path"].removesuffix(".zip") + ".dmg"
    matches = [item for item in files if item.get("path") == dmg_name]
    if len(matches) != 1:
        raise ValueError(f"Missing receipt-bound macOS DMG in candidate manifest: {dmg_name}")
    dmg_item = matches[0]
    source_key = r2.staging_key_for(manifest["archive"], dmg_name)
    expected_url = f"{public_base.rstrip('/')}/{source_key}"
    if (
        dmg_item.get("url") != expected_url
        or type(dmg_item.get("size")) is not int
        or dmg_item["size"] <= 0
        or not re.fullmatch(r"[a-f0-9]{64}", str(dmg_item.get("sha256", "")))
    ):
        raise ValueError(f"Invalid macOS DMG receipt in candidate manifest: {dmg_name}")
    return source_key, dmg_item


def _check_stable_dmg_state(
    existing,
    *,
    tag: str,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    from hermes_cli.update_channel import is_canary_tag, is_commit_tag
    from scripts.releases import semver

    if existing is None:
        return True
    existing_release = existing.header("x-amz-meta-release")
    if existing_release:
        if is_canary_tag(existing_release) or is_commit_tag(existing_release):
            raise ValueError("Refusing to overwrite non-stable macOS DMG alias")
        existing_version = release_version(existing_release)
        order = semver.compare(release_version(tag), existing_version)
        if order < 0:
            raise ValueError("Refusing to move the stable macOS DMG backward")
        if order == 0:
            existing_sha = existing.header("x-amz-meta-sha256")
            if existing_sha and existing_sha != expected_sha256:
                raise ValueError("Refusing to replace published stable macOS DMG version with different artifact")
    matches_live = (
        existing.header("content-length") == str(expected_size)
        and existing.header("x-amz-meta-sha256") == expected_sha256
        and existing.header("x-amz-meta-release") == tag
        and existing.header("cache-control") == "no-store"
        and existing.header("content-disposition") == STABLE_DARWIN_DMG_DISPOSITION
        and existing.header("content-type") == STABLE_DARWIN_DMG_CONTENT_TYPE
    )
    return not matches_live


def _check_stable_page_state(
    existing: tuple[bytes, str] | None,
    *,
    tag: str,
    page_bytes: bytes,
) -> tuple[bytes | None, dict[str, str] | None]:
    """The page bytes this promotion makes live (None: the page is not this promotion's to write
    or read back), and the conditions for writing them (None: nothing to write).

    A page that already describes a newer stable release stays as it is; a canary page at this
    key fails closed. Feed order is checked before promotion, so a newer stable page can only come
    from render-builds-table's `complete` step rendering a newer release while this older advance
    was pending; refusing would wedge the sequencer, which advances oldest first."""
    from hermes_cli.update_channel import published_canary_timestamp
    from scripts.releases.downloads_page import recorded_build, supersedes

    if existing is None:
        return page_bytes, {"If-None-Match": "*"}
    existing_bytes, etag = existing
    existing_page = existing_bytes.decode("utf-8", errors="replace")
    if not supersedes(existing_page, tag):
        if published_canary_timestamp(recorded_build(existing_page)) is not None:
            raise ValueError("Refusing to keep or replace a canary downloads page at the stable key")
        print(f"::warning::the stable downloads page already describes a release newer than {tag}; "
              "leaving it unchanged")
        return None, None
    if existing_bytes == page_bytes:
        return page_bytes, None
    return page_bytes, {"If-Match": etag}


def _verify_public_head_size(public_base: str, key: str, expected_size: int) -> None:
    request = urllib.request.Request(f"{public_base.rstrip('/')}/{key}", method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.headers.get("Content-Length") != str(expected_size):
            raise ValueError(f"Channel read-back differs: {key}")


def _verify_public_bytes(public_base: str, key: str, expected_bytes: bytes) -> None:
    with urllib.request.urlopen(f"{public_base.rstrip('/')}/{key}", timeout=60) as response:
        if response.read() != expected_bytes:
            raise ValueError(f"Channel read-back differs: {key}")


@dataclass(frozen=True)
class StableDownloads:
    """The stable macOS DMG alias and downloads page for one accepted candidate.

    ``prepare`` runs every check (the DMG receipt, its source object and the single-copy limit,
    the order against the live alias and page) before any feed moves. ``promote`` calls
    ``publish_dmg`` right after the macOS feed, so a failed copy leaves the Windows and Termux
    pointers on the previous release, and ``publish_page`` after every feed pointer, so a failed
    feed write leaves the page there too. Each object this promotion writes or finds already in
    place is read back through the public origin."""

    tag: str
    public_base: str
    creds: dict[str, str] = field(repr=False)
    base: str
    bucket: str
    source_dmg: CopySource
    dmg_sha256: str
    dmg_needs_copy: bool
    page_bytes: bytes | None = field(repr=False)
    page_conditions: dict[str, str] | None

    @classmethod
    def prepare(cls, manifest: dict, public_base: str) -> StableDownloads:
        from scripts.releases import r2
        from scripts.releases.channels import R2ChannelStore
        from scripts.releases.downloads_page import DEFAULT_REPO, render_channel_page_from_manifest

        source_dmg_key, dmg_item = _stable_darwin_dmg_item(manifest, public_base)
        repo = os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO
        page_bytes = render_channel_page_from_manifest(manifest, public_base, repo=repo).encode("utf-8")
        creds, base, bucket = r2.credentials()
        now = r2.amz_timestamp()
        source_dmg = r2.copy_source(creds, base, bucket, source_dmg_key, now, expected_size=dmg_item["size"])
        dmg_needs_copy = _check_stable_dmg_state(
            r2.head_object(creds, base, bucket, r2.channel_dmg_key_for("stable"), now),
            tag=manifest["tag"],
            expected_size=dmg_item["size"],
            expected_sha256=dmg_item["sha256"],
        )
        page_bytes, page_conditions = _check_stable_page_state(
            R2ChannelStore(creds, base, bucket).get(r2.channel_page_key_for("stable")),
            tag=manifest["tag"],
            page_bytes=page_bytes,
        )
        return cls(
            tag=manifest["tag"], public_base=public_base, creds=creds, base=base, bucket=bucket,
            source_dmg=source_dmg, dmg_sha256=dmg_item["sha256"],
            dmg_needs_copy=dmg_needs_copy, page_bytes=page_bytes, page_conditions=page_conditions,
        )

    def publish_dmg(self) -> None:
        from scripts.releases import r2

        dmg_key = r2.channel_dmg_key_for("stable")
        if self.dmg_needs_copy:
            r2.copy_object(
                self.creds, self.base, self.bucket, self.source_dmg, dmg_key, r2.amz_timestamp(),
                content_type=STABLE_DARWIN_DMG_CONTENT_TYPE,
                content_disposition=STABLE_DARWIN_DMG_DISPOSITION,
                metadata={"sha256": self.dmg_sha256, "release": self.tag, "source": self.source_dmg.key},
            )
        _verify_public_head_size(self.public_base, dmg_key, self.source_dmg.size)

    def publish_page(self) -> None:
        from scripts.releases import r2

        if self.page_bytes is None:
            return
        page_key = r2.channel_page_key_for("stable")
        if self.page_conditions is not None:
            r2.put_object(
                self.creds, self.base, self.bucket, page_key, self.page_bytes, r2.amz_timestamp(),
                "text/html; charset=utf-8", self.page_conditions,
            )
        _verify_public_bytes(self.public_base, page_key, self.page_bytes)


def promote(manifest: dict, root: Path, public_base: str) -> None:
    """Point the release's channel at an accepted candidate: the Windows App Installer feed, the
    merged macOS feed, the stable macOS DMG alias and downloads page, and the Termux APT indexes.
    ``release_channel`` picks stable or canary."""
    from scripts.releases.r2 import finalize

    channel = release_channel(manifest["tag"])
    # Feeds need the per-arch feed legs and APT indexes, not the packages: those are
    # bound by the accepted candidate's sha256 receipts.
    materialize(manifest, root, public_base=public_base, metadata_only=True)
    windows = next(r for r in manifest["packages"] if r["platform"] == "windows")
    uri = f"{public_base.rstrip('/')}/releases/win32/{channel}/{channel}.appinstaller"
    appinstaller = root / f"{channel}.appinstaller"
    stable_downloads: StableDownloads | None = None
    if channel == "stable":
        write_appinstaller(appinstaller, identity=windows["identity"], publisher=windows["publisher"],
                           version=windows["version"], self_uri=uri, artifact_uri=windows["artifact"]["url"])
        stable_downloads = StableDownloads.prepare(manifest, public_base)
    apt = root / "apt"
    indexes = sorted(p for p in apt.rglob("*") if p.is_file() and not p.relative_to(apt).as_posix().startswith("pool/") and "/by-hash/" not in p.relative_to(apt).as_posix())
    indexes.sort(key=lambda p: p.name == "InRelease")
    # Termux is opt-in: a release that did not build it leaves the APT channel
    # on the previous Termux release, and ships no indexes to publish.
    termux = any(row["platform"] == "termux" for row in manifest["packages"])
    if termux and not any(p.name == "InRelease" for p in indexes):
        raise ValueError("Missing signed APT index")
    if not termux and indexes:
        raise ValueError("APT indexes staged for a release without a Termux candidate")
    # Every package and index must exist before the first channel write.
    # The merged feed points at the attempt archive; the version stays plain.
    finalize(tag=manifest["tag"], dir=root, archive=manifest["archive"], verify_bytes=False)
    publish_pointer = _pointer_publisher(manifest["tag"], public_base)
    if stable_downloads is not None:
        stable_downloads.publish_dmg()
        publish_pointer("releases/win32/stable/stable.appinstaller", appinstaller)
    else:
        # The canary feed keeps its bundle beside it, as it always has: canary archives are pruned
        # after 14 days (scripts/releases/r2.py prune-canaries), and a quiet fortnight must not
        # leave the live feed pointing at a deleted archive. The bytes are the accepted bundle,
        # re-checked against their receipt digest before they are copied.
        paths = {item["url"]: item["path"] for item in manifest["files"]}
        bundle_name = paths[windows["artifact"]["url"]]
        materialize(manifest, root, public_base=public_base, only={bundle_name})
        publish_canary_appinstaller(root, tag=manifest["tag"], variant="bundled", bundle=root / bundle_name,
                                    identity=windows["identity"], publisher=windows["publisher"],
                                    version=windows["version"], public_base=public_base)
        publish_pointer(f"releases/win32/canary/{appinstaller.name}", appinstaller, upload=False)
    for file in indexes:
        publish_pointer(f"releases/termux/{channel}/{file.relative_to(apt).as_posix()}", file)
    if stable_downloads is not None:
        stable_downloads.publish_page()


def _pointer_publisher(tag: str, public_base: str):
    """Write one mutable channel pointer and read it back through the public origin."""
    from scripts.releases.r2 import put

    def publish_pointer(key: str, file: Path, *, upload: bool = True) -> None:
        if upload:
            put(tag=tag, key=key, file=file, key_is_full=True)
        digest = hashlib.sha256()
        with urllib.request.urlopen(f"{public_base.rstrip('/')}/{key}", timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != sha256_file(file):
            raise ValueError(f"Channel read-back differs: {key}")

    return publish_pointer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["record", "publish", "promote", "materialize", "appinstaller", "publish-appinstaller"])
    parser.add_argument("--platform", choices=["windows", "macos", "termux"])
    parser.add_argument("--arch")
    parser.add_argument("--channel-request", type=Path)
    parser.add_argument("--update-policy", choices=["automatic", "pinned"], default="automatic")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tag", default=os.environ.get("RELEASE_TAG"))
    parser.add_argument("--archive", default=os.environ.get("HERMES_ARCHIVE_TAG"),
                        help="Attempt ref the release archive is keyed under when the payload tag is plain vX.Y.Z")
    # A stable run dispatches from the default branch, so GITHUB_SHA is not the admitted claim commit.
    parser.add_argument("--commit", default=os.environ.get("RELEASE_COMMIT") or os.environ.get("GITHUB_SHA"))
    parser.add_argument("--public-base", default=os.environ.get("CLOUDFLARE_R2_PUBLIC_URL"))
    parser.add_argument("--store-only", action="store_true")
    parser.add_argument("--legacy-commit-build", action="store_true",
                        help="Record a commit claim built on a pre-claim commit (tagless X.Y.Z provenance)")
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
        record(args.platform, args.arch, args.root, args.tag, args.commit, args.out,
               channel_request=request, legacy_commit_build=args.legacy_commit_build)
    else:
        if not args.archive:
            parser.error(f"{args.command} requires the admitted --archive ref")
        # read_admitted_candidate binds the manifest to --tag, so the tag decides before any read.
        if args.command == "promote" and release_channel(args.tag) == "stable":
            parser.error("promote refuses stable candidates: stable promotion runs only through the "
                         "Stable Release Publication sequencer (channel_releases.advance_stable), which "
                         "requires the published release and serializes every stable writer")
        manifest = read_admitted_candidate(args.tag, args.commit, args.public_base,
                                          os.environ.get("CANDIDATE_MANIFEST_SHA256", ""),
                                          archive=args.archive)
        if args.command == "materialize":
            materialize(manifest, args.root, public_base=args.public_base, store_only=args.store_only)
        else:
            {"publish": publish, "promote": promote}[args.command](manifest, args.root, args.public_base)


if __name__ == "__main__":
    main()
