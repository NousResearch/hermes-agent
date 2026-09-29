"""Build fork-hosted Windows ARM64 wheels from hash-locked PyPI sdists."""

from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
from urllib.parse import quote
from zipfile import ZipFile

from pm.artifact_mirror import github_repository
from pm.downloader import Download, Source
from scripts.ci.archive_inputs import GhCli

_RELEASE = "wheelhouse"
_WHEEL = re.compile(
    r"(?P<name>[A-Za-z0-9_]+)-(?P<version>[A-Za-z0-9_.!+]+)"
    r"(?:-(?P<build>[0-9][A-Za-z0-9_]*))?"
    r"-(?P<python>[A-Za-z0-9_.]+)-(?P<abi>[A-Za-z0-9_.]+)-(?P<platform>[A-Za-z0-9_.]+)\.whl"
)


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_sdist(lock_path: Path, name: str, version: str | None = None) -> tuple[str, str, str]:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8-sig"))
    matches = [row for row in lock["package"] if _canonical(row["name"]) == _canonical(name)
               and (version is None or row["version"] == version) and "sdist" in row]
    if len(matches) != 1 or matches[0].get("source", {}).get("registry") is None:
        raise ValueError(f"{name}: expected one registry sdist in {lock_path}")
    row = matches[0]
    source = row["sdist"]
    digest = source["hash"].removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or source["hash"] != f"sha256:{digest}":
        raise ValueError(f"{name}: sdist needs a locked SHA-256")
    return row["version"], source["url"], digest


def fetch_locked_sdist(lock_path: Path, name: str, version: str, destination: Path) -> Path:
    locked_version, source_url, digest = _locked_sdist(lock_path, name, version)
    assert locked_version == version
    destination.parent.mkdir(parents=True, exist_ok=True)
    Download([Source(source_url, destination, digest)], partials_dir=destination.parent / "partials").run()
    return destination


def _native_tag(python: str, abi: str, platform: str) -> bool:
    if platform != "win_arm64":
        return False
    match = re.fullmatch(r"cp3(\d{1,2})", python)
    if not match:
        return False
    minor = int(match.group(1))
    return (python == "cp314" and abi == "cp314") or (abi == "abi3" and 2 <= minor <= 14)


def inspect_wheel(path: Path, name: str, version: str) -> str:
    """Admit a wheel only for native CPython 3.14 ARM64 and its claimed dist."""
    match = _WHEEL.fullmatch(path.name)
    if match is None or _canonical(match["name"]) != _canonical(name) or match["version"] != version:
        raise ValueError(f"wrong wheel package/version: {path.name}")
    tags = {f"{python}-{abi}-{platform}"
            for python in match["python"].split(".")
            for abi in match["abi"].split(".")
            for platform in match["platform"].split(".")
            if _native_tag(python, abi, platform)}
    if not tags:
        raise ValueError(f"wheel is not a CPython 3.14 win_arm64 artifact: {path.name}")
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"wheel has corrupt members: {path.name}")
        metadata_paths = [member for member in archive.namelist() if member.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"wheel metadata is missing or ambiguous: {path.name}")
        root = metadata_paths[0].removesuffix("METADATA")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        wheel = Parser().parsestr(archive.read(root + "WHEEL").decode("utf-8"))
        if (_canonical(metadata.get("Name", "")) != _canonical(name)
                or metadata.get("Version") != version
                or not tags.intersection(wheel.get_all("Tag", []))):
            raise ValueError(f"wheel metadata or tags disagree with the locked package: {path.name}")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build(name: str, lock_path: Path, output: Path, python: Path, *, build_tag: str = "") -> Path:
    from pm.store import current_target

    if current_target() != "win32-arm64":
        raise ValueError("wheelhouse builds require a native Windows ARM64 host")
    if build_tag and not re.fullmatch(r"[0-9][A-Za-z0-9_]*", build_tag):
        raise ValueError("wheel build tag must begin with a digit and contain only alphanumerics/underscores")
    version, _, source_hash = _locked_sdist(lock_path, name)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wheelhouse-", dir=output.parent) as scratch:
        source = fetch_locked_sdist(lock_path, name, version, Path(scratch) / "source.tar.gz")
        subprocess.run(["uv", "build", "--wheel", "--no-python-downloads", "--python", str(python),
                        "--out-dir", str(output), str(source)], check=True)
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"{name}: expected one built wheel, got {len(wheels)}")
    wheel = wheels[0]
    if build_tag:
        parts = wheel.name.split("-")
        wheel = wheel.rename(wheel.with_name("-".join([*parts[:2], build_tag, *parts[2:]])))
    digest = inspect_wheel(wheel, name, version)
    receipt = {"name": name, "version": version, "filename": wheel.name,
               "sha256": digest, "source_sha256": source_hash}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return wheel


def publish(receipts: Path, lock_path: Path, repository: str) -> list[dict]:
    if repository != github_repository():
        raise ValueError(f"wheelhouse publication is restricted to {github_repository()}")
    candidates = []
    for directory in sorted(receipts.iterdir()):
        if not directory.is_dir():
            raise ValueError(f"unexpected receipt path: {directory}")
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8-sig"))
        if set(receipt) != {"name", "version", "filename", "sha256", "source_sha256"}:
            raise ValueError(f"invalid wheel receipt: {directory}")
        version, _, source_hash = _locked_sdist(lock_path, receipt["name"], receipt["version"])
        assert version == receipt["version"]
        if receipt["source_sha256"] != source_hash:
            raise ValueError(f"sdist pin changed for {receipt['name']}")
        wheel = directory / receipt["filename"]
        if {path.name for path in directory.iterdir()} != {"receipt.json", wheel.name}:
            raise ValueError(f"unexpected wheel receipt content: {directory}")
        if inspect_wheel(wheel, receipt["name"], version) != receipt["sha256"]:
            raise ValueError(f"wheel archive hash changed for {receipt['name']}")
        candidates.append((wheel, receipt))
    if not candidates:
        raise ValueError("no wheels were produced")

    api = GhCli(repository)
    current = api.release_assets(_RELEASE)
    if current is None:
        subprocess.run(["gh", "release", "create", _RELEASE, "--repo", repository,
                        "--latest=false", "--prerelease", "--title", "Windows ARM64 wheelhouse",
                        "--notes", "Hash-locked wheels built from uv.lock sdists. Assets are immutable by filename."],
                       check=True)
        current = api.release_assets(_RELEASE)
    view = subprocess.run(["gh", "release", "view", _RELEASE, "--repo", repository,
                           "--json", "isPrerelease"], check=True, capture_output=True, text=True)
    if not json.loads(view.stdout)["isPrerelease"]:
        raise ValueError("wheelhouse release must be a prerelease, never /releases/latest")
    assets = {asset["name"]: asset for asset in current or []}
    verified = []
    for wheel, receipt in candidates:
        asset = assets.get(wheel.name)
        if asset is not None and (asset["state"] != "uploaded" or asset["size"] != wheel.stat().st_size):
            raise ValueError(f"existing wheel asset differs: {wheel.name}")
        if asset is None:
            api.upload(_RELEASE, wheel)
        public_url = f"https://github.com/{repository}/releases/download/{_RELEASE}/{quote(wheel.name)}"
        with tempfile.TemporaryDirectory(prefix="wheelhouse-readback-") as scratch:
            public = Path(scratch) / wheel.name
            Download([Source(public_url, public, receipt["sha256"])],
                     partials_dir=Path(scratch) / "partials").run()
            if public.stat().st_size != wheel.stat().st_size:
                raise ValueError(f"published wheel size differs: {wheel.name}")
        verified.append({**receipt, "url": public_url})
        print(f"{receipt['name']}=={receipt['version']}: {public_url} sha256:{receipt['sha256']}", flush=True)
    return verified


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    producer = sub.add_parser("build")
    producer.add_argument("--package", required=True)
    producer.add_argument("--lock", type=Path, default=Path("uv.lock"))
    producer.add_argument("--out", type=Path, required=True)
    producer.add_argument("--python", type=Path, required=True)
    producer.add_argument("--build-tag", default="")
    publisher = sub.add_parser("publish")
    publisher.add_argument("--receipts", type=Path, required=True)
    publisher.add_argument("--lock", type=Path, default=Path("uv.lock"))
    publisher.add_argument("--repository", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(build(args.package, args.lock, args.out, args.python, build_tag=args.build_tag))
    else:
        publish(args.receipts, args.lock, args.repository)


if __name__ == "__main__":
    main()
