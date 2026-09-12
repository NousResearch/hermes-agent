"""Archive pinned binary inputs in R2 and seed existing CI consumers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from urllib.parse import quote, urlsplit

# The runner invokes this before setup-pm has installed the checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pm.artifact_mirror import object_key
from pm.downloader import Download, DownloadError, DownloadTransportError, Source
from pm.lock import SCHEMA
from pm.store import ALL_TARGETS, Store
from scripts.releases import r2

TERMUX_TARGET = "linux-arm64-bionic"


@dataclass(frozen=True)
class InputPin:
    name: str
    url: str
    sha256: str
    kind: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9+_.@-]*", self.name):
            raise ValueError(f"Invalid input name: {self.name!r}")
        object_key(self.sha256)
        parsed = urlsplit(self.url)
        loopback = parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
        if (parsed.scheme != "https" and not loopback) or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError(f"{self.name}: pinned input needs an HTTPS URL")
        if self.kind not in ("library", "license", "tool"):
            raise ValueError(f"Invalid input kind: {self.kind}")


def historical_url(url: str) -> str | None:
    parsed = urlsplit(url)
    prefix = "/apt/termux-main/pool/main/"
    if parsed.scheme != "https" or parsed.netloc != "packages.termux.dev" or not parsed.path.startswith(prefix):
        return None
    parts = parsed.path[len(prefix):].split("/")
    if len(parts) != 3 or any(not part or part in (".", "..") for part in parts):
        return None
    group, package, filename = (quote(part, safe="") for part in parts)
    return f"https://archive.org/download/termux_pkgs_archive_{group}/{package}/{filename}"


def _pin(name: str, row: dict, kind: str) -> InputPin:
    if not isinstance(row, dict) or not isinstance(row.get("url"), str):
        raise ValueError(f"{name}: invalid pinned input row")
    return InputPin(name, row["url"], row.get("sha256"), kind)


def pinned_inputs(repo: Path, *, target: str | None = None, packages: set[str] | None = None) -> list[InputPin]:
    """All pin references, including shared bytes needed at different paths."""
    if target is not None and target not in ALL_TARGETS:
        raise ValueError(f"Unknown target: {target}")
    lock = json.loads((repo / "pm/lock.json").read_text(encoding="utf-8-sig"))
    if not isinstance(lock, dict) or lock.get("schema") != SCHEMA or not isinstance(lock.get("packages"), dict):
        raise ValueError("Invalid PM lockfile")
    if packages is not None and packages - lock["packages"].keys():
        raise ValueError(f"Unknown pinned packages: {sorted(packages - lock['packages'].keys())}")
    pins = []
    for name, package in lock["packages"].items():
        if packages is not None and name not in packages:
            continue
        artifacts = package["artifacts"]
        if target is not None:
            key = target if target in artifacts else "any"
            artifacts = {key: artifacts[key]} if key in artifacts else {}
        for row_target, rows in artifacts.items():
            for row in rows if isinstance(rows, list) else [rows]:
                label = f"{name}@{row_target}"
                if isinstance(row, dict) and isinstance(row.get("url"), str) and row["url"].startswith("docker://"):
                    continue  # OCI digests belong to the container registry, not HTTP archives.
                pins.append(_pin(label, row, "tool"))
    if packages is None and target in (None, TERMUX_TARGET):
        table = json.loads((repo / "scripts/termux/runtime_libs.json").read_text(encoding="utf-8"))
        pins.extend(_pin(name, row, "library") for name, row in table["libs"].items())
        if table.get("licenses") is not None:
            pins.append(_pin("termux-licenses", table["licenses"], "license"))
    if not pins:
        raise ValueError("No pinned HTTP inputs selected")
    return pins


@dataclass(frozen=True)
class Archive:
    creds: dict[str, str]
    base: str
    bucket: str

    def fetch(self, pin: InputPin, destination: Path) -> str:
        """Only R2 404 permits upstream download; every result is read back."""
        key = object_key(pin.sha256)
        url = f"{self.base}/{self.bucket}/{r2.encode_key_path(key)}"
        try:
            head = r2.signed_request("HEAD", url, creds=self.creds, now=r2.amz_timestamp())
        except r2.R2RequestError as exc:
            if exc.status != 404:
                raise
            head = None
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".input-archive-", dir=destination.parent) as temporary:
            work = Path(temporary)
            local = work / "input"
            origin = "R2"
            if head is None:
                origin = _download_upstream(pin, local)
                size = local.stat().st_size
                r2.put_object(self.creds, self.base, self.bucket, key, str(local), r2.amz_timestamp(),
                              "application/octet-stream", conditions={"If-None-Match": "*"})
            else:
                length = head.header("content-length")
                if length is None:
                    raise ValueError(f"R2 did not report the input size: {url}")
                size = int(length)
            verified = work / "verified"
            r2.download_object(self.creds, self.base, self.bucket, key, verified, r2.amz_timestamp(),
                               expected_size=size, expected_sha256=pin.sha256)
            verified.replace(destination)
        return origin


def _download_upstream(pin: InputPin, local: Path) -> str:
    try:
        Download([Source(pin.url, local, pin.sha256)], partials_dir=local.parent / "partials").run()
        return "upstream"
    except DownloadTransportError as exc:
        backup = historical_url(pin.url)
        if exc.status not in (404, 410) or backup is None:
            raise
        try:
            Download([Source(backup, local, pin.sha256)], partials_dir=local.parent / "partials").run()
        except DownloadError as failure:
            raise DownloadError(f"{exc}\n{failure}") from failure
        return "historical archive"


def stage_inputs(pins: list[InputPin], *, archive: Archive, store: Store | None = None, payload: Path | None = None) -> int:
    """Archive all unique digests concurrently; optionally seed the stagers' inputs."""
    from scripts.termux.stage_runtime_libs import download_path

    groups: dict[str, list[InputPin]] = {}
    for pin in pins:
        groups.setdefault(pin.sha256, []).append(pin)
    if not groups:
        return 0

    def stage_digest(digest: str, references: list[InputPin]) -> None:
        # Each worker owns its downloads and cleanup, including on failure.
        with tempfile.TemporaryDirectory(prefix="hermes-inputs-") as temporary:
            local = Path(temporary) / "input"
            origin = archive.fetch(references[0], local)
            for pin in references:
                if pin.kind != "tool" and payload is not None:
                    dest = download_path(payload, pin.name)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, dest)
            if store is not None and any(pin.kind == "tool" for pin in references):
                tool = next(pin for pin in references if pin.kind == "tool")
                filename = PurePosixPath(urlsplit(tool.url).path).name
                if not filename or filename in (".", "..") or "\\" in filename:
                    raise ValueError(f"Invalid archive filename: {tool.url}")
                with store.install_lock(), store.scratch() as scratch:
                    staged = scratch / "archive"
                    staged.mkdir()
                    shutil.copyfile(local, staged / filename)
                    entry = store.entry(f"fetch-{digest}")
                    if entry.exists():
                        shutil.rmtree(entry)
                    store.publish(staged, entry.name)
            print(f"  {references[0].name}: {origin} -> {object_key(digest)}", flush=True)

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(stage_digest, digest, references) for digest, references in groups.items()]
        for future in as_completed(futures):
            future.result()
    return len(groups)


def main(argv=None) -> int:
    from pm import paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=ALL_TARGETS)
    parser.add_argument("--payload", type=Path, help="seed Termux runtime-library downloads")
    parser.add_argument("--store", type=Path, help="seed this PM store's disposable input cache")
    args = parser.parse_args(argv)
    pins = pinned_inputs(paths.repo_root(), target=args.target)
    count = stage_inputs(pins, archive=Archive(*r2.credentials()),
                         store=Store(args.store.resolve()) if args.store else None,
                         payload=args.payload.resolve() if args.payload else None)
    print(f"Verified {count} unique pinned inputs in R2.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
