"""Preserve pinned binary inputs in GitHub releases and R2, then seed CI consumers."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Protocol
from urllib.parse import quote, urlsplit

# The runner invokes this before setup-pm has installed the checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pm.artifact_mirror import github_asset_url, github_release_tag, github_repository, object_key
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
        if self.kind not in ("library", "license", "tool", "prepared"):
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
        for row_target, prepared in package.get("prepared", {}).items():
            if target is not None and target != row_target:
                continue
            sha256 = prepared["sha256"]
            pins.append(_pin(f"{name}@{row_target}.prepared",
                             {"url": github_asset_url(sha256), "sha256": sha256}, "prepared"))
    if packages is None and target in (None, TERMUX_TARGET):
        table = json.loads((repo / "pm" / "termux_runtime_libs.json").read_text(encoding="utf-8-sig"))
        pins.extend(_pin(name, row, "library") for name, row in table["libs"].items())
        if table.get("licenses") is not None:
            pins.append(_pin("termux-licenses", table["licenses"], "license"))
    if not pins:
        raise ValueError("No pinned HTTP inputs selected")
    return pins


class Mirror(Protocol):
    """One place the pinned bytes are preserved. Objects are named by sha256, so an
    existing object is never rewritten."""

    @property
    def name(self) -> str: ...

    def size(self, sha256: str) -> int | None:
        """Bytes of the complete object, or None on a definite miss. Anything else raises."""

    def put(self, sha256: str, local: Path) -> None:
        """Create the object if absent. Losing a creation race is not an error."""

    def read_back(self, sha256: str, destination: Path, size: int) -> None:
        """Download the way consumers do and verify both size and sha256."""


@dataclass(frozen=True)
class R2Mirror:
    creds: dict[str, str]
    base: str
    bucket: str
    name: str = "R2"

    def size(self, sha256: str) -> int | None:
        key = object_key(sha256)
        url = f"{self.base}/{self.bucket}/{r2.encode_key_path(key)}"
        try:
            head = r2.signed_request("HEAD", url, creds=self.creds, now=r2.amz_timestamp())
        except r2.R2RequestError as exc:
            if exc.status != 404:
                raise
            return None
        length = head.header("content-length")
        if length is None:
            raise ValueError(f"R2 did not report the input size: {url}")
        return int(length)

    def put(self, sha256: str, local: Path) -> None:
        r2.put_object(self.creds, self.base, self.bucket, object_key(sha256), str(local), r2.amz_timestamp(),
                      "application/octet-stream", conditions={"If-None-Match": "*"})

    def read_back(self, sha256: str, destination: Path, size: int) -> None:
        r2.download_object(self.creds, self.base, self.bucket, object_key(sha256), destination,
                           r2.amz_timestamp(), expected_size=size, expected_sha256=sha256)


class GhError(RuntimeError):
    def __init__(self, args: tuple[str, ...], proc: subprocess.CompletedProcess):
        self.stderr = proc.stderr.strip()
        super().__init__(f"gh {' '.join(args)} -> {proc.returncode}: {self.stderr[-400:]}")


class GhApi(Protocol):
    def release_assets(self, tag: str) -> list[dict] | None:
        """Every asset (id, name, size, state) of the release, or None when it does not exist."""

    def create_release(self, tag: str) -> None: ...

    def upload(self, tag: str, path: Path) -> None: ...

    def delete_asset(self, asset_id: int) -> None: ...


class GhCli:
    """The gh CLI, authenticated by GH_TOKEN, scoped to one repository."""

    def __init__(self, repository: str, run=subprocess.run):
        self.repository = repository
        self._run = run

    def _gh(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = self._run(["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if check and proc.returncode != 0:
            raise GhError(args, proc)
        return proc

    def release_assets(self, tag: str) -> list[dict] | None:
        found = self._gh("api", f"repos/{self.repository}/releases/tags/{tag}", check=False)
        if found.returncode != 0:
            if "HTTP 404" in found.stderr:
                return None
            raise GhError((f"api releases/tags/{tag}",), found)
        release_id = json.loads(found.stdout)["id"]
        # The release object's own asset list is not guaranteed complete; the paginated endpoint is.
        pages = json.loads(self._gh("api", "--paginate", "--slurp",
                                    f"repos/{self.repository}/releases/{release_id}/assets?per_page=100").stdout)
        return [asset for page in pages for asset in page]

    def create_release(self, tag: str) -> None:
        # On a fork with no stable release, --latest=false alone still leaves
        # an input shard as /releases/latest. Prereleases cannot win that slot.
        self._gh("release", "create", tag, "--repo", self.repository, "--latest=false", "--prerelease",
                 "--title", f"Pinned inputs {tag.rsplit('-', 1)[-1]}",
                 "--notes", "Content-addressed copies of reviewed pm inputs, named by sha256. "
                            "Never edit or delete assets; scripts/ci/archive_inputs.py owns this release.")

    def upload(self, tag: str, path: Path) -> None:
        # No --clobber: an existing name means the bytes are already preserved.
        self._gh("release", "upload", tag, str(path), "--repo", self.repository)

    def delete_asset(self, asset_id: int) -> None:
        self._gh("api", "-X", "DELETE", f"repos/{self.repository}/releases/assets/{asset_id}")


class GitHubMirror:
    """Sharded GitHub releases (`pm.artifact_mirror.github_release_tag`), one asset per sha256."""

    name = "GitHub"

    def __init__(self, api: GhApi, asset_url: Callable[[str], str] = github_asset_url, *, uploads: int = 2):
        self._api = api
        self._asset_url = asset_url
        self._lock = threading.Lock()
        self._listings: dict[str, dict[str, dict] | None] = {}
        # Creating many assets at once trips GitHub's secondary rate limit.
        self._uploads = threading.Semaphore(uploads)

    def _listing(self, tag: str, *, refresh: bool = False) -> dict[str, dict] | None:
        with self._lock:
            if refresh or tag not in self._listings:
                assets = self._api.release_assets(tag)
                self._listings[tag] = None if assets is None else {a["name"]: a for a in assets}
            return self._listings[tag]

    def size(self, sha256: str) -> int | None:
        tag = github_release_tag(sha256)
        asset = (self._listing(tag) or {}).get(sha256)
        if asset is None:
            return None
        if asset["state"] != "uploaded":
            # An interrupted upload leaves a named stub that blocks creating the real asset.
            self._api.delete_asset(asset["id"])
            with self._lock:
                self._listings.pop(tag, None)
            return None
        return asset["size"]

    def put(self, sha256: str, local: Path) -> None:
        tag = github_release_tag(sha256)
        with self._uploads:
            if self._listing(tag) is None:
                try:
                    self._api.create_release(tag)
                except GhError:
                    if self._listing(tag, refresh=True) is None:  # a racing worker made it
                        raise
                self._listing(tag, refresh=True)
            # gh names the asset after the file, and the name IS the digest.
            with tempfile.TemporaryDirectory(dir=local.parent) as scratch:
                named = Path(scratch) / sha256
                try:
                    os.link(local, named)
                except OSError:
                    shutil.copyfile(local, named)
                try:
                    self._api.upload(tag, named)
                except GhError:
                    # Lost a race to an identical, immutable asset: the read-back verifies it.
                    if sha256 not in (self._listing(tag, refresh=True) or {}):
                        raise
            with self._lock:
                self._listings.pop(tag, None)

    def read_back(self, sha256: str, destination: Path, size: int) -> None:
        Download([Source(self._asset_url(sha256), destination, sha256)],
                 partials_dir=destination.parent / "partials").run()
        if destination.stat().st_size != size:
            raise ValueError(f"GitHub asset {sha256} is {destination.stat().st_size} bytes, expected {size}")


@dataclass(frozen=True)
class Archive:
    mirrors: tuple[Mirror, ...]

    def fetch(self, pin: InputPin, destination: Path) -> str:
        """Only a miss on every mirror permits an upstream download. Every mirror ends up
        holding the bytes, and every copy is read back the way consumers read it."""
        sha = pin.sha256
        object_key(sha)
        sizes = [mirror.size(sha) for mirror in self.mirrors]
        source = next((i for i, size in enumerate(sizes) if size is not None), None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".input-archive-", dir=destination.parent) as temporary:
            work = Path(temporary)
            local = work / "input"
            if source is None:
                origin = _download_upstream(pin, local)
            else:
                held = sizes[source]
                assert held is not None
                origin = self.mirrors[source].name
                self.mirrors[source].read_back(sha, local, held)
            size = local.stat().st_size
            for mirror, existing in zip(self.mirrors, sizes):
                if existing is None:
                    mirror.put(sha, local)
            for i, (mirror, existing) in enumerate(zip(self.mirrors, sizes)):
                if i != source:
                    mirror.read_back(sha, work / f"verified-{i}", size if existing is None else existing)
            local.replace(destination)
        return origin


def archive_from_env(env: Mapping[str, str] = os.environ) -> Archive:
    """Publish only from the configured mirror repo; R2 stays optional."""
    mirrors: list[Mirror] = []
    repository = github_repository()
    if env.get("GH_TOKEN") and env.get("GITHUB_REPOSITORY", "").lower() == repository.lower():
        mirrors.append(GitHubMirror(GhCli(repository)))
    if env.get("CLOUDFLARE_R2_ACCOUNT_ID"):
        mirrors.append(R2Mirror(*r2.credentials()))
    if not mirrors:
        raise ValueError("Pinned inputs need GitHub release access or R2 credentials")
    return Archive(tuple(mirrors))


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
                if pin.kind in ("library", "license") and payload is not None:
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

    # A release upload may not be readable from every CDN edge immediately.
    # Unbounded workers flood public readback with transient 500s.
    with ThreadPoolExecutor(max_workers=min(2, len(groups))) as pool:
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
    archive = archive_from_env()
    count = stage_inputs(pins, archive=archive,
                         store=Store(args.store.resolve()) if args.store else None,
                         payload=args.payload.resolve() if args.payload else None)
    print(f"Verified {count} unique pinned inputs in {', '.join(m.name for m in archive.mirrors)}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
