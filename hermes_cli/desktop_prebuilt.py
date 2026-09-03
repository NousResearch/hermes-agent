"""Prebuilt Desktop artifact resolution for ``hermes update``.

When a packaged Desktop app is installed, ``hermes update`` used to always
rebuild it from source (``tsc`` / ``vite`` / electron-builder). That exposes
every user to the full source-build failure surface. This module is the
first rung of a resolution ladder:

  1. Artifact keyed to the current core commit SHA
  2. Artifact for the nearest tagged release within the compatibility window
  3. None — caller falls back to a local source rebuild

The artifact index (schema_version 1) is either fetched from
``desktop.prebuilt_artifacts.index_url`` or discovered from GitHub Releases
(``desktop-index.json`` assets). Every install verifies the published
sha256 before extraction, then stage-and-swaps through the existing
``_swap_staged_desktop_app`` helper so a failed unpack never removes the
working app.

Dev checkouts, forks, and unsupported archs simply miss every rung and
keep the source rebuild. No new ``HERMES_*`` env vars — settings live
under ``desktop.prebuilt_artifacts`` in config.yaml.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_COMPATIBILITY_WINDOW = 64
DEFAULT_RELEASES_API = (
    "https://api.github.com/repos/NousResearch/hermes-agent/releases"
)
INDEX_ASSET_NAME = "desktop-index.json"
USER_AGENT = "hermes-cli-desktop-prebuilt"
HTTP_TIMEOUT_SEC = 15

FetchBytes = Callable[[str], bytes]
RunGit = Callable[[list[str], Path], Optional[str]]


# --------------------------------------------------------------------------------------
# Index contract
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DesktopArtifact:
    commit: str
    platform: str
    arch: str
    url: str
    sha256: str
    tag: str = ""
    filename: str = ""

    def identity(self) -> tuple[str, str, str]:
        return (self.commit, self.platform, self.arch)


@dataclass
class ArtifactIndex:
    schema_version: int
    compatibility_window: int
    artifacts: list[DesktopArtifact]


def normalize_commit(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    sha = value.strip().lower()
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        return None
    return sha


def normalize_sha256(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return None
    return digest


def host_platform(system: Optional[str] = None) -> str:
    raw = (system or sys.platform).lower()
    if raw.startswith("linux"):
        return "linux"
    if raw == "darwin":
        return "darwin"
    if raw.startswith("win"):
        return "win32"
    return raw


def host_arch(machine: Optional[str] = None) -> str:
    raw = (machine or platform.machine() or "").lower()
    if raw in ("x86_64", "amd64", "x64"):
        return "x64"
    if raw in ("aarch64", "arm64"):
        return "arm64"
    return raw or "unknown"


def parse_index(payload: object, *, default_window: int = DEFAULT_COMPATIBILITY_WINDOW) -> ArtifactIndex:
    """Parse and validate an index document. Unknown/invalid rows are dropped."""
    if not isinstance(payload, dict):
        return ArtifactIndex(SCHEMA_VERSION, default_window, [])
    try:
        schema = int(payload.get("schema_version") or SCHEMA_VERSION)
    except (TypeError, ValueError):
        schema = SCHEMA_VERSION
    try:
        window = int(payload.get("compatibility_window") or default_window)
    except (TypeError, ValueError):
        window = default_window
    if window < 0:
        window = 0
    artifacts: list[DesktopArtifact] = []
    raw_rows = payload.get("artifacts") or []
    if not isinstance(raw_rows, list):
        raw_rows = []
    for row in raw_rows:
        parsed = _parse_artifact_row(row)
        if parsed is not None:
            artifacts.append(parsed)
    return ArtifactIndex(schema, window, artifacts)


def _parse_artifact_row(row: object) -> Optional[DesktopArtifact]:
    if not isinstance(row, dict):
        return None
    commit = normalize_commit(row.get("commit"))
    sha256 = normalize_sha256(row.get("sha256"))
    url = row.get("url")
    plat = row.get("platform")
    arch = row.get("arch")
    if commit is None or sha256 is None:
        return None
    if not isinstance(url, str) or not url.strip():
        return None
    if not isinstance(plat, str) or not plat.strip():
        return None
    if not isinstance(arch, str) or not arch.strip():
        return None
    tag = row.get("tag") if isinstance(row.get("tag"), str) else ""
    filename = row.get("filename") if isinstance(row.get("filename"), str) else ""
    return DesktopArtifact(
        commit=commit,
        platform=plat.strip(),
        arch=arch.strip(),
        url=url.strip(),
        sha256=sha256,
        tag=tag.strip(),
        filename=filename.strip(),
    )


def merge_indexes(indexes: Iterable[ArtifactIndex]) -> ArtifactIndex:
    """Last-writer-wins on (commit, platform, arch). Keeps the max window."""
    by_id: dict[tuple[str, str, str], DesktopArtifact] = {}
    window = DEFAULT_COMPATIBILITY_WINDOW
    schema = SCHEMA_VERSION
    for index in indexes:
        schema = max(schema, int(index.schema_version or SCHEMA_VERSION))
        window = max(window, int(index.compatibility_window or 0))
        for art in index.artifacts:
            by_id[art.identity()] = art
    return ArtifactIndex(schema, window, list(by_id.values()))


# --------------------------------------------------------------------------------------
# Resolution ladder
# --------------------------------------------------------------------------------------


def resolve_artifact(
    index: ArtifactIndex,
    *,
    commit: str,
    platform_name: str,
    arch: str,
    distance_fn: Optional[Callable[[str, str], Optional[int]]] = None,
    window: Optional[int] = None,
) -> Optional[DesktopArtifact]:
    """Pick the SHA artifact, else the nearest in-window tag artifact."""
    wanted = normalize_commit(commit)
    if wanted is None:
        return None
    matching = [
        art
        for art in index.artifacts
        if art.platform == platform_name and art.arch == arch
    ]
    for art in matching:
        if art.commit == wanted:
            return art
    limit = index.compatibility_window if window is None else window
    if limit <= 0 or distance_fn is None:
        return None
    best: Optional[DesktopArtifact] = None
    best_distance: Optional[int] = None
    seen: set[str] = set()
    for art in matching:
        if art.commit in seen:
            continue
        seen.add(art.commit)
        distance = distance_fn(art.commit, wanted)
        if distance is None or distance <= 0 or distance > limit:
            continue
        if best_distance is None or distance < best_distance:
            best = art
            best_distance = distance
    return best


def ancestor_distance(ancestor: str, head: str, *, cwd: Path, run_git: Optional[RunGit] = None) -> Optional[int]:
    """Commits from *ancestor* (exclusive) to *head*, if ancestor is on the path.

    ``0`` means the SHAs are the same. ``None`` means *ancestor* is not an
    ancestor of *head* (or git could not answer).
    """
    git = run_git or _run_git
    anc = normalize_commit(ancestor)
    tip = normalize_commit(head)
    if anc is None or tip is None:
        return None
    if anc == tip:
        return 0
    is_anc = git(["merge-base", "--is-ancestor", anc, tip], cwd)
    if is_anc is None:
        return None
    counted = git(["rev-list", "--count", f"{anc}..{tip}"], cwd)
    if counted is None:
        return None
    try:
        n = int(counted.strip())
    except ValueError:
        return None
    return n if n >= 0 else None


# --------------------------------------------------------------------------------------
# Fetch / integrity / extract
# --------------------------------------------------------------------------------------


def default_fetch_bytes(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
        return path.read_bytes()
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"unsupported artifact URL scheme: {parsed.scheme or 'none'}")
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json, application/json, application/octet-stream",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
        return resp.read()


def fetch_index(
    *,
    index_url: str = "",
    fetch_bytes: FetchBytes = default_fetch_bytes,
    releases_api: str = DEFAULT_RELEASES_API,
) -> ArtifactIndex:
    if index_url:
        payload = json.loads(fetch_bytes(index_url).decode("utf-8"))
        return parse_index(payload)
    return fetch_index_from_github_releases(
        fetch_bytes=fetch_bytes, releases_api=releases_api
    )


def fetch_index_from_github_releases(
    *,
    fetch_bytes: FetchBytes = default_fetch_bytes,
    releases_api: str = DEFAULT_RELEASES_API,
    per_page: int = 15,
) -> ArtifactIndex:
    """Discover ``desktop-index.json`` assets on recent GitHub releases."""
    url = releases_api
    if "per_page=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}per_page={per_page}"
    raw = fetch_bytes(url)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        return ArtifactIndex(SCHEMA_VERSION, DEFAULT_COMPATIBILITY_WINDOW, [])
    indexes: list[ArtifactIndex] = []
    for release in payload:
        if not isinstance(release, dict):
            continue
        for asset in release.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            name = asset.get("name") or ""
            asset_url = asset.get("browser_download_url") or ""
            if name != INDEX_ASSET_NAME or not isinstance(asset_url, str) or not asset_url:
                continue
            try:
                doc = json.loads(fetch_bytes(asset_url).decode("utf-8"))
            except Exception as exc:
                logger.debug("desktop-index.json from %s unreadable: %s", asset_url, exc)
                continue
            indexes.append(parse_index(doc))
    if not indexes:
        return ArtifactIndex(SCHEMA_VERSION, DEFAULT_COMPATIBILITY_WINDOW, [])
    return merge_indexes(indexes)


def verify_sha256(data: bytes, expected: str) -> bool:
    digest = normalize_sha256(expected)
    if digest is None:
        return False
    return hashlib.sha256(data).hexdigest() == digest


def extract_zip_bytes(data: bytes, dest: Path) -> None:
    """Extract *data* into *dest*, rejecting zip-slip and absolute entries."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        with zipfile.ZipFile(tmp.name) as zf:
            for info in zf.infolist():
                name = info.filename
                if not name or name.startswith("/") or name.startswith("\\"):
                    raise ValueError(f"refusing absolute zip entry: {name!r}")
                if ".." in Path(name).parts:
                    raise ValueError(f"refusing zip-slip entry: {name!r}")
                target = (dest / name).resolve()
                try:
                    target.relative_to(dest)
                except ValueError as exc:
                    raise ValueError(f"refusing zip-slip entry: {name!r}") from exc
            zf.extractall(dest)


def flatten_single_wrapper(dest: Path) -> None:
    """If the zip wrapped the unpacked dir in one folder, lift its children up."""
    children = [p for p in dest.iterdir() if p.name not in (".", "..")]
    if len(children) != 1 or not children[0].is_dir():
        return
    wrapper = children[0]
    # Already looks like an electron-builder output dir — leave it.
    if wrapper.name in {
        "linux-unpacked",
        "linux-arm64-unpacked",
        "win-unpacked",
        "win-ia32-unpacked",
        "win-arm64-unpacked",
    } or wrapper.name.startswith("mac"):
        return
    for child in wrapper.iterdir():
        child.rename(dest / child.name)
    wrapper.rmdir()


# --------------------------------------------------------------------------------------
# Config + git + install
# --------------------------------------------------------------------------------------


def load_prebuilt_config() -> dict:
    try:
        from hermes_cli.config import load_config_readonly

        desktop = (load_config_readonly() or {}).get("desktop") or {}
        raw = desktop.get("prebuilt_artifacts") or {}
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def current_head_sha(project_root: Path, *, run_git: Optional[RunGit] = None) -> Optional[str]:
    git = run_git or _run_git
    return normalize_commit(git(["rev-parse", "HEAD"], project_root) or "")


def try_install_prebuilt_desktop(
    desktop_dir: Path,
    *,
    project_root: Path,
    fetch_bytes: FetchBytes = default_fetch_bytes,
    run_git: Optional[RunGit] = None,
    platform_name: Optional[str] = None,
    arch: Optional[str] = None,
    index: Optional[ArtifactIndex] = None,
) -> bool:
    """Download + verify + stage-and-swap a prebuilt app.

    Returns True only when a launchable app was swapped into ``release/``.
    Any miss or failure returns False so the caller can source-rebuild.
    """
    cfg = load_prebuilt_config()
    if cfg.get("enabled", True) is False:
        return False
    head = current_head_sha(project_root, run_git=run_git)
    if head is None:
        return False
    plat = platform_name or host_platform()
    machine = arch or host_arch()
    try:
        window = int(cfg.get("compatibility_window") or DEFAULT_COMPATIBILITY_WINDOW)
    except (TypeError, ValueError):
        window = DEFAULT_COMPATIBILITY_WINDOW
    if index is None:
        index_url = cfg.get("index_url") if isinstance(cfg.get("index_url"), str) else ""
        try:
            index = fetch_index(index_url=index_url.strip(), fetch_bytes=fetch_bytes)
        except Exception as exc:
            logger.debug("desktop prebuilt index fetch failed: %s", exc)
            return False
    git = run_git or _run_git

    def _distance(ancestor: str, tip: str) -> Optional[int]:
        return ancestor_distance(ancestor, tip, cwd=project_root, run_git=git)

    chosen = resolve_artifact(
        index,
        commit=head,
        platform_name=plat,
        arch=machine,
        distance_fn=_distance,
        window=window,
    )
    if chosen is None:
        return False
    return install_artifact(chosen, desktop_dir, project_root=project_root, fetch_bytes=fetch_bytes)


def install_artifact(
    artifact: DesktopArtifact,
    desktop_dir: Path,
    *,
    project_root: Path,
    fetch_bytes: FetchBytes = default_fetch_bytes,
) -> bool:
    """Verify checksum, extract to staging, swap over the live app."""
    try:
        blob = fetch_bytes(artifact.url)
    except Exception as exc:
        logger.debug("desktop prebuilt download failed: %s", exc)
        return False
    if not verify_sha256(blob, artifact.sha256):
        logger.warning("desktop prebuilt checksum mismatch; leaving live app in place")
        return False
    try:
        from hermes_cli.main import (
            _desktop_staging_dir,
            _swap_staged_desktop_app,
            _write_desktop_build_stamp,
        )
    except Exception as exc:
        logger.debug("desktop prebuilt cannot import stage-and-swap helpers: %s", exc)
        return False
    staging = _desktop_staging_dir(desktop_dir)
    try:
        extract_zip_bytes(blob, staging)
        flatten_single_wrapper(staging)
    except Exception as exc:
        logger.warning("desktop prebuilt extract failed; live app kept: %s", exc)
        from hermes_cli.main import _discard_desktop_staging

        _discard_desktop_staging(staging)
        return False
    swapped = _swap_staged_desktop_app(desktop_dir, staging)
    if swapped is None:
        return False
    _write_desktop_build_stamp(project_root, source_mode=False)
    return True


def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        from hermes_cli._subprocess_compat import noninteractive_git_env
    except Exception:
        env = None
    else:
        env = noninteractive_git_env()
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            env=env,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout
