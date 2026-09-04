"""hermes pm update: resolve the latest versions and re-pin the lockfile.

Each Package subclass declares how to find its own latest via
``latest_versions(target)`` (see pm/package.py); this module is the driver:
it intersects those candidate lists across the targets the package serves,
compares against the lockfile, and (in real mode) re-pins + reinstalls.

Two version styles (``Package.version_style``):

  semver — one shared version across targets (node 26.7.0, uv 0.12.3).
          The update is the highest version every relevant target serves.
  minor  — the lockfile ``version`` field is major.minor; each target's
          exact patch lives in ITS artifact urls. Used when the package
          has no shared release cadence across its sources (ffmpeg's posix
          martin-riedl builds vs win32 BtbN autobuilds drift in patch).
          An update moves to the highest MAJOR.MINOR every relevant target
          serves; within it each target pins its own newest patch.

The ``--check`` command is dry-run only: it hits the upstream indexes (small
HTTP GETs) but never writes the lockfile, the store, or the venv, and never
downloads an artifact. Real ``pm update`` re-pins changed packages
(``pm lock``-style artifact hashing), installs them, then refreshes the
venv (uv sync) — and with the dep legs enabled, uv.lock + package-lock.json.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from pm.registry import get_package

# ---------------------------------------------------------------------------
# Version parsing / comparison (pure)
# ---------------------------------------------------------------------------

_VERSION_PART_RE = re.compile(r"\d+|[A-Za-z]+")


def version_key(version: str) -> tuple:
    """A sortable key for a version string. Handles dot-separated numerics,
    +suffixes (python's 3.11.16+20260814, git's 2.53.0+3), and plain build
    numbers (llamacpp's 10362). Non-numeric segments sort after numerics so
    a prerelease never beats its release."""
    key = []
    for chunk in version.replace("-", ".").split("."):
        for m in _VERSION_PART_RE.findall(chunk):
            if m.isdigit():
                key.append((0, int(m)))
            else:
                key.append((1, m))
    return tuple(key)


def minor_of(version: str) -> Optional[tuple[int, int]]:
    """(major, minor) of a semver-ish string; None when not 2-part numeric."""
    nums = [int(m) for m in _VERSION_PART_RE.findall(version) if m.isdigit()]
    if len(nums) < 2:
        return None
    return (nums[0], nums[1])


def version_in_minor(version: str, minor: tuple[int, int]) -> bool:
    return minor_of(version) == minor


def best_in_minor(versions: list[str], minor: tuple[int, int]) -> Optional[str]:
    """The highest version in `versions` whose major.minor == `minor`."""
    best = None
    for v in versions:
        if minor_of(v) == minor and (best is None or version_key(v) > version_key(best)):
            best = v
    return best


# ---------------------------------------------------------------------------
# Per-target intersection (pure)
# ---------------------------------------------------------------------------


@dataclass
class Resolved:
    """What `pm update` decided for ONE package."""

    name: str
    locked: Optional[str]
    style: str
    version: Optional[str] = None  # lockfile label: full (semver) or X.Y (minor)
    per_target: dict[str, str] = field(default_factory=dict)  # target -> exact version
    reason: str = ""  # "up to date" / "no source" / "no shared minor" / ""

    @property
    def changed(self) -> bool:
        if self.version is None:
            return False
        if self.locked is None:
            return True
        if self.style == "minor":
            return minor_of(self.version) != minor_of(self.locked)
        return version_key(self.version) > version_key(self.locked)


def resolve_best(
    name: str,
    targets: list[str],
    latest_by_target: dict[str, list[str]],
    locked: Optional[str],
    style: str,
) -> Resolved:
    """Intersect per-target candidate lists into one update decision.

    ``latest_by_target[target]`` is newest-first. Empty candidates for a
    target mean "no auto-update source" — the target is skipped (a missing
    index must not block the others).
    """
    present = {t: vs for t, vs in latest_by_target.items() if vs}
    if not present:
        return Resolved(name, locked, style, reason="no source")

    if style == "minor":
        # Highest major.minor every target has; per-target exact patch.
        minor_sets = [set(minor_of(v) for v in vs if minor_of(v)) for vs in present.values()]
        common = set.intersection(*minor_sets) if len(minor_sets) > 1 else set(minor_sets[0])
        if not common:
            return Resolved(name, locked, style, reason="no shared minor")
        shared_minor = max(common)
        per_target = {
            t: best_in_minor(vs, shared_minor) for t, vs in present.items()
        }
        return Resolved(
            name,
            locked,
            style,
            version=f"{shared_minor[0]}.{shared_minor[1]}",
            per_target=per_target,
        )

    # semver: highest version every target serves.
    sets = [set(vs) for vs in present.values()]
    common = set.intersection(*sets) if len(sets) > 1 else set(sets[0])
    if not common:
        return Resolved(name, locked, style, reason="no shared version")
    version = max(common, key=version_key)
    return Resolved(name, locked, style, version=version, per_target={t: version for t in present})


def resolve_package(package, targets: list[str], locked: Optional[str]) -> Resolved:
    """Call a package's own latest_versions() per target, then intersect."""
    latest = {
        t: list(package.latest_versions(t, locked=locked) or [])
        for t in targets
    }
    return resolve_best(package.name, targets, latest, locked, package.version_style)


# ---------------------------------------------------------------------------
# Upstream index helpers (small GETs; used by package latest_versions())
# ---------------------------------------------------------------------------

_UA = {"User-Agent": "hermes-pm"}


def _github_headers() -> dict:
    """Headers for GitHub API calls. A GH_TOKEN / GITHUB_TOKEN env var lifts
    the unauthenticated 60 req/hr cap to 5000 — a full `pm update --check`
    across every github-sourced package burns the anonymous budget fast."""
    headers = dict(_UA)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(url: str) -> dict | list:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=_github_headers()), timeout=60
    ) as resp:
        return json.load(resp)


def _get_text(url: str, headers: Optional[dict] = None) -> str:
    hdrs = dict(_UA)
    if headers:
        hdrs.update(headers)
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=hdrs), timeout=60
    ) as resp:
        return resp.read().decode("utf-8", "replace")


# ── llama.app installer bucket (ggml-org/install.sh) ──────────────────────
# The llama-install.sh installer resolves its build version from
# `.../resolve/latest` (a short tag like b10679) and downloads prebuilt
# llama-app binaries from the HF bucket tree {ARCH}/{OS}/{backend}/{CONFIG}/
# llama-app[.exe].zst. pm does NOT fetch those binaries (CONFIG codes are
# hardware-probe-derived, not precomputable, and the llama.cpp GitHub
# releases are our artifact source) — but the bucket's version index is a
# better "what should we be on" signal than scraping every GitHub release:
# it is the installer's own updater pointer, needs no API token, and is
# not rate-limited. We resolve versions from it and still fetch artifacts
# from the llama.cpp GitHub releases (every bucket tag corresponds 1:1 to
# a GitHub release tag, so a bump always has our per-target assets).

_LLAMA_BUCKET = "ggml-org/install.sh"
_LLAMA_BUCKET_API = f"https://huggingface.co/api/buckets/{_LLAMA_BUCKET}"
_LLAMA_BUCKET_RESOLVE = f"https://huggingface.co/buckets/{_LLAMA_BUCKET}/resolve"


def _hf_headers() -> dict:
    """Optional bearer auth for HF bucket fetches (HF_TOKEN), like the
    installer's own requests."""
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def llama_app_latest() -> Optional[str]:
    """The build tag the llama.app installer's `latest` pointer currently
    resolves to — the updater's "next version". Returns the bare build
    number ("10679") or None when unreachable."""
    try:
        text = _get_text(f"{_LLAMA_BUCKET_RESOLVE}/latest", headers=_hf_headers())
    except Exception:
        return None
    # The tag is b<digits> (b10679) — the first digit run is the build
    # number. A \b boundary would fail between the 'b' and the digits.
    m = re.search(r"(\d+)", text)
    return m.group(1) if m else None


def llama_app_bucket_versions() -> list[str]:
    """Build numbers visible in the llama.app bucket, newest-first.

    The HF bucket tree API IGNORES the offset param (verified: every
    offset returns the same first page — the oldest ~1000 paths, sorted
    by path ascending). So the tree can only ever enumerate the OLDEST
    builds, never the newest — the `latest` pointer (llama_app_latest)
    is the authoritative "next version" source. This helper returns what
    the tree CAN see (deduped, sorted by build number descending) as a
    bounded supplement; callers should put llama_app_latest() first.
    """
    versions = []
    seen = set()
    try:
        data = _get_json(f"{_LLAMA_BUCKET_API}/tree?limit=1000&offset=0")
    except Exception:
        return []
    for entry in data or []:
        m = re.match(r"^(b\d+)/", entry.get("path", ""))
        if m:
            tag = m.group(1)[1:]  # strip the leading 'b'
            if tag not in seen:
                seen.add(tag)
                versions.append(int(tag))
    versions.sort(reverse=True)
    return [str(v) for v in versions]


def github_release_tags(repo: str, *, strip_prefix: str = "") -> list[str]:
    """Newest-first release tag names for a GitHub repo (releases, not all
    tags — no drafts/prereleases, and a rolling 'latest' pseudo-release is
    skipped). ``strip_prefix`` removes a tag prefix (e.g. 'v', 'b'). Only
    version-shaped tags survive the strip — cua-driver's repo also tags
    sandbox/experimental builds (cua-driver-rs-vsandbox-v0.4.3) that must
    never appear as update candidates."""
    tags = []
    for page in range(1, 4):  # up to 90 releases — far beyond any cadence
        data = _get_json(f"https://api.github.com/repos/{repo}/releases?per_page=30&page={page}")
        if not data:
            break
        for release in data:
            if release.get("draft") or release.get("prerelease"):
                continue
            tag = release.get("tag_name", "")
            if tag == "latest":
                continue
            if strip_prefix and tag.startswith(strip_prefix):
                tag = tag[len(strip_prefix):]
            if not tag or not tag[0].isdigit():
                continue
            tags.append(tag)
        if len(data) < 30:
            break
    return tags


def npm_dist_tags(name: str) -> dict:
    return _get_json(f"https://registry.npmjs.org/{name}").get("dist-tags", {})


def node_latest_versions() -> list[str]:
    """Newest-first node versions from nodejs.org's index (strip the 'v')."""
    out = []
    for entry in _get_json("https://nodejs.org/dist/index.json"):
        v = entry.get("version", "")
        if v.startswith("v"):
            v = v[1:]
        out.append(v)
    return out


def martin_riedl_index() -> dict[str, dict[str, str]]:
    """ffmpeg.martin-riedl.de index, cached: target -> {version: epoch}.

    The site has no directory listing or API — the root page is the index,
    and it lists the CURRENT build per platform (both snapshot builds like
    N-126314-g... and release builds like 9.0.1). Parse every
    <epoch>_<semver> download dir; snapshots never match the numeric
    pattern. The epoch is needed to reconstruct the download URL at pin
    time, so the index maps version -> epoch per target."""
    cached = _martin_cache.get()
    if cached is not None:
        return cached
    try:
        page = _get_text("https://ffmpeg.martin-riedl.de/")
    except Exception:
        _martin_cache.set({})
        return {}
    out: dict[str, dict[str, str]] = {}
    # /download/<os>/<arch>/<epoch>_<version>/ffmpeg.zip
    for m in re.finditer(
        r"/download/(macos|linux)/(amd64|arm64)/(\d+)_(\d+\.\d+\.\d+)/ffmpeg\.zip",
        page,
    ):
        osname, arch, epoch, version = m.groups()
        target = f"{'darwin' if osname == 'macos' else 'linux'}-{arch}"
        out.setdefault(target, {})[version] = epoch
    _martin_cache.set(out)
    return out


def martin_riedl_versions(target: str) -> list[str]:
    """Newest-first release versions the martin-riedl index lists for `target`."""
    return list((martin_riedl_index().get(target) or {}).keys())


def btbn_index() -> dict[str, list[tuple[str, str]]]:
    """BtbN/FFmpeg-Builds autobuild index, cached: version -> [(tag, asset)].

    The release tag is a dated autobuild (autobuild-2026-08-28-17-08); the
    ffmpeg version lives in the asset names (ffmpeg-n9.0.1-11-g<hash>-
    win64-gpl-9.0.zip). Building the download URL at pin time needs BOTH
    the tag and the full asset name, so the index keeps them per version."""
    cached = _btbn_cache.get()
    if cached is not None:
        return cached
    out: dict[str, list[tuple[str, str]]] = {}
    for page in range(1, 4):
        data = _get_json(f"https://api.github.com/repos/BtbN/FFmpeg-Builds/releases?per_page=30&page={page}")
        if not data:
            break
        for release in data:
            if release.get("draft") or release.get("prerelease"):
                continue
            tag = release.get("tag_name", "")
            if tag == "latest":
                continue
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                m = re.search(r"(?:^|-)n(\d+\.\d+\.\d+)(?:-|$)", name)
                if m and m.group(1) not in out:
                    out[m.group(1)] = [(tag, name)]
        if len(data) < 30:
            break
    _btbn_cache.set(out)
    return out


def btbn_versions() -> list[str]:
    """Newest-first ffmpeg versions BtbN autobuilds currently ship."""
    return list(btbn_index().keys())


def pbs_build_tags(minor: str, target: str) -> list[str]:
    """python-build-standalone release tags (20260814, ...) newest-first that
    actually ship the requested cpython <minor> install_only asset for
    `target`. The release list carries assets + digests, so one fetch per
    page is enough — scan newest-first and stop at the first tag with the
    asset (pbs builds every supported line per release)."""
    triple = {
        "win32-x64": "x86_64-pc-windows-msvc",
        "win32-arm64": "aarch64-pc-windows-msvc",
        "linux-x64": "x86_64-unknown-linux-gnu",
        "linux-arm64": "aarch64-unknown-linux-gnu",
        "darwin-x64": "x86_64-apple-darwin",
        "darwin-arm64": "aarch64-apple-darwin",
    }[target]
    wanted = f"cpython-{minor}."
    for page in range(1, 3):
        data = _get_json(f"https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=30&page={page}")
        if not data:
            break
        for release in data:
            if release.get("draft") or release.get("prerelease"):
                continue
            tag = release.get("tag_name", "")
            if not re.fullmatch(r"\d{8}", tag):
                continue
            if any(
                a["name"].startswith(wanted)
                and a["name"].endswith(f"-{triple}-install_only.tar.gz")
                for a in release.get("assets", [])
            ):
                return [tag]
        if len(data) < 30:
            break
    return []


class _TTL:
    """Tiny per-process cache so --check and the pin step share one fetch."""

    def __init__(self) -> None:
        self._value: object = None
        self._set = False

    def get(self):
        return self._value if self._set else None

    def set(self, value: object) -> None:
        self._value = value
        self._set = True


_martin_cache = _TTL()
_btbn_cache = _TTL()
