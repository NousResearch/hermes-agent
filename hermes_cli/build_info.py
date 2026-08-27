"""
Baked-in build metadata for Hermes Agent.

Source installs report their git revision live via ``git rev-parse`` (see
``hermes_cli/dump.py`` and ``hermes_cli/banner.py``).  That doesn't work inside
the published Docker image because ``.dockerignore`` excludes ``.git``, so
those callsites fall back to ``"(unknown)"`` / drop the banner suffix entirely.

To make ``hermes dump`` and the startup banner identify the exact commit the
image was built from, the Docker build writes the build-time ``$HERMES_GIT_SHA``
arg into ``<project_root>/.hermes_build_sha``.  This module is the single
read-side helper consumed by both callsites — keeping the lookup in one place
so the file path and missing-file behaviour stay consistent.

Behaviour:

- Returns ``None`` when the file is absent.  Source installs and dev images
  built without the ``HERMES_GIT_SHA`` build-arg fall through to live-git
  resolution in the caller, so non-Docker installs are unaffected.
- Returns ``None`` on any IO / decoding error.  The build-sha is a nice-to-have
  for support triage; nothing in the CLI is allowed to crash because of it.
- Truncates to ``short`` characters (default 8) to match the format used by
  ``git rev-parse --short=8`` throughout the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Path is resolved relative to this module so it works regardless of cwd —
# matches the pattern used by ``banner._resolve_repo_dir``.
_BUILD_SHA_FILE = Path(__file__).parent.parent / ".hermes_build_sha"


_code_identity_cache: Optional[dict] = None


def _resolve_git_head_sha(project_root: Path) -> Optional[str]:
    """Resolve the checkout's HEAD commit sha by reading .git directly.

    Deliberately NOT ``git rev-parse`` in a subprocess: this helper runs
    inside library paths (gateway runtime-status writes, update receipts)
    where spawning processes is both slow and hostile to tests that mock
    ``subprocess.run`` tightly (call-count asserts, sequenced side effects).
    Handles regular checkouts, worktrees/submodules (``.git`` file with a
    ``gitdir:`` pointer + ``commondir``), loose refs, and packed-refs.
    Returns None on any failure.
    """
    try:
        git_path = project_root / ".git"
        if git_path.is_file():
            # Worktree/submodule: ".git" is a pointer file.
            pointer = git_path.read_text(encoding="utf-8", errors="replace").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = Path(pointer[len("gitdir:"):].strip())
            if not git_dir.is_absolute():
                git_dir = (project_root / git_dir).resolve()
        elif git_path.is_dir():
            git_dir = git_path
        else:
            return None

        # Refs live in the COMMON git dir for worktrees.
        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.is_file():
            rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
            common = Path(rel)
            if not common.is_absolute():
                common = (git_dir / common).resolve()
            common_dir = common

        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if not head.startswith("ref:"):
            # Detached HEAD: the file holds the sha itself.
            return head if len(head) == 40 else None
        ref_name = head[len("ref:"):].strip()

        loose = common_dir / ref_name
        if loose.is_file():
            sha = loose.read_text(encoding="utf-8", errors="replace").strip()
            return sha if len(sha) == 40 else None

        packed = common_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "^")):
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[1].strip() == ref_name:
                    sha = parts[0].strip()
                    return sha if len(sha) == 40 else None
    except Exception:
        return None
    return None


def get_code_identity(refresh: bool = False) -> dict:
    """Return the running checkout's code identity as a dict.

    Shape: ``{"sha": full-or-short sha | None, "short_sha": str | None,
    "version": pyproject version | None, "source": "git" | "build-file" |
    "unknown"}``.

    Resolution order mirrors the banner/dump callsites: live ``git
    rev-parse`` for source installs, the baked ``.hermes_build_sha`` for
    Docker images (no ``.git`` inside the published image), else unknown.

    Cached per process — code identity cannot change while a process is
    running (an updated checkout requires a restart to take effect, which
    is exactly the property the fleet version verification relies on).
    Never raises; every field degrades to ``None`` independently.
    """
    global _code_identity_cache
    if _code_identity_cache is not None and not refresh:
        return dict(_code_identity_cache)

    sha: Optional[str] = None
    source = "unknown"
    project_root = Path(__file__).parent.parent
    resolved = _resolve_git_head_sha(project_root)
    if resolved:
        sha = resolved
        source = "git"
    if sha is None:
        baked = get_build_sha(short=0)
        if baked:
            sha = baked
            source = "build-file"

    version: Optional[str] = None
    try:
        import tomllib

        with open(project_root / "pyproject.toml", "rb") as fh:  # windows-footgun: ok — binary mode, tomllib requires bytes
            raw_version = tomllib.load(fh).get("project", {}).get("version")
        version = str(raw_version) if raw_version else None
    except Exception:
        version = None

    _code_identity_cache = {
        "sha": sha,
        "short_sha": sha[:8] if sha else None,
        "version": version,
        "source": source,
    }
    return dict(_code_identity_cache)


def get_build_sha(short: int = 8) -> Optional[str]:
    """Return the baked-in build SHA, truncated to ``short`` chars, or None.

    Reads ``<project_root>/.hermes_build_sha`` if present.  The file is
    written by the Dockerfile's ``HERMES_GIT_SHA`` build-arg and contains
    the full 40-character commit hash on a single line.
    """
    try:
        if not _BUILD_SHA_FILE.is_file():
            return None
        sha = _BUILD_SHA_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not sha:
        return None
    return sha[:short] if short and short > 0 else sha
