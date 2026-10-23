"""Fail-closed exact source identity for the public status API.

Release install stamps retain upstream's Docker/Nix/Desktop provenance. Nix
wheel builds carry package-relative metadata; unstamped source trees must be
clean. This stricter contract does not change support-display version policy
or the frozen updater shims in build_info.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Optional

# Path is resolved relative to this module so it works regardless of cwd —
# matches the pattern used by ``banner._resolve_repo_dir``.
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_BUILD_METADATA_RELATIVE_PATH = Path("hermes_cli") / "_build_metadata.json"
_FULL_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _read_embedded_source_revision(project_root: Path) -> tuple[bool, Optional[str]]:
    metadata_file = project_root / _BUILD_METADATA_RELATIVE_PATH
    if not metadata_file.is_file():
        return False, None
    try:
        payload = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True, None
    if not isinstance(payload, dict):
        return True, None
    revision = payload.get("source_revision")
    if not isinstance(revision, str) or not _FULL_SOURCE_REVISION.fullmatch(revision):
        return True, None
    return True, validate_source_revision(revision)


def validate_source_revision(value: object) -> Optional[str]:
    """Return ``value`` only when it is an exact lowercase Git SHA."""
    if not isinstance(value, str) or not _FULL_SOURCE_REVISION.fullmatch(value) or value == "0" * 40:
        return None
    return value


def write_build_metadata(
    project_root: Path | str,
    source_revision: object,
) -> Path:
    """Write the canonical package-relative build metadata file."""
    metadata_file = Path(project_root) / _BUILD_METADATA_RELATIVE_PATH
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source_revision": validate_source_revision(source_revision)}
    metadata_file.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_file


def _run_git(project_root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _read_install_source_revision(root: Path) -> tuple[bool, Optional[str]]:
    from pm.paths import install_stamp_path

    stamp = install_stamp_path(root)
    if not stamp.is_file():
        return False, None
    try:
        payload = json.loads(stamp.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True, None
    if not isinstance(payload, dict) or payload.get("dirty") or payload.get("source") == "fallback":
        return True, None
    revision = validate_source_revision(payload.get("commit"))
    if payload.get("payload") == "light" or payload.get("updateMechanism") not in {
        "self", "electron-updater", "app-installer", "external",
    }:
        return True, None
    if payload.get("source") in {"git", "local"} and (root / ".git").exists():
        status = _run_git(root, "--no-optional-locks", "status", "--porcelain", "--untracked-files=normal")
        head = _run_git(root, "rev-parse", "--verify", "HEAD")
        if status is None or status or head != revision:
            return True, None
    return True, revision


def get_source_revision(project_root: Path | str | None = None) -> Optional[str]:
    """Return the exact full revision for this Hermes artifact, if provable.

    Release install stamps take precedence over package-relative metadata.
    A present but invalid stamp or metadata never falls through to Git.
    Unstamped source checkouts report full ``HEAD`` only while clean.
    """
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    stamp_present, stamped_revision = _read_install_source_revision(root)
    if stamp_present:
        return stamped_revision
    metadata_present, embedded_revision = _read_embedded_source_revision(root)
    if metadata_present:
        return embedded_revision
    return _clean_checkout_revision(root)


def _clean_checkout_revision(root: Path) -> Optional[str]:
    if not (root / ".git").exists():
        return None

    status = _run_git(root, "--no-optional-locks", "status", "--porcelain", "--untracked-files=normal")
    if status is None or status:
        return None

    revision = _run_git(root, "rev-parse", "--verify", "HEAD")
    if revision is None or not _FULL_SOURCE_REVISION.fullmatch(revision):
        return None
    return revision


def resolve_build_source_revision(
    project_root: Path | str,
    asserted_revision: object = None,
) -> Optional[str]:
    """Resolve the revision a build may safely embed in a new artifact.

    Existing package metadata is immutable provenance and wins over all other
    inputs. In a Git checkout, a builder assertion is accepted only when it
    matches the checkout's clean full ``HEAD``. A source archive without Git or
    embedded metadata may rely on a strictly validated builder assertion.
    """
    root = Path(project_root).resolve()
    metadata_present, embedded_revision = _read_embedded_source_revision(root)
    if metadata_present:
        return embedded_revision

    asserted = validate_source_revision(asserted_revision)
    if asserted_revision is not None and asserted is None:
        return None

    if (root / ".git").exists():
        checkout_revision = _clean_checkout_revision(root)
        if asserted_revision is None:
            return checkout_revision
        return asserted if asserted == checkout_revision else None

    return asserted

