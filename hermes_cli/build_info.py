"""Exact, offline build identity for Hermes Agent.

Wheel, sdist, Docker, Desktop, and release artifacts carry one
package-relative ``hermes_cli/_build_metadata.json`` file. Its optional
``source_revision`` is authoritative only when it is a full lowercase
40-character Git SHA. A source checkout without embedded metadata falls back
to its full ``HEAD`` only while the working tree is clean. Resolution performs
no network request or Git fetch.

The old root-level ``.hermes_build_sha`` file remains a read-only compatibility
fallback for short support displays from pre-migration Docker images; it never
feeds the exact ``source_revision`` contract.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Optional

# Path is resolved relative to this module so it works regardless of cwd —
# matches the pattern used by ``banner._resolve_repo_dir``.
_BUILD_SHA_FILE = Path(__file__).parent.parent / ".hermes_build_sha"
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_BUILD_METADATA_RELATIVE_PATH = Path("hermes_cli") / "_build_metadata.json"
_FULL_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _read_embedded_source_revision(project_root: Path) -> tuple[bool, Optional[str]]:
    metadata_file = project_root / _BUILD_METADATA_RELATIVE_PATH
    if not metadata_file.is_file():
        return False, None
    try:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True, None
    if not isinstance(payload, dict):
        return True, None
    revision = payload.get("source_revision")
    if not isinstance(revision, str) or not _FULL_SOURCE_REVISION.fullmatch(revision):
        return True, None
    return True, revision


def validate_source_revision(value: object) -> Optional[str]:
    """Return ``value`` only when it is an exact lowercase Git SHA."""
    if not isinstance(value, str) or not _FULL_SOURCE_REVISION.fullmatch(value):
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


def get_source_revision(project_root: Path | str | None = None) -> Optional[str]:
    """Return the exact full revision for this Hermes artifact, if provable.

    Packaged artifacts carry ``hermes_cli/_build_metadata.json``. A present
    metadata file is authoritative and never falls through to Git when its
    contents are malformed. Source checkouts without embedded metadata report
    their full ``HEAD`` only while the checkout is clean.
    """
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    metadata_present, embedded_revision = _read_embedded_source_revision(root)
    if metadata_present:
        return embedded_revision

    status = _run_git(root, "status", "--porcelain", "--untracked-files=normal")
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
        checkout_revision = get_source_revision(root)
        if asserted_revision is None:
            return checkout_revision
        return asserted if asserted == checkout_revision else None

    return asserted


def get_build_sha(
    short: int = 8,
    project_root: Path | str | None = None,
) -> Optional[str]:
    """Return the artifact SHA, truncated to ``short`` chars, or ``None``.

    New artifacts use the canonical package-relative metadata consumed by
    :func:`get_source_revision`. The root-level Docker stamp remains a
    read-only fallback for artifacts built before that metadata existed.
    """
    root = Path(project_root).resolve() if project_root is not None else _PROJECT_ROOT
    metadata_present, _ = _read_embedded_source_revision(root)
    revision = get_source_revision(project_root)
    if revision is not None:
        return revision[:short] if short and short > 0 else revision
    if metadata_present or project_root is not None:
        return None

    try:
        if not _BUILD_SHA_FILE.is_file():
            return None
        sha = _BUILD_SHA_FILE.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not sha:
        return None
    return sha[:short] if short and short > 0 else sha
