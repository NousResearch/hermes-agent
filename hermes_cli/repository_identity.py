"""Fail-closed repository identity checks for workspace resolution.

Filesystem shape (for example, merely having ``pyproject.toml``) is not a
repository identity. Callers provide the expected manifest name and source
markers for the target repository.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from typing import Any, Iterable


class RepositoryIdentityError(ValueError):
    """Raised when a candidate cannot be proven to be the requested repo."""


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _git_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return _canonical(Path(result.stdout.strip()))


def _manifest_name(root: Path, manifest: str) -> str | None:
    path = root / manifest
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project")
    return project.get("name") if isinstance(project, dict) else None


def validate_repository_identity(
    candidate: str | Path,
    *,
    expected_manifest_name: str,
    required_markers: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate and return auditable identity evidence for ``candidate``.

    The candidate and its Git root are canonicalized before comparison. A
    linked worktree therefore cannot evade the manifest/marker checks through
    a symlink or by pointing at its common Git directory. No best-effort
    fallback is performed: every missing or mismatched fact is reported.
    """
    requested = _canonical(Path(candidate))
    root = _git_root(requested)
    missing: list[str] = []
    if root is None:
        missing.append("canonical Git root")
        root = requested
    manifest = "pyproject.toml"
    actual_name = _manifest_name(root, manifest)
    if actual_name != expected_manifest_name:
        missing.append(
            f"{manifest} [project].name={expected_manifest_name!r} "
            f"(found {actual_name!r})"
        )
    markers = tuple(str(marker) for marker in required_markers)
    marker_paths: list[tuple[str, Path]] = []
    invalid_markers: list[str] = []
    for marker in markers:
        marker_path = Path(marker)
        resolved_marker = (root / marker_path).resolve(strict=False)
        if marker_path.is_absolute() or ".." in marker_path.parts:
            invalid_markers.append(marker)
            continue
        try:
            resolved_marker.relative_to(root)
        except ValueError:
            invalid_markers.append(marker)
            continue
        marker_paths.append((marker, resolved_marker))
    if invalid_markers:
        raise RepositoryIdentityError(
            "BLOCKED/needs-input: repository identity required markers must be "
            "relative paths contained by the canonical Git root; invalid markers: "
            + ", ".join(repr(marker) for marker in invalid_markers)
        )
    absent = [marker for marker, path in marker_paths if not path.exists()]
    missing.extend(f"required source marker {marker!r}" for marker in absent)
    evidence = {
        "candidate": str(requested),
        "git_root": str(root) if root else None,
        "manifest": manifest,
        "manifest_name": actual_name,
        "expected_manifest_name": expected_manifest_name,
        "required_markers": list(markers),
        "missing": missing,
    }
    if missing:
        raise RepositoryIdentityError(
            "BLOCKED/needs-input: repository identity could not be established; "
            f"candidate {requested} is missing or mismatched: "
            + "; ".join(missing)
            + ". Candidates must be supplied with an explicit expected identity. "
            f"Evidence: {evidence}"
        )
    return evidence


def select_repository_candidate(
    candidates: Iterable[str | Path],
    *,
    expected_manifest_name: str,
    required_markers: Iterable[str] = (),
) -> tuple[Path, dict[str, Any]]:
    """Select exactly one identity-validated candidate, or fail closed."""
    candidate_list = [_canonical(Path(candidate)) for candidate in candidates]
    accepted: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[str] = []
    for candidate in candidate_list:
        try:
            accepted.append((candidate, validate_repository_identity(
                candidate,
                expected_manifest_name=expected_manifest_name,
                required_markers=required_markers,
            )))
        except RepositoryIdentityError as exc:
            rejected.append(f"{candidate}: {exc}")
    if len(accepted) == 1:
        return accepted[0]
    if len(accepted) > 1:
        details = ", ".join(str(candidate) for candidate, _ in accepted)
        reason = f"ambiguous validated candidates: {details}"
    else:
        reason = "no candidate established the requested identity"
    raise RepositoryIdentityError(
        "BLOCKED/needs-input: " + reason + "; candidates and identity evidence: "
        + " | ".join(rejected or [str(candidate) for candidate in candidate_list])
    )
