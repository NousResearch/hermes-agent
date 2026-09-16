"""Release-channel target resolution for the transactional update pipeline.

The updater keeps branch and release targets separate.  This module owns the
small, side-effect-free argument/config policy and the official-release lookup
contract; checkout mutation remains in ``update_cmd.py`` so it shares the
existing backup, lock, syntax, restart, and receipt stages.

Channel policy (installation-scoped, root decision 1): the channel lives in
``get_default_hermes_root()/update-channel.json`` (see ``update_channel.py``),
not in per-profile config.  ``--branch`` is the one-shot developer override and
never wins over an explicit release request; ``--channel`` persists the record
and selects this run's target kind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

RELEASE_LATEST = "latest"


def _resolve_release_request(args: Any) -> str | None:
    """Return an explicit ``--release`` request, or ``None`` when absent."""
    raw = getattr(args, "release", None)
    if raw is None:
        return None
    value = str(raw).strip()
    return RELEASE_LATEST if not value or value.lower() == RELEASE_LATEST else value


def _configured_release_request(args: Any) -> str | None:
    """Resolve the release target for this run without touching a checkout.

    Precedence: explicit ``--release`` > explicit ``--channel`` > explicit
    ``--branch`` (developer override, wins only as a one-shot branch target) >
    the persisted channel record (default stable).  Only reads; persistence of
    an explicit ``--channel`` is the caller's job (it must happen exactly once,
    with the run continuing on the selected channel).
    """
    explicit = _resolve_release_request(args)
    if explicit is not None:
        return explicit

    channel = str(getattr(args, "channel", "") or "").strip().lower()
    if channel in {"beta", "main", "fast", "fast-track", "fast_track"}:
        return None
    if channel in {"stable", "release", "stable-tags", "stable_tag"}:
        return RELEASE_LATEST

    if getattr(args, "branch", None):
        return None

    # No explicit flag: the install-scoped record decides; missing = stable.
    from hermes_cli.update_channel import read_update_channel

    return RELEASE_LATEST if read_update_channel() == "stable" else None


def _resolve_requested_channel(args: Any) -> str | None:
    """Explicitly requested ``--channel`` value normalized to stable/beta, or None."""
    raw = str(getattr(args, "channel", "") or "").strip().lower()
    if raw in {"stable", "release"}:
        return "stable"
    if raw in {"beta", "main"}:
        return "beta"
    return None


def validate_release_commit(value: Any) -> str | None:
    """Validate an optional exact release SHA, preserving its supplied value."""
    if value is None:
        return None
    raw = str(value).strip()
    if len(raw) != 40 or any(char not in "0123456789abcdefABCDEF" for char in raw):
        raise ValueError("--release-commit must be a full 40-character Git commit SHA")
    return raw


def resolve_official_release_target(
    repo_dir: Path,
    requested_tag: str | None,
    expected_commit: Any = None,
) -> tuple[str, str]:
    """Resolve and verify a published official release, returning ``(tag, sha)``.

    The GitHub Releases API rejects drafts/prereleases and the resolver then
    verifies the exact tag SHA from the canonical repository.  No origin/main
    fallback is permitted.
    """
    from hermes_cli.stable_update import resolve_official_release

    checked_commit = validate_release_commit(expected_commit)
    result = resolve_official_release(
        Path(repo_dir),
        requested_tag=None if requested_tag in {None, "", RELEASE_LATEST} else requested_tag,
    )
    tag = result.get("tag")
    commit = result.get("commit")
    error = result.get("error")
    if error or not isinstance(tag, str) or not tag or not isinstance(commit, str) or not commit:
        raise ValueError(str(error or "No official release found."))
    if checked_commit is not None and commit.lower() != checked_commit.lower():
        raise ValueError(f"Release {tag} no longer resolves to the checked commit.")
    return tag, commit


__all__ = [
    "RELEASE_LATEST",
    "_configured_release_request",
    "_resolve_release_request",
    "_resolve_requested_channel",
    "resolve_official_release_target",
    "validate_release_commit",
]

