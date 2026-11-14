"""Hermes CLI - Unified command-line interface for Hermes Agent."""

import sys

# Last resort only: an install with neither a usable stamp nor git has no date to
# derive from. Every stamped/checked-out install overrides this through __getattr__
# below, which reads the commit date the updater rewrites on each release.
_FALLBACK_RELEASE_DATE = "2026.10.23"
# Declared for type checkers and the old-updater surface audit; served lazily by __getattr__.
__version__: str
__release_date__: str


def _release_date_from_commit(commit_date: int) -> str:
    """Format an epoch commit date as the banner's ``YYYY.M.D`` (UTC)."""
    from datetime import UTC, datetime

    moment = datetime.fromtimestamp(commit_date, tz=UTC)
    return f"{moment.year}.{moment.month}.{moment.day}"


def _release_date() -> str:
    """The installed version's release date, derived so it tracks every upgrade.

    Nothing bumps a date literal when the release pipeline bumps the version, so a
    literal here went stale on the first semver release: v0.22.5 published
    2026-10-23 still printed ``(2026.9.24)``. The install stamp carries
    ``commitDate`` and ``scripts/write_install_stamp.py`` rewrites it on every
    install/update, so deriving from it stays in step with the release for good.

    Resolution mirrors ``version_info``: install stamp first, live git second. Both
    paths are guarded because this runs on lazy ``--version`` paths where a broken
    provenance import must not take the banner down.
    """
    try:
        from hermes_cli.version_info import get_version_info

        commit_date = get_version_info().commit_date
        if isinstance(commit_date, int):
            return _release_date_from_commit(commit_date)
    except Exception:
        # Pre-bootstrap paths may not import the full provenance stack.
        pass
    try:
        import json
        from pathlib import Path

        stamp = json.loads((Path(__file__).resolve().parents[1] / "install-stamp.json").read_text(encoding="utf-8-sig"))
        commit_date = stamp.get("commitDate")
        if isinstance(commit_date, int):
            return _release_date_from_commit(commit_date)
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return _FALLBACK_RELEASE_DATE


def __getattr__(name: str) -> str:
    """Old-updater compat and lazy identity: served here, never a module global.

    ``__version__``: shipped updaters import it after the checkout swap, and
    tests/compat/old_updater_surface.json freezes that import. ``__release_date__``:
    derived from the install stamp so it cannot drift from the release it labels.

    In-tree code resolves identity through hermes_cli.version_info.get_version_info();
    this reads only the install stamp -- never git -- and keeps the pre-stamp
    placeholder when a checkout has no stamp.

    Lazy because ``pm`` is not importable when this package loads: a venv
    editable-installed from a pre-PM tree maps only the top-level packages it knew
    at install time, and the repo root reaches ``sys.path`` only once
    ``hermes_bootstrap`` runs -- after this ``__init__``, from ``hermes_cli.main``.
    """
    if name == "__release_date__":
        return _release_date()
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from hermes_cli.steward import read_install_stamp
    try:
        from pm.paths import repo_root
    except ModuleNotFoundError as exc:
        if exc.name != "pm.paths" and not "pm.paths".startswith(f"{exc.name}."):
            raise
        # The old editable finder may not know the new pm package yet.
        import json
        from pathlib import Path
        try:
            stamp = json.loads((Path(__file__).resolve().parents[1] / "install-stamp.json").read_text(encoding="utf-8-sig"))
            return str(stamp.get("baseVersion") or "0.0.0")
        except (OSError, ValueError, AttributeError):
            return "0.0.0"
    return str(read_install_stamp(repo_root()).get("baseVersion") or "0.0.0")


def _ensure_utf8() -> bool:
    """Force UTF-8 stdout/stderr to prevent UnicodeEncodeError crashes; True when a stream was repaired.

    The CLI prints box-drawing characters and the ☤ glyph in the setup wizard, doctor, and status
    banners; under a non-UTF-8 codec that raises before the command can even start (e.g.
    `hermes setup` on a fresh Pi).
    """
    repaired = False
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            if (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8":
                continue
            # Preferred: reconfigure in place, preserving object identity so code already holding
            # a reference to the old sys.stdout benefits from the repair too.
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
            else:
                # No reconfigure(): reopen the fd as UTF-8 (closefd=False keeps the original fd open).
                new_stream = open(stream.fileno(), "w", encoding="utf-8", errors="replace",  # windows-footgun: ok (stdout re-open for write, not a read)
                                  buffering=1, closefd=False)
                setattr(sys, stream_name, new_stream)
            repaired = True
        except (AttributeError, OSError, ValueError):
            pass
    return repaired


# Import repairs only this process's streams. Gateway, compute host, and test code import this
# package as a library; rewriting their os.environ would leak into every child they spawn, so the
# child-process UTF-8 hint is applied by the CLI entry point (hermes_cli.main.main) instead.
_stdio_repaired = _ensure_utf8()
