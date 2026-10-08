"""Hermes CLI - Unified command-line interface for Hermes Agent."""

import sys

# Declared for type checkers and the old-updater surface audit; served lazily by __getattr__.
__version__: str
# Served lazily from the install stamp like __version__; the literal is only the
# no-stamp fallback.
__release_date__: str

# Stable releases are tag-based (scripts/release.py cuts them without a repo
# commit), so nothing in the release flow bumps a checked-in date literal: the
# stamp's commitDate -- the tagged commit's own date -- is the date a build
# actually shipped (#135217). Unstamped dev checkouts have no release date to
# report and keep this placeholder.
_RELEASE_DATE_FALLBACK = "2026.9.24"


def _read_release_stamp() -> dict:
    """The install stamp, or ``{}`` when it is absent or unreadable.

    Resolves exactly like ``__getattr__`` must (see its docstring for why pm is
    optional here): ``pm.paths`` when importable, else the stamp beside the
    repo root for the pre-PM editable-finder case.
    """
    from hermes_cli.steward import read_install_stamp
    try:
        from pm.paths import repo_root
    except ModuleNotFoundError as exc:
        if exc.name != "pm" and not (exc.name or "").startswith("pm."):
            raise
        # The old editable finder may not know the new pm package yet.
        import json
        from pathlib import Path
        try:
            data = json.loads(
                (Path(__file__).resolve().parents[1] / "install-stamp.json").read_text(
                    encoding="utf-8-sig"
                )
            )
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}
    return read_install_stamp(repo_root())


def _release_date() -> str:
    """``YYYY.M.D`` for the stamp's commitDate, else the placeholder.

    Every stamp writer records commitDate (epoch seconds) for the commit it
    packaged -- the release tag's commit for a stable build -- formatted in UTC
    without zero padding, matching how the literal has always read.
    """
    commit_date = _read_release_stamp().get("commitDate")
    if isinstance(commit_date, int):
        from datetime import datetime, timezone
        moment = datetime.fromtimestamp(commit_date, tz=timezone.utc)
        return f"{moment.year}.{moment.month}.{moment.day}"
    return _RELEASE_DATE_FALLBACK


def __getattr__(name: str) -> str:
    """Old-updater compat: shipped updaters import ``__version__`` after the checkout swap.

    tests/compat/old_updater_surface.json freezes that import. In-tree code resolves
    identity through hermes_cli.version_info.get_version_info(); this reads only the
    install stamp -- never git -- and keeps the pre-stamp placeholder when a checkout
    has no stamp. ``__release_date__`` reads the same stamp so every banner and
    status surface reports the date the running build was tagged, not a checked-in
    literal the tag-based release flow never bumps.

    Lazy because ``pm`` is not importable when this package loads: a venv
    editable-installed from a pre-PM tree maps only the top-level packages it knew
    at install time, and the repo root reaches ``sys.path`` only once
    ``hermes_bootstrap`` runs -- after this ``__init__``, from ``hermes_cli.main``.
    """
    if name == "__release_date__":
        return _release_date()
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return str(_read_release_stamp().get("baseVersion") or "0.0.0")


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
