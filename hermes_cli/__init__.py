"""Hermes CLI - Unified command-line interface for Hermes Agent."""

import sys

__release_date__ = "2026.9.24"
# Declared for type checkers and the old-updater surface audit; served lazily by __getattr__.
__version__: str


def __getattr__(name: str) -> str:
    """Old-updater compat: shipped updaters import ``__version__`` after the checkout swap.

    tests/compat/old_updater_surface.json freezes that import. In-tree code resolves
    identity through hermes_cli.version_info.get_version_info(); this reads only the
    install stamp -- never git -- and keeps the pre-stamp placeholder when a checkout
    has no stamp.

    Lazy because ``pm`` is not importable when this package loads: a venv
    editable-installed from a pre-PM tree maps only the top-level packages it knew
    at install time, and the repo root reaches ``sys.path`` only once
    ``hermes_bootstrap`` runs -- after this ``__init__``, from ``hermes_cli.main``.
    """
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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
