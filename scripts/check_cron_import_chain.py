#!/usr/bin/env python3
"""Fail when the cron dispatch import chain does not import.

The cron external worker imports the LIVE checkout (``cron/scheduler.py`` puts
the repo root on ``sys.path``), so one module-level error anywhere in the shared
delivery/agent chain fails EVERY job that touches the gateway delivery path,
at once, on the next tick.

Field incident (2026-10-24 02:16Z): ``agent/auxiliary_client.py`` raised a
``TypeError`` at import time. 17 distinct cron jobs failed in the same second,
each with the opaque::

    Restart-safe cron worker dispatch failed: cron external worker exited
    before ownership acknowledgement (exit 1)

The tree was already broken 33 minutes before the tick, and nothing said so.

This check is the DEPLOYMENT INTERLOCK: when a change touches the chain, the
chain is imported for real, in a SUBPROCESS (the gateway's own ``sys.modules``
is frozen at boot, so an in-process import cannot reveal a fresh error). A
broken chain is a 30-second error at commit time, not a 17-job outage.

It is deliberately scoped to the chain: a change elsewhere in the tree cannot
break the cron delivery path, so it is not probed (and the probe stays cheap
for the changes that do not matter).

Usage: python3 scripts/check_cron_import_chain.py [paths...]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The modules whose import failure kills every cron job on the delivery path.
# Mirrors ``cron/import_smoke.py::SMOKE_CHAIN``.
CHAIN_FILES = (
    "gateway/delivery.py",
    "gateway/session.py",
    "gateway/session_transcript.py",
    "agent/turn_context.py",
    "agent/conversation_compression.py",
    "agent/auxiliary_client.py",
)

# Suppress a deliberate exception with ``# cron-import-chain: ok -- <why>``.
SUPPRESS = "# cron-import-chain: ok"


def _changed_paths(paths: list[str]) -> list[Path]:
    """The files under *paths* that exist, relative to the repo root."""
    out: list[Path] = []
    for raw in paths:
        target = Path(raw)
        if not target.is_absolute():
            target = ROOT / raw
        if target.is_dir():
            out.extend(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
        elif target.is_file():
            out.append(target)
    return out


def _is_chain_file(path: Path) -> bool:
    """Whether *path* is one of the chain modules the cron worker imports."""
    try:
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return False
    return rel in CHAIN_FILES


def _is_suppressed(path: Path) -> bool:
    """Whether the file carries a deliberate-exception marker."""
    try:
        return SUPPRESS in path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False


def main(argv: list[str]) -> int:
    if not argv:
        # No explicit paths: probe the chain anyway. The check is cheap (one
        # subprocess) and a green run on an unchanged tree is the point.
        touched = list(CHAIN_FILES)
    else:
        touched = [
            p.resolve().relative_to(ROOT.resolve()).as_posix()
            for p in _changed_paths(argv)
            if p.is_file() and _is_chain_file(p)
        ]
        if not touched:
            print("check_cron_import_chain: OK (no chain module touched)")
            return 0

    # The smoke probe lives in the tree under test; if it cannot be imported,
    # the chain cannot be checked, and saying so is better than a false green.
    try:
        from cron.import_smoke import check_import_chain
    except ImportError as exc:
        # The probe module itself cannot be imported — that IS the tree
        # problem the check exists to report, so name it and fail.
        print(
            f"check_cron_import_chain: FAILED — the import probe itself could not "
            f"be loaded: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    result = check_import_chain(force=True)
    if result.ok:
        print(f"check_cron_import_chain: OK ({len(touched)} chain module(s) import cleanly)")
        return 0

    print(
        "check_cron_import_chain: FAILED — the cron dispatch import chain does "
        "not import. The cron external worker imports the live checkout, so "
        "every job on the gateway delivery path would fail at once:\n"
        f"  {result.diagnostic()}",
        file=sys.stderr,
    )
    if result.traceback:
        print("\n--- traceback ---\n" + result.traceback, file=sys.stderr)
    for rel in touched:
        path = ROOT / rel
        if path.is_file() and _is_suppressed(path):
            print(
                f"  (suppressed on {rel} — the marker is there, but the chain "
                f"still does not import, so this is NOT actually waived)",
                file=sys.stderr,
            )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
