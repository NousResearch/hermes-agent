"""Wires :mod:`services.hermes.todo_store` into the deterministic scheduled path.

Nothing in the morning-chain previously called the store's reconciliation
API — the one-time seed (:mod:`services.hermes.todo_store_seed`) populated it
once, but tomorrow's fresh journal contract never reached it. This module is
the fix: two operations, run as steps 1 and 3 of ``deploy/bin/morning-chain.sh``
(see that script), that close the loop every day.

  export_open_closed_keys()
      Runs BEFORE journal's morning run. Auto-reopens any snooze whose date
      has passed, then writes the store's current OPEN_KEYS / CLOSED_KEYS
      projections to disk for journal's own contract generator to read back
      into its prompt (so it does not re-propose an already-closed todo, and
      knows which stable keys are still open). This necessarily reflects the
      store as of the end of *yesterday's* :func:`ingest_contract` run --
      there is no way to make it reflect "today" before today's journal run
      has even produced a contract.

  ingest_contract()
      Runs AFTER journal's morning run. Reconciles that day's
      ``journal_brief.latest.json`` (schema v1.2: a stable ``key`` per todo,
      plus a top-level ``resolved_keys`` list) into the store via
      :func:`todo_store.upsert_from_contract`, then closes out anything
      ``resolved_keys`` says journal already resolved on its own.

Both are callable as a CLI (``python -m services.hermes.todo_store_sync
export`` / ``... ingest [--contract PATH] [--dry-run]``) and as plain
functions for testing.

No LLM/agent-client imports: stdlib + services.hermes.todo_store + this
repo's ``utils.atomic_json_write`` only -- part of the deterministic
scheduled path.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

from services.hermes import todo_store
from utils import atomic_json_write

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home)


def default_open_keys_path() -> str:
    return str(_hermes_home() / "contracts" / "todo-open-keys.json")


def default_closed_keys_path() -> str:
    return str(_hermes_home() / "contracts" / "todo-closed-keys.json")


def default_contract_path() -> str:
    return str(_hermes_home() / "contracts" / "journal_brief.latest.json")


def _today() -> str:
    return _dt.date.today().isoformat()


# ---------------------------------------------------------------------------
# export — morning-chain step 1 (before journal runs)
# ---------------------------------------------------------------------------


def export_open_closed_keys(open_path: str, closed_path: str) -> dict:
    """Reopen expired snoozes, then write OPEN_KEYS/CLOSED_KEYS for journal.

    Writes are atomic (temp file + ``os.replace``, via ``utils.atomic_json_write``).
    Never raises: any failure (store unavailable, disk full, etc.) is logged
    and returned as ``{"error": ...}`` rather than propagated -- this is a
    reconciliation step in a scheduled chain, not something that should take
    the chain down.
    """
    try:
        today = _today()
        todo_store.reopen_expired_snoozes(today)
        open_keys = todo_store.get_open_keys()
        closed_keys = todo_store.get_closed_keys()
        atomic_json_write(open_path, open_keys)
        atomic_json_write(closed_path, closed_keys)
        return {"open_count": len(open_keys), "closed_count": len(closed_keys)}
    except Exception as exc:  # pragma: no cover - defensive backstop
        _log.error("todo_store_sync export failed: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# ingest — morning-chain step 3 (after journal runs)
# ---------------------------------------------------------------------------


def _load_contract(contract_path: str) -> tuple[dict | None, str | None]:
    """Read + parse a ``journal_brief.latest.json``-shaped file.

    Returns ``(contract, error)`` -- ``contract`` is ``None`` and ``error`` is
    a human-readable string when the file is missing, unreadable, not valid
    JSON, or not a JSON object at the top level. Never raises.
    """
    try:
        with open(contract_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as exc:
        return None, f"could not read {contract_path}: {exc}"
    try:
        contract = json.loads(raw)
    except ValueError as exc:
        return None, f"could not parse {contract_path}: {exc}"
    if not isinstance(contract, dict):
        return None, f"{contract_path}: expected a JSON object at the top level"
    return contract, None


def ingest_contract(contract_path: str) -> dict:
    """Morning-chain entrypoint: reconcile today's journal contract.

    Steps: :func:`todo_store.upsert_from_contract` on ``todos[]`` (the daily
    reconciliation -- new/refreshed/reopened/ignored), then
    :func:`todo_store.close_todo` for each ``resolved_keys[]`` entry journal
    already resolved on its own.

    Returns the upsert plan (``inserted``/``refreshed``/``reopened``/
    ``ignored_snoozed``/``ignored_closed``) merged with ``resolved_count``
    (how many ``resolved_keys`` were successfully closed) and
    ``resolved_errors`` (list of ``{"key", "error"}`` for any close that came
    back ``{"ok": False}``, e.g. an unknown key).

    Never raises. Returns ``{"error": ...}`` with no store writes attempted
    when the contract file itself couldn't be read/parsed -- that is the one
    case :func:`main` treats as worth a non-zero exit, since it signals a
    real upstream problem (as opposed to one malformed row within an
    otherwise-good contract, which is logged and skipped).
    """
    contract, error = _load_contract(contract_path)
    if error is not None:
        _log.error("todo_store_sync ingest: %s", error)
        return {"error": error}

    for_date = contract.get("for_date") or _today()
    todos = contract.get("todos") or []

    plan = todo_store.upsert_from_contract(todos, for_date)

    resolved_count = 0
    resolved_errors: list[dict] = []
    for entry in contract.get("resolved_keys") or []:
        if not isinstance(entry, dict):
            _log.warning(
                "todo_store_sync ingest: skipping malformed resolved_keys entry "
                "(not an object): %r", entry,
            )
            continue
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            _log.warning(
                "todo_store_sync ingest: skipping resolved_keys entry missing "
                "'key': %r", entry,
            )
            continue
        result = todo_store.close_todo(key, "done", source="journal:resolution")
        if result.get("ok"):
            resolved_count += 1
        else:
            _log.warning(
                "todo_store_sync ingest: close_todo(%r) failed: %s",
                key, result.get("error"),
            )
            resolved_errors.append({"key": key, "error": result.get("error")})

    summary = dict(plan)
    summary["resolved_count"] = resolved_count
    summary["resolved_errors"] = resolved_errors
    return summary


def _dry_run_ingest(contract_path: str) -> dict:
    """Read-only counterpart of :func:`ingest_contract`, for ``--dry-run``.

    Uses :func:`todo_store.plan_upsert_from_contract` (no writes) for the
    ``todos[]`` classification. ``todo_store`` exposes no read-only "would
    this close succeed" check, so ``resolved_keys[]`` reporting here is
    limited to listing the keys that WOULD be attempted (``would_resolve_keys``)
    without simulating success/failure -- see ``resolved_errors_note`` in the
    returned dict. Deliberately not over-engineered into a full simulation.
    """
    contract, error = _load_contract(contract_path)
    if error is not None:
        _log.error("todo_store_sync ingest --dry-run: %s", error)
        return {"error": error}

    for_date = contract.get("for_date") or _today()
    todos = contract.get("todos") or []

    plan = todo_store.plan_upsert_from_contract(todos, for_date)

    would_resolve_keys: list[str] = []
    for entry in contract.get("resolved_keys") or []:
        if isinstance(entry, dict):
            key = entry.get("key")
            if isinstance(key, str) and key.strip():
                would_resolve_keys.append(key)

    summary = dict(plan)
    summary["would_resolve_keys"] = would_resolve_keys
    summary["resolved_errors_note"] = (
        "dry-run: close_todo is not called, so per-key success/failure "
        "(e.g. unknown key) is not simulated -- these are only the keys "
        "that would be attempted"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sync services.hermes.todo_store with the scheduled morning-chain "
            "path: 'export' writes OPEN_KEYS/CLOSED_KEYS for journal, 'ingest' "
            "reconciles journal's contract back into the store."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser(
        "export", help="Reopen expired snoozes; write OPEN_KEYS/CLOSED_KEYS for journal."
    )
    export_p.add_argument(
        "--open-path",
        default=os.environ.get("HERMES_OPEN_KEYS_PATH", default_open_keys_path()),
        help="Output path for the open-keys JSON (env: HERMES_OPEN_KEYS_PATH).",
    )
    export_p.add_argument(
        "--closed-path",
        default=os.environ.get("HERMES_CLOSED_KEYS_PATH", default_closed_keys_path()),
        help="Output path for the closed-keys JSON (env: HERMES_CLOSED_KEYS_PATH).",
    )

    ingest_p = sub.add_parser(
        "ingest", help="Reconcile today's journal_brief.latest.json into the store."
    )
    ingest_p.add_argument(
        "--contract",
        default=os.environ.get("JOURNAL_BRIEF_PATH", default_contract_path()),
        help="Path to journal_brief.latest.json (env: JOURNAL_BRIEF_PATH).",
    )
    ingest_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen; make no writes.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Always exits 0, EXCEPT when ``ingest``'s contract file
    itself could not be read/parsed at all -- that propagates a non-zero
    exit so it surfaces in the chain's logs, per the module docstring.
    """
    args = _build_parser().parse_args(argv)

    if args.command == "export":
        result = export_open_closed_keys(args.open_path, args.closed_path)
        print(result)
        return 0

    if args.command == "ingest":
        result = _dry_run_ingest(args.contract) if args.dry_run else ingest_contract(args.contract)
        print(result)
        return 1 if "error" in result else 0

    return 0  # pragma: no cover - argparse `required=True` prevents this


if __name__ == "__main__":
    sys.exit(main())
