"""Generic, data-free seed for the persistent todo store (plugins.life_ops.todo_store).

Reconciles any journal_brief.latest.json-shaped contract file's ``todos``
list into the store, one time, idempotently: a todo whose key already exists
is left completely untouched, and only genuinely-new keys are inserted as
``status='open', origin='journal'``. This is deliberately generic — it must
work identically against any contract file (including a synthetic test
fixture), so it contains no seed data of its own.

Usage:
    python -m plugins.life_ops.todo_store_seed
    python -m plugins.life_ops.todo_store_seed --input /path/to/journal_brief.latest.json
    python -m plugins.life_ops.todo_store_seed --for-date 2026-07-15
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
from pathlib import Path

from hermes_cli.sqlite_util import write_txn
from plugins.life_ops import todo_store

# Common filler articles/verbs that don't identify the task; dropped before
# picking the "significant" words for a derived key.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "at", "with",
    "is", "be", "do", "set", "get", "find", "review", "continue", "complete",
    "finish", "fix", "pay", "decide", "lock", "order", "visit",
}

_MAX_WORD_LEN = 24  # guard against a single absurdly long "word" bloating the key

_DEFAULTS = {
    "recurring": False,
    "confidence": 0.5,
    "priority": "medium",
    "category": None,
}


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home)


def _default_input_path() -> Path:
    return _hermes_home() / "contracts" / "journal_brief.latest.json"


def derive_key(text: str) -> str:
    """Pure, deterministic slugifier: same input text -> same output key, always.

    No LLM/network calls. Lowercases, strips punctuation, drops a small
    stopword list, then takes the first 3 remaining significant words (in
    original order) and joins them with hyphens, truncating any single word
    that is absurdly long. If fewer than 1 significant word remains, falls
    back to the first 3 words of the raw text regardless of stopword
    filtering (so short/all-stopword input still yields a stable, non-empty
    key wherever possible).
    """
    raw = text or ""
    lowered = raw.lower()
    cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    words = cleaned.split()

    significant = [w for w in words if w not in _STOPWORDS]
    chosen = significant[:3] if significant else words[:3]
    return "-".join(w[:_MAX_WORD_LEN] for w in chosen)


def _value_or_default(item: dict, field: str, default: object) -> object:
    value = item.get(field)
    return default if value is None else value


def _todo_raw_text(item: dict) -> str:
    text = item.get("content") or item.get("text") or ""
    return str(text)


def seed_from_contract(input_path: str, for_date: str) -> dict:
    """Reconcile ``input_path``'s ``todos[]`` into the todo store. Never raises.

    For each todo dict: use its existing ``key`` if non-empty, otherwise
    derive one via :func:`derive_key` from ``content``/``text``. A todo whose
    key ends up empty is skipped. A todo whose key already exists in the
    store is skipped (existing rows are never touched). Anything new is
    inserted as ``status='open', origin='journal'`` with
    ``first_seen = min(source_dates)`` when present, else ``for_date``, and
    ``last_seen = for_date``.

    Returns ``{"inserted": [...keys], "skipped_existing": [...keys],
    "skipped_empty_key": <count>}``, plus an ``"error"`` key (with empty
    lists/zero count) if ``input_path`` doesn't exist or isn't valid JSON —
    this must never crash a scheduled run.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return {
            "inserted": [],
            "skipped_existing": [],
            "skipped_empty_key": 0,
            "error": str(exc),
        }

    todos = data.get("todos") if isinstance(data, dict) else None
    if not isinstance(todos, list):
        return {
            "inserted": [],
            "skipped_existing": [],
            "skipped_empty_key": 0,
            "error": "input JSON missing a top-level 'todos' list",
        }

    inserted: list[str] = []
    skipped_existing: list[str] = []
    skipped_empty_key = 0

    conn = todo_store.connect()
    try:
        with write_txn(conn):
            for item in todos:
                if not isinstance(item, dict):
                    skipped_empty_key += 1
                    continue

                given_key = item.get("key")
                if isinstance(given_key, str) and given_key.strip():
                    key = given_key.strip()
                else:
                    key = derive_key(_todo_raw_text(item))

                if not key:
                    skipped_empty_key += 1
                    continue

                exists = conn.execute(
                    "SELECT 1 FROM todos WHERE key=?", (key,)
                ).fetchone()
                if exists is not None:
                    skipped_existing.append(key)
                    continue

                source_dates = item.get("source_dates")
                if not isinstance(source_dates, list):
                    source_dates = []
                source_dates = [d for d in source_dates if isinstance(d, str)]
                first_seen = min(source_dates) if source_dates else for_date

                conn.execute(
                    "INSERT INTO todos "
                    "(key, text, category, priority, source_dates, recurring, "
                    " confidence, status, snoozed_until, first_seen, last_seen, "
                    " closed_at, origin) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', NULL, ?, ?, NULL, 'journal')",
                    (
                        key,
                        _todo_raw_text(item),
                        _value_or_default(item, "category", _DEFAULTS["category"]),
                        _value_or_default(item, "priority", _DEFAULTS["priority"]),
                        json.dumps(source_dates),
                        1 if _value_or_default(item, "recurring", _DEFAULTS["recurring"]) else 0,
                        _value_or_default(item, "confidence", _DEFAULTS["confidence"]),
                        first_seen,
                        for_date,
                    ),
                )
                inserted.append(key)
    finally:
        conn.close()

    return {
        "inserted": inserted,
        "skipped_existing": skipped_existing,
        "skipped_empty_key": skipped_empty_key,
    }


def main() -> None:
    """CLI entrypoint: parse argv, seed, print the summary, always exit 0.

    ``--for-date`` resolution (today's system date) is kept here rather than
    in :func:`seed_from_contract` so the core function stays deterministic
    and unit-testable.
    """
    parser = argparse.ArgumentParser(
        description=(
            "One-off, idempotent seed of plugins.life_ops.todo_store from a "
            "journal_brief.latest.json-shaped contract file."
        )
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to a journal_brief.latest.json-shaped file "
            "(default: $HERMES_HOME/contracts/journal_brief.latest.json)"
        ),
    )
    parser.add_argument(
        "--for-date",
        default=None,
        help="ISO date YYYY-MM-DD used for last_seen / fallback first_seen (default: today)",
    )
    args = parser.parse_args()

    input_path = args.input or str(_default_input_path())
    for_date = args.for_date or _dt.date.today().isoformat()

    result = seed_from_contract(input_path, for_date)
    print(result)


if __name__ == "__main__":
    main()
