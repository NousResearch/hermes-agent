#!/usr/bin/env python3
"""Backfill titles for existing untitled Hermes sessions.

This is intended as a one-shot maintenance script.  It finds sessions whose
friendly display name/title is empty, takes the first assistant response from
that session, and runs the same response-title generator used by the API
adapter:

    Provide a title of less than 50 characters for the following text : ...

By default it writes titles to the active Hermes state DB.  Use --dry-run to
preview candidates without calling the title LLM or writing anything.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Allow running this file directly from a source checkout without installing the
# package first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.title_generator import generate_title_from_response_text  # noqa: E402
from hermes_state import SessionDB  # noqa: E402

logger = logging.getLogger("backfill_session_titles")


def _content_to_text(content: Any) -> str:
    """Return human-readable text for stored message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") in {"image_url", "input_image"}:
                    parts.append("[image]")
                elif item.get("type") in {"file", "input_file"}:
                    parts.append("[file]")
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
    return str(content).strip()


def _iter_untitled_sessions(db: SessionDB, *, source: str | None = None) -> Iterable[dict[str, Any]]:
    """Yield sessions whose title/friendly name is empty."""
    clauses = ["(title IS NULL OR TRIM(title) = '')"]
    params: list[Any] = []
    if source:
        clauses.append("source = ?")
        params.append(source)

    query = f"""
        SELECT id, source, started_at, message_count
        FROM sessions
        WHERE {' AND '.join(clauses)}
        ORDER BY started_at ASC, id ASC
    """
    with db._lock:  # Script-level maintenance query; SessionDB has no public all-untitled iterator.
        rows = db._conn.execute(query, params).fetchall()
    for row in rows:
        yield dict(row)


def _first_assistant_response(db: SessionDB, session_id: str) -> str | None:
    for message in db.get_messages(session_id):
        if message.get("role") != "assistant":
            continue
        text = _content_to_text(message.get("content"))
        if text:
            return text
    return None


def _set_unique_title(db: SessionDB, session_id: str, title: str) -> str:
    """Set title, suffixing on collisions caused by similar conversations."""
    base = title.strip()
    candidate = base
    for attempt in range(1, 100):
        try:
            if db.set_session_title(session_id, candidate):
                return candidate
            raise RuntimeError(f"session disappeared before title update: {session_id}")
        except ValueError as exc:
            message = str(exc)
            if "already in use" not in message:
                raise
            # Keep under SessionDB.MAX_TITLE_LENGTH while leaving room for suffix.
            suffix = f" #{attempt + 1}"
            candidate = f"{base[: SessionDB.MAX_TITLE_LENGTH - len(suffix)].rstrip()}{suffix}"
    raise RuntimeError(f"could not find a unique title for {session_id!r} based on {base!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="Path to state.db; defaults to the active Hermes home")
    parser.add_argument("--source", default=None, help="Only backfill one source, e.g. api, cli, telegram")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of sessions to process")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-title LLM timeout in seconds")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between title-generation calls")
    parser.add_argument("--dry-run", action="store_true", help="List candidates only; do not call the LLM or write titles")
    parser.add_argument("--verbose", action="store_true", help="Show debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    db = SessionDB(args.db) if args.db else SessionDB()
    scanned = 0
    skipped = 0
    updated = 0
    failed = 0

    for session in _iter_untitled_sessions(db, source=args.source):
        if args.limit is not None and scanned >= args.limit:
            break
        scanned += 1
        session_id = session["id"]
        first_response = _first_assistant_response(db, session_id)
        if not first_response:
            skipped += 1
            logger.info("SKIP %s (%s): no assistant response", session_id, session.get("source"))
            continue

        preview = " ".join(first_response.split())[:100]
        if args.dry_run:
            logger.info("DRY  %s (%s): %s", session_id, session.get("source"), preview)
            continue

        try:
            title = generate_title_from_response_text(first_response, timeout=args.timeout)
            if not title:
                skipped += 1
                logger.warning("SKIP %s: generator returned no title", session_id)
                continue
            final_title = _set_unique_title(db, session_id, title)
            updated += 1
            logger.info("OK   %s: %s", session_id, final_title)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # Continue the one-shot batch after individual failures.
            failed += 1
            logger.error("FAIL %s: %s", session_id, exc)

        if args.sleep > 0:
            time.sleep(args.sleep)

    logger.info(
        "Done. scanned=%d updated=%d skipped=%d failed=%d%s",
        scanned,
        updated,
        skipped,
        failed,
        " (dry run)" if args.dry_run else "",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
