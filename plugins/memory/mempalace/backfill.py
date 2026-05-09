#!/usr/bin/env python3
"""MemPalace Backfill — mine existing Hermes session history into the palace.

Reads all sessions from the Hermes SQLite state DB, exports them as
conversation files (Claude export format), then mines each one into the
palace using mempalace's convo_miner.

Wing classification uses ~/.mempalace/wing_config.json if it exists,
otherwise all sessions go to wing_general.

Usage:
    python -m plugins.memory.mempalace.backfill
    python -m plugins.memory.mempalace.backfill --limit 50
    python -m plugins.memory.mempalace.backfill --dry-run
    python -m plugins.memory.mempalace.backfill --palace ~/.mempalace/palace
    python -m plugins.memory.mempalace.backfill --wing wing_general --source cli
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# FIX 8: import _extract_text_content from the main module — no duplication
from plugins.memory.mempalace import _extract_text_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_wing_config() -> Dict[str, Any]:
    """Load wing config from ~/.mempalace/wing_config.json if it exists."""
    wing_config_path = Path.home() / ".mempalace" / "wing_config.json"
    if wing_config_path.exists():
        try:
            with open(wing_config_path) as f:
                return json.load(f).get("wings", {})
        except Exception:
            pass
    return {}


def _classify_wing_simple(text: str, wing_config: Dict[str, Any]) -> str:
    """Keyword-based wing classification using loaded wing_config."""
    if not wing_config:
        return "wing_general"
    combined = text.lower()
    scores: Dict[str, int] = {}
    for wing_name, wing_cfg in wing_config.items():
        keywords = wing_cfg.get("keywords", [])
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[wing_name] = score
    return max(scores, key=scores.get) if scores else "wing_general"


def _messages_to_export_format(messages: List[Dict[str, Any]]) -> str:
    """Convert Hermes messages to Claude export format (> User / response pairs)."""
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        raw_content = msg.get("content", "")
        content = _extract_text_content(raw_content)
        if not content or not content.strip():
            continue
        # Skip tool calls and tool results — they're noise for mining
        if msg.get("tool_calls") or msg.get("tool_name"):
            continue
        if role == "user":
            lines.append(f"> {content.strip()}")
        elif role == "assistant":
            lines.append(content.strip())
            lines.append("")  # blank line between exchanges
    return "\n".join(lines)


def _find_hermes_state_db() -> Optional[Path]:
    """Locate the Hermes state DB."""
    # Primary: use hermes_constants if available
    try:
        from hermes_constants import get_hermes_home
        db = get_hermes_home() / "state.db"
        if db.exists():
            return db
    except ImportError:
        pass

    # Fallback: look in ~/.hermes/
    for candidate in [
        Path.home() / ".hermes" / "state.db",
        Path.home() / ".hermes" / "hermes.db",
    ]:
        if candidate.exists():
            return candidate

    return None


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

def backfill(
    palace_path: Optional[str] = None,
    limit: int = 0,
    dry_run: bool = False,
    wing_override: Optional[str] = None,
    source_filter: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Mine all Hermes sessions into the MemPalace.

    Returns the number of sessions successfully mined.
    """
    # ------------------------------------------------------------------
    # 1. Find state DB
    # ------------------------------------------------------------------
    db_path = _find_hermes_state_db()
    if db_path is None:
        print("ERROR: Could not find Hermes state DB (state.db).")
        print("       Make sure HERMES_HOME is set or hermes_state is importable.")
        return 0

    print(f"\n  Source DB:  {db_path}")

    # ------------------------------------------------------------------
    # 2. Resolve palace path
    # ------------------------------------------------------------------
    if palace_path:
        resolved_palace = os.path.expanduser(palace_path)
    else:
        try:
            from mempalace.config import MempalaceConfig
            resolved_palace = MempalaceConfig().palace_path
        except ImportError:
            resolved_palace = os.path.expanduser("~/.mempalace/palace")

    print(f"  Palace:     {resolved_palace}")
    if dry_run:
        print("  DRY RUN — nothing will be filed")
    print()

    # ------------------------------------------------------------------
    # 3. Check mempalace is installed
    # ------------------------------------------------------------------
    try:
        from mempalace.convo_miner import mine_convos
    except ImportError:
        print("ERROR: mempalace package not installed.")
        print("       Run: pip install mempalace")
        return 0

    # ------------------------------------------------------------------
    # 4. Load wing config
    # ------------------------------------------------------------------
    wing_config = _load_wing_config()
    if wing_config:
        print(f"  Wing config: {len(wing_config)} wings loaded from ~/.mempalace/wing_config.json")
    else:
        print("  Wing config: not found — all sessions will go to wing_general")
        print("               (run `mempalace init` to configure wings)")

    # ------------------------------------------------------------------
    # 5. Open session DB
    # ------------------------------------------------------------------
    try:
        # Try the Hermes SessionDB class first (respects WAL mode etc.)
        from hermes_state import SessionDB
        db = SessionDB(db_path)
        use_session_db = True
    except Exception as exc:
        if verbose:
            print(f"  Warning: SessionDB unavailable ({exc}), falling back to raw SQLite")
        import sqlite3
        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row
        use_session_db = False

    # ------------------------------------------------------------------
    # 6. Enumerate sessions
    # ------------------------------------------------------------------
    try:
        if use_session_db:
            sessions = db.list_sessions_rich(
                source=source_filter,
                limit=limit if limit > 0 else 9999,
                include_children=False,
            )
        else:
            query = "SELECT id, source, started_at FROM sessions"
            params: List[Any] = []
            if source_filter:
                query += " WHERE source = ?"
                params.append(source_filter)
            query += " ORDER BY started_at DESC"
            if limit > 0:
                query += f" LIMIT {limit}"
            sessions = [dict(r) for r in db.execute(query, params).fetchall()]
    except Exception as exc:
        print(f"ERROR reading sessions: {exc}")
        return 0

    total = len(sessions)
    print(f"  Sessions found: {total}")
    if total == 0:
        print("  Nothing to backfill.")
        return 0

    print()
    print(f"{'=' * 60}")
    print("  MemPalace Backfill")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # 7. Mine each session
    # ------------------------------------------------------------------
    mined = 0
    skipped = 0
    errors = 0

    for idx, session in enumerate(sessions, 1):
        session_id = session.get("id", "")
        source = session.get("source", "unknown")
        preview = session.get("preview", "")[:50]

        # Fetch messages
        try:
            if use_session_db:
                messages = db.get_messages(session_id)
            else:
                rows = db.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
                    (session_id,),
                ).fetchall()
                messages = [dict(r) for r in rows]
        except Exception as exc:
            if verbose:
                print(f"  [{idx:4}/{total}] SKIP {session_id[:8]}... — message fetch failed: {exc}")
            errors += 1
            continue

        # Build export text
        export_text = _messages_to_export_format(messages)
        if len(export_text.strip()) < 50:
            if verbose:
                print(f"  [{idx:4}/{total}] SKIP {session_id[:8]}... — too short")
            skipped += 1
            continue

        # Detect wing
        if wing_override:
            wing = wing_override
        else:
            wing = _classify_wing_simple(export_text[:1000], wing_config)

        if dry_run:
            word_count = len(export_text.split())
            print(
                f"  [{idx:4}/{total}] DRY {session_id[:8]}... "
                f"source={source:8} wing={wing:20} {word_count}w  \"{preview}\""
            )
            mined += 1
            continue

        # Write to temp dir and mine
        try:
            with tempfile.TemporaryDirectory(prefix="hermes_backfill_") as tmpdir:
                session_file = Path(tmpdir) / f"session_{session_id[:8]}.txt"
                session_file.write_text(export_text, encoding="utf-8")

                # Suppress mine_convos' own print output unless verbose
                if not verbose:
                    import io
                    import contextlib
                    with contextlib.redirect_stdout(io.StringIO()):
                        mine_convos(
                            convo_dir=tmpdir,
                            palace_path=resolved_palace,
                            wing=wing,
                            agent="hermes-backfill",
                        )
                else:
                    mine_convos(
                        convo_dir=tmpdir,
                        palace_path=resolved_palace,
                        wing=wing,
                        agent="hermes-backfill",
                    )

            mined += 1
            print(
                f"  [{idx:4}/{total}] OK  {session_id[:8]}... "
                f"source={source:8} wing={wing:20} \"{preview}\""
            )
        except Exception as exc:
            errors += 1
            print(f"  [{idx:4}/{total}] ERR {session_id[:8]}... — {exc}")

    # ------------------------------------------------------------------
    # 8. Report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Backfill complete.")
    print(f"  Sessions mined:   {mined}")
    print(f"  Sessions skipped: {skipped}  (too short / no content)")
    print(f"  Errors:           {errors}")
    if not dry_run and mined > 0:
        print()
        print(f"  Palace: {resolved_palace}")
        print('  Next:   mempalace wake-up')
    print(f"{'=' * 60}\n")

    return mined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MemPalace Backfill — mine Hermes session history into the palace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--palace",
        default=None,
        help="Palace directory (default: from ~/.mempalace/config.json or ~/.mempalace/palace)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max sessions to process (0 = all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be mined without writing to palace",
    )
    parser.add_argument(
        "--wing",
        default=None,
        help="Override wing for all sessions (default: auto-detect per session)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Only mine sessions from this source (e.g. cli, telegram, discord)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output including mine_convos progress",
    )

    args = parser.parse_args()

    count = backfill(
        palace_path=args.palace,
        limit=args.limit,
        dry_run=args.dry_run,
        wing_override=args.wing,
        source_filter=args.source,
        verbose=args.verbose,
    )

    sys.exit(0 if count >= 0 else 1)


if __name__ == "__main__":
    main()
