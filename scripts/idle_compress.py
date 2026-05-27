#!/usr/bin/env python3
"""
Idle compression integration script.

Triggers context compression on the most recent Hermes agent session,
reusing the Insert-then-Compress pattern so the compression call itself
hits the prompt cache when run within the cache TTL window.

Usage:
    python scripts/idle_compress.py              # compress most recent session
    python scripts/idle_compress.py --dry-run    # show what would happen
    python scripts/idle_compress.py --session <id>  # compress a specific session
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add repo root to path so imports work when run standalone
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for standalone script runs."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config() -> dict:
    """Load the user's hermes_cli config, returning the relevant idle subsection."""
    hermes_home = get_hermes_home()
    config_path = hermes_home / "config.yaml"

    if not config_path.exists():
        logger.warning("No config found at %s — using defaults", config_path)
        return {}

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available — using defaults")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Failed to load config: %s", exc)
        return {}

    compression = config.get("compression", {})
    idle_cfg = compression.get("idle", {})

    if not idle_cfg.get("enabled", True):
        logger.info("Idle compression is disabled in config — exiting")
        sys.exit(0)

    return idle_cfg


def _get_most_recent_session(hermes_home: Path) -> dict | None:
    """Query the session DB for the most recently updated session."""
    db_path = hermes_home / "sessions.db"
    if not db_path.exists():
        logger.warning("No session database at %s", db_path)
        return None

    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as exc:
        logger.warning("Failed to query session DB: %s", exc)
        return None


def _estimate_tokens_from_db(session: dict) -> int:
    """Rough token estimate from the session's message count or transcript."""
    # Try transcript length first
    transcript = session.get("transcript", "") or ""
    if transcript:
        # ~4 chars per token (rough)
        return len(transcript) // 4

    # Fallback: message count * avg tokens per message
    msg_count = session.get("message_count", 0) or 0
    return msg_count * 200  # ~200 tokens per message average


def run_compression(
    session_id: str | None = None,
    dry_run: bool = False,
    delay_seconds: float = 300.0,
    min_tokens: int = 20_000,
) -> None:
    """Main logic — locate session, check floor, trigger compression.

    Args:
        session_id: Specific session ID to compress, or None for most recent.
        dry_run: If True, only report what would happen.
        delay_seconds: Idle delay (informational — script runs immediately).
        min_tokens: Token floor below which compression is skipped.
    """
    hermes_home = get_hermes_home()

    # Locate session
    if session_id:
        logger.info("Targeting session: %s", session_id)
        session = None
        db_path = hermes_home / "sessions.db"
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cur.fetchone()
            conn.close()
            session = dict(row) if row else None
    else:
        logger.info("Finding most recent session...")
        session = _get_most_recent_session(hermes_home)

    if not session:
        logger.error("No session found — nothing to compress")
        sys.exit(1)

    sid = session.get("session_id", "unknown")
    est_tokens = _estimate_tokens_from_db(session)
    logger.info(
        "Session %s: ~%d estimated tokens (floor: %d)",
        sid, est_tokens, min_tokens,
    )

    # Token floor check
    if est_tokens < min_tokens:
        logger.info(
            "Skipping — session is below the token floor (%d < %d). "
            "Compressing a small session wastes API credits.",
            est_tokens, min_tokens,
        )
        if dry_run:
            print(f"[dry-run] Would skip session {sid}: {est_tokens} tokens < {min_tokens} minimum")
        return

    if dry_run:
        print(
            f"[dry-run] Would compress session {sid}: "
            f"~{est_tokens} tokens, idle delay={delay_seconds}s"
        )
        return

    # Trigger actual compression via the idle compression timer path.
    # We import the AIAgent and timer, create a minimal agent instance,
    # and fire compression directly.
    logger.info("Triggering idle compression on session %s...", sid)

    try:
        from run_agent import AIAgent
        from agent.idle_compression import IdleCompressionTimer

        # Build a minimal agent — the compression path needs messages loaded.
        # For a standalone script we use the session transcript.
        agent = AIAgent(
            quiet_mode=True,
            session_id=sid,
        )

        # Load messages from the session if available
        try:
            # Try to load existing conversation
            from hermes_state import SessionDB
            sdb = SessionDB()
            history = sdb.load_messages(sid)
            if history:
                agent.messages = history
                logger.info("Loaded %d messages from session", len(history))
        except Exception as exc:
            logger.debug("Could not load session messages: %s", exc)

        # Create timer and fire compression immediately (delay_seconds=0)
        timer = IdleCompressionTimer(
            agent=agent,
            delay_seconds=0.0,  # immediate
            min_tokens=min_tokens,
        )

        # Bypass the countdown — call compression directly on the timer's
        # internal _run_compression so we reuse the agent's existing
        # compression path (Insert-then-Compress, prompt cache reuse).
        timer._run_compression()

        logger.info("Compression complete for session %s", sid)

    except ImportError as exc:
        logger.error("Could not import required modules: %s", exc)
        logger.error(
            "Make sure you're running from the hermes-agent repo root "
            "with the virtualenv activated."
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Compression failed: %s", exc, exc_info=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trigger idle auto-compression on a Hermes agent session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                       compress the most recent session
  %(prog)s --dry-run             show what would happen
  %(prog)s --session abc123      compress a specific session
  %(prog)s --min-tokens 10000    only compress sessions above 10K tokens
        """,
    )
    parser.add_argument(
        "--session", "-s",
        help="Specific session ID to compress (default: most recent)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Report what would happen without actually compressing",
    )
    parser.add_argument(
        "--min-tokens", "-m",
        type=int,
        default=None,
        help="Token floor (default: from config or 20000)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=300.0,
        help="Idle delay in seconds (informational, default: 300)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)

    # Load config for defaults
    config = _load_config()

    min_tokens = args.min_tokens
    if min_tokens is None:
        min_tokens = config.get("min_tokens", 20_000)

    delay = config.get("delay_seconds", args.delay)

    run_compression(
        session_id=args.session,
        dry_run=args.dry_run,
        delay_seconds=delay,
        min_tokens=min_tokens,
    )


if __name__ == "__main__":
    main()
