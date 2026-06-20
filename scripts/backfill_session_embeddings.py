#!/usr/bin/env python3
"""One-time embedding backfill for semantic session search (#44075).

Embeds every active user/assistant message in state.db that has no
embedding yet, so past conversations become semantically searchable
immediately instead of converging via the opportunistic per-search batch.

Usage:
    python scripts/backfill_session_embeddings.py [--db PATH] [--batch-size N]

Requires ``session_search.semantic: true`` plus a working embedding
provider in config.yaml (see DEFAULT_CONFIG["session_search"]), and the
sqlite-vec package (``pip install hermes-agent[semantic-search]`` or let
lazy-install pull it).
"""

import argparse
import sys
from pathlib import Path

# Allow running straight from a checkout: scripts/ is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Path to state.db (default: the active profile's database)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Messages embedded per provider call (default: 64)",
    )
    args = parser.parse_args()

    from hermes_state import DEFAULT_DB_PATH, SessionDB
    from tools.session_semantic import (
        backfill,
        get_semantic_config,
        semantic_enabled,
        _ensure_sqlite_vec,
    )

    db_path = args.db or DEFAULT_DB_PATH
    if not Path(db_path).exists():
        print(f"No database at {db_path}", file=sys.stderr)
        return 1

    cfg = get_semantic_config()
    if not semantic_enabled(cfg):
        print(
            "session_search.semantic is disabled in config.yaml — enable it "
            "first so searches actually use the embeddings:\n\n"
            "  session_search:\n    semantic: true\n",
            file=sys.stderr,
        )
        return 1
    if not _ensure_sqlite_vec():
        print(
            "sqlite-vec is not installed and could not be lazy-installed. "
            "Run: pip install 'hermes-agent[semantic-search]'",
            file=sys.stderr,
        )
        return 1

    db = SessionDB(db_path)
    if not db.vector_search_available():
        print(
            "sqlite-vec extension failed to load on this Python build "
            "(sqlite3 compiled without extension support?).",
            file=sys.stderr,
        )
        return 1

    pending = len(db.get_unembedded_messages(cfg["embedding_model"], limit=1_000_000))
    print(f"Model: {cfg['embedding_model']}")
    print(f"Pending messages: {pending}")
    if pending == 0:
        print("Nothing to do.")
        return 0

    def progress(done: int) -> None:
        print(f"  embedded {done}/{pending}", flush=True)

    total = backfill(db, cfg, batch_size=args.batch_size, progress=progress)
    print(f"Done — embedded {total} messages.")
    if total < pending:
        print(
            f"Warning: {pending - total} messages were not embedded "
            "(provider failure mid-run?). Re-run to resume.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
