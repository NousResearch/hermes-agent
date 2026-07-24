"""CLI-friendly sync entrypoint for People Phase 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plugins.people.adapters.imessage import IMessageAdapter
from plugins.people.store import PeopleMessageStore
from plugins.people.writer import write_people_markdown


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hermes People message sync (Phase 1)")
    p.add_argument(
        "--db",
        default=None,
        help="Path to people messages.db (default: ~/.hermes/people/messages.db)",
    )
    p.add_argument(
        "--imessage-db",
        default=str(Path.home() / "Library" / "Messages" / "chat.db"),
        help="Read-only path to macOS chat.db",
    )
    p.add_argument("--people-dir", default=None, help="Output dir for <slug>.md")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--skip-imessage", action="store_true")
    p.add_argument("--write-only", action="store_true", help="Only rewrite markdown")
    args = p.parse_args(argv)

    with PeopleMessageStore(args.db) as store:
        if not args.write_only and not args.skip_imessage:
            adapter = IMessageAdapter(args.imessage_db)
            inserted, skipped = adapter.sync_to_store(store, limit=args.limit)
            print(f"imessage: inserted={inserted} skipped={skipped}")
        n = write_people_markdown(store, people_dir=args.people_dir)
        print(f"people markdown files written: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
