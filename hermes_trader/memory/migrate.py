"""SQLite schema migration for trade_episodes."""

from __future__ import annotations

import argparse
import sys

from hermes_trader.memory.episodes import EpisodeStore, SCHEMA_VERSION, default_episodes_db_path


def migrate(db_path: str | None = None) -> int:
    store = EpisodeStore(db_path) if db_path else EpisodeStore()
    version = store.migrate()
    print(f"Migrated {store.db_path} to schema v{version}")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate Hermes trader episodes DB")
    parser.add_argument(
        "--db",
        default=str(default_episodes_db_path()),
        help="Path to trade_episodes.db",
    )
    args = parser.parse_args(argv)
    migrate(args.db)
    if SCHEMA_VERSION < 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())