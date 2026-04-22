from __future__ import annotations

from pathlib import Path

from market_monitor.db import Database, initialize_database
from market_monitor.runners import run_weekly


def run(db_path: Path | str, raw_root: Path | str) -> list[dict]:
    db_path = Path(db_path)
    raw_root = Path(raw_root)
    initialize_database(db_path)
    db = Database(db_path)
    return run_weekly(db=db, raw_root=raw_root)
