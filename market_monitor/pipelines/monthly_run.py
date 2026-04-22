from __future__ import annotations

from pathlib import Path

from market_monitor.db import Database, initialize_database
from market_monitor.runners import render_structured_results, run_monthly


def run(db_path: Path | str, raw_root: Path | str) -> list[dict]:
    db_path = Path(db_path)
    raw_root = Path(raw_root)
    initialize_database(db_path)
    db = Database(db_path)
    return run_monthly(db=db, raw_root=raw_root)


def latest_structured_payload(db_path: Path | str, period_label: str) -> dict:
    db = Database(db_path)
    return render_structured_results(db, period_label=period_label)
