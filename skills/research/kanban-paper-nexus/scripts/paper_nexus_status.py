#!/usr/bin/env python3
"""Inspect existing paper-nexus workflow state for one paper id."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from paper_doc_registry import canonical_paper_id, lookup  # noqa: E402
from paper_nexus_metadata import resolve_canonical_id  # noqa: E402

STAGE_LABELS = {
    "T0": "论点与阅读地图",
    "T1": "主张-证据链 CEL",
    "T2": "方法与复现要点",
    "T3": "对标与开源地图",
    "T4": "实验审计与局限",
    "T5": "飞书精读文档",
    "T6": "QA 门禁",
}


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def board_db(board: str = "paper-nexus") -> Path:
    return hermes_home() / "kanban" / "boards" / board / "kanban.db"


def _query_tasks(canonical_id: str, board: str = "paper-nexus") -> dict:
    db = board_db(board)
    stages = {
        key: {"title": label, "task_id": "", "status": "missing"}
        for key, label in STAGE_LABELS.items()
    }
    if not db.is_file():
        return stages
    conn = sqlite3.connect(db)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            """
            select id, title, status
            from tasks
            where title like ?
            order by created_at
            """,
            (f"%{canonical_id}%",),
        ).fetchall()
    finally:
        conn.close()
    for task_id, title, status in rows:
        for stage in STAGE_LABELS:
            if f"[paper] {canonical_id} " in title and f" {STAGE_LABELS[stage]}" in title:
                stages[stage] = {"title": title, "task_id": task_id, "status": status}
                break
    return stages


def inspect_status(paper_ref: str, board: str = "paper-nexus") -> dict:
    canonical_id = canonical_paper_id(resolve_canonical_id(paper_ref))
    tasks = _query_tasks(canonical_id, board)
    doc = lookup(canonical_id, board)
    exists = any(v["status"] != "missing" for v in tasks.values())
    incomplete = [k for k, v in tasks.items() if v["status"] not in {"done", "missing"}]
    missing = [k for k, v in tasks.items() if v["status"] == "missing"]
    next_stage = incomplete[0] if incomplete else (missing[0] if missing else "")
    return {
        "ok": True,
        "board": board,
        "paper_ref": paper_ref,
        "canonical_id": canonical_id,
        "exists": exists,
        "tasks": tasks,
        "doc_url": (doc or {}).get("doc_url", ""),
        "next_stage": next_stage,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: paper_nexus_status.py <paper_id>", file=sys.stderr)
        return 2
    result = inspect_status(sys.argv[1])
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
