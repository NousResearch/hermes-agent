#!/usr/bin/env python3
"""Feishu session manifest per Kanban board run (skill layer)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from kanban_feishu_boards import get_board_config


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def session_path(entity_id: str, board: str) -> Path:
    eid = (entity_id or "").strip()
    return hermes_home() / "kanban" / "boards" / board / "feishu_sessions" / f"{eid}.json"


def load_session(entity_id: str, board: str) -> dict | None:
    path = session_path(entity_id, board)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(data: dict, board: str) -> Path:
    cfg = get_board_config(board)
    ekey = cfg["entity_key"]
    eid = str(data.get(ekey) or data.get("canonical_id") or data.get("symbol") or "").strip()
    if not eid:
        raise ValueError(f"{ekey} required")
    path = session_path(eid, board)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def new_session(
    *,
    board: str,
    entity_id: str,
    title: str = "",
    chat_id: str,
    thread_id: str = "",
    platform: str = "feishu",
    tasks: dict[str, str] | None = None,
    extra_url: str = "",
    meta: dict | None = None,
) -> dict:
    cfg = get_board_config(board)
    ekey = cfg["entity_key"]
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "version": 1,
        "board": board,
        ekey: entity_id.strip(),
        "title": title.strip(),
        "platform": platform,
        "chat_id": chat_id.strip(),
        "thread_id": (thread_id or "").strip(),
        "tasks": {k: str(v) for k, v in (tasks or {}).items()},
        "created_at": now,
        "updated_at": now,
    }
    if board == "paper-nexus":
        data["paper_id_latest"] = (meta or {}).get("paper_id_latest") or entity_id
        data["feishu_doc_url"] = (extra_url or "").strip()
    if board == "stock-nexus":
        data["stock_name"] = (meta or {}).get("stock_name") or title
        data["deep"] = bool((meta or {}).get("deep"))
    if board == "paper-search":
        data["query_slug"] = entity_id
        data["query_text"] = (meta or {}).get("query_text") or title
    return data
