#!/usr/bin/env python3
"""Subscribe Feishu chat to all tasks in a board feishu_sessions manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from kanban_feishu_boards import get_board_config  # noqa: E402
from kanban_feishu_session import load_session  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("entity_id", nargs="?", default="")
    p.add_argument("--board", default="paper-nexus")
    p.add_argument("--chat-id", default="")
    p.add_argument("--platform", default="feishu")
    p.add_argument("--thread-id", default="")
    p.add_argument("--notifier-profile", default="")
    args = p.parse_args()

    board = args.board
    cfg = get_board_config(board)
    eid = args.entity_id.strip()
    session = load_session(eid, board) if eid else None
    if session:
        eid = str(session.get(cfg["entity_key"]) or eid)
        chat_id = args.chat_id or session.get("chat_id") or ""
        thread_id = args.thread_id or session.get("thread_id") or ""
        tasks = session.get("tasks") or {}
    else:
        chat_id = (args.chat_id or os.environ.get("HERMES_SESSION_CHAT_ID") or "").strip()
        thread_id = (args.thread_id or os.environ.get("HERMES_SESSION_THREAD_ID") or "").strip()
        tasks = {}

    if not chat_id:
        print("chat_id required", file=sys.stderr)
        return 2
    if not tasks:
        print("no tasks in session; run init first", file=sys.stderr)
        return 1

    try:
        from hermes_cli import kanban_db as kb
    except ImportError as exc:
        print(f"hermes_cli import failed: {exc}", file=sys.stderr)
        return 2

    profile = (args.notifier_profile or os.environ.get("HERMES_PROFILE") or "").strip() or None
    subscribed: list[str] = []
    conn = kb.connect(board=board)
    try:
        for stage, tid in sorted(tasks.items()):
            if kb.get_task(conn, tid) is None:
                print(f"skip missing {stage}={tid}", file=sys.stderr)
                continue
            kb.add_notify_sub(
                conn,
                task_id=tid,
                platform=args.platform,
                chat_id=chat_id,
                thread_id=thread_id or None,
                notifier_profile=profile,
            )
            subscribed.append(tid)
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "ok": bool(subscribed),
                "board": board,
                cfg["entity_key"]: eid,
                "subscribed_task_ids": subscribed,
                "count": len(subscribed),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if subscribed else 1


if __name__ == "__main__":
    raise SystemExit(main())
