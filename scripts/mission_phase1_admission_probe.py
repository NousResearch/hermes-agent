#!/usr/bin/env python3
"""Exercise Telegram-shaped gateway admission through the real plugin lifecycle registry."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run_inbound_consumer import run_post_admission_hook
from gateway.session import SessionSource
from hermes_cli import mission_ownership as missions, plugins
from hermes_cli.kanban_db_connect import connect


class Runner:
    config = type("Config", (), {"multiplex_profiles": False})()


def event(message_id: str, chat_id: str, text: str):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="synthetic-user")
    return MessageEvent(text=text, source=source, message_id=message_id), source


async def deliver(message_id: str, chat_id: str, text: str):
    item, source = event(message_id, chat_id, text)
    session_id = f"telegram:phase1:{chat_id}"
    return await run_post_admission_hook(Runner(), item, source, session_id)


def main() -> int:
    root_value = os.environ.get("HERMES_ISOLATION_ROOT")
    if not root_value:
        raise SystemExit("isolation root is required")
    os.environ["HERMES_KANBAN_DB"] = str(Path(root_value) / "work" / "admission-probe-replacement.db")
    plugins.discover_plugins()
    if not plugins.has_hook("post_gateway_admission"):
        raise SystemExit("mission admission hook was not registered")
    request = "Please implement mission ownership in the isolated Hermes system and record evidence."
    first = asyncio.run(deliver("tg-delivery-1", "chat-a", request))
    replay = asyncio.run(deliver("tg-delivery-1", "chat-a", request))
    distinct = asyncio.run(deliver("tg-delivery-2", "chat-b", request))
    with connect(db_path=Path(os.environ["HERMES_KANBAN_DB"])) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "missions" not in tables:
            print(json.dumps({"first": first, "replay": replay, "distinct": distinct,
                              "tables": sorted(tables),
                              "database": os.environ["HERMES_KANBAN_DB"]}, default=str))
            raise SystemExit("real gateway lifecycle invocation did not admit a mission")
        rows = conn.execute(
            "SELECT owner_id,ingress_key FROM missions WHERE status='active' ORDER BY owner_id"
        ).fetchall()
        counts = conn.execute("SELECT count(*) FROM missions").fetchone()[0]
        by_owner = {row["owner_id"]: row["ingress_key"] for row in rows}
    if counts != 2 or set(by_owner) != {"telegram:phase1:chat-a", "telegram:phase1:chat-b"}:
        raise SystemExit(f"unexpected mission admission state: {counts} missions, owners={sorted(by_owner)}")
    if first != (False, None) or replay != (False, None) or distinct != (False, None):
        raise SystemExit("admission plugin unexpectedly consumed a user message")
    print(json.dumps({
        "registered_hook": "post_gateway_admission",
        "duplicate_delivery": "same active mission; one row for chat-a",
        "identical_text_distinct_delivery_session": "separate mission for chat-b",
        "messages_consumed": False,
        "active_mission_count": counts,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
