"""/save md|html is a transcript: after in-place compaction it still holds every turn the chat shows.
/save json stays the live rows import_sessions restores."""
import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _compacted_store(path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=path)
    db.create_session("s1", "telegram")
    for i in range(1, 7):
        db.append_message("s1", "user", f"question {i}")
        db.append_message("s1", "assistant", f"answer {i}")
    tail = [{"role": "user", "content": "question 6"}, {"role": "assistant", "content": "answer 6"}]
    db.archive_and_compact("s1", [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, *tail],
                           watermark=db.get_active_message_watermark("s1"), tail_count=len(tail))
    return db


def _cli_save(db, fmt, out):
    import cli

    stub = SimpleNamespace(_session_db=db, session_id="s1", conversation_history=[], model="m",
                           session_start=datetime(2026, 1, 1))
    cli.HermesCLI.save_conversation(stub, f"/save {fmt} {out}")
    return out.read_text(encoding="utf-8")


def _gateway_save(db, fmt, out):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, SessionSource, build_session_key
    from hermes_state import AsyncSessionDB

    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    runner = object.__new__(GatewayRunner)
    delivered = {}
    adapter = MagicMock()
    adapter.send_document = AsyncMock(side_effect=lambda **kw: delivered.update(
        text=open(kw["file_path"], encoding="utf-8").read()))
    runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: adapter}, {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=build_session_key(source), session_id="s1", created_at=datetime.now(),
        updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm")
    runner._session_db = AsyncSessionDB(db)
    event = MessageEvent(text=f"/save {fmt} {out.name}", source=source, message_id="m1")
    assert asyncio.run(runner._handle_save_command(event)) == "Export complete."
    return delivered["text"]


@pytest.mark.parametrize("save", [_cli_save, _gateway_save], ids=["cli", "gateway"])
@pytest.mark.parametrize("fmt, expected", [("md", 6), ("html", 6), ("json", 1)])
def test_save_transcript_holds_every_turn_the_chat_shows(tmp_path, monkeypatch, save, fmt, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _compacted_store(tmp_path / "state.db")
    try:
        text = save(db, fmt, tmp_path / f"saved.{fmt}")
    finally:
        db.close()
    assert sum(f"answer {i}" in text for i in range(1, 7)) == expected
