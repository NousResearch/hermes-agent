"""/save md (CLI and gateway) carries the display history, compaction-archived turns included."""
import asyncio
import contextlib
import io
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
    db.archive_and_compact(
        "s1", [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, *tail],
        watermark=db.get_active_message_watermark("s1"), tail_count=len(tail),
    )
    return db


def _cli_save(db, fmt, out):
    import cli

    stub = SimpleNamespace(_session_db=db, session_id="s1", conversation_history=[], model="m",
                           session_start=datetime(2026, 1, 1))
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        cli.HermesCLI.save_conversation(stub, f"/save {fmt} {out}")
    return out.read_text(encoding="utf-8") if out.exists() else printed.getvalue()


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
    reply = asyncio.run(runner._handle_save_command(event))
    return delivered["text"] if reply == "Export complete." else reply


@pytest.mark.parametrize("save", [_cli_save, _gateway_save], ids=["cli", "gateway"])
def test_save_transcript_holds_display_history(tmp_path, monkeypatch, save):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _compacted_store(tmp_path / "state.db")
    try:
        text = save(db, "md", tmp_path / "saved.md")
    finally:
        db.close()
    assert [f"answer {i}" in text for i in range(1, 7)] == [True] * 6


@pytest.mark.parametrize("save", [_cli_save, _gateway_save], ids=["cli", "gateway"])
def test_save_json_restores_compacted_history_as_archived(tmp_path, monkeypatch, save):
    """/save json is the snapshot the dashboard import restores: the turns compaction archived come back
    in the display history, and stay out of the live context."""
    import json

    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _compacted_store(tmp_path / "state.db")
    # A model-only live row (a micro-compaction merge) is context, not display history: only the transfer
    # projection carries it, so the snapshot's ids pin that projection, not the display read.
    db.append_message("s1", "user", "merged", display_metadata={"model_only": True})
    db.append_message("s1", "user", "undone secret")
    db.rewind_to_message("s1", db.get_messages("s1")[-1]["id"])

    def shape(store, **flags):
        return [(m["role"], m["content"]) for m in store.get_messages("s1", **flags)]

    try:
        shown, live = shape(db, include_compacted=True), shape(db)
        snapshot = json.loads(save(db, "json", tmp_path / "saved.json"))
        kept = [m["id"] for m in db.get_messages("s1", include_inactive=True) if m["active"] or m["compacted"]]
        assert [m["id"] for m in snapshot["messages"]] == kept  # every kept row, not the deduped display read
        assert sum(snapshot["timings"]["role_counts"].values()) == len(kept)  # timings name no undone row
        # Like `hermes sessions export`, the in-memory backup is capped per session (sessions.max_export_messages).
        monkeypatch.setattr("hermes_state.resolved_max_export_messages", lambda: 5)
        assert "max_export_messages" in save(db, "json", tmp_path / "capped.json")
    finally:
        db.close()
    restored = SessionDB(db_path=tmp_path / "restored.db")
    try:
        assert restored.import_sessions([snapshot])["ok"]
        assert shape(restored, include_compacted=True) == shown
        assert shape(restored) == live
    finally:
        restored.close()
