"""CLI and gateway /save preserve filenames and reject invalid arguments before writing."""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _save(db, args, output, surface):
    if surface == "cli":
        import cli

        stub = SimpleNamespace(_session_db=db, session_id="s1", conversation_history=[],
                               model="m", session_start=datetime(2026, 1, 1))
        cli.HermesCLI.save_conversation(stub, f"/save {args}")
        return output.read_text(encoding="utf-8") if output.exists() else None

    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, SessionSource, build_session_key
    from hermes_state import AsyncSessionDB

    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", chat_type="dm")
    delivered = {}

    async def send_document(**kwargs):
        from pathlib import Path

        delivered.update(name=kwargs["file_name"], text=Path(kwargs["file_path"]).read_text(encoding="utf-8"))

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(send_document=send_document)}
    runner._profile_adapters = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=build_session_key(source), session_id="s1", created_at=datetime.now(),
        updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm")
    runner._session_db = AsyncSessionDB(db)
    event = MessageEvent(text=f"/save {args}", source=source, message_id="m1")
    reply = asyncio.run(runner._handle_save_command(event))
    if delivered:
        assert delivered["name"] == output.name
        return delivered["text"]
    assert "Usage:" in reply
    return None


@pytest.mark.parametrize("surface", ["cli", "gateway"])
@pytest.mark.parametrize(("filename_arg", "filename"), [
    ('"meeting notes.json"', "meeting notes.json"),
    ("'会议 记录.json'", "会议 记录.json"),
    ("O'Reilly.json", "O'Reilly.json"),
    (r"notes\archive.json", r"notes\archive.json"),
    ('"notes\\meeting notes.json"', r"notes\meeting notes.json"),
    ("#notes.json", "#notes.json"),
    ('"redact"', "redact"),
    ('"--redact"', "--redact"),
    ("notes.json --REDACT", "notes.json"),
])
def test_save_preserves_filename(tmp_path, monkeypatch, surface, filename_arg, filename):
    from hermes_state import SessionDB

    monkeypatch.chdir(tmp_path)
    output = tmp_path / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "telegram")
        db.append_message("s1", "user", "hello")
        text = _save(db, f"json {filename_arg}", output, surface)
        assert json.loads(text)["messages"][0]["content"] == "hello"
    finally:
        db.close()


@pytest.mark.parametrize("surface", ["cli", "gateway"])
@pytest.mark.parametrize("args", [
    'json "existing.json" extra',
    'json "existing.json',
    'json ""',
    '"" existing.json',
])
def test_save_rejects_invalid_args_before_writing(tmp_path, monkeypatch, capsys, surface, args):
    from hermes_state import SessionDB

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("keep this file", encoding="utf-8")
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "telegram")
        db.append_message("s1", "user", "hello")
        text = _save(db, args, output, surface)
        if surface == "cli":
            assert "Usage:" in capsys.readouterr().out
        else:
            assert text is None
        assert output.read_text(encoding="utf-8") == "keep this file"
        assert not (tmp_path / '"existing.json').exists()
        assert not (tmp_path / '""').exists()
    finally:
        db.close()
