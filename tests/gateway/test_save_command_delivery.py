"""Gateway /save: the export document is delivered through the requester's live adapter."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _runner(entry, adapters, home):
    runner = object.__new__(GatewayRunner)
    runner.adapters = adapters
    runner._profile_adapters = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.persistence = storage_handle(home)
    db = runner.session_store.persistence
    seed_session(db, "sess-1", "telegram")
    seed_messages(db, "sess-1", {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"})
    return runner


def _save(runner, platform):
    source = SessionSource(platform=platform, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    event = MessageEvent(text="/save md save-stuff.md", source=source, message_id="m1")
    return asyncio.run(runner._handle_save_command(event))


def _entry(platform):
    source = SessionSource(platform=platform, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    return SessionEntry(session_key=build_session_key(source), session_id="sess-1", created_at=datetime.now(),
                        updated_at=datetime.now(), platform=platform, chat_type="dm")


def test_save_sends_document_through_requesting_platform_adapter(tmp_path):
    adapter = MagicMock()
    adapter.send_document = AsyncMock()
    runner = _runner(_entry(Platform.TELEGRAM), {Platform.TELEGRAM: adapter}, tmp_path)

    assert _save(runner, Platform.TELEGRAM) == "Export complete."
    kwargs = adapter.send_document.await_args.kwargs
    assert (kwargs["chat_id"], kwargs["file_name"]) == ("c1", "save-stuff.md")


def test_save_without_live_adapter_reports_missing_adapter_not_a_crash(tmp_path):
    runner = _runner(_entry(Platform.DISCORD), {}, tmp_path)

    assert _save(runner, Platform.DISCORD) == "Platform adapter not found to send the document."
