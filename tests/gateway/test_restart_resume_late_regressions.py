"""Late-finding regressions for restart resume + delivery ledger recovery."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.session_store = None
    store = MagicMock()
    store._store = None
    store.clear_resume_pending = AsyncMock()
    runner._async_session_store = store
    return runner


@pytest.mark.asyncio
async def test_effective_resume_session_id_follows_compression_tip():
    runner = _make_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="u1",
        thread_id="topic-1",
    )
    entry = SessionEntry(
        session_key="agent:main:telegram:dm:chat-1:topic-1",
        session_id="fresh-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
    )
    session_db = MagicMock()
    session_db.get_telegram_topic_binding = AsyncMock(
        return_value={"session_id": "bound-parent"}
    )
    session_db.get_compression_tip = AsyncMock(return_value="bound-child")
    runner._session_db = session_db

    effective = await runner._effective_resume_session_id(entry)

    assert effective == "bound-child"
    session_db.get_compression_tip.assert_awaited_once_with("bound-parent")


def test_sweep_recoverable_skips_platforms_not_connected(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")

    dl.record_obligation(
        obligation_id="ob-telegram",
        session_key="agent:main:telegram:dm:1",
        platform="telegram",
        chat_id="1",
        thread_id=None,
        content="telegram answer",
    )
    dl.record_obligation(
        obligation_id="ob-slack",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="slack answer",
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
        )

    claimed = dl.sweep_recoverable(platforms={"slack"})

    assert [row["obligation_id"] for row in claimed] == ["ob-slack"]
    with dl._connect() as conn:
        telegram_state = conn.execute(
            "SELECT state, owner_pid FROM delivery_obligations WHERE obligation_id=?",
            ("ob-telegram",),
        ).fetchone()
    assert telegram_state[0] == "pending"
    assert telegram_state[1] == 999999999


@pytest.mark.asyncio
async def test_redeliver_pending_obligations_scopes_to_connected_platform(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")

    dl.record_obligation(
        obligation_id="ob-telegram",
        session_key="agent:main:telegram:dm:1",
        platform="telegram",
        chat_id="1",
        thread_id=None,
        content="telegram answer",
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
        )

    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True, error=""))
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: adapter}

    n = await runner._redeliver_pending_obligations(platform=Platform.TELEGRAM)

    assert n == 1
    adapter.send.assert_awaited_once()
