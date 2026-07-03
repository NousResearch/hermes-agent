"""Tests for the goal kickoff media passthrough (#57928).

When a user invokes ``/goal <plan>`` while having attached a document
(e.g. Telegram file caption), the synthesized kickoff MessageEvent must
carry ``media_urls`` / ``media_types`` / ``reply_to_*`` fields. Before
the fix these were silently dropped, so the kickoff agent turn never
saw the file even though the user clearly meant to attach it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


class _RecordingAdapter:
    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self._fifo: list = []  # FIFO-style insert so we can peek
        self.sends: list = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})

        class _R:
            success = True
            message_id = "mock-msg"

        return _R()

    def _enqueue_fifo(self, key, event, adapter):
        # matches GatewayRunner's contract — pairs of (event, adapter)
        self._fifo.append((event, adapter))


def _make_runner_with_adapter(session_id: str = None):
    from gateway.run import GatewayRunner
    import uuid

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}

    src = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id or f"goal-media-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = build_session_key(src)

    adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner, adapter, session_entry, src


@pytest.mark.asyncio
async def test_goal_command_with_attachment_preserves_media_on_kickoff_57928(hermes_home):
    """``/goal <text>`` with an attached file (Telegram doc caption) must
    enqueue a kickoff MessageEvent that still carries the attachment, the
    same way ``/queue`` already does (#57928)."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    # Wire the runner so /goal can talk to its adapter.
    runner.adapters[src.platform] = adapter
    runner._session_key_for_source = lambda source: build_session_key(src)
    runner._enqueue_fifo = adapter._enqueue_fifo

    # Stub the goal manager so it accepts a one-liner goal.
    from hermes_cli import goals as goals_mod

    mgr = MagicMock()
    mgr.set.return_value = SimpleNamespace(
        goal="ship the Q3 plan",
        contract=SimpleNamespace(is_empty=lambda: True),
        has_contract=lambda: False,
        max_turns=20,
    )
    runner._get_goal_manager_for_event = lambda event: (mgr, session_entry)

    event = MessageEvent(
        text="/goal ship the Q3 plan",
        source=src,
        message_id="m_attach",
        media_urls=["/tmp/inbox/plan.pdf"],
        media_types=["document"],
    )

    result = await runner._handle_goal_command(event)

    # The kickoff event landed in the FIFO with the media payload intact.
    assert any(
        ev.media_urls == ["/tmp/inbox/plan.pdf"]
        and ev.media_types == ["document"]
        and ev.text == "ship the Q3 plan"
        for ev, _ in adapter._fifo
    ), f"expected a kickoff event preserving media, got {[(ev.text, ev.media_urls) for ev, _ in adapter._fifo]}"
    # Should not have crashed / returned an error string.
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_goal_command_without_attachment_uses_text_message_type_57928(hermes_home):
    """Regression guard: when there is no attachment, the kickoff event
    must remain MessageType.TEXT — we must not pretend the message_type
    field carries media information that isn't there."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    runner.adapters[src.platform] = adapter
    runner._session_key_for_source = lambda source: build_session_key(src)
    runner._enqueue_fifo = adapter._enqueue_fifo

    from hermes_cli import goals as goals_mod

    mgr = MagicMock()
    mgr.set.return_value = SimpleNamespace(
        goal="polish docs",
        contract=SimpleNamespace(is_empty=lambda: True),
        has_contract=lambda: False,
        max_turns=20,
    )
    runner._get_goal_manager_for_event = lambda event: (mgr, session_entry)

    event = MessageEvent(
        text="/goal polish docs",
        source=src,
        message_id="m_noattach",
    )

    result = await runner._handle_goal_command(event)

    assert any(ev.text == "polish docs" for ev, _ in adapter._fifo)
    # Inspect the kickoff event's message_type explicitly
    kickoff_event = next(ev for ev, _ in adapter._fifo if ev.text == "polish docs")
    from gateway.platforms.base import MessageType
    assert kickoff_event.message_type == MessageType.TEXT
    assert kickoff_event.media_urls == []
    assert isinstance(result, str)
