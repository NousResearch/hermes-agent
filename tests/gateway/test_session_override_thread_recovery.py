"""Regression tests for #30479 — session-scoped /model, /reasoning, and /fast
overrides silently lost on Telegram forum/DM topics and after compression
session splits.

Root cause: ``_handle_message_with_agent`` rewrites ``source.thread_id`` via
``_recover_telegram_topic_thread_id`` (lobby/stripped reply -> the user's
last-active bound topic) *before* deriving the session key for a message turn.
The ``/model``, ``/reasoning``, and ``/fast`` command handlers derived their
override key from the raw inbound ``event.source``, skipping that recovery —
so the override was stored under one key and the next message turn read a
different key, and the override was dropped.

Fix: all three command handlers normalize the source via
``_normalize_source_for_session_key`` before deriving the override key, so
storage and read keys are identical.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_runner(recovered_thread_id=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = None
    runner.session_store = None
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    # Stub topic recovery: returns the bound topic id for a lobby message,
    # None otherwise (the real method's contract).
    runner._recover_telegram_topic_thread_id = MagicMock(return_value=recovered_thread_id)
    return runner


def _make_fast_runner(recovered_thread_id=None):
    runner = _make_runner(recovered_thread_id=recovered_thread_id)
    runner.adapters = {}
    runner._service_tier = None
    runner._try_send_choice_picker = AsyncMock(return_value=False)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    return runner


def _topic_dm_source(thread_id):
    """A Telegram DM in topic mode. thread_id="" / "1" == General/lobby."""
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="555",
        chat_name="Forum DM",
        chat_type="dm",
        user_id="user-1",
        thread_id=thread_id,
    )


def _topic_event(text, thread_id=""):
    return MessageEvent(text=text, source=_topic_dm_source(thread_id), message_id="m1")


def _patch_fast_gateway(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"service_tier": ""}},
    )


def test_normalize_rewrites_lobby_thread_to_bound_topic():
    """A lobby (stripped) reply gets pinned to the user's bound topic id."""
    runner = _make_runner(recovered_thread_id="42")
    src = _topic_dm_source(thread_id="")  # lobby/General — no message_thread_id

    normalized = runner._normalize_source_for_session_key(src)

    assert normalized.thread_id == "42"
    # Original source is left untouched (we return a copy).
    assert src.thread_id == ""


@pytest.mark.asyncio
async def test_fast_pin_uses_recovered_topic_key(monkeypatch, tmp_path):
    """Lobby /fast writes the pin under the recovered topic, not the raw source."""
    _patch_fast_gateway(monkeypatch, tmp_path)
    runner = _make_fast_runner(recovered_thread_id="42")
    lobby = _topic_dm_source(thread_id="")
    recovered = runner._normalize_source_for_session_key(lobby)
    lobby_key = runner._session_key_for_source(lobby)
    topic_key = runner._session_key_for_source(recovered)
    assert lobby_key != topic_key

    response = await runner._handle_fast_command(_topic_event("/fast fast"))

    assert response is not None
    assert "FAST" in response
    assert runner._resolve_session_service_tier(session_key=topic_key) == "priority"
    # Raw lobby key is what a broken /fast would have written; turn never reads it.
    assert runner._resolve_session_service_tier(session_key=lobby_key) is None


@pytest.mark.asyncio
async def test_fast_status_and_next_turn_see_recovered_pin(monkeypatch, tmp_path):
    """After a lobby /fast, status and the next recovered turn share the pin."""
    _patch_fast_gateway(monkeypatch, tmp_path)
    runner = _make_fast_runner(recovered_thread_id="42")

    await runner._handle_fast_command(_topic_event("/fast normal"))

    status = await runner._handle_fast_command(_topic_event("/fast status"))
    assert status is not None
    assert "normal" in status.lower()

    recovered = runner._normalize_source_for_session_key(_topic_dm_source(thread_id=""))
    turn_key = runner._session_key_for_source(recovered)
    # Explicit /fast normal is a pin (None), not "fall back to config".
    assert runner._resolve_session_service_tier(session_key=turn_key) is None
    assert turn_key in runner._session_service_tier_overrides


@pytest.mark.asyncio
async def test_fast_picker_uses_recovered_session_key(monkeypatch, tmp_path):
    """Choice-picker callbacks store the pin under the recovered topic key."""
    _patch_fast_gateway(monkeypatch, tmp_path)
    runner = _make_fast_runner(recovered_thread_id="42")
    captured = {}

    async def _capture(_event, session_key, **_kwargs):
        captured["session_key"] = session_key
        return False

    runner._try_send_choice_picker = _capture
    lobby = _topic_dm_source(thread_id="")
    topic_key = runner._session_key_for_source(
        runner._normalize_source_for_session_key(lobby)
    )

    await runner._handle_fast_command(_topic_event("/fast"))

    assert captured["session_key"] == topic_key


