"""Regression tests for hygiene turn-hold cooldown guard.

When the turn-hold budget expires and the user message is released,
``turnhold_deferred`` must NOT be sent if a failure cooldown is already
active from a prior attempt. Without the guard, every subsequent turn
that still exceeds the hygiene token threshold sends the notice on top
of the running cooldown — causing a message storm on sustained chatty
sessions.

Two paths are covered:
- Path 1 (no watermark fence / fence cancelled): worker exited before
  turn-hold expiry; ``HygieneTurnHoldExceeded`` block — must suppress
  the user notice when a cooldown is active, AND record the cooldown.
- Path 2 (worker still streaming / fence retained): ``continue`` exits
  the loop on the idle extension check — also in the
  ``HygieneTurnHoldExceeded`` block.

Regression test: when no cooldown is active, ``turnhold_deferred`` is
sent normally — the fix must not suppress the notice when safe.
"""

import asyncio
import importlib
import sys
import threading
import time
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource


def _make_history(n_messages: int, content_size: int = 100) -> list:
    history = []
    content = "x" * content_size
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": content, "timestamp": f"t{i}"})
    return history


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM
        )
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="x")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _write_turnhold_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_timeout_seconds: 60\n"
        "  hygiene_total_ceiling_seconds: 600\n"
        "  hygiene_max_turn_hold_seconds: 0.3\n"
        "  hygiene_failure_cooldown_seconds: 120\n"
    )


def _build_runner(gateway_run, adapter, fake_db):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:12345",
        session_id="sess-cooldown-guard",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = _make_history(
        6, content_size=400
    )
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = SimpleNamespace(_db=fake_db)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


def _make_event():
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="12345",
        ),
        message_id="1",
    )


def _install_fakes(monkeypatch, gateway_run, tmp_path, agent_cls):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )


class _UnfencedSpinningAgent:
    """Worker that spins indefinitely with NO watermark fence.

    Triggers ``HygieneTurnHoldExceeded`` → Path 1 (fence was cancelled).
    The outer code catches and falls through the turnhold path that sends
    ``turnhold_deferred`` AND now (with the fix) records the cooldown.
    """

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id", "sess-cooldown-guard")
        self._session_db = kwargs.get("session_db")
        self._last_compaction_in_place = False
        self.context_compressor = SimpleNamespace(
            bind_session_state=MagicMock(),
            _last_compress_aborted=False,
            _last_aux_model_failure_model=None,
        )
        self.shutdown_memory_provider = MagicMock()
        self.close = MagicMock()

    def _compress_context(self, messages, *_args, commit_fence=None, **_kwargs):
        # No mark_commit_watermark_fenced — turn-hold will cancel the fence
        # and treat this as Path 1.
        _spin_started = time.monotonic()
        while time.monotonic() - _spin_started < 5:
            if commit_fence is not None:
                commit_fence.touch_progress()
            time.sleep(0.01)
        # 5s is well past the 0.3s turn-hold → the outer code will
        # raise HygieneTurnHoldExceeded before we get here.
        return (messages, None)


class _StreamingAgent:
    """Worker that streams continuously WITH a watermark fence.

    Triggers Path 2: the loop exits via ``raise`` in the idle extension
    branch (turn-hold budget elapsed while summary was still streaming).
    """

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id", "sess-cooldown-guard")
        self._session_db = kwargs.get("session_db")
        self._last_compaction_in_place = False
        self.context_compressor = SimpleNamespace(
            bind_session_state=MagicMock(),
            _last_compress_aborted=False,
            _last_aux_model_failure_model=None,
        )
        self.shutdown_memory_provider = MagicMock()
        self.close = MagicMock()

    def _compress_context(self, messages, *_args, commit_fence=None, **_kwargs):
        if commit_fence is not None:
            commit_fence.mark_commit_watermark_fenced()
        # Streaming until the outer code cuts us off.
        _spin_started = time.monotonic()
        while time.monotonic() - _spin_started < 5:
            if commit_fence is not None:
                commit_fence.touch_progress()
            time.sleep(0.01)
        return (messages, None)


def _drain_deferred(runner, timeout=10.0):
    """Run the async drain on whatever loop is currently active."""
    return asyncio.wait_for(_drain_deferred_async(runner), timeout=timeout)


async def _drain_deferred_async(runner):
    tasks = getattr(runner, "_deferred_agent_cleanup_tasks", None) or set()
    if tasks:
        await asyncio.gather(*list(tasks), return_exceptions=True)


def _has_turnhold_msg(adapter):
    """Match the user-facing message text rather than the locale key —
    the gateway renders via ``t()`` and only the rendered text lands on
    the adapter."""
    return any(
        "compression deferred" in str(m.get("content", "")).lower()
        for m in adapter.sent
    )


# ---------------------------------------------------------------------
# Path 1: fence cancelled (worker exited, turn-hold raised, fall-through).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_path1_turnhold_msg_suppressed_when_cooldown_active(
    monkeypatch, tmp_path
):
    """Path 1: an active cooldown must suppress the user-facing
    ``turnhold_deferred`` message."""
    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = {
        "cooldown_until": time.time() + 300.0,
        "remaining_seconds": 300.0,
        "error": "prior failure",
    }

    gateway_run = importlib.import_module("gateway.run")
    _write_turnhold_config(tmp_path)
    adapter = _CaptureAdapter()
    _install_fakes(monkeypatch, gateway_run, tmp_path, _UnfencedSpinningAgent)
    runner = _build_runner(gateway_run, adapter, fake_db)

    result = await asyncio.wait_for(runner._handle_message(_make_event()), timeout=15)
    assert result == "ok"

    # The cooldown should NOT have been overwritten — the merge-max
    # path leaves the prior deadline intact. record_compression_failure_cooldown
    # was called for the merge path (but the prior cooldown with a longer
    # deadline remains effective).
    # The important assertion: NO turnhold_deferred message was sent.
    assert not _has_turnhold_msg(adapter), (
        f"turnhold_deferred must be suppressed when a cooldown is active; "
        f"got sent={adapter.sent}"
    )


# ---------------------------------------------------------------------
# Path 2: worker still streaming (loop exited via raise in idle branch).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_path2_turnhold_msg_suppressed_when_cooldown_active(
    monkeypatch, tmp_path
):
    """Path 2: an active cooldown must suppress the user-facing
    ``turnhold_deferred`` message when the worker was still streaming."""
    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = {
        "cooldown_until": time.time() + 300.0,
        "remaining_seconds": 300.0,
        "error": "prior failure",
    }

    gateway_run = importlib.import_module("gateway.run")
    _write_turnhold_config(tmp_path)
    adapter = _CaptureAdapter()
    _install_fakes(monkeypatch, gateway_run, tmp_path, _StreamingAgent)
    runner = _build_runner(gateway_run, adapter, fake_db)

    result = await asyncio.wait_for(runner._handle_message(_make_event()), timeout=15)
    assert result == "ok"

    assert not _has_turnhold_msg(adapter), (
        f"turnhold_deferred must be suppressed when a cooldown is active; "
        f"got sent={adapter.sent}"
    )


# ---------------------------------------------------------------------
# Regression: no cooldown active → message still sent.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_cooldown_turnhold_msg_still_sent(monkeypatch, tmp_path):
    """When no cooldown is active, ``turnhold_deferred`` is sent normally."""
    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None

    gateway_run = importlib.import_module("gateway.run")
    _write_turnhold_config(tmp_path)
    adapter = _CaptureAdapter()
    _install_fakes(monkeypatch, gateway_run, tmp_path, _UnfencedSpinningAgent)
    runner = _build_runner(gateway_run, adapter, fake_db)

    result = await asyncio.wait_for(runner._handle_message(_make_event()), timeout=15)
    assert result == "ok"

    assert _has_turnhold_msg(adapter), (
        f"turnhold_deferred must be sent when no cooldown is active; "
        f"got sent={adapter.sent}"
    )
    # And the new failure-cooldown record call must be made (Path 1 fix).
    assert fake_db.record_compression_failure_cooldown.called, (
        "Path 1 turnhold-exceeded must record the failure cooldown so the "
        "next turn doesn't respawn a compressor immediately"
    )
