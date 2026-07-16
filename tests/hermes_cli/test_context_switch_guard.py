"""Tests for hermes_cli.context_switch_guard."""

from __future__ import annotations

import asyncio
import gc
import threading
import warnings
from types import SimpleNamespace

from hermes_cli.context_switch_guard import (
    enrich_model_switch_warnings_for_gateway,
    merge_preflight_compression_warning,
)
from hermes_cli.model_switch import ModelSwitchResult


def _result(*, model: str = "small-model") -> ModelSwitchResult:
    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider="openrouter",
        provider_changed=False,
        api_key="k",
        base_url="https://example.com/v1",
        api_mode="chat_completions",
        provider_label="openrouter",
        model_info={"context_length": 32_000},
    )


def _compressor(monkeypatch, *, context_length: int = 200_000):
    from agent.context_compressor import ContextCompressor

    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length",
        lambda *a, **k: context_length,
    )
    return ContextCompressor(
        model="big-model",
        threshold_percent=0.5,
        protect_first_n=3,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=context_length,
    )


def test_no_warning_when_below_new_threshold(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    cc = _compressor(monkeypatch)
    cc.last_prompt_tokens = 10_000
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="",
        api_key="",
    )
    result = _result()
    merge_preflight_compression_warning(result, agent=agent)
    assert not result.warning_message


def test_warns_when_estimate_exceeds_new_threshold(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 90_000,
    )
    cc = _compressor(monkeypatch)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="",
        api_key="",
    )
    result = _result()
    merge_preflight_compression_warning(result, agent=agent)
    assert result.warning_message
    assert "preflight compression" in result.warning_message
    assert "shrinks" in result.warning_message


def test_merge_appends_to_existing_warning(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 90_000,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    cc = _compressor(monkeypatch)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result()
    result.warning_message = "expensive"
    merge_preflight_compression_warning(result, agent=agent)
    assert "expensive" in result.warning_message
    assert "preflight compression" in result.warning_message


class _FakeSyncSessionDB:
    """Stands in for the real synchronous hermes_state.SessionDB."""

    def __init__(self, messages):
        self._messages = messages

    def get_messages_as_conversation(self, session_id):
        return self._messages


def test_enrich_awaits_async_session_db_read(monkeypatch):
    """Regression for #63712: the gateway's runner._session_db is an
    AsyncSessionDB, whose every method access returns an async-offloaded
    wrapper. The helper must await that facade — a bare (sync) call hands
    merge_preflight_compression_warning a stray coroutine instead of the
    real message list and leaks an un-awaited coroutine (RuntimeWarning),
    while unwrapping to the sync ``_db`` handle would run blocking SQLite
    on the gateway event loop. Exercises the real AsyncSessionDB facade
    around a small sync double, per the gateway's off-loop contract.
    """
    from hermes_state import AsyncSessionDB

    captured = {}

    def _fake_merge(result, *, agent=None, messages=None, **kwargs):
        captured["messages"] = messages

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.merge_preflight_compression_warning",
        _fake_merge,
    )

    real_messages = [{"role": "user", "content": "hi"}]
    async_db = AsyncSessionDB(_FakeSyncSessionDB(real_messages))
    agent = SimpleNamespace()
    runner = SimpleNamespace(
        _agent_cache_lock=threading.Lock(),
        _agent_cache={"sess-key": (agent, None)},
        _session_db=async_db,
        session_store=SimpleNamespace(
            get_or_create_session=lambda source: SimpleNamespace(session_id="s1"),
        ),
    )
    result = _result()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(
            enrich_model_switch_warnings_for_gateway(
                result,
                runner,
                session_key="sess-key",
                source=SimpleNamespace(),
            )
        )
        # A dropped, un-awaited coroutine only warns at GC time — force it
        # so the assertion below is deterministic, not a coin flip.
        gc.collect()

    assert captured["messages"] == real_messages
    assert not any(
        issubclass(w.category, RuntimeWarning) and "coroutine" in str(w.message)
        for w in caught
    )


def test_enrich_is_a_coroutine_function():
    """Guard the async signature: both gateway call sites await this helper
    from the event loop; a silent revert to a sync def would put either a
    dropped read (no await) or a blocking on-loop SQLite call back in play."""
    assert asyncio.iscoroutinefunction(enrich_model_switch_warnings_for_gateway)
