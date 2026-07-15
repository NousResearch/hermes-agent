"""Tests for hermes_cli.context_switch_guard."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from hermes_cli.context_switch_guard import (
    enrich_model_switch_warnings_for_gateway,
    merge_preflight_compression_warning,
)
from hermes_cli.model_switch import ModelSwitchResult
from hermes_state import AsyncSessionDB
from gateway.session import AsyncSessionStore


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


@pytest.mark.asyncio
async def test_gateway_enrichment_awaits_async_session_history(monkeypatch):
    stored_messages = [{"role": "user", "content": "stored"}] * 30
    observed: dict[str, object] = {}

    def _capture_estimate(agent, messages):
        observed["messages"] = messages
        return 90_000

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        _capture_estimate,
    )

    cc = _compressor(monkeypatch)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    sync_db = SimpleNamespace(
        get_messages_as_conversation=lambda session_id: stored_messages,
    )
    sync_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(
            session_id="stored-session"
        )
    )
    runner = SimpleNamespace(
        _agent_cache_lock=threading.Lock(),
        _agent_cache={"session": (agent,)},
        _session_db=AsyncSessionDB(sync_db),
        async_session_store=AsyncSessionStore(sync_store),
    )
    result = _result()

    await enrich_model_switch_warnings_for_gateway(
        result,
        runner,
        session_key="session",
        source=object(),
    )

    assert observed["messages"] is stored_messages
    assert "preflight compression" in result.warning_message
