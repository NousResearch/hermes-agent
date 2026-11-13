"""Gateway views label a defaulted context window "fallback" like /usage does, and /model never shows a pin
the route change discarded (Greptile on #148801)."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.session import SessionEntry, build_session_key
from tests.gateway.test_status_command import _make_event, _make_runner, _make_source

UNKNOWN = "acme.unknown-model-v1"
RUNTIME_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    from hermes_agent.agent import context_default_record
    monkeypatch.setattr(context_default_record, "_RECORDS", {})
    monkeypatch.setattr("hermes_agent.agent.bedrock_adapter.probe_bedrock_context_length", lambda *a, **k: None)


def _resolve_unknown_bedrock() -> int:
    """What the agent's compressor does when it builds the window."""
    from hermes_agent.agent.model_metadata import get_model_context_length
    return get_model_context_length(UNKNOWN, base_url=RUNTIME_URL, provider="bedrock")


def _bedrock_agent(context_length: int) -> SimpleNamespace:
    compressor = SimpleNamespace(
        model=UNKNOWN, provider="bedrock", base_url=RUNTIME_URL, last_prompt_tokens=9_000,
        context_length=context_length, _config_context_length=None, threshold_tokens=96_000,
        threshold_percent=0.75, compression_count=0, _last_compression_savings_pct=None)
    return SimpleNamespace(
        model=UNKNOWN, provider="bedrock", base_url=RUNTIME_URL, context_compressor=compressor,
        interrupt=lambda *a, **k: None, session_api_calls=1, session_input_tokens=9_000,
        session_output_tokens=100, session_reasoning_tokens=0, session_total_tokens=9_100,
        session_cache_read_tokens=0, session_cache_write_tokens=0)


def _runner():
    entry = SessionEntry(session_key=build_session_key(_make_source()), session_id="sess-fallback",
                         created_at=datetime.now(), updated_at=datetime.now(), platform=Platform.TELEGRAM,
                         chat_type="dm")
    entry.last_prompt_tokens = 9_000
    return _make_runner(entry)


@pytest.mark.asyncio
async def test_status_marks_a_live_agents_fallback_window():
    runner = _runner()
    runner._running_agents[build_session_key(_make_source())] = _bedrock_agent(_resolve_unknown_bedrock())
    result = await runner._handle_message(_make_event("/status"))
    line = next(x for x in result.splitlines() if "**Context:**" in x)
    assert "128,000" in line and line.endswith("(fallback)")


@pytest.mark.asyncio
async def test_status_leaves_a_detected_live_window_unmarked():
    runner = _runner()
    runner._running_agents[build_session_key(_make_source())] = _bedrock_agent(1_050_000)
    result = await runner._handle_message(_make_event("/status"))
    assert "(fallback)" not in result


@pytest.mark.asyncio
async def test_context_marks_a_resident_agents_fallback_window():
    runner = _runner()
    agent = _bedrock_agent(_resolve_unknown_bedrock())
    with patch.object(type(runner), "_resident_agent_for", lambda self, key: agent, create=True), \
            patch.object(type(runner), "_context_breakdown_block", lambda self, *a: [], create=True):
        result = await runner._handle_context_command(_make_event("/context"))
    assert "Window: 128,000 tokens (fallback)" in result


@pytest.mark.asyncio
async def test_context_marks_a_non_resident_sessions_defaulted_route():
    from hermes_agent.gateway.run import _GatewayModelContext
    runner = _runner()
    resolved = _GatewayModelContext(model=UNKNOWN, provider="bedrock", base_url=RUNTIME_URL,
                                    context_length=128_000, context_source="default")

    async def _route(self, source, model_name, route=None):
        return resolved
    with patch.object(type(runner), "_resident_agent_for", lambda self, key: None, create=True), \
            patch.object(type(runner), "_resolve_route_context", _route):
        result = await runner._handle_context_command(_make_event("/context"))
    assert "Window: 128,000 tokens (fallback)" in result


@pytest.mark.asyncio
async def test_model_switch_does_not_show_a_discarded_pin(tmp_path, monkeypatch):
    """The configured model's 128K pin is dropped for the new route, so it must not label the new window."""
    import hermes_yaml as yaml
    from hermes_cli.model_switch import ModelSwitchResult
    from tests.gateway.test_model_command_context_offload import _event, _runner_with_store
    runner = _runner_with_store(tmp_path, monkeypatch)
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.safe_dump(
        {"model": {"default": "old-model", "provider": "openrouter", "context_length": 128_000}}), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model=UNKNOWN, target_provider="bedrock", provider_changed=True, api_key="aws-sdk",
        base_url=RUNTIME_URL, api_mode="bedrock_converse", provider_label="AWS Bedrock"))
    result = await runner._handle_model_command(_event(f"/model {UNKNOWN}"))
    line = next(x for x in result.splitlines() if "128,000" in x)
    assert "(pinned)" not in line and line.endswith("(fallback)")
