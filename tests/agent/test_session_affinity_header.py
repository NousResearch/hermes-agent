"""Per-provider ``session_affinity_header`` (#86241, #104449).

A custom provider entry may name a header that carries Hermes' conversation id so a
session-aware proxy can correlate the requests of one agent loop. Off unless configured.
"""

from __future__ import annotations

from unittest.mock import patch

from agent import auxiliary_client as aux
from agent.chat_completion_helpers import build_api_kwargs
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hello"}]
_BASE = "http://localhost:4000/v1"
_HEADER = "x-litellm-session-id"


def _agent(session_id, api_mode="chat_completions"):
    agent = AIAgent(
        api_key="test-key", base_url=_BASE, model="claude-3-5-sonnet", provider="litellm-lan",
        api_mode=api_mode, quiet_mode=True, skip_context_files=True, skip_memory=True,
        session_id=session_id,
    )
    agent._anthropic_base_url = _BASE
    return agent


def _providers(with_header: bool):
    entry = {"name": "litellm-lan", "provider_key": "litellm-lan", "base_url": _BASE}
    if with_header:
        entry["session_affinity_header"] = _HEADER
    return patch("hermes_cli.config.get_compatible_custom_providers", return_value=[entry])


def _aux_headers(session_id):
    token = aux.set_runtime_main("litellm-lan", "claude-3-5-sonnet", base_url=_BASE, session_id=session_id)
    try:
        return aux._build_call_kwargs("litellm-lan", "claude-3-5-sonnet", _MSGS, base_url=_BASE).get("extra_headers") or {}
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)


def test_configured_header_carries_one_value_per_conversation_on_every_path():
    with _providers(True):
        chat = build_api_kwargs(_agent("sess-A"), _MSGS)["extra_headers"][_HEADER]
        anthropic = build_api_kwargs(_agent("sess-A", api_mode="anthropic_messages"), _MSGS)["extra_headers"][_HEADER]
        assert chat == anthropic == _aux_headers("sess-A")[_HEADER] == "sess-A"
        assert build_api_kwargs(_agent("sess-B"), _MSGS)["extra_headers"][_HEADER] == "sess-B"
        # A caller-pinned value is preserved.
        pinned = _agent("sess-A")
        pinned.request_overrides = {"extra_headers": {_HEADER: "pinned-by-caller"}}
        assert build_api_kwargs(pinned, _MSGS)["extra_headers"][_HEADER] == "pinned-by-caller"


def test_unconfigured_provider_sends_no_session_header():
    with _providers(False):
        for agent in (_agent("sess-A"), _agent("sess-A", api_mode="anthropic_messages")):
            headers = build_api_kwargs(agent, _MSGS).get("extra_headers") or {}
            assert _HEADER not in headers and "x-opencode-session" not in headers
        assert _HEADER not in _aux_headers("sess-A")
