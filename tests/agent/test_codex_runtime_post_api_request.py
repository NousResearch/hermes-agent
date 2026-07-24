"""Codex app-server turns must reach ``post_api_request`` consumers.

The ordinary conversation loop fires ``post_api_request`` for every API turn,
but Codex app-server sessions record usage through
``_record_codex_app_server_usage`` and bypassed the hook — so plugin-side
accounting (usage ledgers, spend guards) was blind to subscription-billed
Codex turns.
"""
from types import SimpleNamespace

import pytest

from agent.codex_runtime import _record_codex_app_server_usage
from hermes_cli.plugins import get_plugin_manager


class _FakeAgent:
    def __init__(self):
        self.model = "gpt-5.1-codex"
        self.provider = "openai-codex"
        self.base_url = ""
        self.api_key = ""
        self.api_mode = "codex-app-server"
        self.platform = "cli"
        self.session_id = "codex-test-session"
        self.context_compressor = None
        self._session_db = None
        self._session_db_created = False
        for attr in (
            "session_api_calls", "session_prompt_tokens",
            "session_completion_tokens", "session_total_tokens",
            "session_input_tokens", "session_output_tokens",
            "session_cache_read_tokens", "session_cache_write_tokens",
            "session_reasoning_tokens",
        ):
            setattr(self, attr, 0)
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = ""
        self.session_cost_source = ""


def _turn():
    return SimpleNamespace(
        token_usage_last={
            "inputTokens": 12000, "cachedInputTokens": 3000,
            "outputTokens": 800, "reasoningOutputTokens": 200,
            "totalTokens": 16000,
        },
        model_context_window=None,
    )


@pytest.fixture()
def hook_capture():
    mgr = get_plugin_manager()
    saved = list(mgr._hooks.get("post_api_request", []))
    captured = []
    mgr._hooks["post_api_request"] = [lambda **kw: captured.append(kw)]
    try:
        yield captured
    finally:
        mgr._hooks["post_api_request"] = saved


def test_codex_app_server_turn_fires_post_api_request(hook_capture):
    agent = _FakeAgent()
    result = _record_codex_app_server_usage(agent, _turn())

    assert len(hook_capture) == 1
    payload = hook_capture[0]
    assert payload["provider"] == "openai-codex"
    assert payload["model"] == "gpt-5.1-codex"
    # canonical usage buckets, same shape conversation_loop delivers
    assert payload["usage"]["input_tokens"] == 12000
    assert payload["usage"]["cache_read_tokens"] == 3000
    assert payload["usage"]["output_tokens"] == 800
    # session accounting and the returned usage dict are unchanged
    assert agent.session_api_calls == 1
    assert agent.session_total_tokens == 16000
    assert result.get("total_tokens") == 16000


def test_broken_hook_never_breaks_usage_recording():
    mgr = get_plugin_manager()
    saved = list(mgr._hooks.get("post_api_request", []))
    mgr._hooks["post_api_request"] = [
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))]
    try:
        agent = _FakeAgent()
        _record_codex_app_server_usage(agent, _turn())
        assert agent.session_api_calls == 1
    finally:
        mgr._hooks["post_api_request"] = saved
