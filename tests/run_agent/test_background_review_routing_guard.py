"""Review runtime normalization and native-tool boundary regressions."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.background_review import _resolve_review_runtime, build_cache_parity_fork


def _parent():
    return SimpleNamespace(
        provider="openai-codex", model="parent", request_overrides={"x": 1},
        _credential_pool=None, max_tokens=123, acp_command=None, acp_args=[],
        platform="test", session_id="parent", _memory_store=None, _memory_enabled=False,
        _user_profile_enabled=False, _cached_system_prompt="system", session_start=None,
        _current_main_runtime=lambda: {"api_mode": "codex_app_server", "api_key": "parent-key"},
    )


@pytest.mark.parametrize("mode", ["codex_app_server", "anthropic_messages"])
@pytest.mark.parametrize("pool", [None, object()])
def test_routed_runtime_normalizes_only_native_mode_preserving_fields(mode, pool):
    runtime = dict(provider="openai-codex", model="review", api_key="key", base_url="url",
                   api_mode=mode, credential_pool=pool, command="cmd", args=["arg"],
                   request_overrides={"extra": True})
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime):
        resolved = _resolve_review_runtime(_parent(), {"provider": "openai-codex", "model": "review"})
    expected = {**runtime, "api_mode": "codex_responses" if mode == "codex_app_server" else mode,
                "routed": True}
    assert resolved == expected
    assert runtime["api_mode"] == mode


def test_same_model_inherits_normalized_runtime():
    resolved = _resolve_review_runtime(_parent(), {"provider": "openai-codex", "model": "parent"})
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["routed"] is False
    assert resolved["api_key"] == "parent-key"
    assert resolved["request_overrides"] == {"x": 1}


@pytest.mark.parametrize("origin", ["background_review", "side_question"])
def test_fork_rejects_residual_native_runtime_before_construction(origin):
    with patch("agent.background_review._resolve_review_runtime", return_value={"api_mode": "codex_app_server"}), patch("run_agent.AIAgent") as constructor:
        with pytest.raises(ValueError, match="codex_app_server"):
            build_cache_parity_fork(_parent(), {}, max_iterations=3, write_origin=origin)
    constructor.assert_not_called()


def test_routed_codex_request_contains_digest_and_medium_wire_effort():
    from agent import background_review as review
    from agent.transports.codex import ResponsesApiTransport
    from unittest.mock import MagicMock
    parent = _parent()
    fork = MagicMock()
    fork._session_messages = []
    snapshot = [{"role": role, "content": f"history {i}"}
                for i in range(30) for role in ("user", "assistant")]
    state = review._ReviewForkState()
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
        "provider": "openai-codex", "model": "gpt-5.6-sol", "api_mode": "codex_app_server",
        "api_key": "test-key", "base_url": "https://chatgpt.com/backend-api/codex",
    }), patch("run_agent.AIAgent", return_value=fork) as constructor, \
         patch.object(review, "_review_tool_whitelist", return_value=(set(), set())), \
         patch.object(review, "_record_review_usage_to_parent"):
        review._run_review_fork(parent, snapshot, "review", {"provider": "openai-codex", "model": "gpt-5.6-sol"}, None, state)
    kwargs = constructor.call_args.kwargs
    assert kwargs["api_mode"] == "codex_responses"
    assert "reasoning_config" not in kwargs
    history = fork.run_conversation.call_args.kwargs["conversation_history"]
    assert history == review._digest_history(snapshot)
    assert history != snapshot
    request = ResponsesApiTransport().build_kwargs(
        model=kwargs["model"], messages=history, tools=None,
        base_url=kwargs["base_url"], reasoning_config=kwargs.get("reasoning_config"),
    )
    assert request["reasoning"]["effort"] == "medium"
    assert "[Earlier conversation digest" in str(request["input"])
