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
                   request_overrides={"extra": True}, max_output_tokens=456)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime):
        resolved = _resolve_review_runtime(_parent(), {"provider": "openai-codex", "model": "review"})
    expected = {**runtime, "api_mode": "codex_responses" if mode == "codex_app_server" else mode,
                "max_tokens": 456, "routed": True}
    del expected["max_output_tokens"]
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
