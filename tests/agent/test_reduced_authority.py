from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _reduced_agent(**overrides):
    kwargs = {
        "model": "openai/gpt-oss-120b",
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "enabled_toolsets": [],
        "skip_memory": True,
        "skip_context_files": True,
        "session_db": None,
        "quiet_mode": True,
        "reduced_authority": True,
    }
    kwargs.update(overrides)
    return AIAgent(**kwargs)


def test_reduced_authority_is_established_during_agent_construction():
    agent = _reduced_agent()

    assert agent._reduced_authority is True
    assert agent.compression_enabled is False
    assert type(agent.context_compressor).__name__ == "ContextCompressor"
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._skip_mcp_refresh is True
    assert agent._skip_plugin_hooks is True
    assert agent._skip_extension_middleware is True
    assert agent._environment_probe is False


def test_reduced_authority_skips_error_hooks_before_discovery():
    agent = _reduced_agent()

    with patch("hermes_cli.plugins.has_hook") as has_hook:
        agent._invoke_api_request_error_hook(
            task_id="task",
            turn_id="turn",
            api_request_id="request",
            api_call_count=1,
            api_start_time=0.0,
            api_kwargs={"messages": [{"role": "user", "content": "private text"}]},
            error_type="RuntimeError",
            error_message="private text",
        )

    has_hook.assert_not_called()


def test_reduced_authority_never_writes_json_session_snapshots(tmp_path):
    agent = object.__new__(AIAgent)
    agent._reduced_authority = True
    agent._session_json_enabled = True
    agent._session_messages = [{"role": "user", "content": "private text"}]
    agent.logs_dir = tmp_path
    agent.session_id = "reduced-session"

    agent._save_session_log()

    assert not (tmp_path / "session_reduced-session.json").exists()


def test_reduced_authority_uses_only_the_explicit_safe_system_prompt():
    from agent.conversation_loop import _restore_or_build_system_prompt

    agent = SimpleNamespace(
        _reduced_authority=True,
        ephemeral_system_prompt="SAFE ATTACHMENT POLICY",
        _cached_system_prompt=None,
        _session_db=None,
        _build_system_prompt=MagicMock(return_value="PRIVATE HOST CONTEXT"),
    )

    _restore_or_build_system_prompt(agent, None, [])

    assert agent._cached_system_prompt == "SAFE ATTACHMENT POLICY"
    agent._build_system_prompt.assert_not_called()


def test_reduced_authority_hides_raw_turn_content_from_log_preview():
    from agent.turn_context import _turn_log_preview

    agent = SimpleNamespace(_reduced_authority=True)
    sentinel = "LEAK_SENTINEL_123456"

    preview = _turn_log_preview(agent, f"Summarize\n\n{sentinel}", lambda value: value)

    assert sentinel not in preview
    assert preview == "[reduced-authority untrusted context]"


def test_reduced_authority_never_writes_api_request_debug_dump():
    agent = object.__new__(AIAgent)
    agent._reduced_authority = True

    with patch("agent.agent_runtime_helpers.dump_api_request_debug") as dump:
        result = agent._dump_api_request_debug(
            {"messages": [{"role": "user", "content": "private attachment text"}]},
            reason="provider-error",
        )

    assert result is None
    dump.assert_not_called()


def test_reduced_authority_ignores_hidden_moa_envelopes():
    from agent.conversation_loop import _decode_allowed_moa_turn

    agent = SimpleNamespace(_reduced_authority=True)
    with patch("hermes_cli.moa_config.decode_moa_turn", return_value=("decoded", {"models": ["other/provider"]})) as decode:
        assert _decode_allowed_moa_turn(agent, "encoded envelope") == ("encoded envelope", None)

    decode.assert_not_called()
