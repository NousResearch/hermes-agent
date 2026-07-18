"""Read-only memory isolation for ephemeral agent turns."""

from unittest.mock import Mock

from agent.agent_runtime_helpers import dump_api_request_debug
from run_agent import AIAgent


def _agent_with_memory_manager(*, read_only: bool):
    agent = object.__new__(AIAgent)
    agent._memory_read_only = read_only
    agent._memory_manager = Mock()
    agent.context_compressor = None
    agent.session_id = "ephemeral-room-turn"
    return agent


def test_read_only_turn_skips_external_memory_sync_and_prefetch():
    agent = _agent_with_memory_manager(read_only=True)

    agent._sync_external_memory_for_turn(
        original_user_message="private room input",
        final_response="private room response",
        interrupted=False,
        messages=[{"role": "user", "content": "private room input"}],
    )

    agent._memory_manager.sync_all.assert_not_called()
    agent._memory_manager.queue_prefetch_all.assert_not_called()


def test_read_only_shutdown_does_not_submit_session_end_observations():
    agent = _agent_with_memory_manager(read_only=True)

    agent.shutdown_memory_provider(
        [{"role": "user", "content": "private room input"}],
    )

    agent._memory_manager.on_session_end.assert_not_called()
    agent._memory_manager.shutdown_all.assert_called_once_with()


def test_normal_shutdown_preserves_session_end_observation():
    agent = _agent_with_memory_manager(read_only=False)
    messages = [{"role": "user", "content": "ordinary input"}]

    agent.shutdown_memory_provider(messages)

    agent._memory_manager.on_session_end.assert_called_once_with(messages)
    agent._memory_manager.shutdown_all.assert_called_once_with()


def test_read_only_agent_loads_builtin_memory_without_exposing_memory_tool(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "memory:\n  memory_enabled: true\n  user_profile_enabled: true\n",
        encoding="utf-8",
    )
    (memories / "MEMORY.md").write_text("- stable identity marker\n", encoding="utf-8")
    (memories / "USER.md").write_text("- stable user marker\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    agent = AIAgent(
        provider="openai",
        api_key="test-only",
        base_url="http://127.0.0.1:1/v1",
        model="test-model",
        enabled_toolsets=["memory"],
        quiet_mode=True,
        persist_disabled=True,
        memory_read_only=True,
        tools_disabled=True,
    )

    snapshot = agent._memory_store._system_prompt_snapshot
    assert "stable identity marker" in snapshot["memory"]
    assert "stable user marker" in snapshot["user"]
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._memory_nudge_interval == 0
    assert agent._skill_nudge_interval == 0
    assert agent._persist_disabled is True


def test_persistence_disabled_agent_never_writes_request_debug_dump(tmp_path):
    agent = Mock()
    agent._persist_disabled = True
    agent.logs_dir = tmp_path

    result = dump_api_request_debug(
        agent,
        {"messages": [{"role": "user", "content": "private room input"}]},
        reason="provider_error",
        error=RuntimeError("mock failure"),
    )

    assert result is None
    assert list(tmp_path.glob("request_dump_*.json")) == []
