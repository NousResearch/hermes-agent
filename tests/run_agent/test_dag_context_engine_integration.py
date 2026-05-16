"""DAG context engine native loading and non-destructive compression tests."""

from unittest.mock import MagicMock, patch


def test_context_engine_default_remains_builtin_compressor():
    cfg = {"agent": {}, "compression": {"enabled": False}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor.name == "compressor"


def test_context_engine_dag_loads_native_without_plugin(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": False}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine") as plugin_loader,
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="sess",
            session_db=db,
        )

    assert agent.context_compressor.name == "dag"
    plugin_loader.assert_not_called()


def test_dag_compress_context_preserves_session_and_skips_destructive_side_effects(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess", "cli")
    db.append_message("sess", "user", "hello")
    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="sess",
            session_db=db,
        )

    db.end_session = MagicMock()
    agent.commit_memory_session = MagicMock()
    old_session_id = agent.session_id

    compressed, _ = agent._compress_context(
        [{"role": "user", "content": "hello"}],
        "",
        approx_tokens=100,
    )

    assert compressed[-1]["content"] == "hello"
    assert agent.session_id == old_session_id
    db.end_session.assert_not_called()
    agent.commit_memory_session.assert_not_called()
