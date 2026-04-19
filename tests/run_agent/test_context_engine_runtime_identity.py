from unittest.mock import MagicMock, patch

from agent.context_engine import ContextEngine


class _StubEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "stub"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None):
        return messages


class _StubLCMEngine(_StubEngine):
    @property
    def name(self) -> str:
        return "lcm"


def test_runtime_identity_reports_builtin_context_engine_and_known_state_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    cfg = {"context": {"engine": "compressor"}, "agent": {}}

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
            session_db=db,
            session_id="host-session-1",
        )

    identity = agent.get_context_engine_runtime_identity()

    assert identity["configured_engine"] == "compressor"
    assert identity["active_engine"] == "compressor"
    assert identity["origin"] == "built-in"
    assert identity["host_session_id"] == "host-session-1"
    assert identity["host_session_known_to_state_db"] is True
    assert identity["plugin_session_id"] is None
    assert identity["warnings"] == []


def test_runtime_identity_reports_plugin_binding_mismatch_and_missing_lcm_command():
    engine = _StubLCMEngine()
    engine.update_model = MagicMock()
    engine._hermes_context_engine_origin = "plugin:user"
    engine._hermes_context_engine_path = "/tmp/hermes/plugins/hermes-lcm"
    engine.get_status = MagicMock(return_value={
        "engine": "lcm",
        "conversation_id": "conv-123",
        "lifecycle": {
            "current_session_id": "plugin-session-9",
            "last_finalized_session_id": "plugin-session-8",
            "conversation_id": "conv-123",
        },
    })

    cfg = {"context": {"engine": "lcm"}, "agent": {}}
    fake_session_db = MagicMock()
    fake_session_db.get_session.return_value = None

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=None),
        patch("hermes_cli.plugins.get_plugin_context_engine", return_value=engine),
        patch("hermes_cli.plugins.get_plugin_commands", return_value={}),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="openrouter/auto",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=fake_session_db,
            session_id="host-session-2",
        )

    identity = agent.get_context_engine_runtime_identity()

    assert identity["configured_engine"] == "lcm"
    assert identity["active_engine"] == "lcm"
    assert identity["origin"] == "plugin:user"
    assert identity["path"] == "/tmp/hermes/plugins/hermes-lcm"
    assert identity["host_session_id"] == "host-session-2"
    assert identity["host_session_known_to_state_db"] is False
    assert identity["plugin_session_id"] == "plugin-session-9"
    assert identity["plugin_conversation_id"] == "conv-123"
    assert identity["plugin_last_finalized_session_id"] == "plugin-session-8"
    assert "state_db_missing_host_session" in identity["warnings"]
    assert "plugin_session_mismatch" in identity["warnings"]
    assert "missing_expected_slash_command:/lcm" in identity["warnings"]