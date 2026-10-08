"""Tests for per-user memory scoping via user_id threading.

Verifies that gateway user_id flows from AIAgent -> MemoryManager -> plugins,
so each gateway user gets their own memory bucket instead of sharing a static one.
"""

import json
from unittest.mock import MagicMock, patch

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager

# ---------------------------------------------------------------------------
# Concrete test provider that records init kwargs
# ---------------------------------------------------------------------------

class RecordingProvider(MemoryProvider):
    """Minimal provider that records what initialize() receives."""

    def __init__(self, name="recording"):
        self._name = name
        self._init_kwargs = {}
        self._init_session_id = None

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._init_session_id = session_id
        self._init_kwargs = dict(kwargs)

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({})

    def shutdown(self):
        pass

# ---------------------------------------------------------------------------
# MemoryManager user_id threading tests
# ---------------------------------------------------------------------------

class TestMemoryManagerUserIdThreading:
    """Verify user_id reaches providers via initialize_all."""

    def test_no_user_id_when_cli(self):
        """CLI sessions should not have user_id in kwargs."""
        mgr = MemoryManager()
        p = RecordingProvider()
        mgr.add_provider(p)

        mgr.initialize_all(
            session_id="sess-456",
            platform="cli",
        )

        assert "user_id" not in p._init_kwargs
        assert p._init_kwargs.get("platform") == "cli"

    def test_multiple_providers_all_receive_user_id(self):
        mgr = MemoryManager()
        # Use one provider named "builtin" (always accepted) and one external
        p1 = RecordingProvider("builtin")
        p2 = RecordingProvider("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        mgr.initialize_all(
            session_id="sess-multi",
            platform="slack",
            user_id="slack_U12345",
        )

        assert p1._init_kwargs.get("user_id") == "slack_U12345"
        assert p1._init_kwargs.get("platform") == "slack"
        assert p2._init_kwargs.get("user_id") == "slack_U12345"
        assert p2._init_kwargs.get("platform") == "slack"

    def test_session_title_provenance_and_cwd_reach_provider(self, tmp_path):
        from run_agent import AIAgent

        provider = RecordingProvider()
        session_db = MagicMock()
        session_db.get_session_title.return_value = "Generated title"
        session_db.get_session_title_source.return_value = "llm"

        with patch(
            "model_tools.get_tool_definitions",
            return_value=[],
        ), patch(
            "model_tools.check_toolset_requirements",
            return_value={},
        ), patch(
            "agent.process_bootstrap.OpenAI",
        ), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"memory": {"provider": "recording"}},
        ), patch(
            "plugins.memory.load_memory_provider",
            return_value=provider,
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                platform="telegram",
                session_id="session-with-title",
                session_db=session_db,
                gateway_session_key="agent:main:telegram:dm:42",
                cwd=str(tmp_path),
            )

        assert provider._init_kwargs["session_title"] == "Generated title"
        assert provider._init_kwargs["session_title_source"] == "llm"
        assert (
            provider._init_kwargs["gateway_session_key"]
            == "agent:main:telegram:dm:42"
        )
        assert provider._init_kwargs["cwd"] == str(tmp_path)
        agent.close()

