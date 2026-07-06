"""CLI resume workspace guard entrypoint tests."""

from __future__ import annotations

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _FakeDB:
    def get_session(self, session_id: str):
        assert session_id == "target-session"
        return {"id": session_id, "git_repo_root": None, "cwd": None}

    def resolve_resume_session_id(self, session_id: str) -> str:
        return session_id

    def get_messages_as_conversation(self, session_id: str):
        assert session_id == "target-session"
        return [
            {
                "role": "user",
                "content": "[CONTEXT COMPACTION] summary\n<!-- HERMES_WORKSPACE:/workspace/other -->",
            }
        ]


class _Harness(CLIAgentSetupMixin):
    def __init__(self):
        self.agent = None
        self._resumed = True
        self._session_db = _FakeDB()
        self.session_id = "target-session"
        self.conversation_history = []
        self.cwd = "/workspace/current"
        self.tool_progress_mode = "off"
        self.restore_cwd_called = False

    def _prepare_runtime(self):  # unused; protects against accidental calls
        raise AssertionError("unexpected runtime preparation")

    def _install_tool_callbacks(self):
        pass

    def _ensure_tirith_security(self):
        pass

    def _ensure_runtime_credentials(self) -> bool:
        return True

    def _restore_session_cwd(self, *_args, **_kwargs):
        self.restore_cwd_called = True


def test_cli_resume_blocks_compaction_stamped_workspace_mismatch(monkeypatch):
    import cli
    import hermes_cli.mcp_startup

    monkeypatch.setattr(cli, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(hermes_cli.mcp_startup, "wait_for_mcp_discovery", lambda: None)

    harness = _Harness()

    assert harness._init_agent() is False
    assert harness.agent is None
    assert harness.conversation_history == []
    assert harness.restore_cwd_called is False
