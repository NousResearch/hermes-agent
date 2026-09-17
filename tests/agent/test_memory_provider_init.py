"""Regression tests for memory provider selection during AIAgent init."""

from types import SimpleNamespace
from unittest.mock import patch

import agent.runtime_cwd as rt


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_shutdown_memory_provider_is_idempotent():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    manager = MagicMock()
    agent = object.__new__(AIAgent)
    agent._memory_manager = manager
    agent.context_compressor = None
    agent.session_id = "session-1"

    agent.shutdown_memory_provider([{"role": "user", "content": "one"}])
    agent.shutdown_memory_provider([{"role": "user", "content": "two"}])

    manager.on_session_end.assert_called_once()
    manager.shutdown_all.assert_called_once()


def test_blank_memory_provider_does_not_auto_enable_honcho():
    """Blank memory.provider should remain opt-out even if Honcho fallback looks configured."""
    cfg = {"memory": {"provider": ""}, "agent": {}}
    honcho_cfg = SimpleNamespace(enabled=True, api_key="stale-key", base_url=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=honcho_cfg,
        ) as from_global_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    from_global_config.assert_not_called()
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_close_shuts_down_memory_provider():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = None
    agent.session_id = ""
    agent._session_messages = []

    agent.close()

    agent._memory_manager.shutdown_all.assert_called_once()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"
    assert "warning_callback" not in provider.init_kwargs
    assert "status_callback" not in provider.init_kwargs


class CoreShadowProvider:
    """Provider that tries to register tools shadowing built-in core tools."""

    name = "core-shadow"

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "shadows built-in clarify"},
            {"name": "delegate_task", "description": "shadows built-in delegate"},
            {"name": "honcho_search", "description": "legit memory tool"},
        ]


def test_core_tool_names_rejected_from_memory_routing_table():
    """Memory tools shadowing core tool names are rejected at registration (#40466).

    Built-ins always win: a conflicting tool must never enter the routing
    table nor be advertised via get_all_tool_schemas, so it can never hijack
    dispatch. The non-conflicting tool is preserved.
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.add_provider(CoreShadowProvider())

    # Reserved names never enter the routing table
    assert not mm.has_tool("clarify")
    assert not mm.has_tool("delegate_task")
    assert "clarify" not in mm._tool_to_provider
    assert "delegate_task" not in mm._tool_to_provider

    # Non-conflicting tool survives
    assert mm.has_tool("honcho_search")
    assert "honcho_search" in mm._tool_to_provider

    # Manager never advertises a schema it would refuse to route
    schema_names = {s.get("name") for s in mm.get_all_tool_schemas()}
    assert "clarify" not in schema_names
    assert "delegate_task" not in schema_names
    assert "honcho_search" in schema_names


# ---------------------------------------------------------------------------
# agent_workspace resolution (workspace identity for memory providers)
# ---------------------------------------------------------------------------


class _FakeSessionStore:
    """Minimal session store: only the reads memory-provider init performs."""

    def __init__(self, row=None, title=None):
        self._row = row
        self._title = title

    def get_session_title(self, session_id):
        return self._title

    def get_session(self, session_id):
        return self._row


class _RecordingAgent:
    """Minimal agent surface for ``_memory_provider_init_kwargs``."""

    def __init__(self, *, session_id="sess-ws", session_db=None):
        self.session_id = session_id
        self._session_db = session_db
        for ident in (
            "user_id", "user_id_alt", "user_name", "chat_id", "chat_name",
            "chat_type", "thread_id", "gateway_session_key",
        ):
            setattr(self, f"_{ident}", "")
        self._emit_warning = lambda *args, **kwargs: None
        self._emit_status = lambda *args, **kwargs: None


def _declare_project(tmp_path, monkeypatch, *, name, slug, folders):
    """Declare one project in a temp projects DB and point the DAO at it."""
    import hermes_cli.projects_db as pdb

    path = tmp_path / "projects.db"
    conn = pdb.connect(db_path=path)
    try:
        pdb.create_project(conn, name=name, slug=slug, folders=folders)
    finally:
        conn.close()
    monkeypatch.setattr(pdb, "projects_db_path", lambda: path)
    return path


def _no_projects_db(tmp_path, monkeypatch):
    """Point the projects DAO at a missing DB (hermetic: no declared projects)."""
    import hermes_cli.projects_db as pdb

    monkeypatch.setattr(pdb, "projects_db_path", lambda: tmp_path / "absent" / "projects.db")


class TestAgentWorkspaceResolution:
    """agent_workspace: declared project → git root → cwd basename; "" for sessions
    with no real workspace (gateway/API turns) — the ambient configured
    TERMINAL_CWD is never promoted to a workspace there."""

    def test_resolves_from_session_row_cwd(self, tmp_path, monkeypatch):
        from agent.agent_init import _memory_provider_init_kwargs

        proj = tmp_path / "myproj"
        proj.mkdir()
        _declare_project(tmp_path, monkeypatch, name="My Project", slug="myproject", folders=[str(proj)])
        store = _FakeSessionStore(row={"cwd": str(proj), "git_repo_root": ""})
        kwargs = _memory_provider_init_kwargs(_RecordingAgent(session_db=store), "cli")
        assert kwargs["agent_workspace"] == "myproject"

    def test_resolves_from_runtime_scope_without_row_cwd(self, tmp_path, monkeypatch):
        # Mirrors cron._CronRunScope.enter(): the job workdir is pinned on the session
        # scope before the agent (and its memory provider) is built.
        from agent.agent_init import _memory_provider_init_kwargs

        workdir = tmp_path / "jobwork"
        workdir.mkdir()
        _declare_project(tmp_path, monkeypatch, name="Job Project", slug="jobproj", folders=[str(workdir)])
        store = _FakeSessionStore(row={"cwd": "", "git_repo_root": ""})
        token = rt.set_session_cwd(str(workdir))
        try:
            kwargs = _memory_provider_init_kwargs(_RecordingAgent(session_db=store), "cron")
        finally:
            rt._SESSION_CWD.reset(token)
        assert kwargs["agent_workspace"] == "jobproj"

    def test_gateway_session_with_no_workspace_is_empty(self, tmp_path, monkeypatch):
        # An explicit empty session cwd (gateway/API turns) must NOT fall back to
        # the ambient configured terminal cwd.
        from agent.agent_init import _memory_provider_init_kwargs

        _no_projects_db(tmp_path, monkeypatch)
        ambient = tmp_path / "workspace-default"
        ambient.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(ambient))
        store = _FakeSessionStore(row={"cwd": None, "git_repo_root": None})
        token = rt.set_session_cwd("")
        try:
            kwargs = _memory_provider_init_kwargs(_RecordingAgent(session_db=store), "discord")
        finally:
            rt._SESSION_CWD.reset(token)
        assert kwargs["agent_workspace"] == ""

    def test_local_cli_without_session_context_uses_surface_cwd(self, tmp_path, monkeypatch):
        from agent.agent_init import _memory_provider_init_kwargs

        _no_projects_db(tmp_path, monkeypatch)
        launch = tmp_path / "launchdir"
        launch.mkdir()
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(launch)
        token = rt._SESSION_CWD.set(rt._UNSET)  # no session context on this task
        try:
            kwargs = _memory_provider_init_kwargs(_RecordingAgent(session_db=None), "cli")
        finally:
            rt._SESSION_CWD.reset(token)
        assert kwargs["agent_workspace"] == "launchdir"

    def test_broken_session_store_is_safe(self, tmp_path, monkeypatch):
        from agent.agent_init import _memory_provider_init_kwargs

        class _BrokenStore:
            def get_session_title(self, session_id):
                raise RuntimeError("locked")

            def get_session(self, session_id):
                raise RuntimeError("locked")

        _no_projects_db(tmp_path, monkeypatch)
        launch = tmp_path / "still-works"
        launch.mkdir()
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(launch)
        token = rt._SESSION_CWD.set(rt._UNSET)
        try:
            kwargs = _memory_provider_init_kwargs(_RecordingAgent(session_db=_BrokenStore()), "cli")
        finally:
            rt._SESSION_CWD.reset(token)
        assert kwargs["agent_workspace"] == "still-works"


def test_aiagent_forwards_agent_workspace_to_memory_provider(tmp_path, monkeypatch):
    """End to end through AIAgent init: the resolved workspace identity reaches the
    provider (replacing the old hard-coded ``"hermes"`` constant)."""
    from unittest.mock import MagicMock

    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}
    proj = tmp_path / "resolved-proj"
    proj.mkdir()
    _declare_project(tmp_path, monkeypatch, name="Resolved", slug="resolved", folders=[str(proj)])
    store = MagicMock()
    store.get_session_title.return_value = None
    store.get_session.return_value = {"cwd": str(proj), "git_repo_root": ""}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-ws",
            platform="cli",
            session_db=store,
        )

    assert provider.init_kwargs is not None
    assert provider.init_kwargs["agent_workspace"] == "resolved"


