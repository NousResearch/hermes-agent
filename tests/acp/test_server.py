"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SessionInfo,
    TextContentBlock,
    Usage,
)
from acp_adapter.server import (
    HermesACPAgent,
    HERMES_VERSION,
    _ROUTE_FORENSICS_LOG,
    _append_route_forensics,
    _build_routed_prompt,
    _load_routed_memory,
    _parse_tool_json,
    _route_turn_id_from_kwargs,
)
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_correct_protocol_version(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == acp.PROTOCOL_VERSION

    @pytest.mark.asyncio
    async def test_initialize_returns_agent_info(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert resp.agent_info is not None
        assert isinstance(resp.agent_info, Implementation)
        assert resp.agent_info.name == "hermes-agent"
        assert resp.agent_info.version == HERMES_VERSION

    @pytest.mark.asyncio
    async def test_initialize_returns_capabilities(self, agent):
        resp = await agent.initialize(protocol_version=1)
        caps = resp.agent_capabilities
        assert isinstance(caps, AgentCapabilities)
        assert caps.load_session is True
        assert caps.session_capabilities is not None
        assert caps.session_capabilities.fork is not None
        assert caps.session_capabilities.list is not None
        assert caps.session_capabilities.resume is not None

    @pytest.mark.asyncio
    async def test_initialize_capabilities_wire_format(self, agent):
        """Verify the JSON wire format uses correct aliases so ACP clients see the right keys."""
        resp = await agent.initialize(protocol_version=1)
        payload = resp.agent_capabilities.model_dump(by_alias=True, exclude_none=True)
        assert payload["loadSession"] is True
        session_caps = payload["sessionCapabilities"]
        assert "fork" in session_caps
        assert "list" in session_caps
        assert "resume" in session_caps


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_matching_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_is_case_insensitive(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="OpenRouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_mismatched_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="totally-invalid-method")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert resp is None


# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:
    @pytest.mark.asyncio
    async def test_new_session_creates_session(self, agent):
        resp = await agent.new_session(cwd="/home/user/project")
        assert isinstance(resp, NewSessionResponse)
        assert resp.session_id
        # Session should be retrievable from the manager
        state = agent.session_manager.get_session(resp.session_id)
        assert state is not None
        assert state.cwd == "/home/user/project"

    @pytest.mark.asyncio
    async def test_new_session_returns_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(model="gpt-5.4", provider="openai-codex")
        )
        acp_agent = HermesACPAgent(session_manager=manager)

        with patch(
            "hermes_cli.models.curated_models_for_provider",
            return_value=[("gpt-5.4", "recommended"), ("gpt-5.4-mini", "")],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert resp.models.available_models[0].model_id == "openai-codex:gpt-5.4"
        assert resp.models.available_models[0].description is not None
        assert "Provider:" in resp.models.available_models[0].description

    @pytest.mark.asyncio
    async def test_new_session_returns_mode_state(self, agent):
        resp = await agent.new_session(cwd="/tmp")

        assert isinstance(resp.modes, SessionModeState)
        assert resp.modes.current_mode_id == "standard"
        assert [mode.id for mode in resp.modes.available_modes] == [
            "standard",
            "auto",
            "force-spar",
            "force-moa",
            "force-moa-spar",
        ]

    @pytest.mark.asyncio
    async def test_available_commands_include_help(self, agent):
        help_cmd = next(
            (cmd for cmd in agent._available_commands() if cmd.name == "help"),
            None,
        )

        assert help_cmd is not None
        assert help_cmd.description == "List available commands"
        assert help_cmd.input is None

    @pytest.mark.asyncio
    async def test_send_available_commands_update(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent._send_available_commands_update("session-123")

        mock_conn.session_update.assert_awaited_once()
        call = mock_conn.session_update.await_args
        assert call.kwargs["session_id"] == "session-123"
        update = call.kwargs["update"]
        assert isinstance(update, AvailableCommandsUpdate)
        assert update.session_update == "available_commands_update"
        assert [cmd.name for cmd in update.available_commands] == [
            "help",
            "model",
            "tools",
            "context",
            "reset",
            "compact",
            "version",
        ]
        model_cmd = next(
            cmd for cmd in update.available_commands if cmd.name == "model"
        )
        assert model_cmd.input is not None
        assert model_cmd.input.root.hint == "model name to switch to"

    @pytest.mark.asyncio
    async def test_cancel_sets_event(self, agent):
        resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(resp.session_id)
        assert not state.cancel_event.is_set()
        await agent.cancel(session_id=resp.session_id)
        assert state.cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session_is_noop(self, agent):
        # Should not raise
        await agent.cancel(session_id="does-not-exist")

    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None

    @pytest.mark.asyncio
    async def test_resume_session_creates_new_if_missing(self, agent):
        resume_resp = await agent.resume_session(cwd="/tmp", session_id="nonexistent")
        assert isinstance(resume_resp, ResumeSessionResponse)


# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id

    @pytest.mark.asyncio
    async def test_list_sessions_includes_title_and_updated_at(self, agent):
        with patch.object(
            agent.session_manager,
            "list_sessions",
            return_value=[
                {
                    "session_id": "session-1",
                    "cwd": "/tmp/project",
                    "title": "Fix Zed session history",
                    "updated_at": 123.0,
                }
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp/project")

        assert isinstance(resp.sessions[0], SessionInfo)
        assert resp.sessions[0].title == "Fix Zed session history"
        assert resp.sessions[0].updated_at == "123.0"

    @pytest.mark.asyncio
    async def test_list_sessions_passes_cwd_filter(self, agent):
        with patch.object(agent.session_manager, "list_sessions", return_value=[]) as mock_list:
            await agent.list_sessions(cwd="/mnt/e/Projects/AI/browser-link-3")

        mock_list.assert_called_once_with(cwd="/mnt/e/Projects/AI/browser-link-3")

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_first_page(self, agent):
        from acp_adapter import server as acp_server

        infos = [
            {"session_id": f"s{i}", "cwd": "/tmp", "title": None, "updated_at": 0.0}
            for i in range(acp_server._LIST_SESSIONS_PAGE_SIZE + 5)
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions()

        assert len(resp.sessions) == acp_server._LIST_SESSIONS_PAGE_SIZE
        assert resp.next_cursor == resp.sessions[-1].session_id

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_no_more(self, agent):
        infos = [
            {"session_id": f"s{i}", "cwd": "/tmp", "title": None, "updated_at": 0.0}
            for i in range(3)
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions()

        assert len(resp.sessions) == 3
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_cursor_resumes_after_match(self, agent):
        infos = [
            {"session_id": "s1", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s2", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s3", "cwd": "/tmp", "title": None, "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions(cursor="s1")

        assert [s.session_id for s in resp.sessions] == ["s2", "s3"]
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_unknown_cursor_returns_empty(self, agent):
        infos = [
            {"session_id": "s1", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s2", "cwd": "/tmp", "title": None, "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions(cursor="does-not-exist")

        assert resp.sessions == []
        assert resp.next_cursor is None

# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:
    def test_parse_tool_json_extracts_wrapped_json_object(self):
        payload = _parse_tool_json(
            "Here is the tool output:\n```json\n{\"success\": true, \"response\": \"ok\"}\n```\nThanks.",
            stage="wrapped",
        )

        assert payload["success"] is True
        assert payload["response"] == "ok"

    def test_parse_tool_json_prefers_outer_payload_over_inner_fence(self):
        """Race regression: a real tool payload whose `response` field contains a
        markdown code block with an inner JSON-like dict must NOT cause the parser
        to latch onto the inner dict. Caught live in the MoA + Spar bridge: a
        successful MoA run with an embedded ```json {...}``` example was being
        mis-parsed as a small inner dict missing `success`/`response`, which the
        bridge then surfaced as 'MoA failed without a usable response.'"""
        import json as _json
        moa_output = _json.dumps({
            "success": True,
            "response": "Run this:\n\n```json\n{\"foo\": \"bar\"}\n```\n\nThen ship.",
            "models_used": {"aggregator_model": "xiaomi/mimo-v2-pro"},
        }, indent=2, ensure_ascii=False)

        payload = _parse_tool_json(moa_output, stage="moa")

        assert payload.get("success") is True
        assert "response" in payload
        assert payload.get("models_used", {}).get("aggregator_model") == "xiaomi/mimo-v2-pro"

    def test_route_turn_id_accepts_acp_and_python_param_names(self):
        assert _route_turn_id_from_kwargs({"messageId": "camel"}) == "camel"
        assert _route_turn_id_from_kwargs({"message_id": "snake"}) == "snake"
        assert _route_turn_id_from_kwargs({}) != ""

    def test_route_forensics_rotates_when_size_cap_is_reached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setattr("acp_adapter.server._ROUTE_FORENSICS_MAX_BYTES", 40)

        _append_route_forensics({"event": "old", "payload": "x" * 80})
        _append_route_forensics({"event": "new"})

        forensic_path = tmp_path / ".hermes" / "logs" / _ROUTE_FORENSICS_LOG
        rotated_path = forensic_path.with_name(f"{forensic_path.name}.1")
        assert json.loads(rotated_path.read_text().splitlines()[0])["event"] == "old"
        assert json.loads(forensic_path.read_text().splitlines()[0])["event"] == "new"

    @pytest.mark.asyncio
    async def test_set_session_mode_returns_response(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn
        resp = await agent.set_session_mode(mode_id="chat", session_id=new_resp.session_id)
        state = agent.session_manager.get_session(new_resp.session_id)

        assert isinstance(resp, SetSessionModeResponse)
        assert getattr(state, "mode", None) == "standard"
        mock_conn.session_update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "chat", "sessionId": new_resp.session_id},
            False,
        )
        config_result = await router(
            "session/set_config_option",
            {
                "configId": "approval_mode",
                "sessionId": new_resp.session_id,
                "value": "auto",
            },
            False,
        )

        assert mode_result == {}
        assert config_result == {"configOptions": []}

    @pytest.mark.asyncio
    async def test_router_accepts_unstable_model_switch_when_enabled(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent, use_unstable_protocol=True)

        result = await router(
            "session/set_model",
            {"modelId": "gpt-5.4", "sessionId": new_resp.session_id},
            False,
        )
        state = agent.session_manager.get_session(new_resp.session_id)

        assert result == {}
        assert state.model == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_set_session_model_accepts_provider_prefixed_choice(self, tmp_path, monkeypatch):
        runtime_calls = []

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            runtime_calls.append(requested)
            provider = requested or "openrouter"
            return {
                "provider": provider,
                "api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "openrouter/gpt-5"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        manager = SessionManager(db=SessionDB(tmp_path / "state.db"))

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            acp_agent = HermesACPAgent(session_manager=manager)
            state = manager.create_session(cwd="/tmp")
            result = await acp_agent.set_session_model(
                model_id="anthropic:claude-sonnet-4-6",
                session_id=state.session_id,
            )

        assert isinstance(result, SetSessionModelResponse)
        assert state.model == "claude-sonnet-4-6"
        assert state.agent.provider == "anthropic"
        assert state.agent.base_url == "https://anthropic.example/v1"
        assert runtime_calls[-1] == "anthropic"


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_build_routed_prompt_truncates_oversized_latest_message(self):
        huge_message = "x" * 20000

        routed = _build_routed_prompt(
            "give me the answer",
            [{"role": "assistant", "content": huge_message}],
            max_messages=8,
            max_chars=120,
        )

        assert "Recent conversation:" in routed
        assert "Assistant: " in routed
        assert "..." in routed
        assert huge_message not in routed
        assert routed.endswith("User: give me the answer")

    def test_build_routed_prompt_includes_existing_local_file(self, tmp_path):
        local_file = tmp_path / "review-me.md"
        local_file.write_text("# Draft\n\nCheck this claim.", encoding="utf-8")

        routed = _build_routed_prompt(f"review {local_file}", [])

        assert "Local file content resolved by Hermes before routing:" in routed
        assert f"--- BEGIN LOCAL FILE: {local_file} ---" in routed
        assert "# Draft\n\nCheck this claim." in routed
        assert f"User: review {local_file}" in routed

    def test_build_routed_prompt_includes_no_tool_framing_no_history(self, tmp_path, monkeypatch):
        """Routed prompt MUST include the no-tool-framing prefix even when
        history is empty, so MoA reference models do not hallucinate
        <tool_call>/<function=...> XML markup."""
        # Isolate from the developer's real ~/.hermes/memories during tests.
        monkeypatch.setattr(
            "acp_adapter.server.Path.home", classmethod(lambda cls: tmp_path)
        )

        routed = _build_routed_prompt("audit this report", [])

        assert "OPERATING CONSTRAINTS" in routed
        assert "no execution tools" in routed.lower()
        assert "<tool_call>" in routed  # forbidden tokens listed in prefix
        assert "<function=" in routed
        assert "CONTENT TO ANALYZE" in routed
        assert "User: audit this report" in routed

    def test_build_routed_prompt_includes_no_tool_framing_with_history(self, tmp_path, monkeypatch):
        """Same prefix must apply when history is present."""
        monkeypatch.setattr(
            "acp_adapter.server.Path.home", classmethod(lambda cls: tmp_path)
        )

        routed = _build_routed_prompt(
            "is this AAA?",
            [{"role": "user", "content": "previous turn"}],
        )

        assert "OPERATING CONSTRAINTS" in routed
        assert "Recent conversation:" in routed
        assert "User: is this AAA?" in routed
        # Framing must precede the conversation transcript so the model reads
        # the constraints before any prior turns.
        assert routed.index("OPERATING CONSTRAINTS") < routed.index(
            "Recent conversation:"
        )

    def test_load_routed_memory_reads_memory_md(self, tmp_path, monkeypatch):
        """MEMORY.md from ~/.hermes/memories/ must be loaded so MoA references
        see the user's persistent memory rather than claiming ignorance."""
        memories = tmp_path / ".hermes" / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text(
            "VPS: 49.12.7.18\nKey: ~/.ssh/binance_futures_tool",
            encoding="utf-8",
        )
        monkeypatch.delenv("HERMES_MOA_NO_MEMORY", raising=False)

        block = _load_routed_memory(home=tmp_path)

        assert "MEMORY (MEMORY.md)" in block
        assert "49.12.7.18" in block
        assert "binance_futures_tool" in block

    def test_load_routed_memory_reads_user_md_alongside(self, tmp_path):
        memories = tmp_path / ".hermes" / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text("memory body", encoding="utf-8")
        (memories / "USER.md").write_text("user profile body", encoding="utf-8")

        block = _load_routed_memory(home=tmp_path)

        assert "MEMORY (MEMORY.md)" in block
        assert "USER PROFILE (USER.md)" in block
        assert "memory body" in block
        assert "user profile body" in block

    def test_load_routed_memory_truncates_oversized_file(self, tmp_path):
        memories = tmp_path / ".hermes" / "memories"
        memories.mkdir(parents=True)
        big = "x" * 8000
        (memories / "MEMORY.md").write_text(big, encoding="utf-8")

        block = _load_routed_memory(max_chars=500, home=tmp_path)

        assert len(block) <= 700  # cap + header overhead
        assert "[truncated by Hermes before routing]" in block

    def test_load_routed_memory_disabled_via_env(self, tmp_path, monkeypatch):
        memories = tmp_path / ".hermes" / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text("should not appear", encoding="utf-8")
        monkeypatch.setenv("HERMES_MOA_NO_MEMORY", "1")

        block = _load_routed_memory(home=tmp_path)

        assert block == ""

    def test_load_routed_memory_returns_empty_when_dir_missing(self, tmp_path):
        # tmp_path has no .hermes/memories
        assert _load_routed_memory(home=tmp_path) == ""

    def test_build_routed_prompt_injects_memory_no_history(self, tmp_path, monkeypatch):
        """When the user has saved memory, MoA references must see it even
        on the first turn (no conversation history yet)."""
        memories = tmp_path / ".hermes" / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text(
            "VPS Brain: ssh -i ~/.ssh/binance_futures_tool root@49.12.7.18",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "acp_adapter.server.Path.home", classmethod(lambda cls: tmp_path)
        )
        monkeypatch.delenv("HERMES_MOA_NO_MEMORY", raising=False)

        routed = _build_routed_prompt("audit my VPS setup", [])

        assert "Persistent memory" in routed
        assert "49.12.7.18" in routed
        # Memory must precede the user's current request so the model reads
        # the facts before being asked the question.
        assert routed.index("Persistent memory") < routed.index(
            "User: audit my VPS setup"
        )

    def test_build_routed_prompt_injects_memory_with_history(self, tmp_path, monkeypatch):
        memories = tmp_path / ".hermes" / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text("VPS: 49.12.7.18", encoding="utf-8")
        monkeypatch.setattr(
            "acp_adapter.server.Path.home", classmethod(lambda cls: tmp_path)
        )
        monkeypatch.delenv("HERMES_MOA_NO_MEMORY", raising=False)

        routed = _build_routed_prompt(
            "follow-up question",
            [{"role": "user", "content": "earlier turn"}],
        )

        assert "Persistent memory" in routed
        assert "49.12.7.18" in routed
        # Order must be: framing -> memory -> conversation -> current request
        assert (
            routed.index("OPERATING CONSTRAINTS")
            < routed.index("Persistent memory")
            < routed.index("Recent conversation:")
            < routed.index("User: follow-up question")
        )

    def test_build_routed_prompt_omits_memory_section_when_empty(self, tmp_path, monkeypatch):
        """When no memory is available, the memory section must be omitted
        cleanly — not rendered as an empty 'Persistent memory:' header."""
        # tmp_path has no .hermes/memories at all
        monkeypatch.setattr(
            "acp_adapter.server.Path.home", classmethod(lambda cls: tmp_path)
        )

        routed = _build_routed_prompt("hello", [])

        assert "Persistent memory" not in routed
        assert "OPERATING CONSTRAINTS" in routed
        assert "User: hello" in routed

    @pytest.mark.asyncio
    async def test_prompt_returns_refusal_for_unknown_session(self, agent):
        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id="nonexistent")
        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_prompt_returns_end_turn_for_empty_message(self, agent):
        new_resp = await agent.new_session(cwd=".")
        prompt = [TextContentBlock(type="text", text="   ")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_prompt_runs_agent(self, agent):
        """The prompt method should call run_conversation on the agent."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        # Mock the agent's run_conversation
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "Hello! How can I help?",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ],
        })

        # Set up a mock connection
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_prompt_force_spar_routes_directly(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-spar"
        state.agent.run_conversation = MagicMock()
        state.history = [
            {"role": "user", "content": "We are comparing two deployment options."},
            {"role": "assistant", "content": "Option A is cheaper; option B is faster."},
        ]

        with patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(
                return_value='{"approved": true, "summary": "ok", "issues": [], "final_response": "spar answer", "disagreement": false}'
            ),
        ) as mock_spar:
            resp = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="ship this fix")],
                session_id=new_resp.session_id,
            )

        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_not_called()
        assert state.history[-1]["content"] == "spar answer"
        routed_prompt = mock_spar.await_args.kwargs["user_prompt"]
        assert "We are comparing two deployment options." in routed_prompt
        assert "Option A is cheaper; option B is faster." in routed_prompt
        assert routed_prompt.endswith("User: ship this fix")

    @pytest.mark.asyncio
    async def test_prompt_auto_routes_review_tasks_to_spar(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "auto"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(
                return_value='{"approved": true, "summary": "ok", "issues": [], "final_response": "auto spar answer", "disagreement": false}'
            ),
        ) as mock_spar:
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="review this PR and fix the regression")],
                session_id=new_resp.session_id,
            )

        mock_spar.assert_awaited_once()
        state.agent.run_conversation.assert_not_called()
        assert state.history[-1]["content"] == "auto spar answer"

    @pytest.mark.asyncio
    async def test_prompt_force_moa_routes_directly(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa"
        state.agent.run_conversation = MagicMock()
        state.history = [
            {"role": "user", "content": "We are choosing between three laptops."},
            {"role": "assistant", "content": "You care most about performance and battery."},
        ]

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value='{"success": true, "response": "moa answer", "models_used": {"reference_models": [], "aggregator_model": "xiaomi/mimo-v2.5-pro"}}'
            ),
        ) as mock_moa:
            resp = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="analyze this hard problem")],
                session_id=new_resp.session_id,
            )

        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_not_called()
        assert state.history[-1]["content"] == "moa answer"
        routed_prompt = mock_moa.await_args.kwargs["user_prompt"]
        assert mock_moa.await_args.kwargs["enable_forensic_analysis"] is False
        assert "We are choosing between three laptops." in routed_prompt
        assert "You care most about performance and battery." in routed_prompt
        assert routed_prompt.endswith("User: analyze this hard problem")

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_routes_moa_then_spar(self, agent, monkeypatch):
        # Legacy text-only verification: the agent-hands path is the new
        # default, but the underlying MoA->Spar wiring must still work
        # when an operator opts out via HERMES_MOA_TEXT_ONLY=1. This test
        # locks down that legacy contract; new tests below cover the
        # default agent-hands path.
        monkeypatch.setenv("HERMES_MOA_TEXT_ONLY", "1")
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"
        state.agent.run_conversation = MagicMock()
        state.history = [
            {"role": "user", "content": "We are choosing between two retirement plans."},
            {"role": "assistant", "content": "You care about safety first, then upside."},
        ]

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "success": True,
                        "response": "moa draft",
                        "models_used": {
                            "reference_models": [
                                "xiaomi/mimo-v2.5-pro (self-draft)",
                                "minimax/MiniMax-M2.7-highspeed",
                                "deepseek/deepseek-reasoner",
                            ],
                            "aggregator_model": "xiaomi/mimo-v2.5-pro",
                        },
                        "failed_models": [],
                        "failed_model_errors": {},
                        "reference_previews": {
                            "minimax/MiniMax-M2.7-highspeed": "minimax preview"
                        },
                        "reference_outputs": {
                            "minimax/MiniMax-M2.7-highspeed": "minimax full output"
                        },
                        "per_model_metrics": {
                            "reference_models": {},
                            "aggregator": {"model": "xiaomi/mimo-v2.5-pro", "success": True},
                            "forensic_analysis": {"skipped": True, "success": False},
                        },
                        "decision_trace": {"final_candidates": ["plan-a", "plan-b"]},
                        "aggregator_influence_log": {"influence_summary": "mimo narrowed the overlap"},
                    }
                )
            ),
        ) as mock_moa, patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "approved": True,
                        "summary": "approved",
                        "issues": [],
                        "final_response": "moa spar answer",
                        "disagreement": False,
                        "judge_verdict": {"approved": True, "summary": "judge ok"},
                    }
                )
            ),
        ) as mock_spar:
            resp = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="which plan is safer?")],
                session_id=new_resp.session_id,
            )

        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_not_called()
        assert state.history[-1]["content"] == "moa spar answer"
        routed_prompt = mock_moa.await_args.kwargs["user_prompt"]
        assert routed_prompt.endswith("User: which plan is safer?")
        assert mock_spar.await_args.kwargs["user_prompt"] == routed_prompt
        assert mock_spar.await_args.kwargs["candidate_response"] == "moa draft"
        assert mock_spar.await_args.kwargs["builder_model"] == "xiaomi/mimo-v2.5-pro"

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_runs_agent_first_then_refines(self, agent, monkeypatch):
        """Default behaviour (no HERMES_MOA_TEXT_ONLY): force-moa-spar must
        run the standard Hermes agent FIRST so tools execute with real
        side effects, then pass the agent's draft to MoA references for
        refinement, then Spar-review the refined output. This is the fix
        for the user complaint that MoA+Spar was a 'no-hands' mode."""
        monkeypatch.delenv("HERMES_MOA_TEXT_ONLY", raising=False)
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"

        # Mock the agent so it returns a 'real' draft as if it had run
        # tools. The test asserts that this draft becomes the input to
        # MoA refinement (not the raw user text).
        agent_draft = "Agent ran SSH and read /etc/os-release. VPS is Ubuntu 24.04."
        state.agent.run_conversation = MagicMock(
            return_value={
                "final_response": agent_draft,
                "messages": [
                    {"role": "user", "content": "what OS is the VPS on?"},
                    {"role": "tool", "content": "uname -a output..."},
                    {"role": "assistant", "content": agent_draft},
                ],
            }
        )

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "success": True,
                        "response": "Ubuntu 24.04 (verified by SSH).",
                        "models_used": {
                            "reference_models": ["xiaomi/mimo-v2.5-pro"],
                            "aggregator_model": "xiaomi/mimo-v2.5-pro",
                        },
                        "failed_models": [],
                        "failed_model_errors": {},
                        "reference_previews": {},
                        "reference_outputs": {},
                        "per_model_metrics": {
                            "reference_models": {},
                            "aggregator": {"model": "xiaomi/mimo-v2.5-pro", "success": True},
                            "forensic_analysis": {"skipped": True, "success": False},
                        },
                        "decision_trace": {},
                        "aggregator_influence_log": {},
                    }
                )
            ),
        ) as mock_moa, patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "approved": True,
                        "summary": "approved",
                        "issues": [],
                        "final_response": "Ubuntu 24.04 (verified by SSH).",
                        "disagreement": False,
                        "judge_verdict": {"approved": True, "summary": "judge ok"},
                    }
                )
            ),
        ) as mock_spar:
            resp = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="what OS is the VPS on?")],
                session_id=new_resp.session_id,
            )

        assert resp.stop_reason == "end_turn"
        # 1. The agent MUST have been called (Phase 1: real tool execution).
        state.agent.run_conversation.assert_called_once()
        # 2. The MoA prompt is the REFINEMENT prompt — it includes the
        #    agent's draft and ends with the improvement marker, not a
        #    raw "User: ..." line.
        moa_prompt = mock_moa.await_args.kwargs["user_prompt"]
        assert "REFINEMENT TASK" in moa_prompt
        assert agent_draft in moa_prompt
        assert moa_prompt.endswith("=== YOUR IMPROVED VERSION ===")
        # 3. Spar reviews the MoA-refined output against the same prompt.
        assert mock_spar.await_args.kwargs["user_prompt"] == moa_prompt
        assert mock_spar.await_args.kwargs["candidate_response"] == "Ubuntu 24.04 (verified by SSH)."
        # 4. The final response shown to the user is the refined version.
        assert state.history[-1]["content"] == "Ubuntu 24.04 (verified by SSH)."

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_falls_back_to_agent_draft_on_refine_failure(
        self, agent, monkeypatch
    ):
        """When the agent succeeded (real tool work was done) but MoA or
        Spar errored, the user must still see the agent's draft —
        discarding completed tool work would be far worse than losing
        the polish layer."""
        monkeypatch.delenv("HERMES_MOA_TEXT_ONLY", raising=False)
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"

        agent_draft = "I edited config.yaml and restarted the service. VPS is healthy."
        state.agent.run_conversation = MagicMock(
            return_value={
                "final_response": agent_draft,
                "messages": [
                    {"role": "user", "content": "fix the config"},
                    {"role": "assistant", "content": agent_draft},
                ],
            }
        )

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(side_effect=RuntimeError("MoA provider down")),
        ):
            resp = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="fix the config")],
                session_id=new_resp.session_id,
            )

        assert resp.stop_reason == "end_turn"
        # User sees the agent's draft, not an error — the actual work happened.
        assert state.history[-1]["content"] == agent_draft

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_skips_refinement_when_agent_errors(
        self, agent, monkeypatch
    ):
        """If the standard agent itself errors out, return that error
        directly. Don't try to MoA-refine an error message — that would
        produce a hallucinated 'success' response on top of a real failure.
        """
        monkeypatch.delenv("HERMES_MOA_TEXT_ONLY", raising=False)
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"

        state.agent.run_conversation = MagicMock(
            return_value={
                "final_response": "Error: provider rate limited",
                "messages": [{"role": "user", "content": "do something"}],
            }
        )

        moa_mock = AsyncMock()
        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool", moa_mock
        ):
            resp = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="do something")],
                session_id=new_resp.session_id,
            )

        assert resp.stop_reason == "end_turn"
        # MoA must NOT have been called — refining an error message would
        # produce a misleading 'fixed' response.
        moa_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_accepts_wrapped_tool_json(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"
        state.agent.run_conversation = MagicMock()

        moa_wrapped = (
            "Draft ready:\n```json\n"
            + json.dumps(
                {
                    "success": True,
                    "response": "wrapped moa draft",
                    "models_used": {
                        "reference_models": [
                            "xiaomi/mimo-v2.5-pro (self-draft)",
                            "minimax/MiniMax-M2.7-highspeed",
                            "deepseek/deepseek-reasoner",
                        ],
                        "aggregator_model": "xiaomi/mimo-v2.5-pro",
                    },
                    "failed_models": [],
                    "failed_model_errors": {},
                    "reference_previews": {},
                    "reference_outputs": {},
                    "per_model_metrics": {
                        "reference_models": {},
                        "aggregator": {"model": "xiaomi/mimo-v2.5-pro", "success": True},
                        "forensic_analysis": {"skipped": True, "success": False},
                    },
                    "decision_trace": {},
                    "aggregator_influence_log": {},
                },
                ensure_ascii=False,
            )
            + "\n```"
        )
        spar_wrapped = (
            "Review verdict:\n"
            + json.dumps(
                {
                    "approved": True,
                    "summary": "ok",
                    "issues": [],
                    "final_response": "wrapped final answer",
                    "disagreement": False,
                    "judge_verdict": {"approved": True, "summary": "judge ok"},
                },
                ensure_ascii=False,
            )
        )

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(return_value=moa_wrapped),
        ), patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(return_value=spar_wrapped),
        ):
            resp = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="review this long brief")],
                session_id=new_resp.session_id,
            )

        assert resp.stop_reason == "end_turn"
        assert state.history[-1]["content"] == "wrapped final answer"

    @pytest.mark.asyncio
    async def test_prompt_force_moa_can_opt_into_llm_forensics(self, agent, monkeypatch):
        monkeypatch.setenv("HERMES_MOA_FORENSIC_ANALYSIS", "1")
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value='{"success": true, "response": "moa answer", "models_used": {"reference_models": [], "aggregator_model": "xiaomi/mimo-v2.5-pro"}}'
            ),
        ) as mock_moa:
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="analyze this hard problem")],
                session_id=new_resp.session_id,
            )

        assert mock_moa.await_args.kwargs["enable_forensic_analysis"] is True

    @pytest.mark.asyncio
    async def test_prompt_force_moa_emits_thought_chunk(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa"
        state.agent.run_conversation = MagicMock()

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value='{"success": true, "response": "moa answer", "models_used": {"reference_models": [], "aggregator_model": "xiaomi/mimo-v2.5-pro"}}'
            ),
        ):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="analyze this hard problem")],
                session_id=new_resp.session_id,
            )

        update_types = [
            getattr((call.kwargs.get("update") or call.args[1]), "session_update", None)
            for call in mock_conn.session_update.await_args_list
        ]
        assert "agent_thought_chunk" in update_types

    @pytest.mark.asyncio
    async def test_prompt_force_moa_writes_forensics_log(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value='{"success": true, "response": "moa answer", "models_used": {"reference_models": ["minimax/MiniMax-M2.7-highspeed", "deepseek/deepseek-reasoner"], "aggregator_model": "xiaomi/mimo-v2.5-pro"}, "failed_models": [], "failed_model_errors": {}, "reference_previews": {"minimax/MiniMax-M2.7-highspeed": "moa answer"}, "reference_outputs": {"minimax/MiniMax-M2.7-highspeed": "full minimax output"}, "per_model_metrics": {"reference_models": {"minimax/MiniMax-M2.7-highspeed": {"attempts": 1, "latency_seconds": 1.2, "success": true}}, "aggregator": {"model": "xiaomi/mimo-v2.5-pro", "latency_seconds": 2.4, "success": true}, "forensic_analysis": {"model": "xiaomi/mimo-v2.5-pro", "latency_seconds": 0.7, "success": true}}, "decision_trace": {"model_proposals": {"minimax/MiniMax-M2.7-highspeed": ["akg", "lithium"]}, "overlap": ["akg"], "conflicts": ["deepseek preferred another stack"], "final_candidates": ["akg", "lithium"], "synthesis_summary": "mimo narrowed to the strongest overlap"}, "aggregator_influence_log": {"kept_from_models": {"minimax/MiniMax-M2.7-highspeed": ["akg"]}, "discarded_or_deprioritized": ["metformin"], "resolution_notes": ["mimo preferred overlap plus safety"], "influence_summary": "mimo used both references but weighted overlap most"}}'
            ),
        ):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="analyze this hard problem")],
                session_id=new_resp.session_id,
            )

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "route_result")
        assert result_event["session_id"] == new_resp.session_id
        assert result_event["route"] == "force-moa"
        assert result_event["tool"]["models_used"]["aggregator_model"] == "xiaomi/mimo-v2.5-pro"
        assert result_event["tool"]["models_used"]["reference_models"] == [
            "minimax/MiniMax-M2.7-highspeed",
            "deepseek/deepseek-reasoner",
        ]
        assert result_event["tool"]["failed_models"] == []
        assert result_event["tool"]["failed_model_errors"] == {}
        assert result_event["tool"]["reference_previews"] == {
            "minimax/MiniMax-M2.7-highspeed": "moa answer"
        }
        assert "reference_outputs" not in result_event["tool"]
        assert result_event["tool"]["reference_output_hashes"] == {
            "minimax/MiniMax-M2.7-highspeed": {"sha256_16": "2e718f248d8ca032", "chars": 19}
        }
        assert result_event["tool"]["per_model_metrics"]["aggregator"]["model"] == "xiaomi/mimo-v2.5-pro"
        assert result_event["tool"]["decision_trace"]["final_candidates"] == ["akg", "lithium"]
        assert result_event["tool"]["aggregator_influence_log"]["influence_summary"] == "mimo used both references but weighted overlap most"

    @pytest.mark.asyncio
    async def test_prompt_force_moa_can_opt_into_full_raw_forensics(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("HERMES_MOA_FULL_FORENSICS", "1")
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value='{"success": true, "response": "moa answer", "models_used": {"reference_models": ["minimax/MiniMax-M2.7-highspeed"], "aggregator_model": "xiaomi/mimo-v2.5-pro"}, "reference_outputs": {"minimax/MiniMax-M2.7-highspeed": "full minimax output"}}'
            ),
        ):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="analyze this hard problem")],
                session_id=new_resp.session_id,
            )

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "route_result")
        assert result_event["tool"]["reference_outputs"] == {
            "minimax/MiniMax-M2.7-highspeed": "full minimax output"
        }
        assert "reference_output_hashes" not in result_event["tool"]

    @pytest.mark.asyncio
    async def test_router_prompt_message_id_becomes_route_turn_id(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa"
        state.agent.run_conversation = MagicMock()
        router = build_agent_router(agent)

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value='{"success": true, "response": "moa answer", "models_used": {"reference_models": [], "aggregator_model": "xiaomi/mimo-v2.5-pro"}}'
            ),
        ):
            await router(
                "session/prompt",
                {
                    "sessionId": new_resp.session_id,
                    "messageId": "router-turn-1",
                    "prompt": [{"type": "text", "text": "analyze this"}],
                },
                False,
            )

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        assert {event["route_turn_id"] for event in events} == {"router-turn-1"}

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_writes_combined_forensics_log(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "success": True,
                        "response": "moa draft",
                        "models_used": {
                            "reference_models": [
                                "xiaomi/mimo-v2.5-pro (self-draft)",
                                "minimax/MiniMax-M2.7-highspeed",
                            ],
                            "aggregator_model": "xiaomi/mimo-v2.5-pro",
                        },
                        "failed_models": [],
                        "failed_model_errors": {},
                        "reference_previews": {
                            "minimax/MiniMax-M2.7-highspeed": "moa answer"
                        },
                        "reference_outputs": {
                            "minimax/MiniMax-M2.7-highspeed": "full minimax output"
                        },
                        "per_model_metrics": {
                            "reference_models": {"minimax/MiniMax-M2.7-highspeed": {"attempts": 1, "latency_seconds": 1.2, "success": True}},
                            "aggregator": {"model": "xiaomi/mimo-v2.5-pro", "latency_seconds": 2.4, "success": True},
                            "forensic_analysis": {"model": "xiaomi/mimo-v2.5-pro", "latency_seconds": 0.0, "success": False, "skipped": True},
                        },
                        "decision_trace": {"final_candidates": ["akg", "lithium"]},
                        "aggregator_influence_log": {"influence_summary": "mimo used overlap"},
                    }
                )
            ),
        ), patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "approved": True,
                        "summary": "ok",
                        "issues": [],
                        "final_response": "moa spar answer",
                        "disagreement": False,
                        "judge_verdict": {"approved": True, "summary": "judge ok"},
                    }
                )
            ),
        ):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="analyze this hard problem")],
                session_id=new_resp.session_id,
            )

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "route_result")
        assert result_event["route"] == "force-moa-spar"
        assert result_event["tool"]["success"] is True
        assert result_event["tool"]["approved"] is True
        assert result_event["tool"]["gate_passed"] is True
        assert result_event["tool"]["judge_verdict"] == {"approved": True, "summary": "judge ok"}
        assert result_event["tool"]["models_used"]["aggregator_model"] == "xiaomi/mimo-v2.5-pro"
        assert result_event["tool"]["reference_output_hashes"] == {
            "minimax/MiniMax-M2.7-highspeed": {"sha256_16": "2e718f248d8ca032", "chars": 19}
        }
        assert result_event["tool"]["moa_candidate_response"] == "moa draft"

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_rejected_review_blocks_gate(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "success": True,
                        "response": "moa draft",
                        "models_used": {"reference_models": [], "aggregator_model": "xiaomi/mimo-v2.5-pro"},
                    }
                )
            ),
        ), patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "approved": False,
                        "summary": "missing evidence",
                        "issues": ["Needs citations"],
                        "final_response": "moa draft",
                        "disagreement": False,
                    }
                )
            ),
        ):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="review this long brief")],
                session_id=new_resp.session_id,
                message_id="turn-123",
            )

        assert state.history[-1]["route_turn_id"] == "turn-123"
        assert state.history[-1]["content"].startswith("Spar review rejected")
        assert "Latest draft (not approved):" in state.history[-1]["content"]
        stored_messages = agent.session_manager._get_db().get_messages(new_resp.session_id)
        assert stored_messages[-1]["tool_call_id"] == "route:turn-123"
        restored_history = agent.session_manager._get_db().get_messages_as_conversation(new_resp.session_id)
        assert restored_history[-1]["route_turn_id"] == "turn-123"
        assert "tool_call_id" not in restored_history[-1]

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "route_result")
        assert result_event["route_turn_id"] == "turn-123"
        assert result_event["tool"]["success"] is False
        assert result_event["tool"]["approved"] is False
        assert result_event["tool"]["gate_passed"] is False

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_preserves_moa_failure_without_fake_spar_success(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "success": False,
                        "response": "MoA processing failed. Please try again or use a single model for this query.",
                        "models_used": {
                            "reference_models": [
                                "xiaomi/mimo-v2.5-pro (self-draft)",
                                "minimax/MiniMax-M2.7-highspeed",
                            ],
                            "aggregator_model": "xiaomi/mimo-v2.5-pro",
                        },
                        "failed_models": ["deepseek/deepseek-reasoner"],
                        "failed_model_errors": {"deepseek/deepseek-reasoner": "timeout"},
                        "reference_previews": {},
                        "reference_outputs": {},
                        "per_model_metrics": {},
                        "decision_trace": {},
                        "aggregator_influence_log": {},
                        "error": "Error in MoA processing: timeout",
                    }
                )
            ),
        ) as mock_moa, patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(),
        ) as mock_spar:
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="review this long brief")],
                session_id=new_resp.session_id,
            )

        mock_moa.assert_awaited_once()
        mock_spar.assert_not_awaited()

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "route_result")
        assert result_event["route"] == "force-moa-spar"
        assert result_event["tool"]["success"] is False
        assert result_event["tool"]["pipeline_stage"] == "moa"
        assert "approved" not in result_event["tool"]
        assert result_event["tool"]["error"] == "Error in MoA processing: timeout"
        assert result_event["tool"]["moa_failure_preview"].startswith("MoA processing failed.")
        assert "\"success\": false" in result_event["tool"]["raw_output_preview"]
        assert result_event["tool"]["models_used"]["aggregator_model"] == "xiaomi/mimo-v2.5-pro"

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_logs_preview_for_thin_moa_failure_payload(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(return_value=json.dumps({"success": False})),
        ) as mock_moa, patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(),
        ) as mock_spar:
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="review this long brief")],
                session_id=new_resp.session_id,
            )

        mock_moa.assert_awaited_once()
        mock_spar.assert_not_awaited()

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "route_result")
        assert result_event["tool"]["success"] is False
        assert result_event["tool"]["pipeline_stage"] == "moa"
        assert result_event["tool"]["raw_output_preview"] == "{\"success\": false}"
        assert "moa_failure_preview" not in result_event["tool"]

    @pytest.mark.asyncio
    async def test_prompt_force_moa_spar_falls_back_to_moa_answer_when_spar_times_out(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-moa-spar"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.mixture_of_agents_tool.mixture_of_agents_tool",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "success": True,
                        "response": "moa draft",
                        "models_used": {
                            "reference_models": [
                                "xiaomi/mimo-v2.5-pro (self-draft)",
                                "minimax/MiniMax-M2.7-highspeed",
                            ],
                            "aggregator_model": "xiaomi/mimo-v2.5-pro",
                        },
                        "failed_models": [],
                        "failed_model_errors": {},
                        "reference_previews": {},
                        "reference_outputs": {},
                        "per_model_metrics": {},
                        "decision_trace": {},
                        "aggregator_influence_log": {},
                    }
                )
            ),
        ) as mock_moa, patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(side_effect=TimeoutError("Request timed out.")),
        ) as mock_spar:
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="review this long brief")],
                session_id=new_resp.session_id,
            )

        mock_moa.assert_awaited_once()
        mock_spar.assert_awaited_once()
        assert state.history[-1]["content"].startswith("MoA + Spar did not return an approved answer.")
        assert "Spar review failed: Request timed out." in state.history[-1]["content"]
        assert "Latest MoA draft (not approved):" in state.history[-1]["content"]

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "route_result")
        assert result_event["route"] == "force-moa-spar"
        assert result_event["tool"]["success"] is False
        assert result_event["tool"]["pipeline_stage"] == "spar"
        assert result_event["tool"]["review_error"] == "Request timed out."
        assert "approved" not in result_event["tool"]

    @pytest.mark.asyncio
    async def test_prompt_force_spar_forensics_marks_completed_review_successfully(self, agent, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.mode = "force-spar"
        state.agent.run_conversation = MagicMock()

        with patch(
            "tools.spar_tool.spar_tool",
            AsyncMock(
                return_value='{"approved": true, "summary": "ok", "issues": [], "final_response": "spar answer", "disagreement": false, "judge_verdict": {"approved": true, "summary": "judge ok"}}'
            ),
        ):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="review this answer")],
                session_id=new_resp.session_id,
            )

        forensic_path = tmp_path / ".hermes" / "logs" / "route_forensics.jsonl"
        events = [json.loads(line) for line in forensic_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "route_result")
        assert result_event["session_id"] == new_resp.session_id
        assert result_event["route"] == "force-spar"
        assert result_event["tool"]["success"] is True
        assert result_event["tool"]["approved"] is True
        assert result_event["tool"]["disagreement"] is False
        assert result_event["tool"]["judge_verdict"] == {"approved": True, "summary": "judge ok"}

    @pytest.mark.asyncio
    async def test_prompt_updates_history(self, agent):
        """After a prompt, session history should be updated."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        expected_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "hey",
            "messages": expected_history,
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hi")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert state.history == expected_history

    @pytest.mark.asyncio
    async def test_prompt_sends_final_message_update(self, agent):
        """The final response should be sent as an AgentMessageChunk."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "I can help with that!",
            "messages": [],
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="help me")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        # session_update should have been called with the final message
        mock_conn.session_update.assert_called()
        # Get the last call's update argument
        last_call = mock_conn.session_update.call_args_list[-1]
        update = last_call[1].get("update") or last_call[0][1]
        assert update.session_update == "agent_message_chunk"

    @pytest.mark.asyncio
    async def test_prompt_auto_titles_session(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "Here is the fix.",
            "messages": [
                {"role": "user", "content": "fix the broken ACP history"},
                {"role": "assistant", "content": "Here is the fix."},
            ],
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("agent.title_generator.maybe_auto_title") as mock_title:
            prompt = [TextContentBlock(type="text", text="fix the broken ACP history")]
            await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        mock_title.assert_called_once()
        assert mock_title.call_args.args[1] == new_resp.session_id
        assert mock_title.call_args.args[2] == "fix the broken ACP history"
        assert mock_title.call_args.args[3] == "Here is the fix."

    @pytest.mark.asyncio
    async def test_prompt_populates_usage_from_top_level_run_conversation_fields(self, agent):
        """ACP should map top-level token fields into PromptResponse.usage."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "usage attached",
            "messages": [],
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
            "reasoning_tokens": 7,
            "cache_read_tokens": 11,
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="show usage")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.usage is not None
        assert resp.usage.input_tokens == 123
        assert resp.usage.output_tokens == 45
        assert resp.usage.total_tokens == 168
        assert resp.usage.thought_tokens == 7
        assert resp.usage.cached_read_tokens == 11

    @pytest.mark.asyncio
    async def test_prompt_cancelled_returns_cancelled_stop_reason(self, agent):
        """If cancel is called during prompt, stop_reason should be 'cancelled'."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            # Simulate cancel being set during execution
            state.cancel_event.set()
            return {"final_response": "interrupted", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="do something")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "cancelled"


# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_stores_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        agent.on_connect(mock_conn)
        assert agent._conn is mock_conn


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command dispatch in the ACP adapter."""

    def _make_state(self, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.model = "test-model"
        return state

    def test_help_lists_commands(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/help", state)
        assert result is not None
        assert "/help" in result
        assert "/model" in result
        assert "/tools" in result
        assert "/reset" in result

    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result

    def test_context_empty(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = []
        result = agent._handle_slash_command("/context", state)
        assert "empty" in result.lower()

    def test_context_with_messages(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = agent._handle_slash_command("/context", state)
        assert "2 messages" in result
        assert "user: 1" in result

    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        result = agent._handle_slash_command("/reset", state)
        assert "cleared" in result.lower()
        assert len(state.history) == 0

    def test_version(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/version", state)
        assert HERMES_VERSION in result

    def test_compact_compresses_context(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        original_session_db = object()
        state.agent._session_db = original_session_db

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id):
            assert state.agent._session_db is None
            assert messages == state.history
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch(
                "agent.model_metadata.estimate_messages_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compact", state)

        assert "Context compressed: 4 -> 1 messages" in result
        assert "~40 -> ~12 tokens" in result
        assert state.history == [{"role": "user", "content": "summary"}]
        assert state.agent._session_db is original_session_db
        state.agent._compress_context.assert_called_once_with(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "system",
            approx_tokens=40,
            task_id=state.session_id,
        )
        mock_save.assert_called_once_with(state.session_id)

    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None

    @pytest.mark.asyncio
    async def test_slash_command_intercepted_in_prompt(self, agent, mock_manager):
        """Slash commands should be handled without calling the LLM."""
        new_resp = await agent.new_session(cwd="/tmp")
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="/help")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        mock_conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_slash_falls_through_to_llm(self, agent, mock_manager):
        """Unknown /commands should be sent to the LLM, not intercepted."""
        new_resp = await agent.new_session(cwd="/tmp")
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        mock_conn.request_permission = AsyncMock(return_value=None)
        agent._conn = mock_conn

        # Mock run_in_executor to avoid actually running the agent
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value={
                "final_response": "I processed /foo",
                "messages": [],
            })
            prompt = [TextContentBlock(type="text", text="/foo bar")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"

    def test_model_switch_uses_requested_provider(self, tmp_path, monkeypatch):
        """`/model provider:model` should rebuild the ACP agent on that provider."""
        runtime_calls = []

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            runtime_calls.append(requested)
            provider = requested or "openrouter"
            return {
                "provider": provider,
                "api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "openrouter/gpt-5"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        manager = SessionManager(db=SessionDB(tmp_path / "state.db"))

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            acp_agent = HermesACPAgent(session_manager=manager)
            state = manager.create_session(cwd="/tmp")
            result = acp_agent._cmd_model("anthropic:claude-sonnet-4-6", state)

        assert "Provider: anthropic" in result
        assert state.agent.provider == "anthropic"
        assert state.agent.base_url == "https://anthropic.example/v1"
        assert runtime_calls[-1] == "anthropic"


# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""

    @pytest.mark.asyncio
    async def test_noop_when_no_servers(self, agent, mock_manager):
        """No-op when mcp_servers is None or empty."""
        state = mock_manager.create_session(cwd="/tmp")
        # Should not raise
        await agent._register_session_mcp_servers(state, None)
        await agent._register_session_mcp_servers(state, [])

    @pytest.mark.asyncio
    async def test_registers_stdio_servers(self, agent, mock_manager):
        """McpServerStdio servers are converted and passed to register_mcp_servers."""
        from acp.schema import McpServerStdio, EnvVariable

        state = mock_manager.create_session(cwd="/tmp")
        # Give the mock agent the attributes _register_session_mcp_servers reads
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerStdio(
            name="test-server",
            command="/usr/bin/test",
            args=["--flag"],
            env=[EnvVariable(name="KEY", value="val")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}

    @pytest.mark.asyncio
    async def test_registers_http_servers(self, agent, mock_manager):
        """McpServerHttp servers are converted correctly."""
        from acp.schema import McpServerHttp, HttpHeader

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerHttp(
            name="http-server",
            url="https://api.example.com/mcp",
            headers=[HttpHeader(name="Authorization", value="Bearer tok")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return []

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "http-server" in registered_config
        cfg = registered_config["http-server"]
        assert cfg["url"] == "https://api.example.com/mcp"
        assert cfg["headers"] == {"Authorization": "Bearer tok"}

    @pytest.mark.asyncio
    async def test_refreshes_agent_tool_surface(self, agent, mock_manager):
        """After MCP registration, agent.tools and valid_tool_names are refreshed."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent._cached_system_prompt = "old prompt"

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools):
            await agent._register_session_mcp_servers(state, [server])

        assert state.agent.tools == fake_tools
        assert state.agent.valid_tool_names == {"mcp_srv_search", "terminal"}
        # _invalidate_system_prompt should have been called
        state.agent._invalidate_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_failure_logs_warning(self, agent, mock_manager):
        """If register_mcp_servers raises, warning is logged but no crash."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        server = McpServerStdio(
            name="bad",
            command="/nonexistent",
            args=[],
            env=[],
        )

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=RuntimeError("boom")):
            # Should not raise
            await agent._register_session_mcp_servers(state, [server])
