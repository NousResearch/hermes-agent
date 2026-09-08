"""MCP security and hybrid registry — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import asyncio
import logging
import sys
import types
from unittest.mock import MagicMock

import pytest

from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
)
from tests.agent.claude_sdk_fakes import (
    TextBlock,
    AssistantMessage,
    ResultMessage,
    _FakeClient,
    _make_session,
    _plant_claude_agent_sdk_stand_in,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


# ---------- direct HTTP MCP security -------------------------------------


class TestHttpMcpSecurity:
    @pytest.fixture(autouse=True)
    def _enable_direct_http_opt_in(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session_config as mod

        monkeypatch.setattr(
            mod,
            "_provider_config",
            lambda: {"hybrid_mcp_bridge": True},
        )

    def test_helper_is_default_off_even_when_called_directly(
        self, monkeypatch, tmp_path
    ):
        from agent.transports import claude_agent_sdk_session_config as mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "mcp_servers:\n  public:\n    url: https://mcp.example.test\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "_provider_config", lambda: {})
        monkeypatch.setattr(
            mod,
            "_configured_hybrid_exclude",
            MagicMock(side_effect=AssertionError("must not read exclusions")),
        )
        assert mod._http_mcp_entries_from_config() == {}

    def test_default_off_never_discovers_direct_http_servers(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session as mod

        discover = MagicMock(return_value={
            "remote": {"type": "http", "url": "https://mcp.example.test"}
        })
        monkeypatch.setattr(mod, "_http_mcp_entries_from_config", discover)
        session, _ = _make_session(script=[ResultMessage(result="ok")])

        fields = session.build_option_fields()

        discover.assert_not_called()
        assert "remote" not in fields["mcp_servers"]

    def test_header_bearing_server_is_refused_without_secret_in_logs(
        self, monkeypatch, tmp_path, caplog
    ):
        from agent.transports.claude_agent_sdk_session_config import _http_mcp_entries_from_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PROFILE", "test")
        monkeypatch.setenv("PRIVATE_MCP_TOKEN", "super-secret-token")
        (tmp_path / "config.yaml").write_text(
            """mcp_servers:
  private-search:
    url: https://mcp.example.test
    headers:
      Authorization: Bearer ${PRIVATE_MCP_TOKEN}
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            entries = _http_mcp_entries_from_config()

        assert entries == {}
        assert "private-search" in caplog.text
        assert "super-secret-token" not in caplog.text
        assert "Authorization" not in caplog.text

    def test_headerless_server_is_safe_to_register(self, monkeypatch, tmp_path):
        from agent.transports.claude_agent_sdk_session_config import _http_mcp_entries_from_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PROFILE", "test")
        (tmp_path / "config.yaml").write_text(
            """mcp_servers:
  public-search:
    url: https://mcp.example.test
""",
            encoding="utf-8",
        )

        assert _http_mcp_entries_from_config() == {
            "public-search": {
                "type": "http",
                "url": "https://mcp.example.test",
            }
        }

    def test_malformed_resolved_url_is_refused_without_url_in_logs(
        self, monkeypatch, tmp_path, caplog
    ):
        from agent.transports.claude_agent_sdk_session_config import _http_mcp_entries_from_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PROFILE", "test")
        monkeypatch.setenv("MISSING_MCP_HOST", "expanded-secret-host")
        (tmp_path / "config.yaml").write_text(
            """mcp_servers:
  broken-search:
    url: https://${MISSING_MCP_HOST}/mcp
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            entries = _http_mcp_entries_from_config()

        assert entries == {}
        assert "broken-search" in caplog.text
        assert "https:///mcp" not in caplog.text
        assert "expanded-secret-host" not in caplog.text

    @pytest.mark.parametrize(
        "url",
        [
            "https://mcp.example.test/${lowercase_secret}/mcp",
            "https://user:password@mcp.example.test/mcp",
            "https://mcp.example.test/mcp?api-key=value",
            "https://mcp.example.test/mcp?apikey=value",
            "https://mcp.example.test/mcp?token=value",
            "https://mcp.example.test/mcp?%74oken=value",
            "https://mcp.example.test/mcp?%2574oken=value",
            "https://mcp.example.test/mcp?signature=value",
            "https://mcp.example.test/mcp#authorization=value",
            "https://mcp.example.test/mcp#%61uth=value",
        ],
    )
    def test_templated_or_credential_bearing_url_is_refused(
        self, monkeypatch, tmp_path, url
    ):
        from agent.transports.claude_agent_sdk_session_config import _http_mcp_entries_from_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            f"mcp_servers:\n  private:\n    url: {url}\n",
            encoding="utf-8",
        )
        assert _http_mcp_entries_from_config() == {}

    def test_truthy_non_mapping_headers_are_refused(self, monkeypatch, tmp_path):
        from agent.transports.claude_agent_sdk_session_config import _http_mcp_entries_from_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "mcp_servers:\n  malformed:\n    url: https://mcp.example.test\n"
            "    headers: bearer-secret\n",
            encoding="utf-8",
        )
        assert _http_mcp_entries_from_config() == {}

    def test_http_server_name_honors_exclusion(self, monkeypatch, tmp_path):
        from agent.transports import claude_agent_sdk_session_config as mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            mod,
            "_provider_config",
            lambda: {
                "hybrid_mcp_bridge": True,
                "hybrid_mcp_bridge_exclude": ["public-search"],
            },
        )
        (tmp_path / "config.yaml").write_text(
            "mcp_servers:\n  public-search:\n"
            "    url: https://mcp.example.test\n",
            encoding="utf-8",
        )
        assert mod._http_mcp_entries_from_config() == {}

    def test_hermes_tools_collision_preserves_hybrid_entry(
        self, monkeypatch, tmp_path
    ):
        from agent.transports import hermes_hybrid_mcp

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "mcp_servers:\n"
            "  hermes-tools:\n    url: https://mcp.example.test/collision\n"
            "  public:\n    url: https://mcp.example.test/public\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_hybrid_mcp,
            "build_hybrid_mcp_server",
            lambda *args, server_name, **kwargs: {
                "type": "sdk",
                "name": server_name,
            },
        )
        session = ClaudeAgentSdkSession(
            cwd="/tmp",
            agent=object(),
            tools={"read_file": object()},
        )
        fields = session.build_option_fields()
        assert fields["mcp_servers"]["hermes-tools"]["type"] == "sdk"
        assert fields["mcp_servers"]["public"] == {
            "type": "http",
            "url": "https://mcp.example.test/public",
        }


class TestHybridRegistryDiff:
    """The bridge is built from ``agent.tools``; servers that register after
    that snapshot are in no bucket at all. These pin the diff that closes it."""

    @staticmethod
    def _spec(name):
        return {
            "type": "function",
            "function": {"name": name, "description": name, "parameters": {}},
        }

    @pytest.fixture
    def captured(self, monkeypatch):
        from agent.transports import hermes_hybrid_mcp

        seen = {}

        def _build(agent, tools, *, server_name, **kwargs):
            seen[server_name] = list(tools)
            return {"type": "sdk", "name": server_name}

        monkeypatch.setattr(
            hermes_hybrid_mcp, "build_hybrid_mcp_server", _build
        )
        return seen

    @staticmethod
    def _session(**kwargs):
        return ClaudeAgentSdkSession(cwd="/tmp", **kwargs)

    def _patch_registry(self, monkeypatch, specs, *, recorder=None):
        from agent.transports import claude_agent_sdk_session_config as mod

        monkeypatch.setattr(
            mod, "_provider_config", lambda: {"hybrid_mcp_bridge": True}
        )
        fake_mcp = types.SimpleNamespace(has_registered_mcp_tools=lambda: True)
        monkeypatch.setitem(sys.modules, "tools.mcp_tool_discovery", fake_mcp)

        def _defs(**kwargs):
            if recorder is not None:
                recorder.update(kwargs)
            return specs

        monkeypatch.setitem(
            sys.modules, "model_tools",
            types.SimpleNamespace(get_tool_definitions=_defs),
        )

    def test_late_registered_mcp_tool_reaches_the_extras_bucket(
        self, monkeypatch, captured
    ):
        self._patch_registry(monkeypatch, [self._spec("mcp__late__do")])
        session = self._session(
            agent=object(), tools=[self._spec("read_file")]
        )

        session.build_option_fields()

        hybrid = [
            (t.get("function") or {}).get("name") for t in captured["hermes-hybrid"]
        ]
        assert "mcp__late__do" in hybrid
        # Legacy bucket is untouched: its grants must keep matching.
        legacy = [
            (t.get("function") or {}).get("name") for t in captured["hermes-tools"]
        ]
        assert legacy == ["read_file"]

    def test_tool_already_in_snapshot_is_not_registered_twice(
        self, monkeypatch, captured
    ):
        self._patch_registry(monkeypatch, [self._spec("mcp__dup__do")])
        session = self._session(
            agent=object(), tools=[self._spec("mcp__dup__do")]
        )

        session.build_option_fields()

        names = [
            (t.get("function") or {}).get("name") for t in captured["hermes-hybrid"]
        ]
        assert names.count("mcp__dup__do") == 1

    def test_non_mcp_registry_entries_are_ignored(self, monkeypatch, captured):
        self._patch_registry(monkeypatch, [self._spec("native_tool")])
        session = self._session(
            agent=object(), tools=[self._spec("read_file")]
        )

        session.build_option_fields()

        names = [
            (t.get("function") or {}).get("name") for t in captured["hermes-hybrid"]
        ]
        assert "native_tool" not in names

    def test_agent_toolset_gating_is_forwarded(self, monkeypatch, captured):
        recorder = {}
        self._patch_registry(monkeypatch, [], recorder=recorder)
        agent = types.SimpleNamespace(
            enabled_toolsets=["mcp-allowed"], disabled_toolsets=["mcp-banned"]
        )
        session = self._session(agent=agent, tools=[self._spec("read_file")])

        session.build_option_fields()

        assert recorder["enabled_toolsets"] == ["mcp-allowed"]
        assert recorder["disabled_toolsets"] == ["mcp-banned"]

    def test_registry_failure_leaves_the_bridge_unchanged(
        self, monkeypatch, captured
    ):
        from agent.transports import claude_agent_sdk_session_config as mod

        monkeypatch.setattr(
            mod, "_provider_config", lambda: {"hybrid_mcp_bridge": True}
        )
        boom = types.SimpleNamespace(
            has_registered_mcp_tools=MagicMock(side_effect=RuntimeError("boom"))
        )
        monkeypatch.setitem(sys.modules, "tools.mcp_tool_discovery", boom)
        session = self._session(
            agent=object(), tools=[self._spec("read_file")]
        )

        fields = session.build_option_fields()

        assert fields["mcp_servers"]["hermes-hybrid"]["type"] == "sdk"
        names = [
            (t.get("function") or {}).get("name") for t in captured["hermes-hybrid"]
        ]
        assert names == ["read_file"]

    def test_skip_mcp_refresh_opt_out_is_honoured(self, monkeypatch, captured):
        """Background review sets this flag on purpose; the registry read must
        not walk past it just because it bypasses the snapshot."""
        self._patch_registry(monkeypatch, [self._spec("mcp__late__do")])
        agent = types.SimpleNamespace(_skip_mcp_refresh=True)
        session = self._session(agent=agent, tools=[self._spec("read_file")])

        session.build_option_fields()

        names = [
            (t.get("function") or {}).get("name") for t in captured["hermes-hybrid"]
        ]
        assert "mcp__late__do" not in names

    def test_no_registry_read_when_the_bridge_is_not_built(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session as mod

        probe = MagicMock(side_effect=AssertionError("must not read registry"))
        monkeypatch.setattr(mod, "_mcp_registry_tool_specs", probe)
        session, _ = _make_session(script=[ResultMessage(result="ok")])

        session.build_option_fields()

        probe.assert_not_called()


# ---------- hermes session id plumbing to the MCP shims (#26567) ----------


class TestMcpEnvMinimal:
    def test_mcp_env_carries_no_secrets(self, monkeypatch):
        # Validator C4 (HIGH): the SDK inlines the MCP config -- env
        # included -- into the claude CLI argv, world-readable via ps. The
        # env must be a minimal allowlist, never the credentialed environ.
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-fake")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
        # (ANTHROPIC_AUTH_TOKEN deliberately NOT set here — the C5 fail-closed
        # guard would refuse startup before the MCP config is even built,
        # which is its own test below. The allowlist excludes it regardless.)
        monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-test-home")
        session, holder = _make_session(
            script=[ResultMessage(result="ok")], hermes_session_id="sess-9"
        )
        try:
            session.run_turn("ping")
        finally:
            session.close()
        env = holder["client"].options["mcp_servers"]["hermes-tools"]["env"]
        for secret in ("CLAUDE_CODE_OAUTH_TOKEN", "OPENROUTER_API_KEY",
                       "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
            assert secret not in env, f"{secret} leaked into the MCP argv env"
        assert "PYTHONPATH" in env
        assert env["HERMES_SESSION_ID"] == "sess-9"
        assert env["HERMES_HOME"] == "/tmp/hermes-test-home"

    def test_state_db_override_rides_the_mcp_env(self, monkeypatch):
        # Validator N1 (round 3): the C4 allowlist dropped HERMES_MCP_STATE_DB,
        # silently killing the shims' documented state-DB override — the MCP
        # subprocess searched the DEFAULT DB with no error. A path, not a
        # secret, so it belongs on the allowlist.
        monkeypatch.setenv("HERMES_MCP_STATE_DB", "/tmp/custom-state.db")
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        env = holder["client"].options["mcp_servers"]["hermes-tools"]["env"]
        assert env["HERMES_MCP_STATE_DB"] == "/tmp/custom-state.db"

    def test_anthropic_auth_token_refuses_startup(self, monkeypatch):
        # Validator C5: the CLI also honors ANTHROPIC_AUTH_TOKEN (bearer,
        # typically metered/proxy) — same fail-closed class as the API key.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "fake-bearer")
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert "ANTHROPIC_AUTH_TOKEN" in (turn.error or "")

    def test_allow_metered_key_via_config_yaml(self, monkeypatch):
        # The explicit override is a config.yaml key (AGENTS.md: behavioral
        # settings live in config, not env); the guard steps aside and the
        # fake-backed session starts normally.
        import hermes_cli.config as cfg

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"allow_metered_key": True}}
            },
        )
        session, _holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            turn = session.run_turn("ping")
        finally:
            session.close()
        assert not turn.should_retire
        assert turn.error is None

    def test_half_connected_client_is_reaped_on_close(self):
        # Validator C6: on a connect failure the client was assigned only
        # AFTER connect() returned, so close() skipped disconnect and the
        # CLI subprocess was orphaned.
        session, holder = _make_session(connect_exc=RuntimeError("connect blew up"))
        turn = session.run_turn("hi")
        assert turn.should_retire
        session.close()
        assert holder["client"].disconnected is True

    def test_mid_stream_interrupt_breaks_and_discards_tail(self):
        # Validator HIGH test-gap: the /stop-arriving-DURING-streaming path
        # was never exercised at session level.
        holder = {}

        class MidStreamClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                self._pending.append(
                    AssistantMessage(content=[TextBlock("first chunk")])
                )
                holder["session"]._interrupt_event.set()
                self._pending.append(
                    AssistantMessage(content=[TextBlock("tail that must be discarded")])
                )
                self._pending.append(ResultMessage(result="tail that must be discarded"))

        def factory(options=None):
            client = MidStreamClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.interrupted is True
        assert all("discarded" not in str(m.get("content")) for m in turn.projected_messages)


class TestSdkBoundedMcpInspectionPermissions:
    """Only fixed Hermes MCP file inspection tools bypass SDK prompting."""

    @pytest.fixture(autouse=True)
    def _sdk_permission_results(self, monkeypatch):
        _plant_claude_agent_sdk_stand_in(monkeypatch)

    @pytest.mark.parametrize("tool_name", [
        "mcp__hermes-tools__read_file",
        "mcp__hermes-tools__search_files",
    ])
    def test_bounded_inspection_mcp_tools_are_auto_allowed(self, tool_name):
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(tool_name, {}, None))
        assert type(result).__name__ == "PermissionResultAllow"
        assert calls == []

    def test_auto_allowed_mcp_tools_match_claude_profile_readers_exactly(self):
        from agent.transports.claude_agent_sdk_session import (
            _SDK_AUTO_ALLOWED_MCP_TOOLS,
        )
        from agent.transports.hermes_tool_exposure import (
            CLAUDE_AGENT_SDK_INSPECTION_TOOLS,
            exposed_tools_for_profile,
        )

        expected = {
            "mcp__hermes-tools__read_file",
            "mcp__hermes-tools__search_files",
        }
        exposed = {
            f"mcp__hermes-tools__{name}"
            for name in exposed_tools_for_profile("claude-agent-sdk")
        }
        profile_inspection = {
            f"mcp__hermes-tools__{name}"
            for name in CLAUDE_AGENT_SDK_INSPECTION_TOOLS
        }
        assert _SDK_AUTO_ALLOWED_MCP_TOOLS == expected
        assert profile_inspection == expected
        assert _SDK_AUTO_ALLOWED_MCP_TOOLS <= exposed

    @pytest.mark.parametrize("tool_name", [
        "mcp__hermes-tools__write_file",
        "mcp__hermes-tools__read_file_evil",
        "mcp__other-server__read_file",
        "Bash",
    ])
    def test_non_bounded_tools_still_use_approval_bridge(self, tool_name):
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(tool_name, {"command": "ls"}, None))
        assert type(result).__name__ == "PermissionResultAllow"
        assert len(calls) == 1
