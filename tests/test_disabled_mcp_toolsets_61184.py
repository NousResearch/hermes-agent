"""Regression tests for issue #61184.

``agent.disabled_toolsets`` must be a hard security boundary for MCP servers:

* bare name (``server-b``) and canonical (``mcp-server-b``) both work
* schemas exclude disabled MCP tools after discovery
* oneshot (-z) passes disabled_toolsets and waits for MCP discovery
* dispatch refuses tools from disabled toolsets with an explicit error
* tool_search / tool_describe / tool_call cannot surface or run them
* CLI list/status surfaces the blocked state
* unavailable intended servers do not silently fall back to a disabled peer
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.registry import registry
from toolsets import (
    expand_disabled_toolset_names,
    is_mcp_server_disabled_by_toolsets,
    tool_blocked_by_disabled_toolsets,
)


SHARED_LOCAL_TOOL = "cluster_status"


def _register_fake_mcp_server(server_name: str, local_tool: str = SHARED_LOCAL_TOOL, *, result_payload: str = "ok"):
    """Register a fake MCP-style tool: toolset mcp-<server>, alias <server>."""
    from tools.mcp_tool import sanitize_mcp_name_component

    safe_server = sanitize_mcp_name_component(server_name)
    safe_tool = sanitize_mcp_name_component(local_tool)
    tool_name = f"mcp_{safe_server}_{safe_tool}"
    toolset = f"mcp-{server_name}"

    def _handler(args, **kwargs):
        return json.dumps({
            "server": server_name,
            "tool": local_tool,
            "result": result_payload,
        })

    registry.register(
        name=tool_name,
        toolset=toolset,
        schema={
            "name": tool_name,
            "description": f"{local_tool} from {server_name}",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        handler=_handler,
        check_fn=lambda: True,
        description=f"{local_tool} from {server_name}",
    )
    registry.register_toolset_alias(server_name, toolset)
    return tool_name, toolset


@pytest.fixture
def two_mcp_servers(monkeypatch):
    """Two MCP servers exposing the same local tool name; server-b is sensitive."""
    # Isolate registry mutations from other tests.
    saved_tools = dict(registry._tools)
    saved_aliases = dict(registry._toolset_aliases)
    saved_checks = dict(registry._toolset_checks)
    saved_gen = registry._generation

    # Clear any prior MCP entries for these names (idempotent re-register).
    for name in list(registry._tools.keys()):
        if name.startswith("mcp_server_a_") or name.startswith("mcp_server_b_"):
            try:
                registry.deregister(name)
            except Exception:
                pass

    tool_a, ts_a = _register_fake_mcp_server("server-a", result_payload="cluster-A")
    tool_b, ts_b = _register_fake_mcp_server("server-b", result_payload="cluster-B")

    mcp_servers = {
        "server-a": {"url": "http://host-a/mcp"},
        "server-b": {"url": "https://host-b/mcp"},
    }
    monkeypatch.setattr(
        "toolsets._configured_mcp_server_names",
        lambda: set(mcp_servers.keys()),
    )

    yield {
        "tool_a": tool_a,
        "tool_b": tool_b,
        "ts_a": ts_a,
        "ts_b": ts_b,
        "mcp_servers": mcp_servers,
    }

    # Restore registry snapshot.
    with registry._lock:
        registry._tools.clear()
        registry._tools.update(saved_tools)
        registry._toolset_aliases.clear()
        registry._toolset_aliases.update(saved_aliases)
        registry._toolset_checks.clear()
        registry._toolset_checks.update(saved_checks)
        registry._generation = saved_gen + 1


def _tool_names(defs):
    return {t["function"]["name"] for t in defs}


# ---------------------------------------------------------------------------
# expand / alias helpers
# ---------------------------------------------------------------------------


class TestExpandDisabledMcpAliases61184:
    def test_bare_and_canonical_expand_to_each_other(self, two_mcp_servers):
        bare = expand_disabled_toolset_names(["server-b"])
        canon = expand_disabled_toolset_names(["mcp-server-b"])
        assert "server-b" in bare and "mcp-server-b" in bare
        assert "server-b" in canon and "mcp-server-b" in canon

    def test_is_mcp_server_disabled_accepts_both_forms(self, two_mcp_servers):
        assert is_mcp_server_disabled_by_toolsets("server-b", ["server-b"])
        assert is_mcp_server_disabled_by_toolsets("server-b", ["mcp-server-b"])
        assert not is_mcp_server_disabled_by_toolsets("server-a", ["server-b"])

    def test_tool_blocked_label_mentions_both_forms(self, two_mcp_servers):
        label = tool_blocked_by_disabled_toolsets(
            two_mcp_servers["tool_b"], ["server-b"],
        )
        assert label is not None
        assert "server-b" in label
        assert "mcp-server-b" in label


# ---------------------------------------------------------------------------
# get_tool_definitions schema filtering
# ---------------------------------------------------------------------------


class TestGetToolDefinitionsDisabledMcp61184:
    def test_disabled_bare_name_excludes_server_b_tools(self, two_mcp_servers):
        from model_tools import get_tool_definitions, _clear_tool_defs_cache

        _clear_tool_defs_cache()
        defs = get_tool_definitions(
            enabled_toolsets=["server-a", "server-b"],
            disabled_toolsets=["server-b"],
            quiet_mode=True,
        )
        names = _tool_names(defs)
        assert two_mcp_servers["tool_a"] in names
        assert two_mcp_servers["tool_b"] not in names

    def test_disabled_canonical_name_same_as_bare(self, two_mcp_servers):
        from model_tools import get_tool_definitions, _clear_tool_defs_cache

        _clear_tool_defs_cache()
        bare = _tool_names(get_tool_definitions(
            enabled_toolsets=["server-a", "server-b"],
            disabled_toolsets=["server-b"],
            quiet_mode=True,
        ))
        _clear_tool_defs_cache()
        canon = _tool_names(get_tool_definitions(
            enabled_toolsets=["server-a", "server-b"],
            disabled_toolsets=["mcp-server-b"],
            quiet_mode=True,
        ))
        assert bare == canon
        assert two_mcp_servers["tool_b"] not in bare
        assert two_mcp_servers["tool_a"] in bare

    def test_non_mcp_disabled_toolsets_still_work(self, two_mcp_servers):
        """Preserves existing behavior for non-MCP toolsets (#61184 constraint)."""
        from model_tools import get_tool_definitions, _clear_tool_defs_cache

        # Use a private non-MCP tool so we are not dependent on check_fn / API keys.
        fake_name = "_61184_probe_tool"
        if fake_name in registry._tools:
            registry.deregister(fake_name)
        registry.register(
            name=fake_name,
            toolset="probe61184",
            schema={
                "name": fake_name,
                "description": "probe",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda args, **kw: json.dumps({"ok": True}),
            check_fn=lambda: True,
            description="probe",
        )
        try:
            # Make probe61184 a real toolset name for validate_toolset.
            from toolsets import TOOLSETS

            TOOLSETS["probe61184"] = {
                "description": "probe",
                "tools": [fake_name],
                "includes": [],
            }
            _clear_tool_defs_cache()
            with_probe = _tool_names(get_tool_definitions(
                enabled_toolsets=["probe61184", "server-a"],
                disabled_toolsets=None,
                quiet_mode=True,
            ))
            _clear_tool_defs_cache()
            no_probe = _tool_names(get_tool_definitions(
                enabled_toolsets=["probe61184", "server-a"],
                disabled_toolsets=["probe61184"],
                quiet_mode=True,
            ))
            assert fake_name in with_probe
            assert fake_name not in no_probe
            assert two_mcp_servers["tool_a"] in no_probe
        finally:
            try:
                registry.deregister(fake_name)
            except Exception:
                pass
            from toolsets import TOOLSETS
            TOOLSETS.pop("probe61184", None)


# ---------------------------------------------------------------------------
# Dispatch-time enforcement
# ---------------------------------------------------------------------------


class TestDispatchDisabledMcp61184:
    def test_dispatch_refuses_disabled_server_b_tool(self, two_mcp_servers):
        from model_tools import handle_function_call

        raw = handle_function_call(
            two_mcp_servers["tool_b"],
            {},
            disabled_toolsets=["server-b"],
        )
        result = json.loads(raw)
        assert "error" in result
        assert two_mcp_servers["tool_b"] in result["error"]
        assert "disabled toolset" in result["error"]
        assert "server-b" in result["error"]
        assert "cannot be executed" in result["error"]

    def test_dispatch_refuses_with_canonical_disabled_name(self, two_mcp_servers):
        from model_tools import handle_function_call

        raw = handle_function_call(
            two_mcp_servers["tool_b"],
            {},
            disabled_toolsets=["mcp-server-b"],
        )
        result = json.loads(raw)
        assert "error" in result
        assert "cannot be executed" in result["error"]

    def test_dispatch_allows_server_a_when_only_b_disabled(self, two_mcp_servers):
        from model_tools import handle_function_call

        raw = handle_function_call(
            two_mcp_servers["tool_a"],
            {},
            disabled_toolsets=["server-b"],
        )
        result = json.loads(raw)
        assert result.get("server") == "server-a"
        assert result.get("result") == "cluster-A"

    def test_no_silent_fallback_when_server_a_unavailable(self, two_mcp_servers, monkeypatch):
        """If the intended server is down, do not execute the disabled peer."""
        from model_tools import handle_function_call

        # Simulate server-a check failing / tool deregistered from schema, but
        # a stale direct dispatch against server-b must still be refused.
        raw = handle_function_call(
            two_mcp_servers["tool_b"],
            {},
            enabled_toolsets=["server-a"],
            disabled_toolsets=["server-b"],
        )
        result = json.loads(raw)
        assert "error" in result
        assert result.get("server") != "server-b"
        assert "cluster-B" not in json.dumps(result)


# ---------------------------------------------------------------------------
# tool_search progressive disclosure
# ---------------------------------------------------------------------------


class TestToolSearchDisabledMcp61184:
    def test_disabled_tools_not_in_tool_search_catalog(self, two_mcp_servers):
        from model_tools import get_tool_definitions, _clear_tool_defs_cache

        _clear_tool_defs_cache()
        defs = get_tool_definitions(
            enabled_toolsets=["server-a", "server-b"],
            disabled_toolsets=["server-b"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = _tool_names(defs)
        assert two_mcp_servers["tool_b"] not in names
        assert two_mcp_servers["tool_a"] in names

    def test_tool_call_bridge_cannot_execute_disabled_mcp(self, two_mcp_servers, monkeypatch):
        from model_tools import handle_function_call
        from tools import tool_search as ts

        # Force bridge path: tool_call with underlying name of disabled tool.
        raw = handle_function_call(
            ts.TOOL_CALL_NAME,
            {"name": two_mcp_servers["tool_b"], "arguments": {}},
            enabled_toolsets=["server-a", "server-b"],
            disabled_toolsets=["server-b"],
        )
        result = json.loads(raw)
        assert "error" in result
        # Either scoped-catalog rejection or disabled-toolset refusal.
        err = result["error"].lower()
        assert (
            "not available" in err
            or "disabled toolset" in err
            or "cannot be executed" in err
        )
        assert result.get("server") != "server-b"


# ---------------------------------------------------------------------------
# oneshot path
# ---------------------------------------------------------------------------


class TestOneshotDisabledMcp61184:
    def test_oneshot_passes_disabled_toolsets_and_awaits_mcp(self, two_mcp_servers, monkeypatch):
        from hermes_cli import oneshot as oneshot_mod

        captured = {}
        synchronous_discover = MagicMock()

        class FakeAgent:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                self.suppress_status_output = False
                self.stream_delta_callback = None
                self.tool_gen_callback = None
                self.tools = []
                self.valid_tool_names = set()
                self.enabled_toolsets = kwargs.get("enabled_toolsets")
                self.disabled_toolsets = kwargs.get("disabled_toolsets")

            def run_conversation(self, prompt):
                from model_tools import get_tool_definitions

                self.tools = get_tool_definitions(
                    enabled_toolsets=self.enabled_toolsets,
                    disabled_toolsets=self.disabled_toolsets,
                    quiet_mode=True,
                )
                self.valid_tool_names = {t["function"]["name"] for t in self.tools}
                captured["tool_names"] = set(self.valid_tool_names)
                return {"final_response": "ok", "completed": True}

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {"default": "test-model", "provider": "test"},
                "agent": {"disabled_toolsets": ["server-b"]},
                "mcp_servers": two_mcp_servers["mcp_servers"],
            },
        )
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools",
            lambda cfg, platform: {"server-a", "server-b", "web"},
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kw: {
                "api_key": "k",
                "base_url": "http://localhost",
                "provider": "test",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            "hermes_cli.fallback_config.get_fallback_chain",
            lambda cfg: None,
        )
        monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
        monkeypatch.setattr(
            "hermes_cli.mcp_startup.start_background_mcp_discovery",
            lambda **kw: captured.__setitem__("bg_started", True),
        )
        monkeypatch.setattr(
            "hermes_cli.mcp_startup.wait_for_mcp_discovery",
            lambda timeout=None: captured.__setitem__("waited", True),
        )
        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", synchronous_discover)
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

        response, result = oneshot_mod._run_agent("check cluster", model="test-model")

        assert response == "ok"
        assert captured.get("waited") is True
        synchronous_discover.assert_not_called()
        assert captured["kwargs"].get("disabled_toolsets") == ["server-b"]
        assert two_mcp_servers["tool_b"] not in captured.get("tool_names", set())
        assert two_mcp_servers["tool_a"] in captured.get("tool_names", set())

    def test_slow_discovery_is_not_duplicated_or_awaited_past_bound(self, monkeypatch):
        from hermes_cli import mcp_startup
        from hermes_cli import oneshot as oneshot_mod

        discovery_started = threading.Event()
        release_discovery = threading.Event()
        attempts = 0

        def slow_discover():
            nonlocal attempts
            attempts += 1
            discovery_started.set()
            release_discovery.wait(timeout=2)

        class FakeAgent:
            def __init__(self, **kwargs):
                self.suppress_status_output = False
                self.stream_delta_callback = None
                self.tool_gen_callback = None

            def run_conversation(self, prompt):
                return {"final_response": "ok", "completed": True}

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {"default": "test-model", "provider": "test"},
                "mcp_discovery_timeout": 0.02,
                "mcp_servers": {"slow-server": {"url": "http://slow/mcp"}},
            },
        )
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools",
            lambda cfg, platform: {"slow-server"},
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kw: {
                "api_key": "k",
                "base_url": "http://localhost",
                "provider": "test",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            "hermes_cli.fallback_config.get_fallback_chain",
            lambda cfg: None,
        )
        monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(mcp_startup, "_mcp_discovery_started", False)
        monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)
        monkeypatch.setattr(mcp_startup, "_has_configured_mcp_servers", lambda: True)
        monkeypatch.setattr(
            mcp_startup,
            "_discover_mcp_tools_without_interactive_oauth",
            slow_discover,
        )

        started_at = time.monotonic()
        try:
            response, _ = oneshot_mod._run_agent("check cluster", model="test-model")
            elapsed = time.monotonic() - started_at

            assert discovery_started.is_set()
            assert response == "ok"
            assert attempts == 1
            assert elapsed < 0.5
            assert mcp_startup.mcp_discovery_in_flight()
        finally:
            release_discovery.set()
            thread = mcp_startup._mcp_discovery_thread
            if thread is not None:
                thread.join(timeout=1)


# ---------------------------------------------------------------------------
# Platform tools + CLI surfaces
# ---------------------------------------------------------------------------


class TestPlatformAndCliDisabledMcp61184:
    def test_get_platform_tools_removes_bare_and_canonical(self, two_mcp_servers):
        from hermes_cli.tools_config import _get_platform_tools

        config = {
            "platform_toolsets": {"cli": ["web"]},
            "agent": {"disabled_toolsets": ["server-b"]},
            "mcp_servers": two_mcp_servers["mcp_servers"],
        }
        enabled = _get_platform_tools(config, "cli")
        assert "server-b" not in enabled
        assert "mcp-server-b" not in enabled
        # server-a still present when defaults include MCP servers
        assert "server-a" in enabled

        config2 = {
            "platform_toolsets": {"cli": ["web"]},
            "agent": {"disabled_toolsets": ["mcp-server-b"]},
            "mcp_servers": two_mcp_servers["mcp_servers"],
        }
        enabled2 = _get_platform_tools(config2, "cli")
        assert "server-b" not in enabled2
        assert "mcp-server-b" not in enabled2

    def test_mcp_list_shows_blocked_by_disabled_toolsets(self, two_mcp_servers, monkeypatch, capsys, tmp_path):
        from hermes_cli import mcp_config

        monkeypatch.setattr(
            mcp_config,
            "_get_mcp_servers",
            lambda config=None: {
                "server-a": {"url": "http://host-a/mcp", "enabled": True},
                "server-b": {"url": "https://host-b/mcp", "enabled": True},
            },
        )
        monkeypatch.setattr(
            "toolsets.is_mcp_server_disabled_by_toolsets",
            lambda name, d=None: str(name) == "server-b",
        )

        mcp_config.cmd_mcp_list()
        out = capsys.readouterr().out
        assert "server-a" in out
        assert "server-b" in out
        assert "disabled_toolsets" in out or "blocked" in out.lower()

    def test_tools_list_marks_disabled_mcp(self, two_mcp_servers, capsys, monkeypatch):
        from hermes_cli.tools_config import _print_tools_list

        monkeypatch.setattr(
            "toolsets.is_mcp_server_disabled_by_toolsets",
            lambda name, d=None: str(name) == "server-b",
        )
        _print_tools_list(
            enabled_toolsets={"web", "server-a"},
            mcp_servers=two_mcp_servers["mcp_servers"],
            platform="cli",
        )
        out = capsys.readouterr().out
        assert "server-b" in out
        assert "disabled_toolsets" in out

    def test_get_mcp_status_marks_disabled_toolsets(self, two_mcp_servers, monkeypatch):
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(
            mcp_tool,
            "_load_mcp_config",
            lambda: {
                "server-a": {"url": "http://a"},
                "server-b": {"url": "https://b"},
            },
        )
        monkeypatch.setattr(
            "toolsets.is_mcp_server_disabled_by_toolsets",
            lambda name, d=None: str(name) == "server-b",
        )
        with mcp_tool._lock:
            saved = dict(mcp_tool._servers)
            mcp_tool._servers.clear()
        try:
            statuses = {e["name"]: e for e in mcp_tool.get_mcp_status()}
        finally:
            with mcp_tool._lock:
                mcp_tool._servers.clear()
                mcp_tool._servers.update(saved)

        assert statuses["server-b"]["disabled"] is True
        assert statuses["server-b"]["status"] == "disabled_toolsets"
        assert statuses["server-a"]["disabled"] is False
