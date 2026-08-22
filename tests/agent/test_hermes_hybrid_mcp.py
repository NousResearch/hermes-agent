"""Tests for the hybrid MCP bridge and the shared exposure module.

The bridge is opt-in behind ``agent.claude_agent_sdk.hybrid_mcp_bridge`` and
splits the Hermes registry into two in-process MCP servers so operator
grants keyed on ``mcp__hermes-tools__<tool>`` keep matching. Enough of the
plumbing runs without ``claude_agent_sdk`` installed to pin the invariants
that matter to fcava's review (`fcavalcantirj/hermes-agent#2`):

* deterministic sort order for cache-prefix stability,
* ``strip_mcp_prefix`` semantics both ways (proxied-MCP names would be
  mangled if flipped the wrong way — the bridge silently loses them),
* the exclude-list contract (both buckets are filtered),
* the bucket split (legacy names stay under ``hermes-tools``).

The ``build_option_fields`` end-to-end wiring is covered indirectly by the
existing provider tests; the bridge-internal behaviour is what this file
pins so a refactor doesn't quietly regress the invariants above.
"""

from __future__ import annotations

import sys
import types
from typing import Any, List

import pytest


# ---- fixtures -----------------------------------------------------------


def _openai_spec(name: str, description: str = "", schema: dict | None = None) -> dict:
    """OpenAI-format tool spec — the shape ``agent.tools`` uses."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or name,
            "parameters": schema or {"type": "object", "properties": {}},
        },
    }


def _anthropic_spec(name: str, description: str = "", schema: dict | None = None) -> dict:
    """Anthropic-format tool spec — the shape an OAuth wire payload uses."""
    return {
        "name": name,
        "description": description or name,
        "input_schema": schema or {"type": "object", "properties": {}},
    }


class _FakeSdkTool:
    """Records what ``sdk.tool(...)`` was called with. The bridge decorates
    handlers with it so we can inspect names/descriptions/schemas without
    driving a real MCP server."""

    def __init__(self, name: str, description: str, schema: dict) -> None:
        self.name = name
        self.description = description
        self.schema = schema
        self.handler = None

    def __call__(self, handler):
        self.handler = handler
        return self


class _FakeSdkServer:
    def __init__(self, *, name: str, version: str, tools: list) -> None:
        self.name = name
        self.version = version
        self.tools = tools


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a minimal ``claude_agent_sdk`` stand-in so the bridge can
    import + register without the real SDK. The stand-in captures the
    per-tool ``sdk.tool(name, description, schema)`` calls in registration
    order — that IS the tools:-block byte order the prompt cache keys on,
    so the sort-order pin below asserts on this list directly.
    """
    module = types.ModuleType("claude_agent_sdk")
    recorded: list[_FakeSdkTool] = []

    def _tool_factory(name, description, schema):
        entry = _FakeSdkTool(name, description, schema)
        recorded.append(entry)
        return entry

    def _create_server(*, name, version, tools):
        return _FakeSdkServer(name=name, version=version, tools=tools)

    module.tool = _tool_factory
    module.create_sdk_mcp_server = _create_server
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return {"recorded": recorded, "create_server": _create_server}


@pytest.fixture
def stub_invoke_deps(monkeypatch):
    """Stub the two heavy imports the bridge does at build time — the tests
    exercise decoration, not invocation, so a benign no-op is enough."""
    invoke_module = types.ModuleType("agent.agent_runtime_helpers")
    invoke_module.invoke_tool = lambda *args, **kwargs: ""
    monkeypatch.setitem(sys.modules, "agent.agent_runtime_helpers", invoke_module)

    dispatch_module = types.ModuleType("agent.tool_dispatch_helpers")
    dispatch_module._maybe_wrap_untrusted = lambda name, content: content
    monkeypatch.setitem(sys.modules, "agent.tool_dispatch_helpers", dispatch_module)


class _StubAgent:
    """Bridge only reads a handful of optional attributes at build time —
    task_id, _task_id, _tool_guardrails. None of them are exercised by the
    decoration path (all guarded)."""

    def __init__(self) -> None:
        self.task_id = "stub-task"


# ---- normalize_tool_spec -------------------------------------------------


class TestNormalizeToolSpec:
    """The dispatcher lookup keys on the ``registry_name`` this function
    returns. Getting ``strip_mcp_prefix`` wrong for a caller silently
    corrupts every proxied MCP tool call: the SDK registers, Claude calls,
    ``invoke_tool``'s registry lookup misses, and the bridge returns a
    ``{"error": "unknown tool"}`` envelope while the SDK reports success
    with ``is_error=True``. So both semantics are pinned explicitly."""

    def test_openai_format(self):
        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        result = normalize_tool_spec(
            _openai_spec("web_search", "Search the web", {"type": "object"})
        )
        assert result == ("web_search", "Search the web", {"type": "object"})

    def test_anthropic_format(self):
        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        result = normalize_tool_spec(
            _anthropic_spec("memory", "Persistent notes", {"type": "object"})
        )
        assert result == ("memory", "Persistent notes", {"type": "object"})

    def test_strip_prefix_default_true_reverses_oauth_encoding(self):
        """OAuth-wire callers see every tool name prefixed with ``mcp__``.
        Default True reverses it so the name matches the registry."""
        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        result = normalize_tool_spec(_openai_spec("mcp__web_search"))
        assert result is not None
        assert result[0] == "web_search"

    def test_strip_prefix_false_preserves_registry_keys(self):
        """The hybrid bridge reads from the internal registry, where
        proxied MCP tools are natively keyed as ``mcp__<server>__<tool>``.
        Stripping here would mangle the key into an unresolvable string
        (``someserver__some_tool``), and the dispatcher would silently
        return ``{"error": "unknown tool"}`` for every proxied call. The
        bridge MUST pass ``strip_mcp_prefix=False``."""
        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        name = "mcp__proton__list_messages"
        result = normalize_tool_spec(_openai_spec(name), strip_mcp_prefix=False)
        assert result is not None
        assert result[0] == name

    def test_missing_name_returns_none(self):
        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        assert normalize_tool_spec({"type": "function", "function": {}}) is None
        assert normalize_tool_spec({"input_schema": {}}) is None

    def test_missing_schema_defaults_to_empty_object(self):
        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        result = normalize_tool_spec({"name": "foo"})
        assert result is not None
        _, _, schema = result
        assert schema == {"type": "object", "properties": {}}

    def test_non_dict_input_rejected(self):
        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        assert normalize_tool_spec(None) is None
        assert normalize_tool_spec("not a dict") is None
        assert normalize_tool_spec(42) is None


# ---- HERMES_TOOLS_LEGACY_NAMES ------------------------------------------


class TestLegacyNames:
    """``~/.claude/settings.json`` grants key on ``mcp__hermes-tools__<tool>``.
    These names must stay reachable under that server when the hybrid bridge
    takes over, or a box on ``permission_mode: default`` gets an approval
    storm for tools it already granted."""

    def test_includes_stateless_curated_set(self):
        from agent.transports.hermes_tool_exposure import (
            CURATED_STATELESS_TOOLS,
            HERMES_TOOLS_LEGACY_NAMES,
        )

        assert set(CURATED_STATELESS_TOOLS).issubset(HERMES_TOOLS_LEGACY_NAMES)

    def test_includes_stdio_shim_tools(self):
        """``memory`` and ``session_search`` are exposed by the stdio
        wrapper through dedicated stateless shims, not through
        ``EXPOSED_TOOLS``. Grants for them must survive too."""
        from agent.transports.hermes_tool_exposure import HERMES_TOOLS_LEGACY_NAMES

        assert "memory" in HERMES_TOOLS_LEGACY_NAMES
        assert "session_search" in HERMES_TOOLS_LEGACY_NAMES

    def test_includes_sdk_profile_inspection_tools(self):
        """Hybrid mode replaces the SDK stdio server, so its bounded readers
        must retain the exact ``mcp__hermes-tools__*`` identities that the
        SDK permission bridge auto-allows."""
        from agent.transports.hermes_tool_exposure import HERMES_TOOLS_LEGACY_NAMES

        assert "read_file" in HERMES_TOOLS_LEGACY_NAMES
        assert "search_files" in HERMES_TOOLS_LEGACY_NAMES

    def test_includes_kanban_review_tools(self):
        """``kanban_request_review`` and ``kanban_request_changes`` ship in
        the stdio wrapper's ``EXPOSED_TOOLS`` but not in the module-level
        ``CURATED_STATELESS_TOOLS`` — the legacy set covers the union so
        grants for them still land in the ``hermes-tools`` bucket."""
        from agent.transports.hermes_tool_exposure import HERMES_TOOLS_LEGACY_NAMES

        assert "kanban_request_review" in HERMES_TOOLS_LEGACY_NAMES
        assert "kanban_request_changes" in HERMES_TOOLS_LEGACY_NAMES


# ---- build_hybrid_mcp_server --------------------------------------------


class TestHybridServerBuild:
    def test_registers_all_tools_by_default(self, fake_sdk, stub_invoke_deps):
        from agent.transports.hermes_hybrid_mcp import build_hybrid_mcp_server

        tools = [
            _openai_spec("web_search"),
            _openai_spec("memory"),
            _openai_spec("mcp__proton__list_messages"),
        ]
        server = build_hybrid_mcp_server(_StubAgent(), tools)
        names = [entry.name for entry in fake_sdk["recorded"]]
        assert set(names) == {"web_search", "memory", "mcp__proton__list_messages"}
        assert server.name == "hermes-hybrid"

    def test_only_names_filters(self, fake_sdk, stub_invoke_deps):
        """The bucket split uses ``only_names`` to isolate legacy names."""
        from agent.transports.hermes_hybrid_mcp import build_hybrid_mcp_server

        tools = [
            _openai_spec("web_search"),
            _openai_spec("memory"),
            _openai_spec("mcp__proton__list_messages"),
        ]
        build_hybrid_mcp_server(
            _StubAgent(),
            tools,
            server_name="hermes-tools",
            only_names={"web_search", "memory"},
        )
        names = [entry.name for entry in fake_sdk["recorded"]]
        assert set(names) == {"memory", "web_search"}

    def test_exclude_names_drops_tools_from_both_buckets(
        self, fake_sdk, stub_invoke_deps
    ):
        """The operator's exclude list is meant to keep a specific
        high-blast tool off the SDK regardless of which bucket would host
        it. Bucket ownership must not become a loophole."""
        from agent.transports.hermes_hybrid_mcp import build_hybrid_mcp_server

        tools = [
            _openai_spec("web_search"),
            _openai_spec("delegate_task"),
            _openai_spec("mcp__proton__list_messages"),
        ]
        build_hybrid_mcp_server(
            _StubAgent(),
            tools,
            exclude_names={"delegate_task"},
        )
        names = [entry.name for entry in fake_sdk["recorded"]]
        assert "delegate_task" not in names
        assert set(names) == {"web_search", "mcp__proton__list_messages"}

    def test_server_name_override(self, fake_sdk, stub_invoke_deps):
        from agent.transports.hermes_hybrid_mcp import build_hybrid_mcp_server

        server = build_hybrid_mcp_server(
            _StubAgent(),
            [_openai_spec("web_search")],
            server_name="hermes-tools",
        )
        assert server.name == "hermes-tools"

    def test_deterministic_sort_order_pin(self, fake_sdk, stub_invoke_deps):
        """The SDK bakes the registered tool order into the ``tools:``
        block sent to Anthropic. Any run-to-run reorder (Python dict
        iteration order across restarts, plugin discovery order, MCP proxy
        refresh sequence) invalidates the prompt cache prefix — cache-hit
        rate collapses from ~90% to ~0. This test fails hard if the sort
        gets removed, so a future refactor cannot silently regress the
        cache posture."""
        from agent.transports.hermes_hybrid_mcp import build_hybrid_mcp_server

        input_a = [
            _openai_spec("web_search"),
            _openai_spec("memory"),
            _openai_spec("mcp__proton__list_messages"),
            _openai_spec("browser_navigate"),
        ]
        input_b = list(reversed(input_a))

        build_hybrid_mcp_server(_StubAgent(), input_a)
        order_a = [entry.name for entry in fake_sdk["recorded"]]
        fake_sdk["recorded"].clear()

        build_hybrid_mcp_server(_StubAgent(), input_b)
        order_b = [entry.name for entry in fake_sdk["recorded"]]

        assert order_a == order_b
        assert order_a == sorted(order_a)

    def test_skips_specs_missing_name(self, fake_sdk, stub_invoke_deps):
        from agent.transports.hermes_hybrid_mcp import build_hybrid_mcp_server

        tools = [
            _openai_spec("web_search"),
            {"type": "function", "function": {}},  # no name
            None,  # not a dict
        ]
        build_hybrid_mcp_server(_StubAgent(), tools)
        names = [entry.name for entry in fake_sdk["recorded"]]
        assert names == ["web_search"]

    def test_preserves_mcp_prefix_for_proxied_tools(
        self, fake_sdk, stub_invoke_deps
    ):
        """Regression pin for the ``strip_mcp_prefix=False`` invariant at
        the call site: if a future refactor flips it, proxied MCP tools
        get renamed at registration time, the SDK's dispatch lookup
        misses, and Claude sees ``{"error": "unknown tool"}`` for every
        proxied call — with ``is_error=True`` masquerading as success."""
        from agent.transports.hermes_hybrid_mcp import build_hybrid_mcp_server

        build_hybrid_mcp_server(
            _StubAgent(),
            [_openai_spec("mcp__proton__list_messages")],
        )
        names = [entry.name for entry in fake_sdk["recorded"]]
        assert names == ["mcp__proton__list_messages"]


# ---- _configured_hybrid_exclude -----------------------------------------


class TestConfiguredHybridExclude:
    def _patch_provider_config(self, monkeypatch, value):
        from agent.transports import claude_agent_sdk_session as mod

        monkeypatch.setattr(mod, "_provider_config", lambda: {"hybrid_mcp_bridge_exclude": value})

    def test_default_empty_list_when_missing(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session as mod

        monkeypatch.setattr(mod, "_provider_config", lambda: {})
        assert mod._configured_hybrid_exclude() == []

    def test_returns_stripped_deduped_names(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session as mod

        self._patch_provider_config(
            monkeypatch,
            ["delegate_task", " read_terminal ", "delegate_task"],
        )
        assert mod._configured_hybrid_exclude() == [
            "delegate_task",
            "read_terminal",
        ]

    def test_non_list_returns_empty(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session as mod

        self._patch_provider_config(monkeypatch, "delegate_task")
        assert mod._configured_hybrid_exclude() == []

    def test_non_string_entries_dropped(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session as mod

        self._patch_provider_config(monkeypatch, ["delegate_task", 42, None, ""])
        assert mod._configured_hybrid_exclude() == ["delegate_task"]


# ---- runtime-level gate --------------------------------------------------


class TestHybridBridgeEnabledGate:
    """The runtime call site in ``claude_sdk_runtime.py`` is the real
    switch: if it always passed ``agent + tools``, the bridge would activate
    everywhere on day one regardless of the config default. This pins the
    default-off contract."""

    def test_default_disabled(self, monkeypatch):
        from agent import claude_sdk_runtime as runtime
        from agent.transports import claude_agent_sdk_session as sess

        monkeypatch.setattr(sess, "_provider_config", lambda: {})
        assert runtime._hybrid_bridge_enabled() is False

    def test_session_fails_closed_even_if_caller_supplies_bridge_inputs(
        self, monkeypatch, fake_sdk, stub_invoke_deps
    ):
        from agent.transports import claude_agent_sdk_session as sess

        http_loader = types.SimpleNamespace(calls=0)

        def _load_http():
            http_loader.calls += 1
            return {
                "remote": {
                    "type": "http",
                    "url": "https://mcp.example.test",
                }
            }

        monkeypatch.setattr(sess, "_provider_config", lambda: {})
        monkeypatch.setattr(sess, "_http_mcp_entries_from_config", _load_http)
        session = sess.ClaudeAgentSdkSession(
            cwd="/tmp",
            agent=_StubAgent(),
            tools=[_openai_spec("web_search")],
        )
        fields = session.build_option_fields()

        assert http_loader.calls == 0
        assert "hermes-hybrid" not in fields["mcp_servers"]
        assert "remote" not in fields["mcp_servers"]

    def test_direct_headerless_http_requires_successful_hybrid_opt_in(
        self, monkeypatch, fake_sdk, stub_invoke_deps
    ):
        from agent.transports import claude_agent_sdk_session as sess

        monkeypatch.setattr(
            sess, "_provider_config", lambda: {"hybrid_mcp_bridge": True}
        )
        monkeypatch.setattr(
            sess,
            "_http_mcp_entries_from_config",
            lambda: {
                "remote": {
                    "type": "http",
                    "url": "https://mcp.example.test",
                }
            },
        )
        session = sess.ClaudeAgentSdkSession(
            cwd="/tmp",
            agent=_StubAgent(),
            tools=[_openai_spec("web_search")],
        )

        fields = session.build_option_fields()

        assert "hermes-hybrid" in fields["mcp_servers"]
        assert fields["mcp_servers"]["remote"] == {
            "type": "http",
            "url": "https://mcp.example.test",
        }

    def test_enabled_when_flag_true(self, monkeypatch):
        from agent import claude_sdk_runtime as runtime
        from agent.transports import claude_agent_sdk_session as sess

        monkeypatch.setattr(
            sess, "_provider_config", lambda: {"hybrid_mcp_bridge": True}
        )
        assert runtime._hybrid_bridge_enabled() is True

    def test_string_true_coerces(self, monkeypatch):
        """``_provider_flag`` accepts ``"true"``/``"1"``/``"yes"`` so YAML
        strings don't silently degrade to ``bool("true") = True`` semantics
        the operator didn't intend — pin the behaviour."""
        from agent import claude_sdk_runtime as runtime
        from agent.transports import claude_agent_sdk_session as sess

        monkeypatch.setattr(
            sess, "_provider_config", lambda: {"hybrid_mcp_bridge": "true"}
        )
        assert runtime._hybrid_bridge_enabled() is True

        monkeypatch.setattr(
            sess, "_provider_config", lambda: {"hybrid_mcp_bridge": "no"}
        )
        assert runtime._hybrid_bridge_enabled() is False
