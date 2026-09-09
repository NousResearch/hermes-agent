"""Unit tests for the public PluginInvocation API (hermes_cli/plugin_invocation.py).

Contracts under test:
  - dispatch_tool fails closed: no authority, no session dispatcher, or a tool
    outside the session's effective tool set are all refused before any execution.
  - arguments must be a dict.
  - the invocation is immutable and carries surface provenance untouched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.plugin_invocation import PluginInvocation


class TestPluginInvocationDispatchFailClosed:
    def test_refuses_when_not_authorized(self):
        invocation = PluginInvocation(
            tool_names=frozenset({"read_file"}), authorized=False,
            _dispatch=lambda name, args: "unreachable",
        )
        with pytest.raises(PermissionError, match="not authorized"):
            invocation.dispatch_tool("read_file", {"path": "x"})

    def test_refuses_without_session_dispatcher(self):
        invocation = PluginInvocation(
            tool_names=frozenset({"read_file"}), authorized=True, _dispatch=None,
        )
        with pytest.raises(RuntimeError, match="[Nn]o session-bound tool dispatcher"):
            invocation.dispatch_tool("read_file", {"path": "x"})

    def test_refuses_tool_outside_session_tool_set(self):
        invocation = PluginInvocation(
            tool_names=frozenset({"read_file"}), authorized=True,
            _dispatch=lambda name, args: "unreachable",
        )
        with pytest.raises(PermissionError, match="not available"):
            invocation.dispatch_tool("terminal", {"command": "whoami"})

    def test_refuses_non_dict_arguments(self):
        invocation = PluginInvocation(
            tool_names=frozenset({"read_file"}), authorized=True,
            _dispatch=lambda name, args: "unreachable",
        )
        with pytest.raises(TypeError, match="dictionary"):
            invocation.dispatch_tool("read_file", ["not", "a", "dict"])

    def test_empty_context_fails_closed_everywhere(self):
        # A surface with no live session builds the default context: dispatch_tool
        # must refuse through the authorization check before anything else.
        invocation = PluginInvocation()
        with pytest.raises(PermissionError, match="not authorized"):
            invocation.dispatch_tool("read_file", {"path": "x"})


class TestPluginInvocationDispatchHappyPath:
    def test_dispatches_through_session_bound_executor(self):
        seen = {}

        def executor(name, args):
            seen["name"] = name
            seen["args"] = dict(args)
            return '{"ok": true}'

        invocation = PluginInvocation(
            session_id="session-1", session_key="session-1", surface="cli", platform="cli",
            cwd=Path("/tmp"), workspace=Path("/tmp"),
            tool_names=frozenset({"read_file", "terminal"}), authorized=True,
            _dispatch=executor,
        )
        result = invocation.dispatch_tool("read_file", {"path": "notes.md"})
        assert result == '{"ok": true}'
        assert seen == {"name": "read_file", "args": {"path": "notes.md"}}


class TestPluginInvocationImmutable:
    def test_fields_are_read_only(self):
        invocation = PluginInvocation(session_id="session-1", surface="gateway")
        with pytest.raises((AttributeError, TypeError)):
            invocation.session_id = "session-2"
        with pytest.raises((AttributeError, TypeError)):
            invocation.surface = "cli"
        with pytest.raises((AttributeError, TypeError)):
            invocation.tool_names = frozenset({"terminal"})
        assert invocation.session_id == "session-1"
        assert invocation.surface == "gateway"

    def test_provenance_survives(self):
        cwd = Path("/home/user/project")
        invocation = PluginInvocation(
            session_id="session-7", session_key="agent:main:telegram:dm:c1",
            surface="gateway", platform="telegram", cwd=cwd, workspace=cwd,
            tool_names=frozenset({"read_file"}), authorized=True,
            _dispatch=lambda name, args: "x",
        )
        assert invocation.session_id == "session-7"
        assert invocation.session_key == "agent:main:telegram:dm:c1"
        assert invocation.surface == "gateway"
        assert invocation.platform == "telegram"
        assert invocation.cwd == cwd
        assert invocation.workspace == cwd
        assert invocation.tool_names == frozenset({"read_file"})
        assert invocation.authorized is True
        # Private wiring never leaks through the public projection.
        assert "_dispatch" not in repr(invocation)
