
"""Regression test: the TUI launcher must not spend time on plugin discovery.

`hermes --tui` just spawns a Node process; the spawned tui_gateway backend
performs its own plugin discovery. Running discover_plugins() in the
launcher added ~0.5s to every `hermes --tui` startup for work the backend
then redoes. Plain chat must still discover plugins.
"""

from __future__ import annotations

from argparse import Namespace
import sys
import types

import pytest

from hermes_cli import main as main_mod
from hermes_cli import mcp_startup


def _install_discover_spy(monkeypatch):
    calls = []

    def _discover():
        calls.append("discover")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(
            discover_plugins=_discover,
            # main.py now kicks discovery off in a background thread; both
            # entry points count as "discovery work happened in the launcher".
            start_background_plugin_discovery=_discover,
        ),
    )
    # The plain-chat path also arms MCP discovery. Its config probe imports
    # ``hermes_cli.plugins`` (replaced by the stub above), fails, and falls
    # back to "assume configured", which spawned a REAL ``cli-mcp-discovery``
    # daemon thread that was still importing ``tools.mcp_tool`` when pytest
    # exited. A daemon thread inside a C-extension import at interpreter
    # finalization dies via pthread_exit → glibc "FATAL: exception not
    # rethrown" → SIGABRT. Plugin discovery is the only subject here.
    monkeypatch.setattr(
        mcp_startup, "start_background_mcp_discovery", lambda **_kw: None
    )
    return calls


def _args(**overrides):
    base = {
        "accept_hooks": False,
        "yolo": False,
        "safe_mode": False,
        "command": None,
        "query": None,
        "image": None,
    }
    base.update(overrides)
    return Namespace(**base)


def test_plugin_discovery_skipped_for_tui_launch(monkeypatch):
    calls = _install_discover_spy(monkeypatch)
    main_mod._prepare_agent_startup(_args(tui=True))
    assert calls == [], (
        "Plugin discovery must not run in the TUI launcher: the spawned "
        "tui_gateway backend discovers plugins itself."
    )


@pytest.mark.parametrize("overrides, speculative", [
    ({"command": None}, False),
    ({"command": "chat"}, False),
    ({"command": "chat", "query": "hello"}, True),
    ({"command": "chat", "query_file": "prompt.txt"}, True),
    ({"command": None, "oneshot": "hello"}, True),
    ({"command": "acp"}, True),
    ({"command": "rl"}, True),
    ({"command": "gateway", "gateway_command": "run"}, True),
    ({"command": "cron", "cron_command": "tick"}, True),
])
def test_plugin_discovery_waits_for_interactive_chat_consumer(monkeypatch, overrides, speculative):
    calls = _install_discover_spy(monkeypatch)
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: False)
    main_mod._prepare_agent_startup(_args(tui=False, **overrides))
    assert calls == (["discover"] if speculative else []), (
        "Only interactive CLI imports must avoid speculative plugin scanning"
    )

    # Discovery is deferred, not disabled: synchronous consumers still get it.
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    assert calls[-1] == "discover"
