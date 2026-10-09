"""Regression test: the TUI launcher must not spend time on plugin discovery.

`hermes --tui` just spawns a Node process; the spawned tui_gateway backend
performs its own plugin discovery. Running discover_plugins() in the
launcher added ~0.5s to every `hermes --tui` startup for work the backend
then redoes. Plain chat must still discover plugins.
"""

from __future__ import annotations

from argparse import Namespace

import plugin_runtime.lifecycle as plugin_lifecycle

from hermes_cli import main as main_mod
from hermes_cli import mcp_startup


def _install_discover_spy(monkeypatch):
    calls = []

    def _discover(*_args, **_kwargs):
        calls.append("discover")

    monkeypatch.setattr(plugin_lifecycle, "discover_plugins", _discover)
    monkeypatch.setattr(plugin_lifecycle, "start_background_plugin_discovery", _discover)
    # The plain-chat path also arms MCP discovery. Keep that separate daemon
    # out of this test: plugin discovery is the only subject here.
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


def test_plugin_discovery_runs_for_plain_chat(monkeypatch):
    calls = _install_discover_spy(monkeypatch)
    main_mod._prepare_agent_startup(_args(tui=False, command="chat"))
    assert calls == ["discover"]
