"""Startup-order contracts for shell-hook registration on Desktop/TUI paths.

Sweeper feedback on PR #53894:
- Dashboard registration must run after interactive auth setup (force-reload
  can clear hooks) and only when HERMES_DESKTOP=1 (per-profile backends).
- TUI entry registers before gateway.ready with accept_hooks=False so
  register_from_config owns hooks_auto_accept / env parsing.
"""

from __future__ import annotations

import io
import sys
import types

import pytest


@pytest.fixture()
def main_mod():
    import hermes_cli.main as main

    return main


def _args(**over):
    base = {
        "host": "127.0.0.1",
        "port": 0,
        "no_open": True,
        "open_profile": None,
        "skip_build": False,
        "headless_backend": True,
        "tui": False,
        "status": False,
        "stop": False,
        "isolated": False,
        "insecure": False,
    }
    base.update(over)
    return types.SimpleNamespace(**base)


def _wire_dashboard(main_mod, monkeypatch, order):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setitem(sys.modules, "fastapi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "hermes_logging",
        types.SimpleNamespace(setup_logging=lambda **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **_k: None,
    )

    def fake_auth(args):
        order.append("auth")

    def fake_register(cfg, accept_hooks=False):
        order.append(("register", accept_hooks))
        return []

    def fake_start(**kwargs):
        order.append("start")

    monkeypatch.setattr(
        main_mod, "_maybe_setup_dashboard_auth_interactively", fake_auth
    )
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config", fake_register
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=fake_start),
    )


def test_dashboard_registers_hooks_for_desktop_after_auth(main_mod, monkeypatch):
    order = []
    _wire_dashboard(main_mod, monkeypatch, order)
    monkeypatch.setenv("HERMES_DESKTOP", "1")

    main_mod.cmd_dashboard(_args())

    assert order == ["auth", ("register", False), "start"]


def test_dashboard_skips_hooks_without_desktop_env(main_mod, monkeypatch):
    order = []
    _wire_dashboard(main_mod, monkeypatch, order)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)

    main_mod.cmd_dashboard(_args())

    assert order == ["auth", "start"]
    assert not any(isinstance(x, tuple) and x[0] == "register" for x in order)


def test_tui_entry_registers_hooks_before_ready(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda cfg, accept_hooks=False: calls.append(
            {"cfg": cfg, "accept_hooks": accept_hooks}
        )
        or [],
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"hooks_auto_accept": "false"},
    )
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {},  # no MCP servers → skip discovery thread
    )

    ready = []

    def fake_write_json(obj):
        ready.append(obj)
        return True

    monkeypatch.setattr("tui_gateway.entry.write_json", fake_write_json)
    monkeypatch.setattr("tui_gateway.entry.resolve_skin", lambda: "default")
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # EOF immediately

    from tui_gateway import entry

    entry.main()

    assert calls == [{"cfg": {"hooks_auto_accept": "false"}, "accept_hooks": False}]
    assert any(
        isinstance(o, dict)
        and (o.get("params") or {}).get("type") == "gateway.ready"
        for o in ready
    )
    assert calls and ready
