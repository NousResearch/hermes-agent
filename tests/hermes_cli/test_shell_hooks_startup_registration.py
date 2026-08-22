"""Session-owned shell-hook registration contracts (PR #53894 / #83980).

Dashboard/TUI no longer process-global-register at startup. Registration
happens at tui_gateway.server._make_agent from the config already loaded under
the active HERMES_HOME (including multi-profile overrides). Idempotence is
keyed by profile + event + matcher + command; callbacks only run under the
owning profile.
"""

from __future__ import annotations

import types
from pathlib import Path

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
    import sys

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

    def fake_register(cfg, accept_hooks=False, profile=None):
        order.append(("register", accept_hooks, profile))
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


def test_dashboard_does_not_register_hooks_at_startup(main_mod, monkeypatch):
    """Startup must not stamp launch-profile hooks; _make_agent owns it."""
    order = []
    _wire_dashboard(main_mod, monkeypatch, order)
    monkeypatch.setenv("HERMES_DESKTOP", "1")

    main_mod.cmd_dashboard(_args())

    assert order == ["auth", "start"]
    assert not any(isinstance(x, tuple) and x[0] == "register" for x in order)


def test_dashboard_skips_hooks_without_desktop_env(main_mod, monkeypatch):
    order = []
    _wire_dashboard(main_mod, monkeypatch, order)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)

    main_mod.cmd_dashboard(_args())

    assert order == ["auth", "start"]
    assert not any(isinstance(x, tuple) and x[0] == "register" for x in order)


def test_tui_entry_does_not_register_hooks_at_startup(monkeypatch):
    import io
    import sys

    calls = []

    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda cfg, accept_hooks=False, profile=None: calls.append(
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

    assert calls == []
    assert any(
        isinstance(o, dict)
        and (o.get("params") or {}).get("type") == "gateway.ready"
        for o in ready
    )


def test_make_agent_registers_hooks_from_loaded_cfg(monkeypatch, tmp_path):
    """Shared agent construction path is the registration boundary."""
    calls = []

    monkeypatch.setattr(
        "tui_gateway.server._load_cfg",
        lambda: {
            "hooks": {"pre_llm_call": [{"command": "true"}]},
            "hooks_auto_accept": True,
        },
    )
    monkeypatch.setattr(
        "tui_gateway.server._parse_tui_skills_env",
        lambda: [],
    )
    monkeypatch.setattr(
        "tui_gateway.server._resolve_startup_runtime",
        lambda: ("test-model", None),
    )
    monkeypatch.setattr(
        "tui_gateway.server._resolve_runtime_with_fallback",
        lambda kwargs: types.SimpleNamespace(
            runtime={
                "provider": "test",
                "base_url": "http://example",
                "api_key": "k",
                "api_mode": "chat_completions",
            },
            used_fallback=False,
            selected_model=None,
        ),
    )
    monkeypatch.setattr("tui_gateway.server._load_provider_routing", lambda: {})
    monkeypatch.setattr("tui_gateway.server._cfg_max_turns", lambda cfg, d: d)
    monkeypatch.setattr("tui_gateway.server._load_reasoning_config", lambda m: None)
    monkeypatch.setattr("tui_gateway.server._load_service_tier", lambda: None)
    monkeypatch.setattr(
        "tui_gateway.server._load_enabled_toolsets", lambda p: None
    )
    monkeypatch.setattr(
        "tui_gateway.server._resolve_agent_platform", lambda p=None: "tui"
    )
    monkeypatch.setattr("tui_gateway.server._get_db", lambda: None)
    monkeypatch.setattr("tui_gateway.server._load_fallback_model", lambda: None)
    monkeypatch.setattr("tui_gateway.server._agent_cbs", lambda sid: {})
    monkeypatch.setattr(
        "tui_gateway.synthetic_turn.maybe_build_synthetic_agent",
        lambda *a, **k: None,
    )

    class _FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        __import__("sys").modules,
        "run_agent",
        types.SimpleNamespace(AIAgent=_FakeAgent),
    )

    def fake_register(cfg, accept_hooks=False, profile=None):
        calls.append(
            {
                "hooks": (cfg or {}).get("hooks"),
                "accept_hooks": accept_hooks,
                "profile": profile,
            }
        )
        return []

    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config", fake_register
    )
    monkeypatch.setattr(
        "hermes_cli.config.resolve_ephemeral_system_prompt_from_config",
        lambda cfg: None,
    )

    from tui_gateway import server as srv

    agent = srv._make_agent("sid1", "key1", session_id="sess1")
    assert isinstance(agent, _FakeAgent)
    assert calls == [
        {
            "hooks": {"pre_llm_call": [{"command": "true"}]},
            "accept_hooks": False,
            "profile": None,
        }
    ]


def test_profile_keyed_idempotence_and_dispatch(tmp_path, monkeypatch):
    """Same command under two profiles registers twice; each fires only at home."""
    import agent.shell_hooks as shell_hooks
    from hermes_cli.plugins import get_plugin_manager
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    shell_hooks.reset_for_tests()
    manager = get_plugin_manager()
    manager._hooks.clear()

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()

    script = tmp_path / "hook.sh"
    if shell_hooks.IS_WINDOWS:
        script = tmp_path / "hook.py"
        script.write_text(
            "import json,sys\njson.dump({}, sys.stdout)\n", encoding="utf-8"
        )
        cmd = f"{Path(__import__('sys').executable).as_posix()} {script.as_posix()}"
    else:
        script.write_text("#!/usr/bin/env bash\necho '{}'\n", encoding="utf-8")
        script.chmod(0o755)
        cmd = str(script)

    cfg = {
        "hooks": {"pre_llm_call": [{"command": cmd}]},
        "hooks_auto_accept": True,
    }

    reg_a = shell_hooks.register_from_config(cfg, accept_hooks=True, profile=str(home_a))
    reg_b = shell_hooks.register_from_config(cfg, accept_hooks=True, profile=str(home_b))
    assert len(reg_a) == 1
    assert len(reg_b) == 1
    # Second pass is idempotent per profile
    assert shell_hooks.register_from_config(cfg, accept_hooks=True, profile=str(home_a)) == []

    callbacks = list(manager._hooks.get("pre_llm_call", []))
    assert len(callbacks) == 2

    spawned = []

    def fake_spawn(spec, stdin_json):
        spawned.append(shell_hooks._resolved_profile_key())
        return {
            "returncode": 0,
            "stdout": "{}",
            "stderr": "",
            "timed_out": False,
            "error": None,
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(shell_hooks, "_spawn", fake_spawn)

    token = set_hermes_home_override(str(home_a))
    try:
        for cb in callbacks:
            cb(session_id="s")
    finally:
        reset_hermes_home_override(token)

    assert spawned == [str(home_a.resolve())]

    spawned.clear()
    token = set_hermes_home_override(str(home_b))
    try:
        for cb in callbacks:
            cb(session_id="s")
    finally:
        reset_hermes_home_override(token)

    assert spawned == [str(home_b.resolve())]
