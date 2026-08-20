"""The dashboard/serve startup path must register shell hooks.

``serve``/``dashboard`` are not in ``_AGENT_COMMANDS``, so the CLI-side
``register_from_config`` in ``_prepare_agent_startup`` never runs for the
desktop backend — without an explicit registration here the desktop app
silently runs WITHOUT pre_tool_call hooks (budget gate, danger, memory
routing, skill gate). ``cmd_dashboard`` registers hooks itself, mirroring
``gateway/run.py``; these tests pin that contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import hermes_cli.main as main_mod


def _dashboard_args(**overrides) -> SimpleNamespace:
    base = dict(
        status=False,
        stop=False,
        ssh_session_token_file=None,
        headless_backend=True,
        isolated=True,
        open_profile="",
        host=None,
        port=0,
        no_open=True,
        insecure=False,
        skip_build=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stub_dashboard_side_effects(monkeypatch):
    """Neutralize everything after hook registration except start_server."""
    # Profile routing must fall through to "default" (no re-exec branch).
    def _no_profile():
        raise RuntimeError("no profile in test")

    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", _no_profile)
    # No real web UI build.
    monkeypatch.setattr(main_mod, "_build_web_ui", lambda *a, **k: True)
    monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
    # No interactive auth prompt.
    monkeypatch.setattr(
        main_mod, "_maybe_setup_dashboard_auth_interactively", lambda args: None
    )
    # Capture the eventual server start instead of binding a port.
    started = {}
    monkeypatch.setattr(
        "hermes_cli.web_server.start_server", lambda **kwargs: started.update(kwargs)
    )
    return started


def test_cmd_dashboard_registers_shell_hooks(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda cfg, accept_hooks=False: calls.append((cfg, accept_hooks)),
    )
    _stub_dashboard_side_effects(monkeypatch)

    main_mod.cmd_dashboard(_dashboard_args())

    assert len(calls) == 1, f"expected one register_from_config call, got {calls}"
    _cfg, accept_hooks = calls[0]
    # Non-TTY backend: allowlisted hooks register, consent prompts never fire.
    assert accept_hooks is False
    assert _cfg is not None


def test_cmd_dashboard_hook_registration_never_blocks_startup(monkeypatch):
    """A failing register_from_config must not prevent the server from starting."""
    def _explode(cfg, accept_hooks=False):
        raise RuntimeError("hooks exploded")

    monkeypatch.setattr("agent.shell_hooks.register_from_config", _explode)
    started = _stub_dashboard_side_effects(monkeypatch)

    # Must not raise: hook failure is logged (debug) and startup continues.
    main_mod.cmd_dashboard(_dashboard_args())

    assert started, "start_server must still be reached"
