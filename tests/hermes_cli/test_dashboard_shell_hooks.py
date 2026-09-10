"""serve/dashboard must register shell hooks so lifecycle hooks (mnemon
prime/remind/nudge, …) fire on the desktop/headless agent host.

Regression: the shell-hook registration in ``_prepare_agent_startup`` is gated
on ``_AGENT_COMMANDS = {None, chat, acp, rl}`` (+ cron/gateway/mcp subcommands),
which excludes ``serve``/``dashboard``. The desktop app runs agent turns through
serve's in-process ``/api/ws`` gateway, so hooks were never registered there and
lifecycle hooks silently stopped firing after an upgrade dropped the earlier
local patch.
"""

from __future__ import annotations

import types

import pytest


def _args(**kw):
    defaults = dict(
        status=False,
        stop=False,
        host="127.0.0.1",
        port=9119,
        no_open=True,
        insecure=False,
        skip_build=False,
        isolated=False,
        open_profile="",
        headless_backend=True,
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


@pytest.fixture
def main_mod():
    import hermes_cli.main as main_mod

    return main_mod


def _neutralize_startup(main_mod, monkeypatch):
    """Mock the expensive/blocking cmd_dashboard steps so the test reaches the
    hook-registration call without a real web server, web build, or config."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr(
        "hermes_cli.config.require_parseable_user_config", lambda **kw: None
    )
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery", lambda **kw: None
    )
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(web_server, "start_server", lambda **kw: None)
    monkeypatch.setattr(
        main_mod, "_maybe_setup_dashboard_auth_interactively", lambda args: None
    )


def test_serve_registers_shell_hooks(main_mod, monkeypatch):
    """Headless serve must register hooks before starting the server."""
    _neutralize_startup(main_mod, monkeypatch)

    calls = []
    monkeypatch.setattr(
        main_mod,
        "_register_shell_hooks",
        lambda accept_hooks=False: calls.append(accept_hooks),
    )

    main_mod.cmd_dashboard(_args())

    assert calls == [False]  # headless: never auto-accept hooks


def test_register_shell_hooks_wires_shell_and_outbound(monkeypatch):
    """The shared helper forwards to both shell-hook and outbound-webhook
    registrars with the caller's accept_hooks flag."""
    import hermes_cli.main as main_mod

    shell_calls = []
    outbound_calls = []

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"hooks": {}})
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda cfg, accept_hooks=False: shell_calls.append((cfg, accept_hooks)),
    )
    monkeypatch.setattr(
        "agent.outbound_webhooks.register_from_config",
        lambda cfg: outbound_calls.append(cfg),
    )

    main_mod._register_shell_hooks(accept_hooks=True)

    assert shell_calls == [({"hooks": {}}, True)]
    assert outbound_calls == [{"hooks": {}}]
