"""Regression tests for #69825: ``hermes serve`` / ``hermes dashboard`` must
register declarative shell hooks from config.yaml.

``serve`` (the desktop app's chat backend) is not in ``_AGENT_COMMANDS``, so
``_prepare_agent_startup()`` never runs for it.  Before the fix, configured
shell hooks passed ``hermes hooks doctor`` but silently never fired during
desktop-app chat activity because ``agent.shell_hooks.register_from_config()``
was never called in the serve process.  ``cmd_dashboard`` (the shared handler
for ``dashboard`` and ``serve``) now mirrors the gateway's call site.
"""

import sys
import types
from argparse import Namespace

import pytest

import hermes_cli.main as main_mod


def _serve_args(**overrides):
    base = {
        "ssh_session_token_file": None,
        "ssh_owner_nonce": None,
        "status": False,
        "stop": False,
        "headless_backend": True,  # the `serve` path: no UI build
        "isolated": False,
        "open_profile": "",
        "host": "127.0.0.1",
        "port": 9119,
        "no_open": True,
        "insecure": False,
        "skip_build": False,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture()
def _stubbed_dashboard_env(monkeypatch):
    """Stub every side-effecting collaborator of ``cmd_dashboard`` and record
    the calls that matter for the shell-hook registration contract."""
    calls = {"order": [], "register_kwargs": None, "register_cfg": None}
    sentinel_cfg = {"hooks": {"pre_tool_call": [{"command": "/tmp/hook.sh"}]}}

    # Touch the env var through monkeypatch first so its mutation inside
    # cmd_dashboard (headless sets HERMES_SERVE_HEADLESS=1) is restored.
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "0")

    def _register_from_config(cfg, *, accept_hooks=False):
        calls["order"].append("register_from_config")
        calls["register_cfg"] = cfg
        calls["register_kwargs"] = {"accept_hooks": accept_hooks}
        return []

    def _discover_plugins(*a, **k):
        calls["order"].append("discover_plugins")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.profiles",
        types.SimpleNamespace(get_active_profile_name=lambda: "default"),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=_discover_plugins),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        types.SimpleNamespace(
            start_background_mcp_discovery=lambda **_k: calls["order"].append(
                "mcp_discovery"
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            load_config=lambda: sentinel_cfg,
            apply_terminal_config_to_env=lambda: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=_register_from_config),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(
            start_server=lambda **_k: calls["order"].append("start_server"),
        ),
    )
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setattr(
        main_mod, "_maybe_setup_dashboard_auth_interactively", lambda _a: None
    )
    return calls, sentinel_cfg


def test_serve_registers_shell_hooks_before_start_server(_stubbed_dashboard_env):
    calls, sentinel_cfg = _stubbed_dashboard_env

    main_mod.cmd_dashboard(_serve_args())

    assert "register_from_config" in calls["order"], (
        "serve/dashboard startup must register shell hooks — configured hooks "
        "otherwise silently never fire in desktop-app chats (#69825)"
    )
    # The loaded config dict is passed through untouched, and consent
    # resolution is delegated to register_from_config (mirrors gateway).
    assert calls["register_cfg"] is sentinel_cfg
    assert calls["register_kwargs"] == {"accept_hooks": False}
    # Registration attaches hooks to the plugin manager, so plugin discovery
    # must have happened first; and the server boots afterwards.
    order = calls["order"]
    assert order.index("discover_plugins") < order.index("register_from_config")
    assert order.index("register_from_config") < order.index("start_server")


def test_serve_hook_registration_failure_does_not_block_startup(
    _stubbed_dashboard_env, monkeypatch
):
    calls, _ = _stubbed_dashboard_env

    def _boom(_cfg, *, accept_hooks=False):
        calls["order"].append("register_from_config")
        raise RuntimeError("registration exploded")

    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=_boom),
    )

    main_mod.cmd_dashboard(_serve_args())

    # The failure is swallowed (logged) and the server still starts.
    assert "start_server" in calls["order"]
