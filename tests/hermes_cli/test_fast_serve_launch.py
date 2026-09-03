from __future__ import annotations

import argparse
import sys

import hermes_cli.config as config_mod
import hermes_cli.main as main_mod
from hermes_cli.subcommands.dashboard import build_dashboard_parser, build_serve_parser


def _capture(_args) -> None:
    return None


def test_lean_serve_parser_matches_full_subcommand_parser() -> None:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    build_dashboard_parser(subparsers, cmd_dashboard=_capture, cmd_dashboard_register=_capture)
    lean = build_serve_parser(cmd_dashboard=_capture)

    argv = [
        "--host", "127.0.0.1", "--port", "0", "--no-open",
        "--ssh-session-token-file", "token.txt", "--ssh-owner-nonce", "0123456789abcdef",
    ]

    assert vars(lean.parse_args(argv)) == vars(root.parse_args(["serve", *argv]))


def test_fast_serve_launch_dispatches_only_unambiguous_serve(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setattr(main_mod, "cmd_dashboard", captured.append)

    monkeypatch.setattr(sys, "argv", ["hermes", "serve", "--host", "127.0.0.1", "--port", "0"])
    assert main_mod._try_fast_serve_launch() is True
    assert (captured[0].command, captured[0].headless_backend, captured[0].no_open, captured[0].port) == (
        "serve", True, True, 0,
    )

    # Every ambiguous shape falls back to the full parser: unknown flags,
    # help, the opt-out, and container routing.
    for argv in (["serve", "--future-flag"], ["serve", "--help"], ["chat"]):
        monkeypatch.setattr(sys, "argv", ["hermes", *argv])
        assert main_mod._try_fast_serve_launch() is False
    monkeypatch.setenv("HERMES_DISABLE_FAST_SERVE_LAUNCH", "1")
    monkeypatch.setattr(sys, "argv", ["hermes", "serve"])
    assert main_mod._try_fast_serve_launch() is False
    monkeypatch.delenv("HERMES_DISABLE_FAST_SERVE_LAUNCH")
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: {"name": "managed"})
    assert main_mod._try_fast_serve_launch() is False
    assert len(captured) == 1


def test_fast_serve_launch_registers_config_hooks(monkeypatch) -> None:
    import agent.outbound_webhooks as outbound_webhooks
    import agent.shell_hooks as shell_hooks
    import hermes_cli.mcp_startup as mcp_startup
    import hermes_cli.plugins as plugins
    import hermes_cli.profiles as profiles
    import hermes_cli.resource_limits as resource_limits
    import hermes_cli.web_server as web_server

    config = {"hooks": {"pre_tool_call": [{"command": "guard"}]}}
    registrations = []

    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "0")
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setattr(config_mod, "require_parseable_user_config", lambda **_kwargs: None)
    monkeypatch.setattr(config_mod, "load_config", lambda: config)
    monkeypatch.setattr(config_mod, "apply_terminal_config_to_env", lambda: None)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(resource_limits, "apply_nofile_soft_limit", lambda: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setattr(main_mod, "_maybe_setup_dashboard_auth_interactively", lambda _args: None)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **_kwargs: None)
    monkeypatch.setattr(web_server, "start_server", lambda **_kwargs: None)
    monkeypatch.setattr(
        shell_hooks,
        "register_from_config",
        lambda cfg, **kwargs: registrations.append(("shell", cfg, kwargs)),
    )
    monkeypatch.setattr(
        outbound_webhooks,
        "register_from_config",
        lambda cfg: registrations.append(("outbound", cfg, {})),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "serve", "--host", "127.0.0.1", "--port", "0"],
    )

    assert main_mod._try_fast_serve_launch() is True
    assert registrations == [
        ("shell", config, {"accept_hooks": False}),
        ("outbound", config, {}),
    ]
