from __future__ import annotations

import argparse
import sys

import hermes_cli.config as config_mod
import hermes_cli.main as main_mod
from hermes_cli.subcommands.dashboard import build_dashboard_parser, build_serve_parser


def _capture(_args) -> None:
    return None


def test_standalone_serve_parser_matches_full_subcommand_parser() -> None:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    build_dashboard_parser(
        subparsers,
        cmd_dashboard=_capture,
        cmd_dashboard_register=_capture,
    )
    lean = build_serve_parser(cmd_dashboard=_capture)

    argv = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--no-open",
        "--ssh-session-token-file",
        "token.txt",
        "--ssh-owner-nonce",
        "0123456789abcdef",
    ]

    assert vars(lean.parse_args(argv)) == vars(root.parse_args(["serve", *argv]))


def test_fast_serve_launch_dispatches_canonical_arguments(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setattr(main_mod, "cmd_dashboard", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ssh-owner-nonce",
            "0123456789abcdef",
        ],
    )

    assert main_mod._try_fast_serve_launch() is True
    assert len(captured) == 1
    assert captured[0].command == "serve"
    assert captured[0].headless_backend is True
    assert captured[0].no_open is True
    assert captured[0].host == "127.0.0.1"
    assert captured[0].port == 0
    assert captured[0].ssh_owner_nonce == "0123456789abcdef"


def test_fast_serve_launch_falls_back_for_unknown_arguments(monkeypatch) -> None:
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hermes", "serve", "--future-flag"])

    assert main_mod._try_fast_serve_launch() is False


def test_fast_serve_launch_preserves_container_routing(monkeypatch) -> None:
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: {"name": "managed"})
    monkeypatch.setattr(sys, "argv", ["hermes", "serve"])

    assert main_mod._try_fast_serve_launch() is False


def test_fast_serve_launch_preserves_help_and_opt_out(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hermes", "serve", "--help"])
    assert main_mod._try_fast_serve_launch() is False

    monkeypatch.setenv("HERMES_DISABLE_FAST_SERVE_LAUNCH", "1")
    monkeypatch.setattr(sys, "argv", ["hermes", "serve"])
    assert main_mod._try_fast_serve_launch() is False
