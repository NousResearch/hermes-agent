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
