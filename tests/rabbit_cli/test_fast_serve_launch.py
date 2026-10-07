from __future__ import annotations

import argparse
import sys

import rabbit_cli.config as config_mod
import rabbit_cli.main as main_mod
from rabbit_cli.subcommands.dashboard import build_dashboard_parser, build_serve_parser


def _capture(_args) -> None:
    return None


def test_lean_serve_parser_matches_full_subcommand_parser() -> None:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    build_dashboard_parser(subparsers, cmd_dashboard=_capture)
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

    monkeypatch.setattr(sys, "argv", ["rabbit", "serve", "--host", "127.0.0.1", "--port", "0"])
    assert main_mod._try_fast_serve_launch() is True
    assert (captured[0].command, captured[0].headless_backend, captured[0].no_open, captured[0].port) == (
        "serve", True, True, 0,
    )

    # Every ambiguous shape falls back to the full parser: unknown flags,
    # help, the opt-out, and container routing.
    for argv in (["serve", "--future-flag"], ["serve", "--help"], ["chat"]):
        monkeypatch.setattr(sys, "argv", ["rabbit", *argv])
        assert main_mod._try_fast_serve_launch() is False
    monkeypatch.setenv("RABBIT_DISABLE_FAST_SERVE_LAUNCH", "1")
    monkeypatch.setattr(sys, "argv", ["rabbit", "serve"])
    assert main_mod._try_fast_serve_launch() is False
    monkeypatch.delenv("RABBIT_DISABLE_FAST_SERVE_LAUNCH")
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: {"name": "managed"})
    assert main_mod._try_fast_serve_launch() is False
    assert len(captured) == 1
