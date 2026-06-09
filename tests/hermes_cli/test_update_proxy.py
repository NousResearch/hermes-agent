"""Tests for ``hermes update --proxy`` forwarding."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace


def test_update_parser_accepts_proxy_flag():
    from hermes_cli.subcommands.update import build_update_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    def _handler(args):  # pragma: no cover - identity only
        return args

    build_update_parser(subparsers, cmd_update=_handler)

    args = parser.parse_args(["update", "--proxy", "http://corp.proxy:8080"])

    assert args.command == "update"
    assert args.proxy == "http://corp.proxy:8080"
    assert args.func is _handler


def test_apply_update_proxy_env_sets_standard_proxy_vars(monkeypatch):
    from hermes_cli.main import _apply_update_proxy_env

    for key in (
        "HERMES_UPDATE_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    proxy = _apply_update_proxy_env(
        SimpleNamespace(proxy=" http://corp.proxy:8080 ")
    )

    assert proxy == "http://corp.proxy:8080"
    for key in (
        "HERMES_UPDATE_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        assert os.environ[key] == "http://corp.proxy:8080"


def test_apply_update_proxy_env_ignores_missing_proxy(monkeypatch):
    from hermes_cli.main import _apply_update_proxy_env

    monkeypatch.delenv("HERMES_UPDATE_PROXY", raising=False)

    assert _apply_update_proxy_env(SimpleNamespace(proxy=None)) is None
    assert "HERMES_UPDATE_PROXY" not in os.environ


def test_update_proxy_guidance_without_proxy_points_to_flag(monkeypatch, capsys):
    from hermes_cli.main import _print_update_proxy_guidance

    for key in (
        "HERMES_UPDATE_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    _print_update_proxy_guidance()

    out = capsys.readouterr().out
    assert "hermes update --proxy http://proxy-host:port" in out
    assert "HTTPS_PROXY/HTTP_PROXY" in out


def test_update_proxy_guidance_with_proxy_asks_to_verify(monkeypatch, capsys):
    from hermes_cli.main import _print_update_proxy_guidance

    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:8080")

    _print_update_proxy_guidance()

    out = capsys.readouterr().out
    assert "A proxy is configured" in out
    assert "Verify the proxy URL" in out
