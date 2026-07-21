"""Contract for the headless ``hermes serve`` backend command.

``serve`` is what the desktop app and remote backends launch — the same gateway
as ``dashboard`` (shared handler) but always headless, and decoupled in name so
the desktop never invokes ``dashboard``. These tests pin that contract:

- ``serve`` routes to the same handler as ``dashboard``;
- ``serve`` is headless by default, ``dashboard`` is not;
- both expose the identical server-runtime flag surface.
"""

from __future__ import annotations

import argparse
import json

import pytest
import uvicorn

from hermes_cli import serve_restart_marker, web_server
from hermes_cli.subcommands.dashboard import build_dashboard_parser


def _dash(args):  # sentinel handler — identity-compared, never invoked
    return args


def _register(args):
    return args


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    build_dashboard_parser(
        parser.add_subparsers(dest="command"),
        cmd_dashboard=_dash,
        cmd_dashboard_register=_register,
    )
    return parser


def test_serve_routes_to_the_shared_dashboard_handler():
    args = _parser().parse_args(["serve"])
    assert args.func is _dash


def test_serve_is_headless_by_default_but_dashboard_is_not():
    assert _parser().parse_args(["serve"]).no_open is True
    assert _parser().parse_args(["dashboard"]).no_open is False


def test_serve_accepts_the_legacy_no_open_flag_as_a_noop():
    # The desktop backend spawn (and old shells) may still pass --no-open;
    # serve must tolerate it rather than erroring on an unknown argument.
    assert _parser().parse_args(["serve", "--no-open"]).no_open is True


def test_serve_takes_the_same_runtime_flags_as_dashboard():
    argv = ["--host", "0.0.0.0", "--port", "0", "--insecure", "--skip-build", "--isolated"]
    serve = _parser().parse_args(["serve", *argv])
    dash = _parser().parse_args(["dashboard", *argv])
    for field in ("host", "port", "insecure", "skip_build", "isolated"):
        assert getattr(serve, field) == getattr(dash, field)


def test_serve_supports_the_lifecycle_flags():
    for flag in ("--stop", "--status"):
        assert getattr(_parser().parse_args(["serve", flag]), flag.lstrip("-")) is True


def test_serve_is_a_headless_backend_but_dashboard_is_not():
    # `headless_backend` is the flag cmd_dashboard reads to skip the web UI
    # build; only `serve` carries it.
    assert getattr(_parser().parse_args(["serve"]), "headless_backend", False) is True
    assert getattr(_parser().parse_args(["dashboard"]), "headless_backend", False) is False


@pytest.fixture
def restart_marker_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        serve_restart_marker,
        "get_hermes_home",
        lambda: tmp_path,
    )
    return tmp_path / "runtime" / "serve_restart.json"


class TestServeRestartMarker:
    def test_matching_fresh_marker_is_consumed(self, restart_marker_path):
        serve_restart_marker.write_restart_markers([12345])

        assert serve_restart_marker.consume_restart_marker(12345) is True
        assert not restart_marker_path.exists()

    def test_matching_pid_is_removed_without_dropping_others(
        self, restart_marker_path
    ):
        serve_restart_marker.write_restart_markers([12345, 54321])

        assert serve_restart_marker.consume_restart_marker(12345) is True
        marker = json.loads(restart_marker_path.read_text(encoding="utf-8"))
        assert marker["pids"] == [54321]
        assert isinstance(marker["written_at"], float)

    def test_pid_mismatch_cleans_marker(self, restart_marker_path):
        serve_restart_marker.write_restart_markers([12345])

        assert serve_restart_marker.consume_restart_marker(54321) is False
        assert not restart_marker_path.exists()

    def test_expired_marker_is_cleaned(self, monkeypatch, restart_marker_path):
        restart_marker_path.parent.mkdir(parents=True)
        restart_marker_path.write_text(
            json.dumps({"pids": [12345], "written_at": 100.0}),
            encoding="utf-8",
        )
        monkeypatch.setattr(serve_restart_marker.time, "time", lambda: 701.0)

        assert serve_restart_marker.consume_restart_marker(12345) is False
        assert not restart_marker_path.exists()

    def test_missing_marker_returns_false(self, restart_marker_path):
        assert serve_restart_marker.consume_restart_marker(12345) is False


def _stub_start_server(monkeypatch):
    class _FakeConfig:
        loaded = True

        def __init__(self, *args, **kwargs):
            pass

    class _FakeServer:
        def __init__(self, _config):
            pass

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(web_server.sys, "platform", "linux")
    monkeypatch.setattr(
        web_server.asyncio,
        "run",
        lambda coro: coro.close(),
    )


class TestServeRestartExit:
    def test_marker_exits_with_restart_code(self, monkeypatch):
        _stub_start_server(monkeypatch)
        calls = iter([False, True])
        monkeypatch.setattr(
            web_server,
            "consume_restart_marker",
            lambda _pid: next(calls),
        )
        with pytest.raises(SystemExit) as exc_info:
            web_server.start_server(open_browser=False)

        assert exc_info.value.code == serve_restart_marker.RESTART_EXIT_CODE

    def test_no_marker_returns_normally(self, monkeypatch):
        _stub_start_server(monkeypatch)
        monkeypatch.setattr(
            web_server,
            "consume_restart_marker",
            lambda _pid: False,
        )

        assert web_server.start_server(open_browser=False) is None
