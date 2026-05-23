"""Tests for the `hermes harness` command."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace

import pytest

from hermes_cli import harness as harness_mod


def test_ensure_harness_running_respects_env_opt_out(monkeypatch):
    calls = []

    monkeypatch.setenv("HYPURA_HARNESS_AUTO_START", "0")
    monkeypatch.setattr(
        harness_mod,
        "_harness_config",
        lambda: {"enabled": True, "auto_start": True},
    )
    monkeypatch.setattr(
        harness_mod, "is_harness_running", lambda: calls.append("probe") or False
    )
    monkeypatch.setattr(
        harness_mod,
        "start_harness_daemon",
        lambda: calls.append("start") or True,
    )

    harness_mod.ensure_harness_running()

    assert calls == []


def test_ensure_harness_running_starts_when_enabled(monkeypatch):
    calls = []

    monkeypatch.delenv("HYPURA_HARNESS_AUTO_START", raising=False)
    monkeypatch.delenv("HERMES_HARNESS_AUTO_START", raising=False)
    monkeypatch.setattr(
        harness_mod,
        "_harness_config",
        lambda: {"enabled": True, "auto_start": True},
    )
    monkeypatch.setattr(
        harness_mod, "is_harness_running", lambda: calls.append("probe") or False
    )
    monkeypatch.setattr(
        harness_mod,
        "start_harness_daemon",
        lambda: calls.append("start") or True,
    )

    harness_mod.ensure_harness_running()

    assert calls == ["probe", "start"]


def test_ensure_harness_running_respects_config_auto_start(monkeypatch):
    calls = []

    monkeypatch.delenv("HYPURA_HARNESS_AUTO_START", raising=False)
    monkeypatch.delenv("HERMES_HARNESS_AUTO_START", raising=False)
    monkeypatch.setattr(
        harness_mod,
        "_harness_config",
        lambda: {"enabled": True, "auto_start": False},
    )
    monkeypatch.setattr(
        harness_mod, "is_harness_running", lambda: calls.append("probe") or False
    )
    monkeypatch.setattr(
        harness_mod,
        "start_harness_daemon",
        lambda: calls.append("start") or True,
    )

    harness_mod.ensure_harness_running()

    assert calls == []


def test_get_harness_url_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("HYPURA_HARNESS_HOST", "localhost")
    monkeypatch.setenv("HYPURA_HARNESS_PORT", "19001")

    assert harness_mod.get_harness_url() == "http://localhost:19001"


def test_get_harness_url_falls_back_on_invalid_port(monkeypatch):
    monkeypatch.setenv("HYPURA_HARNESS_PORT", "not-a-port")

    assert harness_mod.get_harness_url() == "http://127.0.0.1:18794"


def test_is_harness_running_uses_lightweight_health(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            calls.append((url, self.timeout))
            return FakeResponse()

    monkeypatch.setattr(harness_mod, "get_harness_url", lambda: "http://127.0.0.1:18794")
    monkeypatch.setattr(harness_mod.httpx, "Client", FakeClient)

    assert harness_mod.is_harness_running(timeout=0.25) is True
    assert calls == [("http://127.0.0.1:18794/health", 0.25)]


def test_is_harness_running_falls_back_to_legacy_status(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            calls.append((url, self.timeout))
            if url.endswith("/health"):
                return FakeResponse(404)
            return FakeResponse(200)

    monkeypatch.setattr(harness_mod, "get_harness_url", lambda: "http://127.0.0.1:18794")
    monkeypatch.setattr(harness_mod.httpx, "Client", FakeClient)

    assert harness_mod.is_harness_running(timeout=0.25) is True
    assert calls == [
        ("http://127.0.0.1:18794/health", 0.25),
        ("http://127.0.0.1:18794/status", harness_mod.LEGACY_STATUS_TIMEOUT),
    ]


def test_register_harness_subparser_wires_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    harness_mod.register_harness_subparser(subparsers)

    args = parser.parse_args(["harness", "restart"])

    assert args.command == "harness"
    assert args.harness_action == "restart"
    assert args.func is harness_mod._run_harness_command


def test_status_reports_offline_and_missing_script(monkeypatch, tmp_path, capsys):
    missing_script = tmp_path / "missing_harness_daemon.py"
    monkeypatch.setattr(harness_mod, "is_harness_running", lambda: False)
    monkeypatch.setattr(
        harness_mod, "get_harness_url", lambda: "http://127.0.0.1:18794"
    )
    monkeypatch.setattr(harness_mod, "get_harness_script_path", lambda: missing_script)

    rc = harness_mod.harness_command(Namespace(harness_action="status"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "OFFLINE" in captured.out
    assert str(missing_script) in captured.out


def test_start_refuses_missing_script(monkeypatch, tmp_path):
    monkeypatch.setattr(harness_mod, "is_harness_running", lambda: False)
    monkeypatch.setattr(
        harness_mod,
        "get_harness_script_path",
        lambda: tmp_path / "missing_harness_daemon.py",
    )

    assert harness_mod.start_harness_daemon(wait_seconds=0) is False


def test_stop_terminates_process_when_status_endpoint_is_unhealthy(monkeypatch):
    calls = []

    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    class ZombieProcess(Exception):
        pass

    class FakeProc:
        pid = 12345

        def __init__(self):
            self._running = True

        def net_connections(self, kind):
            assert kind == "inet"
            return [Namespace(laddr=Namespace(port=18794))]

        def terminate(self):
            calls.append("terminate")
            self._running = False

        def kill(self):
            calls.append("kill")
            self._running = False

        def is_running(self):
            return self._running

    proc = FakeProc()
    fake_psutil = Namespace(
        process_iter=lambda _attrs: [proc],
        NoSuchProcess=NoSuchProcess,
        AccessDenied=AccessDenied,
        ZombieProcess=ZombieProcess,
    )

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(harness_mod, "is_harness_running", lambda: False)

    assert harness_mod.stop_harness_daemon(timeout=0.1) is True
    assert calls == ["terminate"]


def test_main_accepts_harness_status(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod

    missing_script = tmp_path / "missing_harness_daemon.py"
    monkeypatch.setattr(sys, "argv", ["hermes", "harness", "status"])
    monkeypatch.setattr(harness_mod, "is_harness_running", lambda: False)
    monkeypatch.setattr(
        harness_mod, "get_harness_url", lambda: "http://127.0.0.1:18794"
    )
    monkeypatch.setattr(harness_mod, "get_harness_script_path", lambda: missing_script)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Hypura Harness: OFFLINE" in captured.out
    assert "invalid choice" not in captured.err
