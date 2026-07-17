"""Behavior contracts for selecting one cron scheduler owner per Hermes home."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


class _RecordingThread:
    instances = []

    def __init__(self, *, target, args=(), kwargs=None, daemon=None, name=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.name = name
        self.started = False
        type(self).instances.append(self)

    def start(self):
        self.started = True


def _write_config(home, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "gateway"),
        ({"cron": {}}, "gateway"),
        ({"cron": {"scheduler_owner": "auto"}}, "gateway"),
        ({"cron": {"scheduler_owner": " gateway "}}, "gateway"),
        ({"cron": {"scheduler_owner": "DESKTOP"}}, "desktop"),
    ],
)
def test_scheduler_owner_selection_contract(config, expected):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    assert resolve_cron_scheduler_owner(config=config) == expected


@pytest.mark.parametrize("invalid", ["", "both", 42, None])
def test_invalid_scheduler_owner_fails_closed_without_logging_value(invalid, caplog):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    with caplog.at_level("ERROR"):
        assert resolve_cron_scheduler_owner(
            config={"cron": {"scheduler_owner": invalid}}
        ) is None
    assert "scheduler startup disabled" in caplog.text
    if isinstance(invalid, str) and invalid:
        assert invalid not in caplog.text


def test_default_config_uses_safe_auto_owner():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["cron"]["scheduler_owner"] == "auto"


@pytest.mark.parametrize(
    "body",
    [
        'cron:\n  scheduler_owner: "unterminated\n',
        "cron: desktop\n",
        "cron:\n  scheduler_owner: both\n",
        "null\n",
    ],
)
def test_owner_file_errors_fail_closed(tmp_path, monkeypatch, body):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    _write_config(tmp_path, body)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert resolve_cron_scheduler_owner() is None


def test_named_profile_reads_its_registry_not_default_registry(tmp_path, monkeypatch):
    from cron.scheduler_provider import should_start_cron_scheduler

    default_home = tmp_path / ".hermes"
    profile_home = default_home / "profiles" / "worker"
    _write_config(default_home, "cron:\n  scheduler_owner: gateway\n")
    _write_config(profile_home, "cron:\n  scheduler_owner: desktop\n")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    assert should_start_cron_scheduler("gateway") is False
    assert should_start_cron_scheduler("desktop") is True


def test_gateway_and_desktop_use_same_owner_gate(tmp_path, monkeypatch):
    from gateway import run as gateway_run
    from hermes_cli import web_server

    _write_config(tmp_path, "cron:\n  scheduler_owner: desktop\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setattr(gateway_run.threading, "Thread", _RecordingThread)
    monkeypatch.setattr(web_server.threading, "Thread", _RecordingThread)
    _RecordingThread.instances.clear()
    runner = SimpleNamespace(
        adapters={},
        _draining=False,
        _external_drain_active=False,
    )

    gateway_provider, gateway_thread = gateway_run._start_gateway_cron_scheduler_if_owned(
        runner, threading.Event(), object()
    )
    desktop_stop, desktop_thread = web_server._start_desktop_cron_scheduler_if_owned()

    assert gateway_provider is None
    assert gateway_thread is None
    assert isinstance(desktop_stop, threading.Event)
    assert desktop_thread is _RecordingThread.instances[0]
    assert desktop_thread.started is True
    assert desktop_thread.kwargs["provider"].name == "builtin"


def test_default_owner_starts_gateway_not_desktop(tmp_path, monkeypatch):
    from gateway import run as gateway_run
    from hermes_cli import web_server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setattr(gateway_run.threading, "Thread", _RecordingThread)
    monkeypatch.setattr(web_server.threading, "Thread", _RecordingThread)
    _RecordingThread.instances.clear()
    runner = SimpleNamespace(
        adapters={"discord": object()},
        _draining=False,
        _external_drain_active=False,
    )

    gateway_provider, gateway_thread = gateway_run._start_gateway_cron_scheduler_if_owned(
        runner, threading.Event(), object()
    )
    desktop_stop, desktop_thread = web_server._start_desktop_cron_scheduler_if_owned()

    assert gateway_provider.name == "builtin"
    assert gateway_thread is _RecordingThread.instances[0]
    assert gateway_thread.started is True
    assert callable(gateway_thread.kwargs["can_dispatch"])
    assert desktop_stop is None
    assert desktop_thread is None
