from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hermes_cli import kanban
from hermes_cli import kanban_remote
from hermes_cli import kanban_remote_worker


def test_configured_coordinator_url_prefers_config_over_legacy_environment(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_KANBAN_COORDINATOR_URL", "http://legacy:8788")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"coordinator_url": "http://configured:8788/"}},
    )

    assert kanban_remote.configured_coordinator_url() == "http://configured:8788"


def test_configured_coordinator_url_keeps_legacy_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_COORDINATOR_URL", "http://legacy:8788/")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})

    assert kanban_remote.configured_coordinator_url() == "http://legacy:8788"


def test_remote_worker_retries_coordinator_with_bounded_backoff(monkeypatch):
    class StopLoop(Exception):
        pass

    class Client:
        attempts = 0

        def register_machine(self, *_args, **_kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary outage")

        def claim_next(self, _machine_id):
            raise StopLoop()

    sleeps: list[float] = []
    monkeypatch.setattr(kanban_remote_worker.kb, "get_machine_id", lambda: "machine-id")

    with pytest.raises(StopLoop):
        kanban_remote_worker.run_worker_loop(
            Client(),
            profile="default",
            capabilities=["linux"],
            poll_seconds=2,
            max_retry_seconds=5,
            sleep_fn=sleeps.append,
        )

    assert sleeps == [2]


def test_remote_worker_fences_child_when_renewal_is_rejected(monkeypatch):
    class Process:
        pid = 1234
        returncode = None
        terminated = False
        waited = False

        def poll(self):
            return 1 if self.terminated else None

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            self.waited = True
            return self.returncode

        def kill(self):
            raise AssertionError("graceful termination should have reaped child")

    process = Process()

    class Client:
        base_url = "http://coordinator"
        token = "secret"

        def build_worker_context(self, _conn, _task_id):
            return "context"

        def record_worker_started(self, *_args, **_kwargs):
            return True

        def heartbeat_claim(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(
        kanban_remote_worker.subprocess, "Popen", lambda *_a, **_kw: process
    )
    monkeypatch.setattr(
        kanban_remote_worker.kb, "_resolve_hermes_argv", lambda: ["hermes"]
    )

    kanban_remote_worker._run_task(
        Client(),
        SimpleNamespace(id="t_remote", current_run_id=1, claim_lock="lock"),
        profile="ios",
        machine_id="machine-id",
    )

    assert process.terminated is True
    assert process.waited is True


def _parse_kanban(argv: list[str]):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    kanban.build_parser(subparsers)
    return parser.parse_args(["kanban", *argv])


def test_remote_worker_cli_uses_configured_endpoint(monkeypatch):
    called = {}
    monkeypatch.setattr(
        kanban_remote,
        "configured_coordinator_url",
        lambda: "http://coordinator:8788",
    )
    monkeypatch.setattr(
        kanban_remote_worker,
        "run",
        lambda **kwargs: called.update(kwargs) or 7,
    )

    result = kanban.kanban_command(
        _parse_kanban(["remote-worker", "--profile", "ios", "--capability", "xcode"]),
    )

    assert result == 7
    assert called == {
        "url": "http://coordinator:8788",
        "token_env": "HERMES_KANBAN_COORDINATOR_TOKEN",
        "profile": "ios",
        "capabilities": ["xcode"],
        "poll_seconds": 10.0,
        "max_retry_seconds": 60.0,
    }


def test_coordinator_cli_defaults_to_current_board_database(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(kanban.kb, "kanban_db_path", lambda: tmp_path / "board.db")
    monkeypatch.setattr(
        "hermes_cli.kanban_coordinator.run",
        lambda **kwargs: called.update(kwargs) or 3,
    )

    result = kanban.kanban_command(_parse_kanban(["coordinator", "--port", "9999"]))

    assert result == 3
    assert called == {
        "db_path": tmp_path / "board.db",
        "host": "127.0.0.1",
        "port": 9999,
        "token_env": "HERMES_KANBAN_COORDINATOR_TOKEN",
    }
