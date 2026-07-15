from __future__ import annotations

import base64
from pathlib import Path

import pytest

from gateway import production_execution_readiness as readiness
from tools.browser_controller_client import BrowserControllerClientConfig


REVISION = "a" * 40
CONFIG_SHA256 = "b" * 64


class _WorkerClient:
    instances: list["_WorkerClient"] = []

    def __init__(self, socket_path, **kwargs):
        self.socket_path = socket_path
        self.kwargs = kwargs
        self.cancelled = []
        self.closed = False
        self.__class__.instances.append(self)

    def start(self, command, **kwargs):
        self.command = command
        self.start_kwargs = kwargs
        return "session-1"

    def poll(self, session_id, **_kwargs):
        assert session_id == "session-1"
        return {
            "state": "exited",
            "returncode": 0,
            "stdout_b64": base64.b64encode(
                b"MUNCHO_ISOLATED_WORKER_READY\n"
            ).decode(),
            "stderr_b64": "",
            "drained": True,
            "complete": True,
        }

    def cancel(self, session_id):
        self.cancelled.append(session_id)

    def close(self):
        self.closed = True


def test_worker_readiness_executes_real_fixed_round_trip(monkeypatch):
    _WorkerClient.instances.clear()
    monkeypatch.setattr(readiness, "IsolatedWorkerClient", _WorkerClient)

    receipt = readiness.attest_isolated_worker_execution(
        socket_path=Path("/run/muncho-isolated-worker/worker.sock"),
        server_uid=1001,
        server_gid=1002,
        socket_uid=0,
        socket_gid=1003,
        revision=REVISION,
        config_sha256=CONFIG_SHA256,
    )

    client = _WorkerClient.instances[0]
    assert client.command == "printf 'MUNCHO_ISOLATED_WORKER_READY\\n'"
    assert client.start_kwargs["cwd"] == Path("/workspace")
    assert receipt["execution_round_trip"] is True
    assert receipt["secret_material_recorded"] is False
    assert client.closed is True


def test_worker_readiness_fails_closed_and_cancels(monkeypatch):
    class BadWorker(_WorkerClient):
        def poll(self, *_args, **_kwargs):
            return {
                "state": "exited",
                "returncode": 0,
                "stdout_b64": base64.b64encode(b"wrong").decode(),
                "stderr_b64": "",
                "drained": True,
                "complete": True,
            }

    BadWorker.instances.clear()
    monkeypatch.setattr(readiness, "IsolatedWorkerClient", BadWorker)

    with pytest.raises(
        readiness.ProductionExecutionReadinessError,
        match="worker_readiness_output_invalid",
    ):
        readiness.attest_isolated_worker_execution(
            socket_path=Path("/run/muncho-isolated-worker/worker.sock"),
            server_uid=1001,
            server_gid=1002,
            socket_uid=0,
            socket_gid=1003,
            revision=REVISION,
            config_sha256=CONFIG_SHA256,
        )
    assert BadWorker.instances[0].cancelled == ["session-1"]
    assert BadWorker.instances[0].closed is True


class _BrowserClient:
    instances: list["_BrowserClient"] = []
    result = {"success": True, "data": {"result": "about:blank"}}

    def __init__(self, config, identity):
        self.config = config
        self.identity = identity
        self.closed = False
        self.__class__.instances.append(self)

    def command(self, command, args):
        self.command_value = (command, args)
        return dict(self.result)

    def close(self):
        self.closed = True


def _browser_config(tmp_path: Path) -> BrowserControllerClientConfig:
    return BrowserControllerClientConfig.from_mapping(
        {
            "schema": "hermes-browser-controller-client.v1",
            "socket_path": "/run/muncho-browser-controller/controller.sock",
            "server_uid": 1004,
            "artifact_root": str(tmp_path / "artifacts"),
            "connect_timeout_seconds": 2,
            "request_timeout_seconds": 30,
        }
    )


def test_browser_readiness_runs_release_local_controller_command(
    tmp_path, monkeypatch
):
    _BrowserClient.instances.clear()
    _BrowserClient.result = {
        "success": True,
        "data": {"result": "about:blank"},
    }
    monkeypatch.setattr(readiness, "BrowserControllerClient", _BrowserClient)

    receipt = readiness.attest_browser_controller_execution(
        client_config=_browser_config(tmp_path),
        revision=REVISION,
        config_sha256=CONFIG_SHA256,
    )

    client = _BrowserClient.instances[0]
    assert client.command_value == ("eval", ["window.location.href"])
    assert receipt["command_round_trip"] is True
    assert receipt["secret_material_recorded"] is False
    assert client.closed is True


def test_browser_readiness_rejects_failed_command(tmp_path, monkeypatch):
    _BrowserClient.instances.clear()
    _BrowserClient.result = {
        "success": False,
        "error": "browser_controller_command_failed",
    }
    monkeypatch.setattr(readiness, "BrowserControllerClient", _BrowserClient)

    with pytest.raises(
        readiness.ProductionExecutionReadinessError,
        match="browser_readiness_execution_failed",
    ):
        readiness.attest_browser_controller_execution(
            client_config=_browser_config(tmp_path),
            revision=REVISION,
            config_sha256=CONFIG_SHA256,
        )
    assert _BrowserClient.instances[0].closed is True
