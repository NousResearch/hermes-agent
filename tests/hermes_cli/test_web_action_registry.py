"""Tests for the durable gateway-action registry (issue #25916).

Phase-1 behavior covered:

* ``_record_action_start`` writes a ``running`` record to ``actions.json``.
* ``_record_action_finished`` updates that record with an exit code.
* ``_resolve_action_state`` distinguishes ``running`` / ``succeeded`` /
  ``failed`` / ``lost`` / ``never_started`` across in-memory and on-disk
  state, including the post-restart case where the in-memory Popen handle
  is gone but the pid is still alive.
* ``/api/gateway/restart`` and ``/api/hermes/update`` reject duplicate
  in-flight invocations with HTTP 409.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# fastapi is an optional dep — skip cleanly if it's not installed.
fastapi = pytest.importorskip("fastapi")

from hermes_cli import web_server  # noqa: E402


@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
    """Isolate the action registry into a tmp_path-scoped HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Reset in-memory action handles between tests.
    web_server._ACTION_PROCS.clear()
    yield tmp_path
    web_server._ACTION_PROCS.clear()


class TestActionRegistryIO:
    def test_read_empty_when_missing(self, tmp_hermes_home):
        assert web_server._read_action_registry() == {}

    def test_read_returns_empty_on_malformed(self, tmp_hermes_home):
        path = web_server._action_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json{", encoding="utf-8")
        assert web_server._read_action_registry() == {}

    def test_write_then_read_roundtrip(self, tmp_hermes_home):
        web_server._write_action_registry({"x": {"name": "x", "status": "running"}})
        out = web_server._read_action_registry()
        assert out["x"]["status"] == "running"

    def test_record_start_writes_running(self, tmp_hermes_home):
        log_path = tmp_hermes_home / "logs" / "demo.log"
        web_server._record_action_start("demo", 12345, log_path)
        registry = web_server._read_action_registry()
        assert registry["demo"]["status"] == "running"
        assert registry["demo"]["pid"] == 12345
        assert registry["demo"]["exit_code"] is None
        assert registry["demo"]["log_path"] == str(log_path)
        assert registry["demo"]["start_time"] > 0

    def test_record_finished_updates_exit_code(self, tmp_hermes_home):
        web_server._record_action_start("demo", 1, tmp_hermes_home / "demo.log")
        web_server._record_action_finished("demo", 0)
        rec = web_server._read_action_registry()["demo"]
        assert rec["status"] == "succeeded"
        assert rec["exit_code"] == 0

        web_server._record_action_start("demo2", 2, tmp_hermes_home / "demo2.log")
        web_server._record_action_finished("demo2", 1)
        rec2 = web_server._read_action_registry()["demo2"]
        assert rec2["status"] == "failed"
        assert rec2["exit_code"] == 1

    def test_record_finished_noop_when_missing(self, tmp_hermes_home):
        # Should not raise or create a stray record.
        web_server._record_action_finished("never-existed", 0)
        assert "never-existed" not in web_server._read_action_registry()


class TestResolveActionState:
    def test_never_started(self, tmp_hermes_home):
        state = web_server._resolve_action_state("gateway-restart")
        assert state == {
            "status": "never_started",
            "running": False,
            "exit_code": None,
            "pid": None,
        }

    def test_live_popen_running(self, tmp_hermes_home):
        proc = SimpleNamespace(pid=4242, poll=MagicMock(return_value=None))
        web_server._ACTION_PROCS["gateway-restart"] = proc
        state = web_server._resolve_action_state("gateway-restart")
        assert state["running"] is True
        assert state["status"] == "running"
        assert state["pid"] == 4242

    def test_live_popen_succeeded_updates_registry(self, tmp_hermes_home):
        # Pre-seed registry as running so we exercise the update path.
        web_server._record_action_start("gateway-restart", 7, tmp_hermes_home / "log")
        proc = SimpleNamespace(pid=7, poll=MagicMock(return_value=0))
        web_server._ACTION_PROCS["gateway-restart"] = proc
        state = web_server._resolve_action_state("gateway-restart")
        assert state == {"status": "succeeded", "running": False, "exit_code": 0, "pid": 7}
        assert web_server._read_action_registry()["gateway-restart"]["status"] == "succeeded"

    def test_live_popen_failed(self, tmp_hermes_home):
        web_server._record_action_start("gateway-restart", 7, tmp_hermes_home / "log")
        proc = SimpleNamespace(pid=7, poll=MagicMock(return_value=2))
        web_server._ACTION_PROCS["gateway-restart"] = proc
        state = web_server._resolve_action_state("gateway-restart")
        assert state == {"status": "failed", "running": False, "exit_code": 2, "pid": 7}

    def test_lost_after_restart(self, tmp_hermes_home):
        """Registry says running, in-memory Popen is gone, pid is dead."""
        web_server._record_action_start("hermes-update", 999999, tmp_hermes_home / "log")
        with patch.object(web_server, "_pid_is_alive", return_value=False):
            state = web_server._resolve_action_state("hermes-update")
        assert state["status"] == "lost"
        assert state["running"] is False
        assert state["pid"] == 999999

    def test_running_elsewhere_after_restart(self, tmp_hermes_home):
        """Registry says running, in-memory Popen is gone, pid is still alive
        (covers the case where the web process restarted mid-action)."""
        web_server._record_action_start("hermes-update", 1234, tmp_hermes_home / "log")
        with patch.object(web_server, "_pid_is_alive", return_value=True):
            state = web_server._resolve_action_state("hermes-update")
        assert state["status"] == "running"
        assert state["running"] is True
        assert state["pid"] == 1234

    def test_finished_record_after_restart(self, tmp_hermes_home):
        """Registry has a recorded exit code — propagate it even without Popen."""
        web_server._record_action_start("hermes-update", 1234, tmp_hermes_home / "log")
        web_server._record_action_finished("hermes-update", 0)
        state = web_server._resolve_action_state("hermes-update")
        assert state == {
            "status": "succeeded",
            "running": False,
            "exit_code": 0,
            "pid": 1234,
        }


class TestPidIsAlive:
    def test_invalid_pid_returns_false(self):
        assert web_server._pid_is_alive(0) is False
        assert web_server._pid_is_alive(-1) is False

    def test_current_process_is_alive(self):
        # os.getpid() is guaranteed to exist; harmless signal-0 probe.
        assert web_server._pid_is_alive(os.getpid()) is True

    def test_process_lookup_error_returns_false(self):
        with patch("os.kill", side_effect=ProcessLookupError):
            assert web_server._pid_is_alive(99999) is False

    def test_permission_error_returns_false(self):
        with patch("os.kill", side_effect=PermissionError):
            assert web_server._pid_is_alive(99999) is False


class TestDuplicateActionGuard:
    """The two POST endpoints reject 409 when the same action is already running."""

    def _client(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN
        client = TestClient(web_server.app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        return client

    def test_restart_gateway_rejects_duplicate(self, tmp_hermes_home):
        proc = SimpleNamespace(pid=4321, poll=MagicMock(return_value=None))
        web_server._ACTION_PROCS["gateway-restart"] = proc
        client = self._client()
        resp = client.post("/api/gateway/restart")
        assert resp.status_code == 409
        body = resp.json()["detail"]
        assert body["reason"] == "already_running"
        assert body["pid"] == 4321
        assert body["name"] == "gateway-restart"

    def test_update_rejects_duplicate(self, tmp_hermes_home):
        proc = SimpleNamespace(pid=8765, poll=MagicMock(return_value=None))
        web_server._ACTION_PROCS["hermes-update"] = proc
        client = self._client()
        resp = client.post("/api/hermes/update")
        assert resp.status_code == 409
        body = resp.json()["detail"]
        assert body["reason"] == "already_running"
        assert body["pid"] == 8765

    def test_restart_gateway_allows_when_previous_finished(self, tmp_hermes_home):
        """A previously-finished record must not block a fresh invocation."""
        web_server._record_action_start("gateway-restart", 1, tmp_hermes_home / "log")
        web_server._record_action_finished("gateway-restart", 0)
        client = self._client()
        with patch.object(web_server, "_spawn_hermes_action") as spawn:
            spawn.return_value = SimpleNamespace(pid=999)
            resp = client.post("/api/gateway/restart")
        assert resp.status_code == 200
        assert resp.json()["pid"] == 999


class TestStatusEndpointSurfacesStatusField:
    def _client(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN
        client = TestClient(web_server.app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        return client

    def test_status_endpoint_returns_status_string(self, tmp_hermes_home):
        client = self._client()
        resp = client.get("/api/actions/gateway-restart/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "gateway-restart"
        assert body["status"] == "never_started"
        assert body["running"] is False
        assert body["lines"] == []

    def test_status_endpoint_unknown_action(self, tmp_hermes_home):
        client = self._client()
        resp = client.get("/api/actions/not-a-real-action/status")
        assert resp.status_code == 404
