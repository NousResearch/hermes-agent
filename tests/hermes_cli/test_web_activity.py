"""``GET /api/activity`` — the busy signal an idle reaper must consult before killing a backend.

Contracts: ``busy`` relates to the three counters (true iff any is positive); the route is
token-gated like the other non-public dashboard endpoints; ``running_turns`` counts live
``tui_gateway`` sessions with a turn in flight and ignores finalized ones.
"""

from __future__ import annotations

import threading

import pytest

from hermes_cli.web_routers.activity import activity_snapshot


@pytest.mark.parametrize("turns,subagents,procs", [(1, 0, 0), (0, 2, 0), (0, 0, 1), (3, 1, 4)])
def test_busy_when_any_counter_is_positive(turns, subagents, procs):
    snap = activity_snapshot(turns, subagents, procs)
    assert snap["busy"] is True
    assert (snap["running_turns"], snap["active_subagents"], snap["background_processes"]) == (turns, subagents, procs)


def test_idle_when_every_counter_is_zero():
    snap = activity_snapshot(lambda: 0, lambda: 0, lambda: 0)
    assert snap == {"ok": True, "busy": False, "running_turns": 0, "active_subagents": 0, "background_processes": 0}


def test_failing_counter_source_reads_as_zero_not_busy():
    def boom():
        raise RuntimeError("registry unavailable")
    snap = activity_snapshot(boom, 0, boom)
    assert snap["busy"] is False
    assert snap["running_turns"] == snap["background_processes"] == 0


class TestActivityEndpoint:
    @pytest.fixture(autouse=True)
    def _client(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        self.anon = TestClient(app)
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_requires_the_dashboard_token(self):
        assert self.anon.get("/api/activity").status_code == 401
        resp = self.client.get("/api/activity")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["busy"] == any(body[k] > 0 for k in ("running_turns", "active_subagents", "background_processes"))

    def test_running_turns_counts_live_running_sessions_and_ignores_finalized(self):
        """E2E against the real ``tui_gateway.server._sessions`` registry."""
        from tui_gateway import server

        def _rec(running: bool, finalized: bool = False) -> dict:
            return {"session_key": "k", "history_lock": threading.Lock(), "running": running,
                    "_finalized": finalized}

        fake = {"act-running": _rec(True), "act-running-2": _rec(True), "act-idle": _rec(False),
                "act-finalized-running": _rec(True, finalized=True)}
        with server._sessions_lock:
            server._sessions.update(fake)
        try:
            body = self.client.get("/api/activity").json()
        finally:
            with server._sessions_lock:
                for sid in fake:
                    server._sessions.pop(sid, None)
        assert body["running_turns"] >= 2
        assert body["busy"] is True

        after = self.client.get("/api/activity").json()
        assert after["running_turns"] == body["running_turns"] - 2
