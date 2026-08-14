"""Tests for the gateway diagnostics ring (U3 / KTD2 / KTD4).

Covers the four things the capture bundle depends on:

* a **sub-second** loop block lands in the ring even though it never reaches
  the 5s "event loop stalled" log threshold, and carries a sanitized frame
  summary naming the blocking call site;
* disarmed is genuinely inert — no ring, no thread, and the production 2.0s
  heartbeat interval is untouched;
* ``collect`` returns events + a dropped count and is reachable only through
  the authenticated ``/api/ws`` surface;
* peer identifiers are HMAC'd per capture, so the same peer tokenizes
  differently in two captures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from types import SimpleNamespace

import pytest

from hermes_cli import diagnostics_ring
from hermes_cli import web_server


@pytest.fixture(autouse=True)
def _disarmed_between_tests():
    """Never leave a capture (or its watchdog thread) armed across tests."""
    diagnostics_ring.disarm()
    yield
    diagnostics_ring.disarm()


@pytest.fixture
def loopback_client():
    """``web_server.app`` in loopback (session-token) mode, as the desktop sees it."""
    from fastapi.testclient import TestClient

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    with TestClient(web_server.app, base_url="http://127.0.0.1:8080") as client:
        yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeLoop:
    """Enough of an event loop to drive ``_install_loop_heartbeat`` by hand."""

    def __init__(self) -> None:
        self.now = 100.0
        self.scheduled: list[tuple[float, object, tuple]] = []

    def time(self) -> float:
        return self.now

    def call_later(self, delay, fn, *args):
        self.scheduled.append((delay, fn, args))
        return SimpleNamespace(cancel=lambda: None)

    def tick(self):
        """Fire the most recently scheduled heartbeat callback."""
        delay, fn, args = self.scheduled.pop()
        fn(*args)
        return delay


def _blocking_call_site(seconds: float) -> None:
    """Stand-in for the GIL-heavy work that stalls the loop in production."""
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# Arm / disarm / collect
# ---------------------------------------------------------------------------


class TestArmDisarmCollect:
    def test_arm_is_idempotent_per_capture_id(self):
        first = diagnostics_ring.arm("cap-1", wall_clock_anchor_ms=1234.0)
        again = diagnostics_ring.arm("cap-1", wall_clock_anchor_ms=9999.0)
        assert again["monotonic_anchor_ms"] == first["monotonic_anchor_ms"]
        assert diagnostics_ring.collect("cap-1")["wall_clock_anchor_ms"] == 1234.0

    def test_arm_new_capture_id_replaces_the_window(self):
        diagnostics_ring.arm("cap-1")
        diagnostics_ring.record_loop_drift(1.0)
        diagnostics_ring.arm("cap-2")
        with pytest.raises(LookupError):
            diagnostics_ring.collect("cap-1")
        assert diagnostics_ring.collect("cap-2")["events"] == []

    def test_disarm_is_idempotent_and_clears_the_flag(self):
        diagnostics_ring.arm("cap-1")
        assert diagnostics_ring.ARMED is True
        assert diagnostics_ring.disarm()["armed"] is False
        assert diagnostics_ring.disarm()["armed"] is False
        assert diagnostics_ring.ARMED is False

    def test_collect_reports_events_and_dropped(self, monkeypatch):
        monkeypatch.setattr(diagnostics_ring, "_MAX_EVENTS", 3)
        diagnostics_ring.arm("cap-1")
        for _ in range(5):
            diagnostics_ring.record_loop_drift(0.5)
        out = diagnostics_ring.collect("cap-1")
        assert out["capture_id"] == "cap-1"
        assert len(out["events"]) == 3
        assert out["dropped"] == 2
        assert out["monotonic_anchor_ms"] > 0

    def test_collect_on_unknown_capture_id_raises(self):
        diagnostics_ring.arm("cap-1")
        with pytest.raises(LookupError):
            diagnostics_ring.collect("cap-other")

    def test_drift_below_the_floor_is_not_recorded(self):
        diagnostics_ring.arm("cap-1")
        assert diagnostics_ring.record_loop_drift(0.1) is None
        assert diagnostics_ring.collect("cap-1")["events"] == []


# ---------------------------------------------------------------------------
# Disarmed = inert
# ---------------------------------------------------------------------------


class TestDisarmedIsInert:
    def test_no_ring_and_no_recording_when_disarmed(self):
        assert diagnostics_ring.ARMED is False
        assert diagnostics_ring.record_loop_drift(9.0) is None
        diagnostics_ring.record_ws_write_slow("1.2.3.4:5", 10.0)
        diagnostics_ring.note_loop_tick()
        with pytest.raises(LookupError):
            diagnostics_ring.collect("cap-1")

    def test_watchdog_thread_only_exists_while_armed(self):
        import threading

        def _watchdogs():
            return [t for t in threading.enumerate()
                    if t.name.startswith("hermes-diag-watchdog")]

        assert _watchdogs() == []
        diagnostics_ring.arm("cap-1")
        assert len(_watchdogs()) == 1
        diagnostics_ring.disarm()
        for _ in range(50):
            if not _watchdogs():
                break
            time.sleep(0.02)
        assert _watchdogs() == []

    def test_heartbeat_keeps_the_2s_interval_when_disarmed(self):
        assert web_server._HB_INTERVAL == 2.0
        loop = _FakeLoop()
        web_server._install_loop_heartbeat(loop)
        assert loop.scheduled[0][0] == 2.0
        loop.now += 2.0
        assert loop.tick() == 2.0
        assert loop.scheduled[-1][0] == 2.0

    def test_heartbeat_tightens_only_while_armed(self):
        loop = _FakeLoop()
        web_server._install_loop_heartbeat(loop)
        diagnostics_ring.arm("cap-1")
        loop.now += 2.0
        loop.tick()
        assert loop.scheduled[-1][0] == diagnostics_ring.armed_heartbeat_interval()
        diagnostics_ring.disarm()
        loop.now += 0.05
        loop.tick()
        assert loop.scheduled[-1][0] == 2.0

    def test_log_threshold_is_unchanged_while_armed(self, caplog):
        """Armed capture adds a sink; it must not lower the 5s warning."""
        loop = _FakeLoop()
        web_server._install_loop_heartbeat(loop)
        diagnostics_ring.arm("cap-1")
        with caplog.at_level(logging.WARNING, logger=web_server._log.name):
            # 3.0s of drift: above the ring floor, below the log threshold.
            loop.now += 5.0
            loop.tick()
        assert "event loop stalled" not in caplog.text
        events = diagnostics_ring.collect("cap-1")["events"]
        assert [e["kind"] for e in events] == ["loop_drift"]
        assert events[0]["drift_s"] == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# Sub-second stall + attribution
# ---------------------------------------------------------------------------


class TestSubSecondStallAttribution:
    def test_400ms_block_lands_in_the_ring_with_a_frame_summary(self, monkeypatch):
        # Production ticks every 2s, which cannot resolve a 400ms block at all.
        # Shorten only the *disarmed* interval so the heartbeat starts promptly
        # inside the test; the armed interval is the module's own value.
        monkeypatch.setattr(web_server, "_HB_INTERVAL", 0.05)

        async def _main():
            loop = asyncio.get_running_loop()
            diagnostics_ring.arm("cap-block")
            web_server._install_loop_heartbeat(loop)
            # Let a few ticks establish loop liveness for the watchdog.
            await asyncio.sleep(0.25)
            _blocking_call_site(0.4)
            # Let the post-stall tick record the drift.
            await asyncio.sleep(0.25)

        asyncio.run(_main())

        out = diagnostics_ring.collect("cap-block")
        stalls = [e for e in out["events"] if e["drift_s"] >= 0.25]
        assert stalls, f"no sub-second stall recorded: {out['events']}"
        worst = max(stalls, key=lambda e: e["drift_s"])
        assert worst["kind"] == "loop_drift"
        assert 0.3 <= worst["drift_s"] < 1.0
        assert worst["t_monotonic"] > 0

        attributed = [e for e in stalls if e.get("frames")]
        assert attributed, f"no stall carried a frame summary: {stalls}"
        frames = attributed[0]["frames"]
        assert any(f["function"] == "_blocking_call_site" for f in frames)
        for f in frames:
            assert set(f) == {"module", "function", "line"}
            assert ":" not in f["module"] and "/" not in f["module"]
            assert "\\" not in f["module"]
        blocking = next(f for f in frames if f["function"] == "_blocking_call_site")
        assert blocking["module"].endswith("test_diagnostics_ring")
        assert isinstance(blocking["line"], int)


# ---------------------------------------------------------------------------
# WS write-slow events + per-capture HMAC
# ---------------------------------------------------------------------------


class TestWsWriteSlowEvents:
    def test_peer_is_tokenized_and_correlates_within_a_capture(self):
        diagnostics_ring.arm("cap-1")
        diagnostics_ring.record_ws_write_slow("127.0.0.1:51544", 10.0)
        diagnostics_ring.record_ws_write_slow("127.0.0.1:51544", 10.0)
        diagnostics_ring.record_ws_write_slow("127.0.0.1:60001", 10.0)
        events = diagnostics_ring.collect("cap-1")["events"]
        assert [e["kind"] for e in events] == ["ws_write_slow"] * 3
        assert events[0]["peer"] == events[1]["peer"]
        assert events[0]["peer"] != events[2]["peer"]
        # Non-reversible: the raw identifier appears nowhere in the ring.
        assert "127.0.0.1" not in repr(events)
        assert events[0]["timeout_s"] == 10.0

    def test_same_peer_tokenizes_differently_across_captures(self):
        peer = "127.0.0.1:51544"
        diagnostics_ring.arm("cap-1")
        diagnostics_ring.record_ws_write_slow(peer, 10.0)
        one = diagnostics_ring.collect("cap-1")["events"][0]["peer"]
        diagnostics_ring.arm("cap-2")
        diagnostics_ring.record_ws_write_slow(peer, 10.0)
        two = diagnostics_ring.collect("cap-2")["events"][0]["peer"]
        assert one != two

    def test_per_capture_key_is_never_exposed(self):
        diagnostics_ring.arm("cap-1")
        diagnostics_ring.record_ws_write_slow("peer-a", 10.0)
        out = diagnostics_ring.collect("cap-1")
        assert "key" not in out
        assert set(out) == {
            "capture_id", "monotonic_anchor_ms", "wall_clock_anchor_ms",
            "events", "dropped",
        }

    def test_ws_transport_records_on_the_write_timeout_path(self, monkeypatch):
        """The real ``ws write slow`` branch feeds the ring while armed."""
        import concurrent.futures

        from tui_gateway import ws as ws_mod

        class _StalledFuture:
            """A send scheduled onto a loop that never breathes."""

            def result(self, timeout=None):
                raise concurrent.futures.TimeoutError()

        def _stalled(coro, loop):
            coro.close()  # never awaited; keep the test warning-free
            return _StalledFuture()

        monkeypatch.setattr(
            "agent.async_utils.safe_schedule_threadsafe", _stalled
        )
        transport = ws_mod.WSTransport(
            ws=SimpleNamespace(), loop=_FakeLoop(), peer="127.0.0.1:51544"
        )

        diagnostics_ring.arm("cap-ws")
        assert transport.write({"jsonrpc": "2.0", "id": "1", "result": {}}) is True
        events = diagnostics_ring.collect("cap-ws")["events"]
        assert [e["kind"] for e in events] == ["ws_write_slow"]
        assert "127.0.0.1" not in repr(events)


# ---------------------------------------------------------------------------
# Auth surface
# ---------------------------------------------------------------------------


class TestAuthenticatedSurfaceOnly:
    def test_methods_are_registered_on_the_gateway_rpc_table(self):
        from tui_gateway import server, ws as _ws  # noqa: F401 - installs methods

        for name in (
            "diagnostics.arm", "diagnostics.disarm", "diagnostics.collect",
            "diagnostics/arm", "diagnostics/disarm", "diagnostics/collect",
        ):
            assert name in server._methods

    def test_routes_live_on_the_gated_api_prefix_only(self):
        """The ring rides the existing gate — it must not become public.

        ``/api/`` + absence from ``PUBLIC_API_PATHS`` *is* the auth: any other
        prefix would slip past ``auth_middleware`` entirely.
        """
        from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS

        paths = [getattr(r, "path", "") for r in web_server.app.routes]
        mounted = [
            p
            for p in paths
            if "/diagnostics/" in p
            and "plugins" not in p
            # The U5 test hook mounts only under an env opt-in and has its own
            # coverage in TestLoopBlockHookIsGuarded.
            and p != diagnostics_ring._TEST_BLOCK_PATH
        ]
        assert sorted(mounted) == [
            "/api/diagnostics/arm",
            "/api/diagnostics/collect",
            "/api/diagnostics/disarm",
        ]
        for path in mounted:
            assert path not in PUBLIC_API_PATHS

    def test_unauthenticated_http_pull_is_refused(self, loopback_client):
        """An unauthenticated local caller cannot pull an armed ring."""
        diagnostics_ring.arm("cap-http")
        diagnostics_ring.record_loop_drift(0.4)

        for path, body in (
            ("/api/diagnostics/arm", {"capture_id": "cap-evil"}),
            ("/api/diagnostics/collect", {"capture_id": "cap-http"}),
            ("/api/diagnostics/disarm", {}),
        ):
            resp = loopback_client.post(path, json=body)
            assert resp.status_code == 401, path
        # Still armed, still ours: the refused calls changed nothing.
        assert diagnostics_ring.collect("cap-http")["capture_id"] == "cap-http"

    def test_authenticated_http_round_trip(self, loopback_client):
        headers = {"Authorization": f"Bearer {web_server._SESSION_TOKEN}"}

        armed = loopback_client.post(
            "/api/diagnostics/arm",
            json={"capture_id": "cap-http", "wall_clock_anchor_ms": 17.0},
            headers=headers,
        )
        assert armed.status_code == 200
        assert armed.json()["monotonic_anchor_ms"] > 0

        diagnostics_ring.record_loop_drift(0.4)
        got = loopback_client.post(
            "/api/diagnostics/collect",
            json={"capture_id": "cap-http"},
            headers=headers,
        )
        assert got.status_code == 200
        payload = got.json()
        assert payload["capture_id"] == "cap-http"
        assert payload["dropped"] == 0
        assert payload["events"][0]["kind"] == "loop_drift"

        # 409, not 404 — the desktop client reads 404 as "gateway too old".
        missing = loopback_client.post(
            "/api/diagnostics/collect", json={"capture_id": "other"}, headers=headers
        )
        assert missing.status_code == 409

        assert loopback_client.post(
            "/api/diagnostics/disarm", json={}, headers=headers
        ).json()["armed"] is False
        assert diagnostics_ring.ARMED is False

    def test_unauthenticated_ws_upgrade_is_refused(self, monkeypatch):
        """The ring is only reachable behind ``/api/ws``'s existing gate."""
        monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(web_server.app.state, "bound_port", 8080, raising=False)

        class _QP:
            def __init__(self, q):
                self._q = q

            def get(self, k, default=""):
                return self._q.get(k, default)

        def _ws(query):
            return SimpleNamespace(
                query_params=_QP(query),
                client=SimpleNamespace(host="127.0.0.1"),
                url=SimpleNamespace(path="/api/ws"),
            )

        assert web_server._ws_auth_ok(_ws({})) is False
        assert web_server._ws_auth_ok(_ws({"token": "not-the-token"})) is False
        assert web_server._ws_auth_ok(_ws({"token": web_server._SESSION_TOKEN})) is True

    def test_collect_handler_refuses_an_unarmed_capture(self):
        from tui_gateway import server, ws as _ws  # noqa: F401

        resp = server._methods["diagnostics.collect"]("r1", {"capture_id": "nope"})
        assert resp["error"]["code"] == diagnostics_ring._ERR_NO_CAPTURE

    def test_arm_collect_disarm_round_trip_through_the_rpc_table(self):
        from tui_gateway import server, ws as _ws  # noqa: F401

        armed = server._methods["diagnostics/arm"](
            "r1", {"capture_id": "cap-rpc", "wall_clock_anchor_ms": 42.0}
        )
        assert armed["result"]["monotonic_anchor_ms"] > 0
        diagnostics_ring.record_loop_drift(0.4)
        got = server._methods["diagnostics/collect"]("r2", {"capture_id": "cap-rpc"})
        assert got["result"]["dropped"] == 0
        assert got["result"]["events"][0]["kind"] == "loop_drift"
        assert server._methods["diagnostics/disarm"]("r3", {})["result"]["armed"] is False


# ---------------------------------------------------------------------------
# Test-only loop-block hook (U5)
# ---------------------------------------------------------------------------


class TestLoopBlockHookIsGuarded:
    """The hook that lets the U5 harness stall the gateway on purpose.

    The whole safety property is *registration-time*: without the opt-in env
    var the route does not exist, so there is no handler to reach, no flag to
    misread, and no refusal path to get wrong.
    """

    @staticmethod
    def _fresh_app():
        from fastapi import FastAPI

        return FastAPI()

    def test_not_mounted_without_the_env_var(self, monkeypatch):
        monkeypatch.delenv(diagnostics_ring._TEST_HOOKS_ENV, raising=False)
        app = self._fresh_app()

        assert diagnostics_ring.install_test_routes(app) is False
        assert [getattr(r, "path", "") for r in app.routes if "block-loop" in getattr(r, "path", "")] == []

    def test_a_normally_started_gateway_has_no_such_route(self):
        """The real app, imported without the env var, never mounted it."""
        if os.environ.get(diagnostics_ring._TEST_HOOKS_ENV) == "1":
            pytest.skip("this test process itself opted into the hooks")
        assert diagnostics_ring.test_hooks_enabled() is False
        paths = [getattr(r, "path", "") for r in web_server.app.routes]
        assert diagnostics_ring._TEST_BLOCK_PATH not in paths

    def test_mounted_only_with_the_env_var_and_stays_behind_the_api_gate(self, monkeypatch):
        from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS

        monkeypatch.setenv(diagnostics_ring._TEST_HOOKS_ENV, "1")
        app = self._fresh_app()

        assert diagnostics_ring.install_test_routes(app) is True
        paths = [getattr(r, "path", "") for r in app.routes]
        assert diagnostics_ring._TEST_BLOCK_PATH in paths
        # Same gate as the rest of the ring: under /api/ and not public.
        assert diagnostics_ring._TEST_BLOCK_PATH.startswith("/api/")
        assert diagnostics_ring._TEST_BLOCK_PATH not in PUBLIC_API_PATHS

    def test_the_hook_blocks_for_the_requested_duration(self, monkeypatch):
        from fastapi.testclient import TestClient

        monkeypatch.setenv(diagnostics_ring._TEST_HOOKS_ENV, "1")
        app = self._fresh_app()
        diagnostics_ring.install_test_routes(app)

        started = time.monotonic()
        with TestClient(app) as client:
            resp = client.post(diagnostics_ring._TEST_BLOCK_PATH, json={"seconds": 0.4})
        elapsed = time.monotonic() - started

        assert resp.status_code == 200
        assert resp.json()["blocked_s"] >= 0.4
        assert elapsed >= 0.4

    def test_a_block_lands_in_an_armed_ring_as_loop_drift(self, monkeypatch):
        """What the harness asserts on: the block is what the ring records.

        The heartbeat itself is the recorder, so this drives its two calls the
        way ``_install_loop_heartbeat`` does — the block happens *between* two
        ticks and the drift it caused is the event.
        """
        monkeypatch.setenv(diagnostics_ring._TEST_HOOKS_ENV, "1")
        diagnostics_ring.arm("cap-block")

        started = time.monotonic()
        diagnostics_ring.block_event_loop(0.4)
        drift = time.monotonic() - started
        diagnostics_ring.record_loop_drift(drift)

        events = diagnostics_ring.collect("cap-block")["events"]
        assert [e["kind"] for e in events] == ["loop_drift"]
        assert events[0]["drift_s"] >= 0.4

    @pytest.mark.parametrize("seconds", [None, "soon", 0, -1, 31.0])
    def test_a_nonsense_duration_is_refused(self, seconds):
        with pytest.raises(ValueError):
            diagnostics_ring.block_event_loop(seconds)
