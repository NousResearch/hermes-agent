"""Scale-to-zero: an attached dashboard/desktop/TUI WS client counts as activity.

Background (2026-09-02 fleet audit): 13 of 72 active opted-in prod instances
flapped suspend -> proxy-wake every ~60s. The gateway only stamped
``_last_inbound_at`` for messaging inbound, so it suspended under an open
dashboard client; the client's reconnect loop re-poked the Fly-proxied hostname
and autostart resumed the box. The dashboard runs in a separate process on
hosted instances, so the signal crosses over as a marker-file mtime.

These tests exercise the REAL seams — the pure helpers with a real temp
HERMES_HOME, GatewayRunner._scale_to_zero_is_idle's composition, and
tui_gateway.ws.handle_ws — rather than stubbing the collection under test
(the F25 / #84327 lesson: bugs live at the call site, not in the pure predicate).
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from gateway import scale_to_zero as s2z
from gateway.run import GatewayRunner


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _stat_denying(target):
    """os.stat replacement that denies ONLY the marker path (stdlib callers unaffected)."""
    real = os.stat

    def _stat(path, *a, **k):
        if os.fspath(path) == os.fspath(target):
            raise PermissionError("nope")
        return real(path, *a, **k)

    return _stat


# --- pure helpers -----------------------------------------------------------


def test_heartbeat_path_lives_under_hermes_home_state(hermes_home):
    p = s2z.dashboard_client_heartbeat_path()
    assert p == hermes_home / "state" / "dashboard_clients.heartbeat"


def test_last_seen_missing_marker_is_none_not_fail_awake(hermes_home):
    # Steady state for a box nobody has the dashboard open on: must read as
    # "no client", otherwise no instance would ever suspend.
    assert s2z.dashboard_client_last_seen() is None


def test_touch_creates_state_dir_and_marker(hermes_home):
    assert s2z.touch_dashboard_client_heartbeat() is True
    p = s2z.dashboard_client_heartbeat_path()
    assert p.exists()
    seen = s2z.dashboard_client_last_seen()
    assert seen is not None and abs(seen - time.time()) < 5


def test_last_seen_returns_raw_mtime_without_staleness_cutoff(hermes_home):
    # No liveness cutoff here on purpose: is_idle decides recency. A 1h-old
    # marker still reports its mtime; the gateway then finds it outside
    # idle_timeout, same as an old _last_inbound_at.
    s2z.touch_dashboard_client_heartbeat()
    p = s2z.dashboard_client_heartbeat_path()
    mtime = os.stat(p).st_mtime
    assert s2z.dashboard_client_last_seen(now=mtime + 10) == mtime
    assert s2z.dashboard_client_last_seen(now=mtime + 3600) == mtime


def test_last_seen_future_mtime_is_clamped_to_now(hermes_home):
    # A wall-clock step-back can leave the marker in the future; it must not
    # extend the idle window past "now".
    s2z.touch_dashboard_client_heartbeat()
    p = s2z.dashboard_client_heartbeat_path()
    future = time.time() + 600
    os.utime(p, (future, future))
    now = time.time()
    assert s2z.dashboard_client_last_seen(now=now) == now


def test_last_seen_unreadable_marker_fails_awake(hermes_home, monkeypatch):
    s2z.touch_dashboard_client_heartbeat()

    monkeypatch.setattr(s2z.os, "stat", _stat_denying(s2z.dashboard_client_heartbeat_path()))
    now = 1_000_000.0
    # Unreadable (not missing) => counts as activity right now.
    assert s2z.dashboard_client_last_seen(now=now) == now


def test_touch_never_raises(hermes_home, monkeypatch):
    monkeypatch.setattr(s2z.os, "utime", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert s2z.touch_dashboard_client_heartbeat() is False


# --- gateway side: the idle predicate composition ---------------------------


def _runner(monkeypatch, *, last_inbound_at):
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r._last_inbound_at = last_inbound_at
    r._running_agents = {}
    r._background_tasks = set()
    r.adapters = {}
    monkeypatch.setattr(r, "_scale_to_zero_idle_timeout_seconds", lambda: 120.0, raising=False)
    monkeypatch.setattr(r, "_scale_to_zero_has_live_background_work", lambda: False, raising=False)
    monkeypatch.setattr("cron.scheduler.get_running_job_ids", lambda: [])
    return r


def test_idle_without_dashboard_client_unchanged(hermes_home, monkeypatch):
    r = _runner(monkeypatch, last_inbound_at=time.time() - 600)
    assert r._scale_to_zero_is_idle() is True


def test_attached_dashboard_client_blocks_idle(hermes_home, monkeypatch):
    r = _runner(monkeypatch, last_inbound_at=time.time() - 600)
    s2z.touch_dashboard_client_heartbeat()
    assert r._scale_to_zero_is_idle() is False


def test_client_gets_the_same_idle_grace_as_a_message(hermes_home, monkeypatch):
    """Last WS frame 100s ago with a 120s idle_timeout => still inside the
    window => NOT idle. This is the 2-minute-after-the-app-closes contract; an
    earlier draft cut the marker off at 45s and suspended ~50s after
    disconnect (observed live on staging)."""
    r = _runner(monkeypatch, last_inbound_at=time.time() - 600)
    s2z.touch_dashboard_client_heartbeat()
    p = s2z.dashboard_client_heartbeat_path()
    old = time.time() - 100
    os.utime(p, (old, old))
    assert r._scale_to_zero_is_idle() is False


def test_client_gone_longer_than_idle_timeout_is_idle(hermes_home, monkeypatch):
    r = _runner(monkeypatch, last_inbound_at=time.time() - 600)
    s2z.touch_dashboard_client_heartbeat()
    p = s2z.dashboard_client_heartbeat_path()
    old = time.time() - 121
    os.utime(p, (old, old))
    assert r._scale_to_zero_is_idle() is True


def test_marker_predating_gateway_inbound_does_not_matter(hermes_home, monkeypatch):
    # Ancient marker from a client that left hours ago, gateway idle 600s.
    r = _runner(monkeypatch, last_inbound_at=time.time() - 600)
    s2z.touch_dashboard_client_heartbeat()
    p = s2z.dashboard_client_heartbeat_path()
    old = time.time() - 7200
    os.utime(p, (old, old))
    assert r._scale_to_zero_is_idle() is True


def test_dashboard_client_seen_recently_extends_inbound_clock(hermes_home, monkeypatch):
    # Marker 30s old: inbound clock moves to 30s ago, which is
    # inside the 120s window => not idle, even though the gateway's own
    # _last_inbound_at is ancient.
    r = _runner(monkeypatch, last_inbound_at=time.time() - 600)
    s2z.touch_dashboard_client_heartbeat()
    p = s2z.dashboard_client_heartbeat_path()
    t = time.time() - 30
    os.utime(p, (t, t))
    assert r._scale_to_zero_is_idle() is False


def test_newer_gateway_inbound_wins_over_older_marker(hermes_home, monkeypatch):
    r = _runner(monkeypatch, last_inbound_at=time.time() - 5)
    monkeypatch.setattr(r, "_scale_to_zero_idle_timeout_seconds", lambda: 10.0, raising=False)
    s2z.touch_dashboard_client_heartbeat()
    p = s2z.dashboard_client_heartbeat_path()
    t = time.time() - 40
    os.utime(p, (t, t))
    # Picking the marker (40s > 10s) would read idle; the chat message (5s) wins.
    assert r._scale_to_zero_is_idle() is False
    # _last_inbound_at itself is not mutated by the read.
    assert time.time() - r._last_inbound_at < 10


def test_unreadable_marker_keeps_gateway_awake(hermes_home, monkeypatch):
    r = _runner(monkeypatch, last_inbound_at=time.time() - 600)
    s2z.touch_dashboard_client_heartbeat()
    monkeypatch.setattr(s2z.os, "stat", _stat_denying(s2z.dashboard_client_heartbeat_path()))
    assert r._scale_to_zero_is_idle() is False


# --- dashboard side: the real handle_ws path touches the marker -------------


def test_handle_ws_connect_touches_marker(hermes_home, monkeypatch):
    from tui_gateway import server, ws as ws_mod

    monkeypatch.setattr(server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(server, "_schedule_startup_orphan_sweep", lambda: None, raising=False)
    monkeypatch.setattr(server, "resolve_skin", lambda: "default")
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "register_live_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(ws_mod, "_dashboard_client_touched_at", 0.0)

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    assert s2z.dashboard_client_last_seen() is None
    asyncio.run(ws_mod.handle_ws(FakeWS()))
    seen = s2z.dashboard_client_last_seen()
    assert seen is not None and abs(seen - time.time()) < 5


def test_handle_ws_inbound_frames_refresh_marker(hermes_home, monkeypatch):
    from tui_gateway import server, ws as ws_mod

    monkeypatch.setattr(server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(server, "_schedule_startup_orphan_sweep", lambda: None, raising=False)
    monkeypatch.setattr(server, "resolve_skin", lambda: "default")
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "register_live_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(ws_mod, "_dashboard_client_touched_at", 0.0)
    # Disable the throttle so each frame is observable.
    monkeypatch.setattr(ws_mod, "_DASHBOARD_CLIENT_TOUCH_MIN_INTERVAL_S", 0.0)

    frames = ['{"jsonrpc":"2.0","method":"gateway.ping","id":1}'] * 2
    touches: list[float] = []
    real_touch = s2z.touch_dashboard_client_heartbeat

    def _spy(path=None):
        touches.append(time.time())
        return real_touch(path)

    monkeypatch.setattr(s2z, "touch_dashboard_client_heartbeat", _spy)

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            if frames:
                return frames.pop()
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))
    # 1 on connect + 1 per inbound frame.
    assert len(touches) == 3
    assert s2z.dashboard_client_last_seen() is not None


def test_note_activity_is_throttled(hermes_home, monkeypatch):
    from tui_gateway import ws as ws_mod

    calls = {"n": 0}
    monkeypatch.setattr(
        s2z, "touch_dashboard_client_heartbeat", lambda path=None: calls.__setitem__("n", calls["n"] + 1) or True
    )
    monkeypatch.setattr(ws_mod, "_dashboard_client_touched_at", 0.0)
    ws_mod._note_dashboard_client_activity(force=True)
    ws_mod._note_dashboard_client_activity()
    ws_mod._note_dashboard_client_activity()
    assert calls["n"] == 1
    ws_mod._note_dashboard_client_activity(force=True)
    assert calls["n"] == 2
