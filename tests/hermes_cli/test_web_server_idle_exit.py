"""A Desktop-owned ``serve --isolated`` backend over SSH must retire itself once no client has been
connected for the grace window and no turn is running (#101626): it is detached from any parent on
purpose, so the client count IS its liveness signal."""

from threading import Event

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient

import hermes_cli.web_server as ws_mod
from hermes_cli.web_server_idle_exit import (
    IdleClientTracker, should_exit_idle, start_idle_watchdog, wrap_asgi_with_ws_tracking)


def test_ws_sessions_are_counted_at_the_asgi_boundary_for_any_route():
    app = FastAPI()

    @app.websocket("/api/anything")
    async def anything(ws: WebSocket):
        await ws.accept()
        await ws.receive_text()
        await ws.close()

    @app.websocket("/api/refused")
    async def refused(ws: WebSocket):
        await ws.close(code=4401)  # never accepted: must not count

    tracker = IdleClientTracker(now=lambda: 0.0)
    client = TestClient(wrap_asgi_with_ws_tracking(app, tracker))
    with client.websocket_connect("/api/anything") as a:
        with client.websocket_connect("/api/anything") as b:
            assert tracker.live_count() == 2
            b.send_text("bye")
        assert tracker.live_count() == 1
        a.send_text("bye")
    assert tracker.live_count() == 0
    try:
        with client.websocket_connect("/api/refused"):
            pass
    except Exception:
        pass
    assert tracker.live_count() == 0  # a refused (never accepted) upgrade is not a client


def test_exit_only_after_grace_with_no_client_and_no_running_turn():
    clock = {"t": 0.0}
    tracker = IdleClientTracker(now=lambda: clock["t"])
    grace = 900.0
    assert should_exit_idle(tracker, grace, probe=lambda: False) is False  # just started
    clock["t"] = 901.0
    assert should_exit_idle(tracker, grace, probe=lambda: False) is True
    assert should_exit_idle(tracker, grace, probe=lambda: True) is False   # turn running
    assert should_exit_idle(tracker, grace, probe=lambda: None) is False   # indeterminate: fail closed
    tracker.on_open()
    clock["t"] = 5000.0
    assert should_exit_idle(tracker, grace, probe=lambda: False) is False  # a client is connected
    tracker.on_close()
    assert should_exit_idle(tracker, grace, probe=lambda: False) is False  # grace restarts on close
    clock["t"] = 5000.0 + grace + 1
    assert should_exit_idle(tracker, grace, probe=lambda: False) is True


def test_watchdog_sets_should_exit_and_only_arms_for_ssh_isolated_backends(monkeypatch):
    class _Server:
        should_exit = False

    server = _Server()
    clock = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0])
    tracker = IdleClientTracker(now=lambda: next(clock, 10_000.0))
    start_idle_watchdog(server, tracker, grace_s=1.0, poll_s=0.01, probe=lambda: False).join(timeout=5)
    assert server.should_exit is True

    # The uvicorn builder wraps the app + arms tunnel pings ONLY when a session token was handed over.
    monkeypatch.setattr(ws_mod.app.state, "auth_required", False, raising=False)
    plain, _ = ws_mod._build_uvicorn_server("127.0.0.1", 0)
    assert plain.ws_ping_interval is None and getattr(ws_mod.app.state, "ssh_isolated_clients", None) is None
    try:
        isolated, _ = ws_mod._build_uvicorn_server("127.0.0.1", 0, ssh_isolated=True)
        assert isolated.ws_ping_interval and isolated.ws_ping_timeout > isolated.ws_ping_interval
        assert isinstance(ws_mod.app.state.ssh_isolated_clients, IdleClientTracker)
    finally:
        ws_mod.app.state._state.pop("ssh_isolated_clients", None)  # process-global app: never leak the tracker


def test_attached_ssh_backend_retires_after_checkout_changes_only_with_idle_permit(monkeypatch):
    """An attached WebSocket must not keep a provably idle, code-skewed backend alive forever."""
    from gateway import code_skew
    from hermes_cli import backend_retirement, web_server_idle_proof

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("old", "new"))
    tracker = IdleClientTracker()
    tracker.on_open()  # The SSH Desktop client is still connected.

    class _Server:
        should_exit = False

    server = _Server()
    thread = start_idle_watchdog(server, tracker, grace_s=900, poll_s=0.01, probe=lambda: False)
    try:
        thread.join(timeout=2)
        assert server.should_exit is True
        assert fence.acquire() is False  # New work cannot race a graceful exit.
    finally:
        server.should_exit = True
        thread.join(timeout=2)


@pytest.mark.parametrize("unavailable_or_busy", [None, False])
def test_code_skew_waits_for_human_input_or_unreadable_idle_proof(monkeypatch, unavailable_or_busy):
    from gateway import code_skew
    from hermes_cli import backend_retirement, web_server_idle_proof

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    verdict = [unavailable_or_busy]
    attempted = Event()
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": verdict[0]})
    prepare = fence.prepare

    def prepare_and_signal():
        result = prepare()
        attempted.set()  # The boot-time probe is not the retirement attempt.
        return result

    monkeypatch.setattr(fence, "prepare", prepare_and_signal)
    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("old", "new"))
    tracker = IdleClientTracker()
    tracker.on_open()

    class _Server:
        should_exit = False

    server = _Server()
    thread = start_idle_watchdog(server, tracker, poll_s=0.01)
    try:
        assert attempted.wait(timeout=2)
        assert server.should_exit is False
        verdict[0] = True
        thread.join(timeout=2)
        assert server.should_exit is True
    finally:
        server.should_exit = True
        thread.join(timeout=2)


def test_skewed_backend_waits_for_admitted_work_and_unskewed_backend_does_not_retire(monkeypatch):
    from gateway import code_skew
    from hermes_cli import backend_retirement, web_server_idle_proof

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
    changed = [False]
    checked = Event()

    def detect():
        checked.set()
        return ("old", "new") if changed[0] else None

    monkeypatch.setattr(code_skew, "detect_code_skew", detect)
    tracker = IdleClientTracker()
    tracker.on_open()

    class _Server:
        should_exit = False

    server = _Server()
    thread = start_idle_watchdog(server, tracker, poll_s=0.01)
    try:
        assert checked.wait(timeout=2)
        assert server.should_exit is False
        assert fence.acquire() is True
        fence.release()
        with fence.work() as admitted:
            assert admitted
            changed[0] = True
            checked.clear()
            assert checked.wait(timeout=2)
            assert server.should_exit is False
        thread.join(timeout=2)
        assert server.should_exit is True
    finally:
        server.should_exit = True
        thread.join(timeout=2)


def test_skew_retirement_import_error_is_retried_without_killing_idle_watchdog(monkeypatch):
    from gateway import code_skew
    from hermes_cli import backend_retirement, web_server_idle_proof

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("old", "new"))
    prepare = fence.prepare
    attempts = []

    def flaky_prepare():
        attempts.append(None)
        if len(attempts) == 1:
            raise ImportError("stale module during update")
        return prepare()

    monkeypatch.setattr(fence, "prepare", flaky_prepare)
    tracker = IdleClientTracker()
    tracker.on_open()

    class _Server:
        should_exit = False

    server = _Server()
    thread = start_idle_watchdog(server, tracker, poll_s=0.01)
    try:
        thread.join(timeout=2)
        assert len(attempts) >= 2
        assert server.should_exit is True
    finally:
        server.should_exit = True
        thread.join(timeout=2)


def test_real_git_revision_change_retires_attached_ssh_backend(tmp_path, monkeypatch):
    """Exercise the boot fingerprint, on-disk checkout change, fence and watchdog together."""
    from gateway import code_skew
    from hermes_cli import backend_retirement, web_server_idle_proof

    root = tmp_path / "checkout"
    ref = root / ".git" / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    ref.write_text("a" * 40 + "\n")
    monkeypatch.setattr(code_skew, "_PROJECT_ROOT", root)
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
    code_skew.record_boot_fingerprint()
    assert code_skew.detect_code_skew() is None

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
    tracker = IdleClientTracker()
    tracker.on_open()

    class _Server:
        should_exit = False

    server = _Server()
    thread = start_idle_watchdog(server, tracker, poll_s=0.01)
    try:
        ref.write_text("b" * 40 + "\n")
        assert code_skew.detect_code_skew() == ("a" * 10, "b" * 10)
        thread.join(timeout=2)
        assert server.should_exit is True
        assert fence.acquire() is False
    finally:
        server.should_exit = True
        thread.join(timeout=2)


def test_turn_probe_counts_in_flight_cron_execution():
    """#107485: a cron job mid-run must keep the SSH-isolated backend alive; the run lives outside
    the dashboard session table, in the scheduler's running-job ledger."""
    import cron.scheduler as scheduler
    from hermes_cli.web_server_idle_exit import turn_in_flight

    assert turn_in_flight() is False
    with scheduler._running_lock:
        scheduler._running_job_ids.add("idle-exit-probe-job")
    try:
        assert turn_in_flight() is True
    finally:
        with scheduler._running_lock:
            scheduler._running_job_ids.discard("idle-exit-probe-job")
