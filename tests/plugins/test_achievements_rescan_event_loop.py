"""A manual Achievements rescan must not stop the dashboard from answering other requests.

The dashboard serves every page, API call and the embedded terminal from one event loop. Rescan
waits for the session-history scan, which on a long history takes minutes; while it waits, the
rest of the dashboard must keep answering.
"""

import socket
import sys
import threading

import httpx
import pytest
import uvicorn

from tests.waiting import wait_until

PLUGIN = "/api/plugins/hermes-achievements"


@pytest.fixture
def dashboard():
    """The real dashboard app served by uvicorn on loopback, as ``hermes dashboard`` serves it."""
    from hermes_cli import web_server
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    original_routes = list(app.router.routes)
    web_server._dashboard_plugins_cache = None
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error", timeout_graceful_shutdown=1))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        wait_until(lambda: server.started or not thread.is_alive(), what="uvicorn to start")
        assert server.started
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", {_SESSION_HEADER_NAME: _SESSION_TOKEN}, app
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        app.router.routes[:] = original_routes
        web_server._dashboard_plugins_cache = None


def _plugin_scan_lock(app):
    route = next(r for r in app.router.routes if getattr(r, "path", "") == f"{PLUGIN}/rescan")
    return route.endpoint.__globals__["_SCAN_LOCK"]


def _rescan_is_waiting_for_scan() -> bool:
    return any(
        frame.f_code.co_name == "_run_scan_and_update_cache"
        for top in sys._current_frames().values()
        for frame in _stack(top))


def _stack(frame):
    while frame is not None:
        yield frame
        frame = frame.f_back


def test_rescan_waiting_for_a_running_scan_leaves_the_dashboard_answering(dashboard):
    base, auth, app = dashboard
    with httpx.Client(base_url=base, headers=auth, timeout=30) as client:
        assert client.get(f"{PLUGIN}/scan-status").status_code == 200
    scan_lock = _plugin_scan_lock(app)

    # A background scan of the session history is running (it holds the scan lock) when the user
    # clicks Rescan.
    scan_lock.acquire()
    rescan: dict = {}

    def click_rescan():
        with httpx.Client(base_url=base, headers=auth, timeout=60) as client:
            rescan["response"] = client.post(f"{PLUGIN}/rescan")

    clicker = threading.Thread(target=click_rescan, daemon=True)
    clicker.start()
    try:
        wait_until(_rescan_is_waiting_for_scan, what="POST /rescan to wait for the running scan")
        with httpx.Client(base_url=base, headers=auth, timeout=5) as client:
            try:
                status = client.get(f"{PLUGIN}/scan-status")
            except httpx.TimeoutException:
                pytest.fail("GET /scan-status got no answer while POST /rescan waited for the running scan")
        assert status.status_code == 200
    finally:
        scan_lock.release()
        clicker.join(60)
    assert rescan["response"].status_code == 200
    assert rescan["response"].json()["ok"] is True
