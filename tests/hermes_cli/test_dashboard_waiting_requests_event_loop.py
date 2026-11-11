"""A dashboard request that has to wait must not stop the dashboard from answering other requests.

The dashboard serves every page, API call and the embedded terminal from one event loop. Each test
starts a request that waits on something outside the request (the Kanban board's write lock, the
skill-seeding child process, a WhatsApp pairing being stopped elsewhere) and requires the dashboard
to answer another request while it waits.
"""

import errno
import os
import socket
import sys
import threading
from pathlib import Path

import httpx
import pytest
import uvicorn

from tests.waiting import wait_until


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
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", {_SESSION_HEADER_NAME: _SESSION_TOKEN}
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        app.router.routes[:] = original_routes
        web_server._dashboard_plugins_cache = None


def _send_in_background(base, auth, method, path, **kwargs):
    result: dict = {}

    def send():
        with httpx.Client(base_url=base, headers=auth, timeout=60) as client:
            result["response"] = client.request(method, path, **kwargs)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    return thread, result


def _a_thread_is_in(*function_names: str) -> bool:
    """Whether one thread's stack holds every named function (the request is waiting inside them)."""
    for top in sys._current_frames().values():
        names = set()
        frame = top
        while frame is not None:
            names.add(frame.f_code.co_name)
            frame = frame.f_back
        if names.issuperset(function_names):
            return True
    return False


def _assert_dashboard_answers(base, auth, while_what: str) -> None:
    with httpx.Client(base_url=base, headers=auth, timeout=5) as client:
        try:
            health = client.get("/api/health")
        except httpx.TimeoutException:
            pytest.fail(f"GET /api/health got no answer while {while_what}")
    assert health.status_code == 200


def test_attachment_upload_waiting_for_the_board_write_lock_leaves_the_dashboard_answering(dashboard):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    base, auth = dashboard
    kb.init_db()
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="report")
    finally:
        conn.close()

    # Another writer (a dispatcher claiming tasks, a worker completing one) holds the board's write lock.
    holder = kbc.connect()
    holder.execute("BEGIN IMMEDIATE")
    upload, result = _send_in_background(
        base, auth, "POST", f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("notes.txt", b"draft", "text/plain")})
    try:
        wait_until(lambda: _a_thread_is_in("upload_task_attachment", "write_txn"),
                   what="the upload to wait for the board's write lock")
        _assert_dashboard_answers(base, auth, "a Kanban attachment upload waited for the board's write lock")
    finally:
        holder.execute("ROLLBACK")
        holder.close()
        upload.join(60)
    assert result["response"].status_code == 200
    assert result["response"].json()["attachment"]["filename"] == "notes.txt"


def _open_for_writing(fifo: Path):
    """A write end of *fifo* once a reader has opened it, else None."""
    try:
        return os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno == errno.ENXIO:
            return None
        raise


@pytest.mark.platforms("posix")
def test_profile_creation_waiting_for_skill_seeding_leaves_the_dashboard_answering(dashboard, tmp_path, monkeypatch):
    base, auth = dashboard
    # The seeding child reads every bundled SKILL.md; this one is a FIFO, so the child waits on it until
    # the test writes, in place of a slow copy of the full bundled skill tree.
    held = tmp_path / "bundled-skills" / "held" / "SKILL.md"
    held.parent.mkdir(parents=True)
    os.mkfifo(held)
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(held.parent.parent))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    create, result = _send_in_background(base, auth, "POST", "/api/profiles", json={"name": "research"})
    writer = None
    try:
        writer = wait_until(lambda: _open_for_writing(held), timeout=30,
                            what="the skill-seeding child to read the bundled skills")
        _assert_dashboard_answers(base, auth, "a profile creation waited for skill seeding")
    finally:
        writer = writer or _open_for_writing(held)
        if writer is not None:
            os.write(writer, b"---\nname: held\n---\n")
            os.close(writer)
        create.join(60)
    assert result["response"].status_code == 200, result["response"].text
    assert result["response"].json()["name"] == "research"


def test_whatsapp_cancel_waiting_for_a_pairing_being_stopped_leaves_the_dashboard_answering(dashboard):
    from hermes_cli.web_routers import messaging

    base, auth = dashboard
    # The pairing expiry timer holds the onboarding lock while it stops a bridge (up to 6 s for one that
    # ignores SIGTERM) when the user clicks Cancel.
    messaging._whatsapp_onboarding_lock.acquire()
    cancel, result = _send_in_background(base, auth, "DELETE", "/api/messaging/whatsapp/onboarding/no-such-pairing")
    try:
        wait_until(lambda: _a_thread_is_in("cancel_whatsapp_onboarding"),
                   what="the cancel to wait for the onboarding lock")
        _assert_dashboard_answers(base, auth, "a WhatsApp setup cancel waited for a pairing being stopped")
    finally:
        messaging._whatsapp_onboarding_lock.release()
        cancel.join(60)
    assert result["response"].status_code == 200
    assert result["response"].json() == {"ok": True}
