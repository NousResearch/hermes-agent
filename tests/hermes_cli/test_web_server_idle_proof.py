"""A Desktop-pooled ``hermes serve`` child must PROVE it is idle before the Desktop may retire it
for a foreground open (supersedes #104871, whose idle signal was renderer bookkeeping).

Occupied is not busy: the renderer keeps every pinned tile's backend keepalive-fresh forever, so
the Desktop needs the backend's own ledgers — running sessions, running cron jobs (the desktop
child runs the in-process ticker), and prompts waiting on a human. Anything unreadable is
``None`` and the caller treats it as busy.
"""

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import json
from pathlib import Path

import pytest

from hermes_cli.web_server_idle_proof import idle_proof, pending_human_input

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_idle_proof_is_true_only_when_every_ledger_is_provably_empty():
    assert idle_proof(turn_probe=lambda: False, input_probe=lambda: 0) == {"idle": True, "reason": None}
    assert idle_proof(turn_probe=lambda: True, input_probe=lambda: 0) == {
        "idle": False, "reason": "turn_in_flight", "detail": None}
    assert idle_proof(turn_probe=lambda: "session:abc", input_probe=lambda: 0)["detail"] == "session:abc"
    assert idle_proof(turn_probe=lambda: False, input_probe=lambda: 1) == {
        "idle": False, "reason": "awaiting_human_input"}
    # Fail closed: an indeterminate probe is never reported as idle.
    assert idle_proof(turn_probe=lambda: None, input_probe=lambda: 0)["idle"] is None
    assert idle_proof(turn_probe=lambda: False, input_probe=lambda: None)["idle"] is None


def test_idle_proof_reads_the_real_cron_and_human_input_ledgers():
    """The probe must see a running cron job (invisible to the renderer) and a pending approval
    or open clarify request in the live process ledgers, not a renderer-published flag."""
    import cron.scheduler as scheduler
    from tools import approval
    from tui_gateway import server_requests

    assert idle_proof()["idle"] is True

    with scheduler._running_lock:
        scheduler._running_job_ids.add("idle-proof-live-job")
    try:
        # The busy verdict names the ledger and the job, so a backend that will not retire is diagnosable.
        assert idle_proof() == {"idle": False, "reason": "turn_in_flight", "detail": "cron:idle-proof-live-job"}
    finally:
        with scheduler._running_lock:
            scheduler._running_job_ids.discard("idle-proof-live-job")

    from tools.approval_gateway_wait import _ApprovalEntry

    entry = _ApprovalEntry({"request_id": "idle-proof-approval"})
    with approval._lock:
        approval._gateway_queues.setdefault("idle-proof-session", []).append(entry)
    try:
        assert pending_human_input() == 1
        assert idle_proof() == {"idle": False, "reason": "awaiting_human_input"}
    finally:
        with approval._lock:
            approval._gateway_queues.pop("idle-proof-session", None)

    req = server_requests.ServerRequest("sid-idle-proof", "clarify", {})
    with server_requests._lock:
        server_requests._open[req.id] = req
    try:
        assert idle_proof()["idle"] is False
    finally:
        with server_requests._lock:
            server_requests._open.pop(req.id, None)

    assert idle_proof()["idle"] is True


def test_queued_prompts_and_unwinding_workers_are_not_idle(monkeypatch):
    from tui_gateway import server

    session = {"running": False, "history_lock": threading.RLock()}
    monkeypatch.setitem(server._sessions, "queued-retirement", session)
    server._enqueue_prompt(session, "accepted follow-up", None)
    assert idle_proof()["idle"] is False
    session.pop("queued_prompt")
    release = threading.Event()
    thread = threading.Thread(target=lambda: release.wait(10), daemon=True)
    session["_run_thread"] = thread
    thread.start()
    try:
        assert idle_proof()["idle"] is False
    finally:
        release.set()
        thread.join(10)
    assert idle_proof()["idle"] is True


# ---------------------------------------------------------------------------
# Live children: three real desktop-shaped ``serve`` processes, one held busy
# ---------------------------------------------------------------------------

pytestmark_live = pytest.mark.skipif(sys.platform == "win32", reason="POSIX serve-runner path under test")

TOKEN = "idle-proof-live-token"


def _spawn_desktop_child(tmp_path: Path, name: str, *, busy: bool) -> subprocess.Popen:
    """Mirror the Desktop pool spawn: ``HERMES_DESKTOP=1`` (in-process cron ticker), a per-child
    HERMES_HOME, an ephemeral port, the session token the Desktop probes with. The busy child
    registers a cron run in the scheduler's running-job ledger before it serves — the same
    ledger a real fire uses and the one the probe reads."""
    home = tmp_path / name
    home.mkdir()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for k in ("HERMES_PARENT_PID", "HERMES_PARENT_START_MARKER", "HERMES_PARENT_NONCE"):
        env.pop(k, None)
    env.update(
        HERMES_HOME=str(home),
        HERMES_SERVE_HEADLESS="1",
        HERMES_DESKTOP="1",
        HERMES_DASHBOARD_SESSION_TOKEN=TOKEN,
        PYTHONUNBUFFERED="1",
    )
    hold = (
        "import cron.scheduler as s\n"
        "with s._running_lock:\n"
        "    s._running_job_ids.add('live-idle-proof-job')\n"
    ) if busy else ""
    code = (
        hold
        + "from hermes_cli.web_server import start_server\n"
        "start_server(host='127.0.0.1', port=0, open_browser=False, headless=True)\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )


def _read_until(proc: subprocess.Popen, token: str, timeout: float = 120.0):
    lines: list[str] = []
    hit = threading.Event()

    def _pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if token in line:
                hit.set()
                return

    threading.Thread(target=_pump, daemon=True).start()
    hit.wait(timeout)
    return hit.is_set(), lines


def _probe(port: int, token: str | None = TOKEN) -> tuple[int, dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/health/idle",
                                 headers={"X-Hermes-Session-Token": token} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def _await_verdict(port: int, accept, timeout: float = 60.0) -> tuple[tuple[int, dict], list]:
    """Probe until ``accept(verdict)`` or the deadline; returns the last verdict and every one seen."""
    deadline = time.monotonic() + timeout
    seen: list[tuple[int, dict]] = []
    while True:
        verdict = _probe(port)
        seen.append(verdict)
        if accept(verdict) or time.monotonic() >= deadline:
            return verdict, seen
        time.sleep(0.2)


@pytestmark_live
def test_live_pooled_children_prove_idle_or_busy_over_the_desktop_probe(tmp_path):
    """Three real children at the Desktop's pool cap; exactly one is busy. The Desktop's probe
    must get ``idle: false`` from the busy one and ``idle: true`` from the two residents that are
    merely occupied. The probe is token-gated: an unauthenticated caller learns nothing."""
    children = {
        "resident-a": _spawn_desktop_child(tmp_path, "resident-a", busy=False),
        "resident-b": _spawn_desktop_child(tmp_path, "resident-b", busy=False),
        "cron-busy": _spawn_desktop_child(tmp_path, "cron-busy", busy=True),
    }
    try:
        ports = {}
        for name, proc in children.items():
            ready, lines = _read_until(proc, "HERMES_BACKEND_READY")
            assert ready, f"{name}: no READY sentinel; output:\n{''.join(lines)}"
            ready_line = next(l for l in lines if "HERMES_BACKEND_READY" in l)
            ports[name] = int(ready_line.strip().rsplit("port=", 1)[1])

        # READY precedes quiescence: the desktop child's cron ticker runs its first tick right at
        # start and holds retirement admission through the whole scan (``retirement_admission``),
        # which on a loaded runner outlasts the first probe. A resident is "provably idle" once that
        # startup work drains, so wait for the idle verdict instead of sampling once.
        idle = (200, {"ok": True, "idle": True, "reason": None})
        for name in ("resident-a", "resident-b"):
            verdict, seen = _await_verdict(ports[name], lambda v: v == idle)
            assert verdict == idle, f"{name} never proved idle; observed: {seen}"
        busy = (200, {"ok": True, "idle": False, "reason": "turn_in_flight", "detail": "cron:live-idle-proof-job"})
        verdict, seen = _await_verdict(ports["cron-busy"], lambda v: v == busy)
        assert verdict == busy, f"cron-busy never named its ledger; observed: {seen}"
        # Whatever startup work shadowed the cron ledger, the busy child never once claimed idle.
        assert all(body.get("idle") is False for _status, body in seen), seen

        status, body = _probe(ports["resident-a"], token=None)
        assert status == 401 and "idle" not in body
    finally:
        for proc in children.values():
            proc.terminate()
        for proc in children.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
