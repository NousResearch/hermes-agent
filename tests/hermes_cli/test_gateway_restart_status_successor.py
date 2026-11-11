"""``/api/actions/gateway-restart/status`` answered by the gateway its own restart brought up.

When the gateway serves the dashboard API itself, ``POST /api/gateway/restart`` replaces the very
process that holds the action's in-memory state. The replacement used to answer ``pid null,
running false, exit_code null`` for a restart that worked, and every client polling across the
restart reported it failed. These tests run each process as a real one: the test process is the
old gateway (it holds the home's real gateway runtime lock) and a child process plays the gateway
that comes up after it, acquiring that lock and reading the status through the real routes. The
restart child's argv is the only stand-in (no gateway lifecycle code runs); everything on the
status path is real.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

# The replacement gateway: wait for the old one to hand over (when a port is given), take the
# home's gateway runtime lock as a starting gateway does, then read each named action's status.
_REPLACEMENT = r"""
import json, os, socket, sys
home, out, port, names = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
os.environ["HERMES_HOME"] = home
if port:
    with socket.create_connection(("127.0.0.1", port), timeout=60) as go:
        go.settimeout(60)
        go.recv(1)
from gateway.status import acquire_gateway_runtime_lock, release_gateway_runtime_lock
if not acquire_gateway_runtime_lock():
    sys.exit(5)
try:
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    answers = {}
    for name in names:
        response = client.get(f"/api/actions/{name}/status?lines=1")
        answers[name] = [response.status_code, response.json()]
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "answers": answers}, handle)
finally:
    release_gateway_runtime_lock()
"""

_CHILD_TIMEOUT = 120


def _answer(body: dict) -> dict:
    return {key: body[key] for key in ("pid", "running", "exit_code")}


@pytest.fixture()
def home(monkeypatch):
    from hermes_cli import web_server
    import hermes_cli.web_server_gateway as gw
    from hermes_constants import get_hermes_home

    home = get_hermes_home().resolve()
    monkeypatch.setattr(gw, "_ACTION_LOG_DIR", home / "logs")
    monkeypatch.setattr(web_server, "_LAST_GATEWAY_RESTART", None)
    monkeypatch.setattr(web_server, "GATEWAY_RESTART_COOLDOWN_SECONDS", 0.0)
    for registry in (gw._ACTION_PROCS, gw._ACTION_RESULTS, gw._ACTION_COMMANDS, gw._ACTION_IDS,
                     gw._ACTION_LOG_STARTS):
        monkeypatch.delitem(registry, "gateway-restart", raising=False)
    return home


@pytest.fixture()
def client():
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


@pytest.fixture()
def old_gateway(home):
    """This test process holds the home's gateway runtime lock, as the running gateway does."""
    from gateway.status import acquire_gateway_runtime_lock, release_gateway_runtime_lock

    assert acquire_gateway_runtime_lock()
    yield release_gateway_runtime_lock
    release_gateway_runtime_lock()


class Handover:
    """A loopback socket the replacement waits on until the old gateway released its lock."""

    def __init__(self) -> None:
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(_CHILD_TIMEOUT)
        self.port = self.listener.getsockname()[1]

    def go(self) -> None:
        connection, _ = self.listener.accept()
        with connection:
            connection.sendall(b"g")
        self.listener.close()


@pytest.fixture()
def restart_child(monkeypatch, home):
    """The real restart spawn with the ``hermes gateway restart`` argv swapped for a stand-in."""
    import hermes_cli._launchers as launchers
    from hermes_cli import web_server
    import hermes_cli.web_server_gateway as gw

    env = {**os.environ, "HERMES_HOME": str(home)}
    monkeypatch.setattr(gw, "_action_targets_system_gateway", lambda _subcommand: False)
    monkeypatch.setattr(gw, "_profile_action_environment", lambda _subcommand, _overrides=None: dict(env))
    child: dict[str, list[str]] = {}
    monkeypatch.setattr(launchers, "runtime_command", lambda _root, _subcommand: [sys.executable, *child["argv"]])
    yield child
    recent = web_server._LAST_GATEWAY_RESTART
    if recent is not None and recent[1].poll() is None:
        recent[1].kill()
        recent[1].wait(timeout=10)


def _replacement_argv(home: Path, out: Path, port: int = 0) -> list[str]:
    return ["-c", _REPLACEMENT, str(home), str(out), str(port), "gateway-restart", "gateway-migrate"]


def _run_replacement(home: Path, out: Path) -> dict:
    """A gateway that comes up on its own (as a service manager starts one)."""
    subprocess.run([sys.executable, *_replacement_argv(home, out)], check=True, timeout=_CHILD_TIMEOUT,
                   env={**os.environ, "HERMES_HOME": str(home)}, cwd=str(Path(__file__).resolve().parents[2]))
    return json.loads(out.read_text(encoding="utf-8"))


def _restart(client) -> int:
    response = client.post("/api/gateway/restart")
    assert response.status_code == 200, response.text
    return response.json()["pid"]


def _status(client, name: str = "gateway-restart") -> dict:
    response = client.get(f"/api/actions/{name}/status?lines=1")
    assert response.status_code == 200, response.text
    return _answer(response.json())


def _wait_for_child() -> int:
    from hermes_cli import web_server

    return web_server._LAST_GATEWAY_RESTART[1].wait(timeout=_CHILD_TIMEOUT)


NEVER_STARTED = {"pid": None, "running": False, "exit_code": None}


def test_the_restart_child_that_became_the_gateway_reports_the_restart_finished(
        client, home, old_gateway, restart_child, tmp_path):
    """No service manager: the action child runs the new gateway in-process and answers itself."""
    handover, out = Handover(), tmp_path / "replacement.json"
    restart_child["argv"] = _replacement_argv(home, out, handover.port)

    pid = _restart(client)
    assert _status(client) == {"pid": pid, "running": True, "exit_code": None}
    old_gateway()
    handover.go()
    assert _wait_for_child() == 0

    seen = json.loads(out.read_text(encoding="utf-8"))
    assert seen["pid"] == pid
    status_code, body = seen["answers"]["gateway-restart"]
    assert status_code == 200
    assert _answer(body) == {"pid": pid, "running": False, "exit_code": 0}
    # (e) other actions keep their own answer in the replacement.
    assert _answer(seen["answers"]["gateway-migrate"][1]) == NEVER_STARTED


def test_a_gateway_the_service_manager_started_after_the_child_reports_the_restart_finished(
        client, home, old_gateway, restart_child, tmp_path):
    """Service manager: the child asks the manager and exits; a different process comes up."""
    restart_child["argv"] = ["-c", "pass"]
    pid = _restart(client)
    assert _wait_for_child() == 0
    old_gateway()

    seen = _run_replacement(home, tmp_path / "replacement.json")

    assert seen["pid"] != pid
    assert _answer(seen["answers"]["gateway-restart"][1]) == {"pid": pid, "running": False, "exit_code": 0}


# --- No false success: each control below answers exactly as the base does. ---


@pytest.mark.parametrize("exit_code", [0, 3])
def test_a_separate_dashboard_keeps_its_in_memory_answer(client, home, restart_child, exit_code):
    """(a) A dashboard that is not the gateway reports its own child: running, then its exit code."""
    handover = Handover()
    restart_child["argv"] = ["-c", f"import socket, sys; s = socket.create_connection(('127.0.0.1', {handover.port}),"
                                   f" timeout=60); s.settimeout(60); s.recv(1); sys.exit({exit_code})"]
    pid = _restart(client)
    assert _status(client) == {"pid": pid, "running": True, "exit_code": None}
    handover.go()
    assert _wait_for_child() == exit_code
    assert _status(client) == {"pid": pid, "running": False, "exit_code": exit_code}


def test_a_gateway_never_certifies_a_restart_a_separate_dashboard_started(client, home, restart_child, tmp_path):
    """(a)/(c) The record exists only for a restart the home's own gateway spawned."""
    restart_child["argv"] = ["-c", "pass"]
    _restart(client)
    assert _wait_for_child() == 0

    seen = _run_replacement(home, tmp_path / "replacement.json")

    assert _answer(seen["answers"]["gateway-restart"][1]) == NEVER_STARTED


def test_a_restart_nobody_started_reads_as_never_started(client, home, old_gateway, tmp_path):
    """(b) Nothing spawned in this home: the gateway and a later one both answer all-null."""
    assert _status(client) == NEVER_STARTED
    old_gateway()
    seen = _run_replacement(home, tmp_path / "replacement.json")
    assert _answer(seen["answers"]["gateway-restart"][1]) == NEVER_STARTED


def test_a_later_gateway_never_inherits_an_older_restart(client, home, old_gateway, restart_child, tmp_path):
    """(c) The gateway after the replacement (restarted by other means) does not describe the
    restart, so it answers as if none was started."""
    restart_child["argv"] = ["-c", "pass"]
    _restart(client)
    assert _wait_for_child() == 0
    old_gateway()
    _run_replacement(home, tmp_path / "first.json")

    later = _run_replacement(home, tmp_path / "later.json")

    assert _answer(later["answers"]["gateway-restart"][1]) == NEVER_STARTED


def test_a_new_restart_supersedes_the_record_of_an_older_one(client, home, old_gateway, restart_child, tmp_path):
    """(c) A replacement never reports the restart an older POST started."""
    restart_child["argv"] = ["-c", "pass"]
    older = _restart(client)
    assert _wait_for_child() == 0
    newer = _restart(client)
    assert newer != older
    assert _wait_for_child() == 0
    old_gateway()

    seen = _run_replacement(home, tmp_path / "replacement.json")

    answer = _answer(seen["answers"]["gateway-restart"][1])
    assert answer["pid"] != older
    assert answer in (NEVER_STARTED, {"pid": newer, "running": False, "exit_code": 0})


def test_a_failed_restart_is_reported_failed_and_never_later_certified(
        client, home, old_gateway, restart_child, tmp_path):
    """(d) The child exits non-zero and the old gateway still answers: it reports that code, and a
    gateway started later by other means never turns it into a success."""
    restart_child["argv"] = ["-c", "import sys; sys.exit(1)"]
    pid = _restart(client)
    assert _wait_for_child() == 1
    assert _status(client) == {"pid": pid, "running": False, "exit_code": 1}
    old_gateway()

    seen = _run_replacement(home, tmp_path / "replacement.json")

    assert _answer(seen["answers"]["gateway-restart"][1]) == NEVER_STARTED


@pytest.mark.parametrize("damage", ["not json {", "corrupt-shape", "directory"])
def test_an_unreadable_record_degrades_to_the_base_answer(client, home, old_gateway, tmp_path, damage):
    """A missing, unreadable or corrupt record never raises or answers 500."""
    record = home / "logs" / "gateway-restart.record.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    if damage == "directory":
        record.mkdir()
    else:
        record.write_text(damage if damage != "corrupt-shape" else json.dumps({"action": "x"}), encoding="utf-8")
    old_gateway()

    seen = _run_replacement(home, tmp_path / "replacement.json")

    status_code, body = seen["answers"]["gateway-restart"]
    assert status_code == 200
    assert _answer(body) == NEVER_STARTED


_OUT_OF_RANGE = {
    "a spawn time past any float": {"spawned_at": 10**400},
    "an infinite spawn time": {"spawned_at": float("inf")},
    "a pid past any pid": {"action": [10**400, 1]},
    "a start time past any clock": {"action": [1, 10**400]},
    "a successor past any pid": {"successor": [10**400, 1]},
}


@pytest.mark.parametrize("damage", [*_OUT_OF_RANGE, "nesting past the parser's depth"])
def test_a_record_with_unrepresentable_values_degrades_to_the_base_answer(client, home, old_gateway, damage):
    """Well-formed JSON whose numbers no clock or pid can hold (or that nests past what the parser
    accepts) reads as no record: the gateway answers 200 with the base answer, never a 500."""
    from gateway.status import get_process_start_time

    record = home / "logs" / "gateway-restart.record.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    if damage in _OUT_OF_RANGE:
        # Every other field would qualify this process, so only the damaged value can stop it.
        me = [os.getpid(), get_process_start_time(os.getpid())]
        fields = {"action": me, "spawner": [1, 1], "spawned_at": time.time(), "exit_code": None,
                  "successor": None, **_OUT_OF_RANGE[damage]}
        record.write_text(json.dumps(fields), encoding="utf-8")
    else:
        record.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")

    assert _status(client) == NEVER_STARTED
