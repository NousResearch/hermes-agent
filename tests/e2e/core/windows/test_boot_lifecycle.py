"""Boot and lifecycle of the real Hermes processes on native Windows.

* ``hermes.exe --version`` — the console-script shim users launch starts and reports
  the installed package version.
* ``hermes.exe chat -q`` — one oneshot turn against the loopback provider: the reply is
  printed, the prompt reached the wire, and both messages persisted to ``state.db`` under
  the session id the CLI announced.
* ``hermes serve`` (the Desktop backend) — announces its port on stdout, and after the
  Desktop's Windows quit path (``taskkill /T /F``) leaves no process behind that it
  spawned (found by ownership of the scratch profile, not by parent links, which a
  detached grandchild does not keep) and a fresh backend boots again over the leftover
  host records.
* ``hermes gateway run`` + ``hermes gateway stop`` — the Windows graceful-stop IPC (stop
  marker, not TerminateProcess) drains the gateway: it exits on its own, records
  ``stopped``, removes its pid file, and nothing it spawned survives.
"""

from __future__ import annotations

import importlib.metadata
import json
import queue
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from tests.e2e.core.windows._helpers import (
    WinHome,
    db_rows,
    hermes,
    hermes_argv,
    hermes_exe,
    kill_owned,
    last_user,
    make_home,
    nonce,
    owned_processes,
    owned_survivors,
    run,
    taskkill_tree,
    wait_until,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

# Real process-tree kills (taskkill /T /F, psutil) of children this test spawned with a
# scratch USERPROFILE/HERMES_HOME; the live-system guard would refuse the taskkill argv.
pytestmark = [pytest.mark.windows_only, pytest.mark.integration, pytest.mark.live_system_guard_bypass]

READY_TIMEOUT = 120.0


def test_version_reports_installed_package(tmp_path: Path) -> None:
    home = make_home(tmp_path, "http://127.0.0.1:9/v1")  # never contacted by --version
    res = run([str(hermes_exe()), "--version"], home, timeout=120)
    assert res.returncode == 0, res.tail()
    version = importlib.metadata.version("hermes-agent")
    assert f"v{version}" in res.stdout, f"--version does not report the installed {version}:\n{res.tail()}"


def test_chat_oneshot_turn_persists(tmp_path: Path) -> None:
    prompt_id, reply_id = nonce("PROMPT"), nonce("REPLY")
    with FakeLLMServer([Text(f"The answer is {reply_id}.")]) as srv:
        home = make_home(tmp_path, srv.base_url)
        res = run([str(hermes_exe()), "chat", "-q", f"Say the code {prompt_id}", "-Q"], home)
        assert res.returncode == 0, res.tail()
        assert reply_id in res.stdout, f"reply not printed:\n{res.tail()}"
        mains = srv.main_requests()
        assert mains and prompt_id in last_user(mains[0]), "the prompt never reached the provider"

    announced = re.search(r"session_id:\s*(\S+)", res.stdout + res.stderr)
    assert announced, f"CLI did not announce a session id:\n{res.tail()}"
    rows = db_rows(home, "SELECT session_id, role, content FROM messages ORDER BY id")
    by_role = {r["role"]: r for r in rows}
    assert prompt_id in (by_role.get("user") or {"content": ""})["content"], rows
    assert reply_id in (by_role.get("assistant") or {"content": ""})["content"], rows
    assert {r["session_id"] for r in rows} == {announced.group(1)}, (
        f"persisted rows belong to {[r['session_id'] for r in rows]}, CLI announced {announced.group(1)}")


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _spawn_serve(home: WinHome) -> subprocess.Popen:
    # Desktop spawn shape (electron/main.ts): token + desktop flag in env, stdin closed.
    env = {"HERMES_DESKTOP": "1", "HERMES_DASHBOARD_SESSION_TOKEN": nonce("tok"),
           "TERMINAL_CWD": str(home.project)}
    return subprocess.Popen(
        hermes_argv("serve", "--host", "127.0.0.1", "--port", "0"), cwd=home.profile, env=home.env(env),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def _serve_ready(home: WinHome) -> tuple[subprocess.Popen, int]:
    """Spawn serve and wait for its READY port. A daemon pump keeps draining stdout after
    READY, as the Desktop does, so a full pipe can never stall the backend."""
    proc = _spawn_serve(home)
    found: "queue.Queue[int | None]" = queue.Queue()
    seen: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        announced = False
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            seen.append(line)
            m = None if announced else re.search(r"HERMES_BACKEND_READY port=(\d+)", line)
            if m:
                announced = True
                found.put(int(m.group(1)))
        found.put(None)

    threading.Thread(target=pump, daemon=True).start()
    try:
        port = found.get(timeout=READY_TIMEOUT)
    except queue.Empty:
        port = None
    assert port is not None, f"serve (rc={proc.poll()}) announced no READY port:\n{''.join(seen)[-3000:]}"
    assert _port_open(port), f"READY announced port {port} but nothing accepts on it"
    return proc, port


def test_serve_tree_kill_leaves_no_orphans_and_reboots(tmp_path: Path) -> None:
    with FakeLLMServer() as srv:
        home = make_home(tmp_path, srv.base_url)
        started = time.time()
        try:
            first, port = _serve_ready(home)
            # Positive control: the ownership scan must see the backend itself, or an empty
            # scan after the kill would prove nothing.
            owned = {p.pid for p in owned_processes(home, since=started)}
            assert first.pid in owned, f"ownership scan cannot see serve pid {first.pid} (saw {owned})"

            killed = taskkill_tree(first.pid)
            assert killed.returncode == 0, killed.stderr
            # Ownership, not ancestry: anything the backend spawned detached (a broken
            # parent link taskkill /T cannot follow) still carries this profile's
            # HERMES_HOME / cwd, and is an orphan the Desktop quit leaves behind.
            left = owned_survivors(home, since=started, timeout=30)
            assert not left, f"processes of the killed backend outlived taskkill /T /F: {left}"
            wait_until(lambda: not _port_open(port), 30, f"port {port} to be released")

            second, port2 = _serve_ready(home)
            assert second.poll() is None and _port_open(port2)
        finally:
            kill_owned(home, since=started)


def _gateway_state(home: WinHome) -> dict:
    path = home.hermes_home / "gateway_state.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def test_gateway_stop_drains_gracefully(tmp_path: Path) -> None:
    with FakeLLMServer() as srv:
        home = make_home(tmp_path, srv.base_url)
        log = tmp_path / "gateway.log"
        started = time.time()
        with log.open("wb") as fh:
            gw = subprocess.Popen(hermes_argv("gateway", "run"), cwd=home.project, env=home.env(),
                                  stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: _gateway_state(home).get("gateway_state") == "running" or gw.poll() is not None,
                       READY_TIMEOUT, "gateway_state.json to report running")
            assert gw.poll() is None, f"gateway exited {gw.returncode} during boot:\n{log.read_text(errors='replace')}"

            res = hermes(home, "gateway", "stop")
            assert res.returncode == 0 and "Stopped" in res.stdout, res.tail()
            code = gw.wait(timeout=60)
            state = _gateway_state(home)
            assert state.get("gateway_state") == "stopped", (
                f"gateway did not drain through its graceful path (rc={code}); last state {state}\n"
                f"{log.read_text(errors='replace')[-3000:]}")
            assert not (home.hermes_home / "gateway.pid").exists(), "gateway.pid left behind after stop"
            left = owned_survivors(home, since=started, timeout=30)
            assert not left, f"processes of the gateway survived stop: {left}"
        finally:
            kill_owned(home, since=started)
