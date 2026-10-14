"""POSIX owned-router containment: the llama-server tree dies with its owner, however the owner dies.

All processes and homes are disposable; the owner is a real Hermes process importing the real
``spawn_server(owned=True)`` path, and it is SIGKILLed so no Python cleanup of ours can run.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

pytestmark = pytest.mark.platforms("posix")

_SCRIPT = r'''
import json, subprocess, sys, time
from pathlib import Path
import psutil
root = Path(sys.argv[1])
role = sys.argv[2]
p = psutil.Process()
(root / (role + '.tmp')).write_text(json.dumps({'pid': p.pid, 'created': p.create_time()}))
(root / (role + '.tmp')).replace(root / (role + '.json'))
if role == 'owner':
    from hermes_cli.local_runtime.processes import spawn_server
    proc, _ = spawn_server([sys.executable, __file__, str(root), 'router'], owned=True)
    while True:
        time.sleep(1)
elif role == 'router':
    # llama.cpp's router spawns its model children with posix_spawn and no setpgid, so they
    # stay in the router's group; the stand-in does the same.
    subprocess.Popen([sys.executable, __file__, str(root), 'grandchild'])
    time.sleep(90)
else:
    time.sleep(90)
'''


def _wait(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _read(path):
    assert _wait(path.exists), f"missing child receipt: {path}"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _alive(identity):
    """Zombie-aware: a killed non-child lingers as a zombie until its reaper collects it."""
    try:
        proc = psutil.Process(identity["pid"])
        return proc.create_time() == identity["created"] and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _kill(identity):
    if _alive(identity):
        psutil.Process(identity["pid"]).kill()


# The router and its model child reparent to init once the owner dies, so cleaning them up
# leaves this test's subtree; real signal delivery to a real orphan is the point of the test.
@pytest.mark.live_system_guard_bypass
def test_owner_sigkill_reaps_router_tree_not_an_external_control(tmp_path):
    script = tmp_path / "disposable server.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    with (tmp_path / "children.log").open("w", encoding="utf-8") as log:
        launchers = [subprocess.Popen([sys.executable, str(script), str(tmp_path), role],
                                      env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                     for role in ("control", "owner")]
        ids = {}
        try:
            ids = {role: _read(tmp_path / f"{role}.json") for role in ("owner", "router", "grandchild", "control")}
            assert os.getpgid(ids["grandchild"]["pid"]) == ids["router"]["pid"], "model child left the router's group"

            launchers[1].kill()  # SIGKILL the owner: no atexit, no finally, nothing of ours runs
            launchers[1].wait(timeout=10)

            assert _wait(lambda: not _alive(ids["router"]) and not _alive(ids["grandchild"])), (
                "router tree survived kill -9 of its owner", ids)
            assert _alive(ids["control"]), "an unrelated process was killed"
        finally:
            for identity in ids.values():
                _kill(identity)
            for launcher in launchers:
                if launcher.poll() is None:
                    launcher.kill()
                launcher.wait(timeout=10)


def test_a_graceful_stop_releases_the_routers_group(monkeypatch, tmp_path):
    from hermes_cli.local_runtime import supervisor

    released = []
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    monkeypatch.setattr(supervisor, "release_process_groups",
                        lambda pgids: released.append(("unregister", list(pgids))))
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59998)
    sup.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        sup.stop()
        assert sup.proc.poll() is not None, "stop() left the router running"
        assert released == [("unregister", [sup.proc.pid])], (
            "a router this process stopped itself stayed registered with the owner-death supervisor")
    finally:
        if sup.proc.poll() is None:
            sup.proc.kill()
        sup.proc.wait(timeout=10)
