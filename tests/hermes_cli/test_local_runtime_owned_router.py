"""POSIX owned-router containment: the llama-server tree dies with its owner, however the owner dies.

All processes and homes are disposable; the owner is a real Hermes process importing the real
``spawn_server(owned=True)`` path, and it is SIGKILLed so no Python cleanup of ours can run.
"""

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

from tests.waiting import wait_until

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


def _read(path):
    wait_until(path.exists, timeout=15, what=f"child receipt {path}")
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

            wait_until(lambda: not _alive(ids["router"]) and not _alive(ids["grandchild"]), timeout=15,
                       what=f"router tree to die after kill -9 of its owner: {ids}")
            assert _alive(ids["control"]), "an unrelated process was killed"
        finally:
            for identity in ids.values():
                _kill(identity)
            for launcher in launchers:
                if launcher.poll() is None:
                    launcher.kill()
                launcher.wait(timeout=10)


def test_a_graceful_stop_releases_the_routers_group(monkeypatch, tmp_path):
    from hermes_cli import process_identity
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59998)
    sup.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                stdin=subprocess.DEVNULL, start_new_session=True)
    process_identity.own_process_groups([sup.proc.pid])
    try:
        sup.stop()
        assert sup.proc.poll() is not None, "stop() left the router running"
        assert sup.proc.pid not in process_identity._supervised_pgids, (
            "a router this process stopped itself stayed registered with the owner-death supervisor")
    finally:
        if sup.proc.poll() is None:
            sup.proc.kill()
        sup.proc.wait(timeout=10)
        process_identity.release_process_groups([sup.proc.pid])


def test_stop_after_a_router_crash_still_stops_its_surviving_model_child(monkeypatch, tmp_path):
    """The watcher's restart backoff: the router died, its model child (same group) still runs."""
    from hermes_cli import process_identity
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59997)
    receipt = tmp_path / "model.pid"
    sup.proc = subprocess.Popen(  # detached-spawn: ok — test router stand-in; its group is killed below
        [sys.executable, "-c",
         "import subprocess, sys; c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']);"
         f" open({str(receipt)!r}, 'w').write(str(c.pid))"],
        stdin=subprocess.DEVNULL, start_new_session=True)
    process_identity.own_process_groups([sup.proc.pid])
    pgid = sup.proc.pid
    try:
        sup.proc.wait(timeout=30)  # the router "crashed"; the model child is still in its group
        model = psutil.Process(int(wait_until(lambda: receipt.exists() and receipt.read_text(encoding="utf-8"),
                                              what="model child receipt")))
        assert os.getpgid(model.pid) == pgid
        sup.stop()
        wait_until(lambda: not model.is_running() or model.status() == psutil.STATUS_ZOMBIE, timeout=10,
                   what="stop() to stop the crashed router's surviving model child")
        assert pgid not in process_identity._supervised_pgids
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, 9)  # windows-footgun: ok — posix-marked test
        process_identity.release_process_groups([pgid])


def test_a_crash_restart_stops_the_old_routers_survivors_and_releases_its_group(monkeypatch, tmp_path):
    """The watcher restarts a crashed router: the old group's model child must not outlive the restart."""
    from hermes_cli import process_identity
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    monkeypatch.setattr(supervisor, "_RESTART_BACKOFF_S", (0,))
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59996)
    monkeypatch.setattr(sup, "_reap_orphaned_children", lambda: None)
    monkeypatch.setattr(sup, "_wait_health", lambda timeout: None)
    monkeypatch.setattr(sup, "_spawn", lambda: setattr(sup, "proc", None))  # the replacement; ends the watch
    receipt = tmp_path / "model.pid"
    old = subprocess.Popen(  # detached-spawn: ok — test router stand-in; its group is killed below
        [sys.executable, "-c",
         "import subprocess, sys; c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']);"
         f" open({str(receipt)!r}, 'w').write(str(c.pid))"],
        stdin=subprocess.DEVNULL, start_new_session=True)
    process_identity.own_process_groups([old.pid])
    try:
        old.wait(timeout=30)  # the router crashed; its model child is still in its group
        model = psutil.Process(int(wait_until(lambda: receipt.exists() and receipt.read_text(encoding="utf-8"),
                                              what="model child receipt")))
        sup.proc = old
        sup._watch()
        wait_until(lambda: not model.is_running() or model.status() == psutil.STATUS_ZOMBIE, timeout=10,
                   what="the restart to stop the crashed router's surviving model child")
        assert old.pid not in process_identity._supervised_pgids
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(old.pid, 9)  # windows-footgun: ok — posix-marked test
        process_identity.release_process_groups([old.pid])
