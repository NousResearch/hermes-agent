"""One backend migration at a time per home: two runs started together give one migration and one refusal.

Real processes race through the real ``hermes`` entry point (``backend_migration_race_host``: the engine
tests' fake host for the service manager, process table and network; real home files, manifest and lock).
The first run is held inside its preflight; the second must be refused before it reads any state (exit 3,
says how to watch the first, never touches the manifest), and ``--status`` / ``--dry-run`` must not wait.
A run that is killed while it holds the lock leaves none behind (the kernel drops it with the process).
Then two clicks reach the real app at once: one 200, one 409, one migration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hermes_cli import gateway_migrate_backends_apply as engine
from tests.hermes_cli.backend_migration_race_host import MODULE, RaceHost
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux", "macos")  # the race host's detached runs use start_new_session

_REPO = Path(__file__).resolve().parents[2]
_MIGRATE = ["gateway", "migrate", "--backends"]


@pytest.fixture
def race(tmp_path, monkeypatch):
    home, race_dir = tmp_path / "hermes-home", tmp_path / "race"
    (home / "logs").mkdir(parents=True)
    race_dir.mkdir()
    (home / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (home / ".env").write_text("OPENAI_API_KEY=***\n", encoding="utf-8")
    os.chmod(home / ".env", 0o600)
    monkeypatch.setenv("HERMES_HOME", str(home))
    env = {**os.environ, "HERMES_HOME": str(home), "RACE_DIR": str(race_dir), "PYTHONPATH": str(_REPO)}
    yield home, race_dir, env
    (race_dir / "release").write_text("", encoding="utf-8")  # never leave a held run behind


def _start(env, *flags, hold=False):
    return subprocess.Popen([sys.executable, "-m", MODULE, *_MIGRATE, *flags], cwd=_REPO,
                            env={**env, "RACE_HOLD": "1" if hold else "0"}, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def _run(env, *flags):
    proc = _start(env, *flags)
    out, _ = proc.communicate(timeout=60)
    return proc.returncode, out


def _held(race_dir):
    return [p for p in race_dir.iterdir() if p.name.startswith("held-")]


def test_two_cli_applies_at_once_give_one_migration_and_one_clear_refusal(race):
    home, race_dir, env = race
    first = _start(env, "--apply", "--yes", hold=True)
    wait_until(lambda: _held(race_dir), timeout=60, what="the first run inside its preflight")

    code, out = _run(env, "--apply", "--yes")
    assert code == engine.EXIT_BLOCKED, out
    assert "already running for this home" in out and "--status" in out
    assert "recorded the current state" not in out and not engine.manifest_path(home).exists()
    # Neither the status nor the plan waits for the run that holds the lock.
    assert _run(env, "--status", "--json")[0] == 0
    assert _run(env, "--apply", "--dry-run", "--json")[0] == 0

    (race_dir / "release").write_text("", encoding="utf-8")
    out, _ = first.communicate(timeout=60)
    assert first.returncode == engine.EXIT_OK, out
    assert "Migrated" in out
    manifest = json.loads(engine.manifest_path(home).read_text(encoding="utf-8"))
    assert manifest["pid"] == first.pid and manifest["state"] == "done"


def test_a_killed_run_never_leaves_a_stale_lock(race):
    from hermes_cli import gateway_migrate_backends_lock as run_lock
    home, _race_dir, env = race
    holder = subprocess.Popen(
        [sys.executable, "-c", "import sys, time; from hermes_cli import gateway_migrate_backends_lock as l; "
         "print(l.acquire(sys.argv[1]), flush=True); time.sleep(120)", str(home)],
        cwd=_REPO, env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "True"
        assert run_lock.acquire(home) is False  # held by a live run
        assert run_lock.holder_pid(home) == holder.pid
        holder.kill()  # the migrator dies mid-run
        holder.wait(timeout=30)
        assert run_lock.acquire(home) is True  # the kernel dropped it: the next run can roll back at once
    finally:
        run_lock.release()
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=30)
        holder.stdout.close()


# The real launcher, as the route spawns it, with its own up-front plan never held: only the background
# run it starts stops in its preflight (``RACE_HOLD=1`` set just before that spawn). ``RACE_SCOPE`` makes
# the launcher's host see the old backend as a system unit (manual steps only).
_LAUNCHER = """
import os, sys
from tests.hermes_cli import backend_migration_race_host as race
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost
spawn = race.RaceHost.spawn_detached
def spawn_held(self, flags, log):
    os.environ["RACE_HOLD"] = os.environ.get("RACE_CHILD_HOLD", "0")
    return spawn(self, flags, log)
race.RaceHost.spawn_detached = spawn_held
if os.environ.get("RACE_SCOPE"):
    race.RaceHost.__init__ = lambda self, home, **kw: FakeHost.__init__(self, home, scope=os.environ["RACE_SCOPE"])
race.main()
"""


def _route_spawns_the_real_launcher(monkeypatch, env, race_dir, on_spawn=lambda: None, **extra_env):
    from hermes_cli.web_routers import backend_migration as routes
    monkeypatch.setattr("hermes_cli.web_server_gateway._ACTION_LOG_DIR", race_dir / "action-logs")
    routes.action_log_path().parent.mkdir(parents=True)

    def spawn_launcher(subcommand, name, *, env_overrides=None):  # the real launcher, in a real process
        on_spawn()
        with routes.action_log_path().open("ab") as log:  # where the route reads the launcher's answer
            return subprocess.Popen([sys.executable, "-c", _LAUNCHER, *subcommand], cwd=_REPO,
                                    env={**env, **(env_overrides or {}), "RACE_HOLD": "0", **extra_env},
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.DEVNULL)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn_launcher)


def test_two_apply_clicks_at_once_start_one_run_and_answer_409_to_the_other(race, monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.gateway_migrate_backends_detach import read_refusal
    from hermes_cli.web_routers import backend_migration as routes
    home, race_dir, env = race

    _route_spawns_the_real_launcher(monkeypatch, env, race_dir, RACE_CHILD_HOLD="1")
    monkeypatch.setattr(routes, "_host", lambda _home: RaceHost(home))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    barrier, answers = threading.Barrier(2), []

    def click():
        barrier.wait()
        r = client.post("/api/gateway/migrate/backends/apply", json={})
        answers.append((r.status_code, r.json()))

    clicks = [threading.Thread(target=click) for _ in range(2)]
    for t in clicks:
        t.start()
    for t in clicks:
        t.join(timeout=120)

    assert sorted(code for code, _ in answers) == [200, 409], answers
    refused = next(body for code, body in answers if code == 409)
    assert refused["error"] == "already_running" and "--status" in refused["detail"]
    assert read_refusal(home) is None  # the loser left no refusal over the winner's status
    # Exactly one background run reaches its preflight (the other was refused before reading any state).
    assert len(wait_until(lambda: _held(race_dir), timeout=60, what="the winner in its preflight")) == 1

    (race_dir / "release").write_text("", encoding="utf-8")
    log_path = home / "logs" / "backend-migration.log"
    wait_until(lambda: (engine.read_manifest(home) or {}).get("state") == "done"
               and "Migrated" in log_path.read_text(encoding="utf-8"), timeout=60, what="the one migration")
    assert len(_held(race_dir)) == 1
    log = log_path.read_text(encoding="utf-8")
    assert log.count("recorded the current state") == 1 and log.count("Migrated") == 1, log


def test_a_real_launcher_exit_6_is_the_routes_unsupported_refusal(race, monkeypatch):
    """The backend became manual-only (a system unit) after the route's own plan: the real ``--apply
    --detach`` launcher exits 6 with the steps and starts nothing; the route answers 409 ``unsupported``
    with the steps, as its own pre-check does."""
    from hermes_cli import web_server
    from hermes_cli.web_routers import backend_migration as routes
    home, race_dir, env = race
    host = RaceHost(home)  # a user unit when the route plans; a system unit from the launch on

    def becomes_system_unit():
        host.scope = "system"

    _route_spawns_the_real_launcher(monkeypatch, env, race_dir, becomes_system_unit, RACE_SCOPE="system")
    monkeypatch.setattr(routes, "_host", lambda _home: host)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    r = client.post("/api/gateway/migrate/backends/apply", json={})
    assert r.status_code == 409, r.text
    body = r.json()
    assert (body["error"], body["launcher_exit_code"]) == ("unsupported", 6)
    [u] = body["unsupported"]
    assert u["steps"] and u["unit"] == "hermes-dashboard.service", u
    assert not engine.manifest_path(home).exists() and _held(race_dir) == []
