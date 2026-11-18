"""One backend migration at a time per home: two runs started together give one migration and one refusal.

Real processes race through the real ``hermes`` entry point (``backend_migration_race_host``: the engine
tests' fake host for the service manager, process table and network; real home files, manifest and lock).
The first run is held inside its preflight; the second must be refused before it reads any state (exit 3,
says how to watch the first, never touches the manifest), and ``--status`` / ``--dry-run`` must not wait.
A run that is killed while it holds the lock leaves none behind (the kernel drops it with the process).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from tests.hermes_cli.backend_migration_race_host import MODULE
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

