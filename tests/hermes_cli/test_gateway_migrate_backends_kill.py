"""A real migration process SIGKILLed right after it renamed config.yaml or .env into place, then recovered.

The real ``hermes`` entry point runs in its own process (``backend_migration_race_host``: the engine tests'
fake service manager, process table and network; the home's files, the manifest and the staged copies are
real and written by the real writers). The test pause sits in that test module, right after the rename.
The next run's recovery must either put the file back to its backup's bytes or end incomplete (exit 5) with
the steps printed; it never reports a clean rollback while the migration's settings are still in the file.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from tests.hermes_cli.backend_migration_race_host import MODULE
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux", "macos")  # SIGKILL

_REPO = Path(__file__).resolve().parents[2]
_APPLY = ["gateway", "migrate", "--backends", "--apply", "--yes"]


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    home, race_dir = tmp_path / "hermes-home", tmp_path / "race"
    (home / "logs").mkdir(parents=True)
    race_dir.mkdir()
    (home / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (home / ".env").write_text("OPENAI_API_KEY=***\n", encoding="utf-8")
    os.chmod(home / ".env", 0o600)
    monkeypatch.setenv("HERMES_HOME", str(home))
    env = {**os.environ, "HERMES_HOME": str(home), "RACE_DIR": str(race_dir), "PYTHONPATH": str(_REPO)}
    yield home, race_dir, env
    (race_dir / "release").write_text("", encoding="utf-8")  # never leave a paused run behind


def _killed_after_publishing(env, race_dir, name):
    proc = subprocess.Popen([sys.executable, "-m", MODULE, *_APPLY], cwd=_REPO,
                            env={**env, "KILL_AFTER_PUBLISH": name}, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    marker = race_dir / f"published-{name}"
    try:
        wait_until(marker.exists, timeout=60, what=f"the run right after renaming {name} into place")
    finally:
        proc.send_signal(signal.SIGKILL)
        out, _ = proc.communicate(timeout=60)
    assert proc.returncode == -signal.SIGKILL
    return out


@pytest.mark.parametrize("name", ["config.yaml", ".env"])
def test_a_run_killed_right_after_a_rename_never_recovers_as_rolled_back_with_the_change_in_place(scratch, name):
    home, race_dir, env = scratch
    before = {n: (home / n).read_bytes() for n in ("config.yaml", ".env")}

    first = _killed_after_publishing(env, race_dir, name)
    assert (home / name).read_bytes() != before[name], first  # the kill landed after the migration's rename
    assert engine.read_manifest(home)["state"] == "running"

    recovery = subprocess.run([sys.executable, "-m", MODULE, *_APPLY], cwd=_REPO, env=env, capture_output=True,
                              text=True, timeout=120, check=False)
    out = recovery.stdout + recovery.stderr
    manifest = engine.read_manifest(home)
    if (home / name).read_bytes() == before[name]:
        return  # put back to the backup's bytes: a clean recovery is true
    # The file still holds the migration's write: the run is reported incomplete, with the steps.
    assert recovery.returncode == engine.EXIT_ROLLBACK_INCOMPLETE, out
    assert manifest["state"] == "failed", manifest["state"]
    assert "Rollback incomplete" in out and f"{home}/{name}" in out, out
    (step,) = [s for s in engine.status_payload_from(None, manifest)["manual_steps"] if f"{home}/{name}" in s]
    assert "stopped while writing" in step and "edited" not in step, step  # no edit was made: say what happened
    assert step in out
