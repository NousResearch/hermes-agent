"""Regression: the leaky cron test modules must self-isolate OUTSIDE pytest.

2026-07-15 incident: harnesses (real-agent blackbox sessions / kanban workers
in worktrees sharing the live HERMES_HOME) executed the fixture-creating code
in ``test_ticker_stall_60703.py`` / ``test_per_job_reasoning_effort.py``
WITHOUT pytest's autouse hermetic conftest — leaking ``brief``/``claim job``
fixture jobs into the real ``~/.hermes/cron/jobs.json`` (re-arming
cron-config-lint) and, once, wiping it via ``save_jobs([])``.

This test reproduces that exact vector in a subprocess: no PYTEST_* env, the
modules imported directly by file path, ``HERMES_HOME`` pointing at a fake
"live" home. The self-isolation blocks in those modules must route every
write to a tempdir store and leave the fake live home untouched.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

HARNESS = textwrap.dedent(
    """
    import importlib.util, json, os, sys
    from pathlib import Path

    fake_home = Path(os.environ["FAKE_LIVE_HOME"])
    repo = os.environ["REPO_ROOT"]
    sys.path.insert(0, repo)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    t1 = load("t_re", f"{repo}/tests/cron/test_per_job_reasoning_effort.py")
    t1.TestPersistence().test_create_persists_reasoning_effort()

    t2 = load("t_ts", f"{repo}/tests/cron/test_ticker_stall_60703.py")
    t2.TestFutureDatedClaims().test_expired_fire_claim_is_reclaimable()

    leaked = fake_home / "cron" / "jobs.json"
    print("LEAKED" if leaked.exists() else "CLEAN")
    """
)


def test_leaky_modules_self_isolate_without_pytest(tmp_path):
    fake_live_home = tmp_path / "fake-live-home"
    fake_live_home.mkdir()

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PYTEST", "HERMES"))
    }
    env.update(
        {
            "HERMES_HOME": str(fake_live_home),
            "FAKE_LIVE_HOME": str(fake_live_home),
            "REPO_ROOT": str(REPO),
            "PATH": os.environ.get("PATH", ""),
            # Pin module resolution to THIS repo's source tree so the harness
            # can't vacuously exercise an installed/other cron.jobs copy.
            "PYTHONPATH": str(REPO),
        }
    )

    proc = subprocess.run(
        [sys.executable, "-c", HARNESS],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, f"harness crashed:\n{proc.stdout}\n{proc.stderr}"
    assert "CLEAN" in proc.stdout, (
        "fixture jobs leaked into the fake live HERMES_HOME — the leaky test "
        f"modules lost their self-isolation:\n{proc.stdout}\n{proc.stderr}"
    )
    leaked = fake_live_home / "cron" / "jobs.json"
    assert not leaked.exists(), f"leak artifact present: {leaked.read_text()[:500]}"
