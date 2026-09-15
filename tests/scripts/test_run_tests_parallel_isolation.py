"""The per-file pytest subprocess must not see the host's real home.

Regression for the 2026-09-15 production-config contamination: the runner
host is a production Hermes gateway machine, and pytest subprocesses
inherited ``HOME=/home/agent`` with ``HERMES_HOME`` unset, so any test whose
fixture isolation failed resolved production state directly (and rewrote the
live gateway's config.yaml with test fixture providers).

The contract under test: whatever ``HOME``/``HERMES_HOME`` the parent
process carries, the subprocess that ``run_tests_parallel.py`` spawns gets a
unique throwaway home that is NOT the parent's — set in the child
environment before Python starts, so no import-order trick can bypass it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _probe_file(out_dir: Path, marker: str) -> Path:
    """A one-test file that records the env it actually ran with.

    Writes to ``out_dir`` — a parent-controlled directory OUTSIDE the
    subprocess's throwaway home — because the runner deletes the throwaway
    home when the attempt finishes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"probe_{marker}.py"
    f.write_text(
        "import os\n"
        "def test_record_env():\n"
        f"    with open({str(out_dir / marker)!r}, 'w') as fh:\n"
        "        fh.write(os.environ.get('HERMES_HOME', '<unset>') + '\\n'\n"
        "                 + os.environ.get('HOME', '<unset'))\n"
    )
    return f


def _run_runner(probe: Path, parent_home: str, env_hermes: str) -> int:
    """Invoke the parallel runner over the probe file with a hostile parent env."""
    env = os.environ.copy()
    env["HOME"] = parent_home
    if env_hermes:
        env["HERMES_HOME"] = env_hermes
    else:
        env.pop("HERMES_HOME", None)
    return subprocess.run(
        [sys.executable, str(RUNNER), "--files", str(probe)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(REPO_ROOT),
    ).returncode


def test_subprocess_home_is_not_parent_home(tmp_path):
    """The spawned pytest must never inherit the parent's HOME."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    out_dir = tmp_path / "out"
    probe = _probe_file(out_dir, "home")
    assert _run_runner(probe, str(fake_home), "") == 0, "runner failed on probe"
    recorded = (out_dir / "home").read_text()
    assert "hermes-pytest-home-" in recorded, (
        f"subprocess HOME not isolated: {recorded!r}"
    )


def test_subprocess_hermes_home_is_not_production(tmp_path):
    """The subprocess HERMES_HOME is throwaway, never production's."""
    production_like = tmp_path / "prod-like-home"
    production_like.mkdir()
    out_dir = tmp_path / "out"
    probe = _probe_file(out_dir, "hermes")
    assert _run_runner(probe, str(production_like), str(production_like / ".hermes")) == 0
    recorded = (out_dir / "hermes").read_text()
    hermes_line, home_line = recorded.splitlines()
    assert "hermes-pytest-home-" in hermes_line, (
        f"subprocess HERMES_HOME not isolated: {hermes_line!r}"
    )
    assert "hermes-pytest-home-" in home_line, (
        f"subprocess HOME not isolated: {home_line!r}"
    )
