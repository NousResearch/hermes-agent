"""Runtime-root and isolation proofs for Postiz publishing.

Verifies the env override:
- POSTIZ_DRY_RUN=1 prints the dry-run line and exits 0 before invoking the
  Python publisher.
- The production content_engine / venv python is never reached.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "publish_to_postiz.sh"


def _run(env_extra=None):
    env = dict(os.environ)
    env.pop("POSTIZ_DRY_RUN", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env=env,
        cwd=str(REPO_ROOT), timeout=15,
    )


def _fake_runtime(tmp_path, publisher_body):
    root = tmp_path / "runtime"
    ce = root / "content_engine"
    python = root / ".venv" / "bin" / "python"
    ce.mkdir(parents=True)
    (ce / "db").mkdir()
    (ce / "db" / "content_engine.db").touch()
    python.parent.mkdir(parents=True)
    (ce / "publish_to_postiz.py").write_text(publisher_body)
    python.symlink_to(Path(sys.executable))
    return root


def test_dry_run_invokes_read_only_publisher_mode(tmp_path):
    """Dry-run reaches publisher preflight but explicitly requests no writes."""
    root = _fake_runtime(
        tmp_path,
        "import sys\nprint('publisher-argv:' + ','.join(sys.argv[1:]))\n",
    )
    env = {"POSTIZ_DRY_RUN": "1", "HERMES_AGENT_ROOT": str(root), "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    assert "publisher-argv:--dry-run" in r.stdout


def test_wrapper_uses_one_runtime_root_for_code_python_and_database(tmp_path):
    root = _fake_runtime(
        tmp_path,
        "import os,sys\nprint(os.environ['CONTENT_ENGINE_DB_PATH'])\nprint(sys.executable)\n",
    )
    env = {"POSTIZ_DRY_RUN": "1", "HERMES_AGENT_ROOT": str(root), "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    assert str(root / "content_engine" / "db" / "content_engine.db") in r.stdout
    assert str(root / ".venv" / "bin" / "python") in r.stdout


def test_wrapper_fails_clearly_when_runtime_database_is_missing(tmp_path):
    root = _fake_runtime(tmp_path, "raise SystemExit('publisher should not run')\n")
    (root / "content_engine" / "db" / "content_engine.db").unlink()
    r = _run({"HERMES_AGENT_ROOT": str(root), "HOME": str(tmp_path)})
    assert r.returncode != 0
    assert "content engine database missing" in r.stderr.lower()
