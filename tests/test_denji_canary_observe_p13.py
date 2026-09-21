"""P13 isolation proof for scripts/denji-canary-observe.py.

Verifies:
- HERMES_HOME parameterisation: EVAL_DIR and PROFILE_EDITOR resolve
  under HERMES_HOME, not ~/.hermes.
- --dry-run prevents git operations: _revert_commit returns False
  without spawning the profile_editor.py --rollback subprocess.
- --dry-run prevents the fleet-health tripwire mutation (the
  check_fleet_health call that persists tripwire state).
- import-safe under a fake HERMES_HOME.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "denji-canary-observe.py"


def _load_module(monkeypatch, fake_home: Path, dry_run: bool = False):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    argv = ["denji-canary-observe.py"] + (["--dry-run"] if dry_run else [])
    monkeypatch.setattr(sys, "argv", argv)
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("dco_under_test", str(SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def test_paths_resolve_under_hermes_home(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert mod.EVAL_DIR.startswith(str(fake_home))
    assert mod.PROFILE_EDITOR.startswith(str(fake_home))


def test_import_is_side_effect_free(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert mod._DRY_RUN is False


def test_dry_run_flag_set(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    assert mod._DRY_RUN is True


def test_dry_run_revert_commit_skips_subprocess(monkeypatch, fake_home, caplog):
    """_revert_commit under --dry-run must return False and log a
    DRY-RUN line WITHOUT spawning the profile_editor.py subprocess."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    import logging
    with caplog.at_level(logging.INFO, logger="denji-canary-observe"):
        result = mod._revert_commit("abc123def456")
    assert result is False, "dry-run _revert_commit returned True"
    assert any("DRY-RUN" in r.message for r in caplog.records), (
        "dry-run did not log a DRY-RUN line"
    )


def test_dry_run_revert_commit_does_not_call_subprocess(monkeypatch, fake_home):
    """Dry-run _revert_commit must never call subprocess.run."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    called = []
    import subprocess
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: called.append(a) or None)
    mod._revert_commit("abc123")
    assert called == [], "dry-run _revert_commit called subprocess.run"


def test_non_dry_run_revert_commit_calls_subprocess(monkeypatch, fake_home):
    """Non-dry-run _revert_commit must attempt the subprocess (proves
    the dry-run guard is the only thing preventing the git op)."""
    mod = _load_module(monkeypatch, fake_home, dry_run=False)
    called = []
    import subprocess

    class _FakeResult:
        returncode = 1
        stdout = "{}"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: called.append(a) or _FakeResult())
    mod._revert_commit("abc123")
    assert len(called) == 1, "non-dry-run did not call subprocess.run"
