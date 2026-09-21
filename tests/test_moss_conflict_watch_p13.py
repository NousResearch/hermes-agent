"""P13 isolation proof for scripts/moss-conflict-watch.py.

Verifies:
- HERMES_HOME parameterisation: STATE_FILE resolves under HERMES_HOME,
  not ~/.hermes.
- --dry-run prevents GitHub calls: get_open_prs returns [] without
  spawning the gh subprocess.
- --dry-run prevents state writes: save_state is a no-op, the state
  file is never created.
- main() under --dry-run does not touch the network or write state.
- import-safe under a fake HERMES_HOME.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "moss-conflict-watch.py"


def _load_module(monkeypatch, fake_home: Path, dry_run: bool = False):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    argv = ["moss-conflict-watch.py"] + (["--dry-run"] if dry_run else [])
    monkeypatch.setattr(sys, "argv", argv)
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("mcw_under_test", str(SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def test_state_file_resolves_under_hermes_home(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert mod.STATE_FILE.startswith(str(fake_home))


def test_import_is_side_effect_free(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert mod._DRY_RUN is False
    assert not (fake_home / "data").exists()


def test_dry_run_get_open_prs_skips_subprocess(monkeypatch, fake_home):
    """get_open_prs under --dry-run must return [] without calling gh."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    import subprocess
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: called.append(a) or None)
    result = mod.get_open_prs("NousResearch/hermes-agent", "Sahil-SS9")
    assert result == [], "dry-run get_open_prs did not return []"
    assert called == [], "dry-run get_open_prs called subprocess.run"


def test_dry_run_save_state_writes_nothing(monkeypatch, fake_home):
    """save_state under --dry-run must not create the state file."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    mod.save_state({"updated_at": "now", "tracked": []})
    assert not (fake_home / "data" / "moss-conflict-queue.json").exists()


def test_dry_run_main_no_network_no_state(monkeypatch, fake_home, capsys):
    """main() under --dry-run must not call gh or write state."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    import subprocess
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: called.append(a) or None)
    mod.main()
    assert called == [], "dry-run main called subprocess (gh)"
    assert not (fake_home / "data" / "moss-conflict-queue.json").exists()


def test_non_dry_run_save_state_writes(monkeypatch, fake_home):
    """Non-dry-run save_state must write the file (proves the guard is
    the only thing preventing the write)."""
    mod = _load_module(monkeypatch, fake_home, dry_run=False)
    mod.save_state({"updated_at": "now", "tracked": []})
    assert (fake_home / "data" / "moss-conflict-queue.json").exists()
