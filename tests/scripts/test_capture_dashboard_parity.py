"""P04 dashboard parity capture — harness tests.

Tests the capture harness itself (scripts/capture_dashboard_parity.py):
  - parity success (CLI _cmd_list vs dashboard GET /board agree)
  - expected workflow statuses (completion yields review, not done)
  - temporary-root cleanup (no leaked temp home)
  - no default repository write (capture() returns JSON, doesn't write)

Hermetic: each test uses the harness's own TemporaryDirectory; the repo is
never mutated.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    import subprocess
    _out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True)
    if _out.returncode == 0 and _out.stdout.strip():
        REPO_ROOT = Path(_out.stdout.strip())
except Exception:
    pass
SCRIPT_PATH = REPO_ROOT / "scripts" / "capture_dashboard_parity.py"
_MOD_NAME = "capture_dashboard_parity"


def _load_capture():
    sys.modules.pop(_MOD_NAME, None)
    spec = importlib.util.spec_from_file_location(_MOD_NAME, str(SCRIPT_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exists():
    assert SCRIPT_PATH.is_file(), "scripts/capture_dashboard_parity.py is absent"


def test_parity_success():
    """CLI _cmd_list and dashboard GET /board render the same board."""
    mod = _load_capture()
    report = mod.capture()
    assert report["verdict"] == "PARITY-PROVEN-BY-CAPTURE", report
    co = report["captured_output"]
    assert co["id_set_match"] is True
    assert co["status_counts_match"] is True


def test_expected_workflow_statuses():
    """Completion of a full-tier task yields review (review gate), not done.
    The harness must report the HONEST status, not mislabel it done."""
    mod = _load_capture()
    report = mod.capture()
    seed = report["seed_statuses"]
    # Honest status is 'review' — the description may explain it is NOT done.
    assert seed["parity-dash-task"].startswith("review"), \
        f"expected honest review status, got: {seed['parity-dash-task']}"
    # Both render sides agree on 'review'.
    dash = report["captured_output"]["dashboard_side_after_cli_recompute"]["status_counts"]
    cli = report["captured_output"]["cli_side"]["status_counts"]
    assert dash.get("review", 0) >= 1
    assert cli.get("review", 0) >= 1


def test_temp_root_cleaned_after_capture():
    """capture() removes its TemporaryDirectory; it does not ADD a leaked
    temp root. Compared against the pre-call set so stale dirs from other
    processes/suites do not cause a false failure."""
    mod = _load_capture()
    tmp = __import__("tempfile").gettempdir()
    before = set(str(p) for p in Path(tmp).glob("p04-parity-*"))
    mod.capture()
    after = set(str(p) for p in Path(tmp).glob("p04-parity-*"))
    leaked = after - before
    assert not leaked, f"capture() leaked temp roots: {leaked}"


def test_no_default_repository_write_and_safe_explicit_output(tmp_path, monkeypatch):
    """capture() cannot mutate the tracked artifact; explicit output must be
    new and outside Hermes data paths."""
    mod = _load_capture()
    tracked = REPO_ROOT / "migration" / "evidence" / "2026-07-16" / "P04-dashboard-parity-report.json"
    before = tracked.read_bytes() if tracked.exists() else None
    report = mod.capture()
    assert report["verdict"] == "PARITY-PROVEN-BY-CAPTURE"
    assert (tracked.read_bytes() if tracked.exists() else None) == before

    out = tmp_path / "parity-out.json"
    monkeypatch.setattr(sys, "argv", ["capture_dashboard_parity.py", "--output", str(out)])
    assert mod.main() == 0
    assert json.loads(out.read_text())["verdict"] == "PARITY-PROVEN-BY-CAPTURE"

    # Existing files and live-Hermes paths are non-clobbering hard failures.
    monkeypatch.setattr(sys, "argv", ["capture_dashboard_parity.py", "--output", str(out)])
    assert mod.main() == 2
    live = tmp_path / "live-hermes"
    live.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(live))
    dangerous = live / "kanban.db"
    monkeypatch.setattr(sys, "argv", ["capture_dashboard_parity.py", "--output", str(dangerous)])
    assert mod.main() == 2
    assert not dangerous.exists()
    workspace = tmp_path / "live-workspaces"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspace))
    nested_danger = workspace / "must-not-exist" / "report.json"
    monkeypatch.setattr(sys, "argv", ["capture_dashboard_parity.py", "--output", str(nested_danger)])
    assert mod.main() == 2
    assert not nested_danger.parent.exists()


def test_capture_helpers_reject_unowned_root(tmp_path, monkeypatch):
    mod = _load_capture()
    with pytest.raises(RuntimeError, match="refusing unowned capture root"):
        with mod.owned_env(tmp_path):
            pass
    with pytest.raises(RuntimeError, match="refusing non-temporary capture root"):
        mod._register_owned_root(REPO_ROOT)
    protected = tmp_path / "protected"
    protected.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(protected))
    with pytest.raises(ValueError, match="Hermes data path"):
        mod._write_new_output(protected / "report.json", "{}")


def test_capture_restores_caller_environment(monkeypatch):
    """No caller env variable may point at the deleted TemporaryDirectory."""
    mod = _load_capture()
    monkeypatch.setenv("HERMES_HOME", "/tmp/caller-home")
    monkeypatch.setenv("HERMES_KANBAN_HOME", "/tmp/caller-kanban")
    report = mod.capture()
    assert report["provenance"]["temp_root_cleaned"] is True
    assert os.environ["HERMES_HOME"] == "/tmp/caller-home"
    assert os.environ["HERMES_KANBAN_HOME"] == "/tmp/caller-kanban"
