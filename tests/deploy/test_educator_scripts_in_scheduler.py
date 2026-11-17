"""The educator wake-gate scripts, run the way the cron scheduler runs them:
copied into ``<HERMES_HOME>/scripts/``, executed by ``_run_job_script`` in a
real subprocess, output parsed by ``_parse_wake_gate``. No network: the
profile has no Notion key and no checkouts, so release-watch reports those
problems instead of calling out."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "deploy" / "educator" / "scripts"


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "educator"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    (home / "workspace").mkdir()
    (home / "config.yaml").write_text(
        "notion:\n  write_guard:\n    prds:\n      data_sources: [2daff849-6e1b-4e4f-960a-a07acfbc2a1b]\n"
        "release_updates:\n  release_assets_data_source_id: c7436811-d26d-4a73-8c60-69437f3fcb16\n")
    for f in SCRIPTS.glob("*.py"):
        shutil.copy(f, home / "scripts" / f.name)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("NOTION_API_KEY", "NOTION_API_TOKEN", "JIRA_API_TOKEN", "GITHUB_PR_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return home


def test_import_preflight_accepts_both_scripts(profile):
    from cron.capability_preflight import script_import_problems
    for name in ("educator_release_watch.py", "educator_explore.py"):
        assert script_import_problems(name) is None, name


def test_explore_runs_under_the_scheduler_and_wakes(profile):
    from cron.scheduler import _parse_wake_gate, _run_job_script
    ok, out = _run_job_script("educator_explore.py", workdir=str(profile / "workspace"))
    assert ok, out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["job"] == "explore"
    assert _parse_wake_gate(out) is True
    assert (profile / "cron" / "educator-state" / "explore.json").is_file()


def test_release_watch_runs_under_the_scheduler_and_reports_setup_gaps(profile):
    from cron.scheduler import _parse_wake_gate, _run_job_script
    ok, out = _run_job_script("educator_release_watch.py", workdir=str(profile / "workspace"))
    assert ok, out
    payload = json.loads(out.strip().splitlines()[-1])
    kinds = sorted(e["type"] for e in payload["events"])
    assert kinds == ["checkout_error", "checkout_error", "source_error"]
    assert _parse_wake_gate(out) is True
    # second run: the same problems are not reported again → silent, $0
    ok, out = _run_job_script("educator_release_watch.py", workdir=str(profile / "workspace"))
    assert ok, out
    assert _parse_wake_gate(out) is False


def test_release_watch_without_data_sources_says_so(profile):
    from cron.scheduler import _run_job_script
    (profile / "config.yaml").write_text("{}\n")
    ok, out = _run_job_script("educator_release_watch.py", workdir=str(profile / "workspace"))
    assert ok, out
    payload = json.loads(out.strip().splitlines()[-1])
    [cfg] = [e for e in payload["events"] if e["type"] == "config_error"]
    assert "nothing is watched" in cfg["error"]
