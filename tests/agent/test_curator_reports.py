"""Tests for the curator per-run report writer (run.json + REPORT.md).

Reports live under ``~/.hermes/logs/curator/{YYYYMMDD-HHMMSS}/`` alongside
the standard log dir, not inside the user's ``skills/`` data directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a skills/ dir + reset curator module state."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator
    importlib.reload(curator)
    from tools import skill_usage
    importlib.reload(skill_usage)
    yield {"home": home, "curator": curator, "skill_usage": skill_usage}


def _make_llm_meta(**overrides):
    base = {
        "final": "short summary of the pass",
        "summary": "short summary",
        "model": "test-model",
        "provider": "test-provider",
        "tool_calls": [],
        "error": None,
    }
    base.update(overrides)
    return base




def test_write_run_report_creates_both_files(curator_env):
    """Each run writes both a run.json (machine) and a REPORT.md (human)."""
    curator = curator_env["curator"]
    start = datetime.now(timezone.utc)

    run_dir = curator._write_run_report(
        started_at=start,
        elapsed_seconds=12.345,
        auto_counts={"checked": 5, "marked_stale": 1, "archived": 0, "reactivated": 0},
        auto_summary="1 marked stale",
        before_report=[],
        before_names=set(),
        after_report=[],
        llm_meta=_make_llm_meta(),
    )
    assert run_dir is not None
    assert run_dir.is_dir()
    assert (run_dir / "run.json").exists()
    assert (run_dir / "REPORT.md").exists()

    # The directory name is a timestamp under logs/curator/
    assert run_dir.parent == curator._reports_root()






def test_same_second_reruns_get_unique_dirs(curator_env):
    """If the curator somehow runs twice in the same second, the second
    report still gets its own directory rather than overwriting the first."""
    curator = curator_env["curator"]
    start = datetime(2026, 4, 29, 5, 33, 34, tzinfo=timezone.utc)

    kwargs = dict(
        started_at=start,
        elapsed_seconds=1.0,
        auto_counts={"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=[],
        before_names=set(),
        after_report=[],
        llm_meta=_make_llm_meta(),
    )
    a = curator._write_run_report(**kwargs)
    b = curator._write_run_report(**kwargs)
    assert a != b
    assert a is not None and b is not None
    # Second dir has a numeric disambiguator suffix
    assert b.name.startswith(a.name)






# ---------------------------------------------------------------------------
# Cron job skill reference rewriting (curator ↔ cron integration)
# ---------------------------------------------------------------------------
#
# When the curator consolidates skill X into umbrella Y during a run, any
# cron job that listed X in its ``skills`` field would fail to load X at
# run time — the scheduler logs a warning and skips it, so the scheduled
# job runs without the instructions it was scheduled to follow. These
# tests verify that _write_run_report calls into cron.jobs to repair
# those references and records what it did in both run.json and
# cron_rewrites.json.


@pytest.fixture
def curator_env_with_cron(curator_env, monkeypatch):
    """Extend curator_env with an initialized + repointed cron.jobs module."""
    home = curator_env["home"]
    (home / "cron").mkdir(exist_ok=True)
    (home / "cron" / "output").mkdir(exist_ok=True)

    import importlib
    import cron.jobs as jobs_mod
    importlib.reload(jobs_mod)
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", home / "cron" / "output")

    return {**curator_env, "jobs": jobs_mod}


def test_curator_rewrites_cron_skills_when_skill_consolidated(curator_env_with_cron):
    """A skill consolidated into an umbrella should be rewritten in any
    cron job's skills list; the rewrite should be visible in run.json
    and cron_rewrites.json."""
    curator = curator_env_with_cron["curator"]
    jobs = curator_env_with_cron["jobs"]

    # Create a cron job that depends on a soon-to-be-consolidated skill
    job = jobs.create_job(
        prompt="",
        schedule="every 1h",
        skills=["foo"],
        name="foo-watcher",
    )

    # Simulate a curator pass that consolidated `foo` → `foo-umbrella`
    before = [{"name": "foo", "state": "active", "pinned": False}]
    after = [{"name": "foo-umbrella", "state": "active", "pinned": False}]

    run_dir = curator._write_run_report(
        started_at=datetime.now(timezone.utc),
        elapsed_seconds=3.0,
        auto_counts={"checked": 1, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=before,
        before_names={"foo"},
        after_report=after,
        llm_meta=_make_llm_meta(
            final="Consolidated foo into foo-umbrella.",
            tool_calls=[
                {
                    "name": "skill_manage",
                    "arguments": json.dumps({
                        "action": "write_file",
                        "name": "foo-umbrella",
                        "file_path": "references/foo.md",
                        "file_content": "from foo",
                    }),
                },
            ],
        ),
    )

    # Cron job is rewritten on disk
    loaded = jobs.get_job(job["id"])
    assert loaded["skills"] == ["foo-umbrella"]
    assert loaded["skill"] == "foo-umbrella"

    # Rewrite is recorded in run.json
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["cron_rewrites"]["jobs_updated"] == 1
    assert payload["counts"]["cron_jobs_rewritten"] == 1
    rewrites = payload["cron_rewrites"]["rewrites"]
    assert len(rewrites) == 1
    assert rewrites[0]["mapped"] == {"foo": "foo-umbrella"}

    # Separate cron_rewrites.json is written for convenience
    cron_file = run_dir / "cron_rewrites.json"
    assert cron_file.exists()
    detail = json.loads(cron_file.read_text())
    assert detail["jobs_updated"] == 1

    # Markdown surfaces the change
    md = (run_dir / "REPORT.md").read_text()
    assert "Cron job skill references rewritten" in md
    assert "foo-watcher" in md
    assert "foo-umbrella" in md




