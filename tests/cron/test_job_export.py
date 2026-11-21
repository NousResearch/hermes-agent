"""`hermes cron export` writes a deterministic, state-free, secret-free mirror; `import` reads it back."""
import os
import subprocess
from types import SimpleNamespace

import hermes_yaml as yaml
from cron import jobs
from cron.job_export import export_jobs
from hermes_cli.cron import cron_export, cron_import


def _git(repo, *args):
    env = {k: v for k, v in os.environ.items() if k != "GIT_DIR"}
    cmd = ["git", "-C", str(repo), *args]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True, env=env).stdout


def test_export_is_a_stable_git_mirror_without_run_state_or_secrets(tmp_path, monkeypatch):
    out = tmp_path / "automations"
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    _git(decoy, "init", "-q")
    # An inherited GIT_DIR (a git hook, a dotfiles alias) must not redirect the commit.
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    with jobs.use_cron_store(tmp_path / "home"):
        digest = jobs.create_job(prompt="Summarize my inbox", schedule="0 9 * * *", name="digest")
        leaky = jobs.create_job(
            prompt="Poll https://bot:hunter2secretpw@example.com/feed", schedule="every 2h", name="feed")
        assert cron_export(SimpleNamespace(directory=str(out), git=True, message="first")) == 0
        first_head = _git(out, "rev-parse", "HEAD").strip()
        files = sorted(p.name for p in (out / "jobs").iterdir())
        snapshot = {name: (out / "jobs" / name).read_text(encoding="utf-8-sig") for name in files}

        # A run advances every scheduler-owned field; the export must not see any of it.
        jobs.mark_job_run(digest["id"], success=False, error="boom")
        second = export_jobs(out, git=True)
        assert second.commit is None and second.written == ()
        assert _git(out, "rev-parse", "HEAD").strip() == first_head

        jobs.remove_job(leaky["id"])
        third = export_jobs(out, git=True, message="drop feed")
        assert third.removed == (f"{leaky['id']}.yaml",) and third.commit

    assert files == sorted([f"{digest['id']}.yaml", f"{leaky['id']}.yaml"])
    record = yaml.safe_load(snapshot[f"{digest['id']}.yaml"])
    assert record["name"] == "digest" and record["schedule"]["expr"] == "0 9 * * *"
    assert not {"next_run_at", "last_status", "last_run_at", "enabled", "state", "origin"} & set(record)
    feed = snapshot[f"{leaky['id']}.yaml"]
    assert "hunter2secretpw" not in feed and yaml.safe_load(feed)["redacted_fields"] == ["prompt"]
    assert _git(out, "log", "--format=%s").split("\n")[:2] == ["drop feed", "first"]
    assert _git(out, "ls-files").split() == [f"jobs/{digest['id']}.yaml"]
    assert _git(decoy, "ls-files") == ""


def test_import_adds_unknown_jobs_paused_and_never_writes_a_mask_over_a_real_value(tmp_path):
    out = tmp_path / "automations"
    with jobs.use_cron_store(tmp_path / "src"):
        origin = {"platform": "telegram", "chat_id": "42"}
        job = jobs.create_job(prompt="Poll https://bot:hunter2secretpw@example.com/feed",
                              schedule="every 2h", name="feed", origin=origin, enabled_toolsets=[])
        export_jobs(out)
    with jobs.use_cron_store(tmp_path / "dst"):
        assert cron_import(SimpleNamespace(directory=str(out))) == 0
        landed = jobs.get_job(job["id"])
        assert landed is not None and landed["name"] == "feed" and landed["enabled"] is False and landed["next_run_at"] is None
        assert "hunter2secretpw" not in landed["prompt"] and "***" in landed["prompt"]
        assert landed["enabled_toolsets"] == []  # "no tools" survives the trip
    with jobs.use_cron_store(tmp_path / "src"):
        assert cron_import(SimpleNamespace(directory=str(out))) == 0
        kept = jobs.get_job(job["id"])
        assert kept is not None and "hunter2secretpw" in kept["prompt"]
        assert kept["origin"] == origin and kept["deliver"] == "origin"
