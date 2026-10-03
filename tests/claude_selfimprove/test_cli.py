from __future__ import annotations

import json

from claude_selfimprove import cli, paths, store, writers


def test_install_dry_run_reports_would_create_and_writes_nothing(sandbox, capsys):
    rc = cli.main(["install", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would deploy" in out
    assert "would create" in out
    assert not (paths.hermes_home() / "scripts" / "claude_selfimprove").exists()


def test_install_skips_already_registered_jobs_and_still_deploys(sandbox, capsys):
    cron_dir = paths.hermes_home() / "cron"
    cron_dir.mkdir(parents=True)
    (cron_dir / "jobs.json").write_text(
        json.dumps({"jobs": [
            {"id": "a", "name": "claude-selfimprove-nightly-scan"},
            {"id": "b", "name": "claude-selfimprove-weekly-consolidate"},
        ]}),
        encoding="utf-8",
    )

    rc = cli.main(["install"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "already registered, left untouched" in out
    assert (paths.hermes_home() / "scripts" / "claude_selfimprove" / "pipeline.py").exists()


def test_install_reports_error_and_nonzero_exit_when_hermes_binary_missing(sandbox, capsys, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    rc = cli.main(["install"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "NOT created" in captured.out
    assert "errors:" in captured.out


def test_status_on_empty_profile(sandbox, capsys):
    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "claude-selfimprove status" in out
    assert "(none yet)" in out


def test_status_reports_candidate_counts(sandbox, capsys):
    with store.CandidateStore() as db:
        db.upsert_occurrence(
            canonical_key="k", category="explicit_instruction", scope="global",
            target_kind="rule", target_path=None, title="t", body="b",
            confidence=0.9, session_id="s1", source_hash_value="h1",
        )
    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pending: 1" in out


def test_scan_dry_run_reports_and_does_not_persist(sandbox, capsys):
    sandbox.write_transcript(
        "claude", "proj", "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "hello there"}}],
    )
    rc = cli.main(["scan", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "0 flagged" in out or "flagged" in out


def test_consolidate_reports_nothing_eligible(sandbox, capsys):
    rc = cli.main(["consolidate"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0 pending candidates evaluated" in out


def test_rollback_list_empty(sandbox, capsys):
    rc = cli.main(["rollback", "--list"])
    assert rc == 0
    assert "no backups on record" in capsys.readouterr().out


def test_rollback_list_shows_backups(sandbox, capsys):
    path = paths.rules_dir() / "a.md"
    writers.write_rule_file(path, title="A", body="v1")
    rc = cli.main(["rollback", "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(path) in out


def test_rollback_by_target_restores_prior_content(sandbox, capsys):
    path = paths.rules_dir() / "a.md"
    writers.write_rule_file(path, title="A", body="v1")
    writers.write_rule_file(path, title="A", body="v2")
    assert "v2" in path.read_text()

    rc = cli.main(["rollback", "--target", str(path)])
    assert rc == 0
    assert "rolled back" in capsys.readouterr().out
    assert "v1" in path.read_text()


def test_rollback_missing_target_fails_cleanly(sandbox, capsys):
    rc = cli.main(["rollback", "--target", "/no/such/file.md"])
    assert rc == 1
    assert "no backup found" in capsys.readouterr().err


def test_rollback_requires_a_mode(sandbox, capsys):
    rc = cli.main(["rollback"])
    assert rc == 2
    assert "requires" in capsys.readouterr().err


def test_rollback_notifies_queue(sandbox, capsys):
    path = paths.rules_dir() / "a.md"
    writers.write_rule_file(path, title="A", body="v1")
    writers.write_rule_file(path, title="A", body="v2")
    cli.main(["rollback", "--target", str(path)])

    lines = paths.self_improvement_queue_path().read_text().splitlines()
    events = [json.loads(l) for l in lines]
    assert any(e["action"] == "rollback" for e in events)


def test_scan_skipped_when_lock_held(sandbox, capsys):
    paths.ensure_state_dirs()
    paths.lock_dir_path().mkdir(parents=True)
    rc = cli.main(["scan"])
    assert rc == 0
    assert "skipped" in capsys.readouterr().out
