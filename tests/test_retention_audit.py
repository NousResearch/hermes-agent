import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cleanup.retention_audit import (
    RetentionPolicy,
    apply_manifest,
    autonomous_run,
    inventory_roots,
    plan_inventory,
    render_external_summary,
)


def _age(path: Path, days: int) -> None:
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    os.utime(path, (ts, ts))


def _candidate(manifest, path: Path):
    matches = [c for c in manifest["candidates"] if c["path"] == str(path)]
    assert matches, f"missing candidate for {path}; got {[c['path'] for c in manifest['candidates']]}"
    return matches[0]


def test_hard_no_touch_secret_is_report_only_and_summary_omits_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = home / ".env"
    secret.parent.mkdir(parents=True)
    secret.write_text("TOKEN=secret\n")

    inv = inventory_roots([home])
    manifest = plan_inventory(inv, RetentionPolicy(hermes_home=home))

    item = _candidate(manifest, secret)
    assert item["class"] == "A"
    assert item["proposed_action"] == "report"
    assert item["autonomous"] is False
    assert item["sha256"] is None

    summary = render_external_summary(manifest)
    assert str(secret) not in summary
    assert ".env" not in summary


def test_old_logs_compress_autonomously_but_severe_logs_are_preserved(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    logs = home / "logs"
    logs.mkdir(parents=True)
    normal = logs / "gateway.log.1"
    normal.write_text("routine\n")
    _age(normal, 45)
    severe = logs / "incident.log"
    severe.write_text("CRITICAL data loss warning\n")
    _age(severe, 45)

    manifest = plan_inventory(inventory_roots([logs]), RetentionPolicy(hermes_home=home))

    normal_item = _candidate(manifest, normal)
    assert normal_item["class"] == "B"
    assert normal_item["proposed_action"] == "compress"
    assert normal_item["autonomous"] is True

    severe_item = _candidate(manifest, severe)
    assert severe_item["class"] == "B"
    assert severe_item["proposed_action"] == "report"
    assert severe_item["autonomous"] is False
    assert "severe" in severe_item["reason"]


def test_allowlisted_old_gitignored_cache_delete_is_autonomous(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    cache = home / "cache" / "old.tmp"
    cache.parent.mkdir(parents=True)
    cache.write_text("junk")
    _age(cache, 20)

    policy = RetentionPolicy(hermes_home=home, cache_allowlist=(home / "cache",))
    manifest = plan_inventory(inventory_roots([home / "cache"]), policy)

    item = _candidate(manifest, cache)
    assert item["class"] == "F"
    assert item["proposed_action"] == "delete"
    assert item["autonomous"] is True
    assert item["reversible"] is False


def test_symlink_and_hardlink_are_report_only_even_in_cache_allowlist(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    cache_dir = home / "cache"
    cache_dir.mkdir(parents=True)
    target = cache_dir / "target.tmp"
    target.write_text("junk")
    _age(target, 20)
    hard = cache_dir / "hard.tmp"
    os.link(target, hard)
    link = cache_dir / "link.tmp"
    link.symlink_to(target)

    policy = RetentionPolicy(hermes_home=home, cache_allowlist=(cache_dir,))
    manifest = plan_inventory(inventory_roots([cache_dir]), policy)

    hard_item = _candidate(manifest, hard)
    assert hard_item["proposed_action"] == "report"
    assert hard_item["autonomous"] is False
    assert "hardlink" in hard_item["reason"]

    link_item = _candidate(manifest, link)
    assert link_item["proposed_action"] == "report"
    assert link_item["autonomous"] is False
    assert "symlink" in link_item["reason"]


def test_apply_runs_only_autonomous_actions_and_refuses_changed_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    cache = home / "cache" / "old.tmp"
    cache.parent.mkdir(parents=True)
    cache.write_text("junk")
    _age(cache, 20)
    session = home / "sessions" / "old.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text("important")
    _age(session, 120)

    policy = RetentionPolicy(hermes_home=home, cache_allowlist=(home / "cache",))
    manifest = plan_inventory(inventory_roots([home / "cache", home / "sessions"]), policy)
    result = apply_manifest(manifest, policy, autonomous_only=True)

    assert result["deleted"] == 1
    assert not cache.exists()
    assert session.exists()
    assert result["skipped"] >= 1


def test_apply_compresses_old_log_and_keeps_content(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    log = home / "logs" / "gateway.log.1"
    log.parent.mkdir(parents=True)
    log.write_text("line one\n")
    _age(log, 45)

    policy = RetentionPolicy(hermes_home=home)
    manifest = plan_inventory(inventory_roots([home / "logs"]), policy)
    result = apply_manifest(manifest, policy, autonomous_only=True)

    gz = log.with_suffix(log.suffix + ".gz")
    assert result["compressed"] == 1
    assert not log.exists()
    assert gz.exists()
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        assert f.read() == "line one\n"


def test_autonomous_run_writes_manifest_applies_only_safe_actions_and_returns_summary(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    log = home / "logs" / "gateway.log.1"
    log.parent.mkdir(parents=True)
    log.write_text("routine\n")
    _age(log, 45)
    session = home / "sessions" / "old.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text("important")
    _age(session, 120)
    cache = home / "cache" / "old.tmp"
    cache.parent.mkdir(parents=True)
    cache.write_text("junk")
    _age(cache, 20)

    outcome = autonomous_run(RetentionPolicy(hermes_home=home), output_dir=home / "cleanup" / "manifests")

    assert outcome["apply_result"]["compressed"] == 1
    assert outcome["apply_result"]["deleted"] == 1
    assert not log.exists()
    assert not cache.exists()
    assert session.exists()
    assert Path(outcome["manifest_path"]).exists()
    assert str(session) not in outcome["summary"]
