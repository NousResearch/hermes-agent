"""Unit tests for scripts/setup_identity_loop.py."""

import json
from pathlib import Path

import scripts.setup_identity_loop as setup
from agent.prompt_builder import _scan_context_content


def test_preferences_seed_is_context_file_safe():
    content = (setup._ASSETS / "PREFERENCES.seed.md").read_text(encoding="utf-8")
    assert _scan_context_content(content, "PREFERENCES.md") == content


def test_stage_self_description_clause_creates_pending_soul_proposal(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    soul = home / "SOUL.md"
    soul.write_text("# SOUL\n\nExisting constitution.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert setup._stage_self_description_clause(home) is True

    proposal = home / "identity" / "queue" / "0001"
    meta = json.loads((proposal / "meta.json").read_text(encoding="utf-8"))
    proposed = (proposal / "proposed").read_text(encoding="utf-8")
    assert meta["status"] == "pending_review"
    assert meta["target"] == str(soul)
    assert meta["source"] == "setup-identity-loop"
    assert "self-description clause" in meta["summary"]
    assert "Existing constitution." in proposed
    assert "Do not recite SOUL.md as a self-description" in proposed
    assert "unbacked identity claims" in proposed


def test_stage_self_description_clause_is_idempotent_with_pending_proposal(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "SOUL.md").write_text("# SOUL\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert setup._stage_self_description_clause(home) is True
    assert setup._stage_self_description_clause(home) is False

    proposals = list((home / "identity" / "queue").iterdir())
    assert [p.name for p in proposals] == ["0001"]


def test_register_cron_jobs_includes_identity_review_digest(tmp_path):
    calls = []
    home = tmp_path / ".hermes"
    prompt_dir = home / "cron" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "friday-identity-reflection.md").write_text("reflect\n", encoding="utf-8")

    def fake_create_job(**kwargs):
        calls.append(kwargs)
        return kwargs

    setup._register_cron_jobs(
        home,
        create_job=fake_create_job,
        existing_names=set(),
    )

    digest_jobs = [
        c for c in calls
        if c["name"] == "friday-identity-review-digest"
    ]
    assert len(digest_jobs) == 1
    assert digest_jobs[0]["script"] == "identity_improvement_queue_digest.sh"
    assert digest_jobs[0]["no_agent"] is True
    assert digest_jobs[0]["schedule"] == "12 8 * * *"
    assert digest_jobs[0]["deliver"] == "origin"
    assert digest_jobs[0]["origin"] == {
        "platform": "discord",
        "chat_id": "1512110425389400136",
        "chat_name": "Review Queue",
        "chat_topic": "Review Queue",
        "thread_id": "1512260751514009742",
    }
