"""Tests for background skill evolution pending changes."""

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest


def test_normalize_evolution_mode_accepts_known_modes_and_defaults_unknown():
    from agent.skill_evolution import normalize_evolution_mode

    assert normalize_evolution_mode("auto") == "auto"
    assert normalize_evolution_mode("confirm") == "confirm"
    assert normalize_evolution_mode("readonly") == "readonly"
    assert normalize_evolution_mode(" CONFIRM ") == "confirm"
    assert normalize_evolution_mode("") == "auto"
    assert normalize_evolution_mode(None) == "auto"
    assert normalize_evolution_mode("dangerous") == "auto"


def test_get_evolution_mode_reads_nested_config_and_accepts_test_config(monkeypatch):
    from agent import skill_evolution

    assert skill_evolution.get_evolution_mode({"skills": {"evolution_mode": "confirm"}}) == "confirm"
    assert skill_evolution.get_evolution_mode({"skills": {"evolution_mode": "nope"}}) == "auto"
    assert skill_evolution.get_evolution_mode({}) == "auto"

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"skills": {"evolution_mode": "readonly"}},
    )
    assert skill_evolution.get_evolution_mode() == "readonly"


def test_queue_and_list_pending_changes_use_profile_home(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    change = skill_evolution.queue_pending_change(
        "write_file",
        "daily-review",
        {"path": "SKILL.md", "content": "updated"},
    )

    assert change["success"] is True
    assert len(change["id"]) == 12
    assert change["action"] == "write_file"
    assert change["name"] == "daily-review"
    assert change["payload"] == {"path": "SKILL.md", "content": "updated"}
    assert change["origin"] == "background_review"
    assert change["status"] == "pending"
    assert "created_at" in change

    queue_path = hermes_home / "skills" / ".evolution_pending.json"
    assert skill_evolution.pending_queue_path() == queue_path
    assert queue_path.exists()
    persisted = {key: value for key, value in change.items() if key != "success"}
    assert json.loads(queue_path.read_text()) == {"changes": [persisted]}
    assert skill_evolution.list_pending_changes() == [persisted]


def test_queue_pending_change_writes_pending_manifest_snapshot_and_diff(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    change = skill_evolution.queue_pending_change(
        "edit",
        "daily-review",
        {
            "content": "new skill body",
            "snapshot": "old skill body",
            "diff": "--- old\n+++ new\n@@\n-old\n+new\n",
        },
    )

    pending_dir = hermes_home / "skills" / ".pending" / change["id"]
    manifest_path = pending_dir / "manifest.json"
    snapshot_path = pending_dir / "snapshot" / "SKILL.md"
    diff_path = pending_dir / "diff.md"

    assert manifest_path.exists()
    assert snapshot_path.read_text() == "old skill body"
    assert diff_path.read_text() == "--- old\n+++ new\n@@\n-old\n+new\n"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["id"] == change["id"]
    assert manifest["action"] == "edit"
    assert manifest["name"] == "daily-review"
    assert manifest["snapshot_path"] == "snapshot/SKILL.md"
    assert manifest["diff_path"] == "diff.md"
    assert "snapshot" not in manifest["payload"]
    assert "diff" not in manifest["payload"]
    assert skill_evolution.list_pending_changes()[0]["diff_path"] == str(diff_path)


def test_pending_snapshot_is_not_discovered_as_skill_or_prompt_entry(tmp_path, monkeypatch):
    from agent import prompt_builder, skill_evolution
    from tools import skills_tool

    hermes_home = tmp_path / "profile"
    skills_dir = hermes_home / "skills"
    live_skill = skills_dir / "daily-review"
    live_skill.mkdir(parents=True)
    (live_skill / "SKILL.md").write_text(
        """\
---
name: daily-review
description: Live approved skill.
---

# Daily Review

Use the approved instructions.
"""
    )
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(prompt_builder, "get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills_dir])
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)

    queued = skill_evolution.queue_pending_change(
        "edit",
        "daily-review",
        {
            "content": "new",
            "snapshot": """\
---
name: pending-review
description: Unapproved pending snapshot.
---

# Pending Snapshot

This must not enter skill discovery.
""",
            "diff": "--- old\n+++ new\n@@\n-old\n+new\n",
        },
    )

    listed = json.loads(skills_tool.skills_list())
    assert listed["success"] is True
    assert [skill["name"] for skill in listed["skills"]] == ["daily-review"]

    prompt = prompt_builder.build_skills_system_prompt()
    assert "daily-review" in prompt
    assert "Live approved skill." in prompt
    assert "pending-review" not in prompt
    assert "Unapproved pending snapshot" not in prompt
    assert (hermes_home / "skills" / ".pending" / queued["id"] / "snapshot" / "SKILL.md").exists()


def test_list_pending_changes_rejects_manifest_paths_outside_change_dir(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    pending_dir = hermes_home / "skills" / ".pending" / "change-1"
    pending_dir.mkdir(parents=True)
    (pending_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "change-1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "action": "edit",
                "name": "daily-review",
                "payload": {"content": "new"},
                "origin": "background_review",
                "status": "pending",
                "manifest_path": "manifest.json",
                "diff_path": "../../secret.txt",
            }
        )
    )

    assert skill_evolution.list_pending_changes() == []


def test_list_pending_changes_rejects_queue_artifact_paths_outside_change_dir(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    queue_path = skill_evolution.pending_queue_path()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(
            {
                "changes": [
                    {
                        "id": "change-1",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "action": "edit",
                        "name": "daily-review",
                        "payload": {"content": "new"},
                        "origin": "background_review",
                        "status": "pending",
                        "diff_path": str(tmp_path / "secret.txt"),
                    }
                ]
            }
        )
    )

    assert skill_evolution.list_pending_changes() == []


def test_list_pending_changes_rejects_unsafe_queue_ids(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    queue_path = skill_evolution.pending_queue_path()
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    def write_tampered_queue():
        queue_path.write_text(
            json.dumps(
                {
                    "changes": [
                        {
                            "id": "../../outside",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "action": "edit",
                            "name": "daily-review",
                            "payload": {
                                "content": "new",
                                "snapshot": "old",
                                "diff": "--- old\n+++ new\n",
                            },
                            "origin": "background_review",
                            "status": "pending",
                        }
                    ]
                }
            )
        )

    write_tampered_queue()
    assert skill_evolution.list_pending_changes() == []
    assert not (hermes_home / "outside").exists()

    write_tampered_queue()
    result = skill_evolution.approve_pending_change(
        "../../outside",
        lambda change: {"success": True},
    )
    assert result["success"] is False
    assert not (hermes_home / "outside").exists()

    write_tampered_queue()
    result = skill_evolution.reject_pending_change("../../outside")
    assert result["success"] is False
    assert not (hermes_home / "outside").exists()

    write_tampered_queue()
    assert skill_evolution.cleanup_expired_pending_changes(ttl_days=1) == []
    assert not (hermes_home / "outside").exists()


def test_list_pending_changes_rejects_manifest_id_mismatch(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    pending_dir = hermes_home / "skills" / ".pending" / "change-1"
    pending_dir.mkdir(parents=True)
    (pending_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "other-change",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "action": "edit",
                "name": "daily-review",
                "payload": {"content": "new"},
                "origin": "background_review",
                "status": "pending",
                "manifest_path": "manifest.json",
                "diff_path": "diff.md",
            }
        )
    )

    assert skill_evolution.list_pending_changes() == []


def test_queue_pending_change_preserves_concurrent_appends(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    original_write_queue = skill_evolution._write_queue

    def slow_write_queue(queue):
        time.sleep(0.02)
        original_write_queue(queue)

    monkeypatch.setattr(skill_evolution, "_write_queue", slow_write_queue)

    worker_count = 12
    start = threading.Barrier(worker_count)
    results = []

    def enqueue(index):
        start.wait()
        results.append(
            skill_evolution.queue_pending_change(
                "patch",
                f"skill-{index}",
                {"index": index},
            )
        )

    threads = [threading.Thread(target=enqueue, args=(index,)) for index in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    pending = skill_evolution.list_pending_changes()
    assert len(results) == worker_count
    assert len(pending) == worker_count
    assert {change["name"] for change in pending} == {
        f"skill-{index}" for index in range(worker_count)
    }


def test_list_pending_changes_treats_missing_or_bad_json_as_empty(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    assert skill_evolution.list_pending_changes() == []

    queue_path = skill_evolution.pending_queue_path()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("{not json")
    assert skill_evolution.list_pending_changes() == []


def test_approve_pending_change_applies_and_removes_successful_change(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)
    queued = skill_evolution.queue_pending_change("write_file", "daily-review", {"content": "ok"})
    applied = []

    def apply_func(change):
        applied.append(change)
        return {"success": True, "message": "applied"}

    result = skill_evolution.approve_pending_change(queued["id"], apply_func)

    assert result["success"] is True
    assert result["applied_change_id"] == queued["id"]
    assert result["apply_result"] == {"success": True, "message": "applied"}
    persisted = {key: value for key, value in queued.items() if key != "success"}
    assert applied == [persisted]
    assert skill_evolution.list_pending_changes() == []


def test_approve_pending_change_keeps_pending_when_apply_fails(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)
    queued = skill_evolution.queue_pending_change("write_file", "daily-review", {"content": "bad"})

    result = skill_evolution.approve_pending_change(
        queued["id"],
        lambda change: {"success": False, "error": "denied"},
    )

    assert result["success"] is False
    assert result["change_id"] == queued["id"]
    assert result["apply_result"] == {"success": False, "error": "denied"}
    persisted = {key: value for key, value in queued.items() if key != "success"}
    assert skill_evolution.list_pending_changes() == [persisted]


def test_approve_pending_change_returns_false_for_missing_id(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)
    skill_evolution.queue_pending_change("write_file", "daily-review", {"content": "ok"})

    result = skill_evolution.approve_pending_change(
        "missing",
        lambda change: {"success": True},
    )

    assert result["success"] is False
    assert "not found" in result["error"].lower()
    assert len(skill_evolution.list_pending_changes()) == 1


def test_approve_pending_change_marks_applying_before_apply_to_avoid_replay_after_write_failure(
    tmp_path,
    monkeypatch,
):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)
    queued = skill_evolution.queue_pending_change("patch", "daily-review", {"content": "ok"})
    applied = []

    original_write_queue = skill_evolution._write_queue
    write_calls = 0

    def fail_after_apply(queue):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise RuntimeError("queue write failed")
        original_write_queue(queue)

    monkeypatch.setattr(skill_evolution, "_write_queue", fail_after_apply)

    with pytest.raises(RuntimeError, match="queue write failed"):
        skill_evolution.approve_pending_change(
            queued["id"],
            lambda change: applied.append(change["id"]) or {"success": True},
        )

    pending = skill_evolution.list_pending_changes()
    assert applied == [queued["id"]]
    assert pending[0]["status"] == "applying"

    result = skill_evolution.approve_pending_change(
        queued["id"],
        lambda change: applied.append("again") or {"success": True},
    )

    assert result["success"] is False
    assert "already" in result["error"].lower()
    assert applied == [queued["id"]]


def test_reject_pending_change_removes_pending_change(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)
    queued = skill_evolution.queue_pending_change("write_file", "daily-review", {"content": "no"})

    result = skill_evolution.reject_pending_change(queued["id"])

    assert result == {"success": True, "rejected_change_id": queued["id"]}
    assert skill_evolution.list_pending_changes() == []
    assert not (hermes_home / "skills" / ".pending" / queued["id"]).exists()


def test_reject_pending_change_returns_false_for_missing_id(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    result = skill_evolution.reject_pending_change("missing")

    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_cleanup_expired_pending_changes_removes_old_entries(tmp_path, monkeypatch):
    from agent import skill_evolution

    hermes_home = tmp_path / "profile"
    monkeypatch.setattr(skill_evolution, "get_hermes_home", lambda: hermes_home)

    old = skill_evolution.queue_pending_change("patch", "old-skill", {"content": "old"})
    fresh = skill_evolution.queue_pending_change("patch", "fresh-skill", {"content": "fresh"})

    queue_path = skill_evolution.pending_queue_path()
    data = json.loads(queue_path.read_text())
    expired = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    for change in data["changes"]:
        if change["id"] == old["id"]:
            change["created_at"] = expired
    queue_path.write_text(json.dumps(data))

    old_manifest = hermes_home / "skills" / ".pending" / old["id"] / "manifest.json"
    manifest = json.loads(old_manifest.read_text())
    manifest["created_at"] = expired
    old_manifest.write_text(json.dumps(manifest))

    removed = skill_evolution.cleanup_expired_pending_changes(ttl_days=30)

    assert removed == [old["id"]]
    assert [change["id"] for change in skill_evolution.list_pending_changes()] == [fresh["id"]]
    assert not (hermes_home / "skills" / ".pending" / old["id"]).exists()
    assert (hermes_home / "skills" / ".pending" / fresh["id"]).exists()


def test_conflicting_base_hash_blocks_pending_approval(tmp_path, monkeypatch):
    from hermes_cli.skills_hub import do_review
    from tools import skill_manager_tool
    from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin

    original_skill = """\
---
name: queued-skill
description: Queued skill for approval.
---

# Queued Skill

Original content.
"""
    updated_skill = original_skill.replace("Original content.", "Queued content.")

    hermes_home = tmp_path / "profile"
    skills_dir = hermes_home / "skills"
    monkeypatch.setattr("agent.skill_evolution.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills_dir])
    monkeypatch.setattr(skill_manager_tool, "get_evolution_mode", lambda: "auto")
    skill_manager_tool.skill_manage(
        action="create",
        name="queued-skill",
        content=original_skill,
    )

    monkeypatch.setattr(skill_manager_tool, "get_evolution_mode", lambda: "confirm")
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        raw = skill_manager_tool.skill_manage(
            action="edit",
            name="queued-skill",
            content=updated_skill,
        )
    finally:
        reset_current_write_origin(token)

    queued = json.loads(raw)
    assert queued["success"] is True
    live_skill = skills_dir / "queued-skill" / "SKILL.md"
    live_skill.write_text(original_skill.replace("Original content.", "User changed content."))

    console = __import__("rich.console").console.Console(record=True)
    do_review(approve_id=queued["pending_id"], console=console)

    assert "base file changed" in console.export_text().lower()
    assert live_skill.read_text().endswith("User changed content.\n")
