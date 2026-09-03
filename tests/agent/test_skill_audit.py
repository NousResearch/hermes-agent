"""Tests for agent/skill_audit.py — centralized skill mutation audit log."""

import json
import os
from pathlib import Path

import pytest

from agent.skill_audit import (
    append_skill_audit_record,
    is_skill_dir_path,
)


@pytest.fixture
def fake_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # hermes_constants.get_hermes_home reads from env when present.
    return home


def test_is_skill_dir_path_default_profile(fake_hermes_home):
    skill_path = fake_hermes_home / "skills" / "github" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("test")
    assert is_skill_dir_path(str(skill_path)) is True


def test_is_skill_dir_path_profile_local(fake_hermes_home):
    prof = fake_hermes_home / "profiles" / "work" / "skills" / "github" / "SKILL.md"
    prof.parent.mkdir(parents=True)
    prof.write_text("test")
    assert is_skill_dir_path(str(prof)) is True


def test_is_skill_dir_path_outside_skills(fake_hermes_home, tmp_path):
    outside = tmp_path / "random.md"
    outside.write_text("test")
    assert is_skill_dir_path(str(outside)) is False


def test_append_skill_audit_record_writes_ndjson(fake_hermes_home):
    target = fake_hermes_home / "skills" / "github" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("before")

    append_skill_audit_record(
        tool="write_file",
        path=str(target),
        action="modify",
        session_id="sess_123",
        tool_call_id="call_abc",
    )

    audit_log = fake_hermes_home / "skills" / ".audit.log"
    assert audit_log.exists()
    lines = audit_log.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "write_file"
    assert record["path"] == str(target)
    assert record["action"] == "modify"
    assert record["session_id"] == "sess_123"
    assert record["tool_call_id"] == "call_abc"
    assert record["origin"] == "foreground"
    assert "timestamp" in record
    assert "record_id" in record


def test_append_skill_audit_record_skips_non_skill_paths(fake_hermes_home, tmp_path):
    outside = tmp_path / "random.md"
    outside.write_text("test")
    append_skill_audit_record(tool="write_file", path=str(outside), action="modify")
    audit_log = fake_hermes_home / "skills" / ".audit.log"
    assert not audit_log.exists()


def test_append_skill_audit_record_is_best_effort_on_bad_path(fake_hermes_home):
    # Should not raise even for a nonsense path.
    append_skill_audit_record(tool="write_file", path="\x00invalid", action="modify")
    audit_log = fake_hermes_home / "skills" / ".audit.log"
    assert not audit_log.exists()


def test_append_skill_audit_record_appends_multiple(fake_hermes_home):
    target = fake_hermes_home / "skills" / "github" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("v1")

    append_skill_audit_record(tool="write_file", path=str(target), action="modify")
    append_skill_audit_record(tool="patch", path=str(target), action="modify")

    audit_log = fake_hermes_home / "skills" / ".audit.log"
    lines = audit_log.read_text().strip().splitlines()
    assert len(lines) == 2
    tools = [json.loads(line)["tool"] for line in lines]
    assert tools == ["write_file", "patch"]
