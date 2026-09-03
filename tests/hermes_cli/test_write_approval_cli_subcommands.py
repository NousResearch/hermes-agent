from __future__ import annotations

import argparse
import json

from tools import write_approval as wa


def test_memory_parser_exposes_write_approval_review_commands():
    from hermes_cli.subcommands.memory import build_memory_parser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    build_memory_parser(sub, cmd_memory=lambda args: None)

    args = parser.parse_args(["memory", "pending"])
    assert args.memory_command == "pending"

    args = parser.parse_args(["memory", "approve", "abc123"])
    assert args.memory_command == "approve"
    assert args.id == "abc123"

    args = parser.parse_args(["memory", "approval", "on"])
    assert args.memory_command == "approval"
    assert args.mode == "on"


def test_skills_parser_exposes_write_approval_review_commands():
    from hermes_cli.subcommands.skills import build_skills_parser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    build_skills_parser(sub, cmd_skills=lambda args: None)

    args = parser.parse_args(["skills", "pending"])
    assert args.skills_action == "pending"

    args = parser.parse_args(["skills", "reject", "abc123"])
    assert args.skills_action == "reject"
    assert args.id == "abc123"

    args = parser.parse_args(["skills", "approval", "off"])
    assert args.skills_action == "approval"
    assert args.mode == "off"


def test_cmd_memory_pending_uses_shared_pending_store(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.main import cmd_memory

    wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "user", "content": "User prefers concise receipts."},
        summary="add to user profile: User prefers concise receipts.",
        origin="background_review",
    )

    cmd_memory(argparse.Namespace(memory_command="pending"))
    out = capsys.readouterr().out
    assert "Pending memory writes (1):" in out
    assert "[auto]" in out
    assert "User prefers concise receipts" in out


def test_cmd_skills_diff_pending_id_takes_precedence_over_bundled_diff(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.main import cmd_skills

    rec = wa.stage_write(
        wa.SKILLS,
        {
            "action": "patch",
            "name": "demo-skill",
            "old_string": "old",
            "new_string": "new",
        },
        summary="patch 'demo-skill' SKILL.md (+1/-1 lines)",
        origin="foreground",
    )

    cmd_skills(argparse.Namespace(skills_action="diff", name=rec["id"]))
    out = capsys.readouterr().out
    assert f"Pending skill write {rec['id']}" in out
    assert "demo-skill" in out


def test_cmd_memory_approve_applies_pending_record(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.main import cmd_memory

    rec = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "user", "content": "User likes verified fixes."},
        summary="add to user profile: User likes verified fixes.",
        origin="foreground",
    )

    cmd_memory(argparse.Namespace(memory_command="approve", id=rec["id"]))
    out = capsys.readouterr().out
    assert "Approved 1 memory write(s)." in out
    assert not wa.get_pending(wa.MEMORY, rec["id"])
    assert "User likes verified fixes." in (tmp_path / "memories" / "USER.md").read_text(encoding="utf-8")
