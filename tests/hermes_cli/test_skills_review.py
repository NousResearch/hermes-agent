"""Tests for `hermes skills review` and `/skills review`."""

import sys
import types
from types import SimpleNamespace

import pytest
from rich.console import Console


def _install_fake_skill_evolution(monkeypatch, **funcs):
    module = types.ModuleType("agent.skill_evolution")
    for name, func in funcs.items():
        setattr(module, name, func)
    monkeypatch.setitem(sys.modules, "agent.skill_evolution", module)
    return module


def _capture_console():
    return Console(record=True, force_terminal=False, width=120)


def test_cli_skills_review_parser_routes_flags(monkeypatch):
    from hermes_cli.main import main

    captured = []

    def fake_skills_command(args):
        captured.append(args)

    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", fake_skills_command)

    monkeypatch.setattr(sys, "argv", ["hermes", "skills", "review"])
    main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "skills", "review", "--approve", "change-1"],
    )
    main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "skills", "review", "--reject", "change-2"],
    )
    main()

    assert [args.skills_action for args in captured] == ["review", "review", "review"]
    assert captured[0].approve_id == ""
    assert captured[0].reject_id == ""
    assert captured[1].approve_id == "change-1"
    assert captured[1].reject_id == ""
    assert captured[2].approve_id == ""
    assert captured[2].reject_id == "change-2"


def test_skills_command_review_calls_do_review(monkeypatch):
    from hermes_cli.skills_hub import skills_command

    captured = {}

    def fake_do_review(approve_id="", reject_id="", console=None):
        captured["approve_id"] = approve_id
        captured["reject_id"] = reject_id

    monkeypatch.setattr("hermes_cli.skills_hub.do_review", fake_do_review)

    skills_command(
        SimpleNamespace(
            skills_action="review",
            approve_id="change-1",
            reject_id="",
        )
    )

    assert captured == {"approve_id": "change-1", "reject_id": ""}


def test_slash_skills_review_routes_flags(monkeypatch):
    from hermes_cli.skills_hub import handle_skills_slash

    calls = []

    def fake_do_review(approve_id="", reject_id="", console=None):
        calls.append((approve_id, reject_id, console))

    console = _capture_console()
    monkeypatch.setattr("hermes_cli.skills_hub.do_review", fake_do_review)

    handle_skills_slash("/skills review", console=console)
    handle_skills_slash("/skills review --approve change-1", console=console)
    handle_skills_slash("/skills review --reject change-2", console=console)

    assert calls == [
        ("", "", console),
        ("change-1", "", console),
        ("", "change-2", console),
    ]


def test_do_review_prints_empty_message(monkeypatch):
    from hermes_cli.skills_hub import do_review

    _install_fake_skill_evolution(
        monkeypatch,
        list_pending_changes=lambda: [],
    )

    console = _capture_console()
    do_review(console=console)

    assert "No pending skill evolution changes." in console.export_text()


def test_do_review_prints_pending_changes(monkeypatch):
    from hermes_cli.skills_hub import do_review

    _install_fake_skill_evolution(
        monkeypatch,
        list_pending_changes=lambda: [
            {
                "id": "change-1",
                "action": "patch",
                "name": "writer",
                "created_at": "2026-05-08T10:00:00Z",
            }
        ],
    )

    console = _capture_console()
    do_review(console=console)
    output = console.export_text()

    assert "change-1" in output
    assert "patch" in output
    assert "writer" in output


def test_do_review_rejects_pending_change(monkeypatch):
    from hermes_cli.skills_hub import do_review

    calls = []

    def fake_reject(change_id):
        calls.append(change_id)
        return {"success": True, "id": change_id}

    _install_fake_skill_evolution(
        monkeypatch,
        reject_pending_change=fake_reject,
    )

    console = _capture_console()
    do_review(reject_id="change-1", console=console)

    assert calls == ["change-1"]
    assert "Rejected pending skill evolution change change-1." in console.export_text()


@pytest.mark.parametrize(
    ("action", "function_name", "payload", "expected_args", "expected_kwargs"),
    [
        (
            "create",
            "_create_skill",
            {"content": "skill content", "category": "creative"},
            ("writer", "skill content", "creative"),
            {},
        ),
        (
            "edit",
            "_edit_skill",
            {"content": "updated content"},
            ("writer", "updated content"),
            {},
        ),
        (
            "patch",
            "_patch_skill",
            {
                "old_string": "old",
                "new_string": "new",
                "file_path": "references/notes.md",
                "replace_all": True,
            },
            ("writer", "old", "new", "references/notes.md", True),
            {},
        ),
        (
            "delete",
            "_delete_skill",
            {"absorbed_into": "umbrella"},
            ("writer",),
            {"absorbed_into": "umbrella"},
        ),
        (
            "write_file",
            "_write_file",
            {"file_path": "references/notes.md", "file_content": "notes"},
            ("writer", "references/notes.md", "notes"),
            {},
        ),
        (
            "remove_file",
            "_remove_file",
            {"file_path": "references/notes.md"},
            ("writer", "references/notes.md"),
            {},
        ),
    ],
)
def test_do_review_approve_applies_with_low_level_skill_manager_functions(
    monkeypatch,
    action,
    function_name,
    payload,
    expected_args,
    expected_kwargs,
):
    from hermes_cli.skills_hub import do_review
    from tools import skill_manager_tool

    calls = []

    def fake_low_level(*args, **kwargs):
        calls.append((args, kwargs))
        return {"success": True, "action": action}

    def fake_approve(change_id, apply_func):
        change = {
            "id": change_id,
            "action": action,
            "name": "writer",
            "payload": payload,
        }
        return apply_func(change)

    monkeypatch.setattr(skill_manager_tool, function_name, fake_low_level)
    _install_fake_skill_evolution(
        monkeypatch,
        approve_pending_change=fake_approve,
    )

    console = _capture_console()
    do_review(approve_id="change-1", console=console)

    assert calls == [(expected_args, expected_kwargs)]
    assert "Approved pending skill evolution change change-1." in console.export_text()


def test_do_review_approve_records_skill_manage_success_side_effects(monkeypatch):
    from hermes_cli.skills_hub import do_review
    from tools import skill_manager_tool

    side_effects = []

    def fake_approve(change_id, apply_func):
        change = {
            "id": change_id,
            "action": "create",
            "name": "writer",
            "payload": {"content": "skill content", "category": ""},
        }
        return apply_func(change)

    monkeypatch.setattr(
        skill_manager_tool,
        "_create_skill",
        lambda *args, **kwargs: {"success": True, "message": "created"},
    )
    monkeypatch.setattr(
        skill_manager_tool,
        "_record_skill_manage_success",
        lambda action, name, **kwargs: side_effects.append((action, name, kwargs)),
    )
    _install_fake_skill_evolution(
        monkeypatch,
        approve_pending_change=fake_approve,
    )

    console = _capture_console()
    do_review(approve_id="change-1", console=console)

    assert side_effects == [
        ("create", "writer", {"background_review": True}),
    ]
    assert "Approved pending skill evolution change change-1." in console.export_text()


def test_do_review_approve_skips_success_side_effects_when_apply_fails(monkeypatch):
    from hermes_cli.skills_hub import do_review
    from tools import skill_manager_tool

    side_effects = []

    def fake_approve(change_id, apply_func):
        change = {
            "id": change_id,
            "action": "create",
            "name": "writer",
            "payload": {"content": "skill content", "category": ""},
        }
        apply_result = apply_func(change)
        return {"success": False, "error": "apply failed", "apply_result": apply_result}

    monkeypatch.setattr(
        skill_manager_tool,
        "_create_skill",
        lambda *args, **kwargs: {"success": False, "error": "blocked"},
    )
    monkeypatch.setattr(
        skill_manager_tool,
        "_record_skill_manage_success",
        lambda action, name, **kwargs: side_effects.append((action, name, kwargs)),
    )
    _install_fake_skill_evolution(
        monkeypatch,
        approve_pending_change=fake_approve,
    )

    console = _capture_console()
    do_review(approve_id="change-1", console=console)

    assert side_effects == []
    assert "Failed to approve pending skill evolution change change-1" in console.export_text()


def test_confirm_mode_queue_can_be_approved_into_skill_directory(tmp_path, monkeypatch):
    from hermes_cli.skills_hub import do_review
    from tools import skill_manager_tool
    from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin

    valid_skill = """\
---
name: queued-skill
description: Queued skill for approval.
---

# Queued Skill

Approved content.
"""

    hermes_home = tmp_path / "profile"
    skills_dir = hermes_home / "skills"
    monkeypatch.setattr("agent.skill_evolution.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills_dir])
    monkeypatch.setattr(skill_manager_tool, "get_evolution_mode", lambda: "confirm")

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        raw = skill_manager_tool.skill_manage(
            action="create",
            name="queued-skill",
            content=valid_skill,
        )
    finally:
        reset_current_write_origin(token)

    queued = __import__("json").loads(raw)
    assert queued["success"] is True
    assert queued["queued"] is True
    assert not (skills_dir / "queued-skill" / "SKILL.md").exists()

    console = _capture_console()
    do_review(approve_id=queued["pending_id"], console=console)

    assert "Approved pending skill evolution change" in console.export_text()
    assert (skills_dir / "queued-skill" / "SKILL.md").read_text() == valid_skill
