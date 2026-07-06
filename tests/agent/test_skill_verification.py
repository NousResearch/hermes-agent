import json
import shlex
import sys


def _skill_md(command: str, *, name: str = "verified-skill") -> str:
    return f"""---
name: {name}
description: test skill with runtime verification
metadata:
  hermes:
    verification:
      before_final:
        required: true
        command: "{command}"
        timeout_seconds: 5
        success_exit_codes: [0]
        on_failure: block_final
---

# Verified Skill

Do the workflow, then rely on runtime verification before final.
"""


def test_extract_before_final_spec_from_metadata_hermes():
    from agent.skill_verification import extract_before_final_spec

    spec = extract_before_final_spec(
        {
            "success": True,
            "name": "verified-skill",
            "skill_dir": "/tmp/skill",
            "content": _skill_md("true"),
        }
    )

    assert spec is not None
    assert spec.name == "verified-skill"
    assert spec.command == "true"
    assert spec.timeout_seconds == 5
    assert spec.success_exit_codes == (0,)
    assert spec.on_failure == "block_final"


def test_skill_view_loads_candidate_activation_arms_verifier(tmp_path, monkeypatch):
    from agent.skill_verification import (
        activate_skill_verification,
        clear_session_verifications,
        loaded_verification_specs,
        pending_verification_specs,
        run_before_final_verifications,
    )
    from tools import skills_tool
    from tools.skills_tool import _skill_view_with_bump

    monkeypatch.setattr(
        "agent.skill_verification._command_verifiers_enabled", lambda: True
    )
    skill_dir = tmp_path / "verified-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(_skill_md("true"), encoding="utf-8")

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path)
    clear_session_verifications("session-1")

    result = json.loads(
        _skill_view_with_bump({"name": "verified-skill"}, session_id="session-1")
    )

    assert result["success"] is True
    loaded = loaded_verification_specs("session-1")
    assert len(loaded) == 1
    assert loaded[0].name == "verified-skill"
    assert pending_verification_specs("session-1") == []
    assert run_before_final_verifications("session-1") is None

    assert activate_skill_verification("session-1", "verified-skill") is True
    specs = pending_verification_specs("session-1")
    assert len(specs) == 1
    assert specs[0].name == "verified-skill"

    verification = run_before_final_verifications("session-1")

    assert verification is not None
    assert verification.blocked is False
    assert verification.checks[0].passed is True
    assert pending_verification_specs("session-1") == []


def test_skill_view_support_file_does_not_record_verifier(tmp_path, monkeypatch):
    from agent.skill_verification import (
        clear_session_verifications,
        loaded_verification_specs,
        pending_verification_specs,
    )
    from tools import skills_tool
    from tools.skills_tool import _skill_view_with_bump

    skill_dir = tmp_path / "verified-skill"
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_skill_md("true"), encoding="utf-8")
    (refs_dir / "notes.md").write_text("notes", encoding="utf-8")

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path)
    clear_session_verifications("session-2")

    result = json.loads(
        _skill_view_with_bump(
            {"name": "verified-skill", "file_path": "references/notes.md"},
            session_id="session-2",
        )
    )

    assert result["success"] is True
    assert loaded_verification_specs("session-2") == []
    assert pending_verification_specs("session-2") == []


def test_failed_before_final_verifier_blocks_response(monkeypatch):
    from agent.skill_verification import (
        activate_skill_verification,
        clear_session_verifications,
        record_skill_view_payload,
        run_before_final_verifications,
    )

    monkeypatch.setattr(
        "agent.skill_verification._command_verifiers_enabled", lambda: True
    )
    session_id = "session-fail"
    command = f"{shlex.quote(sys.executable)} -c 'import sys; sys.exit(7)'"
    clear_session_verifications(session_id)

    recorded = record_skill_view_payload(
        session_id,
        {
            "success": True,
            "name": "failing-skill",
            "content": _skill_md(command, name="failing-skill"),
        },
    )
    activated = activate_skill_verification(session_id, "failing-skill")
    verification = run_before_final_verifications(session_id)

    assert recorded is True
    assert activated is True
    assert verification is not None
    assert verification.blocked is True
    assert verification.checks[0].exit_code == 7
    assert "Skill verification failed before the final response." in verification.message
    assert "failing-skill" in verification.message


def test_activation_accepts_requested_skill_alias(monkeypatch):
    from agent.skill_verification import (
        activate_skill_verification,
        clear_session_verifications,
        record_skill_view_payload,
    )

    monkeypatch.setattr(
        "agent.skill_verification._command_verifiers_enabled", lambda: True
    )
    session_id = "session-alias"
    clear_session_verifications(session_id)

    recorded = record_skill_view_payload(
        session_id,
        {
            "success": True,
            "name": "verified-skill",
            "_requested_name": "category/verified-skill",
            "content": _skill_md("true", name="verified-skill"),
        },
    )

    assert recorded is True
    assert activate_skill_verification(session_id, "category/verified-skill") is True


def test_command_verifier_disabled_by_default_blocks_response(monkeypatch):
    from agent.skill_verification import (
        activate_skill_verification,
        clear_session_verifications,
        record_skill_view_payload,
        run_before_final_verifications,
    )

    session_id = "session-disabled"
    clear_session_verifications(session_id)

    recorded = record_skill_view_payload(
        session_id,
        {
            "success": True,
            "name": "disabled-command-skill",
            "content": _skill_md("true", name="disabled-command-skill"),
        },
    )
    activated = activate_skill_verification(session_id, "disabled-command-skill")
    verification = run_before_final_verifications(session_id)

    assert recorded is True
    assert activated is True
    assert verification is not None
    assert verification.blocked is True
    assert verification.checks[0].passed is False
    assert "command verifiers are disabled" in verification.checks[0].error


def test_skill_activate_tool_arms_loaded_verifier(tmp_path, monkeypatch):
    from agent.skill_verification import (
        clear_session_verifications,
        pending_verification_specs,
    )
    from tools import skills_tool
    from tools.skills_tool import _skill_activate, _skill_view_with_bump

    skill_dir = tmp_path / "verified-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(_skill_md("true"), encoding="utf-8")

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path)
    clear_session_verifications("session-tool")

    viewed = json.loads(
        _skill_view_with_bump({"name": "verified-skill"}, session_id="session-tool")
    )
    activated = json.loads(
        _skill_activate({"name": "verified-skill"}, session_id="session-tool")
    )

    assert viewed["success"] is True
    assert activated["success"] is True
    assert activated["verification_armed"] is True
    assert len(pending_verification_specs("session-tool")) == 1
