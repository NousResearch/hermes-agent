#!/usr/bin/env python3
"""End-to-end test for skill precipitation verification.

This script directly exercises the new verification functions in
agent/background_review.py by simulating the message structure that
a real background review fork would produce.

Usage:
    python tests/manual/test_skill_verification.py
"""

import json
import sys
import os

# Ensure the repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.background_review import (
    _extract_precipitated_skill_names,
    _command_escapes_sandbox,
    _scan_executed_commands_for_escape,
    _analyze_skill_manage_activity,
    _VERIFY_PROMPT,
    _VERIFY_MAX_FIX_ATTEMPTS,
    _VERIFY_ACTIONS,
)


def make_assistant_msg(tool_calls: list[dict]) -> dict:
    """Build a synthetic assistant message with tool_calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                },
            }
            for tc in tool_calls
        ],
    }


def make_tool_result(tool_call_id: str, content: dict) -> dict:
    """Build a synthetic tool-result message."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(content, ensure_ascii=False),
    }


def test_extract_single_create():
    """A single skill_manage(create) → should be extracted."""
    review_messages = [
        make_assistant_msg([
            {"id": "tc-1", "name": "skill_manage", "args": {"action": "create", "name": "my-test-skill", "content": "# Test Skill\n..."}},
        ]),
        make_tool_result("tc-1", {"success": True, "message": "Skill created"}),
    ]
    result = _extract_precipitated_skill_names(review_messages, [])
    assert result == [("my-test-skill", "create")], f"Expected [('my-test-skill', 'create')], got {result}"
    print("  ✅ test_extract_single_create PASSED")


def test_extract_single_patch():
    """A single skill_manage(patch) → should be extracted."""
    review_messages = [
        make_assistant_msg([
            {"id": "tc-2", "name": "skill_manage", "args": {"action": "patch", "name": "existing-skill", "old_string": "foo", "new_string": "bar"}},
        ]),
        make_tool_result("tc-2", {"success": True, "message": "Skill patched"}),
    ]
    result = _extract_precipitated_skill_names(review_messages, [])
    assert result == [("existing-skill", "patch")], f"Expected [('existing-skill', 'patch')], got {result}"
    print("  ✅ test_extract_single_patch PASSED")


def test_extract_skips_delete():
    """skill_manage(delete) should NOT be extracted."""
    review_messages = [
        make_assistant_msg([
            {"id": "tc-3", "name": "skill_manage", "args": {"action": "delete", "name": "old-skill"}},
        ]),
        make_tool_result("tc-3", {"success": True, "message": "Skill deleted"}),
    ]
    result = _extract_precipitated_skill_names(review_messages, [])
    assert result == [], f"Expected [], got {result}"
    print("  ✅ test_extract_skips_delete PASSED")


def test_extract_skips_failed():
    """A failed skill_manage should NOT be extracted."""
    review_messages = [
        make_assistant_msg([
            {"id": "tc-4", "name": "skill_manage", "args": {"action": "create", "name": "bad-skill", "content": "..."}},
        ]),
        make_tool_result("tc-4", {"success": False, "error": "Name already exists"}),
    ]
    result = _extract_precipitated_skill_names(review_messages, [])
    assert result == [], f"Expected [], got {result}"
    print("  ✅ test_extract_skips_failed PASSED")


def test_extract_skips_stale_prior():
    """Tool results from prior_snapshot should be skipped."""
    prior_snapshot = [
        make_tool_result("tc-old", {"success": True, "message": "Old skill created"}),
    ]
    review_messages = [
        # The review agent inherits the prior history, so tc-old appears
        # in review_messages too — but should be excluded.
        make_assistant_msg([
            {"id": "tc-old", "name": "skill_manage", "args": {"action": "create", "name": "stale-skill", "content": "..."}},
        ]),
        make_tool_result("tc-old", {"success": True, "message": "Old skill created"}),
        # This is a genuinely new skill
        make_assistant_msg([
            {"id": "tc-5", "name": "skill_manage", "args": {"action": "create", "name": "fresh-skill", "content": "..."}},
        ]),
        make_tool_result("tc-5", {"success": True, "message": "Fresh skill created"}),
    ]
    result = _extract_precipitated_skill_names(review_messages, prior_snapshot)
    assert result == [("fresh-skill", "create")], f"Expected [('fresh-skill', 'create')], got {result}"
    print("  ✅ test_extract_skips_stale_prior PASSED")


def test_extract_dedup_last_wins():
    """If a skill is created then patched, last action wins."""
    review_messages = [
        make_assistant_msg([
            {"id": "tc-6", "name": "skill_manage", "args": {"action": "create", "name": "multi-skill", "content": "..."}},
        ]),
        make_tool_result("tc-6", {"success": True, "message": "Skill created"}),
        make_assistant_msg([
            {"id": "tc-7", "name": "skill_manage", "args": {"action": "patch", "name": "multi-skill", "old_string": "x", "new_string": "y"}},
        ]),
        make_tool_result("tc-7", {"success": True, "message": "Skill patched"}),
    ]
    result = _extract_precipitated_skill_names(review_messages, [])
    assert result == [("multi-skill", "patch")], f"Expected [('multi-skill', 'patch')], got {result}"
    print("  ✅ test_extract_dedup_last_wins PASSED")


def test_extract_multiple_skills():
    """Multiple distinct skills → all extracted."""
    review_messages = [
        make_assistant_msg([
            {"id": "tc-8", "name": "skill_manage", "args": {"action": "create", "name": "skill-a", "content": "..."}},
        ]),
        make_tool_result("tc-8", {"success": True, "message": "Skill A created"}),
        make_assistant_msg([
            {"id": "tc-9", "name": "skill_manage", "args": {"action": "create", "name": "skill-b", "content": "..."}},
        ]),
        make_tool_result("tc-9", {"success": True, "message": "Skill B created"}),
    ]
    result = _extract_precipitated_skill_names(review_messages, [])
    assert len(result) == 2, f"Expected 2 skills, got {len(result)}"
    names = {name for name, _ in result}
    assert names == {"skill-a", "skill-b"}, f"Expected {{skill-a, skill-b}}, got {names}"
    print("  ✅ test_extract_multiple_skills PASSED")


def test_extract_empty_messages():
    """Empty review_messages → empty result."""
    result = _extract_precipitated_skill_names([], [])
    assert result == [], f"Expected [], got {result}"
    print("  ✅ test_extract_empty_messages PASSED")


def test_extract_no_skill_manage():
    """Messages without skill_manage → empty result."""
    review_messages = [
        make_assistant_msg([
            {"id": "tc-10", "name": "memory", "args": {"action": "add", "content": "User likes pizza"}},
        ]),
        make_tool_result("tc-10", {"success": True, "message": "Entry added"}),
    ]
    result = _extract_precipitated_skill_names(review_messages, [])
    assert result == [], f"Expected [], got {result}"
    print("  ✅ test_extract_no_skill_manage PASSED")


# ---------------------------------------------------------------------------
# Sandbox escape detection tests
# ---------------------------------------------------------------------------

def test_escape_cd_home():
    """cd ~ escapes the sandbox."""
    reason = _command_escapes_sandbox("cd ~ && ls", "/tmp/hermes-skill-verify-abc")
    assert reason is not None, "cd ~ should be flagged"
    print("  ✅ test_escape_cd_home PASSED")


def test_escape_cd_absolute_outside():
    """cd to an absolute path outside scratch is an escape."""
    reason = _command_escapes_sandbox("cd /etc && cat passwd", "/tmp/hermes-skill-verify-abc")
    assert reason is not None, "cd /etc should be flagged"
    print("  ✅ test_escape_cd_absolute_outside PASSED")


def test_escape_dotdot():
    """cd .. leaves the scratch dir."""
    reason = _command_escapes_sandbox("cd .. && git status", "/tmp/hermes-skill-verify-abc")
    assert reason is not None, "cd .. should be flagged"
    print("  ✅ test_escape_dotdot PASSED")


def test_escape_write_to_home():
    """Writing to ~/something is an escape."""
    reason = _command_escapes_sandbox("echo x > ~/evil.txt", "/tmp/hermes-skill-verify-abc")
    assert reason is not None, "write to ~ should be flagged"
    print("  ✅ test_escape_write_to_home PASSED")


def test_escape_rm_system_dir():
    """rm -rf on a system dir is an escape."""
    reason = _command_escapes_sandbox("rm -rf /etc/passwd", "/tmp/hermes-skill-verify-abc")
    assert reason is not None, "rm /etc/passwd should be flagged"
    print("  ✅ test_escape_rm_system_dir PASSED")


def test_escape_global_git_config():
    """git config --global writes real user state."""
    reason = _command_escapes_sandbox("git config --global user.name test", "/tmp/hermes-skill-verify-abc")
    assert reason is not None, "git config --global should be flagged"
    print("  ✅ test_escape_global_git_config PASSED")


def test_no_escape_sandbox_commands():
    """Commands operating inside scratch are NOT flagged."""
    scratch = "/tmp/hermes-skill-verify-abc"
    safe_commands = [
        "git init",
        "touch test.txt",
        "git add . && git commit -m 'init'",
        "echo hello > README.md",
        "mkdir -p src && touch src/main.py",
        "ls -la",
        "git -c user.name=X -c user.email=Y commit -m 'msg'",
        "git config user.name test",  # repo-level config, inside sandbox
    ]
    for cmd in safe_commands:
        reason = _command_escapes_sandbox(cmd, scratch)
        assert reason is None, f"Expected safe, got flagged: {cmd!r} -> {reason}"
    print("  ✅ test_no_escape_sandbox_commands PASSED")


def test_scan_executed_commands():
    """_scan_executed_commands_for_escape walks tool messages for command escapes."""
    scratch = "/tmp/hermes-skill-verify-abc"
    session_messages = [
        make_tool_result("tc-t1", {"success": True, "command": "git init"}),
        make_tool_result("tc-t2", {"success": True, "command": "echo x > ~/leak.txt"}),
        make_tool_result("tc-t3", {"success": True, "command": "git status"}),
    ]
    escaped = _scan_executed_commands_for_escape(session_messages, scratch)
    assert len(escaped) == 1, f"Expected 1 escape, got {escaped}"
    assert "leak.txt" in escaped[0], f"Expected leak.txt in {escaped[0]}"
    print("  ✅ test_scan_executed_commands PASSED")


# ---------------------------------------------------------------------------
# Repair-loop tests (_analyze_skill_manage_activity + prompt)
# ---------------------------------------------------------------------------

def test_analyze_no_skill_manage():
    """No skill_manage writes → no violations, not repaired."""
    session_messages = [
        make_assistant_msg([
            {"id": "tc-a1", "name": "skill_view", "args": {"name": "s"}},
        ]),
        make_tool_result("tc-a1", {"success": True, "content": "# Skill"}),
    ]
    violations, repaired = _analyze_skill_manage_activity(session_messages, "s")
    assert violations == [], f"Expected no violations, got {violations}"
    assert repaired is False, f"Expected repaired=False, got {repaired}"
    print("  ✅ test_analyze_no_skill_manage PASSED")


def test_analyze_repair_target_success():
    """A successful patch on the target skill → repaired, no violations."""
    session_messages = [
        make_assistant_msg([
            {"id": "tc-a2", "name": "skill_manage", "args": {"action": "patch", "name": "my-skill", "old_string": "a", "new_string": "b"}},
        ]),
        make_tool_result("tc-a2", {"success": True, "message": "Skill patched"}),
    ]
    violations, repaired = _analyze_skill_manage_activity(session_messages, "my-skill")
    assert violations == [], f"Expected no violations, got {violations}"
    assert repaired is True, f"Expected repaired=True, got {repaired}"
    print("  ✅ test_analyze_repair_target_success PASSED")


def test_analyze_writes_other_skill():
    """A write to an unrelated skill is a state-modification violation."""
    session_messages = [
        make_assistant_msg([
            {"id": "tc-a3", "name": "skill_manage", "args": {"action": "create", "name": "other-skill", "content": "..."}},
        ]),
        make_tool_result("tc-a3", {"success": True, "message": "Skill created"}),
    ]
    violations, repaired = _analyze_skill_manage_activity(session_messages, "my-skill")
    assert len(violations) == 1, f"Expected 1 violation, got {violations}"
    assert "other-skill" in violations[0], f"Expected other-skill in {violations[0]}"
    assert repaired is False, f"Expected repaired=False, got {repaired}"
    print("  ✅ test_analyze_writes_other_skill PASSED")


def test_analyze_failed_write():
    """A failed write on the target skill is a violation (agent lied / was refused)."""
    session_messages = [
        make_assistant_msg([
            {"id": "tc-a4", "name": "skill_manage", "args": {"action": "patch", "name": "my-skill", "old_string": "a", "new_string": "b"}},
        ]),
        make_tool_result("tc-a4", {"success": False, "error": "Not found"}),
    ]
    violations, repaired = _analyze_skill_manage_activity(session_messages, "my-skill")
    assert len(violations) == 1, f"Expected 1 violation, got {violations}"
    assert "failed" in violations[0], f"Expected 'failed' in {violations[0]}"
    assert repaired is False, f"Expected repaired=False, got {repaired}"
    print("  ✅ test_analyze_failed_write PASSED")


def test_analyze_remove_file_on_target():
    """A successful remove_file on the target counts as a repair, not a violation."""
    session_messages = [
        make_assistant_msg([
            {"id": "tc-a5", "name": "skill_manage", "args": {"action": "remove_file", "name": "my-skill", "file_path": "bogus.sh"}},
        ]),
        make_tool_result("tc-a5", {"success": True, "message": "File removed"}),
    ]
    violations, repaired = _analyze_skill_manage_activity(session_messages, "my-skill")
    assert violations == [], f"Expected no violations, got {violations}"
    assert repaired is True, f"Expected repaired=True, got {repaired}"
    print("  ✅ test_analyze_remove_file_on_target PASSED")


def test_verify_prompt_formats_with_repair():
    """_VERIFY_PROMPT formats with all placeholders and carries the repair section."""
    prompt = _VERIFY_PROMPT.format(
        skill_name="my-skill",
        action_label="created",
        scratch="/tmp/hermes-skill-verify-abc",
        max_fix_attempts=_VERIFY_MAX_FIX_ATTEMPTS,
    )
    assert "REPAIR" in prompt, "Expected REPAIR section in prompt"
    assert "my-skill" in prompt, "Expected skill name in prompt"
    assert str(_VERIFY_MAX_FIX_ATTEMPTS) in prompt, "Expected max_fix_attempts value in prompt"
    assert "skill_manage" in prompt, "Expected skill_manage mention in prompt"
    assert "UNABLE" in prompt, "Expected UNABLE outcome in prompt"
    print("  ✅ test_verify_prompt_formats_with_repair PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Skill Verification — Unit Tests")
    print("=" * 60)
    print(f"VERIFY_ACTIONS = {_VERIFY_ACTIONS}")
    print()

    tests = [
        test_extract_single_create,
        test_extract_single_patch,
        test_extract_skips_delete,
        test_extract_skips_failed,
        test_extract_skips_stale_prior,
        test_extract_dedup_last_wins,
        test_extract_multiple_skills,
        test_extract_empty_messages,
        test_extract_no_skill_manage,
        test_escape_cd_home,
        test_escape_cd_absolute_outside,
        test_escape_dotdot,
        test_escape_write_to_home,
        test_escape_rm_system_dir,
        test_escape_global_git_config,
        test_no_escape_sandbox_commands,
        test_scan_executed_commands,
        test_analyze_no_skill_manage,
        test_analyze_repair_target_success,
        test_analyze_writes_other_skill,
        test_analyze_failed_write,
        test_analyze_remove_file_on_target,
        test_verify_prompt_formats_with_repair,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {test.__name__} ERROR: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
