"""The review fork is told up front which consulted skills it may write."""

import json
from unittest.mock import patch

from agent.background_review import _consulted_skill_names, _skill_write_eligibility_block


def _snapshot():
    return [
        {"role": "user", "content": '[IMPORTANT: The user has invoked the "pinned-skill" skill, indicating '
                                    "they want you to follow its instructions.]\n\nfix the thing"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "skill_view",
                                                          "arguments": json.dumps({"name": "managed-skill"})}},
            {"id": "c2", "type": "function", "function": {"name": "skill_view",
                                                          "arguments": json.dumps({"name": "pinned-skill"})}},
            {"id": "c3", "type": "function", "function": {"name": "read_file",
                                                          "arguments": json.dumps({"path": "x"})}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "..."},
    ]


def test_consulted_skill_names_come_from_skill_view_calls_and_invocation_markers():
    assert _consulted_skill_names(_snapshot()) == ["pinned-skill", "managed-skill"]


def test_eligibility_block_names_protected_skills_before_the_fork_tries_them(tmp_path):
    for name in ("pinned-skill", "managed-skill", "manual-skill"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n# {name}\n")
    usage = {
        "pinned-skill": {"pinned": True, "created_by": "agent"},
        "managed-skill": {"pinned": False, "created_by": "agent"},
        "manual-skill": {"pinned": False, "created_by": None},
    }
    snapshot = _snapshot() + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c4", "type": "function", "function": {"name": "skill_view",
                                                      "arguments": json.dumps({"name": "manual-skill"})}}]}]
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch("tools.skill_usage.get_record", side_effect=lambda n: usage.get(n, {})), \
         patch("tools.skill_usage.load_usage", return_value=usage), \
         patch("tools.skill_usage.is_protected_builtin", return_value=False), \
         patch("tools.skill_usage.is_hub_installed", return_value=False), \
         patch("tools.skill_usage.is_bundled", return_value=False):
        block = _skill_write_eligibility_block(snapshot)

    assert "pinned-skill — PROTECTED (pinned)" in block
    assert "manual-skill — PROTECTED (not curator-managed" in block
    assert "managed-skill — writable" in block
    assert _skill_write_eligibility_block([{"role": "user", "content": "hi"}]) == ""
