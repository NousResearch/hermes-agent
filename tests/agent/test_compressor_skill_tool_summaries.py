"""Summaries of skill-tool results must carry the skill names and the failure
outcome across compression — `skill_manage` names live inside ``operations``
(or the legacy flat shape), and ``skills_list`` has no top-level ``name`` at
all. A failed call must never compress into a success-shaped line."""

import json

from agent.context_compressor import _summarize_tool_result


class TestSummarizeSkillManage:
    def test_operations_array_names_the_skills(self):
        args = json.dumps({
            "operations": [
                {
                    "action": "create",
                    "name": "orca-ade",
                    "content": "---\nname: orca-ade\n---\nbody",
                },
                {
                    "action": "patch",
                    "name": "llama-server",
                    "old_string": "a",
                    "new_string": "b",
                },
            ]
        })
        content = json.dumps({
            "success": True,
            "operations_applied": 2,
            "results": [
                {"name": "orca-ade", "action": "create", "success": True},
                {"name": "llama-server", "action": "patch", "success": True},
            ],
        })
        summary = _summarize_tool_result("skill_manage", args, content)
        assert "create orca-ade" in summary
        assert "patch llama-server" in summary
        assert "name=?" not in summary
        assert "FAILED" not in summary

    def test_failed_batch_keeps_error_visible(self):
        args = json.dumps({"operations": [{"action": "create", "name": "orca-ade"}]})
        content = json.dumps({
            "success": False,
            "error": (
                "operations[0] (create on 'orca-ade') failed: content is "
                "required for 'create' — batch aborted, all touched skills "
                "rolled back."
            ),
        })
        summary = _summarize_tool_result("skill_manage", args, content)
        assert "FAILED" in summary
        assert "create orca-ade" in summary
        assert "content is required" in summary

    def test_legacy_flat_shape_names_the_skill(self):
        args = json.dumps({"action": "delete", "name": "stale-skill"})
        content = json.dumps({
            "success": True,
            "message": "Skill 'stale-skill' deleted.",
        })
        summary = _summarize_tool_result("skill_manage", args, content)
        assert "delete stale-skill" in summary
        assert "name=?" not in summary
        assert "FAILED" not in summary

    def test_malformed_ops_render_guarded_not_crash(self):
        args = json.dumps({"operations": ["not-a-dict", 42]})
        content = json.dumps({
            "success": False,
            "error": "operations[0] needs an 'action'.",
        })
        summary = _summarize_tool_result("skill_manage", args, content)
        assert summary.startswith("[skill_manage]")
        assert "FAILED" in summary

    def test_error_text_is_bounded_to_one_line(self):
        args = json.dumps({"operations": [{"action": "patch", "name": "s"}]})
        content = json.dumps({"success": False, "error": "x" * 500})
        summary = _summarize_tool_result("skill_manage", args, content)
        assert "\n" not in summary
        assert len(summary) < 250


class TestSummarizeSkillsList:
    def test_success_lists_count_and_category(self):
        args = json.dumps({"category": "devops"})
        content = json.dumps({
            "success": True,
            "skills": [{"name": "a"}, {"name": "b"}],
            "categories": ["devops"],
            "count": 2,
        })
        summary = _summarize_tool_result("skills_list", args, content)
        assert "category=devops" in summary
        assert "2 skills" in summary
        assert "name=?" not in summary
        assert "FAILED" not in summary

    def test_failure_keeps_error_visible(self):
        args = json.dumps({})
        content = json.dumps({"success": False, "error": "skills directory unreadable"})
        summary = _summarize_tool_result("skills_list", args, content)
        assert "FAILED" in summary
        assert "skills directory unreadable" in summary


class TestSummarizerRegistryEntries:
    def test_skill_tool_stubs_never_render_question_name(self):
        cases = [
            ("skills_list", "{}", '{"success": true, "skills": [], "count": 0}'),
            ("skill_manage", "{}", '{"success": true}'),
            ("skill_view", '{"name": "real-skill"}', "x" * 300),
        ]
        for tool, args, content in cases:
            summary = _summarize_tool_result(tool, args, content)
            assert "name=?" not in summary, (tool, summary)
