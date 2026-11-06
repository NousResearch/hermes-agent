import json

import pytest

from agent.insights import InsightsEngine
from hermes_state import SessionDB


@pytest.mark.parametrize("report_kind", ["full", "usage"])
@pytest.mark.parametrize(
    "malformed_function",
    [None, [], 42, "invalid", {"name": []}, {"name": {}}, {"name": 42}, {"name": None}, {}],
)
def test_insights_preserve_valid_calls_beside_malformed_functions(tmp_path, report_kind, malformed_function):
    calls = [
        {"function": {"name": "skill_view", "arguments": json.dumps({"name": "example"})}},
        {"function": malformed_function},
        {"function": {"name": "read_file", "arguments": json.dumps({"path": "example.txt"})}},
        {"function": {"name": "skill_manage", "arguments": json.dumps({"name": "example"})}},
    ]
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("insights-malformed", "cli")
        db.append_message("insights-malformed", "assistant", tool_calls=calls)
        engine = InsightsEngine(db)

        if report_kind == "full":
            report = engine.generate()
        else:
            report = engine.get_usage_breakdown()

        tool_counts = {}
        for tool in report["tools"]:
            tool_counts[tool["tool"]] = tool["count"]
        assert tool_counts == {"skill_view": 1, "read_file": 1, "skill_manage": 1}
        assert report["skills"]["summary"] == {
            "total_skill_loads": 1,
            "total_skill_edits": 1,
            "total_skill_actions": 2,
            "distinct_skills_used": 1,
        }
        skill = report["skills"]["top_skills"][0]
        assert skill["skill"] == "example"
        assert skill["view_count"] == 1
        assert skill["manage_count"] == 1
        if report_kind == "full":
            assert "read_file" in engine.format_terminal(report)
            assert "read_file" in engine.format_gateway(report)
