"""Usage analytics share SQLite's report contract over real PostgreSQL rows."""

from agent.insights import InsightsEngine
from hermes_state import SessionDB


def test_read_only_postgres_insights_include_tools_skills_and_source_filter(postgres_db, postgres_home):
    postgres_db.create_session("cli-insights", source="cli", model="gpt-4o")
    postgres_db.append_message("cli-insights", "user", "Review a change")
    postgres_db.append_message("cli-insights", "assistant", "Loading review instructions", tool_calls=[
        {"function": {"name": "skill_view", "arguments": '{"name":"review"}'}},
    ])
    postgres_db.append_message("cli-insights", "tool", "Instructions", tool_name="skill_view")
    postgres_db.create_session("telegram-insights", source="telegram")
    postgres_db.append_message("telegram-insights", "user", "Another conversation")

    with SessionDB(postgres_home / "state.db", read_only=True) as reader:
        engine = InsightsEngine(reader)
        report = engine.generate(source="cli")
        all_sessions = engine.generate()

    assert not report["empty"]
    assert report["overview"]["total_sessions"] == 1
    assert all_sessions["overview"]["total_sessions"] == 2
    assert report["tools"] == [{"tool": "skill_view", "count": 1, "percentage": 100}]
    assert report["skills"]["top_skills"][0]["skill"] == "review"
    assert report["skills"]["top_skills"][0]["view_count"] == 1
