import time

from hermes_state import SessionDB
from run_agent import AIAgent


def _make_agent(db: SessionDB, session_id: str, platform: str = "telegram") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent.session_id = session_id
    agent.platform = platform
    return agent


def test_build_skippy_activity_notice_empty_without_prior_session(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("current", source="telegram", model="test")

    agent = _make_agent(db, "current")

    assert agent._build_skippy_activity_notice(is_first_turn=True) == ""
    assert agent._build_skippy_activity_notice(is_first_turn=False) == ""



def test_build_skippy_activity_notice_reports_newer_skippy_activity(tmp_path):
    db = SessionDB(tmp_path / "state.db")

    db.create_session("old-chat", source="telegram", model="test")
    db.append_message("old-chat", "user", "Last time we talked about email triage.")

    time.sleep(0.02)

    db.create_session("skippy-session", source="cli", model="test")
    db.append_message("skippy-session", "assistant", "Skippy and OpenClaw discussed thread routing.")

    time.sleep(0.02)

    db.create_session("current", source="telegram", model="test")
    db.append_message("current", "user", "Hello again")

    agent = _make_agent(db, "current")
    notice = agent._build_skippy_activity_notice(is_first_turn=True)

    assert "New linked activity since the last chat" in notice
    assert "Skippy/OpenClaw" in notice
    assert "thread routing" in notice
    assert "old-chat" not in notice



def test_build_skippy_activity_notice_ignores_older_matches(tmp_path):
    db = SessionDB(tmp_path / "state.db")

    db.create_session("ancient-skippy", source="cli", model="test")
    db.append_message("ancient-skippy", "assistant", "Skippy mentioned OpenClaw months ago.")

    time.sleep(0.02)

    db.create_session("old-chat", source="telegram", model="test")
    db.append_message("old-chat", "user", "Our previous conversation.")

    time.sleep(0.02)

    db.create_session("current", source="telegram", model="test")

    agent = _make_agent(db, "current")

    assert agent._build_skippy_activity_notice(is_first_turn=True) == ""
