from hermes_state import SessionDB
from run_agent import AIAgent


def _agent_for_db_session(*, db, session_id, platform="slack", user_id=None, parent_session_id=None):
    agent = AIAgent.__new__(AIAgent)
    agent._session_db_created = False
    agent._session_db = db
    agent.session_id = session_id
    agent.platform = platform
    agent.model = "test-model"
    agent._session_init_model_config = {}
    agent._cached_system_prompt = ""
    agent._parent_session_id = parent_session_id
    agent._user_id = user_id
    return agent


def test_gateway_agent_session_creation_persists_user_id(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        agent = _agent_for_db_session(
            db=db,
            session_id="gateway_session",
            platform="slack",
            user_id="U_KENNY",
        )

        agent._ensure_db_session()

        session = db.get_session("gateway_session")
        assert session is not None
        assert session["source"] == "slack"
        assert session["user_id"] == "U_KENNY"
    finally:
        db.close()


def test_compressed_child_session_creation_inherits_parent_user_id(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            "parent_session",
            source="slack",
            model="test-model",
            user_id="U_KENNY",
        )
        agent = _agent_for_db_session(
            db=db,
            session_id="compressed_child_session",
            parent_session_id="parent_session",
            platform="slack",
        )

        agent._ensure_db_session()

        session = db.get_session("compressed_child_session")
        assert session is not None
        assert session["source"] == "slack"
        assert session["parent_session_id"] == "parent_session"
        assert session["user_id"] == "U_KENNY"
    finally:
        db.close()


def test_compressed_child_session_creation_inherits_parent_source_when_platform_missing(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            "parent_session",
            source="slack",
            model="test-model",
            user_id="U_KENNY",
        )
        agent = _agent_for_db_session(
            db=db,
            session_id="compressed_child_session",
            parent_session_id="parent_session",
            platform=None,
        )

        agent._ensure_db_session()

        session = db.get_session("compressed_child_session")
        assert session is not None
        assert session["source"] == "slack"
        assert session["parent_session_id"] == "parent_session"
        assert session["user_id"] == "U_KENNY"
    finally:
        db.close()
