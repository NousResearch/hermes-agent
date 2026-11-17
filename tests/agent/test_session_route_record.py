"""A provider fallback is recorded on the session row: what was asked for, and what answered."""

import json
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from run_agent import AIAgent

_SID = "fallback-route-session"


def _agent(db):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="acme/primary-model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id=_SID,
            fallback_model=[{"provider": "zai", "model": "glm-5.2"}],
        )
    agent.client = MagicMock()
    return agent


def _fallback_client():
    client = MagicMock()
    client.base_url, client.api_key = "https://api.z.ai/v1", "fb-key"
    return client


def test_fallback_switch_is_stored_on_the_session_row_and_its_list_projection(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(_SID, source="cli", model="acme/primary-model", model_config={"_branched_from": "parent-1"})
    agent = _agent(db)
    requested_provider = agent.provider

    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "glm-5.2")):
        assert agent._try_activate_fallback() is True

    config = json.loads(db.get_session(_SID)["model_config"])
    assert config["_branched_from"] == "parent-1"  # merged, never clobbered
    route = config["effective_route"]
    assert route["active"] is True and route["route_source"] == "fallback"
    assert route["requested"] == {"model": "acme/primary-model", "provider": requested_provider}
    assert route["effective"] == {"model": "glm-5.2", "provider": "zai"}

    # The list endpoints strip model_config; the row still says which model answered.
    [row] = [r for r in db.list_sessions_rich(compact_rows=True, include_children=True) if r["id"] == _SID]
    assert row["effective_route"]["effective"]["model"] == "glm-5.2"
    assert row["_branched_from"] == "parent-1"


def test_a_turn_back_on_the_requested_model_marks_the_fallback_inactive(tmp_path):
    from agent.session_route_record import settle_effective_route

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(_SID, source="cli", model="acme/primary-model")
    agent = _agent(db)
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "glm-5.2")):
        assert agent._try_activate_fallback() is True

    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._restore_primary_runtime() is True
    settle_effective_route(agent)

    route = json.loads(db.get_session(_SID)["model_config"])["effective_route"]
    assert route["active"] is False and route["route_source"] == "primary"
    assert route["effective"]["model"] == "acme/primary-model"
    assert route["last_fallback"]["effective"] == {"model": "glm-5.2", "provider": "zai"}


def test_a_session_resumed_onto_its_fallback_keeps_the_record_active(tmp_path):
    """A rebuilt agent restored onto the fallback route (gateway_runtime names it) has not recovered."""
    from agent.session_route_record import settle_effective_route

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(_SID, source="cli", model="acme/primary-model")
    first = _agent(db)
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "glm-5.2")):
        assert first._try_activate_fallback() is True
    resumed = _agent(db)
    resumed.model, resumed.provider = "glm-5.2", "zai"
    settle_effective_route(resumed)

    route = json.loads(db.get_session(_SID)["model_config"])["effective_route"]
    assert route["active"] is True and route["requested"]["model"] == "acme/primary-model"


def test_a_compressed_conversation_lists_its_tips_route(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("root", source="desktop", model="m", model_config={"effective_route": {"active": True}})
    db.create_session("tip", source="desktop", model="m", parent_session_id="root",
                      model_config={"effective_route": {"active": False, "route_source": "primary"}})
    db.end_session("root", "compression")

    [row] = [r for r in db.list_sessions_rich(compact_rows=True) if r.get("_lineage_root_id") == "root"]
    assert row["effective_route"] == {"active": False, "route_source": "primary"}
