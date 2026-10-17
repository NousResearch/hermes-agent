"""A compression rotation re-aliases the active turn's review admission to the child session: the
facade's token and the gateway admission's token both follow the rotation, so a review spawned
against the child still sees the live turn, and nothing leaks once the turn finishes."""

from pathlib import Path
from types import SimpleNamespace

from agent import review_admission
from hermes_state import SessionDB
from tests.agent.test_compression_concurrent_fork import _build_agent_with_db


def test_compression_rotation_aliases_active_review_admission(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "REVIEW_ADMISSION_ROTATION_PARENT"
    db.create_session(parent_sid, source="cli")
    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    profile_key = "/profiles/alpha"
    token = review_admission.note_turn_started(parent_sid, profile_key)
    gateway_token = review_admission.note_turn_started(parent_sid, profile_key)
    agent._active_turn_token = token
    agent._active_turn_profile_key = profile_key
    agent._gateway_review_admission = SimpleNamespace(
        token=gateway_token,
        profile_key=profile_key,
        session_id=parent_sid,
    )
    try:
        agent._compress_context(messages, "sys", approx_tokens=120_000)
        child_sid = agent.session_id

        assert child_sid != parent_sid
        assert review_admission.other_live_turn(child_sid, None, profile_key)
        assert review_admission.other_live_turn(child_sid, token, profile_key)
    finally:
        review_admission.note_turn_finished(parent_sid, token, profile_key)
        review_admission.note_turn_finished(parent_sid, gateway_token, profile_key)

    assert review_admission.other_live_turn(child_sid, None, profile_key) is False
