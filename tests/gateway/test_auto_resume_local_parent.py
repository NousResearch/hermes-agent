"""The auto-resume stand-in for a local (desktop/CLI-served) parent runs under that session's frozen policy, even when
the session ended automatically and boot left it unrestored: the scheduler restores it first, as a completion does."""
import pytest


@pytest.mark.asyncio
async def test_an_unrestored_local_parent_is_restored_before_its_stand_in_is_built(tmp_path, monkeypatch):
    from hermes_agent.hermes_state_local import end_idle_local_session
    from tests.gateway.test_session_end_delegations import _local
    from tests.gateway.test_unviewed_completion_notes import _restart, _target
    from gateway import run_auto_resume
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
    assert end_idle_local_session(authority.db, epoch=authority.epoch, session_id=ref.session_id,
                                  target_id=target, reason="agent_close")  # automatic: not a cancel
    runner, authority = await _restart(tmp_path, monkeypatch, authority)
    policies = []

    def stand_in(_runner, _db, _record, _source, _route, policy=None):
        policies.append(policy)
        raise RuntimeError("stop here")
    monkeypatch.setattr(run_auto_resume, "_stand_in_parent", stand_in)
    try:
        out = run_auto_resume._continue_as_parent(runner, None, {"subagent_id": "sa-1"},
                                                  {"origin_session": route, "parent_session_id": target},
                                                  {"session_key": route, "session_id": target}, "go on")
        assert "stop here" in out["error"] and len(policies) == 1 and policies[0] is not None
    finally:
        authority.db.close()
