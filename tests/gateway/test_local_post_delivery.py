"""A local session's turn runs its post-delivery callbacks once it settles, and a deleted local session
leaves nothing keyed by it on the local adapter.

A messaging reply's adapter task (``_process_message_background``) runs the callbacks a turn registered
(background-review release, temp-bubble cleanup, a deferred /goal verdict) after its send. A Desktop/TUI/CLI
turn runs through the authority's drain with no such task: the callbacks must run there, or they never
fire (the /goal verdict never reaches the viewer) and the adapter's per-route slot keeps the turn's whole
context alive until restart.
"""
import pytest

from tests.waiting import async_wait_until


async def _owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from persistence_assembly import open_persistence, profile_scope

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = open_persistence(profile_scope(home=tmp_path))
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner", persistence=db)
    return runner, authority, db


def _viewer():
    from gateway.session_contract import SessionActor
    return SessionActor("test-owner", "default", frozenset(
        {"session:create", "session:submit", "session:read", "session:control"}), "viewer")


@pytest.mark.asyncio
async def test_local_turn_runs_its_post_delivery_callbacks_after_settling(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    from gateway.session_local import create_local_session

    runner, authority, db = await _owner(tmp_path, monkeypatch)
    fired = []

    async def turn(event):
        live = authority.sessions[ref.session_id]
        adapter = runner._delivery_adapter_for(live.source)
        admission = event.message_id

        def after_delivery():  # what TurnRunner / the /goal judge register for the turn
            fired.append((event.text, db.admissions.get(admission_id=admission).status))
        adapter.register_post_delivery_callback(live.route, after_delivery, generation=1)
        if event.text == "fails":
            raise RuntimeError("turn failed")
        return "ok"

    monkeypatch.setattr(runner, "_handle_message", turn)
    try:
        ref = create_local_session(authority, _viewer(), dict(request_id="r", source="tui", model="fixture",
                                                              toolsets=[]))
        live = authority.sessions[ref.session_id]
        for text in ("answers", "fails"):
            await authority.submit(_viewer(), Submission(text, ref, {"text": text}, "queue"))
            await async_wait_until(lambda: text in [t for t, _ in fired], timeout=10, what=f"the {text} turn")
        adapter = runner._delivery_adapter_for(live.source)
        assert adapter._post_delivery_callbacks == {}
    finally:
        db.close()
    # Once each, after the turn's outcome committed: the reply was out before they ran.
    assert fired == [("answers", "terminal"), ("fails", "terminal")]


@pytest.mark.asyncio
async def test_post_delivery_is_taken_by_the_key_the_turn_registered_under(tmp_path, monkeypatch):
    """A route rebind while the turn runs must not strand the slot under the key it registered with."""
    from gateway.session_contract import Submission
    from gateway.session_local import create_local_session

    runner, authority, db = await _owner(tmp_path, monkeypatch)
    fired = []

    async def turn(event):
        live = authority.sessions[ref.session_id]
        runner._delivery_adapter_for(live.source).register_post_delivery_callback(
            live.route, lambda: fired.append(event.text), generation=1)
        live.route = live.route + ":rebound"
        return "ok"

    monkeypatch.setattr(runner, "_handle_message", turn)
    try:
        ref = create_local_session(authority, _viewer(), dict(request_id="r", source="tui", model="fixture",
                                                              toolsets=[]))
        live = authority.sessions[ref.session_id]
        await authority.submit(_viewer(), Submission("answers", ref, {"text": "answers"}, "queue"))
        await async_wait_until(lambda: fired == ["answers"], timeout=10, what="the post-delivery callback")
        assert runner._delivery_adapter_for(live.source)._post_delivery_callbacks == {}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_deleted_local_session_leaves_nothing_on_the_local_adapter(tmp_path, monkeypatch):
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from gateway.session_queued_controls import _interactive

    runner, authority, db = await _owner(tmp_path, monkeypatch)
    try:
        refs = [create_local_session(authority, _viewer(), dict(request_id=f"r{i}", source="gui", model="fixture",
                                                                toolsets=[])) for i in range(2)]
        kept, deleted = (authority.sessions[ref.session_id] for ref in refs)
        adapter = runner._delivery_adapter_for(kept.source)
        for live in (kept, deleted):
            adapter.register_post_delivery_callback(live.route, lambda: None, generation=1)
            with live.event_stream.lock:
                _interactive(authority, live.source.chat_id)
        result = await mutate_session(authority, _viewer(), refs[1], dict(
            session_id=refs[1].session_id, request_id="delete", expected_revision=0,
            expected_generation=0, operation="delete", payload={}))
        assert result.get("deleted_ids") == [refs[1].session_id]
        assert set(adapter.sources) == set(adapter.policies) == set(authority.launch_sources) == {
            refs[0].session_id}
        assert set(adapter._post_delivery_callbacks) == {kept.route}
        assert adapter.authorize_source(kept.source)
    finally:
        db.close()
