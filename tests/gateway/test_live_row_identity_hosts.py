"""The session authority and a managed worker name the stored rows of a turn's live events.

The in-process TUI gateway names them on its own emitters (agent/live_row_identity.py). A turn the
authority runs publishes its tool events from TurnRunner's callbacks, and a managed worker's travel as
frames the owner re-validates and publishes; both settle message.complete away from the agent, so the
reply's row rides on the turn result. Each test reads what a viewer would receive.
"""
from types import SimpleNamespace

from agent.live_row_identity import live_assistant_row
from agent.managed_worker import accept_result, encode_frame, live_rows_asked, read_frame, result_frame, tool_frame
from gateway.config import Platform
from gateway.session_authority import LiveSession, SessionAuthority, admission_row
from gateway.session_settlement_recovery import completion_payload
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import storage_handle
from tests.gateway.test_local_viewer_events_turn import _turn

CALL = {"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
ROW = {"role": "assistant", "content": "Looking now.", "tool_calls": [CALL], "_row_id": 12,
       "_db_persisted": True, "message_uid": "uid-12", "_tool_call_uids": {"call_0": "call-uid-12"}}
NAMED = {"row_id": 12, "message_uid": "uid-12", "tool_call_uid": "call-uid-12"}


def _wire(frame):
    import io
    return read_frame(io.BytesIO(encode_frame(frame)))


def test_authority_tool_events_name_the_running_rounds_row():
    published = []
    turn = _turn(Platform.LOCAL, published)
    agent = SimpleNamespace()
    turn._ctx.agent_holder = [agent]

    with live_assistant_row(agent, ROW):
        turn.combined_tool_start_callback("call_0", "read_file", {"path": "x"})
        turn.combined_tool_complete_callback("call_0", "read_file", {"path": "x"}, '{"content": "hi"}')
        # A call the round's row does not hold (a delegated child's) names nothing.
        turn.combined_tool_start_callback("child_call", "read_file", {"path": "y"})
    turn.combined_tool_start_callback("call_0", "read_file", {"path": "x"})  # after the round

    tools = [(kind, payload) for kind, payload in published if kind.startswith("tool.")]
    named = [{key: payload[key] for key in NAMED if key in payload} for _kind, payload in tools]
    assert [kind for kind, _payload in tools] == ["tool.start", "tool.complete", "tool.start", "tool.start"]
    assert named == [NAMED, NAMED, {}, {}]


def test_authority_completion_names_the_reply_row_only_for_a_completed_turn():
    row = {"admission_id": "a", "request_id": "r"}
    captured = {"result": {"final_response": "done", "reply_row_id": 14, "reply_message_uid": "uid-14"}}

    complete = completion_payload(row, {"outcome": "completed"}, "done", captured)
    assert (complete["row_id"], complete["message_uid"]) == (14, "uid-14")
    failed = completion_payload(row, {"outcome": "failed"}, "", captured)
    assert "row_id" not in failed and "message_uid" not in failed


def test_managed_result_frame_carries_the_reply_row_to_the_owner():
    result = {"final_response": "done", "reply_row_id": 14, "reply_message_uid": "uid-14", "messages": [ROW]}

    accepted, _usage = accept_result(_wire({"type": "result", "result": result_frame(result)})["result"])
    assert (accepted["reply_row_id"], accepted["reply_message_uid"]) == (14, "uid-14")


def test_managed_tool_frames_publish_the_rows_they_name(tmp_path):
    from gateway.run_turn_progress import publish_worker_tool_event
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(db, "s", "cli")
        epoch = db.runtime.begin_epoch(instance_id="owner")
        runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False))
        authority = SessionAuthority(runner, profile_id="owned", instance_id="owner", epoch=epoch,
                                     persistence=storage_handle(tmp_path))
        live = authority.sessions["s"] = LiveSession(SimpleNamespace(platform=None, user_id="human"), "route")
        db.admissions.admit(epoch=epoch, principal_id="human", session_id="s", request_id="r", payload={"text": "go"})
        row = admission_row(db.admissions.claim(epoch=epoch, session_id="s"))
        authority._stamp_claim(SimpleNamespace(session_id="s"), live, row)
        frames = []
        live.event_stream.observers.add(frames.append)

        args = {"path": "x"}
        start = _wire({"type": "tool.start", **tool_frame("call_0", "read_file", args), **NAMED})
        done = _wire({"type": "tool.complete", **tool_frame("call_0", "read_file", args, '{"content": "hi"}'), **NAMED})
        legacy = _wire({"type": "tool.start", **tool_frame("call_1", "read_file", args)})
        assert publish_worker_tool_event(authority, "s", row["generation"], start)
        assert publish_worker_tool_event(authority, "s", row["generation"], done)
        assert publish_worker_tool_event(authority, "s", row["generation"], legacy)

        payloads = [frame["params"]["payload"] for frame in frames]
        assert [{key: payload[key] for key in NAMED if key in payload} for payload in payloads] == [NAMED, NAMED, {}]
        # A malformed identity is dropped, never the frame: refusing it would lose a turn whose tools already ran.
        for bad in ({"row_id": "12"}, {"row_id": True}, {"row_id": 0}, {"message_uid": "uid-12"},
                    {"row_id": 12, "message_uid": ""}, {"row_id": 12, "tool_call_uid": 5},
                    {"row_id": 12, "message_uid": "u" * 129}):
            frames.clear()
            assert publish_worker_tool_event(
                authority, "s", row["generation"], {"type": "tool.start", **tool_frame("call_0", "read_file", args), **bad})
            [payload] = [frame["params"]["payload"] for frame in frames]
            assert not set(NAMED) & set(payload) and payload["tool_call_id"] == "call_0"
        # Anything else unknown still is not a tool frame.
        assert not publish_worker_tool_event(
            authority, "s", row["generation"], {"type": "tool.start", **tool_frame("call_0", "read_file", args), "extra": 1})
    finally:
        db.close()


def test_a_managed_worker_names_rows_only_to_an_owner_that_asked():
    """An older owner refuses a tool or result frame with fields it does not know, so the worker sends
    them only when the owner's bootstrap asks (``live_rows_v1``, set by session_managed_worker._bootstrap)."""
    asked = {"policy": {"request_json": '{"live_rows_v1": true, "turn_v1": {}}'}}
    assert live_rows_asked(asked)
    assert not live_rows_asked({"policy": {"request_json": '{"turn_v1": {}}'}})
    assert not live_rows_asked({"policy": {"request_json": '{"live_rows_v1": "yes"}'}})
    assert not live_rows_asked({"policy": {}})
