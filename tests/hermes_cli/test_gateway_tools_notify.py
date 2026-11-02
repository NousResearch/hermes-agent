"""A notification turn the gateway admits while the classic CLI is idle reaches main's screen.

On main the CLI drained ``tools.process_registry`` itself and ran the ``[IMPORTANT: Background
process ...]`` turn. Over the session authority the process (and its completion) live in the
gateway, which admits and runs that turn on its own; the CLI viewer must paint main's
``◈ Background Process Finished: ...`` header and stream the turn's reply through the same callbacks.
"""
import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import Mock

from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore
from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with, frame
from tests.waiting import wait_until

DISPLAY = "Background Process Finished: sleep 6; echo BGDONE"
MODEL_TEXT = "[IMPORTANT: Background process proc_1 completed (exit 0).\nOutput:\nBGDONE]"


def _gateway_frames():
    """What the authority publishes for one admitted local automation turn (admission adm-9)."""
    return [
        # _admitted -> _publish_pending: the queued row (its public input text) before any claim.
        {"jsonrpc": "2.0", "method": "event", "params": {
            "type": "session.info", "session_id": SID, "payload": {
                "running": False, "pending": [{"admission_id": "adm-9", "status": "queued", "input_id": "x", "text": MODEL_TEXT}]}}},
        frame("message.start", {"display_kind": "process_complete", "display_text": DISPLAY, "text": MODEL_TEXT},
              generation=9, admission="adm-9"),
        frame("message.delta", {"text": "STEP13 notified "}, generation=9, admission="adm-9"),
        frame("message.delta", {"text": "reply."}, generation=9, admission="adm-9"),
        frame("message.complete", {"text": "STEP13 notified reply.", "status": "complete"},
              generation=9, admission="adm-9"),
    ]


def _connected():
    client = FakeClient([])
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    conn.state(SID)
    return conn, client


def _play(conn, client, frames):
    async def put():
        for item in frames:
            await client.events.put(item)
    asyncio.run_coroutine_threadsafe(put(), conn.loop).result(5)
    wait_until(lambda: conn.events.qsize() >= len(frames), timeout=5)


def _idle_cli(conn):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = SID
    cli._pending_input = queue.Queue()
    cli._session_db = GatewaySessionStore(connection=conn)
    cli._agent_running = False
    return cli


def test_idle_cli_paints_header_and_streams_the_gateway_admitted_turn(monkeypatch, capsys):
    conn, client = _connected()
    try:
        # The CLI process never owns the registry over the gateway: draining it is a bug.
        monkeypatch.setattr("tools.process_registry.process_registry", SimpleNamespace(
            drain_notifications=Mock(side_effect=AssertionError("CLI drained the process registry"))))
        cli = _idle_cli(conn)
        _play(conn, client, _gateway_frames())
        cli._drain_process_notifications("cli-idle")
        queued = cli._pending_input.get_nowait()
        assert cli._pending_input.empty()
        assert queued.display_kind == "process_complete" and queued.display_text == DISPLAY
        assert str(queued) == MODEL_TEXT

        # Main's input path renders the header exactly as for a local completion.
        for attr, value in {"_pending_resume_sessions": [], "_typed_voice_stop": lambda t: False,
                            "handle_bang_shell": lambda t: False, "_turn_summary_begin": lambda: None,
                            "_tui_after_turn": lambda: None, "_app": SimpleNamespace(invalidate=lambda: None),
                            "chat": Mock()}.items():
            setattr(cli, attr, value)
        cli._tui_process_one_input(queued)
        assert f"◈ {DISPLAY}" in capsys.readouterr().out
        message = cli.chat.call_args.args[0]
        assert message is queued

        # chat(): stage the row, then run_conversation pumps the ALREADY-admitted turn.
        deltas = []
        agent = GatewayAgent(connection=conn, session_id=SID, stream_delta_callback=deltas.append)
        cli.conversation_history = []
        cli._chat_stage_user_message(agent, message)
        result = agent.run_conversation(str(message), conversation_history=[])
        assert "prompt.submit" not in [method for method, _ in client.calls]
        assert deltas == ["STEP13 notified ", "reply."]
        assert result["final_response"] == "STEP13 notified reply."
        assert result["failed"] is False and result["interrupted"] is False
        assert ("prompt.receipt", {"session_id": SID, "admission_id": "adm-9", "include_result": True}) in client.calls
        assert result["messages"][0]["display_kind"] == "process_complete"
    finally:
        conn.close()


def test_adoption_is_one_shot_and_never_hijacks_a_typed_turn():
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        _play(conn, client, _gateway_frames()[:2])
        cli._drain_process_notifications("cli-idle")
        assert not cli._pending_input.empty()
        # The user's own turn runs instead (e.g. chat() bailed before run_conversation): it submits.
        client.script = [frame("message.complete", {"text": "typed", "outcome": "completed"})]
        agent = GatewayAgent(connection=conn, session_id=SID)
        agent._pending_cli_user_message = {"role": "user", "content": "hello"}
        result = agent.run_conversation("hello")
        assert "prompt.submit" in [method for method, _ in client.calls]
        assert result["final_response"] == "typed"
        assert getattr(conn.state(SID), "admitted_turn", None) is None
    finally:
        conn.close()


def test_post_turn_adoption_runs_before_inputs_queued_behind_the_running_turn():
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        cli._pending_input.put("typed while the last turn ran")
        _play(conn, client, _gateway_frames()[:2])
        cli._drain_process_notifications("cli-post-turn")
        first = cli._pending_input.get_nowait()
        assert getattr(first, "display_text", None) == DISPLAY
        assert cli._pending_input.get_nowait() == "typed while the last turn ran"
    finally:
        conn.close()


def test_idle_tick_without_an_admitted_turn_queues_nothing():
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        _play(conn, client, [frame("session.usage", {"usage": {"api_calls": 1}}),
                             frame("message.complete", {"text": "late"}, admission="adm-old")])
        cli._drain_process_notifications("cli-idle")
        assert cli._pending_input.empty()
    finally:
        conn.close()


def test_stamp_claim_publishes_the_card_and_an_ordinary_claim_stays_empty():
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_events import SessionEvents
    published = []
    stream = SessionEvents()
    stream.publish = lambda sid, payload, *, event_type: published.append((event_type, payload))
    authority = SimpleNamespace(epoch=3, executing={}, _publish_pending=lambda ref: None)
    live = SimpleNamespace(event_stream=stream)
    ref = SessionRef("p", SID)
    automation = {"text": MODEL_TEXT, "local_automation_v1": {
        "identity": "i", "owner": SID, "route": "r", "target": SID,
        "display_kind": "process_complete", "display_text": DISPLAY}}
    SessionAuthority._stamp_claim(authority, ref, live, {"generation": 9, "admission_id": "adm-9", "payload": automation})
    SessionAuthority._stamp_claim(authority, ref, live, {"generation": 10, "admission_id": "adm-10",
                                                         "payload": {"text": "typed"}})
    assert published == [("message.start", {"display_kind": "process_complete", "display_text": DISPLAY,
                                            "text": MODEL_TEXT}), ("message.start", {})]


def test_claim_stamp_carries_the_automation_card_on_message_start():
    from gateway.session_automation import admission_display
    local = {"text": MODEL_TEXT, "local_automation_v1": {
        "identity": "i", "owner": SID, "route": "r", "target": SID,
        "display_kind": "process_complete", "display_text": DISPLAY}}
    assert admission_display(local) == {"display_kind": "process_complete", "display_text": DISPLAY,
                                        "text": MODEL_TEXT}
    diagnostic = {"text": "x", "native_text_v1": {"automation": {
        "display_kind": "process_complete", "display_text": DISPLAY, "notification_category": "diagnostic"}}}
    assert admission_display(diagnostic)["notification_category"] == "diagnostic"
    assert admission_display({"text": "typed by the user"}) == {}
    assert admission_display({"text": "x", "local_automation_v1": {"identity": "i"}}) == {}


def _delegation_completion(goal):
    return {"type": "async_delegation", "delegation_id": "deleg_1", "parent_session_id": SID,
            "goals": [goal], "results": [{"task_index": 0, "status": "completed", "summary": "Two nits.",
                                           "api_calls": 3, "duration_seconds": 12}]}


def _claim_card(evt, text):
    """The ``message.start`` card the gateway publishes for this LOCAL completion's admission: the
    producer's metadata, validated as ``snapshot_local_automation`` does, read back at the claim."""
    from gateway.config import Platform
    from gateway.run_notifications import _watch_notification_metadata
    from gateway.session import SessionSource
    from gateway.session_automation import admission_display, automation_display_metadata
    metadata = _watch_notification_metadata(evt, SessionSource(platform=Platform.LOCAL, chat_id=SID))
    descriptor = {"identity": "i", "owner": SID, "route": "r", "target": SID,
                  **automation_display_metadata(metadata)}
    return admission_display({"text": text, "local_automation_v1": descriptor})


def test_idle_cli_shows_a_gateway_review_subagents_completion(capsys):
    """``/review`` runs its reviewer as a gateway-side async delegation (#134889 M4/CLI-6). Its
    completion turn must reach an idle classic CLI as main's ``◈ Subagent Task ...`` header plus the
    streamed reply; a claim without the card was skipped and the result never shown."""
    from tools.process_registry_notifications import format_process_notification
    evt = _delegation_completion("Review the work just discussed")
    text = format_process_notification(evt)
    card = _claim_card(evt, text)
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        _play(conn, client, [frame("message.start", card, generation=9, admission="adm-9"),
                             frame("message.delta", {"text": "Review: two nits."}, generation=9, admission="adm-9"),
                             frame("message.complete", {"text": "Review: two nits.", "status": "complete"},
                                   generation=9, admission="adm-9")])
        cli._drain_process_notifications("cli-idle")
        queued = cli._pending_input.get_nowait()
        assert queued.display_kind == "async_delegation_complete"
        assert queued.display_text == "Subagent Task Completed: Review the work just discussed"
        assert str(queued) == text
        for attr, value in {"_pending_resume_sessions": [], "_typed_voice_stop": lambda t: False,
                            "handle_bang_shell": lambda t: False, "_turn_summary_begin": lambda: None,
                            "_tui_after_turn": lambda: None, "_app": SimpleNamespace(invalidate=lambda: None),
                            "chat": Mock()}.items():
            setattr(cli, attr, value)
        cli._tui_process_one_input(queued)
        assert f"◈ {queued.display_text}" in capsys.readouterr().out
        deltas = []
        agent = GatewayAgent(connection=conn, session_id=SID, stream_delta_callback=deltas.append)
        cli.conversation_history = []
        cli._chat_stage_user_message(agent, queued)
        result = agent.run_conversation(str(queued), conversation_history=[])
        assert "prompt.submit" not in [method for method, _ in client.calls]
        assert deltas == ["Review: two nits."] and result["final_response"] == "Review: two nits."
    finally:
        conn.close()


def test_a_long_subagent_goal_still_admits_its_completion_card():
    """The card's title is bounded by the admission's display-text limit: an over-long goal must not
    turn the completion into an ``invalid_params`` refusal (a delivery that never lands)."""
    card = _claim_card(_delegation_completion("audit " * 1000), "model text")
    assert card["display_kind"] == "async_delegation_complete"
    assert card["display_text"].startswith("Subagent Task Completed: audit")


def test_idle_cli_prints_the_gateway_background_review_summary(capsys):
    """``/refine`` promises "any memory/skill updates will be reported when done" (#134889 M3): the
    gateway publishes the review's summary to the local session as ``review.summary`` after the turn
    settled. An idle classic CLI prints it (main's ``display.review.summary_cli`` line) instead of
    discarding it with the between-turn leftovers."""
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        _play(conn, client, [frame("review.summary", {"text": "💾 Self-improvement review: Memory updated"},
                                   admission=None)])
        cli._drain_process_notifications("cli-idle")
        assert "  💾 Self-improvement review: Memory updated" in capsys.readouterr().out
        assert cli._pending_input.empty()  # a report, not a turn
    finally:
        conn.close()
