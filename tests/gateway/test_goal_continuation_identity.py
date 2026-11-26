"""A queued /goal continuation belongs to the goal that judged it, and always has a drainer.

A continuation the judge queued for goal A must never start once A is gone: cleared, replaced by
goal B (with or without a /goal clear first), or set again with the same text, even when the row
is already committed to the session authority's durable FIFO. It names A by its persistent
``GoalState.created_at`` (as a heartbeat admission names its ``HeartbeatState``), so a new goal
with the same text is not mistaken for it, and a criterion added to A (/subgoal) keeps it.

A continuation drained from the adapter slot, and the one /goal resume queues, join the FIFO as the
same identified wake, so they are still checked when they wait behind a follow-up; one whose wake
is refused is not run as a plain row.

A follow-up sent while a turn runs (``dispatch_shared_busy``) is answered by its observer, which
delivers the reply but never drains the adapter's pending slot: a continuation parked there after
that turn would wait for an unrelated message (#145865).

The judge runs for seconds: a /goal pause or clear sent meanwhile is not written back over, and a
/subgoal added meanwhile is kept.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent

MARKER = "[Continuing toward your standing goal]"


async def _telegram_gateway(tmp_path, monkeypatch, instance_id=None):
    import gateway.run as gr
    from gateway.config import GatewayConfig
    from gateway.platforms.base import SendResult
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_context import register_transport_home
    from hermes_cli import goals
    from persistence_assembly import open_persistence
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    authority = None  # without one: the classic in-process path
    if instance_id is not None:
        authority = await initialize_session_authority(runner, profile_id="default", instance_id=instance_id)
    db = open_persistence(runner.session_store.persistence.scope)
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", typing_indicator=False))
    adapter.gateway_runner, runner.adapters = runner, {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append((chat_id, content))
        return SendResult(success=True, message_id=str(len(sent)))

    monkeypatch.setattr(adapter, "send", send)
    return runner, authority, adapter, db, sent


def _reply(text):
    return {"final_response": text, "messages": [], "tools": [], "history_offset": 0, "last_prompt_tokens": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["clear_then_set", "replace", "same_text_again", "storage_reset",
                                    "storage_reset_before_admission", "subgoal_added"])
async def test_durable_continuation_never_starts_for_a_goal_that_is_gone(tmp_path, monkeypatch, change):
    """Goal A's continuation is committed to the durable FIFO (a delegation result started the judged
    turn); before it runs, the real /goal handlers change the goal. Only a continuation of the goal
    still standing reaches the agent."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-identity")
    calls = []

    async def respond(**kw):
        calls.append(kw["message"])
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "one read remains", False, None, False)])
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "finished", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    real_judge = runner._post_turn_goal_continuation
    armed, judged = [], []

    async def command(text):
        handler = runner._handle_subgoal_command if text.startswith("/subgoal") else runner._handle_goal_command
        return await handler(MessageEvent(text=text, source=source, message_id=f"cmd-{text}"))

    async def judge_then_change(**kw):
        await real_judge(**kw)
        if not armed or judged:
            return
        judged.append(True)
        # The continuation is queued (wherever this build queues it) and has not started.
        if change == "storage_reset_before_admission":
            return
        if change == "storage_reset":  # a policy reset rotates the route without /new
            runner.session_store.reset_session(route)
            return
        for text in {"clear_then_set": ["/goal clear", "/goal ship release B"],
                     "replace": ["/goal ship release B"],
                     "same_text_again": ["/goal clear", "/goal ship release A"],
                     "subgoal_added": ["/subgoal also tag the release"]}[change]:
            await command(text)

    monkeypatch.setattr(runner, "_post_turn_goal_continuation", judge_then_change)
    if change == "storage_reset_before_admission":
        from gateway import session_automation
        real_wake = session_automation.admit_route_wake

        async def reset_then_wake(*args, **kwargs):
            runner.session_store.reset_session(route)  # between the judgment and the commit
            return await real_wake(*args, **kwargs)

        monkeypatch.setattr(session_automation, "admit_route_wake", reset_then_wake)
    try:
        await adapter.handle_message(MessageEvent(text="seed the route", source=source, message_id="1"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks, what="human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        goals.GoalManager(sid).set("ship release A")
        live = authority.sessions[sid]
        armed.append(True)  # the next judgment is goal A's
        assert await runner._inject_watch_notification("[IMPORTANT: background result arrived]", {
            "type": "async_delegation", "delegation_id": "deleg_identity", "status": "completed",
            "parent_session_id": sid, "session_key": route, "platform": "telegram",
            "chat_type": "dm", "chat_id": "42", "user_id": "42", "output": "read-only result",
        }) is True
        await live.task
        # A later message drains whatever is still parked for this chat.
        await adapter.handle_message(MessageEvent(text="next question", source=source, message_id="2"))
        await async_wait_until(lambda: "next question" in calls and not adapter._background_tasks
                               and (live.task is None or live.task.done()), what="the later message ran")

        assert judged, "the judged turn never ran"
        rows = authority.persistence.admissions.list(session_id=authority.logical_owner(sid), pending_only=False)
        assert all(row.status == "terminal" for row in rows), [(r.request_id[:40], r.status) for r in rows]
        continued = [c for c in calls if c.startswith(MARKER)]
        if change == "subgoal_added":
            assert len(continued) == 1 and "ship release A" in continued[0], calls
        else:
            assert continued == [], f"a continuation of the goal that is gone reached the agent: {calls}"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_real_new_still_cancels_a_queued_continuation(tmp_path, monkeypatch):
    """The user's /new, through the adapter, after the continuation was committed: it never runs,
    and the fresh session's next message is answered normally."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-new")
    calls = []

    async def respond(**kw):
        calls.append((kw["session_id"], kw["message"]))
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("continue", "one read remains", False, None, False))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    real_judge = runner._post_turn_goal_continuation
    armed, reset = [], []

    async def judge_then_new(**kw):
        await real_judge(**kw)
        if armed and not reset:
            reset.append(True)
            await adapter.handle_message(MessageEvent(text="/new", source=source, message_id="new"))
            await async_wait_until(lambda: runner.session_store.peek_session_id(route) != sid,
                                   what="the route was reset")

    monkeypatch.setattr(runner, "_post_turn_goal_continuation", judge_then_new)
    try:
        await adapter.handle_message(MessageEvent(text="seed the route", source=source, message_id="1"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks, what="human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        goals.GoalManager(sid).set("ship release A")
        live = authority.sessions[sid]
        armed.append(True)
        assert await runner._inject_watch_notification("[IMPORTANT: background result arrived]", {
            "type": "async_delegation", "delegation_id": "deleg_new", "status": "completed",
            "parent_session_id": sid, "session_key": route, "platform": "telegram",
            "chat_type": "dm", "chat_id": "42", "user_id": "42", "output": "read-only result",
        }) is True
        await live.task
        await async_wait_until(lambda: not adapter._background_tasks, what="the /new reply was sent")
        fresh = runner.session_store.peek_session_id(route)
        await adapter.handle_message(MessageEvent(text="fresh question", source=source, message_id="2"))
        await async_wait_until(lambda: ("fresh question" in [m for _, m in calls]) and not adapter._background_tasks,
                               what="the fresh session answered")

        assert reset and fresh != sid
        assert [m for _, m in calls if m.startswith(MARKER)] == [], calls
        assert (fresh, "fresh question") in calls
    finally:
        db.close()


@pytest.mark.asyncio
async def test_busy_follow_up_turn_continues_the_goal_after_the_first_task_ended(tmp_path, monkeypatch):
    """#145865's busy path: a second message, sent while the first turn runs, goes through
    ``dispatch_shared_busy``. Its turn ends after the first message's adapter task is gone, the judge
    says continue: a third turn must run without another message."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, _authority, adapter, db, sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-busy")
    calls, gates = [], {"first": asyncio.Event(), "second": asyncio.Event()}

    async def respond(**kw):
        message = kw["message"]
        calls.append(message)
        if message in gates:
            await gates[message].wait()
        return _reply("continued" if message.startswith(MARKER) else f"answered {message}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "one step remains", False, None, False), ("done", "all done", False, None, False)])
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    try:
        await adapter.handle_message(MessageEvent(text="first", source=source, message_id="1"))
        await async_wait_until(lambda: calls == ["first"], what="the first turn started")
        await adapter.handle_message(MessageEvent(text="second", source=source, message_id="2"))
        gates["first"].set()
        await async_wait_until(lambda: calls == ["first", "second"] and route not in adapter._active_sessions,
                               what="the second turn started after the first task ended")
        sid = runner.session_store.peek_session_id(route)
        goals.GoalManager(sid).set("finish the report")
        gates["second"].set()
        await async_wait_until(lambda: len(calls) == 3 and ("42", "continued") in sent,
                               what="the continuation ran and was answered without another message")
        await async_wait_until(lambda: goals.GoalManager(sid).state.status == "done", what="the goal was judged done")

        assert calls[2].startswith(MARKER) and "finish the report" in calls[2], calls
        assert len(calls) == 3
        assert route not in adapter._pending_messages
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_authority", [True, False])
async def test_adapter_slot_continuation_never_runs_for_a_replaced_goal(tmp_path, monkeypatch, with_authority):
    """A human turn's continuation waits in the adapter slot (its task drains it after the reply).
    A goal set meanwhile (as /goal <text> does) does not clear that slot: the continuation of the
    replaced goal must be dropped when it drains, and the new goal's next turn still judged."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, _authority, adapter, db, _sent = await _telegram_gateway(
        tmp_path, monkeypatch, "goal-slot" if with_authority else None)
    calls = []

    async def respond(**kw):
        calls.append(kw["message"])
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "more work remains", False, None, False)])
    judged = []

    def judge(*a, **k):
        judged.append(True)
        return next(verdicts, ("done", "finished", False, None, False))

    monkeypatch.setattr(goals, "judge_goal", judge)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    session_a = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(session_a).set("ship release A")
    real_enqueue, enqueued = runner._enqueue_fifo, []

    def enqueue_then_replace(session_key, queued_event, fifo_adapter):
        real_enqueue(session_key, queued_event, fifo_adapter)
        enqueued.append(queued_event.text)
        goals.GoalManager(session_a).set("ship release B")

    monkeypatch.setattr(runner, "_enqueue_fifo", enqueue_then_replace)
    try:
        await adapter.handle_message(MessageEvent(text="start on the release", source=source, message_id="1"))
        await async_wait_until(lambda: enqueued and route not in adapter._pending_messages
                               and not adapter._background_tasks, what="the continuation slot drained")
        await adapter.handle_message(MessageEvent(text="fresh question", source=source, message_id="2"))
        await async_wait_until(lambda: "fresh question" in calls and not adapter._background_tasks and len(judged) == 2,
                               what="the new goal's turn ran and was judged")

        assert enqueued and enqueued[0].startswith(MARKER) and "ship release A" in enqueued[0]
        assert [c for c in calls if c.startswith(MARKER)] == [], calls
        assert goals.GoalManager(session_a).state.goal == "ship release B"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_new_goal_judged_while_the_old_continuation_waits_still_continues(tmp_path, monkeypatch):
    """Goal A's continuation is still queued when goal B is set and B's turn is judged: A's waiting
    continuation does not stand in for B's (it is dropped when it would start), so B continues."""
    from gateway import session_automation
    from hermes_cli import goals

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-dedupe")
    calls = []

    async def respond(**kw):
        calls.append(kw["message"])
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "A needs a read", False, None, False), ("continue", "B needs a read", False, None, False)])
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "finished", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    real_wake, wakes = session_automation.admit_route_wake, []

    async def wake_then_new_goal(*args, **kwargs):
        accepted = await real_wake(*args, **kwargs)
        wakes.append(args[3])
        if len(wakes) == 1:
            await runner._handle_goal_command(MessageEvent(text="/goal ship release B", source=source, message_id="g"))
            entry = await runner.async_session_store.get_or_create_session(source, touch_activity=False)
            await runner._post_turn_goal_continuation(session_entry=entry, source=source, final_response="answered",
                                                      authority_owned_wake=True)
        return accepted

    monkeypatch.setattr(session_automation, "admit_route_wake", wake_then_new_goal)
    try:
        from tests.waiting import async_wait_until
        await adapter.handle_message(MessageEvent(text="seed the route", source=source, message_id="1"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks, what="human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        goal_a = goals.GoalManager(sid).set("ship release A").created_at
        live = authority.sessions[sid]
        assert await runner._inject_watch_notification("[IMPORTANT: background result arrived]", {
            "type": "async_delegation", "delegation_id": "deleg_dedupe", "status": "completed",
            "parent_session_id": sid, "session_key": route, "platform": "telegram",
            "chat_type": "dm", "chat_id": "42", "user_id": "42", "output": "read-only result",
        }) is True
        await live.task

        assert len(wakes) == 2 and "ship release A" in wakes[0] and "ship release B" in wakes[1], wakes
        assert [c.split("\n")[1] for c in calls if c.startswith(MARKER)] == ["Goal: ship release B"], calls
        # Each goal's identity is in its stored row: read back and restored by the native restore a
        # restarted gateway uses, the rows still name goal A and goal B.
        from gateway.run_busy import GatewayBusySessionMixin
        from gateway.session_envelope import restore_native
        goal_b = goals.GoalManager(sid).state.created_at
        stored = [row for row in db.admissions.list(session_id=authority.logical_owner(sid), pending_only=False)
                  if "goal-continuation" in row.request_id]
        restored = [restore_native(row.payload, runner) for row in stored]
        assert all(event.internal for event in restored)
        assert [GatewayBusySessionMixin._goal_continuation_generation(e) for e in restored] == [goal_a, goal_b]
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["cleared_during_follow_up", "judged_during_follow_up",
                                  "handoff_in_cleanup", "handoff_in_flight", "handoff_then_continue"])
async def test_drained_continuation_behind_a_follow_up_is_still_the_goals(tmp_path, monkeypatch, case):
    """A message sent during a goal turn is queued ahead of that turn's continuation, which waits in the
    adapter slot until the turn's task drains it into the FIFO. While the follow-up runs, a /goal clear
    must still stop the continuation, and the follow-up's own judgment must not queue a second one."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-follow-up")
    assert authority is not None
    calls, gates = [], {"work": asyncio.Event(), "meanwhile": asyncio.Event()}
    conversion_started, release_conversion, followup_judged = (asyncio.Event() for _ in range(3))
    handoff = case.startswith("handoff_")
    if handoff:
        admit, post_turn = runner._admit_goal_continuation, runner._post_turn_goal_continuation
        stop_typing = adapter._stop_typing_refresh

        async def hold_first_conversion(*args, **kwargs):
            if not conversion_started.is_set():
                conversion_started.set()
                await release_conversion.wait()
            return await admit(*args, **kwargs)

        async def signal_followup_judged(**kwargs):
            await post_turn(**kwargs)
            if kwargs["final_response"] == "answered 2":
                followup_judged.set()

        async def hold_adapter_cleanup(*args, **kwargs):
            event = kwargs.get("event")
            if (getattr(event, "message_id", None) == "1" and route not in adapter._pending_messages
                    and not conversion_started.is_set()):
                conversion_started.set()
                await release_conversion.wait()
            return await stop_typing(*args, **kwargs)

        if case == "handoff_in_cleanup":
            monkeypatch.setattr(adapter, "_stop_typing_refresh", hold_adapter_cleanup)
        else:
            monkeypatch.setattr(runner, "_admit_goal_continuation", hold_first_conversion)
        monkeypatch.setattr(runner, "_post_turn_goal_continuation", signal_followup_judged)

    async def respond(**kw):
        message = kw["message"]
        calls.append(message)
        if message in gates:
            await gates[message].wait()
        return _reply(f"answered {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "one step remains", False, None, False)] * (3 if case == "handoff_then_continue" else 2))
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "all done", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    sid = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(sid).set("finish the report")
    early_wakes = 0
    try:
        await adapter.handle_message(MessageEvent(text="work", source=source, message_id="1"))
        await async_wait_until(lambda: calls == ["work"], what="the goal turn started")
        await adapter.handle_message(MessageEvent(text="meanwhile", source=source, message_id="2"))
        gates["work"].set()
        await async_wait_until(lambda: calls == ["work", "meanwhile"] and route not in adapter._pending_messages,
                               what="the follow-up runs and the continuation left the slot")
        if case == "cleared_during_follow_up":
            await runner._handle_goal_command(MessageEvent(text="/goal clear", source=source, message_id="c"))
        if handoff:
            await asyncio.wait_for(conversion_started.wait(), 15)
        gates["meanwhile"].set()
        if handoff:
            await asyncio.wait_for(followup_judged.wait(), 15)
            early_wakes = sum("goal-continuation" in row.request_id for row in authority.persistence.admissions.list(
                session_id=authority.logical_owner(sid), pending_only=False,
            ))
            release_conversion.set()
        live = authority.sessions[authority.logical_owner(sid)]
        await async_wait_until(lambda: not adapter._background_tasks and (live.task is None or live.task.done())
                               and all(r.status == "terminal" for r in authority.persistence.admissions.list(
                                   session_id=authority.logical_owner(sid), pending_only=False)),
                               what="the route settled")

        continued = [c for c in calls if c.startswith(MARKER)]
        expected = {"cleared_during_follow_up": 0, "handoff_then_continue": 2}.get(case, 1)
        assert len(continued) == expected, calls
        assert [c for c in calls if not c.startswith(MARKER)] == ["work", "meanwhile"]
        if handoff:
            # Count committed inputs too: a second wake discarded after the first finishes is
            # still a duplicate admission, even though it did not reach the model.
            rows = authority.persistence.admissions.list(
                session_id=authority.logical_owner(sid), pending_only=False,
            )
            assert sum("goal-continuation" in row.request_id for row in rows) == expected
            assert early_wakes == 0
    finally:
        release_conversion.set()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["not_admitted", "raises"])
async def test_drained_continuation_is_not_run_unidentified_when_its_wake_is_refused(tmp_path, monkeypatch, refusal):
    """The slot continuation's wake is refused when its task drains it: it must not fall back to a
    plain row that no goal check would stop. The user is told, and their own messages still run."""
    from gateway import session_automation
    from hermes_cli import goals
    from persistence.core.errors import AuthorityRefusal
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-refused")
    calls, gates = [], {"work": asyncio.Event(), "meanwhile": asyncio.Event()}

    async def respond(**kw):
        calls.append(kw["message"])
        if kw["message"] in gates:
            await gates[kw["message"]].wait()
        return _reply(f"answered {len(calls)}")

    async def refuse_wake(*args, **kwargs):
        if refusal == "raises":
            raise AuthorityRefusal("runtime_draining")
        return False

    monkeypatch.setattr(runner, "_run_agent", respond)
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("continue", "one step remains", False, None, False))
    monkeypatch.setattr(session_automation, "admit_route_wake", refuse_wake)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    sid = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(sid).set("finish the report")
    try:
        await adapter.handle_message(MessageEvent(text="work", source=source, message_id="1"))
        await async_wait_until(lambda: calls == ["work"], what="the goal turn started")
        await adapter.handle_message(MessageEvent(text="meanwhile", source=source, message_id="2"))
        gates["work"].set()
        await async_wait_until(lambda: calls == ["work", "meanwhile"] and route not in adapter._pending_messages,
                               what="the follow-up runs and the continuation left the slot")
        await runner._handle_goal_command(MessageEvent(text="/goal clear", source=source, message_id="c"))
        gates["meanwhile"].set()
        live = authority.sessions[authority.logical_owner(sid)]
        await async_wait_until(lambda: not adapter._background_tasks and (live.task is None or live.task.done())
                               and all(r.status == "terminal" for r in authority.persistence.admissions.list(
                                   session_id=authority.logical_owner(sid), pending_only=False)),
                               what="the route settled")

        assert calls == ["work", "meanwhile"], calls
        assert ("42", "Goal is still active, but Hermes couldn’t queue the next turn.") in sent
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["cleared_while_it_waits", "kept"])
async def test_goal_resume_continuation_is_the_goals_own(tmp_path, monkeypatch, case):
    """``/goal resume``, sent through the adapter while a turn runs and a follow-up is queued, queues the
    goal's continuation behind that follow-up. A /goal clear while it waits stops it; otherwise it runs
    once (the turn judged meanwhile does not queue a second one)."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-resume")
    calls, gates = [], {"work": asyncio.Event(), "meanwhile": asyncio.Event()}

    async def respond(**kw):
        calls.append(kw["message"])
        if kw["message"] in gates:
            await gates[kw["message"]].wait()
        return _reply(f"answered {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "one step remains", False, None, False)] * 2)
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "all done", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    sid = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(sid).set("finish the report")
    goals.GoalManager(sid).pause()
    try:
        await adapter.handle_message(MessageEvent(text="work", source=source, message_id="1"))
        await async_wait_until(lambda: calls == ["work"], what="the turn started")
        await adapter.handle_message(MessageEvent(text="meanwhile", source=source, message_id="2"))
        await adapter.handle_message(MessageEvent(text="/goal resume", source=source, message_id="3"))
        await async_wait_until(lambda: goals.GoalManager(sid).is_active(), what="the goal resumed")
        gates["work"].set()
        await async_wait_until(lambda: calls == ["work", "meanwhile"], what="the follow-up runs")
        if case == "cleared_while_it_waits":
            await runner._handle_goal_command(MessageEvent(text="/goal clear", source=source, message_id="c"))
        gates["meanwhile"].set()
        live = authority.sessions[authority.logical_owner(sid)]
        await async_wait_until(lambda: not adapter._background_tasks and (live.task is None or live.task.done())
                               and all(r.status == "terminal" for r in authority.persistence.admissions.list(
                                   session_id=authority.logical_owner(sid), pending_only=False)),
                               what="the route settled")

        continued = [c for c in calls if c.startswith(MARKER)]
        assert len(continued) == (0 if case == "cleared_while_it_waits" else 1), calls
    finally:
        db.close()


@pytest.mark.asyncio
async def test_goal_resume_during_a_background_result_turn_still_continues(tmp_path, monkeypatch):
    """``/goal resume`` sent while a delegation result's turn runs: no adapter task will drain that
    turn's slot, so the resume continuation must not wait there. It runs once after the turn (the
    turn's own judgment sees it waiting and queues no second one)."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-resume-bg")
    calls, gate = [], asyncio.Event()

    async def respond(**kw):
        calls.append(kw["message"])
        if "background result" in kw["message"]:
            await gate.wait()
        return _reply(f"answered {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "one step remains", False, None, False)])
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "all done", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    try:
        await adapter.handle_message(MessageEvent(text="seed the route", source=source, message_id="1"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks, what="human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        goals.GoalManager(sid).set("finish the report")
        goals.GoalManager(sid).pause()
        live = authority.sessions[sid]
        assert await runner._inject_watch_notification("[IMPORTANT: background result arrived]", {
            "type": "async_delegation", "delegation_id": "deleg_resume", "status": "completed",
            "parent_session_id": sid, "session_key": route, "platform": "telegram",
            "chat_type": "dm", "chat_id": "42", "user_id": "42", "output": "read-only result",
        }) is True
        await async_wait_until(lambda: len(calls) == 2, what="the background-result turn started")
        await adapter.handle_message(MessageEvent(text="/goal resume", source=source, message_id="2"))
        await async_wait_until(lambda: goals.GoalManager(sid).is_active(), what="the goal resumed")
        gate.set()
        await async_wait_until(lambda: any(c.startswith(MARKER) for c in calls) and not adapter._background_tasks
                               and (live.task is None or live.task.done()),
                               what="the resume continuation ran without another message")

        assert len([c for c in calls if c.startswith(MARKER)]) == 1, calls
        assert route not in adapter._pending_messages
    finally:
        db.close()


@pytest.mark.asyncio
async def test_goal_resume_whose_wake_is_refused_does_not_run_unidentified(tmp_path, monkeypatch):
    """No ``goal-continuation`` wake can be admitted for ``/goal resume``: its continuation is not run
    as a plain row (which no later pause or clear would stop); the user is told, the goal stays."""
    from gateway import session_automation
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, _authority, adapter, db, sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-resume-refused")
    calls = []

    async def respond(**kw):
        calls.append(kw["message"])
        return _reply(f"answered {len(calls)}")

    async def refuse_wake(*args, **kwargs):
        return False

    monkeypatch.setattr(runner, "_run_agent", respond)
    monkeypatch.setattr(session_automation, "admit_route_wake", refuse_wake)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    sid = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(sid).set("finish the report")
    goals.GoalManager(sid).pause()
    try:
        await adapter.handle_message(MessageEvent(text="/goal resume", source=source, message_id="1"))
        await async_wait_until(lambda: not adapter._background_tasks
                               and ("42", "Goal is still active, but Hermes couldn’t queue the next turn.") in sent,
                               what="the resume was answered")

        assert [c for c in calls if c.startswith(MARKER)] == [], calls
        assert goals.GoalManager(sid).is_active()
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/goal clear", "/goal pause", "/subgoal also tag the release"])
async def test_a_goal_changed_while_its_judge_runs_stays_changed(tmp_path, monkeypatch, command):
    """The judge takes seconds. A /goal clear or pause sent through the adapter meanwhile is persisted
    before the judge returns ``continue``: the judge's result must not write the goal back as active,
    and no continuation may start. A /subgoal added meanwhile is kept, and the continuation carries it."""
    import threading

    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-judging")
    calls, judging, release = [], threading.Event(), threading.Event()

    async def respond(**kw):
        calls.append(kw["message"])
        return _reply(f"answered {len(calls)}")

    def slow_judge(*a, **k):
        if judging.is_set():
            return ("done", "all done", False, None, False)
        judging.set()
        assert release.wait(30), "the test never released the judge"
        return ("continue", "one step remains", False, None, False)

    monkeypatch.setattr(runner, "_run_agent", respond)
    monkeypatch.setattr(goals, "judge_goal", slow_judge)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    sid = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(sid).set("finish the report")
    try:
        await adapter.handle_message(MessageEvent(text="work", source=source, message_id="1"))
        await async_wait_until(judging.is_set, what="the judge is running")
        await adapter.handle_message(MessageEvent(text=command, source=source, message_id="2"))
        expected = {"/goal clear": "cleared", "/goal pause": "paused"}.get(command, "active")
        await async_wait_until(lambda: goals.load_goal(sid).status == expected
                               and (expected != "active" or goals.load_goal(sid).subgoals),
                               what="the command was persisted")
        release.set()
        live = authority.sessions[authority.logical_owner(sid)]
        await async_wait_until(lambda: not adapter._background_tasks and (live.task is None or live.task.done())
                               and all(r.status == "terminal" for r in authority.persistence.admissions.list(
                                   session_id=authority.logical_owner(sid), pending_only=False)),
                               what="the route settled")

        continued = [c for c in calls if c.startswith(MARKER)]
        if expected == "active":
            assert goals.load_goal(sid).subgoals == ["also tag the release"]
            assert len(continued) == 1 and "also tag the release" in continued[0], calls
        else:
            assert goals.load_goal(sid).status == expected
            assert continued == [], calls
    finally:
        release.set()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before_its_wake", "after_its_wake"])
async def test_drained_continuation_never_runs_after_its_route_was_reset(tmp_path, monkeypatch, when):
    """A storage-level rotation of the route (not /new, which fences it) lands right before or right
    after the drained continuation is committed: it never reaches the agent, nothing stays queued, and
    the next message is answered on the fresh session."""
    from gateway import session_automation
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-rotated")
    calls = []

    async def respond(**kw):
        calls.append((kw["session_id"], kw["message"]))
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "more work remains", False, None, False)])
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "finished", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    session_a = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(session_a).set("ship release A")
    real_wake = session_automation.admit_route_wake

    async def rotate_around_wake(*args, **kwargs):
        if when == "before_its_wake":
            runner.session_store.reset_session(route)
        accepted = await real_wake(*args, **kwargs)
        if when == "after_its_wake":
            runner.session_store.reset_session(route)
        return accepted

    monkeypatch.setattr(session_automation, "admit_route_wake", rotate_around_wake)
    try:
        await adapter.handle_message(MessageEvent(text="start on the release", source=source, message_id="1"))
        await async_wait_until(lambda: calls and not adapter._background_tasks, what="the first turn settled")
        await adapter.handle_message(MessageEvent(text="fresh question", source=source, message_id="2"))
        await async_wait_until(lambda: any(m == "fresh question" for _, m in calls) and not adapter._background_tasks,
                               what="the fresh session answered")

        fresh = runner.session_store.peek_session_id(route)
        assert fresh != session_a and (fresh, "fresh question") in calls
        assert [m for _, m in calls if m.startswith(MARKER)] == [], calls
        rows = authority.persistence.admissions.list(session_id=authority.logical_owner(session_a), pending_only=False)
        assert all(row.status == "terminal" for row in rows), [(r.request_id[:40], r.status) for r in rows]
    finally:
        db.close()


def test_waiting_continuation_counts_only_for_the_judged_goal():
    """A slot continuation of an earlier goal is dropped when it would start: it must not stop the
    new goal's judgment from queuing its own."""
    from types import SimpleNamespace

    from gateway.run_busy import GatewayBusySessionMixin

    key = "agent:main:telegram:dm:42"
    slot = {key: SimpleNamespace(text=f"{MARKER}\nGoal: ship", internal=False,
                                 metadata={"gateway_session_id": "s1", "goal_created_at": 100.5})}
    stub = SimpleNamespace(
        _session_key_for_source=lambda source: key,
        _delivery_adapter_for=lambda source: SimpleNamespace(_pending_messages=slot),
        _overflow_queue=lambda session_key: [],
        _is_goal_continuation_event=GatewayBusySessionMixin._is_goal_continuation_event,
        _goal_continuation_generation=GatewayBusySessionMixin._goal_continuation_generation,
    )
    waiting = GatewayBusySessionMixin._goal_continuation_waiting
    assert waiting(stub, object(), "s1", 100.5) is True
    assert waiting(stub, object(), "s1", 200.25) is False


def test_in_band_follow_up_check_names_the_goal(tmp_path, monkeypatch):
    """``_run_agent_queued_followup`` drops a drained continuation through ``_stale_goal_continuation``:
    a continuation of the goal now standing runs, one of a goal set again since does not, and any
    other input is never treated as one."""
    from types import SimpleNamespace

    from gateway.run_busy import GatewayBusySessionMixin
    from hermes_cli import goals

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    goals._DB_CACHE.clear()
    try:
        manager = goals.GoalManager("sess-inband")
        first = manager.set("ship release A").created_at
        runner = SimpleNamespace(
            _is_goal_continuation_event=GatewayBusySessionMixin._is_goal_continuation_event,
            _goal_continuation_generation=GatewayBusySessionMixin._goal_continuation_generation,
            _goal_still_active_for_session=lambda sid, gen: GatewayBusySessionMixin._goal_still_active_for_session(
                runner, sid, gen),
        )
        stale = GatewayBusySessionMixin._stale_goal_continuation

        def queued(text, generation):
            return SimpleNamespace(text=text, internal=False, metadata={"goal_created_at": generation})

        continuation = f"{MARKER}\nGoal: ship release A"
        assert stale(runner, queued(continuation, first), "sess-inband") is False
        manager.clear()
        second = goals.GoalManager("sess-inband").set("ship release A").created_at
        assert second != first
        assert stale(runner, queued(continuation, first), "sess-inband") is True
        assert stale(runner, queued(continuation, second), "sess-inband") is False
        assert stale(runner, queued("a person's /queue item", first), "sess-inband") is False
    finally:
        goals._DB_CACHE.clear()
