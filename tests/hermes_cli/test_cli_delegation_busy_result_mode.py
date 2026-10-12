"""CLI side of delegation.busy_result_mode=steer: only this session's results are offered to the
running agent, and an offer no tool batch carried still arrives as the next turn."""
from __future__ import annotations

import queue

from cli import HermesCLI


def test_offered_result_the_turn_never_carried_becomes_the_next_turn(monkeypatch):
    from tools.process_registry import process_registry
    monkeypatch.setattr(process_registry, "completion_queue", queue.Queue())
    monkeypatch.setattr(process_registry, "_completions_restored", True)
    acks = []
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda evt, consumer: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda evt, claim: acks.append(claim))
    mine = {"type": "async_delegation", "delegation_id": "deleg_x", "session_key": "visible-session",
            "status": "completed", "summary": "reviewer verdict: BLOCKER in step 3", "goal": "review"}
    foreign = {**mine, "delegation_id": "deleg_y", "session_key": "other-session"}
    process_registry.completion_queue.put(mine)
    process_registry.completion_queue.put(foreign)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = None
    cli._pending_input = queue.Queue()
    agent = type("Agent", (), {})()

    assert cli._offer_delegation_results_to_running_turn(agent) == 1
    cli._requeue_undelivered_delegation_results(agent)

    assert process_registry.completion_queue.get_nowait() is foreign
    assert acks == ["claim"]
    queued = cli._pending_input.get_nowait()
    assert "BLOCKER in step 3" in str(queued) and queued.display_kind == "async_delegation_complete"
    assert cli._pending_input.empty()


def test_steered_notice_prints_when_the_result_lands_not_after_the_reply(monkeypatch):
    from agent.background_results import attach_to_tool_tail, offer
    from tools.process_registry_notifications import TimelineNotification
    printed = []
    monkeypatch.setattr("cli._cprint", printed.append)
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    agent = type("Agent", (), {})()
    offer(agent, TimelineNotification("full", "Subagent Task Completed: BALTA11", "async_delegation_complete"), "full")
    attach_to_tool_tail(agent, [{"role": "tool", "content": "out", "tool_call_id": "c1"}])

    cli._announce_delegation_results_carried_so_far(agent)
    assert len(printed) == 1 and "BALTA11" in printed[0]

    cli._requeue_undelivered_delegation_results(agent)
    assert len(printed) == 1 and cli._pending_input.empty()
