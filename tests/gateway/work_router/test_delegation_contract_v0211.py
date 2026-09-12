"""Real SQLite/service contracts; no profile spawning or network."""
from dataclasses import replace
import json
import sqlite3

import pytest

from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.models import CanonicalEvent, RouteAction, RouteDecision
from gateway.work_router.service import WorkRouter


@pytest.fixture
def router(tmp_path):
    config = RouterConfig.from_mapping({
        "enabled": True, "channel_allowlist": ["CTEST"],
        "db_path": str(tmp_path / "router.db"),
        "bot_registry": {name: f"UBOT{i}" for i, name in enumerate(DEFAULT_PROFILES)},
    }).require_ready()
    instance = WorkRouter(config)
    yield instance
    instance.store.close()


def event(targets=("UBOT1", "UBOT2")):
    return CanonicalEvent(
        event_id="EvDelegate", channel_id="CTEST", thread_ts="1700000000.1",
        text=" ".join(f"<@{uid}>" for uid in targets) + " 검토해줘",
        author_kind="bot", author_profile="Demian", author_user_id="UBOT0",
        mentioned_user_ids=targets,
    )


class RecordingSender:
    def __init__(self, statuses=None, fail=None):
        self.calls = []
        self.executions = []
        self.statuses = statuses or {}
        self.fail = fail

    async def __call__(self, **kw):
        self.calls.append(kw)
        action = kw["decision"].action
        if action.kind == "dispatch":
            self.executions.append(action.target_profile)
            if action.target_profile == self.fail:
                raise TimeoutError("unknown external result")
        return {"success": True, "message_id": f"1700000000.{len(self.calls)+1}",
                "execution_status": self.statuses.get(action.target_profile, "completed"),
                "result_text": f"{action.target_profile} recorded result"}


async def enqueue(router, e):
    router.ack_gate.complete(e.event_id, success=True)
    assert await router.enqueue_after_ack(e)


@pytest.mark.asyncio
@pytest.mark.parametrize("targets", [("UBOT1",), ("UBOT1", "UBOT2")])
async def test_pm_explicit_delegation_executes_and_summary_once_after_reopen(router, targets):
    e = event(targets)
    await enqueue(router, e)
    sender = RecordingSender()
    result = await router.process_once(e.event_id, sender=sender)
    assert result.status == "delegation_completed", result
    assert sender.executions == list(DEFAULT_PROFILES[1:1+len(targets)])
    assert sender.calls[-1]["decision"].action.kind == "summary"
    assert all(c["event"] == e for c in sender.calls)
    assert router.store.get_thread(e.channel_id, e.thread_ts).owner == "Demian"
    assert router.store.get_handoff_count(e.channel_id, e.thread_ts) == 1
    other = WorkRouter(router.config)
    try:
        assert (await other.process_once(e.event_id, sender=sender)).status == "duplicate_or_not_pending"
        assert len(sender.calls) == len(targets) + 1
    finally:
        other.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state,outcome", [("failed", "failed"), ("waiting", "waiting"),
                                           ("resume_required", "resume_required"),
                                           ("resumed", "resumed"), ("completed", "completed")])
async def test_execution_status_is_not_transport_success(router, state, outcome):
    e = event(("UBOT1",))
    await enqueue(router, e)
    sender = RecordingSender({"Hans": state})
    result = await router.process_once(e.event_id, sender=sender)
    assert result.status == f"delegation_{outcome}"
    assert f"[{state}]" in sender.calls[-1]["decision"].action.content
    if state != "completed":
        assert "재개 조건:" in sender.calls[-1]["decision"].action.content
    with sqlite3.connect(router.config.db_path) as db:
        assert db.execute("SELECT outcome FROM router_completions WHERE event_id=?", (e.event_id,)).fetchone()[0] == result.status


@pytest.mark.asyncio
async def test_failed_child_does_not_short_circuit_siblings_and_is_visible(router):
    e = event()
    await enqueue(router, e)
    sender = RecordingSender(fail="Hans")
    result = await router.process_once(e.event_id, sender=sender)
    assert result.status == "delegation_quarantined", result
    assert sender.executions == ["Hans", "Wendy"]
    summary = sender.calls[-1]["decision"].action.content
    assert "Hans: [quarantined]" in summary and "Wendy: [completed]" in summary
    await router.process_once(e.event_id, sender=sender)
    assert len(sender.calls) == 3


@pytest.mark.asyncio
async def test_handoff_limit_posts_actual_noncompletion_summary(router):
    e = event()
    await enqueue(router, e)
    for _ in range(5):
        router.store.increment_handoff_count(e.channel_id, e.thread_ts)
    sender = RecordingSender()
    result = await router.process_once(e.event_id, sender=sender)
    assert result.status == "handoff_limited"
    assert sender.executions == []
    assert len(sender.calls) == 1
    assert "[handoff_limited]" in sender.calls[0]["decision"].action.content
    assert router.store.outbox_status(result.operation_id) == "confirmed"
    await router.process_once(e.event_id, sender=sender)
    assert len(sender.calls) == 1


@pytest.mark.asyncio
async def test_crash_recovery_reuses_confirmed_child_and_quarantines_unknown(router):
    e = event()
    await enqueue(router, e)
    claim = router.store.claim_inbox_with_lease(e.event_id, "crashed")
    state = router.store.get_thread(e.channel_id, e.thread_ts)
    fence = dict(lease_owner="crashed", lease_generation=claim.generation)
    for target in ("Hans", "Wendy"):
        child = RouteDecision(RouteAction("dispatch", target, response_kind="team"), state)
        op = router.store.prepare_outbox(replace(e, event_id=f"{e.event_id}:staff:{target}"), child, **fence)
        router.store.mark_outbox_attempting(op, "before_crash", **fence)
        if target == "Hans":
            router.store.record_delegation_result(op, json.dumps({"profile": target, "status": "completed", "text": "durable result"}), **fence)
            router.store.finalize_confirmed(op, target_profile=target, message_id="old", next_state=None, **fence)
    router.store.release_lease(e.channel_id, e.thread_ts, "crashed", claim.generation)
    sender = RecordingSender()
    result = await router.process_once(e.event_id, sender=sender)
    assert result.status == "delegation_quarantined", result
    assert sender.executions == []
    assert "durable result" in sender.calls[-1]["decision"].action.content
    assert "Wendy: [quarantined]" in sender.calls[-1]["decision"].action.content


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"author_user_id": "USPOOF"}, {"author_user_id": None},
    {"mentioned_user_ids": (), "text": "Hans Wendy 검토해줘"},
    {"mentioned_user_ids": ("UBOT1", "UBOT1")},
    {"mentioned_user_ids": ("UNREGISTERED",)},
])
async def test_untrusted_or_implicit_pm_delegation_never_executes(router, change):
    e = replace(event(), **change)
    await enqueue(router, e)
    sender = RecordingSender()
    result = await router.process_once(e.event_id, sender=sender)
    assert result.status == "silence"
    assert not sender.executions


@pytest.mark.asyncio
async def test_summary_delivery_failure_consumes_budget_and_does_not_reexecute(router):
    e = event()
    await enqueue(router, e)
    sender = RecordingSender()

    async def fail_summary(**kw):
        result = await sender(**kw)
        if kw["decision"].action.kind == "summary":
            return {"success": False}
        return result

    result = await router.process_once(e.event_id, sender=fail_summary)
    assert result.status == "quarantined"
    assert router.store.get_handoff_count(e.channel_id, e.thread_ts) == 1
    assert router.store.inbox_status(e.event_id) == "quarantined"
    await router.process_once(e.event_id, sender=fail_summary)
    assert len(sender.calls) == 3


@pytest.mark.asyncio
async def test_reserved_fifth_hop_recovers_without_sixth_relay(router):
    e = event(("UBOT1",))
    await enqueue(router, e)
    claim = router.store.claim_inbox_with_lease(e.event_id, "crashed")
    state = router.store.get_thread(e.channel_id, e.thread_ts)
    from gateway.work_router.rules import route
    decision = route(state, e, router.registry)
    fence = dict(lease_owner="crashed", lease_generation=claim.generation)
    for _ in range(4):
        router.store.increment_handoff_count(e.channel_id, e.thread_ts)
    op = router.store.prepare_outbox(e, decision, **fence)
    router.store.reserve_delegation_handoff(op, **fence)
    router.store.reserve_delegation_handoff(op, **fence)
    router.store.release_lease(e.channel_id, e.thread_ts, "crashed", claim.generation)
    sender = RecordingSender()
    result = await router.process_once(e.event_id, sender=sender)
    assert result.status == "delegation_completed", result
    assert sender.executions == ["Hans"]
    assert router.store.get_handoff_count(e.channel_id, e.thread_ts) == 5
    next_event = replace(e, event_id="EvSixth")
    await enqueue(router, next_event)
    result = await router.process_once(next_event.event_id, sender=sender)
    assert result.status == "handoff_limited"
    assert sender.executions == ["Hans"]
