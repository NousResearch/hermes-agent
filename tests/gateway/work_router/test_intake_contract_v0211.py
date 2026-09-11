"""Intake acceptance probes using real canonicalization, service and SQLite.

No live profiles/network. These are RED acceptance tests until the native
Demian-producer/Owner execution contract is bound; do not infer live execution
from the recording transport receipt.
"""
from unittest.mock import patch

import pytest

from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.rules import DirectiveRejection, parse_work_directive
from gateway.work_router.service import WorkRouter


@pytest.fixture
def router(tmp_path):
    config = RouterConfig.from_mapping({
        "enabled": True, "channel_allowlist": ["CEXEC"],
        "db_path": str(tmp_path / "intake.db"),
        "bot_registry": {name: f"UBOT{i}" for i, name in enumerate(DEFAULT_PROFILES)},
        "work_execution": {"enabled": True, "channel_allowlist": ["CEXEC"],
                           "slack_team_id": "TTEST", "max_body_bytes": 1024},
    }).require_ready()
    instance = WorkRouter(config)
    yield instance
    instance.store.close()


def native_event(router, *, team="TTEST", human=False):
    message = {"channel": "CEXEC", "ts": "1700000000.2",
               "thread_ts": "1700000000.1", "user": "UBOT0",
               "text": "[WORK] <@UBOT1> :: Review\nEvidence"}
    if not human:
        message.update(bot_id="BDEMIAN", subtype="bot_message")
    return router.canonicalize_slack_event(
        message, {"event_id": "EvIntake", "team_id": team})


async def process(router, event):
    calls = []

    async def sender(**kwargs):
        calls.append(kwargs)
        return {"success": True, "message_id": "1700000000.3"}

    router.ack_gate.complete(event.event_id, success=True)
    assert await router.enqueue_after_ack(event)
    result = await router.process_once(event.event_id, sender=sender)
    return result, calls


def test_native_human_is_not_authorized_demian_directive(router):
    assert parse_work_directive(native_event(router, human=True), router.config) == (
        DirectiveRejection("issuer_not_demian"))


@pytest.mark.asyncio
async def test_intake_consumer_reaches_existing_parser_once(router):
    # create=True exposes the currently missing service import as a RED call-count,
    # rather than replacing the parser with fabricated successful output.
    with patch("gateway.work_router.service.parse_work_directive",
               wraps=parse_work_directive, create=True) as parser:
        await process(router, native_event(router))
    assert parser.call_count == 1


@pytest.mark.asyncio
async def test_valid_native_intake_persists_authoritative_owner(router):
    event = native_event(router)
    await process(router, event)
    reopened = WorkRouter(router.config)
    try:
        execution = reopened.store.get_work_execution(event.event_id)
        assert execution is not None
        assert execution.owner_profile == "Hans"
        assert execution.owner_slack_user_id == "UBOT1"
        assert reopened.store.get_thread(event.channel_id, event.thread_ts).owner == "Hans"
    finally:
        reopened.store.close()


@pytest.mark.asyncio
async def test_invalid_reserved_directive_cannot_fall_through_to_delegation(router):
    event = native_event(router, team="TOTHER")
    assert parse_work_directive(event, router.config) == DirectiveRejection("workspace_mismatch")
    result, calls = await process(router, event)
    assert result.status == "rejected"
    assert result.error == "work_directive_rejected:workspace_mismatch"
    assert router.store.inbox_status(event.event_id) == "quarantined"
    assert router.store.get_work_execution(event.event_id) is None
    assert not calls, "Rejected reserved directive reached an execution/delivery sender"
