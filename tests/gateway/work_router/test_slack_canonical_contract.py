from dataclasses import replace

import pytest

from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.rules import DirectiveRejection, ParsedWorkDirective, parse_work_directive
from gateway.work_router.service import WorkRouter


@pytest.fixture
def router(tmp_path):
    config = RouterConfig.from_mapping({
        "enabled": True,
        "channel_allowlist": ["CEXEC"],
        "db_path": str(tmp_path / "router.db"),
        "bot_registry": {name: f"UBOT{i}" for i, name in enumerate(DEFAULT_PROFILES)},
        "work_execution": {
            "enabled": True, "channel_allowlist": ["CEXEC"],
            "slack_team_id": "TTEST", "max_body_bytes": 1024,
        },
    }).require_ready()
    instance = WorkRouter(config)
    yield instance
    instance.store.close()


def test_slack_normalization_preserves_directive_rejection_evidence(router):
    event = {
        "channel": "CEXEC", "ts": "1700000000.1", "user": "UBOT0",
        "bot_id": "BDEMIAN", "text": "[WORK] <@UBOT1> :: Review\nEvidence",
    }
    envelope = {"event_id": "EvDirective", "team_id": "TTEST"}
    canonical = router.canonicalize_slack_event(event, envelope)
    assert isinstance(parse_work_directive(canonical, router.config), ParsedWorkDirective)
    for key, value in (
        ("edited", {"user": "UBOT0", "ts": "1700000000.2"}),
        ("deleted", True), ("file_only", True), ("forwarded", True),
    ):
        rejected = router.canonicalize_slack_event({**event, key: value}, envelope)
        assert parse_work_directive(rejected, router.config) == DirectiveRejection(
            "unsupported_event"
        ), key
        # Persist only the rejection signal, not an extra raw metadata payload.
        assert rejected.metadata[key] is True
    assert parse_work_directive(
        replace(canonical, metadata={"slack_team_id": "TOTHER"}), router.config
    ) == DirectiveRejection("workspace_mismatch")


@pytest.mark.asyncio
async def test_real_router_ack_receipt_duplicate_and_storage_failure(router, monkeypatch):
    event = router.canonicalize_slack_event(
        {"channel": "CEXEC", "ts": "1700000000.1", "text": "한스야 확인해"},
        {"event_id": "EvReceipt", "team_id": "TTEST"},
    )
    router.ack_gate.complete(event.event_id, success=True)
    assert await router.enqueue_after_ack(event)
    assert router.store.get_event(event.event_id) == event
    assert not await router.enqueue_after_ack(event)
    assert router.store.inbox_count() == 1

    failed = replace(event, event_id="EvStorageFailure")
    router.ack_gate.complete(failed.event_id, success=True)

    def fail_commit(*args, **kwargs):
        raise OSError("injected storage failure")

    monkeypatch.setattr(router.store, "enqueue_after_ack", fail_commit)
    with pytest.raises(OSError, match="injected storage failure"):
        await router.enqueue_after_ack(failed)
    assert router.store.get_event(failed.event_id) is None
