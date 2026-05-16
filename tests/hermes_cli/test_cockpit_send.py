"""Tests for BIF-513 Cockpit send foundation phase 2."""

from __future__ import annotations

import json

from hermes_cli.cockpit_send import (
    CockpitSendConfig,
    CockpitSendLane,
    CockpitSendService,
    CockpitSendStore,
    FakeGatewayMutationAdapter,
)

RAW_VALUES = {
    "C1234567890",
    "U1234567890",
    "T1234567890",
    "discord:C1234567890:U1234567890:T1234567890",
    "sk-testSecret123",
    "token=raw-secret-token",
}


class NoPlatformDispatchAdapter(FakeGatewayMutationAdapter):
    def __init__(self):
        super().__init__()
        self.forbidden_calls: list[str] = []

    def send(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.forbidden_calls.append("send")
        raise AssertionError("platform/gateway send must not be called")

    def send_message(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.forbidden_calls.append("send_message")
        raise AssertionError("platform/gateway send_message must not be called")

    def dispatch(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.forbidden_calls.append("dispatch")
        raise AssertionError("platform/gateway dispatch must not be called")

    def resolve_delivery_target(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.forbidden_calls.append("resolve_delivery_target")
        raise AssertionError("delivery target resolution must not be called")


def _lane() -> CockpitSendLane:
    return CockpitSendLane(
        alias="%ops/current",
        label="Ops C1234567890 for U1234567890",
        platform_label="discord",
        canonical_target={
            "platform": "discord",
            "chat_id": "C1234567890",
            "user_id": "U1234567890",
            "thread_id": "T1234567890",
            "canonical_id": "discord:C1234567890:U1234567890:T1234567890",
        },
    )


def _assert_display_safe(value):
    encoded = json.dumps(value, sort_keys=True)
    for raw in RAW_VALUES:
        assert raw not in encoded
    forbidden_keys = {"canonical_target", "chat_id", "user_id", "thread_id", "canonical_id"}

    def walk(nested):
        if isinstance(nested, dict):
            assert not (set(nested) & forbidden_keys)
            for item in nested.values():
                walk(item)
        elif isinstance(nested, list):
            for item in nested:
                walk(item)

    walk(value)
    assert "idempotency-key-1" not in encoded


def test_send_foundation_is_default_off_and_does_not_record_or_mutate():
    store = CockpitSendStore()
    adapter = NoPlatformDispatchAdapter()
    service = CockpitSendService(store=store, gateway_adapter=adapter)

    result = service.submit_send_intent(
        actor_user_id="marco",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="hello C1234567890",
    )

    assert result == {"ok": False, "status": "feature_disabled", "error_code": "feature_disabled", "record": None}
    assert store.records_by_id == {}
    assert store.audit_events == []
    assert adapter.mutations == []
    assert adapter.forbidden_calls == []


def test_kill_switch_blocks_even_when_fake_gateway_flag_is_enabled():
    store = CockpitSendStore()
    adapter = NoPlatformDispatchAdapter()
    service = CockpitSendService(
        config=CockpitSendConfig(send_enabled=True, fake_gateway_enabled=True, kill_switch=True),
        store=store,
        gateway_adapter=adapter,
    )

    result = service.submit_send_intent(
        actor_user_id="marco",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="hello",
    )

    assert result["ok"] is False
    assert result["status"] == "kill_switch_active"
    assert store.records_by_id == {}
    assert adapter.mutations == []
    assert adapter.forbidden_calls == []


def test_same_key_same_normalized_payload_returns_existing_record_and_fake_mutates_once():
    store = CockpitSendStore()
    adapter = NoPlatformDispatchAdapter()
    service = CockpitSendService(
        config=CockpitSendConfig(send_enabled=True, fake_gateway_enabled=True, kill_switch=False),
        store=store,
        gateway_adapter=adapter,
    )

    first = service.submit_send_intent(
        actor_user_id="marco",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="  hello C1234567890  \r\nsecond line  ",
    )
    replay = service.submit_send_intent(
        actor_user_id="marco",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="hello C1234567890\nsecond line",
    )

    assert first["ok"] is True
    assert replay["ok"] is True
    assert replay["idempotent_replay"] is True
    assert replay["record"]["send_record_id"] == first["record"]["send_record_id"]
    assert replay["record"]["dispatch_count"] == 1
    assert len(adapter.mutations) == 1
    assert len(store.records_by_id) == 1
    assert [event.event_type for event in store.audit_events] == ["cockpit_send_requested", "cockpit_send_dispatched"]
    assert adapter.forbidden_calls == []
    _assert_display_safe(first)
    _assert_display_safe(replay)


def test_same_key_different_payload_returns_conflict_and_no_second_mutation():
    store = CockpitSendStore()
    adapter = NoPlatformDispatchAdapter()
    service = CockpitSendService(
        config=CockpitSendConfig(send_enabled=True, fake_gateway_enabled=True, kill_switch=False),
        store=store,
        gateway_adapter=adapter,
    )

    first = service.submit_send_intent(
        actor_user_id="marco",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="first payload",
    )
    conflict = service.submit_send_intent(
        actor_user_id="marco",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="different payload",
    )

    assert first["ok"] is True
    assert conflict["ok"] is False
    assert conflict["status"] == "idempotency_conflict"
    assert conflict["error_code"] == "idempotency_key_conflict"
    assert conflict["record"]["send_record_id"] == first["record"]["send_record_id"]
    assert conflict["record"]["dispatch_count"] == 1
    assert len(adapter.mutations) == 1
    assert len(store.records_by_id) == 1
    assert len(store.audit_events) == 2
    assert adapter.forbidden_calls == []


def test_audit_events_redact_raw_ids_secrets_and_message_body():
    store = CockpitSendStore()
    adapter = NoPlatformDispatchAdapter()
    service = CockpitSendService(
        config=CockpitSendConfig(send_enabled=True, fake_gateway_enabled=True, kill_switch=False),
        store=store,
        gateway_adapter=adapter,
    )

    result = service.submit_send_intent(
        actor_user_id="marco",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="DM U1234567890 in C1234567890 using token=raw-secret-token and sk-testSecret123",
    )

    assert result["ok"] is True
    audit_payloads = [event.to_display_dict() for event in store.audit_events]
    assert len(audit_payloads) == 2
    for audit in audit_payloads:
        assert audit["idempotency_key_hash"] != "idempotency-key-1"
        assert audit["content_preview_redacted"] == "DM [id] in [id] using [redacted] and [redacted]"
    _assert_display_safe(audit_payloads)
    _assert_display_safe(adapter.mutations)
    assert adapter.forbidden_calls == []


def test_platform_actor_and_canonical_ids_are_not_exposed_in_display_audit_or_fake_adapter():
    store = CockpitSendStore()
    adapter = NoPlatformDispatchAdapter()
    service = CockpitSendService(
        config=CockpitSendConfig(send_enabled=True, fake_gateway_enabled=True, kill_switch=False),
        store=store,
        gateway_adapter=adapter,
    )

    result = service.submit_send_intent(
        actor_user_id="U1234567890",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="safe operational update",
    )

    assert result["ok"] is True
    audit_payloads = [event.to_display_dict() for event in store.audit_events]
    assert len(audit_payloads) == 2
    for audit in audit_payloads:
        assert "actor_user_id" not in audit
        assert audit["actor_user_id_hash"].startswith("sha256:")
        assert audit["actor_user_id_hash"] != "U1234567890"
        assert audit["idempotency_key_hash"] != "idempotency-key-1"

    _assert_display_safe(result)
    _assert_display_safe(audit_payloads)
    _assert_display_safe(adapter.mutations)
    assert adapter.forbidden_calls == []


def test_arbitrary_canonical_values_are_redacted_from_serialized_send_payloads():
    store = CockpitSendStore()
    adapter = NoPlatformDispatchAdapter()
    service = CockpitSendService(
        config=CockpitSendConfig(send_enabled=True, fake_gateway_enabled=True, kill_switch=False),
        store=store,
        gateway_adapter=adapter,
    )
    raw_values = {
        "U-ACTOR-RAW-513",
        "U-APPROVER-RAW-513",
        "C-PLATFORM-RAW-513",
        "THREAD-RAW-513",
        "CANONICAL-RAW-513",
        "idempotency-key-raw-513",
    }
    lane = CockpitSendLane(
        alias="%ops/arbitrary",
        label="Ops lane for U-ACTOR-RAW-513 on C-PLATFORM-RAW-513 via CANONICAL-RAW-513",
        platform_label="discord",
        canonical_target={
            "platform_chat_id": "C-PLATFORM-RAW-513",
            "actor_user_id": "U-ACTOR-RAW-513",
            "approver_user_id": "U-APPROVER-RAW-513",
            "thread_id": "THREAD-RAW-513",
            "canonical_id": "CANONICAL-RAW-513",
        },
    )

    result = service.submit_send_intent(
        actor_user_id="U-ACTOR-RAW-513",
        lane=lane,
        idempotency_key="idempotency-key-raw-513",
        message_text="Send U-ACTOR-RAW-513 to C-PLATFORM-RAW-513 via CANONICAL-RAW-513",
    )

    assert result["ok"] is True
    audit_payloads = [event.to_display_dict() for event in store.audit_events]
    serialized_payloads = json.dumps(
        {"result": result, "audit": audit_payloads, "mutations": adapter.mutations},
        sort_keys=True,
    )
    for raw in raw_values:
        assert raw not in serialized_payloads
    assert "canonical_target" not in serialized_payloads
    assert "platform_chat_id" not in serialized_payloads
    assert "canonical_id" not in serialized_payloads
    assert result["record"]["lane_label"] == "Ops lane for [id] on [id] via [id]"
    assert result["record"]["content_preview_redacted"] == "Send [id] to [id] via [id]"
    assert adapter.mutations[0]["lane_label"] == "Ops lane for [id] on [id] via [id]"
    assert adapter.forbidden_calls == []


def test_missing_fake_gateway_flag_blocks_without_mutation():
    store = CockpitSendStore()
    adapter = NoPlatformDispatchAdapter()
    service = CockpitSendService(
        config=CockpitSendConfig(send_enabled=True, fake_gateway_enabled=False, kill_switch=False),
        store=store,
        gateway_adapter=adapter,
    )

    result = service.submit_send_intent(
        actor_user_id="marco",
        lane=_lane(),
        idempotency_key="idempotency-key-1",
        message_text="hello",
    )

    assert result["ok"] is False
    assert result["status"] == "feature_disabled"
    assert store.records_by_id == {}
    assert adapter.mutations == []
    assert adapter.forbidden_calls == []
