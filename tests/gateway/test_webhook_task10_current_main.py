"""Current-main contracts for webhook Task 10 intake closure.

These tests exercise the production helper boundaries that make provider retry
identity, scoped idempotency, cache bounds, and raw template rendering safe.
The HTTP-level cases are intentionally kept in test_webhook_adapter.py; this
module pins the new class-level contract without relying on staged patch files.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import (
    IdempotencyResult,
    WebhookAdapter,
    _IDEMPOTENCY_DEFAULT_MAX_ENTRIES,
    _IDEMPOTENCY_MAX_ENTRIES_LIMIT,
    _RAW_PAYLOAD_DEFAULT_CAP_BYTES,
    _RAW_PAYLOAD_MAX_CAP_BYTES,
    _RAW_PAYLOAD_MIN_CAP_BYTES,
)


def _adapter(max_entries: object = 8) -> WebhookAdapter:
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "routes": {},
                "rate_limit": 30,
                "idempotency_max_entries": max_entries,
            },
        )
    )


class TestIdempotencyContract:
    def test_scope_contains_profile_route_provider_and_id(self) -> None:
        adapter = _adapter()
        for profile, route, provider in (
            ("alpha", "one", "github"),
            ("beta", "one", "github"),
            ("alpha", "two", "github"),
            ("alpha", "one", "gitlab"),
        ):
            assert (
                adapter._record_delivery_id(
                    "same-id",
                    1000.0,
                    "same-body",
                    profile=profile,
                    route=route,
                    provider=provider,
                )
                is IdempotencyResult.ACCEPTED
            )
        assert len(adapter._seen_deliveries) == 4

    def test_duplicate_and_conflict_are_distinct(self) -> None:
        adapter = _adapter()
        kwargs = {"profile": "default", "route": "r", "provider": "github"}
        assert adapter._record_delivery_id("d", 1.0, "a", **kwargs) is IdempotencyResult.ACCEPTED
        assert adapter._record_delivery_id("d", 2.0, "a", **kwargs) is IdempotencyResult.DUPLICATE
        assert adapter._record_delivery_id("d", 3.0, "b", **kwargs) is IdempotencyResult.CONFLICT

    def test_same_key_concurrency_has_one_winner(self) -> None:
        adapter = _adapter()

        def record(_: int) -> IdempotencyResult:
            return adapter._record_delivery_id(
                "d",
                1000.0,
                "body",
                profile="default",
                route="r",
                provider="github",
            )

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(record, range(32)))
        assert results.count(IdempotencyResult.ACCEPTED) == 1
        assert results.count(IdempotencyResult.DUPLICATE) == 31

    def test_hard_ceiling_holds_after_every_insert(self) -> None:
        adapter = _adapter(4)
        for i in range(30):
            assert (
                adapter._record_delivery_id(
                    str(i),
                    float(i),
                    f"body-{i}",
                    profile="default",
                    route="r",
                    provider="github",
                )
                is IdempotencyResult.ACCEPTED
            )
            assert len(adapter._seen_deliveries) <= 4
            assert len(adapter._seen_delivery_bodies) <= 4
        assert {key[-1] for key in adapter._seen_deliveries} == {"26", "27", "28", "29"}

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            (0, 1),
            (-1, 1),
            ("bad", _IDEMPOTENCY_DEFAULT_MAX_ENTRIES),
            (True, _IDEMPOTENCY_DEFAULT_MAX_ENTRIES),
            (False, _IDEMPOTENCY_DEFAULT_MAX_ENTRIES),
            (float("inf"), _IDEMPOTENCY_DEFAULT_MAX_ENTRIES),
            (_IDEMPOTENCY_MAX_ENTRIES_LIMIT + 1, _IDEMPOTENCY_MAX_ENTRIES_LIMIT),
        ],
    )
    def test_ceiling_normalization(self, configured: object, expected: int) -> None:
        assert _adapter(configured)._idempotency_max_entries == expected


class TestRawEnvelopeContract:
    @pytest.mark.parametrize("cap", [64, 128, 4000, 8192])
    def test_envelope_is_valid_json_and_never_exceeds_utf8_cap(self, cap: int) -> None:
        adapter = _adapter()
        payload = {"text": "🦊" * 5000, "quote": '"\\' * 200}
        rendered = adapter._render_raw_payload(payload, cap)
        encoded = rendered.encode("utf-8")
        assert len(encoded) <= cap
        envelope = json.loads(rendered)
        assert set(envelope) == {"payload", "truncated", "original_bytes"}
        assert isinstance(envelope["payload"], str)
        assert envelope["original_bytes"] > 0

    def test_default_raw_token_uses_complete_envelope(self) -> None:
        adapter = _adapter()
        rendered = adapter._render_prompt("raw={__raw__}", {"a": "b"}, "push", "r")
        envelope = json.loads(rendered.split("raw=", 1)[1])
        assert envelope["truncated"] is False
        assert '"a": "b"' in envelope["payload"]
        assert len(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= _RAW_PAYLOAD_DEFAULT_CAP_BYTES

    def test_explicit_raw_cap_is_supported(self) -> None:
        adapter = _adapter()
        rendered = adapter._render_prompt("{__raw__:128}", {"x": "z" * 5000}, "push", "r")
        assert len(rendered.encode("utf-8")) <= 128
        assert json.loads(rendered)["truncated"] is True

    @pytest.mark.parametrize(
        "template",
        [
            "{__raw__:63}",
            f"{{__raw__:{_RAW_PAYLOAD_MAX_CAP_BYTES + 1}}}",
            "{__raw__:banana}",
            "{__raw__:}",
        ],
    )
    def test_invalid_raw_caps_fail_closed(self, template: str) -> None:
        with pytest.raises(ValueError):
            _adapter()._render_prompt(template, {"x": 1}, "push", "r")

    def test_raw_payload_does_not_trigger_second_template_pass(self) -> None:
        adapter = _adapter()
        rendered = adapter._render_prompt("{__raw__}", {"x": "{event_type}"}, "push", "r")
        assert "{event_type}" in json.loads(rendered)["payload"]

    def test_constants_are_ordered(self) -> None:
        assert _RAW_PAYLOAD_MIN_CAP_BYTES < _RAW_PAYLOAD_DEFAULT_CAP_BYTES < _RAW_PAYLOAD_MAX_CAP_BYTES
