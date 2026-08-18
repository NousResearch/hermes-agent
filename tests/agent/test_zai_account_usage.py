from __future__ import annotations

from datetime import datetime, timezone

import httpx

from agent.account_usage import (
    _zai_reset_ms,
    _zai_window_label,
    fetch_account_usage,
)


def test_zai_reset_ms_parses_epoch_milliseconds():
    assert _zai_reset_ms(1781719502287) == datetime.fromtimestamp(
        1781719502287 / 1000.0, tz=timezone.utc
    )


def test_zai_reset_ms_falls_back_to_parse_dt():
    assert _zai_reset_ms(None) is None


def test_zai_window_labels_cover_credit_limits():
    # Live Coding Plan payload (verified 2026-08-19) returns CREDIT_LIMIT
    # entries: unit 3 / number 5 (rolling 5h) and unit 6 / number 1 (weekly).
    assert _zai_window_label({"type": "CREDIT_LIMIT", "unit": 3, "number": 5}) == "5h credits"
    assert _zai_window_label({"type": "CREDIT_LIMIT", "unit": 6, "number": 1}) == "Weekly credits"


def test_zai_window_labels_cover_tokens_limits():
    # Other observed shapes: TOKENS_LIMIT variants and TIME_LIMIT for tools.
    assert _zai_window_label({"type": "TOKENS_LIMIT", "unit": 3, "number": 5}) == "5h credits"
    assert _zai_window_label({"type": "TIME_LIMIT", "unit": 3, "number": 5}) == "MCP/tools"
    assert _zai_window_label({"type": "OTHER_LIMIT", "unit": 3, "number": 5}) == "Other Limit"
    assert _zai_window_label({}) == "Quota"


def test_zai_account_usage_snapshot_from_live_payload(monkeypatch):
    """End-to-end parse of the payload shape returned by the live endpoint."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = str(request.headers.get("Authorization", ""))
        return httpx.Response(
            200,
            json={
                "success": True,
                "msg": "Operation successful",
                "data": {
                    "level": "pro",
                    "limits": [
                        {
                            "type": "CREDIT_LIMIT",
                            "unit": 3,
                            "number": 5,
                            "abilityType": 1,
                            "funcScope": "ALL",
                            "codingPlanId": "coding-pro",
                            "percentage": 2,
                            "remaining": 11665,
                            "currentValue": 334,
                            "nextResetTime": 1787108192489,
                        },
                        {
                            "type": "CREDIT_LIMIT",
                            "unit": 6,
                            "number": 1,
                            "abilityType": 1,
                            "funcScope": "ALL",
                            "codingPlanId": "coding-pro",
                            "percentage": 1,
                            "remaining": 59665,
                            "currentValue": 334,
                            "nextResetBackground": "false",
                            "nextResetTime": 1787693832994,
                        },
                    ],
                },
            },
        )

    original_client = httpx.Client

    class ClientShim:
        def __init__(self, *a, **k):
            self._real = original_client(transport=httpx.MockTransport(handler), **k)

        def __enter__(self):
            self._real.__enter__()
            return self._real

        def __exit__(self, *exc):
            return self._real.__exit__(*exc)

    monkeypatch.setattr(httpx, "Client", ClientShim)

    snapshot = fetch_account_usage(
        "zai", base_url="https://api.z.ai/api/coding/paas/v4", api_key="zai-test-key"
    )
    assert snapshot is not None
    assert snapshot.provider == "zai"
    assert snapshot.plan == "Pro"
    labels = [w.label for w in snapshot.windows]
    assert labels == ["5h credits", "Weekly credits"]
    assert snapshot.windows[0].used_percent == 2.0
    assert snapshot.windows[0].reset_at == datetime.fromtimestamp(
        1787108192489 / 1000.0, tz=timezone.utc
    )
    assert "remaining 11665 / used 334" == snapshot.windows[0].detail
    assert any("z.ai/manage-apikey/subscription" in d for d in snapshot.details)
    assert seen["url"] == "https://api.z.ai/api/monitor/usage/quota/limit"
    assert seen["auth"] == "Bearer zai-test-key"


def test_zai_account_usage_unavailable_payload(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": False, "msg": "no subscription"})

    original_client = httpx.Client

    class ClientShim:
        def __init__(self, *a, **k):
            self._real = original_client(transport=httpx.MockTransport(handler), **k)

        def __enter__(self):
            self._real.__enter__()
            return self._real

        def __exit__(self, *exc):
            return self._real.__exit__(*exc)

    monkeypatch.setattr(httpx, "Client", ClientShim)

    snapshot = fetch_account_usage(
        "zai-coding",
        base_url="https://api.z.ai/api/coding/paas/v4",
        api_key="zai-test-key",
    )
    assert snapshot is not None
    assert snapshot.unavailable_reason == "no subscription"
    assert not snapshot.windows


def test_zai_account_usage_no_key_returns_none(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda **_: {"api_key": "", "base_url": None},
    )
    assert fetch_account_usage(
        "zai", base_url="https://api.z.ai/api/coding/paas/v4"
    ) is None


def test_zai_account_usage_ignores_unconfirmed_routes(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda **_: (_ for _ in ()).throw(AssertionError("route must be rejected before credential resolution")),
    )

    for base_url in (
        "https://api.z.ai/api/paas/v4",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        "https://lookalike.example/api/coding/paas/v4",
    ):
        assert fetch_account_usage("zai", base_url=base_url, api_key="unused") is None
