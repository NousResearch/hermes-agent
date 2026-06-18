from __future__ import annotations

from datetime import datetime, timezone

import httpx

from agent.account_usage import (
    _zai_reset_ms,
    _zai_window_label,
    fetch_account_usage,
    render_account_usage_lines,
)


def test_zai_reset_ms_parses_epoch_milliseconds():
    assert _zai_reset_ms(1781719502287) == datetime.fromtimestamp(
        1781719502287 / 1000.0, tz=timezone.utc
    )


def test_zai_window_labels_known_limit_shapes():
    assert _zai_window_label({"type": "TOKENS_LIMIT", "unit": 3, "number": 5}) == "5h tokens"
    assert _zai_window_label({"type": "TOKENS_LIMIT", "unit": 6, "number": 1}) == "Weekly tokens"
    assert _zai_window_label({"type": "TIME_LIMIT"}) == "MCP/tools"


def test_zai_fetch_renders_real_quota_api(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda **_: {"api_key": "zai-test-key", "base_url": "https://api.z.ai/api/coding/paas/v4"},
    )

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "code": 200,
                "msg": "Operation successful",
                "success": True,
                "data": {
                    "level": "pro",
                    "limits": [
                        {
                            "type": "TOKENS_LIMIT",
                            "unit": 3,
                            "number": 5,
                            "percentage": 2,
                            "nextResetTime": 1781719502287,
                        },
                        {
                            "type": "TOKENS_LIMIT",
                            "unit": 6,
                            "number": 1,
                            "percentage": 1,
                            "nextResetTime": 1782306187986,
                        },
                        {
                            "type": "TIME_LIMIT",
                            "unit": 5,
                            "number": 1,
                            "currentValue": 0,
                            "remaining": 1000,
                            "percentage": 0,
                            "nextResetTime": 1784293387993,
                            "usageDetails": [
                                {"modelCode": "search-prime", "usage": 0},
                                {"modelCode": "web-reader", "usage": 0},
                            ],
                        },
                    ],
                },
            },
        )

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *args, **kwargs: original_client(transport=httpx.MockTransport(handler)),
    )

    snapshot = fetch_account_usage("zai")
    assert snapshot is not None
    assert snapshot.provider == "zai"
    assert snapshot.source == "quota_api"
    assert snapshot.plan == "Pro"
    assert [w.label for w in snapshot.windows] == ["5h tokens", "Weekly tokens", "MCP/tools"]
    assert snapshot.windows[0].used_percent == 2.0
    assert "search-prime: 0" in snapshot.details
    assert "web-reader: 0" in snapshot.details
    assert seen == {
        "url": "https://api.z.ai/api/monitor/usage/quota/limit",
        "auth": "Bearer zai-test-key",
    }

    lines = render_account_usage_lines(snapshot)
    assert lines[0] == "📈 Z.AI (GLM) quotas"
    assert "Provider: zai (Pro)" in lines
    assert any("5h tokens: 98% remaining (2% used)" in line for line in lines)
    assert any("Usage dashboard: https://z.ai/manage-apikey/subscription" == line for line in lines)


def test_zai_fetch_none_without_key(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda **_: {"api_key": "", "base_url": "https://api.z.ai/api/coding/paas/v4"},
    )
    assert fetch_account_usage("glm") is None
