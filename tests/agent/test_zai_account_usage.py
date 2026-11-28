from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from agent.account_usage import (
    _zai_reset_ms,
    _zai_window_label,
    fetch_account_usage,
)


def test_zai_reset_ms_parses_epoch_milliseconds():
    assert _zai_reset_ms(1781719502287) == datetime.fromtimestamp(
        1781719502287 / 1000.0, tz=UTC
    )


def test_zai_reset_ms_falls_back_to_parse_dt():
    assert _zai_reset_ms(None) is None


@pytest.mark.parametrize("kind", ["CREDIT_LIMIT", "TOKENS_LIMIT"])
def test_zai_window_labels_cover_model_limits(kind):
    # Pro credit and Lite token quotas share durations, not billing units.
    assert _zai_window_label({"type": kind, "unit": 3, "number": 5}) == "5-hour window"
    assert _zai_window_label({"type": kind, "unit": 6, "number": 1}) == "Weekly window"
    assert _zai_window_label({"type": kind, "unit": 5, "number": 1}) == kind.replace("_", " ").title()


def test_zai_window_labels_only_name_known_tool_duration():
    assert _zai_window_label({"type": "TIME_LIMIT", "unit": 5, "number": 1}) == "Monthly MCP/tools"
    assert _zai_window_label({"type": "TIME_LIMIT", "unit": 3, "number": 5}) == "MCP/tools"
    assert _zai_window_label({"type": "TIME_LIMIT", "unit": 5, "number": 2}) == "MCP/tools"
    assert _zai_window_label({"type": "TIME_LIMIT", "unit": 5.5, "number": 1}) == "MCP/tools"
    assert _zai_window_label({"type": "TIME_LIMIT", "unit": 5, "number": 1.5}) == "MCP/tools"
    assert _zai_window_label({"type": "TIME_LIMIT", "unit": None}) == "MCP/tools"
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
    assert labels == ["5-hour window", "Weekly window"]
    assert snapshot.windows[0].used_percent == 2.0
    assert snapshot.windows[0].reset_at == datetime.fromtimestamp(
        1787108192489 / 1000.0, tz=UTC
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


@pytest.mark.parametrize("kind", ["CREDIT_LIMIT", "TOKENS_LIMIT"])
@pytest.mark.parametrize("model_percent", [12, 100])
def test_zai_tool_quota_does_not_limit_model_login(monkeypatch, tmp_path, kind, model_percent):
    """Field regression: exhausted MCP calls must not bench an otherwise usable login."""
    import json

    from agent import account_usage_cache as cache
    from agent.credential_pool import load_pool
    from hermes_cli.inventory import _apply_usage

    base_url = "https://api.z.ai/api/coding/paas/v4"
    auth_path = tmp_path / "auth.json"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    auth_path.write_text(json.dumps({"version": 1, "credential_pool": {"zai": [
        {"id": f"c{i}", "label": f"Login {i}", "source": "manual", "auth_type": "api_key",
         "access_token": f"test-key-{i}", "base_url": base_url} for i in (1, 2)
    ]}}), encoding="utf-8")
    payload = {"success": True, "data": {"level": "lite", "limits": [
        {"type": kind, "unit": 3, "number": 5, "percentage": model_percent,
         "nextResetTime": "2099-01-01T00:00:00Z"},
        {"type": kind, "unit": 6, "number": 1, "percentage": 20},
        {"type": "TIME_LIMIT", "unit": 5, "number": 1, "percentage": 100,
         "nextResetTime": "2099-02-01T00:00:00Z"},
    ]}}
    original_client = httpx.Client
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    monkeypatch.setattr(httpx, "Client", lambda **kw: original_client(transport=transport, **kw))
    # Exercise real resolution, caching and inventory; suppress only background I/O.
    monkeypatch.setattr(cache, "refresh_account_usage_async", lambda *_: None)
    monkeypatch.setattr(cache, "refresh_account_usage_entries_async", lambda *_: None)
    for entry in load_pool("zai").entries():
        snapshot = fetch_account_usage("zai", base_url=base_url, api_key=entry.access_token,
                                       identity_id=cache._identity_id_for("zai", entry))
        assert snapshot is not None
    before = auth_path.read_bytes()
    row = {"slug": "zai", "authenticated": True, "models": ["glm-test"]}
    _apply_usage([row])

    accounts = row["usage"]["accounts"]
    assert len(accounts) == 2
    for account in accounts:
        assert account["state"] == ("limited" if model_percent == 100 else "ready")
        assert account["resets_at"] == ("2099-01-01T00:00:00+00:00" if model_percent == 100 else None)
        assert [w["scope"] for w in account["windows"][:2]] == ["account", "account"]
        tool = account["windows"][2]
        assert tool["scope"] == "tool"
        assert tool["used_percent"] == 100.0
    assert auth_path.read_bytes() == before, "usage telemetry must not mutate authentication"


@pytest.mark.parametrize("percentage", [None, 100, 125])
def test_zai_percentage_only_never_invents_counters(monkeypatch, percentage):
    from agent.account_usage import render_account_usage_lines

    payload = {"success": True, "data": {"limits": [
        {"type": kind, "percentage": percentage, "usageDetails": [
            {"modelCode": "missing-counter"}, {"modelCode": "measured-zero", "usage": 0},
        ]} for kind in ("CREDIT_LIMIT", "TOKENS_LIMIT", "TIME_LIMIT")
    ]}}
    original_client = httpx.Client
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    monkeypatch.setattr(httpx, "Client", lambda **kw: original_client(transport=transport, **kw))
    snapshot = fetch_account_usage("zai", base_url="https://api.z.ai/api/coding/paas/v4", api_key="test-key")
    assert snapshot is not None
    for window in snapshot.windows:
        assert window.used_percent == (None if percentage is None else 100.0)
        assert window.detail is None
    assert not any("missing-counter" in detail for detail in snapshot.details)
    assert "measured-zero: 0" in snapshot.details
    lines = render_account_usage_lines(snapshot)
    for window in snapshot.windows:
        expected = "unavailable" if percentage is None else "0% remaining (100% used)"
        assert f"{window.label}: {expected}" in lines
