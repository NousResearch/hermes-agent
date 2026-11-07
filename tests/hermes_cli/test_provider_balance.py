"""GET /api/model/balance — probe parsing (fixture payloads, no network) + row/route shape."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from hermes_cli import provider_balance

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVIDER_DIR = REPO_ROOT / "plugins" / "model-providers"


def _load_provider(name: str):
    """Import one bundled plugin module by path.

    Loaded directly instead of through ``get_provider_profile`` so the result cannot be
    shadowed by a same-named ``$HERMES_HOME`` copy of the plugin.
    """
    path = PROVIDER_DIR / name / "__init__.py"
    module_name = f"hermes_test_provider_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path, submodule_search_locations=[str(path.parent)])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def deepseek_payload():
    return {
        "is_available": True,
        "balance_infos": [
            {"currency": "USD", "total_balance": "0.80", "granted_balance": "0.00", "topped_up_balance": "0.80"},
            {"currency": "CNY", "total_balance": "5.77", "granted_balance": "0.00", "topped_up_balance": "5.77"},
        ],
    }


@pytest.fixture
def amd_payload():
    return {
        "rpm_limit": 60,
        "daily_cost_limit_usd": 5,
        "daily_cost_used_usd": 0.0351,
        "daily_cost_remaining_usd": 4.9648,
        "daily_reset_at": "2026-11-05T09:36:12.414Z",
        "daily_reset_timezone": None,
        "today": {"requests": 12, "errors": 0, "total_tokens": 4200, "cost": 0.0351},
    }


def _stub_get(monkeypatch, module, payload, status_code: int = 200):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers or {}, "timeout": timeout})
        return _FakeResponse(payload, status_code)

    monkeypatch.setattr(module.httpx, "get", fake_get)
    return calls


def test_deepseek_probe_prefers_cny_and_keeps_key_out_of_the_payload(monkeypatch, deepseek_payload):
    module = _load_provider("deepseek")
    calls = _stub_get(monkeypatch, module, deepseek_payload)

    result = module.balance_probe("sk-test-secret")

    assert calls[0]["url"] == "https://api.deepseek.com/user/balance"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test-secret"
    assert calls[0]["timeout"] == provider_balance.PROBE_TIMEOUT_SECONDS
    assert result["primary"] == {"amount": 5.77, "unit": "CNY", "label": "Total balance"}
    assert result["breakdown"]["topped_up_balance"] == 5.77
    assert "sk-test-secret" not in repr(result)


def test_deepseek_probe_falls_back_to_usd_when_cny_is_absent(monkeypatch, deepseek_payload):
    module = _load_provider("deepseek")
    usd_only = dict(deepseek_payload, balance_infos=deepseek_payload["balance_infos"][:1])
    _stub_get(monkeypatch, module, usd_only)

    result = module.balance_probe("sk-test-secret")

    assert result["primary"] == {"amount": 0.80, "unit": "USD", "label": "Total balance"}


def test_amd_probe_reports_points_quota_and_beijing_reset(monkeypatch, amd_payload):
    module = _load_provider("amd")
    calls = _stub_get(monkeypatch, module, amd_payload)

    result = module.balance_probe("amd-test-secret")

    assert calls[0]["url"] == "https://developer.amd.com.cn/radeon/api/v1/usage"
    # The daily_cost_*_usd fields are points, never money: the unit is pts.
    assert result["primary"] == {"amount": 4.9648, "unit": "pts", "label": "Remaining today"}
    assert result["quota"] == {
        "used": 0.0351,
        "limit": 5,
        "remaining": 4.9648,
        "reset_at": "2026-11-05 17:36 GMT+8",
    }
    assert result["extra"]["rpm_limit"] == 60
    assert result["extra"]["requests_today"] == 12
    assert "amd-test-secret" not in repr(result)


def test_openrouter_probe_subtracts_usage_from_credits(monkeypatch):
    module = _load_provider("openrouter")
    _stub_get(monkeypatch, module, {"data": {"total_credits": "10.00", "total_usage": "2.50"}})

    result = module.balance_probe("or-test-secret")

    assert result["primary"] == {"amount": 7.5, "unit": "USD", "label": "Remaining credits"}
    assert result["breakdown"]["total_credits"] == 10.0
    assert "quota" not in result or not result.get("quota")


def test_probe_registry_only_lists_profiles_that_declare_a_probe():
    registry = provider_balance.probe_registry()

    assert registry["deepseek"].__name__ == "balance_probe"
    # Bundled AMD ships its probe even where a $HERMES_HOME copy shadows the profile.
    assert registry["amd"].__name__ == "balance_probe"
    # A provider with no documented balance endpoint must not appear (→ unsupported).
    assert "anthropic" not in registry


def test_probe_row_statuses_and_redacted_error():
    # Configured, no probe → unsupported (never a guessed URL).
    row = provider_balance._probe_row("anthropic", "Anthropic", True, "sk-x", None)
    assert row["status"] == "unsupported" and row["error"] is None

    # Not configured → no_key, and no probe is attempted.
    row = provider_balance._probe_row("openrouter", "OpenRouter", False, "", None)
    assert row["status"] == "no_key" and row["configured"] is False

    # A probe failure degrades to the exception's FIRST line, key-redacted, truncated.
    secret = "sk-super-secret-key"

    def boom(_key):
        raise RuntimeError(f"{secret} expired\nsecond line is dropped")

    row = provider_balance._probe_row("deepseek", "DeepSeek", True, secret, boom)
    assert row["status"] == "error"
    assert row["error"] == "RuntimeError: *** expired"
    assert secret not in row["error"]


def test_probe_row_truncates_long_errors(monkeypatch):
    monkeypatch.setattr(provider_balance, "_ERROR_MAX_CHARS", 40)
    row = provider_balance._probe_row("deepseek", "DeepSeek", True, "sk-x", lambda _k: (_ for _ in ()).throw(ValueError("x" * 500)))
    assert row["status"] == "error" and len(row["error"]) <= 40


def test_fetch_balances_caches_and_refresh_busts(monkeypatch):
    calls = []
    monkeypatch.setattr(provider_balance, "_CACHE", {})
    monkeypatch.setattr(provider_balance, "_collect", lambda profile: calls.append(profile) or [{"id": "deepseek"}])

    first = provider_balance.fetch_balances(profile=None)
    second = provider_balance.fetch_balances(profile=None)
    third = provider_balance.fetch_balances(profile=None, refresh=True)

    assert first is second
    assert third is not first
    assert calls == [None, None]


def test_balance_route_forwards_profile_and_refresh(monkeypatch):
    from hermes_cli.web_routers import models as models_router

    seen = {}

    def fake_fetch(profile=None, refresh=False):
        seen.update({"profile": profile, "refresh": refresh})
        return {"fetched_at": "2026-11-04T00:00:00+00:00", "providers": []}

    monkeypatch.setattr("hermes_cli.provider_balance.fetch_balances", fake_fetch)
    payload = asyncio.run(models_router.get_model_balance(profile="iris", refresh=True))

    assert payload["providers"] == []
    assert seen == {"profile": "iris", "refresh": True}
