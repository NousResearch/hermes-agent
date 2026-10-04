"""Billing recovery must cross the real auth-store merge without lifting 429 windows."""
import base64
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.credential_pool import load_pool
from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import _arm_rate_limit_cooldown, _probe_primary_billing_recovery
from hermes_cli.auth import read_credential_pool, write_credential_pool
from hermes_cli.auth_constants import NOUS_INFERENCE_INVOKE_SCOPE


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path / "auth.json"


def _row(cid, reason="billing", status="exhausted"):
    claims = {"scope": NOUS_INFERENCE_INVOKE_SCOPE, "exp": time.time() + 86400}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return {
        "id": cid, "source": "manual", "auth_type": "api_key",
        "access_token": "eyJhbGciOiJub25lIn0." + payload + ".test", "priority": 0,
        "label": "manual", "request_count": 0,
        "last_error_message": None, "last_error_reason": None, "last_error_reset_at": None,
        "last_status": status, "last_status_at": time.time(),
        "failure_reason": reason, "last_error_code": 402 if reason == "billing" else 429,
    }


def _agent(pool=None):
    return SimpleNamespace(
        _primary_runtime={"provider": "nous", "model": "test-model"},
        provider="openrouter", _credential_pool=pool, _fallback_activated=True,
        _rate_limited_until=time.monotonic() + 3600, _rate_limit_backoff_count=7,
    )


def _portal(monkeypatch, total=10):
    probe = Mock(return_value=SimpleNamespace(
        paid_service_access_info=SimpleNamespace(total_usable_credits=total), is_paid=True,
    ))
    monkeypatch.setattr("hermes_cli.nous_account.get_nous_portal_account_info", probe)
    return probe


def test_clear_billing_benches_survives_persistence_and_reload(store):
    rows = [_row("billing"), _row("rate", "rate_limit"),
            _row("auth", "auth"), _row("unverified", "billing_unverified"),
            _row("dead", "billing", "dead")]
    write_credential_pool("nous", rows)
    before = {r["id"]: r for r in read_credential_pool("nous")}
    pool = load_pool("nous")
    assert pool.clear_billing_benches() == 1
    disk = {r["id"]: r for r in read_credential_pool("nous")}
    assert disk["billing"]["last_status"] == "ok"
    assert disk["billing"]["last_status_at"] is None
    assert disk["billing"].get("failure_reason") is None
    assert disk["billing"]["status_cleared_at"] > 0
    for cid in ("rate", "auth", "unverified", "dead"):
        assert disk[cid] == before[cid]
    reloaded = load_pool("nous")
    recovered = next(e for e in reloaded.entries() if e.id == "billing")
    assert recovered.last_status == "ok"
    assert recovered.failure_reason is None
    assert reloaded.has_available()
    assert reloaded.clear_billing_benches() == 0


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("cooldown", [False, True])
def test_funded_billing_recovery_clears_real_primary_store(store, monkeypatch, attached, cooldown):
    write_credential_pool("nous", [_row("billing")])
    agent = _agent(load_pool("nous") if attached else load_pool("openrouter"))
    if not cooldown:
        agent._rate_limited_until = 0
    probe = _portal(monkeypatch)
    assert _probe_primary_billing_recovery(agent)
    probe.assert_called_once_with(force_fresh=True)
    assert agent._rate_limited_until <= time.monotonic()
    assert agent._rate_limit_backoff_count == 0
    assert load_pool("nous").has_available()
    assert read_credential_pool("nous")[0]["last_status"] == "ok"
    assert not _probe_primary_billing_recovery(agent)
    probe.assert_called_once()


@pytest.mark.parametrize("reason", [None, "rate_limit", "auth", "billing_unverified"])
def test_nonbilling_cooldown_never_probes_or_changes_state(store, monkeypatch, reason):
    write_credential_pool("nous", [] if reason is None else [_row("control", reason)])
    agent = _agent(load_pool("nous"))
    before = store.read_bytes()
    until = agent._rate_limited_until
    probe = _portal(monkeypatch)
    assert not _probe_primary_billing_recovery(agent)
    assert agent._rate_limited_until == until
    assert agent._rate_limit_backoff_count == 7
    assert store.read_bytes() == before
    probe.assert_not_called()


@pytest.mark.parametrize("rate_bench", [False, True])
def test_pure429_backoff_still_escalates_to_four_hours(store, monkeypatch, rate_bench):
    write_credential_pool("nous", [_row("rate", "rate_limit")] if rate_bench else [])
    agent = _agent(load_pool("nous"))
    agent._rate_limit_backoff_count = 0
    probe = _portal(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr("agent.fallback_cooldown.time.monotonic", lambda: clock[0])
    for count, seconds in enumerate([60, 120, 240, 480, 960, 1920, 3840, 7680, 14400, 14400], 1):
        agent.provider = "nous"
        assert _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit) == seconds
        agent.provider = "openrouter"
        assert not _probe_primary_billing_recovery(agent)
        assert agent._rate_limited_until == clock[0] + seconds
        assert agent._rate_limit_backoff_count == count
        clock[0] += seconds + 1
    probe.assert_not_called()


@pytest.mark.parametrize("total", [0, -1])
def test_unfunded_billing_probe_preserves_store_and_cooldown(store, monkeypatch, total):
    write_credential_pool("nous", [_row("billing")])
    agent = _agent(load_pool("nous"))
    before = store.read_bytes()
    until = agent._rate_limited_until
    _portal(monkeypatch, total)
    assert not _probe_primary_billing_recovery(agent)
    assert agent._rate_limited_until == until
    assert agent._rate_limit_backoff_count == 7
    assert store.read_bytes() == before


def test_portal_failure_is_throttled_and_keeps_billing_state(store, monkeypatch):
    write_credential_pool("nous", [_row("billing")])
    agent = _agent(load_pool("nous"))
    before = store.read_bytes()
    until = agent._rate_limited_until
    probe = _portal(monkeypatch)
    probe.side_effect = RuntimeError("portal unavailable")
    assert not _probe_primary_billing_recovery(agent)
    assert not _probe_primary_billing_recovery(agent)
    probe.assert_called_once_with(force_fresh=True)
    assert agent._rate_limited_until == until
    assert agent._rate_limit_backoff_count == 7
    assert store.read_bytes() == before


@pytest.mark.parametrize("failure", [False, True])
def test_unsuccessful_unbench_does_not_clear_session_cooldown(store, monkeypatch, failure):
    write_credential_pool("nous", [_row("billing")])
    pool = load_pool("nous")
    agent = _agent(pool)
    before = store.read_bytes()
    until = agent._rate_limited_until
    _portal(monkeypatch)
    clear = Mock(return_value=0)
    if failure:
        clear.side_effect = RuntimeError("persist unavailable")
    monkeypatch.setattr(pool, "clear_billing_benches", clear)
    assert not _probe_primary_billing_recovery(agent)
    assert agent._rate_limited_until == until
    assert agent._rate_limit_backoff_count == 7
    assert store.read_bytes() == before
