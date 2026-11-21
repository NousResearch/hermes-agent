import time
from types import SimpleNamespace

from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import switch_deferred_by_reset


def _cfg(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"fallback": {"min_switch_reset_seconds": 120}})


def test_billing_reason_is_never_deferred(monkeypatch):
    _cfg(monkeypatch)
    agent = SimpleNamespace(_fallback_activated=False)
    assert switch_deferred_by_reset(agent, FailoverReason.billing, time.time() + 30) is False


def test_rate_limit_with_short_reset_still_deferred(monkeypatch):
    _cfg(monkeypatch)
    agent = SimpleNamespace(_fallback_activated=False)
    assert switch_deferred_by_reset(agent, FailoverReason.rate_limit, time.time() + 30) is True
