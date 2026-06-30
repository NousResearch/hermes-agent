"""Tests for the LOUD cron fallback alert (`_emit_cron_fallback_alert`).

A cross-provider opt-in cron job (e.g. codex-primary digest with an Anthropic
Opus fallback) runs on its primary in the normal case. If the primary fails and
the agent walks the chain to a different provider, the agent records the switch
as `_last_fallback_event`. Cron passes no `status_callback`, so that interactive
announce reaches nobody — `_emit_cron_fallback_alert` closes the gap by reading
the event after the run and firing a deliberate #alerts ping.

Invariants under test:
  INV-3  a fallback that FIRES is never silent in cron.
  INV-4  the alert can never crash the job (best-effort, swallowed).
  INV-5  the alert routes to the alert target, NOT the digest channel.
"""
import cron.scheduler as sched


class _FakeAgent:
    def __init__(self, event):
        self._last_fallback_event = event


def _capture_deliver(monkeypatch):
    """Patch _deliver_result to record (job, content, success) and return None."""
    calls = []

    def fake(job, content, success=True, adapters=None, loop=None):
        calls.append({"job": job, "content": content, "success": success})
        return None  # delivery OK

    monkeypatch.setattr(sched, "_deliver_result", fake)
    return calls


FALLBACK_EVENT = {
    "old_model": "gpt-5.5",
    "new_model": "claude-opus-4-8",
    "old_provider": "openai-codex",
    "new_provider": "claude-app",
}


def test_alert_fires_when_fallback_event_present(monkeypatch):
    """INV-3: a fired fallback produces exactly one loud alert."""
    calls = _capture_deliver(monkeypatch)
    job = {"id": "md", "name": "morning-digest", "deliver": "discord:LOGS"}
    agent = _FakeAgent(FALLBACK_EVENT)

    sched._emit_cron_fallback_alert(job, agent)

    assert len(calls) == 1
    content = calls[0]["content"]
    # carries the full route old→new (provider/model both sides)
    assert "openai-codex/gpt-5.5" in content
    assert "claude-app/claude-opus-4-8" in content
    assert "FALLBACK" in content.upper()
    # framed as a failure-style (⚠️) alert, not a success delivery
    assert calls[0]["success"] is False


def test_alert_routes_to_default_alerts_channel_not_digest(monkeypatch):
    """INV-5: with no fallback_alert_deliver, route to #alerts — NOT the job's
    own deliver (#logs/digest channel)."""
    calls = _capture_deliver(monkeypatch)
    job = {"id": "md", "name": "morning-digest", "deliver": "discord:LOGS_CHANNEL"}
    agent = _FakeAgent(FALLBACK_EVENT)

    sched._emit_cron_fallback_alert(job, agent)

    alert_deliver = calls[0]["job"]["deliver"]
    assert alert_deliver == sched._DEFAULT_FALLBACK_ALERT_DELIVER
    assert alert_deliver != job["deliver"]


def test_alert_honors_explicit_fallback_alert_deliver(monkeypatch):
    calls = _capture_deliver(monkeypatch)
    job = {
        "id": "md", "name": "morning-digest", "deliver": "discord:LOGS",
        "fallback_alert_deliver": "discord:CUSTOM_ALERTS",
    }
    agent = _FakeAgent(FALLBACK_EVENT)

    sched._emit_cron_fallback_alert(job, agent)

    assert calls[0]["job"]["deliver"] == "discord:CUSTOM_ALERTS"


def test_no_alert_when_no_fallback(monkeypatch):
    """No false positives: the normal (primary-only) run emits nothing."""
    calls = _capture_deliver(monkeypatch)
    job = {"id": "md", "name": "morning-digest", "deliver": "discord:LOGS"}

    # agent with no event attr at all, and agent with explicit None
    sched._emit_cron_fallback_alert(job, _FakeAgent(None))
    sched._emit_cron_fallback_alert(job, object())  # no attr

    assert calls == []


def test_no_alert_when_event_missing_new_model(monkeypatch):
    calls = _capture_deliver(monkeypatch)
    job = {"id": "md", "name": "morning-digest", "deliver": "discord:LOGS"}
    agent = _FakeAgent({"old_model": "gpt-5.5", "new_model": ""})  # incomplete

    sched._emit_cron_fallback_alert(job, agent)

    assert calls == []


def test_alert_never_raises_when_deliver_fails(monkeypatch):
    """INV-4: a raising deliver fn must not propagate — the job stays green."""
    def boom(*a, **k):
        raise RuntimeError("discord 503")

    monkeypatch.setattr(sched, "_deliver_result", boom)
    job = {"id": "md", "name": "morning-digest", "deliver": "discord:LOGS"}
    agent = _FakeAgent(FALLBACK_EVENT)

    # Must return None without raising.
    assert sched._emit_cron_fallback_alert(job, agent) is None


def test_alert_handles_event_missing_providers(monkeypatch):
    """Degrade gracefully to bare model slugs if provider labels are absent."""
    calls = _capture_deliver(monkeypatch)
    job = {"id": "md", "name": "morning-digest", "deliver": "discord:LOGS"}
    agent = _FakeAgent({"old_model": "gpt-5.5", "new_model": "claude-opus-4-8"})

    sched._emit_cron_fallback_alert(job, agent)

    content = calls[0]["content"]
    assert "gpt-5.5" in content and "claude-opus-4-8" in content
