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
    """Patch _deliver_result to record (job, content, success, wrap_override)."""
    calls = []

    def fake(job, content, success=True, adapters=None, loop=None, wrap_override=None):
        calls.append({
            "job": job, "content": content, "success": success,
            "wrap_override": wrap_override,
        })
        return None  # delivery OK

    monkeypatch.setattr(sched, "_deliver_result", fake)
    return calls


FALLBACK_EVENT = {
    "old_model": "gpt-5.5",
    "new_model": "claude-opus-4-8",
    "old_provider": "openai-codex",
    "new_provider": "claude-app",
    "reason_label": "rate limit",
}

# A job whose ``fallback`` chain DECLARES the model the run fell back to — the
# operator-configured safety net. Firing this is expected/calm (→ #logs).
DECLARED_JOB = {
    "id": "md", "name": "morning-digest", "deliver": "discord:LOGS",
    "fallback": [{"provider": "claude-app", "model": "claude-opus-4-8"}],
}
# A job with NO declared fallback — the agent walked to an undeclared model.
# That is the genuine "shouldn't happen" case (→ #alerts, loud).
UNDECLARED_JOB = {"id": "md", "name": "morning-digest", "deliver": "discord:LOGS"}


def test_declared_fallback_is_calm_and_routes_to_logs(monkeypatch):
    """A DECLARED fallback firing is the safety net working as designed:
    calm wording, routed to #logs — NOT a loud #alerts page."""
    calls = _capture_deliver(monkeypatch)
    agent = _FakeAgent(FALLBACK_EVENT)

    sched._emit_cron_fallback_alert(DECLARED_JOB, agent)

    assert len(calls) == 1
    content = calls[0]["content"]
    # calm, not the 🚨 shouldn't-happen framing
    assert "🚨" not in content
    assert "shouldn't happen" not in content.lower()
    assert "as designed" in content.lower()
    # both models still reported
    assert "claude-opus-4-8" in content and "gpt-5.5" in content
    # WHY the primary failed is surfaced (not guessed)
    assert "rate limit" in content
    # routed to #logs, not #alerts
    assert calls[0]["job"]["deliver"] == sched._DEFAULT_FALLBACK_LOG_DELIVER
    assert calls[0]["job"]["deliver"] != sched._DEFAULT_FALLBACK_ALERT_DELIVER


def test_undeclared_fallback_is_loud_and_routes_to_alerts(monkeypatch):
    """An UNDECLARED provider/model switch is genuinely unexpected: loud 🚨
    framing, routed to #alerts."""
    calls = _capture_deliver(monkeypatch)
    agent = _FakeAgent(FALLBACK_EVENT)

    sched._emit_cron_fallback_alert(UNDECLARED_JOB, agent)

    assert len(calls) == 1
    content = calls[0]["content"]
    assert "🚨" in content
    assert "UNDECLARED" in content.upper()
    # WHY the primary failed is surfaced (not guessed)
    assert "rate limit" in content
    assert calls[0]["job"]["deliver"] == sched._DEFAULT_FALLBACK_ALERT_DELIVER


def test_reason_degrades_gracefully_when_unrecorded(monkeypatch):
    """If the runtime recorded no reason, the notice says so honestly rather
    than guessing a cause."""
    calls = _capture_deliver(monkeypatch)
    event = {k: v for k, v in FALLBACK_EVENT.items() if k != "reason_label"}
    agent = _FakeAgent(event)

    sched._emit_cron_fallback_alert(DECLARED_JOB, agent)

    content = calls[0]["content"]
    assert "unrecorded" in content.lower()
    # must NOT fabricate a specific cause
    assert "rate limit" not in content


def test_alert_fires_when_fallback_event_present(monkeypatch):
    """INV-3: a fired fallback produces exactly one alert (undeclared here)."""
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
    # delivered UNWRAPPED (no "Cronjob Failed/Response" envelope) — the reported
    # job succeeded on its fallback; the alert is self-contained.
    assert calls[0]["wrap_override"] is False


def test_alert_routes_to_default_alerts_channel_not_digest(monkeypatch):
    """INV-5: an UNDECLARED fallback with no fallback_alert_deliver routes to
    #alerts — NOT the job's own deliver (#logs/digest channel)."""
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
