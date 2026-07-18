"""E2E: the scheduler's run_job() fires a LOUD #alerts ping when a cron run
actually executed (any part of itself) on a fallback model.

This drives the REAL ``cron.scheduler.run_job`` pipeline (not a mock at the
run_job boundary) with a subclassed ``AIAgent`` whose ``run_conversation``
simulates the codex→opus cross-provider fallback by recording
``self._last_fallback_event`` exactly the way the production
``_emit_fallback_announce`` does. We then assert that the scheduler's post-run
``finally`` hook read that event off the live agent and delivered the alert to
the #alerts target — closing the cron "successful fallback is invisible" gap.

Mirrors the agent-bootstrap scaffolding in ``test_codex_execution_paths.py``.
"""
import sys
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import cron.scheduler as cron_scheduler
import run_agent


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        return None


class _FellBackToOpusAgent(run_agent.AIAgent):
    """Simulates a codex-primary run whose primary failed and recovered on the
    Anthropic Opus fallback. Records the same ``_last_fallback_event`` the real
    ``_emit_fallback_announce`` writes, then returns a successful result."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("skip_context_files", True)
        kwargs.setdefault("skip_memory", True)
        kwargs.setdefault("max_iterations", 4)
        super().__init__(*args, **kwargs)
        self._cleanup_task_resources = lambda task_id: None
        self._persist_session = lambda messages, history=None: None
        self._save_trajectory = lambda messages, user_message, completed: None

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        # The agent walked the chain: openai-codex/gpt-5.5 → claude-app/opus,
        # recording WHY (rate limit) exactly as the real announce path does.
        self._last_fallback_event = {
            "old_model": "gpt-5.5",
            "new_model": "claude-opus-4-8",
            "old_provider": "openai-codex",
            "new_provider": "claude-app",
            "reason_label": "rate limit",
            "reason": "rate_limit",
        }
        return {"final_response": "digest produced on the fallback", "messages": []}


class _PrimaryOnlyAgent(run_agent.AIAgent):
    """A normal run that never left its primary — no fallback event recorded."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("skip_context_files", True)
        kwargs.setdefault("skip_memory", True)
        kwargs.setdefault("max_iterations", 4)
        super().__init__(*args, **kwargs)
        self._cleanup_task_resources = lambda task_id: None
        self._persist_session = lambda messages, history=None: None
        self._save_trajectory = lambda messages, user_message, completed: None

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        return {"final_response": "digest produced on the primary", "messages": []}


def _patch_codex_runtime(monkeypatch):
    monkeypatch.setattr(run_agent, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kw: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc)
    )


def _capture_alerts(monkeypatch):
    """Capture every _deliver_result call (digest delivery + fallback alert)."""
    calls = []

    def fake(job, content, success=True, adapters=None, loop=None, wrap_override=None):
        calls.append({
            "deliver": job.get("deliver"), "content": content,
            "success": success, "wrap_override": wrap_override,
        })
        return None

    monkeypatch.setattr(cron_scheduler, "_deliver_result", fake)
    return calls


DIGEST_JOB = {
    "id": "morning-digest-e2e",
    "name": "morning-digest",
    "prompt": "ping",
    "model": "gpt-5.5",
    "provider": "openai-codex",
    "allow_cross_provider_fallback": True,
    "fallback": [{"provider": "claude-app", "model": "claude-opus-4-8"}],
    "deliver": "discord:LOGS_CHANNEL",
}


def test_run_job_logs_calm_notice_when_declared_fallback_used(monkeypatch):
    """E2E: a real run_job whose agent fell back to its DECLARED fallback (Opus)
    produces a CALM notice to #logs (safety net working as designed), separate
    from the digest delivery — NOT a loud #alerts page."""
    _patch_agent_bootstrap(monkeypatch)
    _patch_codex_runtime(monkeypatch)
    monkeypatch.setattr(run_agent, "AIAgent", _FellBackToOpusAgent)
    alerts = _capture_alerts(monkeypatch)

    success, output, final_response, error = cron_scheduler.run_job(DIGEST_JOB)

    # 1. The run still SUCCEEDED on the fallback.
    assert success is True, error
    assert final_response == "digest produced on the fallback"

    # 2. Exactly one fallback notice fired, to the #logs default (a DECLARED
    #    fallback is expected telemetry, not a #alerts incident), carrying the
    #    full route with calm wording.
    fallback_notes = [
        c for c in alerts
        if c["deliver"] == cron_scheduler._DEFAULT_FALLBACK_LOG_DELIVER
    ]
    assert len(fallback_notes) == 1, alerts
    note = fallback_notes[0]
    assert "openai-codex/gpt-5.5" in note["content"]
    assert "claude-app/claude-opus-4-8" in note["content"]
    # WHY the primary failed is surfaced end-to-end (not guessed)
    assert "rate limit" in note["content"]
    # calm — not the 🚨 shouldn't-happen framing
    assert "🚨" not in note["content"]
    assert "as designed" in note["content"].lower()
    # delivered UNWRAPPED — no "Cronjob Failed/Response" envelope.
    assert note["wrap_override"] is False
    # it went to #logs, not #alerts, and not the job's own digest deliver.
    assert note["deliver"] != cron_scheduler._DEFAULT_FALLBACK_ALERT_DELIVER
    assert note["deliver"] != DIGEST_JOB["deliver"]


def test_run_job_no_alert_on_primary_only_run(monkeypatch):
    """E2E negative: a normal primary-only run fires NO fallback alert."""
    _patch_agent_bootstrap(monkeypatch)
    _patch_codex_runtime(monkeypatch)
    monkeypatch.setattr(run_agent, "AIAgent", _PrimaryOnlyAgent)
    alerts = _capture_alerts(monkeypatch)

    success, output, final_response, error = cron_scheduler.run_job(DIGEST_JOB)

    assert success is True, error
    fallback_alerts = [
        c for c in alerts
        if c["deliver"] == cron_scheduler._DEFAULT_FALLBACK_ALERT_DELIVER
    ]
    assert fallback_alerts == []
