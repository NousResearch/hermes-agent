"""User-facing cron failure notices: plain words, the real output path, and the exact `hermes cron`
command to act on. Contract tests, not snapshots (root AGENTS.md).

The classifier is `agent.error_classifier.classify_api_error`; these tests pin what the copy table
does with its verdict, not the verdict itself.
"""

import re

import cron.scheduler as scheduler
from cron.scheduler import _compose_run_delivery, _summarize_cron_failure_for_delivery

JOB = {"name": "Morning brief", "id": "ab12cd34"}
_HTTP_LEAD = re.compile(r"failed: (HTTP|Error code:|provider )")


def _no_chain(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])






def test_auth_failure_names_the_pinned_provider_and_the_failing_profile(monkeypatch, tmp_path):
    """A profile's credentials are its own (93889b770da): the notice must send the operator to
    THIS profile's sign-in for the job's pinned provider, never a bare placeholder (#114012)."""
    _no_chain(monkeypatch)
    profile_home = tmp_path / ".hermes" / "profiles" / "ops"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    msg = _summarize_cron_failure_for_delivery(
        {**JOB, "provider": "openai-codex"}, "Error code: 401 - Unauthorized")
    assert "`hermes -p ops auth add openai-codex --type oauth`" in msg, msg
    assert "<provider>" not in msg
    unpinned = _summarize_cron_failure_for_delivery(JOB, "Error code: 401 - Unauthorized")
    assert "`hermes -p ops auth add <provider>`" in unpinned, unpinned


def test_rate_and_usage_limit_phrases_still_yield_a_provider_notice(monkeypatch):
    """The old cron regex ladder matched these substrings; the shared classifier must too, or a
    Nous Portal limit turns into a raw generic notice."""
    _no_chain(monkeypatch)
    for text in (
        "Nous Portal rate limit active until 15:00",
        "RuntimeError: usage limit reached for this key",
        "You have hit your weekly usage limit",
        "insufficient quota",
    ):
        msg = _summarize_cron_failure_for_delivery(JOB, text)
        assert "limit" in msg.lower(), msg
        assert not _HTTP_LEAD.search(msg), msg
        assert "`hermes cron run ab12cd34`" in msg or "`hermes cron edit ab12cd34" in msg, msg


def test_cron_cause_gloss_is_the_shared_table():
    """Cron, subagent and chat notices read one reason->cause table (agent/turn_failure_copy.py)."""
    from agent.turn_failure_copy import FAILURE_CAUSE_GLOSS
    from cron.scheduler_failure_copy import provider_failure_notice

    for reason in FAILURE_CAUSE_GLOSS:
        notice = provider_failure_notice("Morning brief", "ab12cd34", reason, backup_provider_phrase="x.")
        assert notice is not None and "`hermes cron" in notice, reason
    assert provider_failure_notice("Morning brief", "ab12cd34", "unknown", backup_provider_phrase="x.") is None






def test_blocked_config_notice_says_it_did_not_run_and_will_self_heal():
    text, blocked, *_ = _compose_run_delivery(
        JOB, success=False, error="[blocked_config] provider credential missing: no key",
        final_response="", output_file=None)
    assert blocked is True
    assert "provider credential missing: no key" in text
