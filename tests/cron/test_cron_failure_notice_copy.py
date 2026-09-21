"""User-facing cron failure notices: plain words, the real output path, and the exact `hermes cron`
command to act on. Contract tests, not snapshots (root AGENTS.md).

The classifier is `agent.error_classifier.classify_api_error`; these tests pin what the copy table
does with its verdict, not the verdict itself.
"""

import re

import cron.scheduler as scheduler
from cron.scheduler import _compose_run_delivery, _summarize_cron_failure_for_delivery
from hermes_constants import display_hermes_home

JOB = {"name": "Morning brief", "id": "ab12cd34"}
_HTTP_LEAD = re.compile(r"failed: (HTTP|Error code:|provider )")


def _no_chain(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])


def test_generic_failure_names_runs_and_pause_commands_and_the_real_output_dir():
    msg = _summarize_cron_failure_for_delivery(JOB, "[Errno 2] No such file or directory: '/x.py'")
    assert "/x.py" in msg  # the raw detail survives as the cause
    for cmd in ("hermes cron runs ab12cd34", "hermes cron run ab12cd34", "hermes cron pause ab12cd34"):
        assert f"`{cmd}`" in msg
    assert f"{display_hermes_home()}/cron/output/ab12cd34/" in msg
    assert "cron output" not in msg  # the unnamed internal location is gone


def test_auth_failure_points_at_login_and_a_retry_command(monkeypatch):
    _no_chain(monkeypatch)
    msg = _summarize_cron_failure_for_delivery(JOB, "Error code: 401 - Unauthorized")
    assert "/login" in msg and "`hermes auth add <provider>`" in msg
    assert "hermes login" not in msg  # that command was removed
    assert "`hermes cron run ab12cd34`" in msg
    assert not _HTTP_LEAD.search(msg)
    assert "401" not in msg


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


def test_waf_block_names_the_header_fix_not_a_bare_rerun(monkeypatch):
    """A firewall refusing the SDK's User-Agent is healed by a header or another provider,
    never by `hermes cron run` alone — the action must say so (#53099, #70566)."""
    _no_chain(monkeypatch)
    msg = _summarize_cron_failure_for_delivery(JOB, "Error code: 403 - Sorry, you have been blocked")
    assert "extra_headers" in msg and "`hermes cron edit ab12cd34 --provider <name>`" in msg, msg
    assert "Run it again with" not in msg, msg
    assert "rejected" not in msg.lower(), msg  # not read as a key rejection


def test_transient_provider_failures_never_lead_with_jargon(monkeypatch):
    _no_chain(monkeypatch)
    for err in ("Request timed out.", "HTTP 429: Too Many Requests"):
        msg = _summarize_cron_failure_for_delivery(JOB, err)
        assert not _HTTP_LEAD.search(msg), msg
        assert "fallback chain" not in msg.lower()
        assert "`hermes cron run ab12cd34`" in msg
        assert "`hermes cron runs ab12cd34`" in msg


def test_blocked_config_notice_says_it_did_not_run_and_will_self_heal():
    text, blocked, *_ = _compose_run_delivery(
        JOB, success=False, error="[blocked_config] provider credential missing: no key",
        final_response="", output_file=None)
    assert blocked is True
    assert "did not run" in text
    assert "provider credential missing: no key" in text
    assert "Nothing was charged" in text
    # Some blocks (MCP server temporarily down) clear on their own, so the retry
    # line must not condition the retry on the user fixing something.
    assert "once this is fixed" not in text
    assert "`hermes cron doctor`" in text
    for jargon in ("configuration validation", "LLM call", "pre-dispatch"):
        assert jargon not in text
