"""User-facing cron failure notices: plain words and the exact `hermes cron`
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


def test_generic_notice_drops_the_job_name_echoed_by_the_script():
    """Scripts conventionally prefix their own stderr with the job name, and the notice
    header already carries it, so the body must not repeat it. The echo is not at index 0:
    a script failure arrives as "Script exited with code 1 stderr: <name>: ..."."""
    from cron.scheduler_failure_copy import generic_failure_notice

    real_shape = generic_failure_notice(
        "Morning brief", "ab12cd34", "Script exited with code 1 stderr: Morning brief: detector timed out")
    assert real_shape.count("Morning brief") == 1, real_shape
    assert "detector timed out" in real_shape
    # Caseless: the echo need not match the job's casing.
    assert "fetch failed" in generic_failure_notice("Nightly digest", "ab12cd34", "nightly digest: fetch failed")
    # An error that merely contains the name elsewhere keeps its full text.
    intact = generic_failure_notice("Morning brief", "ab12cd34", "[Errno 28] No space left on device")
    assert "[Errno 28] No space left on device" in intact


def test_generic_notice_never_delivers_an_empty_or_mangled_body():
    """The echo strip must not eat the error it was meant to clean up."""
    from cron.scheduler_failure_copy import generic_failure_notice

    # Error that is nothing BUT the echo: stripping it blindly leaves no reason at all.
    only_echo = generic_failure_notice("Morning brief", "ab12cd34", "Morning brief:")
    assert only_echo.splitlines()[0].rstrip().endswith(("brief:", "captured")), only_echo
    # str.lower() is not length-preserving (U+0130 folds to two code points). The old
    # slice-by-len(job_name) ate a character of the real error; a regex match cannot,
    # so the worst case here is that the echo survives -- never that the error is cut.
    turkish = generic_failure_notice("\u0130ssues", "ab12cd34", "i\u0307ssues: HTTP 500 from API")
    assert "HTTP 500 from API" in turkish, turkish
    # A blank name must not turn every colon-leading error into a match.
    assert "not an echo" in generic_failure_notice("   ", "ab12cd34", ": not an echo")


def test_failure_notices_carry_the_run_log_command_and_stay_plain_text():
    """Every failure notice must name the one command that shows the run, and must survive
    the plain-text sinks in the delivery fan-out (ntfy and email render markdown literally)."""
    from cron.scheduler_failure_copy import (blocked_config_notice, generic_failure_notice,
                                             inactivity_notice, script_timeout_notice)

    for notice in (
        generic_failure_notice("Morning brief", "ab12cd34", "boom"),
        script_timeout_notice("Morning brief", "ab12cd34"),
        inactivity_notice("Morning brief", "ab12cd34"),
    ):
        assert "`hermes cron runs ab12cd34`" in notice, notice
        # Sibling tests assert this word (test_cron_failure_deliver.py:125,
        # test_fire_claim_lost_after_delivery.py:129); routing itself is structural
        # via _resolve_delivery_targets(..., for_failure=True).
        assert "failed" in notice.lower(), notice

    for notice in (
        generic_failure_notice("Morning brief", "ab12cd34", "boom"),
        script_timeout_notice("Morning brief", "ab12cd34"),
        inactivity_notice("Morning brief", "ab12cd34"),
        blocked_config_notice("Morning brief", "provider credential missing"),
    ):
        assert "**" not in notice, notice


def test_blocked_config_notice_keeps_an_ellipsis_and_never_empties_the_reason():
    """rstrip('.') strips a whole run of dots, deleting the marker that says the
    reason was truncated."""
    from cron.scheduler_failure_copy import blocked_config_notice

    assert "3 files..." in blocked_config_notice("Morning brief", "waiting for 3 files...")
    assert "no key" in blocked_config_notice("Morning brief", "provider credential missing: no key.")
    assert "configuration check failed" in blocked_config_notice("Morning brief", "")
