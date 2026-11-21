"""User-facing cron failure notices: plain words, the real output path, and the exact `hermes cron`
command to act on. Contract tests, not snapshots (root AGENTS.md).

The classifier is `agent.error_classifier.classify_api_error`; these tests pin what the copy table
does with its verdict, not the verdict itself.
"""

import re

from cron import scheduler
from cron.scheduler import _compose_run_delivery, _summarize_cron_failure_for_delivery
from hermes_constants import display_hermes_home

JOB = {"name": "Morning brief", "id": "ab12cd34"}
_HTTP_LEAD = re.compile(r"failed: (HTTP|Error code:|provider )")


def _no_chain(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", dict)
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






def test_crashed_script_is_never_blamed_on_the_model_service(monkeypatch):
    """A monitor/pre-run script that dies is not a provider fault (#78503, #82460 precedent): its
    traceback contains "timed out", so provider classification reported "the AI model service did
    not respond in time" for a job that never reached a model (field-reported 2026-09-27)."""
    _no_chain(monkeypatch)
    crashed = (
        "Script exited with code 1\nstderr:\nTraceback (most recent call last):\n"
        '  File "/home/u/.hermes/scripts/monitor.py", line 34, in main\n'
        "subprocess.TimeoutExpired: Command '['curl', '-u', 'user:secret-token', url] "
        "timed out after 30 seconds"
    )
    msg = _summarize_cron_failure_for_delivery(JOB, crashed)
    assert "never ran" in msg and "No model was invoked" in msg, msg
    assert "model service" not in msg, msg  # the misdiagnosis this guard exists for
    assert "secret-token" not in msg and "Traceback" not in msg, msg  # stderr stays local
    assert f"{display_hermes_home()}/cron/output/ab12cd34/" in msg, msg
    for cmd in ("hermes cron runs ab12cd34", "hermes cron run ab12cd34", "hermes cron pause ab12cd34"):
        assert f"`{cmd}`" in msg, msg


def test_monitor_source_failure_shapes_take_the_script_notice(monkeypatch):
    """Every non-timeout shape the script runner and monitor produce lands on the same notice —
    including a monitor URL that fails, whose text can carry a token in the query string."""
    _no_chain(monkeypatch)
    for text in (
        "Script execution failed: <urlopen error timed out>",
        "Script cancelled because cron fire ownership was lost",
        "monitor_url fetch failed: <urlopen error timed out>",
        "monitor job has neither monitor_script nor monitor_url",
    ):
        msg = _summarize_cron_failure_for_delivery(JOB, text)
        assert "never ran" in msg, (text, msg)
        assert "model service" not in msg, (text, msg)


def test_monitor_url_rejections_do_not_echo_credentials_or_blame_provider(monkeypatch):
    """Monitor URL failures precede the agent and may contain the configured URL verbatim."""
    from cron.monitor import MAX_URL_REDIRECTS, _fetch_monitor_url

    _no_chain(monkeypatch)
    url = "ftp://monitor:secret-canary@invalid.example/health"
    ok, rejected = _fetch_monitor_url(url)
    assert not ok
    for error in (
        rejected,
        "monitor_url blocked: private/internal or unsafe destination",
        "monitor_url redirect without a Location header",
        f"monitor_url exceeded {MAX_URL_REDIRECTS} redirects",
    ):
        notice = _summarize_cron_failure_for_delivery(JOB, error)
        assert "No model was invoked" in notice, (error, notice)
        assert "model service" not in notice, (error, notice)
        assert "secret-canary" not in notice, (error, notice)


def test_script_not_found_keeps_only_its_short_diagnostic(monkeypatch):
    """A path diagnostic is self-contained (never captured stderr), so it travels — bounded to its
    first sentence: the runner's remaining prose is help text, and a notice is one line."""
    _no_chain(monkeypatch)
    text = (
        "Script not found: /home/u/.hermes/scripts/x.py. Cron scripts are looked up only in this "
        "profile's folder (/home/u/.hermes/scripts); if the job was copied from another profile, "
        "copy the script there too, or edit the job with `hermes cron edit`."
    )
    msg = _summarize_cron_failure_for_delivery(JOB, text)
    assert "Script not found: /home/u/.hermes/scripts/x.py." in msg, msg
    assert "copied from another profile" not in msg, msg
    assert "never ran" in msg, msg


def test_a_real_provider_timeout_still_names_the_model_service(monkeypatch):
    """Contract kept: the guard matches leading script shapes only, so an actual provider timeout
    must not be swallowed by it."""
    _no_chain(monkeypatch)
    msg = _summarize_cron_failure_for_delivery(JOB, "Request timed out.")
    assert "AI model service did not respond in time" in msg, msg
    assert "never ran" not in msg, msg


def test_blocked_config_notice_says_it_did_not_run_and_will_self_heal():
    text, blocked, *_ = _compose_run_delivery(
        JOB, success=False, error="[blocked_config] provider credential missing: no key",
        final_response="", output_file=None)
    assert blocked is True
    assert "provider credential missing: no key" in text
