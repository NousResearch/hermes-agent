"""User-facing cron failure notices: plain words, the real output path, and the exact `hermes cron`
command to act on. Contract tests, not snapshots (root AGENTS.md).

The classifier is `agent.error_classifier.classify_api_error`; these tests pin what the copy table
does with its verdict, not the verdict itself.
"""

import re

from cron import scheduler
from cron.scheduler import _compose_run_delivery, _summarize_cron_failure_for_delivery

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






def test_blocked_config_notice_says_it_did_not_run_and_will_self_heal():
    text, blocked, *_ = _compose_run_delivery(
        JOB, success=False, error="[blocked_config] provider credential missing: no key",
        final_response="", output_file=None)
    assert blocked is True
    assert "provider credential missing: no key" in text


def test_script_failure_keeps_its_kind_and_saved_cause(monkeypatch, tmp_path):
    """A monitor script runs before the model; its HTTP errors are not model failures."""
    _no_chain(monkeypatch)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "monitor.py").write_text(
        'import sys\nprint("HTTP 401: deploy token revoked", file=sys.stderr)\nsys.exit(1)\n')
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {**JOB, "monitor_script": "monitor.py"}
    early, _, _ = scheduler._apply_monitor_gate(job, JOB["id"], JOB["name"], None)
    assert early is not None and early[0] is False
    real_error = early[3]
    for error, cause in (
        (real_error, "HTTP 401: deploy token revoked"),
        ("Script execution failed: HTTP 429: remote service rate limit", "HTTP 429: remote service rate limit"),
        ("Script exited with code 1\nstderr:\nidle for 600s (limit 600s)", "idle for 600s (limit 600s)"),
    ):
        msg = _summarize_cron_failure_for_delivery(job, error)
        assert cause not in msg, msg
        assert "Execution failed" in msg, msg
        assert "Details:" in msg, msg
        assert "Script exited with code" not in msg, msg
        assert "Script execution failed:" not in msg, msg
        assert "stderr:" not in msg, msg
        assert "backup provider" not in msg.lower(), msg
        assert "auth add" not in msg, msg
        assert "--provider" not in msg, msg
    provider = _summarize_cron_failure_for_delivery(
        {**JOB, "script": "collect.py"}, "HTTP 401: Unauthorized")
    assert "auth add" in provider, provider


def test_cron_actions_target_the_owning_profile(monkeypatch, tmp_path):
    """Copy-pasted recovery commands must act on this profile's job, not the root store."""
    _no_chain(monkeypatch)
    profile_home = tmp_path / ".hermes" / "profiles" / "ops"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    for error in ("HTTP 401: Unauthorized", "Request timed out.", "unknown failure",
                  "Script timed out after 60s: collect.py", "idle for 600s (limit 600s)"):
        msg = _summarize_cron_failure_for_delivery(JOB, error)
        assert "hermes -p ops cron" in msg, msg
        assert "`hermes cron " not in msg, msg
    job = {**JOB, "failure_streak": 27, "schedule": {"kind": "interval"}}
    from cron.scheduler_failure_copy import _failure_streak_nudge, blocked_config_notice

    assert "hermes -p ops cron doctor" in blocked_config_notice(JOB["name"], "missing key")
    nudge = _failure_streak_nudge(job)
    assert "hermes -p ops cron pause ab12cd34" in nudge, nudge


def test_generic_failure_is_readable_and_keeps_one_profile_scoped_details_pointer(monkeypatch, tmp_path):
    profile_home = tmp_path / ".hermes" / "profiles" / "ops"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    cause = "Task API read failed (timeout 15s). No task command was submitted."
    for error in (
        f"Script exited with code 1\nstderr:\n{cause}",
        f"Script exited with code 2\nstdout:\n{cause}",
        f"Script execution failed: {cause}",
        "Script exited with code 7",
        "Script execution failed:",
    ):
        msg = _summarize_cron_failure_for_delivery({**JOB, "no_agent": True}, error)
        lines = msg.splitlines()
        assert len(lines) == 3, msg
        assert lines[0] == "⚠️ Cron 'Morning brief' failed", msg
        code = re.match(r"^Script exited with code (-?\d+)\b", error)
        expected = f"Execution failed (exit code {code.group(1)})." if code else "Execution failed."
        assert lines[1] == expected, msg
        assert cause not in msg, msg
        assert lines[2] == "Details: `hermes -p ops cron runs ab12cd34`.", msg
        assert "Output:" not in msg and "History:" not in msg, msg
        assert "cron run " not in msg and "cron edit " not in msg and "cron pause " not in msg, msg


def test_monitor_failure_notice_never_publishes_script_diagnostics(monkeypatch, tmp_path):
    """Real monitor errors remain inspectable locally, never interpolated into chat."""
    _no_chain(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    diagnostics = (
        "The export subagent timed out after 30 minutes waiting on the database.",
        "HTTP 401: the report service needs its account renewed.",
        "HTTP 401: token=FixtureOpaqueSecret123456789ABC",
        "safe operation failed " + "x" * 140 + " token=FixtureBoundaryCredential123456789ABC",
        "HTTP 401: https://fixture-user:fixture-password@example.invalid/?token=fixture-query-secret",
    )
    job = {**JOB, "monitor_script": "monitor.py"}
    for diagnostic in diagnostics:
        (scripts / "monitor.py").write_text(
            f"import sys\nprint({diagnostic!r}, file=sys.stderr)\nsys.exit(1)\n")
        early, _, _ = scheduler._apply_monitor_gate(job, JOB["id"], JOB["name"], None)
        assert early is not None and early[0] is False
        assert diagnostic.split(":")[0] in early[3]
        msg = _summarize_cron_failure_for_delivery(job, early[3])
        assert msg.splitlines()[1] == "Execution failed (exit code 1).", msg
        assert "HTTP 401" not in msg and "safe operation failed" not in msg
        assert "export subagent" not in msg
        assert "Fixture" not in msg and "fixture-" not in msg and "token=" not in msg
        assert "Details: `hermes cron runs ab12cd34`." in msg
