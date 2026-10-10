"""_summarize_cron_failure_for_delivery must not mislabel a ``script_failure_policy=fail``
gate failure as a provider failure.

The gate (cron/scheduler._prepare_job_prompt, #123441) fails the run BEFORE any agent is
constructed, and its error text embeds the pre-run script's stderr verbatim. That stderr can
contain "Read timed out", "429" or "401" — wording that classifies as provider trouble even
though no model was ever touched (the whole point of the policy is to fail before the model).
The summarizer must recognize the gate's marker first and deliver a script-origin notice that
quotes the script's own output, the same way the script-timeout and inactivity shapes are
matched before the provider classifier.
"""

from cron import scheduler
from cron.scheduler import _summarize_cron_failure_for_delivery


def _gate_error(script_output: str) -> str:
    """The error string the fail-gate returns from _prepare_job_prompt."""
    return f"Pre-run script failed (script_failure_policy=fail): {script_output}"


def test_script_stderr_read_timeout_is_not_blamed_on_the_model():
    msg = _summarize_cron_failure_for_delivery(
        {"name": "Nightly Pulse", "id": "job_pulse"},
        _gate_error("stderr:\nrequests.exceptions.ReadError: Read timed out. (read timeout=30)"),
    )
    assert "did not respond in time" not in msg
    assert "rate-limited" not in msg
    assert "sign in" not in msg.lower()
    assert "pre-run script exited non-zero" in msg
    assert "Read timed out" in msg


def test_script_stderr_429_is_not_reported_as_rate_limit():
    msg = _summarize_cron_failure_for_delivery(
        {"name": "Ledger Sync", "id": "job_sync"},
        _gate_error("Script exited with code 1\nstderr:\nError code: 429 - Too Many Requests"),
    )
    assert "rate-limited" not in msg
    assert "Too Many Requests" in msg
    assert "pre-run script exited non-zero" in msg


def test_script_stderr_401_is_not_reported_as_auth_failure():
    msg = _summarize_cron_failure_for_delivery(
        {"name": "Ledger Sync", "id": "job_sync"},
        _gate_error("Script exited with code 2\nstderr:\n401 Unauthorized: bad key"),
    )
    assert "sign in again" not in msg.lower()
    assert "rejected the sign-in" not in msg
    assert "401 Unauthorized" in msg


def test_plain_script_failure_names_the_script_and_quotes_output():
    """Non-provider stderr must still get the dedicated copy: WHAT happened (the script failed
    on its own), WHERE to look, WHAT to do."""
    msg = _summarize_cron_failure_for_delivery(
        {"name": "Nightly Pulse", "id": "job_pulse"},
        _gate_error("Script exited with code 3\nstderr:\nboom: data collection failed"),
    )
    assert "pre-run script exited non-zero" in msg
    assert "without running the model" in msg
    assert "boom: data collection failed" in msg
    assert "hermes cron runs job_pulse" in msg
    assert "hermes cron run job_pulse" in msg


def test_gate_marker_is_matched_verbatim_only():
    """Only the gate's own start-of-text marker takes the script-origin branch; a job error
    that merely mentions the marker (e.g. a script echoing it) must not be hijacked — a
    no_agent job falls through to the generic notice."""
    msg = _summarize_cron_failure_for_delivery(
        {"name": "Nightly Pulse", "id": "job_pulse", "no_agent": True},
        "script printed: Pre-run script failed (script_failure_policy=fail): example",
    )
    assert "exited non-zero" not in msg


def test_genuine_provider_failures_still_classify(monkeypatch):
    """The script-origin branch must not swallow real provider failures: non-marker text keeps
    the provider lane (regression guard for the branch's placement)."""
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    msg = _summarize_cron_failure_for_delivery(
        {"name": "CI Autofix Poller", "id": "job_poll"}, "Request timed out.")
    assert "did not respond in time" in msg
