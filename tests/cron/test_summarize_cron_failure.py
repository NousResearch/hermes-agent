"""Tests for #70908: the cron failure summarizer must not misattribute a
``no_agent`` script failure to a provider error.

``no_agent=True`` jobs run a bash script with no LLM/provider involved. When
such a script exits non-zero, its entire stdout is handed to
``_summarize_cron_failure_for_delivery`` as the error string. The provider
classification branches (rate limit / timeout / authentication) match bare
tokens anywhere in that text, so ordinary script output that merely mentions
``401``/``403``/``429``/``timeout`` (e.g. a passing auth-gate test printing
"returns 401") produced a misleading "provider authentication error"
notification for a failure that had nothing to do with any provider.

These tests pin that a no_agent job never gets a provider-flavoured summary,
while agent jobs keep the existing classification.
"""

from cron.scheduler import _summarize_cron_failure_for_delivery


# --- no_agent jobs: never classified as a provider error ------------------

def test_no_agent_401_in_stdout_is_not_provider_auth_error():
    job = {"name": "verify", "no_agent": True}
    error = "PASS  auth gate returns 401\nFAIL  some other check\n"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider authentication error" not in msg
    assert "verify" in msg


def test_no_agent_403_in_stdout_is_not_provider_auth_error():
    job = {"name": "verify", "no_agent": True}
    error = "ok: endpoint returns 403 for anonymous\nunrelated failure\n"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider authentication error" not in msg


def test_no_agent_429_in_stdout_is_not_provider_rate_limit():
    job = {"name": "verify", "no_agent": True}
    error = "checked that throttled route returns 429\nassertion failed\n"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider" not in msg


def test_no_agent_timeout_word_in_stdout_is_not_provider_timeout():
    job = {"name": "verify", "no_agent": True}
    error = "config sets request timeout=30\nstep 2 failed\n"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider timeout" not in msg


# --- agent jobs: provider classification unchanged ------------------------

def test_agent_job_401_still_classified_as_provider_auth():
    job = {"name": "chat", "no_agent": False}
    error = "HTTP 401 Unauthorized from provider"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider authentication error" in msg


def test_agent_job_without_flag_401_still_classified_as_provider_auth():
    # A legacy job dict that predates the no_agent field must still classify.
    job = {"name": "chat"}
    error = "401 Unauthorized"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider authentication error" in msg


def test_agent_job_429_still_classified_as_provider_rate_limit():
    job = {"name": "chat", "no_agent": False}
    error = "429 Too Many Requests"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider rate limit" in msg
