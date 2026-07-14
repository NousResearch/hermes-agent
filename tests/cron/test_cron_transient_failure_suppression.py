"""Tests for transient-failure classification + immediate-page suppression.

A self-healing RECURRING cron job (e.g. the skill-patch-applier queue-drain)
that fails on a momentary provider-capacity blip — the Claude sub relay
returning 503 ``{"error":"no eligible sub"}`` while the subscription pool is
capped — should NOT fire an immediate per-run ⚠️ failure page. The run re-fires
on its own schedule and the separate consecutive-sample monitor still owns the
loud path when the condition actually persists.

Invariants under test:
  INV-A  a capacity/rate/timeout 503 is classified transient; a real defect is not.
  INV-B  the delivery summarizer NAMES the capacity condition instead of leaking
         a raw ``RuntimeError: HTTP 503: {...}`` provider blob.
  INV-C  suppression fires ONLY for a transient failure on a RECURRING job with
         the toggle on; a one-shot (or a real defect, or the toggle off) is
         never suppressed.
"""
import cron.scheduler as sched


# --- fixtures -------------------------------------------------------------

RECURRING_JOB = {
    "id": "abc123",
    "name": "skill-patch-applier",
    "schedule": {"kind": "cron", "expr": "*/10 * * * *"},
    "repeat": {"times": None, "completed": 200},
}

ONESHOT_JOB = {
    "id": "def456",
    "name": "one-time-reminder",
    "schedule": {"kind": "once"},
    "repeat": {"times": 1, "completed": 0},
}

FINITE_JOB = {
    "id": "ghi789",
    "name": "finite-3x",
    "schedule": {"kind": "cron", "expr": "0 * * * *"},
    "repeat": {"times": 3, "completed": 1},
}

NO_ELIGIBLE_SUB = 'RuntimeError: HTTP 503: {"error":"no eligible sub"}'
POOL_AT_CAPACITY = 'RuntimeError: HTTP 503: {"error":"pool at capacity"}'
RATE_LIMIT = "Error: 429 Too Many Requests (rate limit)"
TIMEOUT = "httpx.ReadTimeout: timed out"
REAL_DEFECT = "KeyError: 'target_skill' in applier note parser"


def _force_toggle(monkeypatch, value):
    """Pin cron.suppress_transient_failure_page to a known value."""
    monkeypatch.setattr(
        sched, "load_config",
        lambda: {"cron": {"suppress_transient_failure_page": value}},
    )


# --- INV-A: classifier ----------------------------------------------------

def test_no_eligible_sub_is_transient():
    assert sched._is_transient_cron_failure(NO_ELIGIBLE_SUB) is True


def test_pool_at_capacity_is_transient():
    assert sched._is_transient_cron_failure(POOL_AT_CAPACITY) is True


def test_all_capped_is_transient():
    assert sched._is_transient_cron_failure('{"error":"all_capped"}') is True


def test_rate_limit_is_transient():
    assert sched._is_transient_cron_failure(RATE_LIMIT) is True


def test_timeout_is_transient():
    assert sched._is_transient_cron_failure(TIMEOUT) is True


def test_real_defect_is_not_transient():
    assert sched._is_transient_cron_failure(REAL_DEFECT) is False


def test_none_and_empty_are_not_transient():
    assert sched._is_transient_cron_failure(None) is False
    assert sched._is_transient_cron_failure("") is False


# --- INV-B: summarizer names the condition --------------------------------

def test_summarizer_names_no_eligible_sub_not_raw_blob():
    msg = sched._summarize_cron_failure_for_delivery(RECURRING_JOB, NO_ELIGIBLE_SUB)
    assert "provider capacity" in msg.lower()
    assert "capped" in msg.lower()
    # the raw provider JSON must not leak into chat
    assert "RuntimeError" not in msg
    assert '{"error"' not in msg
    # job name is surfaced
    assert "skill-patch-applier" in msg


def test_summarizer_names_pool_at_capacity():
    msg = sched._summarize_cron_failure_for_delivery(RECURRING_JOB, POOL_AT_CAPACITY)
    assert "provider capacity" in msg.lower()
    assert "at capacity" in msg.lower()
    assert "RuntimeError" not in msg


def test_summarizer_recurring_capacity_says_retrying():
    msg = sched._summarize_cron_failure_for_delivery(RECURRING_JOB, NO_ELIGIBLE_SUB)
    assert "retrying on the next scheduled tick" in msg.lower()
    assert "will not retry" not in msg.lower()


def test_summarizer_oneshot_capacity_says_no_auto_retry():
    # A one-shot capacity failure is NOT suppressed and does NOT self-heal —
    # the message must not falsely promise a retry (Greptile P1).
    msg = sched._summarize_cron_failure_for_delivery(ONESHOT_JOB, NO_ELIGIBLE_SUB)
    assert "provider capacity" in msg.lower()
    assert "will not retry automatically" in msg.lower()
    assert "retrying on the next scheduled tick" not in msg.lower()


def test_summarizer_bare_quota_gets_quota_framing():
    # A standalone "quota" error (no 429/rate-limit wording) must still get the
    # provider-quota-limit framing, not fall through to generic text (P2).
    msg = sched._summarize_cron_failure_for_delivery(RECURRING_JOB, "QuotaExceededError: over the limit")
    assert "quota limit" in msg.lower()


def test_summarizer_still_handles_real_defect_generically():
    msg = sched._summarize_cron_failure_for_delivery(RECURRING_JOB, REAL_DEFECT)
    assert "provider capacity" not in msg.lower()
    assert "failed" in msg.lower()


# --- recurring detection --------------------------------------------------

def test_recurring_job_detected():
    assert sched._job_is_recurring(RECURRING_JOB) is True


def test_oneshot_job_not_recurring():
    assert sched._job_is_recurring(ONESHOT_JOB) is False


def test_finite_repeat_not_recurring():
    assert sched._job_is_recurring(FINITE_JOB) is False


# --- INV-C: suppression gate ---------------------------------------------

def test_suppress_transient_on_recurring_when_toggle_on(monkeypatch):
    _force_toggle(monkeypatch, True)
    assert sched._should_suppress_transient_failure_page(RECURRING_JOB, NO_ELIGIBLE_SUB) is True


def test_no_suppress_when_toggle_off(monkeypatch):
    _force_toggle(monkeypatch, False)
    assert sched._should_suppress_transient_failure_page(RECURRING_JOB, NO_ELIGIBLE_SUB) is False


def test_no_suppress_real_defect_even_on_recurring(monkeypatch):
    _force_toggle(monkeypatch, True)
    assert sched._should_suppress_transient_failure_page(RECURRING_JOB, REAL_DEFECT) is False


def test_no_suppress_transient_on_oneshot(monkeypatch):
    # A one-shot's only signal is that one delivery — never suppress it.
    _force_toggle(monkeypatch, True)
    assert sched._should_suppress_transient_failure_page(ONESHOT_JOB, NO_ELIGIBLE_SUB) is False


def test_no_suppress_transient_on_finite_repeat(monkeypatch):
    _force_toggle(monkeypatch, True)
    assert sched._should_suppress_transient_failure_page(FINITE_JOB, NO_ELIGIBLE_SUB) is False


def test_default_toggle_is_on(monkeypatch):
    # With no cron config present, the default is suppress-on.
    monkeypatch.setattr(sched, "load_config", lambda: {})
    assert sched._should_suppress_transient_failure_page(RECURRING_JOB, NO_ELIGIBLE_SUB) is True


def test_config_load_failure_defaults_to_suppress(monkeypatch):
    def _boom():
        raise RuntimeError("config unreadable")
    monkeypatch.setattr(sched, "load_config", _boom)
    assert sched._should_suppress_transient_failure_page(RECURRING_JOB, NO_ELIGIBLE_SUB) is True
