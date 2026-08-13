"""_summarize_cron_failure_for_delivery must not mislabel the scheduler's own
inactivity-timeout abort as a provider/fallback-chain failure, and must not
claim a fallback chain was "exhausted" when none is configured.

Regression for t_29b8da55 (2026-08-08, Daily Repo Sweep): a stuck `terminal`
tool call tripped the 600s cron inactivity watchdog. The TimeoutError raised
by the watchdog contains the substring "limit 600s" and its message reads
"idle for 1239s (limit 600s)" -- no provider or fallback chain was ever
involved -- but the old branch order matched the generic "timed out"/"timeout"
substring check before any inactivity-specific check existed, so the operator
saw "provider timeout. Fallback chain was exhausted or unavailable." for a
failure that had nothing to do with either.

Second bug bundled into the same task: even on a *genuine* provider failure,
"Fallback chain was exhausted or unavailable." fired unconditionally --
regardless of whether fallback_providers was ever configured. Both the root
and cto profile config.yaml have fallback_providers: [] (confirmed
2026-08-08), so the message always implied an attempted-and-failed fallback
that never existed. _fallback_chain_phrase() now checks the effective chain
via get_fallback_chain() and reports "No fallback chain configured." when
it's empty.
"""

import cron.scheduler as scheduler
from cron.scheduler import _summarize_cron_failure_for_delivery


def test_inactivity_timeout_is_not_reported_as_provider_timeout():
    job = {"name": "Daily Repo Sweep", "id": "82d65bdd5ba9"}
    error = (
        "TimeoutError: Cron job 'Daily Repo Sweep' idle for 1239s "
        "(limit 600s) — last activity: terminal command running (30s elapsed)"
    )
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider timeout" not in msg
    assert "fallback chain" not in msg.lower()
    assert "stalled" in msg.lower()
    assert "Daily Repo Sweep" in msg


def test_genuine_provider_timeout_with_no_fallback_configured(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "CI Autofix Poller", "id": "f7fe78574bda"}
    error = "Request timed out."
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider timeout" in msg
    assert "No fallback chain configured" in msg
    assert "exhausted or unavailable" not in msg


def test_genuine_provider_timeout_with_fallback_configured(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {
        "fallback_providers": [{"provider": "openrouter", "model": "anthropic/claude-sonnet-5"}]
    })
    monkeypatch.setattr(
        scheduler,
        "get_fallback_chain",
        lambda cfg: [{"provider": "openrouter", "model": "anthropic/claude-sonnet-5"}],
    )
    job = {"name": "CI Autofix Poller", "id": "f7fe78574bda"}
    error = "Request timed out."
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider timeout" in msg
    assert "Fallback chain was exhausted or unavailable." in msg
    assert "No fallback chain configured" not in msg


def test_fallback_chain_phrase_fails_open_on_config_error(monkeypatch):
    def _raise():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(scheduler, "load_config", _raise)
    assert scheduler._fallback_chain_phrase() == "Fallback chain was exhausted or unavailable."


def test_readtimeout_error_still_classified_as_provider_timeout(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "some-job", "id": "abc123"}
    error = "httpx.ReadTimeout: The read operation timed out"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider timeout" in msg


def test_rate_limit_classification_still_takes_priority_over_inactivity_text(monkeypatch):
    # A rate-limit error mentioning "usage limit" must still classify as a
    # rate limit even though it could theoretically also contain "timeout"-
    # adjacent wording; rate-limit check runs first and should be unaffected
    # by the new inactivity branch inserted after it.
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "some-job", "id": "abc123"}
    error = "HTTP 429: weekly usage limit exceeded"
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "weekly usage limit" in msg
    assert "No fallback chain configured" in msg
