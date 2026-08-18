from cron.scheduler import _summarize_cron_failure_for_delivery


def test_scheduler_inactivity_timeout_is_not_reported_as_provider_failure():
    message = _summarize_cron_failure_for_delivery(
        {"name": "daily-review"},
        "TimeoutError: Cron job 'daily-review' idle for 601s (limit 600s) "
        "— last activity: executing tool: terminal",
    )

    assert "agent inactivity timeout" in message
    assert "executing tool: terminal" in message
    assert "provider" not in message
    assert "Fallback chain" not in message


def test_provider_read_timeout_keeps_provider_fallback_summary():
    message = _summarize_cron_failure_for_delivery(
        {"name": "daily-review"},
        "ReadTimeout: upstream request timed out",
    )

    assert "provider timeout" in message
    assert "fallback chain" in message.lower()


def test_unclassified_timeout_is_reported_verbatim_not_as_provider_failure():
    message = _summarize_cron_failure_for_delivery(
        {"name": "daily-review"},
        "TimeoutError: local subprocess exceeded its deadline",
    )

    assert "local subprocess exceeded its deadline" in message
    assert "provider" not in message
    assert "Fallback chain" not in message