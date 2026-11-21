"""The ``inbox`` deliver target passes the scheduler's delivery preflight: it needs no adapter, so a
job filed into the Desktop/Web Inbox must not be blocked as an unknown platform before it runs."""
from cron.scheduler_preflight import _preflight_check_delivery


def test_preflight_lets_an_inbox_job_run():
    assert _preflight_check_delivery({"id": "j1", "deliver": "inbox"}) is None
    assert _preflight_check_delivery({"id": "j1", "deliver": "local,inbox", "failure_deliver": "inbox"}) is None
