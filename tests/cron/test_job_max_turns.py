from cron import scheduler


def test_job_max_turns_overrides_global_limit():
    cfg = {"agent": {"max_turns": 150}}
    assert scheduler._resolve_cron_max_iterations({"max_turns": 20}, cfg) == 20


def test_job_without_override_keeps_global_limit():
    cfg = {"agent": {"max_turns": 150}}
    assert scheduler._resolve_cron_max_iterations({}, cfg) == 150
