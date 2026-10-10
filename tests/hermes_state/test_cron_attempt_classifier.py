from hermes_state_common import is_scheduler_finalized_cron


def test_cron_attempt_classifier_requires_scheduler_finalization():
    assert is_scheduler_finalized_cron({"source": "cron", "end_reason": "cron_complete"})
    assert is_scheduler_finalized_cron({"source": "cron", "end_reason": "cron_incomplete_no_output"})
    assert not is_scheduler_finalized_cron({"source": "cron", "end_reason": "ws_orphan_reap"})
    assert not is_scheduler_finalized_cron({"source": "desktop", "end_reason": "cron_complete"})


def test_cron_attempt_classifier_survives_resume_from_durable_marker():
    assert is_scheduler_finalized_cron({
        "source": "cron", "end_reason": None,
        "model_config": '{"_cron_finalized":"cron_incomplete_no_output"}',
    })