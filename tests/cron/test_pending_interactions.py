from pathlib import Path

from cron.scheduler import _record_cron_pending_interactions
from gateway.pending_interactions import STATUS_OPEN, load_pending_interactions


def test_cron_to_discord_follow_up_records_pending_interaction():
    output_file = Path("/tmp/hermes-cron-output.md")
    job = {
        "id": "daily-brief",
        "deliver": "discord:111111111111111111:222222222222222222",
    }

    records = _record_cron_pending_interactions(
        job,
        "Cron found two options. Which one should I continue?",
        output_file,
    )

    assert len(records) == 1
    record = records[0]
    assert record["status"] == STATUS_OPEN
    assert record["platform"] == "discord"
    assert record["channel_id"] == "111111111111111111"
    assert record["thread_id"] == "222222222222222222"
    assert record["job_id"] == "daily-brief"
    assert record["source_session_id"] is None
    assert record["artifact_paths"] == [str(output_file)]
    assert load_pending_interactions()[0]["id"] == record["id"]


def test_cron_non_question_response_does_not_record_pending_interaction():
    job = {
        "id": "daily-brief",
        "deliver": "discord:111111111111111111:222222222222222222",
    }

    records = _record_cron_pending_interactions(job, "Cron completed successfully.", None)

    assert records == []
    assert load_pending_interactions() == []
