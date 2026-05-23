from datetime import datetime, timedelta

from hermes_cli.runtime_safety import scan_resume_pending_sessions


def test_naive_resume_pending_timestamp_is_local_time():
    now = datetime.now().astimezone()
    marked = (datetime.now() - timedelta(seconds=900)).isoformat()

    stale = scan_resume_pending_sessions(
        [{
            "session_key": "agent:main:discord:thread:1",
            "session_id": "s1",
            "platform": "discord",
            "resume_pending": True,
            "last_resume_marked_at": marked,
        }],
        now=now,
        ttl_seconds=600,
    )

    assert stale
    assert stale[0]["reason"] == "stale_resume_pending"
