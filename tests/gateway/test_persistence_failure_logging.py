"""A session-persistence failure must be logged, not just replaced by the storage notice.

The persistence branch in ``_normalize_empty_agent_response`` returns before the
sanitized-failure log line, so before this the user was told "session storage
unavailable" and nothing was written to any log — the one failure nobody could
trace afterwards.
"""
import logging

from gateway.run import _normalize_empty_agent_response


def _persistence_failure(**extra):
    return {
        "failed": True,
        "error": "session storage unavailable: database is locked",
        "failure_reason": "session_persistence_failed",
        **extra,
    }


def test_persistence_failure_is_logged(caplog):
    with caplog.at_level(logging.WARNING):
        reply = _normalize_empty_agent_response(_persistence_failure(), "")

    assert "storage" in reply.lower()
    records = [r for r in caplog.records if "session persistence" in r.getMessage()]
    assert records, "persistence failure produced no log record"
    assert records[0].levelno == logging.WARNING
    # The reason travels with the record; raw provider text stays out of the reply.
    assert "session_persistence_failed" in records[0].getMessage()


def test_disk_variant_is_logged_and_distinguished(caplog):
    with caplog.at_level(logging.WARNING):
        reply = _normalize_empty_agent_response(
            {
                "failed": True,
                "error": "session storage unavailable: disk",
                "failure_reason": "session_persistence_failed:disk",
            },
            "",
        )

    assert reply
    assert any("session persistence" in r.getMessage() for r in caplog.records)
