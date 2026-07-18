import pytest

from gateway.config import Platform
from gateway.run import _require_isolated_handoff_thread


def test_new_handoff_thread_is_used():
    assert (
        _require_isolated_handoff_thread(
            platform=Platform.TELEGRAM,
            chat_id="8650058832",
            effective_thread_id="topic-77",
        )
        == "topic-77"
    )


def test_explicitly_configured_home_thread_is_allowed():
    assert (
        _require_isolated_handoff_thread(
            platform=Platform.TELEGRAM,
            chat_id="8650058832",
            effective_thread_id="88",
        )
        == "88"
    )


def test_handoff_fails_closed_when_no_distinct_thread_exists():
    with pytest.raises(RuntimeError, match="distinct thread/topic"):
        _require_isolated_handoff_thread(
            platform=Platform.TELEGRAM,
            chat_id="8650058832",
            effective_thread_id=None,
        )
