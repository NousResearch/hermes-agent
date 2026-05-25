"""Regression tests for safe agent-result access in the queued-message drain path.

Bug: #25152 — ``/queue`` crashes with ``AttributeError: 'SendResult' object has
no attribute 'get'``.  The queued follow-up drain path in ``gateway/run.py``
called ``result.get("messages", ...)`` on what could be a ``SendResult`` delivery
object rather than the expected dict.  The fix introduces two helpers —
``_agent_result_get`` (safe ``.get()`` for any result type) and
``_messages_from_agent_result`` (safe history extraction) — and replaces all
bare ``.get()`` calls in that path.
"""

from gateway.platforms.base import SendResult
from gateway.run import _agent_result_get, _messages_from_agent_result


# ---------------------------------------------------------------------------
# _agent_result_get
# ---------------------------------------------------------------------------

def test_agent_result_get_reads_dict_values():
    assert _agent_result_get({"interrupted": True}, "interrupted") is True
    assert _agent_result_get({"final_response": "hello"}, "final_response") == "hello"


def test_agent_result_get_returns_default_for_non_dict():
    """SendResult (or any non-dict) must not crash — return the default."""
    delivery = SendResult(success=True, message_id="tg-1")
    assert _agent_result_get(delivery, "interrupted") is None
    assert _agent_result_get(delivery, "final_response", "") == ""


def test_agent_result_get_returns_default_for_none():
    assert _agent_result_get(None, "failed", False) is False


# ---------------------------------------------------------------------------
# _messages_from_agent_result
# ---------------------------------------------------------------------------

def test_messages_from_agent_result_returns_dict_messages():
    messages = [{"role": "assistant", "content": "new"}]
    fallback = [{"role": "user", "content": "old"}]

    assert _messages_from_agent_result({"messages": messages}, fallback) is messages


def test_messages_from_agent_result_empty_messages_list_is_valid():
    """An empty messages list is a legitimate conversation history."""
    fallback = [{"role": "user", "content": "old"}]

    result = _messages_from_agent_result({"messages": []}, fallback)
    assert result == []


def test_messages_from_agent_result_falls_back_for_send_result():
    fallback = [{"role": "user", "content": "queued"}]
    delivery = SendResult(success=True, message_id="tg-1")

    assert _messages_from_agent_result(delivery, fallback) is fallback


def test_messages_from_agent_result_falls_back_for_failed_send_result():
    """A failed delivery result is still a non-dict — must not crash."""
    fallback = [{"role": "user", "content": "queued"}]
    delivery = SendResult(success=False, error="timeout")

    assert _messages_from_agent_result(delivery, fallback) is fallback


def test_messages_from_agent_result_falls_back_for_none_result():
    fallback = [{"role": "user", "content": "queued"}]

    assert _messages_from_agent_result(None, fallback) is fallback


def test_messages_from_agent_result_falls_back_when_messages_missing_or_invalid():
    fallback = [{"role": "user", "content": "queued"}]

    assert _messages_from_agent_result({}, fallback) is fallback
    assert _messages_from_agent_result({"messages": None}, fallback) is fallback
    assert _messages_from_agent_result({"messages": "not-a-list"}, fallback) is fallback
