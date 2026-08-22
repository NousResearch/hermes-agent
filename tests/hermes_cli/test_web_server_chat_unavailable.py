"""Readable-error rendering for the dashboard's embedded-chat spawn failures.

Regression guard for the cryptic ``Chat unavailable: 1`` banner: when the
embedded TUI fails to start (most commonly ``SystemExit(1)`` because Node.js
is missing), the dashboard must surface actionable text, not a bare exit code.
"""

import pytest
from fastapi import HTTPException

from hermes_cli.web_server import _chat_unavailable_message


def test_system_exit_one_is_not_a_bare_code_and_mentions_node():
    msg = _chat_unavailable_message(SystemExit(1))
    # The whole point: never leak the raw "1" as the message.
    assert msg != "1"
    assert "Node.js" in msg


@pytest.mark.parametrize("code", [None, 0])
def test_system_exit_none_or_zero_mentions_node(code):
    assert "Node.js" in _chat_unavailable_message(SystemExit(code))


def test_system_exit_other_code_reports_the_code():
    assert "exit code 2" in _chat_unavailable_message(SystemExit(2))


def test_http_exception_uses_its_detail():
    exc = HTTPException(status_code=404, detail="unknown profile 'foo'")
    assert _chat_unavailable_message(exc) == "unknown profile 'foo'"


def test_http_exception_without_detail_falls_back():
    msg = _chat_unavailable_message(HTTPException(status_code=500))
    assert msg  # non-empty, readable
    assert "1" != msg


def test_generic_exception_falls_back_to_str():
    class _Boom(Exception):
        pass

    assert "disk full" in _chat_unavailable_message(_Boom("disk full"))


def test_empty_exception_uses_class_name():
    class _Quiet(Exception):
        def __str__(self) -> str:
            return ""

    assert "_Quiet" in _chat_unavailable_message(_Quiet())
