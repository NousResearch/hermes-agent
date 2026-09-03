"""Regression tests for bytes input to strip_think_blocks (#66267 follow-up).

PR #66567 added coercion for list/dict/non-str content.  One shape was left
unhandled: ``bytes``.  ``str(b"hello")`` produces the literal string
``"b'hello'"`` rather than decoding the bytes, which leaks the repr into
assistant responses.  This file covers only the bytes gap so there is no
overlap with the tests added in #66567.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agent.agent_runtime_helpers import strip_think_blocks


@pytest.fixture()
def agent():
    return MagicMock()


class TestBytesInput:
    def test_plain_bytes_decoded(self, agent):
        assert strip_think_blocks(agent, b"hello world") == "hello world"

    def test_bytes_think_block_stripped(self, agent):
        result = strip_think_blocks(agent, b"<think>secret</think>visible")
        assert "secret" not in result
        assert "visible" in result

    def test_empty_bytes_returns_empty(self, agent):
        assert strip_think_blocks(agent, b"") == ""

    def test_bytes_non_ascii_decoded(self, agent):
        result = strip_think_blocks(agent, "em\u2014dash".encode("utf-8"))
        assert "\u2014" in result

    def test_bytes_invalid_utf8_uses_replacement(self, agent):
        # Must not raise; replacement char used for undecodable bytes
        result = strip_think_blocks(agent, b"ok\xff\xfedone")
        assert "ok" in result
        assert "done" in result

    def test_bytes_not_repr_leaked(self, agent):
        # The old str() path turned b"hi" into "b'hi'" — verify that is gone
        result = strip_think_blocks(agent, b"hi")
        assert result == "hi"
        assert "b'" not in result
