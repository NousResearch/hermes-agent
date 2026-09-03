"""Unit tests for gateway.inbound_context — the bounded reply-context owner (#101866).

The reviewer's acceptance list requires the behavioral tests to live with
the owner module; test_queued_reply_to_prefix.py keeps the integration
assertions (idle + queued paths consume the owner).
"""

from types import SimpleNamespace

import pytest

from gateway.inbound_context import (
    apply_reply_to_prefix,
    build_reply_to_prefix,
    log_inbound_reply_context,
)


def _event(reply_to_text=None, reply_to_id=None, own=True, text="msg"):
    return SimpleNamespace(
        text=text,
        reply_to_text=reply_to_text,
        reply_to_message_id=reply_to_id,
        reply_to_is_own_message=own,
    )


class TestBuildReplyToPrefix:
    def test_own_message_form(self):
        prefix = build_reply_to_prefix(
            _event(reply_to_text="Draft A", reply_to_id="a")
        )
        assert prefix == '[Replying to your previous message: "Draft A"]\n\n'

    def test_foreign_form(self):
        prefix = build_reply_to_prefix(
            _event(reply_to_text="Bob's note", reply_to_id="m1", own=False)
        )
        assert prefix == "[Replying to: \"Bob's note\"]\n\n"

    def test_no_context_no_prefix(self):
        assert build_reply_to_prefix(_event()) == ""

    def test_text_without_id_no_prefix(self):
        assert build_reply_to_prefix(
            _event(reply_to_text="orphan", reply_to_id=None)
        ) == ""

    def test_id_without_text_no_prefix(self):
        assert build_reply_to_prefix(
            _event(reply_to_text=None, reply_to_id="m1")
        ) == ""

    def test_snippet_truncated_to_500(self):
        prefix = build_reply_to_prefix(
            _event(reply_to_text="x" * 900, reply_to_id="m1")
        )
        assert "x" * 500 in prefix
        assert "x" * 501 not in prefix


class TestApplyReplyToPrefix:
    def test_prefix_prepended(self):
        out = apply_reply_to_prefix(
            "yes, send the same",
            _event(reply_to_text="Draft A", reply_to_id="a"),
        )
        assert out.startswith('[Replying to your previous message: "Draft A"]')
        assert out.endswith("yes, send the same")

    def test_no_context_passthrough(self):
        assert apply_reply_to_prefix("plain", _event()) == "plain"


class TestLogInboundReplyContext:
    def test_log_line_carries_reply_identity_and_queued_marker(self, caplog):
        import logging

        src = SimpleNamespace(
            platform=None, user_id="u1", chat_id="c1"
        )
        ev = _event(reply_to_text="Draft A", reply_to_id="a")
        with caplog.at_level(logging.INFO, logger="gateway.inbound_context"):
            log_inbound_reply_context(
                source=src, message_text="yes", event=ev, queued=True
            )
        assert any(
            "reply_to_id=a" in r.getMessage() and "queued=True" in r.getMessage()
            for r in caplog.records
        )

    def test_event_optional(self, caplog):
        import logging

        src = SimpleNamespace(platform=None, user_id="u", chat_id="c")
        with caplog.at_level(logging.INFO, logger="gateway.inbound_context"):
            log_inbound_reply_context(source=src, message_text="m")
        assert any(
            "queued=False" in r.getMessage() for r in caplog.records
        )
