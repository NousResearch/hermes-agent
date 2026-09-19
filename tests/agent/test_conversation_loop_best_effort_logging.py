"""Regression coverage for #50459: best-effort failures in the conversation loop
must be visible in the logs.

The handlers under test swallow exceptions on purpose — the call must never break
the turn — but a swallowed failure with no log line is invisible when a user asks
"why did the prompt lack skills last turn?".  These tests pin BOTH halves of the
contract for each site:

  * the behavior is unchanged (same return value, no raised exception), and
  * a DEBUG record naming the failure is emitted with the exception attached.

Sites covered here: ``_bot_chat_prompt_stale`` (agent-home probe),
``_compression_deferred_result`` (status-buffer flush), and
``_decode_inline_moa_turn`` (inline MoA preset decode).  The fourth site —
the skills-cache eviction inside ``_restore_or_build_system_prompt`` — lives in
``test_system_prompt_restore.py`` alongside that function's other silent-failure
coverage.
"""

from __future__ import annotations

import logging
import types

import pytest

from agent.conversation_loop import (
    _bot_chat_prompt_stale,
    _compression_deferred_result,
    _decode_inline_moa_turn,
)


def _raiser(exc: Exception):
    """Build a zero-arg callable that raises ``exc``."""

    def _raise(*args, **kwargs):
        raise exc

    return _raise


def _debug_records(caplog) -> list:
    """DEBUG records emitted by the module under test."""
    return [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and r.name == "agent.conversation_loop"
    ]


def _record_text(record) -> str:
    """A record's message, plus the formatted traceback when a site logs via ``exc_info``.

    Carrying both keeps the exception-visible contract valid if a site migrates from
    ``logger.debug("...: %s", exc)`` to ``logger.debug("...", exc_info=True)``.
    """
    return record.getMessage() + (record.exc_text or "")


# ---------------------------------------------------------------------------
# Site: _bot_chat_prompt_stale — the agent-home probe feeding the capability check
# ---------------------------------------------------------------------------


class TestBotChatPromptStaleHomeProbe:
    def test_agent_home_failure_does_not_propagate_and_logs_debug(self, monkeypatch, caplog):
        """A broken ``_agent_home`` probe must not escape into the caller — and must be visible.

        The probe is best-effort: ``home`` stays ``None``, so the capability comparison falls
        through to ambient home resolution instead of stopping. That fallback can decide a
        rebuild against the wrong profile (see ``_agent_home``'s own docstring), which is why
        the failure has to show up at DEBUG. An unstamped prompt returns ``False`` whatever
        ``home`` is, so this test pins non-propagation and the log record, not that verdict.
        """
        monkeypatch.setattr(
            "agent.system_prompt._agent_home", _raiser(RuntimeError("home probe exploded"))
        )
        agent = types.SimpleNamespace(
            session_id="sess-1",
            _session_title_hint="",
            _session_db=None,
            _bot_mode_protocol=False,
        )

        with caplog.at_level(logging.DEBUG, logger="agent.conversation_loop"):
            stale = _bot_chat_prompt_stale(agent, "STORED PROMPT WITHOUT AN EPOCH STAMP")

        # Behavior unchanged: the probe failure does not propagate, and the documented
        # return value for an unstamped prompt still comes back.
        assert stale is False
        # Visibility contract: the swallowed failure reaches the log at DEBUG, and the
        # record carries the exception plus the context the caller can act on. The
        # assertions are on the exception text and the session id — not on the wording
        # of the message, which is free to change.
        records = _debug_records(caplog)
        assert records, f"Expected a DEBUG record from the module, got: {caplog.text!r}"
        text = " | ".join(_record_text(r) for r in records)
        assert "home probe exploded" in text
        assert "sess-1" in text


# ---------------------------------------------------------------------------
# Site: _compression_deferred_result — the status-buffer flush before returning
# ---------------------------------------------------------------------------


class TestCompressionDeferredResultFlush:
    @pytest.mark.parametrize("reason", ["lock", "transient_block"])
    def test_flush_failure_still_returns_deferred_result_and_logs_debug(self, caplog, reason):
        """The deferred turn result is the contract; a failing flush may not eat it.

        Both reasons share the flush site, so both must log and both must return
        the same ``compression_deferred`` shape.
        """
        agent = types.SimpleNamespace(
            session_id="sess-3",
            _flush_status_buffer=_raiser(RuntimeError("status buffer write failed")),
        )

        with caplog.at_level(logging.DEBUG, logger="agent.conversation_loop"):
            result = _compression_deferred_result(agent, [], 1, reason=reason)

        # Behavior unchanged: the soft deferred result is still produced.
        assert result["compression_deferred"] is True
        assert result["failed"] is False
        assert result["partial"] is True
        # Visibility contract (exception + session context, not message wording).
        records = _debug_records(caplog)
        assert records, f"Expected a DEBUG record from the module, got: {caplog.text!r}"
        text = " | ".join(_record_text(r) for r in records)
        assert "status buffer write failed" in text
        assert "sess-3" in text


# ---------------------------------------------------------------------------
# Site: _decode_inline_moa_turn — an undecodable inline preset is a plain message
# ---------------------------------------------------------------------------


class TestInlineMoaDecodeFailure:
    def test_decode_failure_returns_plain_message_and_logs_debug(self, monkeypatch, caplog):
        """A raising ``decode_moa_turn`` degrades to a plain turn, visibly.

        The preset decode is best-effort: the turn must fall through as a normal
        message, but the decode failure must not vanish.
        """
        monkeypatch.setattr(
            "hermes_cli.moa_config.decode_moa_turn",
            _raiser(RuntimeError("preset payload corrupt")),
        )

        with caplog.at_level(logging.DEBUG, logger="agent.conversation_loop"):
            result = _decode_inline_moa_turn("hello world", "persist me")

        # Behavior unchanged: inputs pass through with moa_config None.
        assert result == ("hello world", None, "persist me")
        # Visibility contract (the exception is attached, not the exact wording).
        records = _debug_records(caplog)
        assert records, f"Expected a DEBUG record from the module, got: {caplog.text!r}"
        assert "preset payload corrupt" in " | ".join(_record_text(r) for r in records)
