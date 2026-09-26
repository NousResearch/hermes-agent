"""Regression: a thinking signature minted on a non-Anthropic Anthropic-Messages
endpoint (Kimi/DeepSeek/MiniMax/etc.) must never be replayed verbatim to
api.anthropic.com after a cross-provider fallback, and the one-shot
thinking-signature recovery must actually repair the shape that carries it.

Root cause (kanban t_d6552433, reproduced live 2026-09-12)
-----------------------------------------------------------
``_manage_thinking_signatures`` treats "is this base_url native Anthropic"
as the only signal for whether to keep a turn's signed thinking blocks. It
never asks WHERE the signature was minted. A session whose primary provider
was kimi-coding (an Anthropic-Messages-wire endpoint) failing over to direct
api.anthropic.com replays kimi-minted signatures on the "latest assistant
turn keeps signed thinking" path and gets a hard 400 ("Invalid signature in
thinking block"), twice, burning restart budget before a healthy rung is
ever tried.

The existing one-shot recovery (``_recover_format_errors`` /
``FailoverReason.thinking_signature``) was also a no-op for this exact shape:
it stripped only ``reasoning_details``, but ``_convert_assistant_message``
prefers the verbatim ``anthropic_content_blocks`` channel when present
(interleaved-thinking replay), so the foreign signature survived the "fix"
and the retry hit the identical 400 again.

Fix under test
--------------
1. ``build_assistant_message`` stamps ``_thinking_signed_base_url`` on any
   Anthropic-wire assistant turn that carries a signature.
2. ``_manage_thinking_signatures`` demotes (to plain text) any latest-turn
   thinking block whose stamped origin doesn't match the endpoint we're
   converting FOR right now — the same demotion path already used for an
   orphan-stripped/invalidated turn.
3. ``_recover_format_errors`` strips ``anthropic_content_blocks`` (not just
   ``reasoning_details``) so the one-shot recovery repairs turns using the
   ordered-block replay fast path too.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.transports import get_transport

SIG = "sig-minted-on-kimi"


def _kimi_signed_tool_turn():
    """An assistant turn with signed thinking + tool_use, as minted by a
    Kimi-coding (Anthropic-Messages-wire) primary provider."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="plan the read", signature=SIG),
            SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
        ],
        stop_reason="tool_use",
        usage=None,
    )


def _stored_message_from_kimi(signed_base_url: str = "https://api.kimi.com/coding") -> dict:
    """Reproduce exactly what build_assistant_message stores for a kimi-coding turn:
    reasoning_details + anthropic_content_blocks + the new provenance stamp."""
    response = _kimi_signed_tool_turn()
    normalized = get_transport("anthropic_messages").normalize_response(response)
    provider_data = normalized.provider_data or {}
    msg = {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": provider_data.get("reasoning_details"),
        "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in (normalized.tool_calls or [])
        ],
        # Stamped by build_assistant_message at write time (agent.api_mode ==
        # "anthropic_messages" and agent._anthropic_base_url == the kimi endpoint).
        "_thinking_signed_base_url": signed_base_url,
    }
    if provider_data.get("anthropic_content_blocks"):
        msg["anthropic_content_blocks"] = provider_data["anthropic_content_blocks"]
    return msg


def _thinking_blocks(assistant_content):
    return [b for b in assistant_content if isinstance(b, dict) and b.get("type") == "thinking"]


class TestForeignSignatureDemotedOnFallback:
    def test_kimi_signature_replayed_to_anthropic_is_demoted_not_kept(self):
        """The invariant: a signature minted on a non-Anthropic endpoint must never reach
        api.anthropic.com intact. Converting FOR direct Anthropic (base_url=None) must
        strip the foreign signature and demote the block to plain text."""
        assistant_msg = _stored_message_from_kimi()
        messages = [
            {"role": "user", "content": "inspect a.py"},
            assistant_msg,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
        ]
        _sys, out = convert_messages_to_anthropic(messages, base_url=None, model="claude-sonnet-5")
        assistant_out = [m for m in out if m["role"] == "assistant"][-1]
        thinking = _thinking_blocks(assistant_out["content"])
        assert not any(b.get("signature") == SIG for b in thinking), (
            "a Kimi-minted thinking signature was replayed verbatim to direct Anthropic — "
            f"this is the exact shape that produces HTTP 400 'Invalid signature in thinking "
            f"block': {thinking}"
        )
        # The reasoning content itself is preserved as plain text (not silently dropped).
        assert any(b.get("type") == "text" and "plan the read" in b.get("text", "") for b in assistant_out["content"])

    def test_same_endpoint_signature_survives_replay(self):
        """Control: a signature minted on the SAME endpoint we're converting for must
        still be kept — this is the normal, working case and must not regress."""
        assistant_msg = _stored_message_from_kimi(signed_base_url="https://api.kimi.com/coding")
        messages = [
            {"role": "user", "content": "inspect a.py"},
            assistant_msg,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
        ]
        _sys, out = convert_messages_to_anthropic(
            messages, base_url="https://api.kimi.com/coding", model="kimi-k2.5",
        )
        assistant_out = [m for m in out if m["role"] == "assistant"][-1]
        # Kimi does not enforce signatures (is_kimi path passes through as-is) — the
        # signed block must survive untouched when replayed to the endpoint that minted it.
        thinking = _thinking_blocks(assistant_out["content"])
        assert any(b.get("signature") == SIG for b in thinking)

    def test_unstamped_legacy_turn_is_not_treated_as_foreign(self):
        """Absence of the stamp (pre-fix data, or a non-Anthropic-wire provider) must not
        be misread as 'foreign' — only an explicit mismatch demotes. Regression guard for
        the null-signal case."""
        assistant_msg = _stored_message_from_kimi()
        del assistant_msg["_thinking_signed_base_url"]
        messages = [
            {"role": "user", "content": "inspect a.py"},
            assistant_msg,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
        ]
        _sys, out = convert_messages_to_anthropic(messages, base_url=None, model="claude-sonnet-5")
        assistant_out = [m for m in out if m["role"] == "assistant"][-1]
        thinking = _thinking_blocks(assistant_out["content"])
        assert any(b.get("signature") == SIG for b in thinking), (
            "an unstamped (legacy) turn was demoted even though no origin mismatch was proven"
        )

    def test_internal_stamp_never_reaches_the_wire(self):
        """The provenance flag is bookkeeping only — it must never survive conversion,
        on ANY endpoint (mirrors the existing _thinking_signature_invalidated leak guard)."""
        assistant_msg = _stored_message_from_kimi()
        messages = [
            {"role": "user", "content": "inspect a.py"},
            assistant_msg,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
        ]
        for base_url, model in ((None, "claude-sonnet-5"), ("https://api.kimi.com/coding", "kimi-k2.5")):
            _sys, out = convert_messages_to_anthropic(messages := [dict(m) for m in messages], base_url=base_url, model=model)
            assistant_out = [m for m in out if m["role"] == "assistant"][-1]
            assert "_thinking_signed_base_url" not in assistant_out, (base_url, assistant_out.keys())


class TestRecoverFormatErrorsStripsOrderedBlockChannel:
    """``_recover_format_errors`` must repair BOTH channels a signature can ride in, or the
    one-shot recovery is a no-op for the ordered-block replay shape and the retry 400s again."""

    def test_strips_anthropic_content_blocks_not_just_reasoning_details(self):
        from agent.turn_recovery import _recover_format_errors
        from agent.turn_retry_state import TurnRetryState
        from agent.error_classifier import FailoverReason

        assistant_msg = _stored_message_from_kimi()
        assert "anthropic_content_blocks" in assistant_msg, "fixture must exercise the ordered-block channel"
        api_messages = [
            {"role": "user", "content": "inspect a.py"},
            dict(assistant_msg),
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
        ]
        agent = SimpleNamespace(log_prefix="", quiet_mode=True, verbose_mode=False, _vprint=lambda *a, **k: None)
        classified = SimpleNamespace(reason=FailoverReason.thinking_signature)
        retry = TurnRetryState()

        recovered = _recover_format_errors(
            agent, RuntimeError("Invalid signature in thinking block"), classified, retry,
            messages=[], api_messages=api_messages,
        )
        assert recovered is True

        repaired_assistant = next(m for m in api_messages if m.get("role") == "assistant")
        assert "reasoning_details" not in repaired_assistant, "reasoning_details survived the strip"
        assert "anthropic_content_blocks" not in repaired_assistant, (
            "anthropic_content_blocks survived the strip — this is the exact shape "
            "documented as a no-op recovery in kanban t_d6552433: "
            "_convert_assistant_message prefers this channel over reasoning_details, "
            "so a foreign signature stripped only from reasoning_details still rides "
            "the wire and the retry gets the identical 400."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
