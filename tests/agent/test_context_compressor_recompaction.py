"""Regression test for issue #17344: recompaction must not keep re-anchoring
the original first user exchange across every cycle.

Pre-fix, ``protect_first_n=3`` preserved ``[system, user1, assistant1]``
on every compaction. After 6 compactions the head still contained the
*original* user1 \u2014 and the model, presented with a stale user-role
message right next to the handoff summary, would re-execute it as if it
were active.

Fix: when the system prompt already carries the compaction note from a
previous run, ``protect_first_n`` is shrunk to 1 (system only) for that
compaction, so the stale exchange flows into the summariser pool where
the structured ``## Active Task`` section replaces it as the steering
signal.
"""

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _COMPRESSION_NOTE_SENTINEL,
)


def _make_compressor(protect_first_n: int = 3):
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=protect_first_n,
            protect_last_n=2,
            quiet_mode=True,
        )


class TestIsRecompaction:
    """Sentinel detection: cheap, side-effect-free, and conservative."""

    def test_fresh_system_prompt_is_not_recompaction(self):
        c = _make_compressor()
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Original first request"},
            {"role": "assistant", "content": "Sure."},
        ]
        assert c._is_recompaction(msgs) is False

    def test_system_prompt_with_compaction_note_is_recompaction(self):
        c = _make_compressor()
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant.\n\n"
                    "[Note: Some earlier conversation turns have been "
                    "compacted into a handoff summary to preserve context "
                    "space.]"
                ),
            },
            {"role": "user", "content": "u1"},
        ]
        assert c._is_recompaction(msgs) is True

    def test_empty_messages_safe(self):
        c = _make_compressor()
        assert c._is_recompaction([]) is False

    def test_non_system_first_message_is_not_recompaction(self):
        c = _make_compressor()
        msgs = [
            {"role": "user", "content": _COMPRESSION_NOTE_SENTINEL},
        ]
        assert c._is_recompaction(msgs) is False

    def test_multimodal_system_content_is_inspected(self):
        c = _make_compressor()
        msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {
                        "type": "text",
                        "text": (
                            "[Note: Some earlier conversation turns have "
                            "been compacted into a handoff summary.]"
                        ),
                    },
                ],
            },
        ]
        assert c._is_recompaction(msgs) is True

    def test_garbage_content_does_not_raise(self):
        c = _make_compressor()
        msgs = [{"role": "system", "content": object()}]
        # Must not raise; should default to False.
        assert c._is_recompaction(msgs) is False


class TestRecompactionShrinksProtectFirstN:
    """Behaviour test: on recompaction the head shrinks to {system}, so the
    stale first exchange is summarised instead of re-anchored."""

    def _build_session(self, n_middle: int = 30, recompacted: bool = False):
        system_text = "You are a helpful assistant."
        if recompacted:
            system_text += (
                "\n\n[Note: Some earlier conversation turns have been "
                "compacted into a handoff summary to preserve context "
                "space.]"
            )
        msgs = [{"role": "system", "content": system_text}]
        # First exchange: this is what we want demoted on recompaction.
        msgs.append({"role": "user", "content": "ORIGINAL_FIRST_REQUEST"})
        msgs.append({"role": "assistant", "content": "Sure, started on it."})
        for i in range(n_middle):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"middle-msg-{i} " * 80})
        # Latest user message anchors the tail.
        msgs.append({"role": "user", "content": "LATEST_USER_REQUEST"})
        return msgs

    def test_first_compaction_preserves_first_exchange_in_head(self):
        c = _make_compressor(protect_first_n=3)
        msgs = self._build_session(recompacted=False)
        with patch.object(c, "_generate_summary", return_value="SUMMARY_BODY"):
            out = c.compress(msgs, current_tokens=80_000)
        # Head: [system, user(ORIGINAL_FIRST_REQUEST), assistant(...)]
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"
        assert "ORIGINAL_FIRST_REQUEST" in out[1]["content"]
        assert out[2]["role"] == "assistant"

    def test_recompaction_demotes_first_exchange_to_summary(self):
        c = _make_compressor(protect_first_n=3)
        msgs = self._build_session(recompacted=True)
        with patch.object(c, "_generate_summary", return_value="SUMMARY_BODY"):
            out = c.compress(msgs, current_tokens=80_000)
        # Head: [system] only \u2014 the original first exchange must NOT be
        # sitting at messages[1] anymore.
        assert out[0]["role"] == "system"
        # No preserved-head user message carrying ORIGINAL_FIRST_REQUEST.
        head_user_texts = [
            (m.get("content") if isinstance(m.get("content"), str) else "")
            for m in out[:2]
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert all(
            "ORIGINAL_FIRST_REQUEST" not in (t or "") for t in head_user_texts
        ), (
            "Recompaction must drop the stale first user exchange from the "
            "preserved head; otherwise the model keeps re-executing it. "
            f"head_user_texts={head_user_texts}"
        )

    def test_recompaction_preserves_latest_user_message_in_tail(self):
        c = _make_compressor(protect_first_n=3)
        msgs = self._build_session(recompacted=True)
        with patch.object(c, "_generate_summary", return_value="SUMMARY_BODY"):
            out = c.compress(msgs, current_tokens=80_000)
        # Latest user request must still be reachable (not summarised away).
        last_text = (
            out[-1].get("content")
            if isinstance(out[-1].get("content"), str)
            else ""
        )
        # The TODO snapshot may append an extra user message; scan the tail.
        tail_text = "".join(
            (m.get("content") or "") if isinstance(m.get("content"), str) else ""
            for m in out[-5:]
            if isinstance(m, dict)
        )
        assert "LATEST_USER_REQUEST" in tail_text

    def test_recompaction_keeps_protect_first_n_attribute_unchanged(self):
        # The shrink is per-call; the configured value must not mutate so
        # subsequent fresh sessions still get the default head.
        c = _make_compressor(protect_first_n=3)
        msgs = self._build_session(recompacted=True)
        with patch.object(c, "_generate_summary", return_value="SUMMARY_BODY"):
            c.compress(msgs, current_tokens=80_000)
        assert c.protect_first_n == 3

    def test_protect_first_n_one_no_op_for_recompaction(self):
        # When the compressor is already configured with protect_first_n=1
        # the shrink branch should be a no-op (no further shrinking needed).
        c = _make_compressor(protect_first_n=1)
        msgs = self._build_session(recompacted=True)
        with patch.object(c, "_generate_summary", return_value="SUMMARY_BODY"):
            out = c.compress(msgs, current_tokens=80_000)
        # Still produces a valid compressed transcript.
        assert out[0]["role"] == "system"
        assert len(out) >= 3  # system + summary + at least one tail message


class TestRecompactionMinForCompressGate:
    """``_min_for_compress`` early-return must scale with the *effective*
    head size on recompaction, otherwise small recompacted sessions would
    skip compression entirely."""

    def test_recompaction_can_compress_small_session(self):
        c = _make_compressor(protect_first_n=3)
        # 1 system + 2 first-exchange + 4 middle + 1 latest = 8 messages.
        # Default protect_first_n=3 \u2192 _min_for_compress = 7, which is < 8 so
        # compression DOES run, but the effective shrink to 1 makes
        # _min_for_compress = 5, leaving more room to summarise.
        system_text = (
            "You are a helpful assistant.\n\n"
            "[Note: Some earlier conversation turns have been compacted "
            "into a handoff summary.]"
        )
        msgs = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "ORIGINAL"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "m1 " * 200},
            {"role": "assistant", "content": "m2 " * 200},
            {"role": "user", "content": "m3 " * 200},
            {"role": "assistant", "content": "m4 " * 200},
            {"role": "user", "content": "LATEST"},
        ]
        with patch.object(c, "_generate_summary", return_value="SUMMARY_BODY"):
            out = c.compress(msgs, current_tokens=80_000)
        # ORIGINAL must be summarised away (not in preserved head).
        head = out[:2]
        for m in head:
            text = m.get("content") if isinstance(m.get("content"), str) else ""
            assert "ORIGINAL" not in text
