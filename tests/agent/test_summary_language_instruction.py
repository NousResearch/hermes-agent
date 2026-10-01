"""Tests for the summarizer language instruction (fix/summary-language-user-turns).

The drift incident (issue #130117): with a mixed-language transcript, the old
instruction ("same language the user was using in the conversation") let the
summarizer read the assistant's drifted Chinese as the conversation language, so
every compaction summary re-seeded Chinese into the context. The fixed instruction
must:
- derive summary language from user-authored turns only,
- explicitly instruct that assistant-message language is never a precedent,
- keep the byte-pinned template structure (plain wording, no injection framing).
"""
from __future__ import annotations

import re

from agent.context_compressor import _SECTION_INSTRUCTIONS


def test_user_turn_language_instruction_keys_on_user_messages():
    text = _SECTION_INSTRUCTIONS[True]["language"]
    assert "user" in text.lower()
    # must tell the model to derive language from user messages specifically
    assert re.search(r"user.{0,40}(messages|turns)", text, re.IGNORECASE), (
        "language instruction must point at user-authored turns as the language source"
    )


def test_assistant_language_is_disclaimed_as_non_precedent():
    text = _SECTION_INSTRUCTIONS[True]["language"]
    assert "assistant" in text.lower(), (
        "fixed instruction must explicitly exclude assistant-message language"
    )
    # the failure mode being fixed: assistant drift must be called out as not precedent
    assert re.search(r"(ignore|never|not a precedent|noise)", text, re.IGNORECASE)


def test_no_user_turn_branch_disfavors_mixed_assistant_language():
    text = _SECTION_INSTRUCTIONS[False]["language"]
    # the old text made "the most recent assistant turn" authoritative — the exact
    # mechanism that let one drifted reply own the summary language forever.
    assert "most recent" not in text.lower() or "exclude" in text.lower()


def test_templates_remain_plain_wording():
    # byte-pinned template contract: plain sentences, no injection-style framing.
    for branch in (True, False):
        text = _SECTION_INSTRUCTIONS[branch]["language"]
        assert "do not respond" not in text.lower()
        assert "ignore previous" not in text.lower()
        assert len(text) < 700, "language instruction must stay lean"
