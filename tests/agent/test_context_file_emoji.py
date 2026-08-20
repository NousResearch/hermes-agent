"""Regression tests for issue #59492.

The injection scanner (``tools/threat_patterns.scan_for_threats``) used to
flag every U+200D (Zero-Width Joiner) as ``invisible_unicode_U+200D``,
which caused ``agent/prompt_builder._scan_context_content`` to BLOCK the
entire context file (SOUL.md / AGENTS.md / .cursorrules / USER.md /
MEMORY.md) whenever it contained a compound emoji like the family
``U+1F468 U+200D U+1F469 U+200D U+1F467 U+200D U+1F466`` or the
technologist ``U+1F468 U+200D U+1F4BB``.

The fix exempts ZWJ when it sits inside a legitimate emoji sequence
(both neighbours fall inside the documented Extended_Pictographic ranges
from Unicode TR51, optionally with a Variation Selector-16 U+FE0F on
either side).  ZWJ hidden between ASCII / Latin text or next to
non-emoji code points is STILL flagged.

These tests pin:

1. Compound emoji in a context file -> preserved (the regression we are
   fixing).
2. Single emoji -> preserved (no regression for plain emoji).
3. Pure ASCII -> preserved (no regression for normal content).
4. ACTUAL suspicious control characters -> still stripped (no
   regression on the underlying security feature).
"""

from __future__ import annotations

import pytest

from agent.prompt_builder import _scan_context_content


# =========================================================================
# Helpers — build content strings with stable, well-known compound emoji.
# =========================================================================


# Family: man + ZWJ + woman + ZWJ + girl + ZWJ + boy
FAMILY_EMOJI = "👨‍👩‍👧‍👦"
# Rainbow flag (with VS-16): white flag + VS16 + ZWJ + rainbow
RAINBOW_FLAG = "🏳️‍🌈"
# Technologist: man + ZWJ + laptop
TECHNOLOGIST = "👨‍💻"


def _has_zwj_finding(findings: list[str]) -> bool:
    return any(f == "invisible_unicode_U+200D" for f in findings)


# =========================================================================
# 1. The original bug: compound emoji in SOUL.md / context file
# =========================================================================


class TestCompoundEmojiPreserved:
    """The actual user-reported regression — ZWJ inside a compound emoji
    sequence must NOT cause the whole context file to be blocked.
    """

    def test_family_emoji_in_soul_md_preserved(self):
        content = (
            "# My Agent Persona\n\n"
            "- **Family:** " + FAMILY_EMOJI + " (a happy family)\n"
            "- **Mood:** cheerful and supportive\n"
        )
        result = _scan_context_content(content, "SOUL.md")
        assert result == content, (
            "SOUL.md containing the family compound emoji must NOT be "
            "replaced with a [BLOCKED: ...] placeholder. "
            f"Got: {result!r}"
        )

    def test_technologist_emoji_in_agents_md_preserved(self):
        content = (
            "# AGENTS\n\n"
            "Role: " + TECHNOLOGIST + " expert coding assistant.\n"
        )
        result = _scan_context_content(content, "AGENTS.md")
        assert result == content

    def test_rainbow_flag_in_user_md_preserved(self):
        # Flag includes a Variation Selector-16 (U+FE0F) before the ZWJ.
        content = (
            "# About Me\n\n"
            "- **Pride:** " + RAINBOW_FLAG + "\n"
            "- **Role:** senior engineer\n"
        )
        result = _scan_context_content(content, "USER.md")
        assert result == content

    def test_compound_emoji_no_zwj_finding_in_context_scope(self):
        """At the ``context`` scope used by ``_scan_context_content``,
        a legitimate compound emoji must NOT produce the
        ``invisible_unicode_U+200D`` finding that previously blocked
        the file.
        """
        from tools.threat_patterns import scan_for_threats

        findings = scan_for_threats(
            "Persona uses " + FAMILY_EMOJI + " everywhere", scope="context"
        )
        assert not _has_zwj_finding(findings), (
            f"ZWJ inside the family compound emoji must not be flagged. "
            f"Got findings: {findings}"
        )


# =========================================================================
# 2. No regression for plain (non-ZWJ) emoji
# =========================================================================


class TestSingleEmojiPreserved:
    def test_smiley_in_soul_md_preserved(self):
        content = "Hello, I am your assistant 😊\n"
        result = _scan_context_content(content, "SOUL.md")
        assert result == content

    def test_cat_face_emoji_preserved(self):
        content = "I love cats 🐱 (and black cats 🐈‍⬛ too).\n"
        result = _scan_context_content(content, "AGENTS.md")
        # 🐈‍⬛ = cat + ZWJ + black large square. Must pass.
        assert result == content


# =========================================================================
# 3. No regression for plain ASCII
# =========================================================================


class TestPlainAsciiPreserved:
    def test_ascii_soul_md(self):
        content = (
            "# My Agent\n\n"
            "You are a helpful assistant.\n"
            "- Be concise.\n"
            "- Be accurate.\n"
        )
        result = _scan_context_content(content, "SOUL.md")
        assert result == content

    def test_ascii_agents_md_with_code_blocks(self):
        content = (
            "# AGENTS\n\n"
            "## Conventions\n\n"
            "```python\n"
            "def hello() -> str:\n"
            "    return 'world'\n"
            "```\n"
        )
        result = _scan_context_content(content, "AGENTS.md")
        assert result == content


# =========================================================================
# 4. Security feature still works — real suspicious chars must still be
# blocked.  This guards against an over-broad fix that drops ALL
# invisible-unicode detection.
# =========================================================================


class TestSuspiciousCharsStillBlocked:
    """The whole point of the fix is to keep detecting real injection
    vectors.  These tests confirm the scanner still fires on:

    - U+200D hidden between ASCII / Latin text (ZWJ smuggling attack).
    - BiDi override characters (U+202E RLO, U+2066-U+2069 isolates).
    - Zero-width space (U+200B) between arbitrary characters.
    - Mixed content: a legitimate compound emoji alongside a suspicious
      ZWJ smuggling attack in the same file must still be blocked.
    """

    def test_zwj_hidden_between_ascii_still_flagged(self):
        from tools.threat_patterns import scan_for_threats

        findings = scan_for_threats("Hello‍World", scope="context")
        assert _has_zwj_finding(findings)

    def test_zwj_at_start_of_text_still_flagged(self):
        from tools.threat_patterns import scan_for_threats

        # ZWJ at idx 0 has no left neighbour -> cannot be joining emoji.
        findings = scan_for_threats("‍hidden text", scope="context")
        assert _has_zwj_finding(findings)

    def test_zwj_at_end_of_text_still_flagged(self):
        from tools.threat_patterns import scan_for_threats

        # ZWJ at the end has no right neighbour -> cannot be joining emoji.
        findings = scan_for_threats("hidden text‍", scope="context")
        assert _has_zwj_finding(findings)

    def test_rlo_still_detected_at_context_scope(self):
        from tools.threat_patterns import scan_for_threats

        # U+202E RIGHT-TO-LEFT OVERRIDE is a classic file-name spoofing
        # and prompt-hiding vector.  Must still be flagged.
        findings = scan_for_threats("evil ‮ text", scope="context")
        assert "invisible_unicode_U+202E" in findings

    def test_lri_isolate_still_detected(self):
        from tools.threat_patterns import scan_for_threats

        findings = scan_for_threats("attack ⁦ text", scope="context")
        assert "invisible_unicode_U+2066" in findings

    def test_zero_width_space_still_detected(self):
        from tools.threat_patterns import scan_for_threats

        findings = scan_for_threats("zero​width", scope="context")
        assert "invisible_unicode_U+200B" in findings

    def test_mixed_legitimate_and_smuggled_zwj_still_blocked(self):
        """A file that has BOTH a legitimate compound emoji AND a smuggled
        ZWJ between ASCII text must still be blocked.  This is the most
        important security regression guard.
        """
        from tools.threat_patterns import scan_for_threats

        content = (
            "Persona: " + FAMILY_EMOJI + "\n"
            "Smuggled: Hello‍World\n"
        )
        findings = scan_for_threats(content, scope="context")
        assert _has_zwj_finding(findings), (
            "Smuggled ZWJ between ASCII chars must still be flagged even "
            "if the file also contains a legitimate compound emoji."
        )

    def test_blocked_placeholder_returned_for_rlo_context_file(self):
        """End-to-end: a context file with RLO must be replaced with a
        [BLOCKED: ...] placeholder by ``_scan_context_content``.
        """
        content = (
            "# SOUL\n\n"
            "I am your assistant.\n"
            "Hidden direction: ‮ payload\n"
        )
        result = _scan_context_content(content, "SOUL.md")
        assert result.startswith("[BLOCKED:")
        assert "SOUL.md" in result
        assert "invisible_unicode_U+202E" in result


# =========================================================================
# 5. Scope behaviour for ZWJ findings
# =========================================================================


class TestZWJScopeBehaviour:
    """Confirm the ZWJ exemption is consistent across all scanner
    scopes, not just ``context``.  The fix is in the shared invisible-
    char detector, so every scope inherits the behaviour.
    """

    def test_family_emoji_no_zwj_finding_at_all_scope(self):
        from tools.threat_patterns import scan_for_threats

        findings = scan_for_threats(FAMILY_EMOJI, scope="all")
        assert not _has_zwj_finding(findings)

    def test_family_emoji_no_zwj_finding_at_strict_scope(self):
        from tools.threat_patterns import scan_for_threats

        findings = scan_for_threats(FAMILY_EMOJI, scope="strict")
        assert not _has_zwj_finding(findings)

    def test_smuggled_zwj_still_flagged_at_all_scopes(self):
        from tools.threat_patterns import scan_for_threats

        for scope in ("all", "context", "strict"):
            findings = scan_for_threats("a‍b", scope=scope)
            assert _has_zwj_finding(findings), (
                f"Smuggled ZWJ must be flagged at scope={scope!r}; "
                f"got {findings}"
            )


# =========================================================================
# 6. Internal helpers — pin the heuristic at the unit level.
# =========================================================================


class TestZWJHelpers:
    def test_zwj_between_emoji_neighbours_is_legitimate(self):
        from tools.threat_patterns import _zwj_has_emoji_neighbours

        # ZWJ sits between U+1F468 (MAN) and U+1F469 (WOMAN).
        family = FAMILY_EMOJI
        idx = family.index("‍")
        assert _zwj_has_emoji_neighbours(family, idx) is True

    def test_zwj_between_ascii_is_not_legitimate(self):
        from tools.threat_patterns import _zwj_has_emoji_neighbours

        text = "Hello‍World"
        idx = text.index("‍")
        assert _zwj_has_emoji_neighbours(text, idx) is False

    def test_zwj_at_string_start_has_no_left_neighbour(self):
        from tools.threat_patterns import _zwj_has_emoji_neighbours

        text = "‍abc"
        idx = 0
        assert _zwj_has_emoji_neighbours(text, idx) is False

    def test_zwj_at_string_end_has_no_right_neighbour(self):
        from tools.threat_patterns import _zwj_has_emoji_neighbours

        text = "abc‍"
        idx = text.index("‍")
        assert _zwj_has_emoji_neighbours(text, idx) is False

    def test_variation_selector_16_is_skipped(self):
        """U+FE0F (VS-16) before/after the ZWJ should not break the
        emoji-neighbour check.  The rainbow flag has VS-16 on the left.
        """
        from tools.threat_patterns import _zwj_has_emoji_neighbours

        text = RAINBOW_FLAG
        idx = text.index("‍")
        assert _zwj_has_emoji_neighbours(text, idx) is True

    @pytest.mark.parametrize(
        "cp",
        [
            0x1F468,  # MAN (main supplementary block)
            0x2603,   # SNOWMAN (miscellaneous symbols)
            0x2708,   # AIRPLANE (dingbats)
            0x231A,   # WATCH (miscellaneous technical)
            0x1F1FA,  # REGIONAL INDICATOR U (flags)
            0x20E3,   # COMBINING ENCLOSING KEYCAP
        ],
    )
    def test_is_emoji_codepoint_recognises_emoji_ranges(self, cp: int):
        from tools.threat_patterns import _is_emoji_codepoint

        assert _is_emoji_codepoint(cp) is True

    @pytest.mark.parametrize(
        "cp",
        [
            ord("a"),
            ord(" "),
            ord("0"),
            0x00A9,  # COPYRIGHT SIGN (So, but not emoji)
            0x2122,  # TRADE MARK SIGN
            0x202E,  # RLO -- directional override, NOT emoji
            0x200B,  # zero-width space
        ],
    )
    def test_is_emoji_codepoint_rejects_non_emoji(self, cp: int):
        from tools.threat_patterns import _is_emoji_codepoint

        assert _is_emoji_codepoint(cp) is False