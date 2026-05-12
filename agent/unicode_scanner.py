"""Shared invisible-unicode scanner for context files, memory, cron, and skills.

Centralises the blocklist and the ZWJ-aware scan logic so that all scanners
behave consistently.  The key insight is that ZWJ (U+200D) is a **required
component** of emoji grapheme clusters (🧙‍♂️, 👩‍⚕️, 👨‍👩‍👧, 🏳️‍🌈) and should
not be treated as malicious when it appears **between two pictographic
codepoints**.

Public API
----------
find_unsafe_invisibles(content, blocklist)
    Return a list of ``(char, code_point)`` tuples for invisible chars that
    are genuinely suspicious — i.e. not ZWJ inside an emoji grapheme cluster.
"""

import unicodedata
from typing import AbstractSet, List, Tuple

# ── Blocklist ──────────────────────────────────────────────────────────────

# Characters that are invisible / zero-width and could be used for prompt
# injection.  ZWJ (U+200D) is *included* here because it IS dangerous when
# used outside emoji sequences; the scanner below filters out the safe cases.
INVISIBLE_CHARS_BLOCKLIST: frozenset[str] = frozenset({
    '\u200b',  # ZERO WIDTH SPACE
    '\u200c',  # ZERO WIDTH NON-JOINER
    '\u200d',  # ZERO WIDTH JOINER  (safe inside emoji clusters)
    '\u2060',  # WORD JOINER
    '\ufeff',  # BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
    '\u202a',  # LEFT-TO-RIGHT EMBEDDING
    '\u202b',  # RIGHT-TO-LEFT EMBEDDING
    '\u202c',  # POP DIRECTIONAL FORMATTING
    '\u202d',  # LEFT-TO-RIGHT OVERRIDE
    '\u202e',  # RIGHT-TO-LEFT OVERRIDE
})

# ── Pictographic predicate ────────────────────────────────────────────────

# Unicode categories and ranges that count as "pictographic" for the purpose
# of deciding whether a ZWJ between two characters is part of an emoji
# grapheme cluster.
_EMOJI_CATEGORIES = {"So", "Sk"}  # Symbol-other, Modifier-symbol
_EMOJI_RANGES: List[Tuple[int, int]] = [
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
    (0x1F680, 0x1F6FF),  # Transport and Map
    (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    (0x1FA00, 0x1FA6F),  # Chess Symbols
    (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
    (0x2600,  0x26FF),   # Misc Symbols
    (0x2700,  0x27BF),   # Dingbats
    (0xFE00,  0xFE0F),   # Variation Selectors (VS1–VS16)
    (0x200D,  0x200D),   # ZWJ itself (so ZWJ+ZWJ is not "safe")
    (0x20E3,  0x20E3),   # Combining Enclosing Keycap
    (0x00A9,  0x00AE),   # © ®
    (0x203C,  0x3299),   # ‼ through㊙ (misc symbols in BMP)
    (0x1F018, 0x1F270),  # Additional emoji
    (0x2388,  0x2388),   # ⎈
    (0x2B50,  0x2B55),   # ⭐–⭕
    (0x231A,  0x231B),   # ⌚ ⌛
    (0x23E9,  0x23F3),   # ⏩–⏳
    (0x23F8,  0x23FA),   # ⏸–⏺
    (0x25AA,  0x25FE),   # ▪–◾
    (0x2934,  0x2935),   # ⤴ ⤵
    (0x2B05,  0x2B07),   # ⬅⬆⬇
]

ZWJ = '\u200d'


def _is_pictographic(cp: int) -> bool:
    """Return True if *cp* is a pictographic / emoji codepoint."""
    try:
        cat = unicodedata.category(chr(cp))
    except (ValueError, OverflowError):
        return False
    if cat in _EMOJI_CATEGORIES:
        return True
    for lo, hi in _EMOJI_RANGES:
        if lo <= cp <= hi:
            return True
    return False


# ── Public scanner ────────────────────────────────────────────────────────

def find_unsafe_invisibles(
    content: str,
    blocklist: AbstractSet[str] = INVISIBLE_CHARS_BLOCKLIST,
) -> List[Tuple[str, int]]:
    """Return suspicious invisible characters found in *content*.

    Each entry is ``(character, code_point)``.  ZWJ (U+200D) is excluded
    from the result when it sits **between two pictographic codepoints**
    (i.e. it is part of an emoji grapheme cluster like 🧙‍♂️).

    All other invisible characters from *blocklist* are always reported.
    """
    findings: List[Tuple[str, int]] = []
    chars = list(content)

    for idx, ch in enumerate(chars):
        if ch not in blocklist:
            continue

        # ZWJ between two pictographic neighbours is safe — skip it.
        if ch == ZWJ:
            left_ok = (idx > 0 and _is_pictographic(ord(chars[idx - 1])))
            right_ok = (idx + 1 < len(chars) and _is_pictographic(ord(chars[idx + 1])))
            if left_ok and right_ok:
                continue
            # Also allow ZWJ preceded by VS16 (U+FE0F) — flag sequences etc.
            if idx > 0 and ord(chars[idx - 1]) == 0xFE0F and right_ok:
                continue

        findings.append((ch, ord(ch)))

    return findings
