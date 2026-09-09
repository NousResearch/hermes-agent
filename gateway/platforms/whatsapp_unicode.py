"""Unicode "font style" mapper for recovering rich formatting in WhatsApp.

WhatsApp's native text-formatting dialect is tiny (bold, italic,
strikethrough, inline code, fenced code), so the CommonMark renderer
down-converts richer constructs (e.g. heading *levels*, compound
bold-italic) to something flat. When enabled, we reclaim some of that
richness with Unicode Mathematical Alphanumeric Symbols (U+1D400–U+1D7FF)
— the glyphs most mobile chat clients render as styled (bold / italic /
sans-serif / monospace / …) text.

Each ASCII letter/digit is mapped to its styled counterpart; punctuation,
whitespace and non-ASCII characters pass through **unchanged** (there is no
styled glyph for them), so the result stays readable and never *loses* the
visible characters.

These are opt-in via ``WHATSAPP_UNICODE_FORMATTING`` / ``unicode_formatting``
because a very old or stripped phone font may lack a block (showing a box
instead of a styled glyph). Default is therefore OFF.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# (cap_A_base, small_a_base, digit_0_base or None)  → Unicode math block.
STYLES: Dict[str, Tuple[int, int, Optional[int]]] = {
    "bold": (0x1D400, 0x1D41A, 0x1D7CE),
    "italic": (0x1D434, 0x1D44E, None),  # no italic digits in the block
    "bold_italic": (0x1D468, 0x1D482, None),
    "sans_serif": (0x1D5A0, 0x1D5BA, 0x1D7E2),
    "sans_serif_bold": (0x1D5D4, 0x1D5EE, 0x1D7EC),
    "sans_serif_italic": (0x1D608, 0x1D622, None),
    "monospace": (0x1D670, 0x1D68A, 0x1D7F6),
}

# Heading levels render in distinct Unicode fonts so the document's hierarchy
# survives the down-conversion (native WhatsApp flattens every heading to the
# same bold). Kept to well-supported blocks; numbers/letters map cleanly.
HEADING_STYLE = {1: "bold", 2: "bold_italic", 3: "italic",
                 4: "sans_serif", 5: "monospace", 6: "sans_serif_italic"}

# Mathematical Alphanumeric Symbols deliberately leaves a few code points
# unassigned because they duplicate pre-existing letter-like symbols.
# Mapping a letter there produces U+FFFD-style tofu on every device, so
# substitute the canonical compatibility character instead.  Only U+1D455
# is reachable from the STYLES above (italic lowercase "h" → Planck
# constant); the rest are future-proofing for script/fraktur styles.
SKIPPED_GLYPH_SUBSTITUTES = {
    0x1D455: 0x210E,  # italic small h      → ℎ PLANCK CONSTANT
    0x1D49D: 0x212C,  # script capital B    → ℬ
    0x1D4A0: 0x2130,  # script capital E    → ℰ
    0x1D4A1: 0x2131,  # script capital F    → ℱ
    0x1D4A3: 0x210B,  # script capital H    → ℋ
    0x1D4A4: 0x2110,  # script capital I    → ℐ
    0x1D4A7: 0x2112,  # script capital L    → ℒ
    0x1D4A8: 0x2133,  # script capital M    → ℳ
    0x1D4AD: 0x211B,  # script capital R    → ℛ
    0x1D4BA: 0x212F,  # script small e      → ℯ
    0x1D4BC: 0x210A,  # script small g      → ℊ
    0x1D4C4: 0x2134,  # script small o      → ℴ
    0x1D506: 0x212D,  # fraktur capital C   → ℭ
    0x1D50B: 0x210A,  # fraktur small g     → ℊ
    0x1D50C: 0x2113,  # fraktur small l     → ℓ
    0x1D515: 0x212C,  # fraktur capital B   → ℬ
    0x1D51D: 0x2130,  # fraktur capital E   → ℰ
    0x1D53A: 0x210E,  # fraktur small h     → ℎ
    0x1D53F: 0x2113,  # fraktur small l     → ℓ
    0x1D545: 0x2134,  # fraktur small o     → ℴ
}

_CAP_A, _SMALL_A, _DIGIT_0 = ord("A"), ord("a"), ord("0")
_CAP_Z, _SMALL_Z, _DIGIT_9 = ord("Z"), ord("z"), ord("9")


class GlyphStyler:
    """Translate ASCII letters/digits into a Unicode styled equivalent.

    Usage::
        GlyphStyler.stylize("Title 2", "bold")     # → "𝐓𝐢𝐭𝐥𝐞 2"
        GlyphStyler.stylize("Hi", "sans_serif")    # → "𝖧𝗂"
    """

    _cache: Dict[Tuple[str, str], str] = {}

    @classmethod
    def stylize(cls, text: str, style: str) -> str:
        """Return ``text`` with every mappable character restyled to ``style``.

        Characters with no glyph in the style (punctuation, whitespace,
        non-ASCII, digits that have no styled block) pass through unchanged.
        Unknown style names are treated as a no-op.
        """
        bases = STYLES.get(style)
        if bases is None or not text:
            return text
        key = (text, style)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached

        cap_base, small_base, digit_base = bases
        out: list[str] = []
        for ch in text:
            code = ord(ch)
            if _CAP_A <= code <= _CAP_Z:
                out.append(chr(cap_base + (code - _CAP_A)))
            elif _SMALL_A <= code <= _SMALL_Z:
                out.append(chr(small_base + (code - _SMALL_A)))
            elif digit_base is not None and _DIGIT_0 <= code <= _DIGIT_9:
                out.append(chr(digit_base + (code - _DIGIT_0)))
            else:
                out.append(ch)
        result = "".join(
            chr(SKIPPED_GLYPH_SUBSTITUTES.get(ord(g), ord(g))) for g in out
        )
        cls._cache[key] = result
        return result