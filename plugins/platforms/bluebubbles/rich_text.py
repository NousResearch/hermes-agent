"""Native iMessage text styles and send effects for the BlueBubbles adapter.

Markdown emphasis becomes BlueBubbles ``textFormatting`` runs (bold, italic, strikethrough) over the
same plain text ``strip_markdown`` would produce, and an ``[[effect:NAME]]`` marker in a reply becomes
the iMessage ``effectId`` for that send. Both need the Private API helper; the adapter decides when
they apply and falls back to today's plain text otherwise.
"""

import re
from typing import Optional

from gateway.platforms.helpers import strip_markdown

# Private-use sentinels that wrap a styled span while strip_markdown runs; stripped from input first.
_OPEN = {"bold": "", "italic": "", "strikethrough": ""}
_CLOSE = {"bold": "", "italic": "", "strikethrough": ""}
_SENTINEL_STYLE = {**{c: (s, True) for s, c in _OPEN.items()}, **{c: (s, False) for s, c in _CLOSE.items()}}
_SENTINEL_RE = re.compile("[-]")

# The same emphasis patterns strip_markdown removes (gateway.platforms.helpers._STRIP_RULES), in the
# same order, plus ~~strike~~, which plain text keeps as typed but iMessage can render.
_EMPHASIS_RULES = (
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), "bold"),
    (re.compile(r"\*(.+?)\*", re.DOTALL), "italic"),
    (re.compile(r"\b__(?![\s_])(.+?)(?<![\s_])__\b", re.DOTALL), "bold"),
    (re.compile(r"\b_(?![\s_])(.+?)(?<![\s_])_\b", re.DOTALL), "italic"),
    (re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.DOTALL), "strikethrough"),
)
_STYLE_ORDER = ("bold", "italic", "strikethrough")

# Expressive send effects Messages has offered since iOS 10 / macOS 10.12, by the names people use.
IMESSAGE_EFFECTS = {
    "slam": "com.apple.MobileSMS.expressivesend.impact",
    "loud": "com.apple.MobileSMS.expressivesend.loud",
    "gentle": "com.apple.MobileSMS.expressivesend.gentle",
    "invisibleink": "com.apple.MobileSMS.expressivesend.invisibleink",
    "confetti": "com.apple.messages.effect.CKConfettiEffect",
    "balloons": "com.apple.messages.effect.CKHappyBirthdayEffect",
    "fireworks": "com.apple.messages.effect.CKFireworksEffect",
    "heart": "com.apple.messages.effect.CKHeartEffect",
    "lasers": "com.apple.messages.effect.CKLasersEffect",
    "sparkles": "com.apple.messages.effect.CKSparklesEffect",
    "spotlight": "com.apple.messages.effect.CKSpotlightEffect",
    "echo": "com.apple.messages.effect.CKEchoEffect",
    "shootingstar": "com.apple.messages.effect.CKShootingStarEffect",
}
_EFFECT_MARKER_RE = re.compile(r"[ \t]*\[\[effect:\s*([A-Za-z_ -]+?)\s*\]\][ \t]*", re.IGNORECASE)


def extract_effect(text: str) -> tuple[str, Optional[str]]:
    """Remove every ``[[effect:NAME]]`` marker; return the text and the first known effect's id.

    Unknown names are removed too, so a typo never reaches the recipient as literal brackets."""
    found: list[str] = []

    def _take(match: re.Match[str]) -> str:
        found.append(re.sub(r"[\s_-]", "", match.group(1).lower()))
        return " " if match.start() and match.end() < len(match.string) else ""

    cleaned = _EFFECT_MARKER_RE.sub(_take, text or "")
    effect_id = next((IMESSAGE_EFFECTS[name] for name in found if name in IMESSAGE_EFFECTS), None)
    return cleaned, effect_id


def mark_emphasis(content: str) -> str:
    """Markdown -> the plain text strip_markdown makes, with each emphasis span wrapped in sentinels."""
    text = _SENTINEL_RE.sub("", content or "")
    for pattern, style in _EMPHASIS_RULES:
        text = pattern.sub(lambda m, s=style: f"{_OPEN[s]}{m.group(1)}{_CLOSE[s]}", text)
    return strip_markdown(text, keep_link_targets=True)


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def styled_chunk(marked: str, active: tuple[str, ...] = ()) -> tuple[str, list[dict], tuple[str, ...]]:
    """One bubble of sentinel-marked text -> (plain text, textFormatting runs, styles still open).

    Offsets are UTF-16 code units, as BlueBubbles and Messages count them. ``active`` carries styles
    left open by the previous bubble, so a span split across bubbles stays styled in both."""
    open_styles = set(active)
    plain: list[str] = []
    runs: list[dict] = []
    offset = 0
    for piece in re.split(r"([-])", marked):
        if piece in _SENTINEL_STYLE:
            style, opening = _SENTINEL_STYLE[piece]
            (open_styles.add if opening else open_styles.discard)(style)
            continue
        if not piece:
            continue
        length = _utf16_len(piece)
        styles = [s for s in _STYLE_ORDER if s in open_styles]
        if styles and piece.strip():
            if runs and runs[-1]["styles"] == styles and runs[-1]["start"] + runs[-1]["length"] == offset:
                runs[-1]["length"] += length
            else:
                runs.append({"start": offset, "length": length, "styles": styles})
        plain.append(piece)
        offset += length
    text = "".join(plain)
    lead, stripped = _utf16_len(text) - _utf16_len(text.lstrip()), text.strip()
    end = _utf16_len(stripped)
    trimmed = []
    for run in runs:
        start, stop = max(run["start"] - lead, 0), min(run["start"] + run["length"] - lead, end)
        if stop > start:
            trimmed.append({"start": start, "length": stop - start, "styles": run["styles"]})
    return stripped, trimmed, tuple(s for s in _STYLE_ORDER if s in open_styles)


def strip_sentinels(text: str) -> str:
    return _SENTINEL_RE.sub("", text)
