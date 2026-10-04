"""Nominal text-capacity estimation for pptx body placeholders.

python-pptx cannot measure rendered text, and overfilling a placeholder is
silent in the outline. PowerPoint normally auto-shrinks fixed-placeholder
text to fit; other renderers or templates may clip it. This module provides
a conservative *estimate* of the unscaled rendered height of bullet text so
callers (and the agent) get an explicit warning before either failure mode
produces an unreadable slide.

The estimate is heuristic on purpose: it uses per-character width factors
(no font files needed) and mirrors the default python-pptx template's
master text styles, which inherit font sizes of 32/28/24/20/20pt for
bullet levels 0-4 unless a run sets an explicit size. Estimates are
biased to OVER-predict height (warnings may fire on borderline slides;
missed overflow defeats the guardrail's purpose). It is only meaningful
for the fixed-frame placeholders pptx_create.py writes into — not for
auto-fitting textboxes, which grow instead of clipping.
"""

# Rough advance-width per character as a fraction of font size, tuned for
# Calibri/Arial-class sans faces at mixed-case prose. Deliberately rounded
# up: false positives are cheap (an extra look), false negatives hide bugs.
_DEFAULT_WIDTH = 0.52
_WIDE_CHARS = set("MWmw@%")
_NARROW_CHARS = set("iljtfIr.,:;'|!()[] ")
# CJK / full-width ideographs and kana render at ~1em advance width.
_CJK_THRESHOLD = 0x2E80

# Default-template master bodyStyle: inherited run size (pt) and left
# margin (inches, includes the bullet hang) per bullet level 0-4.
INHERITED_SIZE_PT = {0: 32.0, 1: 28.0, 2: 24.0, 3: 20.0, 4: 20.0}
INHERITED_INDENT_IN = {0: 0.375, 1: 0.8125, 2: 1.25, 3: 1.75, 4: 2.25}

# Master bodyStyle adds spcBef of 20% of the line before each paragraph;
# single line spacing for sans faces approximates 1.22 x size.
_LINE_FACTOR = 1.22
_SPC_BEFORE = 0.20


def _char_width(char, size_pt):
    if ord(char) >= _CJK_THRESHOLD:
        return size_pt * 1.0
    if char in _WIDE_CHARS:
        return size_pt * 0.85
    if char in _NARROW_CHARS:
        return size_pt * 0.30
    return size_pt * _DEFAULT_WIDTH


def text_width_pt(text, size_pt=18.0):
    """Estimated rendered width of `text` in points."""
    return sum(_char_width(c, size_pt) for c in text)


def _wrap_segment(text, wrap_width_pt, size_pt):
    """Greedy wrap of one hard-line segment; splits oversized tokens."""
    lines, current = 0, ""
    space = _char_width(" ", size_pt)
    for word in text.split():
        while text_width_pt(word, size_pt) > wrap_width_pt:
            # Unbreakable token longer than the frame: flush what fits.
            if current:
                lines += 1
                current = ""
            cut = len(word)
            while cut > 1 and text_width_pt(word[:cut], size_pt) > wrap_width_pt:
                cut -= 1
            lines += 1
            word = word[cut:]
        candidate = f"{current} {word}".strip()
        if not current:
            current = word
        elif text_width_pt(candidate, size_pt) <= wrap_width_pt:
            current = candidate
        else:
            lines += 1
            current = word
    if current:
        lines += 1
    return max(lines, 1)


def wrapped_line_count(text, wrap_width_pt, size_pt=18.0):
    """Greedy word-wrap line count using the estimated widths.

    Honors hard line breaks ('\\n') as mandatory line boundaries.
    """
    total = 0
    for segment in text.split("\n"):
        if not segment.strip():
            total += 1  # empty line still occupies one row
        else:
            total += _wrap_segment(segment, wrap_width_pt, size_pt)
    return max(total, 1)


def effective_size_pt(item):
    """Effective run size for a bullet spec item: explicit `size` when
    truthy (same semantics as style_run), else the level's inherited
    master size. Accepts a bare string for convenience."""
    if isinstance(item, str):
        item = {"text": item}
    explicit = item.get("size")
    if explicit:
        return float(explicit)
    level = int(item.get("level", 0))
    return INHERITED_SIZE_PT.get(min(level, 4), 20.0)


def effective_indent_in(level):
    return INHERITED_INDENT_IN.get(min(int(level), 4), 2.25)


def paragraph_height_pt(text, size_pt=None, level=0, wrap_width_pt=612.0,
                        indent_in=None):
    """Estimated height of one bullet paragraph in points.

    Default wrap width matches the default template's content placeholder
    (9.0in wide minus ~0.2in of side insets); the level's master indent is
    subtracted from the wrap width automatically unless `indent_in` is
    given explicitly.
    """
    if size_pt is None:
        size_pt = INHERITED_SIZE_PT.get(min(int(level), 4), 20.0)
    if indent_in is None:
        indent_in = effective_indent_in(level)
    usable = max(wrap_width_pt - indent_in * 72.0, wrap_width_pt * 0.35)
    lines = wrapped_line_count(text, usable, size_pt)
    return lines * size_pt * _LINE_FACTOR * (1.0 + _SPC_BEFORE)


def estimate_bullets_overflow(bullets, frame_width_in=None,
                              frame_height_in=None):
    """Estimate whether bullet content overflows its body placeholder.

    `bullets` is the create-script spec list (strings or dicts with
    `text`, optional `level`/`size`). Frame dimensions come from the
    placeholder when known; defaults mirror the default template's
    title_content body placeholder (9.0in wide x 4.95in tall).

    Returns None when there is nothing to report (no bullets, no known
    frame, or the unscaled estimate fits), otherwise a dict with the
    estimated vs. fitted heights and a remediation hint. The warning is a
    nominal-capacity guardrail: PowerPoint may shrink the text rather than
    clip it, so callers must verify the rendered slide.
    """
    # Default template content placeholder: 9.0in wide x 4.95in tall.
    width_in = frame_width_in if frame_width_in else 9.0
    height_in = frame_height_in if frame_height_in else 4.95
    if not bullets:
        return None

    wrap_pt = (width_in - 0.2) * 72.0
    total = 0.0
    for item in bullets:
        if isinstance(item, str):
            item = {"text": item}
        total += paragraph_height_pt(
            item.get("text", ""),
            size_pt=effective_size_pt(item),
            level=int(item.get("level", 0)),
            wrap_width_pt=wrap_pt)

    # Usable frame height subtracts the 0.05in top/bottom text insets.
    # Warn slightly BEFORE the theoretical fit (2% early) so borderline
    # overfill surfaces instead of silently shrinking or clipping.
    fitted = max(height_in * 72.0 - 0.1 * 72.0, 1.0)
    if total <= fitted * 0.98:
        return None
    return {
        "estimated_text_height_pt": round(total, 1),
        "frame_usable_height_pt": round(fitted, 1),
        "hint": ("content exceeds the body placeholder's nominal text "
                 "capacity and may be auto-shrunk or clipped; verify with a "
                 "render, then split across slides, shorten bullets, or set "
                 "smaller per-bullet \"size\" values"),
    }
