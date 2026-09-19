"""Rich-markup rendering for the TUI input separator."""

from collections import OrderedDict
from functools import lru_cache
import logging

from prompt_toolkit.utils import get_cwidth

_logger = logging.getLogger(__name__)
_BAD_INPUT_RULE_ART_SEEN: OrderedDict[str, None] = OrderedDict()
_BAD_INPUT_RULE_ART_SEEN_MAXSIZE = 128


def _should_log_bad_input_rule_art(art: str) -> bool:
    """Return whether malformed ``art`` is new to the bounded seen-set.

    Eviction re-arms logging, which is acceptable for this debug-only diagnostic.
    """
    if art in _BAD_INPUT_RULE_ART_SEEN:
        return False
    _BAD_INPUT_RULE_ART_SEEN[art] = None
    if len(_BAD_INPUT_RULE_ART_SEEN) > _BAD_INPUT_RULE_ART_SEEN_MAXSIZE:
        _BAD_INPUT_RULE_ART_SEEN.popitem(last=False)
    return True


def _prompt_toolkit_style(style) -> str:
    """Convert supported Rich styles to prompt_toolkit syntax.

    Supported attributes are color, bgcolor, bold, italic, underline, strike, reverse, and blink;
    Rich dim, overline, and conceal have no prompt_toolkit equivalent and are omitted.
    """
    parts = ["class:input-rule"]
    if style.color is not None:
        parts.append(f"fg:{style.color.get_truecolor().hex}")
    if style.bgcolor is not None:
        parts.append(f"bg:{style.bgcolor.get_truecolor(foreground=False).hex}")
    for enabled, name in (
        (style.bold, "bold"),
        (style.italic, "italic"),
        (style.underline, "underline"),
        (style.strike, "strike"),
        (style.reverse, "reverse"),
        (style.blink, "blink"),
    ):
        if enabled:
            parts.append(name)
    return " ".join(parts)


@lru_cache(maxsize=64)
def _parsed_input_rule_art(art: str) -> tuple[tuple[str, str], ...] | None:
    """Parse one-line Rich markup once into prompt_toolkit fragments."""
    try:
        from rich.console import Console
        from rich.text import Text

        text = Text.from_markup(art.splitlines()[0])
        plain = text.plain
        if not plain.strip():
            return None
        console = Console()
        fragments = []
        for index, char in enumerate(plain):
            style = text.get_style_at_offset(console, index)
            fragments.append((_prompt_toolkit_style(style), char))
        return tuple(fragments)
    except Exception as exc:
        if _should_log_bad_input_rule_art(art):
            _logger.debug("Malformed input_rule_art markup; using plain rule", exc_info=exc)
        return None


@lru_cache(maxsize=128)
def _input_rule_fragments(art: str, width: int) -> tuple[tuple[str, str], ...]:
    """Return exactly ``width`` columns of a tiled, clipped Rich-markup rule."""
    width = max(0, width)
    plain_style = "class:input-rule"
    if not art or not width:
        return ((plain_style, "─" * width),) if width else ()
    parsed = _parsed_input_rule_art(art)
    if not parsed:
        return ((plain_style, "─" * width),)

    result: list[tuple[str, str]] = []
    columns = 0
    index = 0
    while columns < width and index < len(parsed) * (width + 1):
        style, char = parsed[index % len(parsed)]
        cell_width = get_cwidth(char)
        index += 1
        if cell_width <= 0:
            continue
        remaining = width - columns
        if cell_width > remaining:
            style, char, cell_width = plain_style, "─", 1
        if result and result[-1][0] == style:
            result[-1] = (style, result[-1][1] + char)
        else:
            result.append((style, char))
        columns += cell_width
    if columns < width:
        result.append((plain_style, "─" * (width - columns)))
    return tuple(result)


def input_rule_fragments(art, width) -> tuple[tuple[str, str], ...]:
    """Return a cached rule; non-string art falls back to a plain rule.

    Only the first line of multiline art is used. ``width`` is the terminal column count (int).
    """
    normalized_art = art if isinstance(art, str) else ""
    normalized_width = width if isinstance(width, int) and not isinstance(width, bool) else 0
    # Cached value stays an immutable tuple (cache cannot be poisoned), but
    # prompt_toolkit's to_formatted_text accepts only a LIST of fragments — a
    # bare tuple raises ValueError at FIRST RENDER (construction is fine, which
    # is why green tests missed it). Copy at the boundary.
    return list(_input_rule_fragments(normalized_art, normalized_width))
