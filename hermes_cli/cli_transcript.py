"""Classic CLI transcript chrome: user-message band, spinner hint, turn receipt extras, queue recall.

Display-only helpers for the inline scrollback. Nothing here touches the messages sent to the
model or the system prompt; every function is pure or works on CLI-side state only, so the
rendering rules can be tested without a terminal.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# Share of the skin accent mixed into the neutral band surface: enough to tint the band
# towards the skin, little enough that it reads as a background rather than a highlight.
_BAND_ACCENT_MIX = 0.10
_BAND_BASE = {True: "#F1F1F1", False: "#262626"}
_BAND_TEXT = {True: "#1A1A1A", False: "#ECECEC"}
# The interrupt hint appears only once a turn has run this long, so quick replies never flash it.
SPINNER_HINT_AFTER_S = 3.0


def _hex_rgb(value: str) -> tuple[int, int, int] | None:
    s = (value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None


def luminance(value: str) -> float | None:
    """Rec.709 luma in [0, 1], or None for a malformed colour."""
    rgb = _hex_rgb(value)
    if rgb is None:
        return None
    r, g, b = rgb
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def mix_hex(top: str, base: str, amount: float) -> str:
    """``amount`` of ``top`` over ``base`` (both ``#RRGGBB``); ``base`` when ``top`` is malformed."""
    a, b = _hex_rgb(top), _hex_rgb(base)
    if b is None:
        return base
    if a is None:
        return "#{:02X}{:02X}{:02X}".format(*b)
    return "#{:02X}{:02X}{:02X}".format(*(round(x * amount + y * (1 - amount)) for x, y in zip(a, b)))


def skin_is_light(colors: dict[str, str]) -> bool:
    """A skin authored for light backgrounds sets dark body text (daylight, warm-lightmode)."""
    lum = luminance(colors.get("banner_text", ""))
    return lum is not None and lum < 0.5


def user_band_colors(colors: dict[str, str], light: bool) -> dict[str, str]:
    """``{"bg", "fg", "symbol"}`` for the submitted-message band on a ``light`` or dark surface.

    ``colors.user_message_bg`` wins when a skin sets it; otherwise the band is a neutral surface
    of the right polarity tinted with the skin accent. The text colour is the skin's body text
    when that contrasts with the band, else a neutral ink, so the band never carries its own
    background with an unreadable terminal-default foreground.
    """
    accent = colors.get("ui_accent") or colors.get("banner_accent") or "#FFBF00"
    bg = colors.get("user_message_bg") or mix_hex(accent, _BAND_BASE[light], _BAND_ACCENT_MIX)
    text = colors.get("banner_text", "")
    text_lum, bg_lum = luminance(text), luminance(bg)
    readable = text_lum is not None and bg_lum is not None and abs(text_lum - bg_lum) >= 0.45
    fg = text if readable else _BAND_TEXT[light]
    accent_lum = luminance(accent)
    symbol = accent if accent_lum is not None and bg_lum is not None and abs(accent_lum - bg_lum) >= 0.3 else fg
    return {"bg": bg, "fg": fg, "symbol": symbol}


def active_band_colors() -> dict[str, str]:
    """:func:`user_band_colors` for the active skin and the detected terminal polarity."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        skin = get_active_skin()
        base = dict(skin.colors)
    except Exception:  # no skin yet: neutral band
        logger.debug("skin colours unavailable for the user band", exc_info=True)
        skin, base = None, {}
    try:
        from cli import _detect_light_mode
        terminal_light = bool(_detect_light_mode())
    except Exception:  # polarity probe is best-effort
        logger.debug("terminal polarity probe failed", exc_info=True)
        terminal_light = False
    light = terminal_light or skin_is_light(base)
    overlay = (getattr(skin, "light_colors", None) if light else getattr(skin, "dark_colors", None)) or {}
    return user_band_colors({**base, **overlay}, light)


def preview_lines(text: str, first: int, last: int) -> tuple[list[str], int, list[str]]:
    """``(head, hidden_count, tail)`` of a submitted message for the scrollback echo."""
    lines = text.split("\n")
    head = lines[:max(1, first)]
    tail_count = min(max(0, last), max(0, len(lines) - len(head)))
    tail = lines[-tail_count:] if tail_count else []
    return head, max(0, len(lines) - len(head) - len(tail)), tail


def is_thinking_announcement(text: str, faces: Iterable[str], verbs: Iterable[str]) -> bool:
    """True for the per-API-call ``"<face> <verb>..."`` spinner label the agent loop emits."""
    if not isinstance(text, str) or not text.endswith("..."):
        return False
    body = text[:-3]
    face_set = set(faces)
    return any(body.endswith(" " + verb) and body[: -len(verb) - 1] in face_set for verb in verbs if verb)


def stable_thinking_label(cli: Any, text: str) -> str:
    """Keep the first thinking label of a turn for every later API call in that turn.

    The agent picks a random face and verb per API call; re-randomising the spinner each
    tool round reads as churn. Other spinner text (wait notices, diagnostics) passes through.
    """
    try:
        from agent.display import KawaiiSpinner
        if not is_thinking_announcement(text, KawaiiSpinner.get_thinking_faces(),
                                        KawaiiSpinner.get_thinking_verbs()):
            return text
    except Exception:  # unknown spinner text passes through unchanged
        logger.debug("thinking label lookup failed", exc_info=True)
        return text
    label = getattr(cli, "_turn_thinking_label", None) or text
    cli._turn_thinking_label = label
    return label


def spinner_elapsed_label(seconds: float) -> str:
    """Fixed-width timer (``" 5.2s"`` / ``"01m05s"``) so the spinner line never jitters on repaint."""
    from agent.i18n import t
    seconds = max(0.0, seconds)
    if seconds >= 60:
        return t("cli.shared.duration_ms_compact",
                 minutes=f"{int(seconds // 60):02d}", seconds=f"{int(seconds % 60):02d}")
    return t("cli.shared.duration_s", seconds=f"{seconds:5.1f}")


def spinner_suffix(elapsed: float | None, flow: str, *, hint: str = "") -> str:
    """The parenthesised spinner tail: ``(12.3s · ↓ 1.2k tok · ctrl+c to interrupt)``."""
    parts = [spinner_elapsed_label(elapsed)] if elapsed is not None else []
    parts += [p for p in (flow, hint) if p]
    return f"  ({' · '.join(parts)})" if parts else ""


def receipt_extras(agent: Any, token_baseline: int, cost_baseline: float | None) -> list[str]:
    """Token and cost segments for the turn receipt, from counters the agent already keeps.

    Cost appears only when this turn's pricing is a real estimate (never for ``included`` or
    ``unknown`` pricing, where a ``$0.00`` would mislead).
    """
    if agent is None:
        return []
    from agent.turn_summary import format_token_flow
    extras: list[str] = []
    try:
        flow = format_token_flow((getattr(agent, "session_output_tokens", 0) or 0) - (token_baseline or 0))
    except (TypeError, ValueError):
        flow = ""
    if flow:
        extras.append(flow)
    if cost_baseline is None or getattr(agent, "session_cost_status", "") not in ("estimated", "actual"):
        return extras
    try:
        delta = float(getattr(agent, "session_estimated_cost_usd", 0.0) or 0.0) - float(cost_baseline)
        if delta > 0:
            from agent.usage_pricing import format_cost_label
            extras.append(format_cost_label(Decimal(str(delta))))
    except Exception:  # cost is optional in the receipt
        logger.debug("turn cost segment failed", exc_info=True)
    return extras


# ── Alt+Up: pull the last queued prompt back into the composer ──────────────────────────────


def editable_payload(item: Any) -> tuple[str, list] | None:
    """``(text, images)`` for a prompt the user typed, or None for system-queued turns.

    Notifications, process batches and seeded ``-q`` prompts are not the user's drafts and
    must never be lifted into the composer (editing them would resend them as user text).
    """
    from tools.process_registry_notifications import TimelineNotification
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[0], list(item[1] or [])
    if isinstance(item, str) and not isinstance(item, TimelineNotification):
        return str(item), []
    if type(item).__name__ == "_VoiceInputMessage":
        return str(getattr(item, "text", "") or ""), []
    return None


def pop_last_editable(items: list, payload: Callable[[Any], Any] = editable_payload) -> tuple[list, Any]:
    """Remove the newest user-typed item: ``(remaining, (text, images))`` or ``(items, None)``."""
    for idx in range(len(items) - 1, -1, -1):
        found = payload(items[idx])
        if found is not None:
            return items[:idx] + items[idx + 1:], found
    return items, None


def has_editable_queued(cli: Any) -> bool:
    pending = getattr(cli, "_pending_input", None)
    if pending is None:
        return False
    with pending.mutex:
        items = list(pending.queue)
    return pop_last_editable(items)[1] is not None


def recall_last_queued(cli: Any, buffer: Any) -> bool:
    """Move the newest typed queued prompt into an empty composer. Returns True when moved."""
    if buffer.text or getattr(cli, "_attached_images", None):
        return False
    taken: list = []

    def _take(items: list) -> list:
        remaining, found = pop_last_editable(items)
        if found is not None:
            taken.append(found)
        return remaining

    cli._mutate_pending_input(_take)
    if not taken:
        return False
    text, images = taken[0]
    buffer.text = text
    buffer.cursor_position = len(text)
    if images and isinstance(getattr(cli, "_attached_images", None), list):
        cli._attached_images.extend(images)
    refresh = getattr(cli, "_tui_dock_drop_dequeued", None)
    if callable(refresh):
        refresh()
    return True


def bind_queue_recall(cli: Any, kb: Any) -> None:
    """Alt+Up on an empty composer pulls the last queued prompt back for editing (Pi/Codex)."""
    from prompt_toolkit.filters import Condition

    def _ready() -> bool:
        try:
            from hermes_cli.cli_subagent_monitor import modal_prompt_active
            if modal_prompt_active(cli):
                return False
            app = cli._app
            return not app.current_buffer.text and has_editable_queued(cli)
        except Exception:  # a key filter must never raise into prompt_toolkit
            logger.debug("queue recall filter failed", exc_info=True)
            return False

    def _handler(event) -> None:
        if recall_last_queued(cli, event.app.current_buffer):
            event.app.invalidate()

    kb.add("escape", "up", filter=Condition(_ready))(_handler)
