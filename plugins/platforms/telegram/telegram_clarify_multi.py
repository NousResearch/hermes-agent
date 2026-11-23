"""Multi-select clarify on Telegram: toggle buttons, then Send.

A ``multi_select`` clarify renders one ``cl:<id>:t<idx>`` toggle per choice plus "Other" and
"Send" (``cl:<id>:go``). A toggle only flips the checkmark; Send resolves the entry once with a
JSON array of the checked labels, the same shape the typed ``"1, 3"`` reply resolves to, so the
tool reports each pick as its own answer.
"""

from __future__ import annotations

import html
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def pending_multi_entry(clarify_id: str) -> Optional[Any]:
    """The pending entry when it is a multi-select with choices, else None."""
    from tools.clarify_gateway import get_pending
    entry = get_pending(clarify_id)
    return entry if entry is not None and entry.multi_select and entry.choices else None


def multi_keyboard(adapter: Any, clarify_id: str, count: int, selected: set[int]) -> Any:
    """One toggle row per choice, then Other and Send; callback data stays under Telegram's 64 bytes."""
    from plugins.platforms.telegram import adapter as tg
    button, t = tg.InlineKeyboardButton, tg.t
    rows = [[button(f"{'☑' if idx in selected else '☐'} {idx + 1}", callback_data=f"cl:{clarify_id}:t{idx}")]
            for idx in range(count)]
    rows.append([button(t("platform.telegram.prompt.other"), callback_data=f"cl:{clarify_id}:other")])
    rows.append([button(t("platform.telegram.prompt.send"), callback_data=f"cl:{clarify_id}:go")])
    return tg.InlineKeyboardMarkup(rows)


def multi_prompt_extras(adapter: Any, clarify_id: str, count: int) -> Optional[tuple[str, Any]]:
    """``(hint_html, keyboard)`` for a multi-select prompt, None for a single-select one."""
    from plugins.platforms.telegram.adapter import t
    if pending_multi_entry(clarify_id) is None:
        return None
    return (f"\n\n<i>{html.escape(t('platform.telegram.prompt.multi_hint'))}</i>",
            multi_keyboard(adapter, clarify_id, count, set()))


async def handle_multi_token(adapter: Any, query: Any, clarify_id: str, token: str, prompt: Any,
                             user_display: str) -> bool:
    """Handle a ``t<idx>`` toggle or the ``go`` Send tap; False when *token* is neither."""
    from plugins.platforms.telegram.adapter import _toast
    from plugins.platforms.telegram.callback_answer import answer_callback
    is_toggle = token.startswith("t") and token[1:].isdigit()
    if not is_toggle and token != "go":
        return False
    entry = pending_multi_entry(clarify_id)
    if entry is None:  # timed out, answered by text, or reset: retire the card instead of a dead keyboard
        adapter._clarify_state.pop(clarify_id, None)
        adapter._clarify_multi.pop(clarify_id, None)
        await adapter._notify_clarify_expired(query, user_display)
        return True
    idx = int(token[1:]) if is_toggle else -1
    if is_toggle and not 0 <= idx < len(entry.choices):
        await answer_callback(query, _toast("platform.telegram.prompt.invalid_choice"))
        return True
    selected = adapter._clarify_multi.setdefault(clarify_id, set())
    if is_toggle:
        await _toggle(adapter, query, clarify_id, entry, selected, idx, user_display)
        return True
    # The tool strips the "(Recommended)" display label per item when it decodes the array.
    labels = [str(entry.choices[i]).strip() for i in sorted(selected) if i < len(entry.choices)]
    if not labels:
        await answer_callback(query, _toast("platform.telegram.prompt.select_one"))
        return True
    await resolve_clarify_choice(adapter, query, clarify_id, prompt, json.dumps(labels, ensure_ascii=False),
                                 user_display, shown=", ".join(labels))
    return True


async def _toggle(adapter: Any, query: Any, clarify_id: str, entry: Any, selected: set[int], idx: int,
                  user_display: str) -> None:
    """Flip *idx* only once Telegram may show the new ticks, so Send never submits a pick the card hides."""
    from telegram.error import TelegramError
    from plugins.platforms.telegram.adapter import _toast
    from plugins.platforms.telegram.callback_answer import answer_callback
    from plugins.platforms.telegram.telegram_send_errors import MAY_HAVE_DELIVERED, send_error_kind
    from tools.clarify_gateway import get_pending
    failure = None
    try:
        await query.edit_message_reply_markup(
            reply_markup=multi_keyboard(adapter, clarify_id, len(entry.choices), selected ^ {idx}))
    except TelegramError as exc:
        failure = exc
    # A reset or timeout while the edit was in flight retired this question: retire the card too.
    if get_pending(clarify_id) is not entry:
        adapter._clarify_state.pop(clarify_id, None)
        adapter._clarify_multi.pop(clarify_id, None)
        await adapter._notify_clarify_expired(query, user_display)
        return
    # "Not modified" proves the card already shows the requested ticks (an earlier unconfirmed edit
    # that did not land, re-tapped): adopt them, or every later tap is refused the same way.
    if failure is not None and "not modified" not in str(failure).lower():
        if send_error_kind(failure) not in MAY_HAVE_DELIVERED:  # refused, flood-waited or never sent
            logger.warning("Telegram clarify toggle not applied (id=%s): %s", clarify_id, failure)
            await answer_callback(query, _toast("platform.telegram.prompt.toggle_failed"))
            return
        # The request left and no answer came back: the edit may be on screen, so keep the flip.
        logger.warning("Telegram clarify toggle edit unconfirmed (id=%s): %s", clarify_id, failure)
    selected.symmetric_difference_update({idx})
    await answer_callback(query, _toast("platform.telegram.prompt.selected_count", count=len(selected)))


async def resolve_clarify_choice(adapter: Any, query: Any, clarify_id: str, prompt: Any, resolved_text: str,
                                 user_display: str, shown: Optional[str] = None) -> None:
    """Resolve the pending entry with *resolved_text* and replace the prompt body with *shown*
    (default: the same text); an entry already gone gets the expired notice."""
    from plugins.platforms.telegram.callback_answer import answer_callback
    from plugins.platforms.telegram.telegram_clarify_binding import log_resolved
    shown = resolved_text if shown is None else shown
    adapter._clarify_state.pop(clarify_id, None)
    adapter._clarify_multi.pop(clarify_id, None)
    from tools.clarify_gateway import resolve_gateway_clarify
    resolved = resolve_gateway_clarify(clarify_id, resolved_text)  # a locked dict read + Event.set
    if not resolved:  # entry evicted / gateway restarted between ask and tap
        await adapter._notify_clarify_expired(query, user_display)
        logger.warning("Telegram clarify button: resolve_gateway_clarify returned False (id=%s)", clarify_id)
        return
    await answer_callback(query, f"✓ {shown[:60]}")
    body = html.escape(query.message.text or "")
    await adapter._edit_html_quiet(query, f"❓ {body}\n\n<b>{html.escape(user_display)}:</b> {html.escape(shown)}")
    log_resolved(query, clarify_id, prompt, shown, user_display)
