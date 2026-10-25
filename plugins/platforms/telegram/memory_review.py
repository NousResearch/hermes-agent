"""Telegram review controls for durable memory proposals."""

from __future__ import annotations

import asyncio
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools import write_approval as wa
from tools.memory_prompt_send import record_lock, review_buttons
from utils import atomic_json_write


def _now():
    return time.time()


_SNOOZE_SECONDS = {"1h": 3600, "4h": 4 * 3600, "3d": 3 * 86400}


def _snooze(home, pending_id, duration, revision):
    from hermes_time import get_timezone, safe_strftime

    with _home_scope(home), record_lock(pending_id):
        record = wa.get_pending(wa.MEMORY, pending_id)
        if record is None or record.get("revision", 1) != revision:
            return None, "This review has changed. Use the latest card."
        zone = get_timezone()
        now = _now()
        local_now = datetime.fromtimestamp(now, zone)
        if duration == "tomorrow9":
            until = (local_now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0).timestamp()
        else:
            until = now + _SNOOZE_SECONDS[duration]
        record["snooze_until"] = until
        record["revision"] = revision + 1
        atomic_json_write(wa._pending_path(wa.MEMORY, pending_id), record)
        return record, "⏰ Snoozed until " + safe_strftime(datetime.fromtimestamp(until, zone), "%Y-%m-%d %H:%M %Z")


def snooze_buttons(pending_id, revision):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=f"mem:snooze:{pending_id}:{revision}:{value}")
         for label, value in (("1h", "1h"), ("4h", "4h"), ("tomorrow 9am", "tomorrow9"), ("3d", "3d"))],
        [InlineKeyboardButton("Cancel", callback_data=f"mem:cancel:{pending_id}:{revision}")]])


@contextmanager
def _home_scope(home):
    # Full runtime scope also binds the owning config, credentials and memory-provider limits.
    from gateway.run import _profile_runtime_scope

    with _profile_runtime_scope(Path(home)):
        yield


def _find_home(adapter, query, pending_id, revision, *, draft_token=None):
    """Locate only a served record with a durable delivery receipt for this exact prompt."""
    homes = [get_hermes_home()]
    runner = getattr(adapter, "gateway_runner", None)
    homes.extend((getattr(runner, "_served_profile_homes", None) or {}).values())
    for home in dict.fromkeys(Path(h) for h in homes):
        token = set_hermes_home_override(home)
        try:
            record = wa.get_pending(wa.MEMORY, pending_id)
        finally:
            reset_hermes_home_override(token)
        if (record is None or record.get("revision", 1) != revision
                or record.get("payload", {}).get("action") == "batch" or record.get("conversion_pending")
                or record.get("snooze_until", 0) > time.time()):
            continue
        if draft_token is not None:
            draft = record.get('discussion_draft') or {}
            prompts = [draft.get('receipt') or {}] if draft.get('token') == draft_token else []
        else:
            prompts = record.get("prompts") or [record.get("prompt") or {}]
        for prompt in prompts:
            message = query.message
            if (prompt.get("revision", 1) == revision
                    and str(message.chat_id) == prompt.get("chat_id")
                    and str(message.message_id) in prompt.get("message_ids", [])[-1:]
                    and str(getattr(message.from_user, "id", "")) == prompt.get("bot_id")
                    and (str(message.message_thread_id) if message.message_thread_id is not None else None)
                    == prompt.get("thread_id")):
                return home
    return None


def _decide(home, pending_id, action, revision):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_rejections import remember_rejection
    from tools.memory_tool import load_on_disk_store

    with _home_scope(home), record_lock(pending_id):
        record = wa.get_pending(wa.MEMORY, pending_id)
        if record is None or record.get("revision", 1) != revision:
            return None, None, "This review has changed or was already handled."
        if action == "reject":
            remember_rejection(record)
            wa.discard_pending(wa.MEMORY, pending_id)
            return record, "🗑 Disapproved · Memory unchanged", None
        from tools.memory_review_records import review_issue
        store = load_on_disk_store()
        if issue := review_issue(record, store):
            return record, None, issue
        result = handle_pending_subcommand(wa.MEMORY, ["approve", pending_id], memory_store=store)
        if wa.get_pending(wa.MEMORY, pending_id) is not None:
            return record, None, result
        return record, "✅ Approved · This memory is saved", None


async def show_status(query, record, status, *, buttons=False):
    """Keep the comparison and append the status; handle old receipt footers too."""
    from tools.memory_prompt_text import ReviewMessage, render_cards, units
    receipt = next((p for p in record.get("prompts", [record.get("prompt", {})])
                    if str(query.message.message_id) in p.get("message_ids", [])[-1:]), {})
    data = receipt.get("card")
    if data is None:
        card = render_cards(record, limit=3700)[-1]
    else:
        card = ReviewMessage(data["text"], list(data["entities"]))
    legacy_footer = 'These actions affect this memory only.\nNothing changes until you approve.'
    position = len(card.text) - len(legacy_footer)
    offset = units(card.text[:position]) if position >= 0 else 0
    if card.text.endswith(legacy_footer) and not any(
            e['type'] in {'blockquote', 'expandable_blockquote'}
            and e['offset'] <= offset < e['offset'] + e['length'] for e in card.entities):
        card.text = card.text[:position]
        offset = units(card.text)
        card.entities = [e for e in card.entities if e['offset'] + e['length'] <= offset]
    card.append(status, 'bold')
    await query.edit_message_text(text=card.text, entities=card.entities, parse_mode=None,
        reply_markup=review_buttons(record['id'], record.get('revision', 1)) if buttons else None)


async def _handle_decision(adapter, query, cb, home, pending_id, action, extra):
    revision = int(extra[0])
    record, status, error = await asyncio.to_thread(_decide, home, pending_id, action, revision)
    if error:
        await query.answer(text="Could not apply this change. The proposal is still awaiting review.")
        if record:
            # Long store errors may echo the original: the full source is already in the card.
            detail = error if len(error) <= 320 else error[:320].rsplit(' ', 1)[0] + '…'
            await show_status(query, record, '⚠ Could not apply this change\n' + detail, buttons=True)
        return
    await query.answer()
    await show_status(query, record, status)
    from tools.memory_space_refresh import refresh_in_home
    await asyncio.to_thread(refresh_in_home, home)


async def _handle_snooze(adapter, query, cb, home, pending_id, action, extra):
    revision = int(extra[0])
    if len(extra) == 1:
        await query.answer(text="How long would you like to snooze?")
        await query.edit_message_reply_markup(reply_markup=snooze_buttons(pending_id, revision))
        return
    duration = extra[1]
    if duration not in {*_SNOOZE_SECONDS, "tomorrow9"}:
        await query.answer(text="Invalid snooze duration")
        return
    record, status = await asyncio.to_thread(_snooze, home, pending_id, duration, revision)
    await query.answer()
    if record:
        await show_status(query, record, status)
        from tools.memory_space_refresh import refresh_in_home
        await asyncio.to_thread(refresh_in_home, home)


async def _handle_cancel(adapter, query, cb, home, pending_id, action, extra):
    await query.answer(text="Cancelled")
    from plugins.platforms.telegram.memory_review_edit import cancel_edit
    if await cancel_edit(adapter, query, pending_id):
        return
    await query.edit_message_reply_markup(reply_markup=review_buttons(pending_id, int(extra[0])))


async def handle_callback(adapter, query, data: str, cb: dict) -> None:
    if not await adapter._callback_authorized(query, cb, "You are not authorized to review this memory."):
        return
    parts = data.split(":")
    if (len(parts) not in {4, 5} or not re.fullmatch(r"[a-f0-9]{8}", parts[2])
            or not re.fullmatch(r"[1-9][0-9]{0,8}", parts[3])):
        await query.answer(text="This is an older review. Use the latest independent memory card.")
        return
    _, action, pending_id, *extra = parts
    revision = int(extra[0])
    from plugins.platforms.telegram.memory_review_edit import begin_edit, cancel_home
    from plugins.platforms.telegram.memory_review_discussion import begin_discussion
    from plugins.platforms.telegram.memory_review_draft import use_draft
    handlers = {"use": use_draft, "discuss": begin_discussion, "approve": _handle_decision, "reject": _handle_decision,
                "snooze": _handle_snooze, "cancel": _handle_cancel, "edit": begin_edit}
    handler = handlers.get(action)
    if (handler is None or (len(extra) > 1 and action not in {"snooze", "use"})
            or (action == "use" and (len(extra) != 2 or not re.fullmatch(r"[a-f0-9]{12}", extra[1])))):
        await query.answer(text="Invalid memory review action")
        return
    home = cancel_home(adapter, query, pending_id, revision) if action == "cancel" else None
    if home is None:
        home = await asyncio.to_thread(_find_home, adapter, query, pending_id, revision,
                                       draft_token=extra[1] if action == "use" else None)
    if home is None:
        await query.answer(text="This review changed or was already handled. Use its latest card.")
        return
    await handler(adapter, query, cb, home, pending_id, action, extra)
