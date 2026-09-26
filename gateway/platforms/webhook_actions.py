"""Opt-in webhook discussion controls, using the owning session's clarify and wake routes."""
import asyncio
import hashlib
import json
import re
import time

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.wake import admit_internal_event
from tools import clarify_gateway

KEY = "webhook_discussion_action"
CHOICES = ["Ask about this task", "Show decision"]
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def binding(payload):
    """Accept references only: no arbitrary callback command or approval semantics."""
    value = payload.get("discussion_action") if isinstance(payload, dict) else None
    if not isinstance(value, dict) or set(value) != {"eventId", "taskId", "cardId", "sourceSessionId"}:
        return None
    if any(not isinstance(item, str) or not ID.fullmatch(item) or ".." in item for item in value.values()):
        return None
    return value


def _entry(owner, platform, chat_id, thread_id, profile):
    runner = owner.gateway_runner
    entries = []
    for entry in runner.session_store.list_sessions()[:128]:
        source = runner._restored_source(entry)
        if (source and source.platform == platform and str(source.chat_id) == str(chat_id)
                and str(source.thread_id or "") == str(thread_id or "")
                and (source.profile or "default") == (profile or "default")
                and source.user_id and source.chat_type in {"dm", "private"}
                and runner._is_user_authorized(source)):
            entries.append((entry, source))
    return entries[0] if len(entries) == 1 else None


def _start_wait(owner, entry, source, adapter, record, profile):
    cid = record["clarifyId"]
    if clarify_gateway.has_pending(entry.session_key):
        return False
    clarify_gateway.register(cid, entry.session_key, record["question"], CHOICES,
                             owner_user_id=str(source.user_id))
    async def wait():
        with owner._profile_scope(profile):
            remaining = max(0.0, record["expiresAt"] - time.time())
            response = await asyncio.to_thread(clarify_gateway.wait_for_response, cid, remaining or .001)
            current = owner.gateway_runner.session_store.get_session_metadata(entry.session_key, KEY)
            if not current or current.get("clarifyId") != cid or current.get("state") != "pending":
                return
            if response not in CHOICES:
                current["state"] = "expired"
            else:
                live = owner.gateway_runner.session_store.lookup_by_session_key(entry.session_key)
                if not live or live.session_id != entry.session_id or not owner.gateway_runner._is_user_authorized(source):
                    current["state"] = "expired"
                    owner.gateway_runner.session_store.set_session_metadata(entry.session_key, KEY, current)
                    return
                current["state"] = "answered"
                # Persist the claimed response before admission so replay never grants a second action.
                owner.gateway_runner.session_store.set_session_metadata(entry.session_key, KEY, current)
                refs = json.dumps(current["binding"], sort_keys=True, separators=(",", ":"))
                event = MessageEvent(text=f"The user selected '{response}' for the existing task references {refs}. "
                                     "Report its current status or pending decision. This is a discussion request, not approval or a new task.",
                                     message_type=MessageType.TEXT, source=source, internal=True,
                                     message_id=f"discussion:{cid}", metadata={"gateway_session_key": entry.session_key,
                                                                             "gateway_session_id": entry.session_id})
                try:
                    await admit_internal_event(adapter, event)
                    current["state"] = "admitted"
                except Exception:
                    current["state"] = "admission_failed"
                    # Keep exact binding and response on existing session metadata; no approval or retry loop.
            owner.gateway_runner.session_store.set_session_metadata(entry.session_key, KEY, current)
    task = asyncio.create_task(wait())
    owner._background_tasks.add(task)
    task.add_done_callback(owner._background_tasks.discard)
    return True


async def deliver(owner, adapter, platform, chat_id, thread_id, content, delivery):
    refs = binding(delivery.get("payload"))
    if platform.value not in {"discord", "whatsapp"}:
        return None
    if not refs:
        return None
    profile = delivery.get("profile")
    resolved = _entry(owner, platform, chat_id, thread_id, profile)
    if not resolved:
        return SendResult(success=False, error="Discussion controls require one authorized paired session")
    entry, source = resolved
    store = owner.gateway_runner.session_store
    previous = store.get_session_metadata(entry.session_key, KEY)
    if previous and previous.get("binding") == refs:
        return SendResult(success=True)
    if previous and previous.get("state") == "pending" and previous.get("expiresAt", 0) > time.time():
        return None  # Deliver the actual alert as text; a second poll would bind to the wrong FIFO reply.
    cid = hashlib.sha256(json.dumps([entry.session_id, refs], sort_keys=True).encode()).hexdigest()[:24]
    record = {"binding": refs, "clarifyId": cid, "sessionId": entry.session_id,
              "userId": str(source.user_id), "profile": profile or "default",
              "question": content, "state": "pending", "expiresAt": time.time() + 300}
    if not store.set_session_metadata(entry.session_key, KEY, record):
        return SendResult(success=False, error="Discussion binding could not be persisted")
    if not _start_wait(owner, entry, source, adapter, record, profile):
        record["state"] = "deferred"
        store.set_session_metadata(entry.session_key, KEY, record)
        return None  # Preserve notification delivery without stealing the ordinary clarification.
    result = await adapter.send_clarify(chat_id=chat_id, question=content, choices=CHOICES,
                                      clarify_id=cid, session_key=entry.session_key,
                                      metadata={"thread_id": thread_id} if thread_id else None)
    current = store.get_session_metadata(entry.session_key, KEY)
    if current and current.get("clarifyId") == cid:
        current["messageId"] = result.message_id
        store.set_session_metadata(entry.session_key, KEY, current)
    if not result.success:
        clarify_gateway.resolve_gateway_clarify(cid, "", user_id=str(source.user_id))
    return result


async def restore(owner):
    """Restore reply interception without resending notifications or waking models."""
    runner = owner.gateway_runner
    if not runner or not getattr(runner, "session_store", None):
        return
    for entry in runner.session_store.list_sessions()[:128]:
        record = entry.metadata.get(KEY)
        if not isinstance(record, dict) or record.get("state") != "pending":
            continue
        if (not binding({"discussion_action": record.get("binding")})
                or not isinstance(record.get("expiresAt"), (int, float))
                or not isinstance(record.get("question"), str)
                or not isinstance(record.get("clarifyId"), str) or not ID.fullmatch(record["clarifyId"])):
            continue
        profile = record.get("profile")
        with owner._profile_scope(profile):
            source = runner._restored_source(entry)
            if (record.get("expiresAt", 0) <= time.time() or record.get("sessionId") != entry.session_id
                    or not source or str(source.user_id or "") != record.get("userId")
                    or (source.profile or "default") != (profile or "default")
                    or not runner._is_user_authorized(source)):
                record["state"] = "expired"
                runner.session_store.set_session_metadata(entry.session_key, KEY, record)
                continue
            adapter = runner._delivery_adapter_for(source)
            if adapter and _start_wait(owner, entry, source, adapter, record, profile):
                restore_card = getattr(adapter, "restore_clarify_card", None)
                if callable(restore_card) and record.get("messageId"):
                    await restore_card(record["clarifyId"], CHOICES, record["messageId"])
