"""Requester-only Telegram group controls via Bot API 10.2/10.3 ephemeral messages.

Parent-stamped metadata (authentic source only): ``telegram_requester_user_id``
(str; required by the send — a configured group without it FAILS CLOSED, never
public), ``telegram_chat_type`` (dm/private/channel keep the public path),
``telegram_callback_query_id`` / ``telegram_ephemeral_reply_id`` (the two 15s
non-admin delivery triggers; chat admins may send any time — Telegram rejects
non-privileged attempts, we report them as failures, never as delivery).

Ephemeral messages have ``message_id == 0`` and carry ``receiver_user`` +
``ephemeral_message_id`` (PTB 22.8 preserves them in ``api_kwargs``; outbound
extras ride ``api_kwargs`` / raw ``do_api_request``); edits/deletes go through
``editEphemeralMessageText`` / ``deleteEphemeralMessage`` with ``(chat_id,
receiver_user_id, ephemeral_message_id)`` — receiver is separate from chat and
a regular ``message_id`` is never a target.

Rich embedding (``extra.rich_controls`` on): sends go through ``sendRichMessage``
with ``ephemeral_message_parameters`` and buttons embedded as native
``<tg-button-row>`` blocks (via the adapter's ``_rich_control_payload`` helper —
``sendRichMessage`` accepts ``ephemeral_message_parameters``); edits carry
``rich_message`` EXCLUSIVELY (never alongside ``text``/``parse_mode``/
``reply_markup`` — no contradictory content fields). The flag is respected and
rich mode is never mandatory: flag off, missing helper, conversion failure, or a
permanent/capability rejection (adapter's ``_is_rich_fallback_error``) degrades
to the plain ephemeral ``sendMessage``/text-edit path; anything else fails
closed. The runtime return may be a raw dict (``do_api_request`` without
``return_type``) or a PTB ``Message`` — ``_ephemeral_fields`` reads both.

Ephemeral overlay (Bot API 10.3 ``EphemeralMessageParameters``): a send
triggered by a FRESH public callback query (parent-stamped
``telegram_callback_query_id`` NOT originating from an ephemeral message) may
set ``replace_callback_query_message: True`` — the new private content replaces
the public overlay in place. A callback from an ephemeral message never
replaces (API contract: those must use the ``editEphemeralMessage…`` methods),
so its replacement flow edits the SAME ephemeral message instead of sending a
duplicate. Media edits go through ``editEphemeralMessageMedia`` with
``InputMedia`` (file id/URL) or a newly uploaded local file (PTB 22.8
``InputFile`` inside the ``media`` parameter of ``do_api_request`` — verified
against PTB source: ``RequestParameter.from_input`` extracts InputMedia
``InputFile`` payloads into multipart data; there is no separate ``files``
parameter on ``do_api_request``), and captions through
``editEphemeralMessageCaption``; both stay requester-gated and private.

Parent integration (this module edits no adapter/gateway file): mixin first in
the ``TelegramAdapter`` bases; ``__init__`` sets ``self._private_controls =
self._coerce_bool_extra("private_controls", False)``; ``_send_control_message``
calls ``private_control_requested`` → ``_send_private_control`` BEFORE the
public send (failures via ``send_private_control_prompt``'s decline contract —
never a public fallback; ``send()`` fails closed on
``metadata.telegram_private_control``); ``_handle_callback_query`` wraps with
``wrap_private_control_query`` + ``gate_private_control_query`` before
``_accept_update`` and the allowlist auth (receiver gate layers ON TOP of the
allowlist); ``_callback_ctx`` captures ``query.id`` / ``query.from_user.id``;
``delete_message`` routes ``eph:`` ids to ``delete_ephemeral_control``; picker
text fallbacks gate on ``gateway.relay.egress.declined_send(result)``.

Callback-context propagation (parent-owned): stamp these metadata keys from the
TRUSTED update object only — never from message text or other user-controllable
content:
``telegram_callback_query_id`` = ``query.id`` (the fresh public trigger),
``telegram_requester_user_id`` = ``query.from_user.id`` (authentic requester),
``telegram_callback_from_ephemeral`` = True when ``query.message`` is itself an
ephemeral message (its ``message_id == 0`` with receiver/ephemeral fields —
suppresses replace_callback_query_message), and
``telegram_ephemeral_reply_id`` = the ephemeral id when replying inside an
ephemeral flow. With those keys, ``_send_private_control`` decides the overlay:
a fresh public callback id + non-ephemeral origin →
``replace_callback_query_message: True``; an ephemeral-origin callback → edit
of the existing ephemeral anchor (call ``edit_ephemeral_control_text``/
``edit_ephemeral_control_media`` with the anchor's record/handle), never a
duplicate send.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from gateway.platforms.base import SendResult
from gateway.platforms.helpers import bounded_put
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

logger = logging.getLogger(__name__)

EPH_PREFIX = "eph:"
GROUP_CHAT_TYPES = frozenset({"group", "supergroup", "forum"})
TOAST_LIMIT = 200
_RECEIVERS_ONLY_TOAST = "This control is addressed to another member."
DECLINE_ERROR = "telegram egress declined: private control delivery unavailable"
_TEXT_SEND_DEADLINE = 60.0


class PrivateControlError(Exception):
    """Private control failed; never convert into a public send (content is requester-only)."""


def _redact_private_error(exc: object) -> str:
    """Redacted transport error text; never logs raw exceptions (tokens/secrets)."""
    try:
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text
        return _redact_telegram_error_text(exc)
    except Exception:
        return "<private control error redacted>"

async def _bounded_api_call(awaitable, label: str):
    """One Telegram call under the adapter's wall-clock deadline (thread-timer —
    survives a blocked loop / cancellation-shielded PTB+httpcore init)."""
    from plugins.platforms.telegram.adapter import _await_with_thread_deadline
    return await _await_with_thread_deadline(
        awaitable, timeout=_TEXT_SEND_DEADLINE, label=label, dump_on_blocked_loop=False)


def _rich_embedding_enabled(self) -> bool:
    """``extra.rich_controls`` — embeds rich buttons in ephemeral sends/edits when on
    (best effort; never required for private controls to work)."""
    enabled = getattr(self, "_rich_controls_enabled", None)
    try:
        return bool(enabled()) if callable(enabled) else False
    except Exception:
        return False


def _ephemeral_fields(message: Any) -> Optional[Tuple[int, int, Any]]:
    """``(receiver_user_id, ephemeral_message_id, chat_id)`` of an ephemeral Message, else None.

    Ephemeral messages carry ``receiver_user`` + ``ephemeral_message_id`` in
    ``api_kwargs`` (PTB 22.8); dict form (raw API echo) is accepted too.
    """
    if message is None:
        return None
    if isinstance(message, dict):
        kw = dict(message.get("api_kwargs") or {})
        kw.update({k: message[k] for k in ("receiver_user", "ephemeral_message_id") if k in message})
        chat = message.get("chat")
        chat_id = chat.get("id") if isinstance(chat, dict) else message.get("chat_id")
    else:
        kw = dict(getattr(message, "api_kwargs", None) or {})
        receiver = getattr(message, "receiver_user", None)
        if receiver is not None and "receiver_user" not in kw:
            kw["receiver_user"] = receiver
        chat_id = getattr(message, "chat_id", None)
        if chat_id is None:
            chat_id = getattr(getattr(message, "chat", None), "id", None)
    receiver_user = kw.get("receiver_user") or {}
    try:
        receiver = int(receiver_user.get("id") if isinstance(receiver_user, dict) else getattr(receiver_user, "id", None))
        ephemeral = int(kw.get("ephemeral_message_id"))
    except (TypeError, ValueError):
        return None
    if receiver <= 0 or ephemeral <= 0:  # message_id is 0 for ephemeral; ids are positive
        return None
    return receiver, ephemeral, chat_id


def parse_private_handle(message_id: Any) -> Optional[Tuple[int, int]]:
    """``eph:<receiver>:<ephemeral>`` → (receiver, ephemeral); rejects non-positive halves."""
    if not isinstance(message_id, str) or not message_id.startswith(EPH_PREFIX):
        return None
    parts = message_id.split(":")
    if len(parts) != 3:
        return None
    try:
        receiver, ephemeral = int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if receiver <= 0 or ephemeral <= 0:
        return None
    return receiver, ephemeral


def is_private_control_handle(message_id: Any) -> bool:
    """True when a message id is an ``eph:`` private-control handle."""
    return parse_private_handle(message_id if isinstance(message_id, str) else None) is not None


def _record_key(chat_id: Any, receiver_user_id: int, ephemeral_message_id: int) -> Tuple[Any, int, int]:
    """Registry key: chat + receiver + ephemeral. The ephemeral id is reusable after
    delete/expiry and the same id may exist in two groups simultaneously — the chat
    disambiguates, and both stay addressable."""
    return (normalize_telegram_chat_id(chat_id), receiver_user_id, ephemeral_message_id)


class _PrivateRecord(dict):
    """Registry record: chat-scoped address plus metadata for later edits/deletes."""

    def __init__(self, chat_id: Any, receiver_user_id: int, ephemeral_message_id: int,
                 metadata: Optional[Dict[str, Any]] = None, callback_query_id: Optional[str] = None):
        super().__init__(chat_id=chat_id, receiver_user_id=receiver_user_id,
                         ephemeral_message_id=ephemeral_message_id,
                         metadata=dict(metadata or {}), callback_query_id=callback_query_id)
        self.handle = f"{EPH_PREFIX}{receiver_user_id}:{ephemeral_message_id}"

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class PrivateQuery:
    """CallbackQuery facade whose message is ephemeral (requester-only).

    Identity attributes delegate to the wrapped query; ``edit_message_text`` and
    ``delete`` route to the raw ephemeral endpoints. NEVER issues a public
    send/edit — failure degrades to nothing, never to the shared chat message.
    Rich buttons: only when the adapter's ``extra.rich_controls`` is on — the
    ``_rich_control_payload(text, parse_mode=…, reply_markup=…)`` helper
    (rich-control module) converts to ``InputRichMessage`` and the edit carries
    ``rich_message`` alone (never alongside text/parse_mode/reply_markup);
    without the flag (or the helper), the edit is plain/parse_mode text.
    """

    def __init__(self, query: Any, adapter: Any, record: _PrivateRecord):
        object.__setattr__(self, "_query", query)
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "record", record)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._query, name)  # only reached for non-facade attrs

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name == "record":
            object.__setattr__(self, name, value)
        else:
            setattr(self._query, name, value)

    @property
    def is_private_control(self) -> bool:
        return True

    def receiver_matches(self) -> bool:
        """True when the tap comes from the record's receiver (layers ON TOP of the allowlist)."""
        from_user = getattr(self._query, "from_user", None)
        try:
            return int(getattr(from_user, "id", 0)) == self.record["receiver_user_id"]
        except (TypeError, ValueError):
            return False

    async def answer_receivers_only(self) -> None:
        """Per-tapper toast: the only public-surface response a mismatched tap gets."""
        try:
            await _bounded_api_call(
                self._query.answer(text=_RECEIVERS_ONLY_TOAST[:TOAST_LIMIT]),
                label="telegram-private-control-toast")
        except Exception:
            pass  # the toast is best-effort; a failed one must not break the gate

    async def edit_message_text(self, *args: Any, text: Optional[str] = None, parse_mode: Any = None,
                                reply_markup: Any = None, **kwargs: Any) -> bool:
        """``editEphemeralMessageText`` — never the shared message, never a ``message_id``.

        Rich: gated on ``extra.rich_controls``; a rich edit carries ``rich_message``
        EXCLUSIVELY. A permanent rich rejection (adapter's ``_is_rich_fallback_error``
        classifying THIS call's own failure — per-call result, no adapter-wide
        last-error slot that a concurrent edit could stale-poison) retries the same
        edit as plain text (fixed message id — idempotent, no duplicate risk); a
        transient/ambiguous rich failure returns False — the rich edit may have landed,
        so a plain edit could race it. No text AND no rich payload → refused.
        """
        if text is None and args:
            text = args[0]
        if not self.receiver_matches():
            await self.answer_receivers_only()
            return False
        rich = None
        if text is not None and _rich_embedding_enabled(self._adapter):
            helper = getattr(self._adapter, "_rich_control_payload", None)
            if callable(helper):
                try:
                    rich = helper(text, parse_mode=parse_mode, reply_markup=reply_markup)
                except Exception:
                    rich = None
        if rich:
            ok, exc = await self._adapter.edit_ephemeral_control_text(
                self.record, None, rich_message=rich, return_error=True)
            if ok:
                return True
            classifier = getattr(self._adapter, "_is_rich_fallback_error", None)
            if not (callable(classifier) and exc is not None and classifier(exc)):
                return False  # transient/ambiguous: never race a plain edit
        if text is None:
            return False
        return await self._adapter.edit_ephemeral_control_text(
            self.record, text, parse_mode=parse_mode, reply_markup=reply_markup)

    async def edit_message_media(self, media: Any, reply_markup: Any = None) -> bool:
        """``editEphemeralMessageMedia`` — receiver-gated, never a public editMessageMedia.

        ``media`` may be a PTB ``InputMedia*`` (file id / HTTPS URL / newly
        uploaded local file via PTB 22.8 ``InputFile`` — the module's media-edit
        endpoint handles the multipart extraction) or a JSON-ready dict.
        """
        if not self.receiver_matches():
            await self.answer_receivers_only()
            return False
        return await self._adapter.edit_ephemeral_control_media(self.record, media, reply_markup=reply_markup)

    async def edit_message_caption(self, caption: Optional[str], parse_mode: Any = None,
                                   show_caption_above_media: Any = None,
                                   reply_markup: Any = None) -> bool:
        """``editEphemeralMessageCaption`` — receiver-gated, never a public editMessageCaption."""
        if not self.receiver_matches():
            await self.answer_receivers_only()
            return False
        return await self._adapter.edit_ephemeral_control_caption(
            self.record, caption, parse_mode=parse_mode,
            show_caption_above_media=show_caption_above_media, reply_markup=reply_markup)

    async def delete(self) -> bool:
        """``deleteEphemeralMessage`` for the wrapped ephemeral control message."""
        if not self.receiver_matches():
            await self.answer_receivers_only()
            return False
        return await self._adapter.delete_ephemeral_control(self.record)


class TelegramPrivateControlsMixin:
    """Requester-only group controls (``extra.private_controls``) via ephemeral messages."""

    def _private_controls_enabled(self) -> bool:
        flag = getattr(self, "_private_controls", None)
        if flag is not None:
            return bool(flag)
        coerce = getattr(self, "_coerce_bool_extra", None)  # object.__new__ test adapters
        try:
            return bool(coerce("private_controls", False)) if callable(coerce) else False
        except Exception:
            return False

    def private_control_requested(self, chat_id: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Whether this control send must attempt the requester-only ephemeral path.

        True iff ``extra.private_controls`` is on and the source chat is a group or
        forum (DMs/channels excluded). Requester-INDEPENDENT: a configured group
        without the stamp still requests the private path, and the send FAILS CLOSED.
        """
        if not self._private_controls_enabled():
            return False
        chat_type = str((metadata or {}).get("telegram_chat_type") or (metadata or {}).get("chat_type") or "")
        return chat_type.strip().lower() in GROUP_CHAT_TYPES

    # ---------------------------------------------------------------- registry

    def _private_records(self) -> Dict[Tuple[Any, int, int], _PrivateRecord]:
        if not hasattr(self, "_private_control_records"):
            self._private_control_records: Dict[Tuple[Any, int, int], _PrivateRecord] = {}
        return self._private_control_records

    def _lookup_private_record(self, chat_id: Any, receiver_user_id: int, ephemeral_message_id: int) -> Optional[_PrivateRecord]:
        """Registry hit requires the full ``(chat, receiver, ephemeral)`` key."""
        return self._private_records().get(_record_key(chat_id, receiver_user_id, ephemeral_message_id))

    def resolve_private_control(self, obj: Any, chat_id: Any = None) -> Optional[_PrivateRecord]:
        """Record for a handle/record/message; None for non-private or chat-mismatched input.

        A handle known under a different chat refuses (ephemeral ids are reusable
        across chats); a handle with no chat known refuses rather than guessing.
        The caller keeps the regular (public) path untouched on None.
        """
        if obj is None:
            return None
        if isinstance(obj, _PrivateRecord):
            return obj
        if isinstance(obj, str):
            parsed = parse_private_handle(obj)
            if parsed is None:
                return None
            receiver, ephemeral = parsed
            if chat_id is None:
                return None  # no chat known — refuse rather than guessing
            record = self._lookup_private_record(chat_id, receiver, ephemeral)
            if record is not None:
                return record
            # The registry key includes the chat: an ``eph:`` handle from chat A
            # must never drive an edit in chat B, and an unknown (receiver,
            # ephemeral, chat) triple still has a valid API address.
            return _PrivateRecord(chat_id, receiver, ephemeral)
        fields = _ephemeral_fields(obj)
        if fields is None:
            return None
        receiver, ephemeral, obj_chat = fields
        if chat_id is None:
            chat_id = obj_chat
        record = self._lookup_private_record(chat_id, receiver, ephemeral) if chat_id is not None else None
        if record is not None:
            return record
        if chat_id is None:
            return None  # not a known private control and no addressable chat
        return _PrivateRecord(chat_id, receiver, ephemeral)

    # -------------------------------------------------------------------- send

    async def _send_private_control(self, kwargs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Send one control as a requester-only ephemeral message; returns a message-like
        handle (opaque ``eph:<receiver>:<ephemeral>``, scoped chat, feeds ``on_sent``).

        RAISES ``PrivateControlError`` on ANY failure — never a falsy success-shape
        (that would make the caller fall back to the public send and leak the
        requester-only content to the group).
        """
        if not self._bot:
            raise PrivateControlError("not_connected")
        meta = metadata or {}
        try:
            receiver = int(str(meta.get("telegram_requester_user_id")).strip())
        except (TypeError, ValueError):
            receiver = 0
        if receiver <= 0:
            raise PrivateControlError("missing_requester_user_id")  # fail closed

        send_kwargs = dict(kwargs)
        ephemeral_params: Dict[str, Any] = {"receiver_user_id": receiver}
        callback_query_id = str(meta.get("telegram_callback_query_id") or "").strip()
        # Ephemeral-origin flag comes ONLY from parent-stamped trusted update state
        # (``query.message`` being ephemeral), never inferred from user text.
        callback_from_ephemeral = bool(meta.get("telegram_callback_from_ephemeral"))
        if callback_query_id:
            # Non-admin delivery needs a 15s trigger (callback query or ephemeral
            # reply); admins send any time. We do NOT preflight privilege —
            # Telegram rejects non-privileged sends and we report them.
            ephemeral_params["callback_query_id"] = callback_query_id
            if not callback_from_ephemeral:
                # Overlay: show this private content IN PLACE OF the public message
                # the fresh callback query came from. Must stay False for callbacks
                # from ephemeral messages (API contract) — those edit instead.
                ephemeral_params["replace_callback_query_message"] = True
        # A regular reply anchor is meaningless on an ephemeral send; an ``eph:``
        # anchor becomes the ephemeral reply_parameters form.
        anchor = send_kwargs.pop("reply_to_message_id", None)
        # Replacement flow from an EPHEMERAL callback: never a duplicate send.
        # The parent stamps ``telegram_ephemeral_reply_id`` (or passes the
        # ``eph:`` reply anchor) with the same ephemeral message the callback hit;
        # when we can resolve that anchor to a record, the replacement flow EDITS
        # it instead. The caller may also force the edit path via the explicit
        # ``telegram_replaces_ephemeral_id`` stamp when the anchor is known.
        replaces_ephemeral = None
        if callback_from_ephemeral:
            explicit = str(meta.get("telegram_replaces_ephemeral_id") or "").strip()
            parsed_replaces = parse_private_handle(explicit) or parse_private_handle(
                str(meta.get("telegram_ephemeral_reply_id") or anchor or ""))
            if parsed_replaces is not None:
                if parsed_replaces[0] == receiver:
                    replaces_ephemeral = parsed_replaces[1]
                elif explicit:
                    # An EXPLICIT replacement anchor for another receiver's
                    # ephemeral message is inconsistent trusted state — fail
                    # closed rather than silently re-addressing it or sending.
                    raise PrivateControlError("ephemeral_anchor_receiver_mismatch")
        if "reply_parameters" not in send_kwargs:
            reply_ephemeral = parse_private_handle(str(meta.get("telegram_ephemeral_reply_id") or anchor))
            if reply_ephemeral is not None:
                send_kwargs["reply_parameters"] = {"ephemeral_message_id": reply_ephemeral[1]}
        # Non-destructive merge: a caller-supplied ``api_kwargs`` dict keeps its
        # own keys (PTB 22.8 ``Bot._post`` shallow-merges it into the request).
        merged_api_kwargs = {**(send_kwargs.pop("api_kwargs", None) or {}),
                             "ephemeral_message_parameters": ephemeral_params}
        text = send_kwargs.get("text")

        # Ephemeral-origin replacement: prefer editing the same ephemeral message (a fresh send could
        # duplicate requester-only content; the API forbids replace_callback_query_message there).
        # The edit is the whole flow — a failed API reply may already have been delivered, never resend.
        if replaces_ephemeral is not None:
            edit_record = self._lookup_private_record(
                send_kwargs.get("chat_id"), receiver, replaces_ephemeral) or _PrivateRecord(
                send_kwargs.get("chat_id"), receiver, replaces_ephemeral)
            rich_replacement = None
            if _rich_embedding_enabled(self) and text is not None:
                supported = getattr(self, "_rich_control_markup_supported", None)
                helper = getattr(self, "_rich_control_payload", None)
                if not (callable(supported) and not supported(send_kwargs.get("reply_markup"))) and callable(helper):
                    try:
                        rich_replacement = helper(text, parse_mode=send_kwargs.get("parse_mode"),
                                                   reply_markup=send_kwargs.get("reply_markup"))
                    except Exception:
                        rich_replacement = None
            edited, error = await self.edit_ephemeral_control_text(
                edit_record, text, parse_mode=send_kwargs.get("parse_mode"),
                reply_markup=send_kwargs.get("reply_markup"), rich_message=rich_replacement, return_error=True)
            classifier = getattr(self, "_is_rich_fallback_error", None)
            if not edited and rich_replacement and error is not None and callable(classifier) and classifier(error):
                edited = await self.edit_ephemeral_control_text(
                    edit_record, text, parse_mode=send_kwargs.get("parse_mode"), reply_markup=send_kwargs.get("reply_markup"))
            if not edited:
                raise PrivateControlError("ephemeral_replacement_edit_failed")
            return _HandleFacade(edit_record)

        # Rich embed (``extra.rich_controls``): one ``sendRichMessage`` with
        # buttons as native ``<tg-button-row>`` blocks. Never mandatory — a
        # permanent/capability rejection (adapter classifier) falls back to the
        # plain ephemeral ``sendMessage``; ANY other failure raises (fail closed:
        # the request may have reached Telegram, so a second send could duplicate
        # the requester-only content).
        if _rich_embedding_enabled(self) and text is not None:
            supported = getattr(self, "_rich_control_markup_supported", None)
            if callable(supported) and not supported(send_kwargs.get("reply_markup")):
                rich = None
            else:
                helper = getattr(self, "_rich_control_payload", None)
                try:
                    rich = helper(text, parse_mode=send_kwargs.get("parse_mode"),
                                  reply_markup=send_kwargs.get("reply_markup")) if callable(helper) else None
                except Exception:
                    rich = None
            if rich:
                rich_payload: Dict[str, Any] = {
                    "chat_id": send_kwargs.get("chat_id"), "rich_message": rich,
                    "ephemeral_message_parameters": ephemeral_params}
                for key in ("reply_parameters", "message_thread_id", "direct_messages_topic_id",
                            "disable_notification"):
                    if send_kwargs.get(key) is not None:
                        rich_payload[key] = send_kwargs[key]
                preview = self._rich_control_link_preview(send_kwargs) if callable(
                    getattr(self, "_rich_control_link_preview", None)) else {}
                rich_payload.update(preview)
                try:
                    message = await _bounded_api_call(
                        self._bot.do_api_request("sendRichMessage", api_kwargs=rich_payload),
                        label="telegram-private-rich-control")
                except Exception as exc:
                    classifier = getattr(self, "_is_rich_fallback_error", None)
                    permanent = callable(classifier) and classifier(exc)
                    if not permanent:
                        raise PrivateControlError(_redact_private_error(exc)) from exc
                    logger.debug(
                        "[%s] private control sendRichMessage rejected (%s) — plain ephemeral send",
                        getattr(self, "name", "telegram"), _redact_private_error(exc))
                else:
                    fields = _ephemeral_fields(message)
                    if fields is not None and fields[1] > 0:
                        return self._register_private_record(
                            fields, send_kwargs.get("chat_id"), meta, callback_query_id)
                    raise PrivateControlError("missing_ephemeral_message_id")

        send_kwargs["api_kwargs"] = merged_api_kwargs
        try:
            message = await _bounded_api_call(
                self._bot.send_message(**send_kwargs), label="telegram-private-control")
        except Exception as exc:
            raise PrivateControlError(_redact_private_error(exc)) from exc

        fields = _ephemeral_fields(message)
        if fields is None or fields[1] <= 0:
            # "Success" without an ephemeral id is unmanageable (no edit/delete
            # target) — fail closed, never public-fallback, never pretend delivered.
            raise PrivateControlError("missing_ephemeral_message_id")
        return self._register_private_record(fields, send_kwargs.get("chat_id"), meta, callback_query_id)

    def _register_private_record(self, fields: Tuple[int, int, Any], chat_id: Any,
                                 meta: Dict[str, Any], callback_query_id: Optional[str]) -> _HandleFacade:
        """Registry bookkeeping shared by the rich and plain send paths."""
        receiver_id, ephemeral_id, _msg_chat = fields
        record = _PrivateRecord(chat_id, receiver_id, ephemeral_id, metadata=meta,
                                callback_query_id=callback_query_id or None)
        bounded_put(self._private_records(), _record_key(chat_id, receiver_id, ephemeral_id), record, 1024)
        return _HandleFacade(record)

    async def send_private_control_prompt(self, kwargs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None,
                                          on_sent: Any = None) -> SendResult:
        """SendResult wrapper. EVERY failure of a requested private control is a
        uniform egress decline (privacy boundary, not a network fact — no retry,
        no transient kind, no public fallback):
        ``raw_response={"success": False, "code": "egress_declined", "error":
        DECLINE_ERROR}`` plus the same ``error`` text, the exact shape
        ``gateway.relay.egress.declined_send`` recognizes, so the gateway's
        existing fallback suppression fires and nothing is public-posted. Never
        ``{"private_control": True, "declined": True}`` (not recognized)."""
        try:
            handle = await self._send_private_control(kwargs, metadata)
        except PrivateControlError as exc:
            logger.warning("[%s] private control send declined (%s) — no public fallback", getattr(self, "name", "telegram"), exc)
            return SendResult(
                success=False, error=DECLINE_ERROR, retryable=False,
                raw_response={"success": False, "code": "egress_declined", "error": DECLINE_ERROR})
        if on_sent is not None:
            try:
                on_sent(handle)
            except Exception:
                # Delivered; a state-hook failure is non-fatal and never public-fallback material.
                logger.debug("[%s] private control on_sent hook failed", getattr(self, "name", "telegram"), exc_info=True)
        return SendResult(success=True, message_id=handle.message_id, raw_response={"private_control": True})

    # ------------------------------------------------------------- edit/delete

    async def edit_ephemeral_control_text(self, record_or_handle: Any, text: Optional[str], *,
                                          chat_id: Any = None, parse_mode: Any = None, reply_markup: Any = None,
                                          rich_message: Optional[Dict[str, Any]] = None,
                                          return_error: bool = False):
        """``editEphemeralMessageText`` for a private control. Returns False on failure
        (quiet-edit call sites treat it as non-fatal); never retries via editMessageText.

        Content fields are EXCLUSIVE: ``rich_message`` alone when given (never
        alongside text/parse_mode/reply_markup — the API takes one content form);
        otherwise plain ``text``/``parse_mode``/``reply_markup``.

        ``return_error=True`` returns ``(ok, exc_or_None)`` PER CALL so the rich
        caller can classify permanent vs transient without any adapter-wide
        last-error slot (a stale global would misroute a concurrent edit's
        failure into this call's retry decision).
        """
        record = self.resolve_private_control(
            record_or_handle, chat_id=chat_id if chat_id is not None else (
                record_or_handle.get("chat_id") if isinstance(record_or_handle, _PrivateRecord) else None))
        if record is None or record["ephemeral_message_id"] <= 0:
            logger.debug("[%s] private control edit refused: no valid ephemeral target", getattr(self, "name", "telegram"))
            return (False, None) if return_error else False
        if text is None and rich_message is None:
            return (False, None) if return_error else False
        if not self._bot:
            return (False, None) if return_error else False
        payload: Dict[str, Any] = {
            "chat_id": record["chat_id"], "receiver_user_id": record["receiver_user_id"],
            "ephemeral_message_id": record["ephemeral_message_id"]}
        if rich_message:
            payload["rich_message"] = rich_message
        else:
            if text is not None:
                payload["text"] = str(text)
            if parse_mode is not None:
                payload["parse_mode"] = str(getattr(parse_mode, "value", parse_mode))
            if reply_markup is not None:
                payload["reply_markup"] = _serialize_markup(reply_markup)
        try:
            await _bounded_api_call(
                self._bot.do_api_request("editEphemeralMessageText", api_kwargs=payload),
                label="telegram-private-control-edit")
            return (True, None) if return_error else True
        except Exception as exc:
            logger.debug(
                "[%s] editEphemeralMessageText failed (chat=%s receiver=%s ephemeral=%s): %s",
                getattr(self, "name", "telegram"), record["chat_id"], record["receiver_user_id"],
                record["ephemeral_message_id"], _redact_private_error(exc))
            return (False, exc) if return_error else False

    async def edit_ephemeral_control_media(self, record_or_handle: Any, media: Any, *,
                                           chat_id: Any = None, reply_markup: Any = None) -> bool:
        """``editEphemeralMessageMedia`` for a private control (Bot API 10.3).

        ``media`` accepts a PTB ``InputMedia*`` object (file id, HTTPS URL, or a
        newly uploaded local file via a PTB 22.8 ``InputFile`` as its
        ``media`` argument — ``do_api_request``'s ``api_kwargs["media"]`` rides
        ``RequestParameter.from_input`` which extracts the file into multipart
        data; PTB has no separate ``files`` parameter on ``do_api_request``) or
        an already JSON-serializable dict. Failures return False — never a
        public editMessageMedia fallback. Same receiver gate as text edits.
        """
        record = self.resolve_private_control(record_or_handle, chat_id=chat_id)
        if record is None or record["ephemeral_message_id"] <= 0:
            logger.debug("[%s] private control media edit refused: no valid ephemeral target", getattr(self, "name", "telegram"))
            return False
        if media is None:
            return False
        if not self._bot:
            return False
        payload: Dict[str, Any] = {
            "chat_id": record["chat_id"], "receiver_user_id": record["receiver_user_id"],
            "ephemeral_message_id": record["ephemeral_message_id"],
            "media": _serialize_media(media)}
        if reply_markup is not None:
            payload["reply_markup"] = _serialize_markup(reply_markup)
        try:
            await _bounded_api_call(
                self._bot.do_api_request("editEphemeralMessageMedia", api_kwargs=payload),
                label="telegram-private-control-media-edit")
            return True
        except Exception as exc:
            logger.debug(
                "[%s] editEphemeralMessageMedia failed (chat=%s receiver=%s ephemeral=%s): %s",
                getattr(self, "name", "telegram"), record["chat_id"], record["receiver_user_id"],
                record["ephemeral_message_id"], _redact_private_error(exc))
            return False

    async def edit_ephemeral_control_caption(self, record_or_handle: Any, caption: Optional[str], *,
                                             chat_id: Any = None, parse_mode: Any = None,
                                             show_caption_above_media: Any = None,
                                             reply_markup: Any = None) -> bool:
        """``editEphemeralMessageCaption`` for a private control (Bot API 10.3).
        Failures return False — never a public editMessageCaption fallback."""
        record = self.resolve_private_control(record_or_handle, chat_id=chat_id)
        if record is None or record["ephemeral_message_id"] <= 0:
            logger.debug("[%s] private control caption edit refused: no valid ephemeral target", getattr(self, "name", "telegram"))
            return False
        if caption is None:
            return False
        if not self._bot:
            return False
        payload: Dict[str, Any] = {
            "chat_id": record["chat_id"], "receiver_user_id": record["receiver_user_id"],
            "ephemeral_message_id": record["ephemeral_message_id"], "caption": str(caption)}
        if parse_mode is not None:
            payload["parse_mode"] = str(getattr(parse_mode, "value", parse_mode))
        if show_caption_above_media is not None:
            payload["show_caption_above_media"] = bool(show_caption_above_media)
        if reply_markup is not None:
            payload["reply_markup"] = _serialize_markup(reply_markup)
        try:
            await _bounded_api_call(
                self._bot.do_api_request("editEphemeralMessageCaption", api_kwargs=payload),
                label="telegram-private-control-caption-edit")
            return True
        except Exception as exc:
            logger.debug(
                "[%s] editEphemeralMessageCaption failed (chat=%s receiver=%s ephemeral=%s): %s",
                getattr(self, "name", "telegram"), record["chat_id"], record["receiver_user_id"],
                record["ephemeral_message_id"], _redact_private_error(exc))
            return False

    async def delete_ephemeral_control(self, record_or_handle: Any, *, chat_id: Any = None) -> bool:
        """``deleteEphemeralMessage`` for a private control; never a public delete."""
        record = self.resolve_private_control(record_or_handle, chat_id=chat_id)
        if record is None or record["ephemeral_message_id"] <= 0:
            logger.debug("[%s] private control delete refused: no valid ephemeral target", getattr(self, "name", "telegram"))
            return False
        if not self._bot:
            return False
        payload: Dict[str, Any] = {
            "chat_id": record["chat_id"], "receiver_user_id": record["receiver_user_id"],
            "ephemeral_message_id": record["ephemeral_message_id"]}
        try:
            await _bounded_api_call(
                self._bot.do_api_request("deleteEphemeralMessage", api_kwargs=payload),
                label="telegram-private-control-delete")
            return True
        except Exception as exc:
            logger.debug(
                "[%s] deleteEphemeralMessage failed (chat=%s receiver=%s ephemeral=%s): %s",
                getattr(self, "name", "telegram"), record["chat_id"], record["receiver_user_id"],
                record["ephemeral_message_id"], _redact_private_error(exc))
            return False

    # ------------------------------------------------------------- inbound wrap

    def wrap_private_control_query(self, query: Any) -> Optional[PrivateQuery]:
        """Facade for a callback query on an ephemeral message; None keeps the public path.

        The wrapped record's metadata is stamped from the TRUSTED update only:
        ``telegram_callback_from_ephemeral`` = True (the tap's message is itself
        ephemeral — its ``message_id == 0`` with receiver/ephemeral fields, which
        is the ephemeral incoming-command case) and ``telegram_callback_query_id``
        = ``query.id``. The parent copies these into the send metadata for the
        replacement flow; nothing here is inferred from user text.
        """
        if query is None:
            return None
        fields = _ephemeral_fields(getattr(query, "message", None))
        if fields is None:
            return None
        receiver, ephemeral, chat_id = fields
        record = self._lookup_private_record(chat_id, receiver, ephemeral)
        if record is None:
            record = _PrivateRecord(chat_id, receiver, ephemeral)
        try:
            record["metadata"]["telegram_callback_from_ephemeral"] = True
            record["metadata"]["telegram_callback_query_id"] = str(getattr(query, "id", "") or "")
            record["metadata"]["telegram_requester_user_id"] = str(receiver)
        except Exception:
            pass  # bookkeeping only; the gate does not depend on it
        return PrivateQuery(query, self, record)

    def callback_receiver_matches(self, query: Any, record: _PrivateRecord) -> bool:
        """True when the tap's user is the receiver the ephemeral control addresses."""
        from_user = getattr(query, "from_user", None)
        try:
            return int(getattr(from_user, "id", 0)) == record["receiver_user_id"]
        except (TypeError, ValueError):
            return False

    async def gate_private_control_query(self, wrapped: Optional[PrivateQuery]) -> bool:
        """Receiver gate BEFORE ``_accept_update`` and the allowlist auth. None (regular
        public control) passes — the allowlist decides. Mismatched tap: toast only,
        pending state untouched, handler never runs."""
        if wrapped is None:
            return True
        if wrapped.receiver_matches():
            return True
        await wrapped.answer_receivers_only()
        return False


class _HandleFacade:
    """Message-like handle for a sent ephemeral control (feeds ``on_sent`` closures).

    ``message_id`` is the opaque ``eph:<receiver>:<ephemeral>`` (never a regular
    id — those are 0 for ephemeral); ``chat`` is scoped to the id alone.
    """

    def __init__(self, record: _PrivateRecord):
        object.__setattr__(self, "_record", record)

    def __getattr__(self, name: str) -> Any:
        if name == "message_id":
            return self._record.handle
        if name == "chat_id":
            return self._record["chat_id"]
        if name == "chat":
            return SimpleNamespace(id=self._record["chat_id"])
        if name == "receiver_user":
            return SimpleNamespace(id=self._record["receiver_user_id"])
        if name == "ephemeral_message_id":
            return self._record["ephemeral_message_id"]
        raise AttributeError(name)


def _serialize_markup(reply_markup: Any) -> Any:
    """``to_dict()`` when available (PTB InlineKeyboardMarkup), else the value as-is."""
    to_dict = getattr(reply_markup, "to_dict", None)
    if callable(to_dict):
        try:
            return reply_markup.to_dict()
        except Exception:
            return reply_markup
    return reply_markup

def _serialize_media(media: Any) -> Any:
    """Pass ``media`` through for the media-edit endpoint, untouched.

    PTB 22.8 ``RequestParameter.from_input`` serializes ``InputMedia*`` objects
    and hoists an ``InputFile`` nested in one into multipart data — but NOT one
    nested inside a plain dict (what a ``to_dict()`` here would produce: the
    ``InputFile`` instance would sit in an unhoisted dict's ``media`` key). So
    ``media`` rides as the CALLER passed it: a PTB ``InputMedia*`` object, a
    JSON-ready dict (``{"type": "photo", "media": "file-id"}``), or a string.
    No type import, no isinstance, no pre-serialization.
    """
    return media
