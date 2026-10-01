"""Yuanbao inbound middleware and message preparation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Tuple

from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.access_policy_mixin import OwnAccessPolicyMixin

logger = logging.getLogger("gateway.platforms.yuanbao")


@dataclass
class InboundContext:
    """Mutable context passed through every inbound middleware in registration order."""
    adapter: Any  # YuanbaoAdapter (forward-ref avoids circular import)
    raw_frames: list = dc_field(default_factory=list)  # debounce-aggregated raw frames
    push: Optional[dict] = None  # DecodeMiddleware
    decoded_via: str = ""  # "json" | "protobuf"
    from_account: str = ""  # ExtractFieldsMiddleware …
    group_code: str = ""
    group_name: str = ""
    sender_nickname: str = ""
    msg_body: list = dc_field(default_factory=list)
    msg_id: str = ""
    cloud_custom_data: str = ""
    chat_id: str = ""  # ChatRoutingMiddleware …
    chat_type: str = ""  # "dm" | "group"
    chat_name: str = ""
    raw_text: str = ""  # ExtractContentMiddleware …
    media_refs: list = dc_field(default_factory=list)
    forwarded_records: Optional[dict] = None  # parsed ForwardMsgData for elem_type 1009
    owner_command: Optional[str] = None  # OwnerCommandMiddleware
    source: Optional[Any] = None  # SessionSource, BuildSourceMiddleware
    msg_type: Optional[Any] = None  # MessageType | YuanbaoMessageType, ClassifyMessageTypeMiddleware
    reply_to_message_id: Optional[str] = None  # QuoteContextMiddleware …
    reply_to_text: Optional[str] = None
    quote_media_refs: list = dc_field(default_factory=list)  # (rid, kind, filename)
    # MediaResolveMiddleware: deduped local paths — own media, then quoted media, else group-observed media
    media_urls: list = dc_field(default_factory=list)
    media_types: list = dc_field(default_factory=list)
    channel_prompt: Optional[str] = None  # GroupAttributionMiddleware


class InboundMiddleware(ABC):
    """Set class-level ``name`` and implement ``handle(ctx, next_fn)``; ``await next_fn()``
    continues the pipeline, returning without it stops."""
    name: str = ""

    @abstractmethod
    async def handle(self, ctx: InboundContext, next_fn: Callable) -> None: ...

    async def __call__(self, ctx: InboundContext, next_fn: Callable) -> None:
        return await self.handle(ctx, next_fn)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class InboundPipeline:
    """Onion-model middleware pipeline: named middlewares, ``when`` guards, use_before/use_after/
    remove. Accepts ``InboundMiddleware`` instances or plain ``async def(ctx, next_fn)`` callables."""
    def __init__(self) -> None:
        self._middlewares: list = []  # (name, handler, when_fn | None)

    @staticmethod
    def _normalize(name_or_mw, handler=None):
        if isinstance(name_or_mw, InboundMiddleware):
            return name_or_mw.name, name_or_mw
        return name_or_mw, handler

    def use(self, name_or_mw, handler=None, when=None) -> InboundPipeline:
        """Append ``pipeline.use(SomeMiddleware())`` or ``pipeline.use("name", fn)``."""
        return self._insert_relative(None, 0, name_or_mw, handler, when)

    def _insert_relative(self, target: Optional[str], offset: int, name_or_mw, handler, when) -> InboundPipeline:
        """Insert at index(target)+offset; appends when *target* is None or not registered."""
        name, h = self._normalize(name_or_mw, handler)
        idx = next((i for i, (n, _, _) in enumerate(self._middlewares) if n == target), None)
        self._middlewares.insert(len(self._middlewares) if idx is None else idx + offset, (name, h, when))
        return self

    def use_before(self, target: str, name_or_mw, handler=None, when=None) -> InboundPipeline:
        return self._insert_relative(target, 0, name_or_mw, handler, when)

    def use_after(self, target: str, name_or_mw, handler=None, when=None) -> InboundPipeline:
        return self._insert_relative(target, 1, name_or_mw, handler, when)

    def remove(self, name: str) -> InboundPipeline:
        self._middlewares = [(n, h, w) for n, h, w in self._middlewares if n != name]
        return self

    @property
    def middleware_names(self) -> list:
        return [n for n, _, _ in self._middlewares]

    async def execute(self, ctx: InboundContext) -> None:
        """Run the chain; each middleware receives ``(ctx, next_fn)``."""
        chain = self._middlewares
        index = 0

        async def next_fn() -> None:
            nonlocal index
            while index < len(chain):
                name, handler, when_fn = chain[index]
                index += 1
                if when_fn is not None and not when_fn(ctx):
                    continue
                try:
                    await handler(ctx, next_fn)
                except Exception:
                    logger.error("[InboundPipeline] middleware [%s] error", name, exc_info=True)
                    raise
                return
        await next_fn()


class DecodeMiddleware(InboundMiddleware):
    """Decode raw inbound frames (JSON or protobuf via ``decode_inbound_push``) into ctx.push."""
    name = "decode"

    @staticmethod
    def convert_json_msg_body(raw_body: list) -> list:
        """Normalize JSON msg_body (PascalCase or snake_case keys) to [{"msg_type", "msg_content"}]."""
        from gateway.platforms.yuanbao import (
            json,
        )

        result = []
        for item in raw_body or []:
            if not isinstance(item, dict):
                continue
            msg_type = item.get("msg_type") or item.get("MsgType", "")
            msg_content = item.get("msg_content") or item.get("MsgContent", {})
            if isinstance(msg_content, str):
                try:
                    msg_content = json.loads(msg_content)
                except Exception:
                    msg_content = {"text": msg_content}
            result.append({"msg_type": msg_type, "msg_content": msg_content or {}})
        return result

    @staticmethod
    def json_sender_fields(raw_json: dict) -> tuple[str, str]:
        """(from_account, group_code) accepting both Tencent IM PascalCase and internal snake_case keys."""
        from_account = raw_json.get("from_account", "") or raw_json.get("From_Account", "")
        group_code = raw_json.get("group_code", "") or raw_json.get("GroupId", "") or raw_json.get("group_id", "")
        return from_account, group_code

    @staticmethod
    def parse_json_push(raw_json: dict) -> dict | None:
        """JSON push → dict shaped like ``decode_inbound_push`` output; accepts both the callback
        format (callback_command/from_account/msg_body) and legacy keys (GroupId/MsgSeq/MsgKey/MsgBody)."""
        if not raw_json:
            return None
        from_account, group_code = DecodeMiddleware.json_sender_fields(raw_json)
        msg_body = DecodeMiddleware.convert_json_msg_body(raw_json.get("msg_body", []) or raw_json.get("MsgBody", []))
        # Recall callbacks may have neither from_account nor msg_body.
        if not from_account and not msg_body and not raw_json.get("callback_command"):
            return None
        return {
            "callback_command": raw_json.get("callback_command", ""),
            "from_account": from_account,
            "to_account": raw_json.get("to_account", "") or raw_json.get("To_Account", ""),
            "sender_nickname": raw_json.get("sender_nickname", "") or raw_json.get("nick_name", ""),
            "group_code": group_code,
            "group_name": raw_json.get("group_name", ""),
            "msg_seq": raw_json.get("msg_seq", 0) or raw_json.get("MsgSeq", 0),
            "msg_id": raw_json.get("msg_id", "") or raw_json.get("msg_key", "") or raw_json.get("MsgKey", ""),
            "msg_body": msg_body,
            "cloud_custom_data": raw_json.get("cloud_custom_data", "") or raw_json.get("CloudCustomData", ""),
            "bot_owner_id": raw_json.get("bot_owner_id", "") or raw_json.get("botOwnerId", ""),
            "recall_msg_seq_list": raw_json.get("recall_msg_seq_list") or None,
            "trace_id": (raw_json.get("log_ext") or {}).get("trace_id", "") if isinstance(raw_json.get("log_ext"), dict) else "",
        }

    def _decode_single(self, adapter, data: bytes) -> tuple:
        """One raw frame → (push_dict, decoded_via) or (None, '')."""
        from gateway.platforms.yuanbao import (
            decode_inbound_push,
            json,
        )

        try:
            conn_json = json.loads(data.decode("utf-8"))
        except Exception:
            conn_json = None
        if isinstance(conn_json, dict):
            push = self.parse_json_push(conn_json)
            return (push, "json") if push else (None, "")
        try:
            push = decode_inbound_push(data)
        except Exception:
            push = None
        return (push, "protobuf") if push else (None, "")

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        from gateway.platforms.yuanbao import (
            Optional,
            _text_elem,
        )

        if not ctx.raw_frames:
            return  # Stop pipeline — nothing to decode
        merged: Optional[dict] = None
        for data in ctx.raw_frames:
            push, via = self._decode_single(ctx.adapter, data)
            if not push:
                logger.info("[%s] Push decoded but no valid message. raw hex(first64)=%s",
                            ctx.adapter.name, data.hex()[:128] if data else "(empty)")
            elif merged is None:
                merged, ctx.decoded_via = push, via
                logger.info("[%s] Frame decoded (via=%s): len=%d", ctx.adapter.name, via, len(data))
            elif push.get("msg_body", []):  # subsequent pushes: append msg_body, newline-separated
                merged["msg_body"] = merged.get("msg_body", []) + [_text_elem("\n")] + push["msg_body"]
                logger.info("[%s] Merged %d extra msg_body elements from aggregated push", ctx.adapter.name, len(push["msg_body"]))
        if not merged:
            return  # Stop pipeline
        ctx.push = merged
        logger.info(
            "[%s] Push decoded (via=%s): from=%s group=%s msg_id=%s msg_types=%s",
            ctx.adapter.name, ctx.decoded_via, merged.get("from_account", ""), merged.get("group_code", ""),
            merged.get("msg_id", ""), [e.get("msg_type", "") for e in merged.get("msg_body", [])],
        )
        logger.debug("[%s] Push payload: %s", ctx.adapter.name, ctx.push)
        await next_fn()


class ExtractFieldsMiddleware(InboundMiddleware):
    """Copy common push fields onto ctx."""
    name = "extract-fields"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        for f in ("from_account", "group_code", "group_name", "sender_nickname", "msg_id", "cloud_custom_data"):
            setattr(ctx, f, ctx.push.get(f, ""))
        ctx.msg_body = ctx.push.get("msg_body", [])
        await next_fn()


class DedupMiddleware(InboundMiddleware):
    name = "dedup"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if ctx.msg_id and ctx.adapter._dedup.is_duplicate(ctx.msg_id):
            logger.debug("[%s] Duplicate message ignored: msg_id=%s", ctx.adapter.name, ctx.msg_id)
            return  # Stop pipeline
        await next_fn()


def _session_store(adapter):
    """Adapter's SessionStore, or None before ``set_session_store`` ran."""
    return getattr(adapter, "_session_store", None)


class RecallGuardMiddleware(InboundMiddleware):
    """Recall callbacks (Group.CallbackAfterRecallMsg / C2C.CallbackAfterMsgWithDraw).
    A: in transcript → redact; B: not in transcript → system note; C: being processed → interrupt + delayed redact."""
    name = "recall_guard"
    _RECALL_COMMANDS = frozenset({"Group.CallbackAfterRecallMsg", "C2C.CallbackAfterMsgWithDraw"})
    _REDACTED = "[This message was recalled/withdrawn by the sender; original content removed]"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        cmd = (ctx.push or {}).get("callback_command", "")
        if cmd in self._RECALL_COMMANDS:
            self._handle_recall(ctx, cmd)  # terminal: recalls never dispatch
        else:
            await next_fn()

    @staticmethod
    def _build_source(adapter, group_code: str, from_account: str):
        return adapter.build_source(
            chat_id=(f"group:{group_code}" if group_code else f"direct:{from_account}"),
            chat_type="group" if group_code else "dm",
            user_id=from_account or None,
            thread_id="main" if group_code else None,
        )

    @classmethod
    def _resolve_sid(cls, store, adapter, group_code: str, from_account: str) -> str:
        return store.get_or_create_session(cls._build_source(adapter, group_code, from_account)).session_id

    @classmethod
    def _redact(cls, adapter, store, sid: str, transcript: list, entry: dict, ok_msg: str, fail_msg: str, *ok_args) -> None:
        """Blank *entry* in place and persist *transcript* (warns, never raises)."""
        entry["content"] = cls._REDACTED
        try:
            store.rewrite_transcript(sid, transcript, active_only=True)
            logger.info(ok_msg, adapter.name, *ok_args)
        except Exception as exc:
            logger.warning(fail_msg, adapter.name, exc)

    def _handle_recall(self, ctx: InboundContext, cmd: str) -> None:
        adapter = ctx.adapter
        push = ctx.push or {}
        if cmd == "Group.CallbackAfterRecallMsg":
            seq_list = push.get("recall_msg_seq_list") or []
        else:
            mid, seq = push.get("msg_id") or "", push.get("msg_seq")
            seq_list = [{"msg_id": mid, "msg_seq": seq}] if (mid or seq) else []
        if not seq_list:
            logger.debug("[%s] Recall callback with empty seq_list, skipping", adapter.name)
            return
        group_code = (push.get("group_code") or "").strip()
        from_account = (push.get("from_account") or "").strip()
        for seq_entry in seq_list:
            recalled_id = seq_entry.get("msg_id") or str(seq_entry.get("msg_seq") or "")
            if not recalled_id:
                continue
            matched_sk = self._find_processing_session(adapter, recalled_id)
            if matched_sk is not None:
                self._interrupt_for_recall(adapter, matched_sk, recalled_id, group_code, from_account)
            else:
                self._patch_transcript(adapter, recalled_id, group_code, from_account, adapter._msg_content_cache.get(recalled_id))

    # -- Branch C: interrupt currently-processing message ---------------

    @staticmethod
    def _find_processing_session(adapter, recalled_id: str) -> Optional[str]:
        return next((sk for sk, mid in adapter._processing_msg_ids.items()
                     if mid == recalled_id and sk in adapter._active_sessions), None)

    @classmethod
    def _interrupt_for_recall(cls, adapter, session_key: str, recalled_id: str, group_code: str, from_account: str) -> None:
        from gateway.platforms.yuanbao import (
            MessageEvent,
            MessageType,
        )

        where = f"group {group_code}" if group_code else f"direct chat with {from_account}"
        recall_text = (
            f"[CRITICAL — MESSAGE RECALLED] The user message that triggered your current task "
            f"(message_id=\"{recalled_id}\") in {where} has been recalled/withdrawn by the sender. "
            "IGNORE any prior system note asking you to finish processing tool results — the original request is void. "
            "Do NOT continue the task, do NOT call more tools, do NOT reference the recalled content. "
            "Reply only with a brief acknowledgment such as \"The message has been recalled.\" in the "
            "language the user was using."
        )
        # Set pending + signal directly (bypass handle_message to avoid busy-ack).
        # May overwrite a user message pending in the same ~200ms window — acceptable.
        adapter._pending_messages[session_key] = MessageEvent(
            text=recall_text, message_type=MessageType.TEXT, source=cls._build_source(adapter, group_code, from_account), internal=True)
        active_event = adapter._active_sessions.get(session_key)
        if active_event is not None:
            active_event.set()
        logger.info("[%s] Recall interrupt: msg_id=%s session=%s", adapter.name, recalled_id, session_key[:30])
        # The interrupted turn persists the recalled content *after* our interrupt — redact later.
        recalled_text = adapter._processing_msg_texts.get(session_key, "")
        if recalled_text:
            cls._schedule_content_redact(adapter, session_key, recalled_text, group_code, from_account)

    @classmethod
    def _schedule_content_redact(cls, adapter, session_key: str, recalled_text: str, group_code: str, from_account: str) -> None:
        from gateway.platforms.yuanbao import (
            TranscriptReadError,
            asyncio,
        )

        async def _redact() -> None:
            store = _session_store(adapter)
            if not store:
                return
            try:
                sid = cls._resolve_sid(store, adapter, group_code, from_account)
            except Exception:
                return
            # Poll until the recalled content appears — the interrupted turn hasn't finished writing yet.
            for _ in range(30):
                await asyncio.sleep(0.5)
                try:
                    transcript = store.load_transcript(sid)
                except TranscriptReadError as exc:
                    # No readable rows means nothing to redact; polling on
                    # would just re-log the same failure (#100788).
                    logger.warning(
                        "[%s] Recall redact: transcript unreadable for "
                        "session %s: %s", adapter.name, sid, exc,
                    )
                    return
                except Exception:
                    continue
                for entry in transcript:
                    if entry.get("role") == "user" and entry.get("content") == recalled_text:
                        cls._redact(adapter, store, sid, transcript, entry, "[%s] Recall redact: session %s",
                                    "[%s] Recall redact failed: %s", session_key[:30])
                        return
            logger.debug("[%s] Recall redact: content not found after polling, session %s", adapter.name, session_key[:30])
        adapter._track_task(asyncio.create_task(_redact()))

    # -- Branch A/B: patch transcript (session idle) --------------------

    @classmethod
    def _patch_transcript(cls, adapter, recalled_id: str, group_code: str,
                          from_account: str, recalled_content: Optional[str] = None) -> None:
        from gateway.platforms.yuanbao import (
            TranscriptReadError,
            datetime,
            UTC,
        )

        store = _session_store(adapter)
        if not store:
            return
        try:
            sid = cls._resolve_sid(store, adapter, group_code, from_account)
        except Exception as exc:
            logger.warning("[%s] Recall: failed to resolve session: %s", adapter.name, exc)
            return
        try:
            # Load transcript from canonical store (state.db). Since PR #29278 added a
            # ``platform_message_id`` column to the messages table and ``append_to_transcript`` wires the
            # incoming dict's ``message_id`` into it, ``load_transcript`` returns rows with ``message_id``
            # set for any message that was observed with one — Branch A1 (exact id match) is the canonical
            # path again.
            transcript = store.load_transcript(sid)
        except TranscriptReadError as exc:
            # Not an empty transcript — the rows are unreadable, so recall has
            # nothing to match against (#100788).
            logger.warning("[%s] Recall: transcript unreadable: %s", adapter.name, exc)
            return
        except Exception as exc:
            logger.warning("[%s] Recall: failed to load transcript: %s", adapter.name, exc)
            return
        # A1: exact platform message_id match; A2: content-match fallback for rows without a
        # platform id (agent-processed @bot messages — run.py doesn't carry msg_id — or pre-column rows).
        target = next((e for e in transcript if e.get("message_id") == recalled_id), None)
        branch_label = "branch A1: id match"
        if target is None and recalled_content:
            target = next((e for e in transcript if e.get("role") == "user" and e.get("content") == recalled_content), None)
            branch_label = "branch A2: content match"
        if target is not None:
            cls._redact(adapter, store, sid, transcript, target, "[%s] Recall: redacted msg_id=%s (%s)",
                        "[%s] Recall: rewrite_transcript failed: %s", recalled_id, branch_label)
            return
        # Branch B: not found in transcript → append system note
        store.append_to_transcript(sid, {
            "role": "system", "timestamp": datetime.now(tz=UTC).isoformat(),
            "content": f'[recall] message_id="{recalled_id}" has been recalled; do not quote or reference it.',
        })
        logger.info("[%s] Recall: system note for msg_id=%s (branch B)", adapter.name, recalled_id)


class SkipSelfMiddleware(InboundMiddleware):
    """Drop the bot's own messages."""
    name = "skip-self"

    @staticmethod
    def _is_self_reference(from_account: str, bot_id: Optional[str]) -> bool:
        return bool(from_account) and from_account == bot_id

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if self._is_self_reference(ctx.from_account, ctx.adapter._bot_id):
            logger.debug("[%s] Ignoring self-sent message from %s", ctx.adapter.name, ctx.from_account)
            return  # Stop pipeline
        await next_fn()


class ChatRoutingMiddleware(InboundMiddleware):
    """Derive chat_id / chat_type / chat_name from push fields."""
    name = "chat-routing"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if ctx.group_code:
            ctx.chat_id, ctx.chat_type, ctx.chat_name = f"group:{ctx.group_code}", "group", ctx.group_name or ctx.group_code
        else:
            ctx.chat_id, ctx.chat_type, ctx.chat_name = f"direct:{ctx.from_account}", "dm", ctx.sender_nickname or ctx.from_account
        await next_fn()


class AccessPolicy(OwnAccessPolicyMixin):
    """DM / group access rules shared by inbound middleware and outbound ``send_dm``."""
    ALLOW_ALL_ENV_PREFIX = "YUANBAO"

    def __init__(self, dm_policy: str, dm_allow_from: list[str], group_policy: str, group_allow_from: list[str]) -> None:
        self._dm_policy = dm_policy
        self._allow_from = dm_allow_from
        self._group_policy = group_policy
        self._group_allow_from = group_allow_from

    def is_dm_allowed(self, sender_id: str) -> bool:
        """Strict DM authorization — pairing does not imply access."""
        return self._is_dm_allowed(sender_id.strip())

    def is_dm_intake_allowed(self, sender_id: str) -> bool:
        """Whether a DM may reach gateway intake (pairing handshake path)."""
        return self._is_dm_intake_allowed(sender_id)

    def is_group_allowed(self, group_code: str) -> bool:
        """Unlike the shared rule, an ``open`` group still needs the allow-all opt-in: Yuanbao groups
        have no runner-side mention gate, so open-without-opt-in would forward every member."""
        if self._group_policy == "open":
            return self._open_dm_opted_in()
        return self._is_group_allowed(group_code.strip())

    @property
    def dm_policy(self) -> str:
        return self._dm_policy

    @property
    def group_policy(self) -> str:
        return self._group_policy


class AccessGuardMiddleware(InboundMiddleware):
    """Platform-level DM/group access filter."""
    name = "access-guard"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        policy: AccessPolicy = ctx.adapter._access_policy
        if ctx.chat_type == "dm" and not policy.is_dm_intake_allowed(ctx.from_account):
            logger.debug("[%s] DM from %s blocked by dm_policy=%s", ctx.adapter.name, ctx.from_account, policy.dm_policy)
            return  # Stop pipeline
        if ctx.chat_type == "group" and not policy.is_group_allowed(ctx.group_code):
            logger.debug("[%s] Group %s blocked by group_policy=%s", ctx.adapter.name, ctx.group_code, policy.group_policy)
            return  # Stop pipeline
        await next_fn()


class AutoSetHomeMiddleware(InboundMiddleware):
    """Silently designate the first inbound conversation as home channel (config.yaml + env);
    a group home is upgraded by the first DM. Runs after GroupAtGuard so unaddressed group traffic
    never claims it; only strictly-authorized senders (allowlist / open opt-in / pairing-approved)
    may — intake-only pairing forwards must not."""
    name = "auto-sethome"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        from gateway.platforms.yuanbao import (
            _yb_secret,
        )

        adapter = ctx.adapter
        if not adapter._auto_sethome_done and adapter._sender_may_designate_home(ctx):
            _cur_home = _yb_secret("YUANBAO_HOME_CHANNEL", "") or ""
            _should_set = not _cur_home or (_cur_home.startswith("group:") and ctx.chat_type == "dm")
            if ctx.chat_type == "dm":
                adapter._auto_sethome_done = True  # DM seen — no further upgrades needed
            if _should_set:
                self._persist_home(adapter, ctx)
        await next_fn()

    @staticmethod
    def _persist_home(adapter, ctx: InboundContext) -> None:
        from gateway.platforms.yuanbao import (
            Platform,
            _profile_scoped,
            os,
        )

        try:
            from gateway.config import HomeChannel, persist_home_channel
            home = HomeChannel(platform=Platform.YUANBAO, chat_id=str(ctx.chat_id), name=str(ctx.chat_name or "Home"))
            # ``platforms.yuanbao.home_channel`` in the owning profile's config.yaml is the durable record
            # ``load_gateway_config`` reads back; the live PlatformConfig is updated so cron/home-channel
            # delivery in THIS process has a target without a restart.
            persist_home_channel(home)
            adapter.config.home_channel = home
            # Under a multiplexed secondary's scope the process env is the DEFAULT profile's; writing there
            # would make this tenant's chat the default profile's cron/notification home.
            if not _profile_scoped():
                os.environ["YUANBAO_HOME_CHANNEL"] = str(ctx.chat_id)
            logger.info("[%s] Auto-sethome: designated %s (%s) as Yuanbao home channel", adapter.name, ctx.chat_id, ctx.chat_name)
        except Exception as e:
            logger.warning("[%s] Auto-sethome failed: %s", adapter.name, e)


def _iter_custom_elems(msg_body: list) -> Iterator[tuple[Any, dict]]:
    """Yield ``(custom, content)`` for each TIMCustomElem whose ``data`` parses as JSON (any type)."""
    from gateway.platforms.yuanbao import (
        contextlib,
        json,
    )

    for elem in msg_body or []:
        if not isinstance(elem, dict) or elem.get("msg_type") != "TIMCustomElem":
            continue
        content = elem.get("msg_content", {}) or {}
        data_str = content.get("data", "") if isinstance(content, dict) else ""
        if data_str:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                yield json.loads(data_str), content


def _file_name(content: dict) -> str:
    """First non-empty of file_name / fileName / filename, stripped."""
    return (str(content.get("file_name") or "").strip() or str(content.get("fileName") or "").strip()
            or str(content.get("filename") or "").strip())


def _media_ref(kind: str, url: str, name: str = "") -> dict[str, str]:
    """media_refs entry; ``name`` is only emitted when non-empty."""
    from gateway.platforms.yuanbao import (
        Dict,
    )

    ref: dict[str, str] = {"kind": kind, "url": url}
    if name:
        ref["name"] = name
    return ref


class ExtractContentMiddleware(InboundMiddleware):
    """Extract raw text, media refs and forwarded records from msg_body."""
    name = "extract-content"
    _CARD_CONTENT_MAX_LENGTH = 1000
    _UNSUPPORTED = "[unsupported message type]"

    @staticmethod
    def _format_shared_link(custom: dict) -> str:
        """elem_type 1010 (share card) → bracket-placeholder text."""
        title, link = custom.get("title", ""), custom.get("link", "")
        lines = [f"[share_card: {title} | {link}]" if link else f"[share_card: {title}]"]
        max_len = ExtractContentMiddleware._CARD_CONTENT_MAX_LENGTH
        preview = next((v for v in (custom.get("card_content"), custom.get("wechat_des")) if v and isinstance(v, str)), None)
        if preview:
            lines.append(f"Preview: {preview[:max_len] + '...(truncated)' if len(preview) > max_len else preview}")
        if link:
            lines.append("[visit link for full content]")
        return "\n".join(lines)

    @staticmethod
    def _format_link_understanding(custom: dict) -> Optional[str]:
        """elem_type 1007 (link understanding card) → bracket-placeholder text."""
        from gateway.platforms.yuanbao import (
            json,
        )

        content = custom.get("content")
        if not content:
            return None
        try:
            parsed = json.loads(content)
            link = parsed.get("link") if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            link = None
        return f"[link: {link} | visit link for full content]" if link and isinstance(link, str) else None

    @staticmethod
    def _parse_resource_id(url: str) -> str:
        """resourceId from a Yuanbao resource URL's query string, or ""."""
        from gateway.platforms.yuanbao import (
            urllib,
        )

        try:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query) if url else {}
            ids = query.get("resourceId") or query.get("resourceid") or []
            return str(ids[0]).strip() if ids else ""
        except Exception:
            return ""

    @staticmethod
    def _pick_image_url(content: dict) -> str:
        """URL of the medium image (index 1), falling back to index 0, else ""."""
        arr = content.get("image_info_array")
        arr = arr if isinstance(arr, list) else []
        image_info = arr[1] if len(arr) > 1 and isinstance(arr[1], dict) else arr[0] if arr and isinstance(arr[0], dict) else None
        return str((image_info or {}).get("url") or "").strip()

    @classmethod
    def _extract_text(cls, msg_body: list) -> str:
        """Plain text from MsgBody: text elems verbatim; media as ``[kind|ybres:RID]`` / ``[kind]``
        (file: ``[file:{name}|ybres:RID]``); TIMFaceElem ``[emoji: name]``; custom elems by
        elem_type. Parts are space-joined."""
        from gateway.platforms.yuanbao import (
            _TEXT_ELEM_TYPE,
            json,
        )

        parts: list[str] = []
        for elem in msg_body:
            elem_type: str = elem.get("msg_type", "")
            content: dict = elem.get("msg_content", {})
            if elem_type == _TEXT_ELEM_TYPE:
                if content.get("text", ""):
                    parts.append(content["text"])
            elif elem_type in ("TIMImageElem", "TIMSoundElem", "TIMVideoFileElem"):
                kind = {"TIMImageElem": "image", "TIMSoundElem": "voice", "TIMVideoFileElem": "video"}[elem_type]
                url = cls._pick_image_url(content) if kind == "image" else str(content.get("url") or "").strip()
                rid = cls._parse_resource_id(url)
                parts.append(f"[{kind}|ybres:{rid}]" if rid else f"[{kind}]")
            elif elem_type == "TIMFileElem":
                filename = content.get("file_name", content.get("fileName", content.get("filename", "")))
                rid = cls._parse_resource_id(str(content.get("url") or "").strip())
                if rid:
                    parts.append(f"[file:{filename}|ybres:{rid}]" if filename else f"[file|ybres:{rid}]")
                else:
                    parts.append(f"[file: {filename}]" if filename else "[file]")
            elif elem_type == "TIMCustomElem":
                parts.append(cls._custom_elem_text(content.get("data", "")))
            elif elem_type == "TIMFaceElem":
                face_name = ""
                if content.get("data", ""):
                    try:
                        face_name = (json.loads(content["data"]).get("name") or "").strip()
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                parts.append(f"[emoji: {face_name}]" if face_name else "[emoji]")
            elif elem_type:
                parts.append(f"[{elem_type}]")  # unknown element type — keep as placeholder
        return " ".join(parts)

    @classmethod
    def _custom_elem_text(cls, data_val: str) -> str:
        """Text for a TIMCustomElem by elem_type: 1002 mention, 1010 share card, 1007 link card,
        1009 forwarded chat-record summary; malformed JSON is passed through verbatim."""
        from gateway.platforms.yuanbao import (
            json,
        )

        if not data_val:
            return cls._UNSUPPORTED
        try:
            custom = json.loads(data_val)
        except (json.JSONDecodeError, TypeError):
            return data_val
        if not isinstance(custom, dict):
            return cls._UNSUPPORTED
        ctype = custom.get("elem_type")
        if ctype == 1002:
            return custom.get("text", "[mention]")
        if ctype == 1009:
            return custom.get("text", "[chat record]")
        if ctype == 1010:
            return cls._format_shared_link(custom)
        return (cls._format_link_understanding(custom) if ctype == 1007 else None) or cls._UNSUPPORTED

    @staticmethod
    def _rewrite_slash_command(text: str) -> str:
        """Strip; convert a leading full-width slash (Chinese IME) to ASCII so commands match."""
        text = text.strip()
        return '/' + text[1:] if text.startswith('\uff0f') else text

    @staticmethod
    def _extract_inbound_media_refs(msg_body: list) -> list[dict[str, str]]:
        """Inbound image/file refs: ``[{"kind": "image", "url": ...}, {"kind": "file", "url": ..., "name": ...}]``."""
        from gateway.platforms.yuanbao import (
            Dict,
            List,
        )

        refs: list[dict[str, str]] = []
        for elem in msg_body or []:
            if not isinstance(elem, dict):
                continue
            msg_type = elem.get("msg_type", "")
            content = elem.get("msg_content", {}) or {}
            if not isinstance(content, dict):
                continue
            if msg_type == "TIMImageElem":
                image_url = ExtractContentMiddleware._pick_image_url(content)
                if image_url:
                    refs.append(_media_ref("image", image_url))
            elif msg_type == "TIMFileElem":
                file_url = str(content.get("url") or "").strip()
                if file_url:
                    refs.append(_media_ref("file", file_url, _file_name(content)))
        return refs

    @staticmethod
    def _extract_forwarded_records(msg_body: list, user_id: str = "") -> Optional[dict]:
        """ForwardMsgData for elem_type 1009 (WeChat forward), or None. Payload lives in
        ``msg_content.ext_map`` (pb field 999) under ``wexin_forward_msg_[id]_[userid]`` keys as
        base64 protobuf (NOT JSON); the first entry decoding to ``sub_type == 1`` wins."""
        from gateway.platforms.yuanbao import (
            base64,
            binascii,
            contextlib,
            decode_forward_msg_data,
        )

        for custom, content in _iter_custom_elems(msg_body):
            if not (isinstance(custom, dict) and custom.get("elem_type") == 1009):
                continue
            ext_map = content.get("ext_map") or {}
            if not isinstance(ext_map, dict) or not ext_map:
                return None
            for key, value in ext_map.items():
                if not key.startswith("wexin_forward_msg_") or not isinstance(value, str) or not value:
                    continue
                with contextlib.suppress(binascii.Error, ValueError):
                    data = decode_forward_msg_data(base64.b64decode(value))
                    if isinstance(data, dict) and data.get("sub_type") == 1:
                        return data
        return None

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.raw_text = self._rewrite_slash_command(self._extract_text(ctx.msg_body))
        ctx.media_refs = self._extract_inbound_media_refs(ctx.msg_body)
        ctx.forwarded_records = self._extract_forwarded_records(ctx.msg_body, ctx.from_account)
        await next_fn()


class PlaceholderFilterMiddleware(InboundMiddleware):
    """Skip pure placeholder messages (e.g. '[image]' with no media)."""
    name = "placeholder-filter"
    SKIPPABLE_PLACEHOLDERS: frozenset = frozenset({"[image]", "[图片]", "[file]", "[文件]", "[video]", "[视频]", "[voice]", "[语音]"})

    @classmethod
    def is_skippable_placeholder(cls, text: str, media_count: int = 0) -> bool:
        return media_count <= 0 and text.strip() in cls.SKIPPABLE_PLACEHOLDERS

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        if self.is_skippable_placeholder(ctx.raw_text, len(ctx.media_refs)):
            logger.debug("[%s] Skipping placeholder message: %r", ctx.adapter.name, ctx.raw_text)
            return  # Stop pipeline
        await next_fn()


class OwnerCommandMiddleware(InboundMiddleware):
    """Bot-owner slash commands in groups: allowlisted commands skip @Bot; non-owner attempts are rejected."""
    name = "owner-command"
    ALLOWLIST: frozenset = frozenset({"/new", "/reset", "/retry", "/undo", "/stop", "/approve", "/deny", "/bg", "/btw", "/queue", "/q"})
    _rewrite_slash_command = staticmethod(ExtractContentMiddleware._rewrite_slash_command)

    @classmethod
    def _detect_owner_command(cls, *, push: dict, msg_body: list, chat_type: str, from_account: str) -> tuple[Optional[str], Optional[str], bool]:
        """→ (cmd, cmd_line, is_owner); (None, None, False) when not an allowlisted command."""
        from gateway.platforms.yuanbao import (
            _TEXT_ELEM_TYPE,
        )

        if chat_type != "group" or not cls.ALLOWLIST:
            return None, None, False
        # Only recognise commands when there is exactly one text segment.
        text_elems = [e for e in (msg_body or []) if e.get("msg_type") == _TEXT_ELEM_TYPE]
        if len(text_elems) != 1:
            return None, None, False
        cmd_line = cls._rewrite_slash_command((text_elems[0].get("msg_content") or {}).get("text", ""))
        if not cmd_line.startswith("/"):
            return None, None, False
        cmd = cmd_line.split(maxsplit=1)[0].lower()
        if cmd not in cls.ALLOWLIST:
            return None, None, False
        # Owner ⇔ push.from_account == push.bot_owner_id; these commands are privileged
        # (/approve, /stop, /reset…) so a non-owner must never run them.
        owner_id = str((push or {}).get("bot_owner_id") or "").strip()
        return cmd, cmd_line, bool(owner_id) and owner_id == from_account

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        from gateway.platforms.yuanbao import (
            asyncio,
            t,
        )

        adapter = ctx.adapter
        matched_cmd, cmd_line, is_owner = self._detect_owner_command(
            push=ctx.push, msg_body=ctx.msg_body, chat_type=ctx.chat_type, from_account=ctx.from_account,
        )
        if matched_cmd and not is_owner:
            logger.info("[%s] Reject non-owner slash command: chat=%s from=%s cmd=%s", adapter.name, ctx.chat_id, ctx.from_account, matched_cmd)
            adapter._track_task(asyncio.create_task(
                adapter.send(ctx.chat_id, t("platform.yuanbao.owner_command_denied", command=matched_cmd)),
                name=f"yuanbao-owner-cmd-denial-{matched_cmd}"))
            return  # Stop pipeline
        if matched_cmd and is_owner and cmd_line:
            logger.info("[%s] Bot owner slash command: chat=%s from=%s cmd=%s", adapter.name, ctx.chat_id, ctx.from_account, matched_cmd)
            ctx.owner_command = matched_cmd
            ctx.raw_text = cmd_line  # clean command text
        await next_fn()


class BuildSourceMiddleware(InboundMiddleware):
    name = "build-source"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.source = ctx.adapter.build_source(
            chat_id=ctx.chat_id, chat_type=ctx.chat_type, chat_name=ctx.chat_name,
            user_id=ctx.from_account or None, user_name=ctx.sender_nickname or ctx.from_account,
            thread_id="main" if ctx.chat_type == "group" else None,
        )
        await next_fn()


class GroupAtGuardMiddleware(InboundMiddleware):
    """Group chat: observe non-@bot messages into the transcript; only @Bot (or owner commands) proceed."""
    name = "group-at-guard"

    @staticmethod
    def _iter_bot_mentions(msg_body: list, bot_id: Optional[str]) -> Iterator[dict]:
        """Yield @bot elems: TIMCustomElem whose data JSON has elem_type 1002 and user_id == bot_id."""
        if not bot_id:
            return
        for custom, _content in _iter_custom_elems(msg_body):
            if custom.get("elem_type") == 1002 and custom.get("user_id") == bot_id:
                yield custom

    @classmethod
    def _is_at_bot(cls, msg_body: list, bot_id: Optional[str]) -> bool:
        return any(True for _ in cls._iter_bot_mentions(msg_body, bot_id))

    @classmethod
    def _extract_bot_mention_text(cls, msg_body: list, bot_id: Optional[str]) -> str:
        """Display text used to @-mention this bot (e.g. ``@yuanbao-bot``), or ""."""
        return next((t for t in (str(c.get("text") or "").strip() for c in cls._iter_bot_mentions(msg_body, bot_id)) if t), "")

    @staticmethod
    def _build_group_channel_prompt(msg_body: list, bot_id: Optional[str]) -> str:
        """Per-turn group-chat prompt that highlights which message to respond to."""
        bot_mention = GroupAtGuardMiddleware._extract_bot_mention_text(msg_body, bot_id) or "unknown"
        return (
            "You are handling a Yuanbao group chat message.\n"
            f"- Your identity: user_id={bot_id or 'unknown'}, @-mention name in this group={bot_mention}\n"
            "- Lines in history prefixed with `[nickname|user_id]` are observed group context "
            "and are not necessarily addressed to you.\n"
            "- Treat only the current new message as a request explicitly directed at you, "
            "and answer it directly."
        )

    @classmethod
    def _observe_group_message(cls, adapter, source, sender_display: str, text: str, *, ctx: InboundContext,
                               msg_id: Optional[str] = None, forwarded_records: Optional[dict] = None) -> None:
        """Record a group message as ``role: user`` ``[nickname|user_id]\\n<content>`` without
        invoking the agent, so later @bot turns see the full conversation."""
        from gateway.platforms.yuanbao import (
            datetime,
            UTC,
        )

        store = _session_store(adapter)
        if not store:
            return
        try:
            session_entry = store.get_or_create_session(source)
            body_text = text
            if forwarded_records:
                summary = ForwardedRecordsParseMiddleware.build_forward_text(forwarded_records, ctx=ctx, is_dispatch=False)
                if summary:
                    body_text = f"{text}\n{summary}" if text else summary
            entry: dict = {
                "role": "user", "content": f"[{sender_display}|{source.user_id or 'unknown'}]\n{body_text}",
                "timestamp": datetime.now(tz=UTC).isoformat(), "observed": True,
            }
            if msg_id:
                entry["message_id"] = msg_id
            store.append_to_transcript(session_entry.session_id, entry)
        except Exception as exc:
            logger.warning("[%s] Failed to observe group message: %s", adapter.name, exc)

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        adapter = ctx.adapter
        if ctx.chat_type == "group" and not ctx.owner_command and not self._is_at_bot(ctx.msg_body, adapter._bot_id):
            self._observe_group_message(
                adapter, ctx.source, ctx.sender_nickname or ctx.from_account, ctx.raw_text,
                msg_id=ctx.msg_id or None, forwarded_records=ctx.forwarded_records, ctx=ctx,
            )
            logger.info("[%s] Group message observed (no @bot): chat=%s from=%s", adapter.name, ctx.chat_id, ctx.from_account)
            return  # Stop pipeline — message observed but not dispatched
        await next_fn()


class GroupAttributionMiddleware(InboundMiddleware):
    """Group @bot turns: build channel_prompt, rewrite raw_text to ``[nickname|user_id]\\n<content>``
    (matches observed-history format) and clear ``source.user_name`` to suppress the runner's
    ``[user_name]`` prefix."""
    name = "group-attribution"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        from gateway.platforms.yuanbao import (
            dataclasses,
        )

        if ctx.chat_type == "group" and not ctx.owner_command:
            ctx.channel_prompt = GroupAtGuardMiddleware._build_group_channel_prompt(ctx.msg_body, ctx.adapter._bot_id)
            ctx.raw_text = f"[{ctx.sender_nickname or ctx.from_account or 'unknown'}|{ctx.from_account or 'unknown'}]\n{ctx.raw_text}"
            if ctx.source is not None:
                ctx.source = dataclasses.replace(ctx.source, user_name=None)
        await next_fn()


class YuanbaoMessageType(Enum):
    CHAT_RECORD = "chat_record"  # yuanbao-local subtype; coerced back to MessageType in DispatchMiddleware


_ELEM_MESSAGE_TYPES = {"TIMImageElem": MessageType.PHOTO, "TIMSoundElem": MessageType.VOICE,
                       "TIMVideoFileElem": MessageType.VIDEO, "TIMFileElem": MessageType.DOCUMENT}


class ClassifyMessageTypeMiddleware(InboundMiddleware):
    """MessageType (or yuanbao-local YuanbaoMessageType) from text and msg_body elements."""
    name = "classify-msg-type"

    @staticmethod
    def _classify(text: str, msg_body: list):
        from gateway.platforms.yuanbao import (
            MessageType,
            json,
        )

        if text.startswith("/"):
            return MessageType.COMMAND
        for elem in msg_body:
            etype = elem.get("msg_type", "")
            mapped = _ELEM_MESSAGE_TYPES.get(etype)
            if mapped is not None:
                return mapped
            if etype == "TIMCustomElem":
                try:
                    custom = json.loads((elem.get("msg_content") or {}).get("data", ""))
                except (json.JSONDecodeError, TypeError):
                    custom = None
                if isinstance(custom, dict) and custom.get("elem_type") == 1009:
                    return YuanbaoMessageType.CHAT_RECORD
        return MessageType.TEXT

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.msg_type = self._classify(ctx.raw_text, ctx.msg_body)
        await next_fn()


class QuoteContextMiddleware(InboundMiddleware):
    """Extract quote/reply context from cloud_custom_data."""
    name = "quote-context"

    def _extract_quote_context(self, cloud_custom_data: str) -> tuple[Optional[str], Optional[str]]:
        """(quote_id, quote_text) from cloud_custom_data → MessageEvent.reply_to_*."""
        from gateway.platforms.yuanbao import (
            json,
        )

        try:
            parsed = json.loads(cloud_custom_data) if cloud_custom_data else None
        except (json.JSONDecodeError, TypeError):
            parsed = None
        quote = parsed.get("quote") if isinstance(parsed, dict) else None
        if not isinstance(quote, dict):
            return None, None
        quote_id = str(quote.get("id") or "").strip() or None
        desc = str(quote.get("desc") or "").strip()
        sender = str(quote.get("sender_nickname") or quote.get("sender_id") or "").strip()
        return quote_id, (f"{sender}: {desc}" if sender else desc) if desc else None

    async def _extract_media_refs_from_transcript(self, ctx: InboundContext) -> list[tuple[str, str, str]]:
        """``(rid, kind, filename)`` for ybres anchors in the quoted transcript message; [] when
        there is no reply_to id, no store/source, or no resolvable anchors."""
        from gateway.platforms.yuanbao import (
            List,
            TranscriptReadError,
            Tuple,
            _YB_RES_REF_RE,
            _iter_ybres_refs,
        )

        if ctx.reply_to_message_id is None:
            return []
        adapter = ctx.adapter
        media_refs: list[tuple[str, str, str]] = []
        try:
            store = _session_store(adapter)
            if not store or ctx.source is None:
                return []
            history = store.load_transcript(store.get_or_create_session(ctx.source).session_id)
            for msg in reversed(history or []):
                mid = msg.get("message_id", "")
                if not mid or mid != ctx.reply_to_message_id:
                    continue
                _content = msg.get("content", "")
                if isinstance(_content, str) and "|ybres:" in _content:
                    media_refs.extend(_iter_ybres_refs(_YB_RES_REF_RE.finditer(_content)))
                break
        except TranscriptReadError as exc:
            # Quote resolution degrades to "no refs" rather than pretending
            # the quoted message was never seen (#100788).
            logger.warning(
                "[%s] quote transcript lookup: transcript unreadable: %s",
                getattr(adapter, "name", "yuanbao"), exc,
            )
        except Exception as exc:
            logger.warning("[%s] quote transcript lookup failed: %s", getattr(adapter, "name", "yuanbao"), exc)
        return media_refs

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.reply_to_message_id, ctx.reply_to_text = self._extract_quote_context(ctx.cloud_custom_data)
        ctx.quote_media_refs = await self._extract_media_refs_from_transcript(ctx)
        await next_fn()


class ForwardedRecordsParseMiddleware(InboundMiddleware):
    """Deep-parse WeChat forwarded chat records (elem_type 1009) on ``ctx.forwarded_records``:
    render media as ``[kind|ybres:RID]``, append refs to ``ctx.media_refs`` and rewrite raw_text.
    No run-time fallback for earlier forwards — GroupAtGuard already rendered summaries at observe
    time. On any failure raw_text is left untouched."""
    name = "forwarded-records-parse"
    FORWARD_MSG_TEXT_MAX_CHARS = 1000  # per-record text cap; record count is NOT capped

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        try:
            if ctx.forwarded_records:
                await self._send_loading_heartbeat(ctx)
                ctx.raw_text = self.build_forward_text(ctx.forwarded_records, ctx=ctx, is_dispatch=True)
        except Exception as exc:
            logger.warning("[%s] forwarded-records deep parse failed: %s", getattr(ctx.adapter, "name", "yuanbao"), exc)
        await next_fn()

    @staticmethod
    async def _send_loading_heartbeat(ctx: InboundContext) -> None:
        """Best-effort RUNNING heartbeat so the user sees a loading bubble."""
        from gateway.platforms.yuanbao import (
            WS_HEARTBEAT_RUNNING,
            contextlib,
        )

        with contextlib.suppress(Exception):
            await ctx.adapter._outbound.heartbeat.send_heartbeat_once(ctx.chat_id, WS_HEARTBEAT_RUNNING)

    @classmethod
    def _media_marker(cls, media: dict, plain_text: str = "") -> tuple[str, Optional[dict[str, str]]]:
        """One ``multimedia`` entry → ``(marker, ref)``: ``[kind|ybres:RID]`` + media_refs dict when a
        RID/URL is usable, else a plain ``[kind] name`` marker and ``ref=None``."""
        media_type = (media.get("type", "") or media.get("doc_type", "")).strip().lower()
        url = str(media.get("url") or "").strip()
        file_name = str(media.get("file_name") or "").strip()
        # media_id is directly usable as a ybres RID; else parse resourceId from the URL.
        rid = str(media.get("media_id") or "").strip() or ExtractContentMiddleware._parse_resource_id(url)
        kind = {"image": "image", "file": "file", "document": "file", "code": "file", "video": "video"}.get(media_type)
        if kind and url and rid:
            return f"[{kind}|ybres:{rid}] {file_name}".rstrip(), _media_ref(kind, url, file_name if kind == "file" else "")
        if kind == "image":
            return f"[image] {file_name or plain_text}".rstrip(), None
        if kind == "file":
            return f"[file] {file_name}".rstrip(), None
        if kind == "video":
            return f"[video] {file_name or url}".rstrip(), None
        if media_type == "url":  # link share (e.g. WeChat article) — keep URL for the agent
            return f"[link] {file_name or str(media.get('title') or '')} {url}".rstrip(), None
        return f"[{media_type or 'media'}] {url or file_name}".rstrip(), None

    @classmethod
    def _walk_forward_msgs(cls, forward_data: dict) -> Iterator[tuple[str, str, list[dict[str, str]]]]:
        """Yield ``(sender, body, refs)`` per ``ForwardMsgData['msg']`` record; body capped at
        FORWARD_MSG_TEXT_MAX_CHARS. ``refs`` keeps textual order — PatchAnchorsMiddleware relies on it."""
        from gateway.platforms.yuanbao import (
            Dict,
            List,
        )

        for msg in (forward_data.get("msg") if isinstance(forward_data, dict) else None) or []:
            if not isinstance(msg, dict):
                continue
            plain_text = msg.get("plainText", "")
            refs: list[dict[str, str]] = []
            parts: list[str] = []
            for mc in msg.get("msgContent", []) or []:
                if not isinstance(mc, dict):
                    continue
                mc_type = mc.get("type", 0)  # EnumMsgContentType: 1 TEXT, 2 MULTIMEDIA, 3 nested FORWARD
                if mc_type == 1:
                    parts.append(mc.get("text", ""))
                elif mc_type == 2:
                    for media in mc.get("multimedia", []) or []:
                        if isinstance(media, dict):
                            marker, ref = cls._media_marker(media, plain_text)
                            parts.append(marker)
                            if ref is not None:
                                refs.append(ref)
                elif mc_type == 3:
                    parts.append("[嵌套聊天记录]")
                elif plain_text:
                    parts.append(plain_text)
            rendered = "  ".join(p for p in parts if p) or plain_text
            if len(rendered) > cls.FORWARD_MSG_TEXT_MAX_CHARS:
                rendered = rendered[: cls.FORWARD_MSG_TEXT_MAX_CHARS] + "…(已截断)"
            yield msg.get("sender", ""), rendered, refs

    @classmethod
    def build_forward_text(cls, forward_data: dict, *, ctx: InboundContext, is_dispatch: bool) -> str:
        """Render ``ForwardMsgData`` as ``发送人：正文`` lines with media markers. When ``is_dispatch``,
        refs go to ``ctx.media_refs`` and a ``用户附言：`` footer is added (observe-time callers skip both)."""
        lines = [f"当前用户的昵称为{ctx.sender_nickname or '用户'}", "以下为用户的聊天记录"]
        for sender, body, refs in cls._walk_forward_msgs(forward_data):
            lines.append(f"{sender}：{body}")
            if is_dispatch:
                ctx.media_refs.extend(refs)
        text = "\n".join(lines)
        if is_dispatch and ctx.raw_text.strip():
            text += f"\n\n用户附言：{ctx.raw_text.strip()}"
        return text


class MediaResolveMiddleware(InboundMiddleware):
    """Resolve inbound media references to local cached files. Yuanbao COS hostnames resolve to
    private IPs (tripping vision_tools' SSRF guard), so we download ourselves and hand the model
    local paths."""
    name = "media-resolve"
    # Resource download cache keyed by resourceId: rid -> (local_path, mime, ts)
    _resource_cache: ClassVar[dict[str, tuple[str, str, float]]] = {}
    _RESOURCE_CACHE_TTL_S: ClassVar[int] = 24 * 60 * 60
    _RESOURCE_CACHE_MAX_SIZE: ClassVar[int] = 256

    @classmethod
    def _get_cached_resource(cls, resource_id: str) -> Optional[tuple[str, str]]:
        """Cached ``(local_path, mime)`` if unexpired and the file still exists (cache dir may be swept)."""
        from gateway.platforms.yuanbao import (
            os,
            time,
        )

        entry = cls._resource_cache.get(resource_id) if resource_id else None
        if entry is None:
            return None
        local_path, mime, ts = entry
        if time.time() - ts > cls._RESOURCE_CACHE_TTL_S or not os.path.isfile(local_path):
            cls._resource_cache.pop(resource_id, None)
            return None
        return local_path, mime

    @classmethod
    def _put_cached_resource(cls, resource_id: str, local_path: str, mime: str) -> None:
        """Cache a download result; evicts the oldest 25% when at capacity."""
        from gateway.platforms.yuanbao import (
            time,
        )

        if not resource_id:
            return
        if len(cls._resource_cache) >= cls._RESOURCE_CACHE_MAX_SIZE:
            sorted_keys = sorted(cls._resource_cache, key=lambda k: cls._resource_cache[k][2])
            for k in sorted_keys[: cls._RESOURCE_CACHE_MAX_SIZE // 4]:
                cls._resource_cache.pop(k, None)
        cls._resource_cache[resource_id] = (local_path, mime, time.time())

    @classmethod
    def _append_cached_resource(cls, adapter, resource_id: str, media_paths: list[str], mimes: list[str]) -> bool:
        """Append a cached resource to the output lists; False on miss."""
        hit = cls._get_cached_resource(resource_id)
        if hit is None:
            return False
        logger.debug("[%s] resource cache hit: rid=%s path=%s", adapter.name, resource_id, hit[0])
        media_paths.append(hit[0])
        mimes.append(hit[1])
        return True

    @staticmethod
    def _guess_image_ext_from_url(url: str) -> str:
        from gateway.platforms.yuanbao import (
            os,
            urllib,
        )

        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        return ext if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tiff"} else ".jpg"

    @staticmethod
    async def _fetch_resource_url(adapter, resource_id: str) -> str:
        """Exchange a ``resourceId`` for a direct download URL via ``/api/resource/v1/download``,
        with a single 401-retry after token force-refresh. Raises on failure."""
        from gateway.platforms.yuanbao import (
            Optional,
            SignManager,
            httpx,
        )

        resource_id = resource_id.strip()
        if not resource_id:
            raise RuntimeError("missing resource_id")
        def _auth_headers(token_data: dict, fallback_source: str) -> Optional[dict]:
            token = str(token_data.get("token") or "").strip()
            bot_id = str(token_data.get("bot_id") or adapter._bot_id or adapter._app_key).strip()
            if not token or not bot_id:
                return None
            source = str(token_data.get("source") or fallback_source).strip() or "web"
            return {"Content-Type": "application/json", "X-ID": bot_id, "X-Token": token, "X-Source": source}
        headers = _auth_headers(await adapter._get_cached_token(), "web")
        if headers is None:
            raise RuntimeError("missing token or bot_id for resource download")
        api_url = f"{adapter._api_domain}/api/resource/v1/download"
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for attempt in range(2):
                resp = await client.get(api_url, params={"resourceId": resource_id}, headers=headers)
                if resp.status_code == 401 and attempt == 0:
                    token_data = await SignManager.force_refresh(adapter._app_key, adapter._app_secret, adapter._api_domain)
                    headers = _auth_headers(token_data, headers["X-Source"] or "web")
                    if headers is None:
                        break
                    continue
                resp.raise_for_status()
                payload = resp.json()
                code = payload.get("code")
                if code not in {None, 0}:
                    raise RuntimeError(f"resource/v1/download failed: code={code}, msg={payload.get('msg', '')}")
                data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
                real_url = str((data or {}).get("url") or (data or {}).get("realUrl") or "").strip()
                if real_url:
                    return real_url
                raise RuntimeError("resource/v1/download missing url/realUrl")
        raise RuntimeError("resource/v1/download did not return a URL")

    @staticmethod
    async def _resolve_download_url(adapter, url: str) -> str:
        """Resolve a Yuanbao resource placeholder URL (``…/api/resource/download?resourceId=…``,
        which 401s on direct GET) to a fetchable URL via the business API; passthrough otherwise."""
        resource_id = ExtractContentMiddleware._parse_resource_id(url)
        if not resource_id:
            return url
        try:
            return await MediaResolveMiddleware._fetch_resource_url(adapter, resource_id)
        except Exception:
            return url

    @classmethod
    async def _download_and_cache(
        cls, adapter, *, fetch_url: str, kind: str,
        file_name: Optional[str] = None, log_tag: str = "", resource_id: str = "",
    ) -> Optional[tuple[str, str]]:
        """Download a Yuanbao resource into the local media cache → ``(local_path, mime)`` or None.
        A *resource_id* is checked against the in-memory cache first."""
        from gateway.platforms.yuanbao import (
            cache_document_from_bytes_async,
            cache_image_from_bytes_async,
            cache_video_from_bytes_async,
            guess_mime_type,
            media_download_url,
            os,
            urllib,
        )

        if resource_id:
            hit = cls._get_cached_resource(resource_id)
            if hit is not None:
                logger.debug("[%s] resource cache hit: rid=%s path=%s", adapter.name, resource_id, hit[0])
                return hit
        try:
            file_bytes, content_type = await media_download_url(fetch_url, max_size_mb=adapter.MEDIA_MAX_SIZE_MB)
        except Exception as exc:
            logger.warning("[%s] inbound media download failed: kind=%s %s err=%s", adapter.name, kind, log_tag, exc)
            return None
        if kind == "image":
            ext = cls._guess_image_ext_from_url(fetch_url)
            try:
                local_path = await cache_image_from_bytes_async(file_bytes, ext=ext)
            except ValueError as exc:
                logger.warning("[%s] inbound image cache rejected: %s err=%s", adapter.name, log_tag, exc)
                return None
            mime = guess_mime_type(f"image{ext}")
            if not mime.startswith("image/"):
                mime = content_type if content_type.startswith("image/") else "image/jpeg"
        elif kind == "video":
            # Yuanbao video resources carry no reliable extension; default to mp4.
            local_path = await cache_video_from_bytes_async(file_bytes)
            mime = guess_mime_type(local_path) or (content_type if content_type.startswith("video/") else "video/mp4")
        else:  # file
            file_name = file_name or os.path.basename(urllib.parse.urlparse(fetch_url).path) or "file"
            try:
                local_path = await cache_document_from_bytes_async(file_bytes, file_name)
            except Exception as exc:
                logger.warning("[%s] inbound file cache failed: %s err=%s", adapter.name, log_tag, exc)
                return None
            mime = guess_mime_type(file_name) or content_type or "application/octet-stream"
        cls._put_cached_resource(resource_id, local_path, mime)
        return local_path, mime

    @classmethod
    async def _resolve_media_urls(cls, adapter, media_refs: list[dict[str, str]]) -> tuple[list[str], list[str]]:
        """Resolve inbound media refs → (local_paths, mime_types); same bounded-concurrency,
        order-preserving, exception-isolated contract as :meth:`_resolve_ybres_refs`."""
        from gateway.platforms.yuanbao import (
            List,
            Tuple,
            _RESOLVABLE_MEDIA_KINDS,
        )

        media_urls: list[str] = []
        media_types: list[str] = []
        active: list[tuple[str, str, str, str]] = []  # (kind, filename, rid, url)
        for ref in media_refs:
            kind = str(ref.get("kind") or "").strip().lower()
            url = str(ref.get("url") or "").strip()
            if kind not in _RESOLVABLE_MEDIA_KINDS or not url:
                continue
            rid = ExtractContentMiddleware._parse_resource_id(url)
            if rid and cls._append_cached_resource(adapter, rid, media_urls, media_types):
                continue
            active.append((kind, str(ref.get("name") or "").strip(), rid or "", url))
        if active:
            await cls._gather_resolve(
                adapter, active, "media", media_urls, media_types,
                get_url=lambda url: cls._resolve_download_url(adapter, url),
                fail_fmt="[%s] inbound media resolve failed: kind=%s url=%s err=%s", fail_args=lambda kind, url: (kind, url),
                crash_fmt="[%s] inbound media resolve crashed: kind=%s url=%s err=%s", crash_args=lambda kind, url: (kind, url[:80]),
                log_tag=lambda url: f"placeholder_url={url[:80]}",
            )
        return media_urls, media_types

    @classmethod
    async def _gather_resolve(cls, adapter, active, scope, out_paths, out_mimes, *,
                              get_url, fail_fmt, fail_args, crash_fmt, crash_args, log_tag) -> None:
        """Resolve ``(kind, filename, rid, key)`` items under bounded concurrency — ``await get_url(key)``
        then download+cache — appending successes in input order. ``return_exceptions=True`` isolates
        per-item failures; the batch summary line keeps stable fields (concurrency vs elapsed_ms) for
        offline aggregation."""
        from gateway.platforms.yuanbao import (
            Optional,
            Tuple,
            asyncio,
            time,
        )

        semaphore = asyncio.Semaphore(adapter.media_resolve_concurrency)

        async def _one(kind: str, filename: str, rid: str, key: str) -> Optional[tuple[str, str]]:
            async with semaphore:
                try:
                    fetch_url = await get_url(key)
                except Exception as exc:
                    logger.warning(fail_fmt, adapter.name, *fail_args(kind, key), exc)
                    return None
                return await cls._download_and_cache(
                    adapter, fetch_url=fetch_url, kind=kind, file_name=filename or None, log_tag=log_tag(key), resource_id=rid,
                )
        _t0 = time.monotonic()
        results = await asyncio.gather(*(_one(*item) for item in active), return_exceptions=True)
        _elapsed_ms = int((time.monotonic() - _t0) * 1000)
        _failed = 0
        for (kind, _filename, _rid, key), result in zip(active, results):
            if isinstance(result, BaseException):
                logger.warning(crash_fmt, adapter.name, *crash_args(kind, key), result)
            if result is None or isinstance(result, BaseException):
                _failed += 1
            else:
                out_paths.append(result[0])
                out_mimes.append(result[1])
        logger.info(
            "[%s] media resolve batch: scope=%s concurrency=%d total=%d ok=%d failed=%d elapsed_ms=%d",
            adapter.name, scope, adapter.media_resolve_concurrency, len(active), len(out_paths), _failed, _elapsed_ms,
        )

    @classmethod
    async def _resolve_ybres_refs(cls, adapter, refs: list[tuple[str, str, str]], *, log_prefix: str) -> tuple[list[str], list[str]]:
        """Resolve ``(rid, kind, filename)`` ybres tuples to local paths (bounded concurrency,
        input order preserved, per-rid failures isolated). Cache hits are served without a fetch."""
        from gateway.platforms.yuanbao import (
            List,
            _RESOLVABLE_MEDIA_KINDS,
        )

        media_paths: list[str] = []
        mimes: list[str] = []
        active = [(kind, filename, rid, rid) for rid, kind, filename in refs
                  if kind in _RESOLVABLE_MEDIA_KINDS and not cls._append_cached_resource(adapter, rid, media_paths, mimes)]
        if active:
            await cls._gather_resolve(
                adapter, active, "ybres", media_paths, mimes,
                get_url=lambda rid: cls._fetch_resource_url(adapter, rid),
                fail_fmt="[%s] %s resolve failed: rid=%s kind=%s err=%s", fail_args=lambda kind, rid: (log_prefix, rid, kind),
                crash_fmt="[%s] %s resolve crashed: rid=%s kind=%s err=%s", crash_args=lambda kind, rid: (log_prefix, rid, kind),
                log_tag=lambda rid: f"{log_prefix} rid={rid}",
            )
        return media_paths, mimes

    @classmethod
    async def _collect_observed_media(cls, adapter, source) -> tuple[list[str], list[str]]:
        """Resolve recent observed image/file anchors from the transcript into ``(local_paths, mimes)``."""
        from gateway.platforms.yuanbao import (
            List,
            OBSERVED_MEDIA_BACKFILL_LOOKBACK,
            OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN,
            TranscriptReadError,
            Tuple,
            _YB_RES_REF_RE,
            _iter_ybres_refs,
        )

        store = _session_store(adapter)
        if not store:
            return [], []
        try:
            history = store.load_transcript(store.get_or_create_session(source).session_id)
        except TranscriptReadError as exc:
            # Hydrate nothing rather than silently acting as if the session had no observed media.
            logger.warning("[%s] Observed-media hydration: transcript unreadable: %s", adapter.name, exc)
            return [], []
        except Exception as exc:
            logger.warning("[%s] Observed-media hydration setup failed: %s", adapter.name, exc)
            return [], []
        # Walk newest→oldest (matches within a message too) so the per-turn cap keeps the
        # *latest* refs; ``order`` is reversed back to chronological before resolving.
        order: list[tuple[str, str, str]] = []  # (rid, kind, filename)
        seen: set = set()
        for msg in reversed((history or [])[-OBSERVED_MEDIA_BACKFILL_LOOKBACK:]):
            content = msg.get("content")
            if not isinstance(content, str) or "|ybres:" not in content:
                continue
            for rid, kind, filename in _iter_ybres_refs(reversed(list(_YB_RES_REF_RE.finditer(content)))):
                if rid not in seen:
                    seen.add(rid)
                    order.append((rid, kind, filename))
                if len(order) >= OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN:
                    break
            if len(order) >= OBSERVED_MEDIA_BACKFILL_MAX_RESOLVE_PER_TURN:
                break
        if not order:
            return [], []
        return await cls._resolve_ybres_refs(adapter, order[::-1], log_prefix="observed-media")

    @classmethod
    async def _resolve_quote_media(cls, adapter, quote_media_refs: list[tuple[str, str, str]]) -> tuple[list[str], list[str]]:
        """Resolve ybres anchors of the quoted message (from QuoteContextMiddleware)."""
        return await cls._resolve_ybres_refs(adapter, quote_media_refs, log_prefix="quote")

    @staticmethod
    def _collect_quote_local_media(ctx: InboundContext) -> tuple[list[str], list[str]]:
        """DM quote fallback: ``(local_paths, mimes)`` for media PatchAnchorsMiddleware already
        rewrote to ``[image: /path]`` / ``[file: name → /path]`` on the original turn. Unresolved
        anchors were that turn's failure — no re-download here."""
        from gateway.platforms.yuanbao import (
            List,
            _YB_LOCAL_MEDIA_RE,
            guess_mime_type,
            os,
        )

        paths: list[str] = []
        mimes: list[str] = []
        cache = getattr(ctx.adapter, "_msg_content_cache", None)
        text = cache.get(ctx.reply_to_message_id) if ctx.reply_to_message_id and cache else None
        for m in _YB_LOCAL_MEDIA_RE.finditer(text if isinstance(text, str) else ""):
            kind = (m.group(1) or "").strip().lower()
            path = (m.group(2) or "").strip()
            if not path or path in paths or not os.path.exists(path):
                continue
            paths.append(path)
            mimes.append(guess_mime_type(os.path.basename(path)) or ("image/jpeg" if kind == "image" else "application/octet-stream"))
        return paths, mimes

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        # In groups only @bot / owner-command turns reach here (GroupAtGuard short-circuits the
        # rest), so media download and observed-media hydration need no @bot re-check.
        from gateway.platforms.yuanbao import (
            List,
            Tuple,
        )

        adapter = ctx.adapter
        urls: list[str] = []
        types: list[str] = []

        def _add_unique_pairs(pair_lists: tuple[list[str], list[str]]) -> None:
            for u, m in zip(*pair_lists):
                if u and u not in urls:
                    urls.append(u)
                    types.append(m)
        own_pairs = await self._resolve_media_urls(adapter, ctx.media_refs)  # 1) media carried by this message
        own_count = sum(1 for u in own_pairs[0] if u)
        _add_unique_pairs(own_pairs)
        # 2) Quoted media takes priority; else observed-media backfill in groups only (DM media
        #    was already resolved on its own turn).
        if ctx.reply_to_message_id is not None:
            if ctx.quote_media_refs:
                _add_unique_pairs(await self._resolve_quote_media(adapter, ctx.quote_media_refs))
            else:  # DM rows carry no platform message_id → recover already-local media from the msg cache.
                _add_unique_pairs(self._collect_quote_local_media(ctx))
        elif ctx.chat_type == "group":
            try:
                _add_unique_pairs(await self._collect_observed_media(adapter, ctx.source))
            except Exception as exc:
                logger.warning("[%s] observed-image hydration raised, continuing anyway: %s", adapter.name, exc)
        ctx.media_urls = urls
        ctx.media_types = types
        # Re-check placeholder using ``own_count``: placeholder text with only quote/observed
        # media (no fresh attachment of its own) is still skippable.
        if PlaceholderFilterMiddleware.is_skippable_placeholder(ctx.raw_text, own_count):
            logger.debug("[%s] Skip placeholder after media download: %r", adapter.name, ctx.raw_text)
            return  # Stop pipeline
        await next_fn()


class PatchAnchorsMiddleware(InboundMiddleware):
    """Replace ``[kind|ybres:RID]`` anchors in raw_text with the local paths MediaResolveMiddleware
    produced, so the transcript records usable paths. Only resolved media (paths starting with
    ``/``) are substituted; other anchors stay untouched."""
    name = "patch-anchors"

    @staticmethod
    def _patch(text: str, urls: list[str], types: list[str]) -> str:
        from gateway.platforms.yuanbao import (
            _YB_RES_REF_RE,
            os,
        )

        patched = text
        for u, m in zip(urls, types):
            if not u.startswith("/"):
                continue
            anchor_match = _YB_RES_REF_RE.search(patched)
            if not anchor_match:
                break
            kind, _, filename = anchor_match.group(1).partition(":")
            kind = kind.strip()
            if kind == "image" and m.startswith("image/"):
                replacement = f"[image: {u}]"
            elif kind == "file":
                replacement = f"[file: {filename.strip() or os.path.basename(u)} → {u}]"
            elif kind == "video":
                replacement = f"[video: {u}]"
            else:
                continue
            patched = patched[: anchor_match.start()] + replacement + patched[anchor_match.end():]
        return patched

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        ctx.raw_text = self._patch(ctx.raw_text, ctx.media_urls, ctx.media_types)
        await next_fn()


class DispatchMiddleware(InboundMiddleware):
    """Build the MessageEvent and dispatch it (groups: serialised per session via a queue)."""
    name = "dispatch"

    async def handle(self, ctx: InboundContext, next_fn) -> None:
        from gateway.platforms.yuanbao import (
            MessageEvent,
            MessageType,
            asyncio,
        )

        adapter = ctx.adapter
        # The adapter seam: keyed in the owner profile's namespace, same as ``handle_message``.
        _sk = adapter._source_session_key(ctx.source)

        async def _dispatch_inbound_event() -> None:
            if any(mt.startswith(("application/", "text/")) for mt in ctx.media_types):
                # Classification: DOCUMENT wins over PHOTO/VIDEO/AUDIO for mixed attachments — run.py's
                # image handling keys off the per-path image/* mime types regardless of message_type, but
                # document-context injection gates strictly on MessageType.DOCUMENT (same precedence as
                # Email/Signal, PR #44695).
                msg_type = MessageType.DOCUMENT
            else:  # yuanbao-local subtypes (CHAT_RECORD) are deep-parsed into text → TEXT downstream
                msg_type = ctx.msg_type if isinstance(ctx.msg_type, MessageType) else MessageType.TEXT
            event = MessageEvent(
                text=ctx.raw_text, message_type=msg_type, source=ctx.source, message_id=ctx.msg_id or None,
                raw_message=ctx.push, media_urls=list(ctx.media_urls), media_types=list(ctx.media_types),
                reply_to_message_id=ctx.reply_to_message_id, reply_to_text=ctx.reply_to_text,
                channel_prompt=ctx.channel_prompt,
            )
            if _sk and ctx.msg_id:
                adapter._processing_msg_ids[_sk] = ctx.msg_id
                adapter._processing_msg_texts[_sk] = ctx.raw_text or ""
            if ctx.msg_id and ctx.raw_text:
                cache = adapter._msg_content_cache
                cache[ctx.msg_id] = ctx.raw_text
                for k in list(cache)[:max(0, len(cache) - 200)]:  # bounded: drop oldest
                    del cache[k]
            await adapter.handle_message(event)
        if ctx.chat_type == "group":
            is_new = _sk not in adapter._group_queues
            queue = adapter._group_queues.setdefault(_sk, asyncio.Queue())
            queue.put_nowait(_dispatch_inbound_event)
            logger.info("[%s] Group message enqueued (qsize=%d) for %s", adapter.name, queue.qsize(), (_sk or "")[:50])
            if is_new:
                self._track_inbound(adapter, self._consume_group_queue(adapter, _sk), f"yuanbao-group-consumer-{(_sk or '')[:30]}")
        else:
            self._track_inbound(adapter, _dispatch_inbound_event(), f"yuanbao-inbound-{ctx.msg_id or 'unknown'}")
        await next_fn()

    @staticmethod
    def _track_inbound(adapter, coro, name: str) -> None:
        from gateway.platforms.yuanbao import (
            asyncio,
        )

        task = asyncio.create_task(coro, name=name)
        adapter._inbound_tasks.add(task)
        task.add_done_callback(adapter._inbound_tasks.discard)

    @staticmethod
    async def _consume_group_queue(adapter: YuanbaoAdapter, session_key: str) -> None:
        """Drain the group queue one dispatch at a time, waiting for each to finish; exits after 2s idle."""
        from gateway.platforms.yuanbao import (
            asyncio,
        )

        queue = adapter._group_queues.get(session_key)
        if not queue:
            return
        try:
            while True:
                try:
                    dispatch_fn = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    break
                logger.debug("[%s] Group queue: dispatching for %s (remaining=%d)", adapter.name, (session_key or "")[:50], queue.qsize())
                try:
                    await dispatch_fn()
                    while session_key in adapter._active_sessions:
                        await asyncio.sleep(0.1)
                except Exception:
                    logger.exception("[%s] Group queue consumer error", adapter.name)
        finally:
            adapter._group_queues.pop(session_key, None)


class InboundPipelineBuilder:
    """Assembles the default Yuanbao inbound pipeline (order matters)."""
    _DEFAULT_MIDDLEWARES: list[type] = [
        DecodeMiddleware, ExtractFieldsMiddleware, RecallGuardMiddleware, DedupMiddleware, SkipSelfMiddleware,
        ChatRoutingMiddleware, AccessGuardMiddleware, ExtractContentMiddleware, PlaceholderFilterMiddleware,
        OwnerCommandMiddleware, BuildSourceMiddleware, GroupAtGuardMiddleware, AutoSetHomeMiddleware,
        GroupAttributionMiddleware, ClassifyMessageTypeMiddleware, QuoteContextMiddleware,
        ForwardedRecordsParseMiddleware, MediaResolveMiddleware, PatchAnchorsMiddleware, DispatchMiddleware,
    ]

    @classmethod
    def build(cls) -> InboundPipeline:
        pipeline = InboundPipeline()
        for mw_cls in cls._DEFAULT_MIDDLEWARES:
            pipeline.use(mw_cls())
        return pipeline
