"""RelayAdapter — one generic gateway adapter fronted by the connector. EXPERIMENTAL.

A single ``BasePlatformAdapter`` subclass that receives a ``CapabilityDescriptor`` at
handshake (which platform it fronts, which capabilities to advertise) and delegates
all wire I/O to an injected transport. There is NO per-platform gateway code: only
the connector knows "this chat_id maps to a Discord channel"; the gateway sees an
ordinary ``MessageEvent`` in and calls ``adapter.send`` out. Transport protocol and
descriptor schema may change without a deprecation cycle until >=2 Class-1 platforms
validate them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Union

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult,
)
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.media import RelayMediaClient
from gateway.relay.transport import RelayTransport
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

# The drain-path going-idle ACK budget must stay strictly under the runner's
# default adapter disconnect timeout (5s) or cancellation fires before
# transport.disconnect() and leaves the websocket open. With transport teardown
# budgets of 1s each for supervisor, reader and ws.close, the drain stays <5s.
_RELAY_GO_IDLE_ON_DISCONNECT_TIMEOUT_S = 2.0
_RELAY_REVOCATION_MONITOR_TEARDOWN_TIMEOUT_S = 1.0

# Link detection for the fresh-final unfurl route: raw URLs, Slack mrkdwn links
# and markdown links. Permissive on purpose — a false positive costs one fresh
# (non-edited) final; a false negative silently loses the preview.
_URL_RE = re.compile(r"https?://|<https?:|\]\(https?:")

# Already-answered prompt ids to remember so a duplicate answer (double tap or
# connector redelivery) reads as a repeat, not a stale prompt.
_RESOLVED_PROMPT_MEMORY = 256

# Connector promptCodec.decodePromptCallback id alphabet ([A-Za-z0-9_.-], <=32).
_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

_SLACK = Platform.SLACK.value

# Prompt option id -> in-channel ack label (the option set doubles as the choice allowlist).
_EXEC_APPROVAL_LABELS = {
    "once": "✅ Approved once",
    "session": "✅ Approved for session",
    "always": "✅ Approved permanently",
    "deny": "❌ Denied",
}
_SLASH_CONFIRM_LABELS = {"once": "✅ Approved once", "always": "🔒 Always approve", "cancel": "❌ Cancelled"}


def _utf16_len(text: str) -> int:
    """Count UTF-16 code units (Telegram's length unit)."""
    return len(text.encode("utf-16-le")) // 2


_LEN_FNS: Dict[str, Callable[[str], int]] = {"chars": len, "utf16": _utf16_len}


def _send_result(result: Dict[str, Any], **extra: Any) -> SendResult:
    """Project a connector ``outbound_result`` dict onto a SendResult."""
    return SendResult(
        success=bool(result.get("success")), message_id=result.get("message_id"),
        error=result.get("error"), **extra,
    )


def _event_ids(event) -> Tuple[Optional[str], Optional[str]]:
    """(message_id, chat_id) of an inbound event; message_id lives on the event, falls back to source."""
    message_id = getattr(event, "message_id", None) or getattr(event.source, "message_id", None)
    return message_id, getattr(event.source, "chat_id", None)


class RelayAdapter(BasePlatformAdapter):
    """Generic relay adapter advertising a connector-negotiated capability profile."""

    def __init__(
        self,
        config: PlatformConfig,
        descriptor: CapabilityDescriptor,
        transport: Optional[RelayTransport] = None,
    ) -> None:
        # Fronts many platforms but presents to the runner as Platform.RELAY.
        super().__init__(config, Platform.RELAY)
        self._transport = transport
        self._apply_descriptor(descriptor)
        # Per-chat egress routing caches learned from inbound events (send() only
        # receives a chat_id). The connector's egress guard resolves the owning tenant
        # from OUTBOUND metadata.scope_id / user_id, so we re-attach what we saw
        # inbound (_capture_scope).
        self._scope_by_chat: Dict[str, str] = {}
        self._dm_user_by_chat: Dict[str, str] = {}
        # chat_id -> chat_type: reproduces native Slack's synthetic DM-thread
        # suppression (a raw reply_to becomes a thread_ts connector-side, so a plain
        # DM reply would thread under the user).
        self._chat_type_by_chat: Dict[str, str] = {}
        # chat_id -> last triggering Slack message ts (typing/status lane's
        # synthetic thread anchor in thread-per-message mode).
        self._last_inbound_ts_by_chat: Dict[str, str] = {}
        # chat_id -> UNDERLYING platform ("discord", ...): one adapter fronts N
        # platforms on one WS and a reply must egress through the platform the
        # inbound came from. Empty for a single-platform gateway (connector default).
        self._platform_by_chat: Dict[str, str] = {}
        # chat_id -> (thread_id, initial_name) of the auto-thread the CONNECTOR
        # created for our latest send; read by the semantic thread-rename lane.
        self._auto_thread_by_chat: Dict[str, Tuple[str, str]] = {}
        # chat_id -> event fired when the entry above lands (wait_for_auto_thread_info).
        self._auto_thread_waiters: Dict[str, asyncio.Event] = {}
        # Bounded FIFO seen-set for inbound replay dedupe (insertion-ordered dict).
        self._seen_inbound: Dict[str, None] = {}
        # Live cards: draft_key -> draft_id of the OPEN native stream. Armed by
        # send_draft; consumed by send() to convert the turn-final into
        # draft(final=true) instead of a duplicate post. Keyed by _draft_key (chat +
        # per-turn identity), NOT bare chat: parallel turns in one DM are distinct
        # streams (per-chat keying merged three concurrent turns).
        self._open_draft_by_chat: Dict[str, int] = {}
        # draft_key -> draft_id of the most recently SEALED stream (mirror of the
        # connector's sealed-key tombstone): post-seal stragglers must neither
        # re-arm interception nor re-open a stream.
        self._sealed_draft_by_chat: Dict[str, int] = {}
        # Draft keys whose post-seal swallow has been logged once (bounded FIFO).
        self._tombstone_swallow_logged: Dict[str, int] = {}
        # Strong refs for fire-and-forget lifecycle acks (asyncio holds tasks weakly).
        self._lifecycle_ack_tasks: set = set()
        # Stream-is-the-message marker read by the stream consumer to keep ONE draft
        # stream per turn instead of bumping draft_id at tool boundaries. SLACK-ONLY:
        # the base send_draft contract is Telegram-shaped (draft clears, final is a
        # separate real send); setting this for any "draft" connector intercepted the
        # turn-final into draft(final=true) and no history message was ever posted.
        # A future platform with this semantic should advertise it via the descriptor.
        self.draft_stream_is_message = str(getattr(descriptor, "platform", "") or "").lower() == "slack"
        # Watches the transport for a terminal auth revocation (4401 after a
        # successful handshake = operator opted this instance out) and surfaces a
        # clean non-retryable "relay disabled" fatal instead of a retry spin.
        self._revocation_monitor: Optional[asyncio.Task[None]] = None
        # Lazily built client for the connector's /relay/media routes; None when
        # dial URL or creds are absent (media lanes degrade to text fallbacks).
        self._media_client: Optional["RelayMediaClient"] = None
        # prompt_id -> pending-prompt state for the interactive `prompt` op; the
        # user's pick comes back as a prompt_response naming this id and resolves the
        # waiting primitive like native button callbacks. Expire lazily (_pop_prompt).
        self._pending_prompts: Dict[str, Dict[str, Any]] = {}
        # Per-process marker prefixed onto every prompt id we mint. WHY: button
        # presses ride the passthrough plane, which the connector fans out to EVERY
        # live gateway session of the tenant, while _pending_prompts is process-local.
        # Without the marker a sibling cannot tell "someone else owns this" from "my
        # prompt expired", and the id-shaped text ("/c1") falls through to chat
        # dispatch as "Unknown command" — once per sibling (common in a DM).
        self._prompt_owner_nonce: str = secrets.token_hex(3)
        # Prompt ids this process already resolved, newest last (repeat answers are
        # consumed silently instead of treated as stale).
        self._resolved_prompts: "OrderedDict[str, float]" = OrderedDict()

    # ── capability surface (from descriptor) ─────────────────────────────
    @property
    def authorization_is_upstream(self) -> bool:
        """The connector enforces authorization (owner-only author-binding before
        delivery), so relay users must not be default-denied for lack of a local
        ``RELAY_ALLOWED_USERS`` allowlist."""
        return True

    @property
    def message_len_fn(self) -> Callable[[str], int]:
        return _LEN_FNS.get(self.descriptor.len_unit, len)

    @property
    def supports_status_text(self) -> bool:  # type: ignore[override]
        """Whether the fronted platform renders a TEXT status line: Slack's typing
        surface is the assistant status line, so run.py's live-status lane may feed
        per-tool phrases; other platforms have textless bubbles and must NOT receive
        them. Reflects the PRIMARY identity, like the scalar ``descriptor``."""
        return self.descriptor.platform == _SLACK

    # ── per-chat capability resolution (multi-platform) ──────────────────
    def _negotiated_descriptor(self, platform: Optional[str]) -> Optional[CapabilityDescriptor]:
        """The transport's negotiated descriptor for ``platform``, or None (unknown
        platform, no transport, or a transport predating ``descriptor_for_platform``).
        Never raises — capability lookup must never break a send."""
        resolve = getattr(self._transport, "descriptor_for_platform", None) if platform else None
        if not callable(resolve):
            return None
        try:
            return resolve(platform)
        except Exception:  # noqa: BLE001
            return None

    def _chat_platform(self, chat_id: str) -> Optional[str]:
        """The chat's underlying platform as seen inbound, else the primary's."""
        return self._platform_by_chat.get(str(chat_id)) or self.descriptor.platform

    def _descriptor_for_chat(self, chat_id: str) -> CapabilityDescriptor:
        """The descriptor governing a specific chat. Platform caps genuinely differ
        (Discord 2000 / Telegram 4096 / Slack 39000), so the primary's scalar cap
        either fragments needlessly or over-sends into a platform 400. Falls back to
        the scalar when the chat's platform is unknown (never saw inbound)."""
        per_platform = self._negotiated_descriptor(self._platform_by_chat.get(str(chat_id)))
        return per_platform if per_platform is not None else self.descriptor

    def max_message_length_for_chat(self, chat_id: str) -> int:
        return self._descriptor_for_chat(chat_id).max_message_length

    def message_len_fn_for_chat(self, chat_id: str) -> Callable[[str], int]:
        return _LEN_FNS.get(self._descriptor_for_chat(chat_id).len_unit, len)

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> bool:
        # Needs BOTH the descriptor flag and an explicit "draft" op: supported_ops is
        # fail-open for legacy connectors, but "draft" did not exist pre-contract, so
        # it must NOT fail open. Resolved per chat when the caller names one (a
        # Telegram primary must not starve a Slack chat).
        desc = self._descriptor_for_chat(str(chat_id)) if chat_id is not None else self.descriptor
        if not (desc.supports_draft_streaming and "draft" in (desc.supported_ops or ())):
            return False
        # Slack chat.*Stream has no unfurl_links / unfurl_media; like native
        # SlackAdapter, refuse streaming when those knobs are set so chat.postMessage
        # can carry them.
        platform = self._chat_platform(chat_id) if chat_id is not None else desc.platform
        return not self._slack_unfurl_hints(platform)

    def prefers_fresh_final_streaming(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> bool:
        """Deliver streamed finals as a FRESH send when Slack unfurl is forced on.

        Slack evaluates link previews exactly once, at ``chat.postMessage``; a
        ``chat.update`` that INTRODUCES the URL never unfurls. Edit-based streaming
        posts its first frame before any URL exists, so a configured ``unfurl_*: true``
        can only surface via a fresh final that ``send()`` stamps with the hints. ONLY
        when the hints contain an explicit True: false-only hints (fail-closed) ride
        the placeholder post fine. Only link-bearing finals qualify — the relay has no
        delete op in contract v1, so a linkless fresh final would just be a duplicate.
        """
        platform = self._platform_by_chat.get(str(chat_id)) if chat_id is not None else None
        # The stream consumer's hook passes (content, metadata=...) only.
        if platform is None and isinstance(metadata, dict):
            platform = metadata.get("platform")
        if platform is None:
            platform = self.descriptor.platform
        hints = self._slack_unfurl_hints(platform)
        return bool(hints) and any(v is True for v in hints.values()) and bool(_URL_RE.search(content or ""))

    def stream_is_message_for_chat(self, chat_id: str) -> bool:
        """Per-chat stream-is-the-message semantic (see ``draft_stream_is_message``).
        A Slack primary must not impose seal semantics on a Telegram chat (its
        turn-final would become draft(final=true) — no history message), nor a
        Telegram primary deny a Slack chat native streaming. Platform-name inference
        is deliberate; a descriptor field is the eventual contract."""
        return str(self._descriptor_for_chat(str(chat_id)).platform or "").lower() == "slack"

    # ── Live cards: native draft streaming + task cards ──────────────────
    # Additive relay ops within contract v1, emitted when the negotiated descriptor
    # advertises them; the connector owns the platform API mechanics and the
    # send+edit fallback. Semantic bridge: the base send_draft contract is
    # Telegram-shaped (draft clears, final is a separate send()); Slack native
    # streaming makes the stream THE message, so the adapter tracks the open draft
    # per turn and converts that turn's final send() into draft(final=true).

    def supports_native_task_cards(self) -> bool:
        """Explicit advertisement required — same no-fail-open rule as "draft"."""
        return "task_card" in (self.descriptor.supported_ops or ())

    def native_task_cards_enabled(self) -> bool:
        """TurnRunner opt-in probe (gateway/run.py calls THIS name, same contract as
        native Slack); without the alias the card lane silently stays text-mode."""
        return self.supports_native_task_cards()

    @staticmethod
    def _draft_key(chat_id: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Coordination key for one turn's stream. Prefers a PER-TURN identity (the
        triggering inbound ``message_id`` / ``reply_to_message_id``) over the thread
        anchor: two parallel turns inside ONE thread share thread_ts (turn A's final
        sealed turn B's stream), and a flat DM with no anchor degraded to the bare
        chat id. Anchor is the fallback for placement-only callers; bare chat last."""
        md = metadata or {}
        turn_id = md.get("message_id") or md.get("reply_to_message_id")
        if turn_id:
            return f"{chat_id}:turn:{turn_id}"
        anchor = md.get("thread_ts") or md.get("thread_id") or ""
        return f"{chat_id}:{anchor}"

    # Cap for the draft/seal coordination dicts (per-turn keys); matches the
    # connector's tombstone store size.
    _DRAFT_STATE_CAP = 512

    @classmethod
    def _evict_oldest(cls, d: Dict[str, Any], cap: Optional[int] = None) -> None:
        """FIFO-bound an insertion-ordered dict in place (default cap: draft state)."""
        while len(d) > (cls._DRAFT_STATE_CAP if cap is None else cap):
            d.pop(next(iter(d)), None)

    @staticmethod
    def _card_key(reply_to: Optional[str], metadata: Optional[Dict[str, Any]]) -> str:
        """Per-turn task-card identity — same precedence as ``_draft_key``; one
        derivation for send AND stop so the stop always hits the stream the send opened."""
        md = metadata or {}
        anchor = (
            reply_to
            or md.get("message_id")
            or md.get("reply_to_message_id")
            or md.get("thread_ts")
            or md.get("thread_id")
            or "root"
        )
        return f"turn:{anchor}"

    def _match_open_draft(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Resolve which open stream (if any) a turn-final send belongs to. Exact key
        match first. Callers carrying a per-turn MESSAGE id never fall back — their
        identity is authoritative. Callers without one may absorb into the chat's
        single open stream; with several open the send stays plain: a duplicate
        message is recoverable, sealing someone else's stream is not."""
        key = self._draft_key(str(chat_id), metadata)
        if key in self._open_draft_by_chat:
            return key
        md = metadata or {}
        if md.get("message_id") or md.get("reply_to_message_id"):
            return None
        prefix = f"{chat_id}:"
        candidates = [k for k in self._open_draft_by_chat if k.startswith(prefix)]
        if len(candidates) == 1:
            # Absorbing a send into a stream is a significant decision (the
            # prompt-ack-seals-own-stream bug); log it so the next mismatch is a grep.
            logger.info(
                "relay: absorbing identity-less send into the single open "
                "stream %s (single-open-stream fallback)",
                candidates[0],
            )
            return candidates[0]
        return None

    async def _outbound(self, chat_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Send one outbound frame tagged with the chat's underlying platform."""
        return await self._transport.send_outbound(  # type: ignore[union-attr]
            action, platform=self._platform_by_chat.get(str(chat_id))
        )

    async def _gated_op(
        self,
        chat_id: str,
        action: Dict[str, Any],
        *,
        decline_level: Optional[int] = logging.WARNING,
        subject: Any = None,
        platform: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Emit one best-effort, op-gated frame; None when the caller must fall back.

        None covers every unavailability: op not advertised (probe the descriptor
        instead of parsing a connector error), no transport, transport raised, or a
        structured connector decline (logged at ``decline_level``; None = silent).
        """
        op = action["op"]
        if self._transport is None or not self.descriptor.supports_op(op):
            return None
        try:
            result = await self._transport.send_outbound(
                action, platform=platform or self._platform_by_chat.get(str(chat_id))
            )
        except Exception:  # noqa: BLE001 - transport failure degrades to the caller's fallback
            logger.debug("relay %s transport failure", op, exc_info=True)
            return None
        if not result.get("success"):
            if decline_level is not None:
                logger.log(
                    decline_level, "relay %s declined for %s: %s",
                    op, chat_id if subject is None else subject, result.get("error"),
                )
            return None
        return result

    def _text_metadata(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Metadata for a text egress frame: format hints + tenant discriminators.
        Draft, seal, send and edit are all text lanes — a streamed final can only
        render blocks if every frame carries the hint (a hintless seal is the
        plain-code-block downgrade)."""
        return self._with_scope(chat_id, self._with_format_hints_for_chat(chat_id, metadata))

    def _draft_frame(
        self, chat_id: str, draft_id: int, content: str, final: bool, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """One ``draft`` op frame (``final=True`` seals the stream)."""
        return {
            "op": "draft",
            "chat_id": chat_id,
            "draft_id": draft_id,
            "content": content,
            "final": final,
            "metadata": self._text_metadata(chat_id, dict(metadata or {})),
        }

    async def send_draft(
        self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self.supports_draft_streaming(chat_id=str(chat_id)):
            raise NotImplementedError("connector does not advertise the 'draft' relay op")
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        # Arm optimistically BEFORE the transport call (a lossy ack often means
        # delivered), but NEVER for a draft_id already sealed on this key: a straggler
        # after the seal re-armed interception with no live stream, and the next
        # unrelated send was wrongly converted into a seal.
        chat_key = self._draft_key(str(chat_id), metadata)
        if self._sealed_draft_by_chat.get(chat_key) == draft_id:
            # Post-seal straggler: content is already in the sealed message; report
            # success, send nothing. Log the FIRST swallow per key — one straggler is
            # the normal race, but a burst means something sealed a live stream
            # mid-flight (silence here cost a forensic hunt).
            if chat_key not in self._tombstone_swallow_logged:
                self._tombstone_swallow_logged[chat_key] = draft_id
                self._evict_oldest(self._tombstone_swallow_logged)
                logger.warning(
                    "relay: draft frame for %s swallowed by post-seal "
                    "tombstone (draft_id=%s) — expected for a straggler; "
                    "a live stream freezing NOW means something sealed it "
                    "mid-flight",
                    chat_key,
                    draft_id,
                )
            return SendResult(success=True)
        # Arm seal-interception ONLY for stream-is-the-message chats: on a
        # Telegram-shaped connector the final MUST go out as a real send.
        if self.stream_is_message_for_chat(str(chat_id)):
            self._open_draft_by_chat[chat_key] = draft_id
            self._evict_oldest(self._open_draft_by_chat)
        try:
            result = await self._outbound(chat_id, self._draft_frame(chat_id, draft_id, content, False, metadata))
        except Exception as e:
            # Ambiguous (stale socket, mid-write drop): may have been delivered;
            # keep interception armed.
            return SendResult(success=False, error=f"draft transport error: {e}")
        if result.get("success"):
            return SendResult(success=True)
        if result.get("ambiguous"):
            # Ack lost (transport timeout, returned rather than raised): same
            # contract as the except branch — keep interception armed.
            return SendResult(success=False, error=str(result.get("error") or "draft ack lost"))
        # DEFINITE connector rejection: disarm. The stream consumer falls back to
        # edit-based streaming and its turn-final must go out as a REAL send, not a
        # seal on a stream the connector just declared unusable.
        if self._open_draft_by_chat.get(chat_key) == draft_id:
            self._open_draft_by_chat.pop(chat_key, None)
        return SendResult(success=False, error=str(result.get("error") or "draft failed"))

    async def _seal_open_draft(
        self,
        chat_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]],
        *,
        draft_key: Optional[str] = None,
    ) -> SendResult:
        """Convert the turn-final send into the sealing draft frame."""
        if draft_key is None:
            draft_key = self._draft_key(str(chat_id), metadata)
        draft_id = self._open_draft_by_chat.pop(draft_key)
        # Tombstone BEFORE the transport call: whatever the ack says, this draft_id
        # must never be re-armed by a straggler frame.
        self._sealed_draft_by_chat[draft_key] = draft_id
        self._evict_oldest(self._sealed_draft_by_chat)
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        seal_frame = self._draft_frame(chat_id, draft_id, content, True, metadata)

        _seal_platform = self._platform_by_chat.get(str(chat_id))
        _transport = self._transport  # narrowed by the None-guard above

        async def _attempt() -> Optional[Dict[str, Any]]:
            """One seal attempt; None means ambiguous (exception or lost ack)."""
            try:
                r = await _transport.send_outbound(seal_frame, platform=_seal_platform)
            except Exception as e:
                logger.warning("relay seal transport error (ambiguous): %s", e)
                return None
            if r.get("ambiguous"):
                logger.warning("relay seal ack lost (ambiguous): %s", r.get("error"))
                return None
            return r

        # Ambiguous outcomes retry the SAME idempotent frame once: the connector's
        # sealed-key tombstone returns the original stream ts for a repeated final
        # and never opens a second stream. Two consecutive ack losses on one socket
        # almost always mean the transport is down. Cancellation safety: the open
        # entry was popped and the tombstone written BEFORE the await; restore both
        # before re-raising so the later abandon pass can still seal the stream.
        try:
            result = await _attempt()
            if result is None:
                result = await _attempt()
        except asyncio.CancelledError:
            self._open_draft_by_chat[draft_key] = draft_id
            if self._sealed_draft_by_chat.get(draft_key) == draft_id:
                self._sealed_draft_by_chat.pop(draft_key, None)
            raise
        if result is None:
            return SendResult(success=False, error="draft seal ambiguous after retry (transport ack lost)")
        if result.get("success"):
            # The connector returns the stream's ts as the message identity.
            return SendResult(success=True, message_id=str(result.get("message_id") or "") or None)
        return SendResult(success=False, error=str(result.get("error") or "draft seal failed"))

    async def _absorb_into_open_draft(
        self, chat_id: str, content: str, metadata: Dict[str, Any], interim: bool
    ) -> Optional[SendResult]:
        """Seal an open native stream with this turn-final; None = do a plain send.

        An open stream absorbs the turn-final whichever egress door it arrives
        through (send / send_for_platform) — otherwise the stream is left frozen
        mid-word AND the final posts as a duplicate. A failed seal must NOT swallow
        the final: the consumer already disabled the draft transport, so fall through
        to a plain send (the orphaned stream is sealed connector-side). Interim sends
        (commentary, tail flush, lifecycle acks) never seal.
        """
        if interim:
            return None
        key = self._match_open_draft(str(chat_id), metadata)
        if key is None:
            return None
        seal = await self._seal_open_draft(chat_id, content, metadata, draft_key=key)
        if seal.success:
            return seal
        logger.warning("relay seal failed (%s); delivering turn-final as plain send", seal.error)
        return None

    async def _card_frame(
        self, chat_id: str, op: str, reply_to: Optional[str], metadata: Dict[str, Any], **fields: Any
    ) -> Union[SendResult, Dict[str, Any]]:
        """Emit one task-card op: the connector result dict, or a failed SendResult
        when the lane is unavailable / the transport raised.

        Card frames are advisory and run inside the progress loop / turn-cleanup
        path: an escaping exception there skipped final delivery, so transport
        errors degrade to the TurnRunner's text fallback instead of raising.
        """
        if not self.supports_native_task_cards():
            return SendResult(success=False, error="connector does not advertise task_card")
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        frame = {
            "op": op,
            "chat_id": chat_id,
            "card_id": self._card_key(reply_to, metadata),
            **fields,
            "metadata": self._with_scope(chat_id, metadata),
        }
        try:
            return await self._outbound(chat_id, frame)
        except Exception as e:
            return SendResult(success=False, error=f"{op} transport error: {e}")

    async def send_native_task_card_progress(
        self,
        chat_id: str,
        tasks: list,
        *,
        title: str = "Hermes is working",
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fallback_text: Optional[str] = None,
    ) -> SendResult:
        """Relay leg of the task-card lane: emit one card frame.

        SIGNATURE CONTRACT: the TurnRunner calls this with the NATIVE Slack
        adapter's keyword contract, not a card_id. ``fallback_text``/``title``
        are accepted for parity but not forwarded (the connector's plan-mode
        stream renders task chunks; field limits are enforced connector-side).

        See #85476.
        """
        merged_meta = dict(metadata or {})
        if reply_to and "thread_ts" not in merged_meta:
            # Slack card streams are thread replies anchored on the trigger.
            merged_meta["thread_ts"] = str(reply_to)
        result = await self._card_frame(
            chat_id, "task_card", reply_to, merged_meta, chunks=[dict(t) for t in tasks]
        )
        if isinstance(result, SendResult):
            return result
        if result.get("success"):
            return SendResult(success=True)
        return SendResult(success=False, error=str(result.get("error") or "task_card failed"))

    async def stop_native_task_card_progress(
        self,
        chat_id: str,
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Seal the card stream at turn end (idempotent connector-side); same key derivation as send."""
        result = await self._card_frame(chat_id, "task_card_stop", reply_to, dict(metadata or {}))
        if isinstance(result, SendResult):
            return result
        return SendResult(success=bool(result.get("success")))

    async def abandon_open_draft(
        self, chat_id: str, content: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Seal an orphaned stream when its turn dies (/stop, /new, supersede), in
        place with ``content`` (the text already on screen) so the seal adds and
        claims nothing; otherwise the live indicator stays forever and the NEXT turn
        could inherit the armed interception state. Failure is reported, never raised."""
        draft_key = self._match_open_draft(str(chat_id), metadata)
        if draft_key is None:
            return SendResult(success=True)  # nothing armed — no-op
        try:
            return await self._seal_open_draft(chat_id, content, metadata, draft_key=draft_key)
        except Exception as e:
            return SendResult(success=False, error=f"abandon seal transport error: {e}")

    # ── abstract methods (delegated to the transport) ────────────────────
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # ``is_reconnect`` is part of the BasePlatformAdapter.connect contract (the
        # reconnect watcher passes it; refusing the kwarg would break recovery).
        # Relay IGNORES it: messages buffered during a gap live in the CONNECTOR's
        # durable buffer and replay on re-handshake; routine WS drops are handled by
        # the transport's own reconnect supervisor.
        if self._transport is None:
            # ``is_reconnect`` is part of the BasePlatformAdapter.connect contract: the gateway's reconnect
            # watcher (gateway/run.py) re-establishes a platform after a fatal adapter error by building a
            # fresh adapter and calling ``connect(is_reconnect=True)``. Relay MUST accept the kwarg or that
            # recovery path raises TypeError and the relay platform can never come back through the watcher.
            # The flag exists so adapters with a server-side update queue (e.g. Telegram's Bot API) preserve
            # that queue across an outage instead of dropping it (#46621). Routine WS drops are handled
            # entirely by the transport's own reconnect supervisor (WebSocketRelayTransport,
            # reconnect=True); a watcher-driven reconnect builds a fresh transport from scratch (the
            # fatal-error handler disconnect()s the old adapter first, cancelling its supervisor), so there
            # is nothing at the adapter layer to preserve.
            raise RuntimeError("RelayAdapter has no transport configured")
        self._transport.set_inbound_handler(self._on_inbound)
        # Interrupts and passthrough-plane forwards (Discord interactions, Twilio, …)
        # ride the SAME outbound WS — no inbound HTTP receiver, no public port.
        for setter_name, handler in (
            ("set_interrupt_inbound_handler", self.on_interrupt),
            ("set_passthrough_handler", self._on_passthrough),
        ):
            setter = getattr(self._transport, setter_name, None)
            if callable(setter):
                setter(handler)
        if not await self._transport.connect():
            return False
        # Adopt the connector-advertised descriptor in place of the placeholder.
        try:
            descriptor = await self._transport.handshake()
        except Exception as exc:  # noqa: BLE001 - a failed handshake = a failed connect
            logger.warning("relay handshake failed: %s", exc)
            return False
        self._apply_descriptor(descriptor)
        # Only the production WebSocket transport exposes `auth_revoked`.
        if hasattr(self._transport, "auth_revoked"):
            self._start_revocation_monitor()
        return True

    def _start_revocation_monitor(self) -> None:
        """Spawn (once) the task turning a transport auth-revocation into a clean
        non-retryable 'relay disabled' fatal. Idempotent."""
        if self._revocation_monitor is not None and not self._revocation_monitor.done():
            return
        try:
            self._revocation_monitor = asyncio.create_task(
                self._watch_for_revocation(), name="relay-revocation-monitor"
            )
        except RuntimeError:
            # No running loop (a unit test calling connect() via a stub).
            self._revocation_monitor = None

    async def _watch_for_revocation(self, poll_interval_s: float = 1.0) -> None:
        """Poll for a terminal 4401 revocation (opt-out), then surface a non-retryable
        `relay_disabled` fatal so the adapter is cleanly removed rather than queued
        for reconnection (the credential is dead until the instance is recreated)."""
        transport = self._transport
        if transport is None:
            return
        while not getattr(transport, "auth_revoked", False):
            await asyncio.sleep(poll_interval_s)
        logger.warning("relay credential revoked (opt-out) — marking the relay adapter disabled")
        self._set_fatal_error(
            "relay_disabled", "Relay disabled (opted out — recreate the instance to re-enable)",
            retryable=False,
        )
        try:
            await self._notify_fatal_error()
        except Exception:  # noqa: BLE001 - notification is best-effort
            logger.debug("relay revocation fatal-error notify failed", exc_info=True)

    def _apply_descriptor(self, descriptor: CapabilityDescriptor) -> None:
        """Adopt a (re)negotiated descriptor into the live capability surface."""
        self.descriptor = descriptor
        self.MAX_MESSAGE_LENGTH = descriptor.max_message_length
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")
        # Cron in_channel continuable surface (D6 gate in cron/scheduler.py);
        # class default is False, so only an explicit descriptor bit turns it on.
        self.supports_inchannel_continuable = bool(getattr(descriptor, "supports_inchannel_continuable", False))

    async def _on_inbound(self, event) -> None:
        """Bridge a connector-delivered MessageEvent into the normal adapter path."""
        # Inbound replay dedupe: the relay leg is at-least-once — on WS re-handshake
        # the connector replays its durable buffer, and a long turn straddling a
        # quiet socket drop got re-run (final answer 2-5x). Platform message identity
        # is stable across replays.
        dedupe_key = self._inbound_dedupe_key(event)
        if dedupe_key is not None:
            if dedupe_key in self._seen_inbound:
                logger.info("relay inbound dropped as replay (dedupe key=%s)", dedupe_key)
                return
            self._seen_inbound[dedupe_key] = None
            self._evict_oldest(self._seen_inbound, self._SEEN_INBOUND_MAX)
        self._capture_scope(event)
        self._stamp_slack_session_thread(event)
        # A structured prompt answer resolves its waiting primitive and is CONSUMED —
        # never also dispatched as chat.
        if await self._consume_prompt_response(event):
            return
        await self._localize_inbound_media(event)
        await self.handle_message(event)

    _SEEN_INBOUND_MAX = 512

    def _inbound_dedupe_key(self, event) -> Optional[str]:
        """Stable replay identity: (platform, chat, platform message id). The platform
        joins the key because one relay socket can front several platforms whose
        numeric ids may collide. None when the event carries no platform message id —
        those never dedupe (fail-open: dropping a real message beats rerunning one)."""
        source = getattr(event, "source", None)
        message_id = getattr(event, "message_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not message_id or not chat_id:
            return None
        # Enum value when present, plain string otherwise: both spellings of one
        # platform must produce ONE key.
        raw_platform = getattr(source, "platform", None)
        platform = getattr(raw_platform, "value", raw_platform) or ""
        return f"{platform}:{chat_id}:{message_id}"

    def _relay_platform_extra(self, platform: str) -> Dict[str, Any]:
        """``platforms.relay.extra.<platform>.*`` — relay-namespaced mirror of a native
        platform's knobs (``platforms.<platform>`` keeps meaning native settings).
        Legacy fallback: flat keys on the relay extra when no ``<platform>`` object exists."""
        extra = getattr(self.config, "extra", None) or {}
        sub = extra.get(platform)
        return sub if isinstance(sub, dict) else extra

    def _relay_slack_extra(self) -> Dict[str, Any]:
        return self._relay_platform_extra("slack")

    @staticmethod
    def _coerce_flag(raw: Any, default: bool) -> bool:
        """Coerce an operator-supplied boolean exactly as native Slack does: a
        YAML-quoted ``"false"`` must turn the flag OFF (bare ``bool()`` would read
        the non-empty string as True and silently ignore the switch)."""
        if raw is None:
            return default
        return raw if isinstance(raw, bool) else str(raw).strip().lower() in _TRUTHY

    def _slack_flag(self, knob: str, default: bool) -> bool:
        """A coerced boolean knob from the relay Slack extra; ``default`` on any config-shape error."""
        try:
            return self._coerce_flag(self._relay_slack_extra().get(knob), default)
        except Exception:  # noqa: BLE001 - config shape is operator-owned
            return default

    def _effective_reply_in_thread(self) -> bool:
        """Resolve the thread-per-message vs flat-DM mode for fronted Slack."""
        return self._slack_flag("reply_in_thread", True)

    def _dm_top_level_threads_as_sessions(self) -> bool:
        """Native-parity escape hatch: per-message DM sessions on/off. Default True:
        in thread-per-message mode each top-level DM message keys its own session.
        False keeps threaded PLACEMENT but ONE rolling DM session (legacy steer/queue
        posture), decoupled from reply_in_thread."""
        return self._slack_flag("dm_top_level_threads_as_sessions", True)

    def _slack_unfurl_hints(self, platform: Optional[str]) -> Optional[Dict[str, bool]]:
        """Slack-only outbound link-preview knobs (``unfurl_links``/``unfurl_media``)
        from the relay namespace. Only explicitly configured booleans are returned
        (omitted keys preserve Slack's default); YAML strings are coerced, junk
        dropped. Non-Slack platforms return None so their metadata is never polluted."""
        if str(platform or "").lower() != _SLACK:
            return None
        extra = self._relay_slack_extra()
        hints: Dict[str, bool] = {}
        for knob in ("unfurl_links", "unfurl_media"):
            val = extra.get(knob)
            if isinstance(val, bool):
                hints[knob] = val
            elif isinstance(val, str) and val.strip().lower() in (_TRUTHY | _FALSY):
                hints[knob] = val.strip().lower() in _TRUTHY
        return hints or None

    def _stamp_slack_unfurl(self, platform: Optional[str], metadata: Dict[str, Any]) -> None:
        unfurl = self._slack_unfurl_hints(platform)
        if unfurl:
            metadata.update(unfurl)

    def _stamp_slack_session_thread(self, event) -> None:
        """Native session-keying parity for fronted Slack DMs.

        Native Slack stamps ``thread_ts = event.thread_ts or ts``, so each TOP-LEVEL
        message keys a FRESH session (parallel turns). The connector normalizes
        top-level messages with thread_id=null, so without this every top-level DM
        collapsed into ONE session and message 2 pre-empted message 1. Only in
        thread-per-message mode (flat mode keeps the shared rolling session on
        purpose); never overwrites a real thread_id.
        """
        try:
            src = getattr(event, "source", None)
            if not src:
                return
            platform = getattr(src, "platform", None)
            if getattr(platform, "value", platform) != _SLACK:
                return
            if getattr(src, "thread_id", None):
                return  # real thread — its session key is already correct
            message_id, _chat = _event_ids(event)
            if not message_id or not self._effective_reply_in_thread():
                return
            if not self._dm_top_level_threads_as_sessions():
                return  # opt-out: threaded replies, one rolling session
            src.thread_id = str(message_id)
        except Exception:  # noqa: BLE001 - session stamping must never break inbound
            logger.debug("slack session-thread stamp failed", exc_info=True)

    async def _localize_inbound_media(self, event) -> None:
        """Download connector re-hosted attachments to local temp paths — every NATIVE
        adapter presents inbound media as LOCAL FILE PATHS (vision/file tools consume
        paths). Best-effort per entry: a failed download drops that entry, never the
        message; with no client only re-host URLs are dropped (they'd 401 downstream)."""
        try:
            urls = list(getattr(event, "media_urls", None) or [])
            if not urls:
                return
            # media_types is INDEXED IN PARALLEL with media_urls by every downstream
            # classifier: carry (url, mime) PAIRS through the loop or surviving
            # attachments inherit a neighbour's type.
            types = list(getattr(event, "media_types", None) or [])
            pairs = [(u, types[i] if i < len(types) else "") for i, u in enumerate(urls)]
            client = self._get_media_client()
            localized: list[tuple[str, str]] = []
            for url, mime in pairs:
                if not isinstance(url, str) or not url:
                    continue
                if client is None:
                    if "/relay/media/" not in url:
                        localized.append((url, mime))
                    continue
                path = await client.download(url)
                if path:
                    localized.append((path, mime))
                elif "/relay/media/" not in url:
                    # A public URL still has value as a URL; a dead re-host does not.
                    localized.append((url, mime))
            event.media_urls = [u for u, _ in localized]
            event.media_types = [m for _, m in localized]
        except Exception:  # noqa: BLE001 - media localization must never break inbound
            logger.debug("relay inbound media localization failed", exc_info=True)

    def prime_routing_cache(self, event) -> None:
        """Warm the per-chat egress routing caches from a SYNTHETIC event: a completion
        turn injected right after a restart (durable async-delegation replay) reaches
        handle_message with the caches COLD, so its replies egress without
        scope_id/user_id and the connector's fail-closed tenant guard declines them."""
        if event is None or getattr(event, "source", None) is None:
            return
        self._capture_scope(event)

    def _capture_scope(self, event) -> None:
        """Remember a chat's egress discriminators from an inbound event. Never raises.

        scope_id: scoped (guild/channel) message → routing-table resolution. user_id:
        authentic author id, captured for EVERY message — sole discriminator for a DM
        AND the author-first fallback for a scoped reply whose guild has no route row
        (managed agents join guilds dynamically). Without a resolvable discriminator
        the connector declines egress as 'target not routed to an onboarded tenant'.
        """
        try:
            src = getattr(event, "source", None)
            chat = getattr(src, "chat_id", None) if src else None
            if not chat:
                return
            chat = str(chat)
            # Underlying platform's string VALUE, skipping the generic RELAY
            # fallback (the connector's session default handles egress then).
            platform = getattr(src, "platform", None)
            platform_value = getattr(platform, "value", platform)
            if platform_value and platform_value != "relay":
                self._platform_by_chat[chat] = str(platform_value)
            for attr, cache in (
                ("user_id", self._dm_user_by_chat), ("scope_id", self._scope_by_chat),
                ("chat_type", self._chat_type_by_chat),
            ):
                value = getattr(src, attr, None)
                if value:
                    cache[chat] = str(value)
            # Triggering message ts for the typing/status lane's synthetic thread anchor.
            message_id, _chat = _event_ids(event)
            if message_id:
                self._last_inbound_ts_by_chat[chat] = str(message_id)
        except Exception:  # noqa: BLE001 - scope tracking must never break inbound
            pass

    def _with_scope(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Outbound metadata carrying the tenant discriminators (see _capture_scope).
        Both are attached when known and not already set; the connector tries scope_id
        first and only falls back to user_id on a route miss, so carrying both never
        overrides routing-table resolution."""
        meta: Dict[str, Any] = dict(metadata or {})
        for key, cache in (("scope_id", self._scope_by_chat), ("user_id", self._dm_user_by_chat)):
            if not meta.get(key):
                value = cache.get(str(chat_id))
                if value:
                    meta[key] = value
        return meta

    def fronts_platform(self, platform: Any) -> bool:
        """Whether the authenticated relay transport advertises ``platform`` — a
        restart-safe ownership signal from the identity set sent at handshake, not
        from an inbound chat cache."""
        platform_value = getattr(platform, "value", platform)
        ids = getattr(self._transport, "_identities", None) or ()
        return bool(platform_value) and any(p == str(platform_value) for p, _ in ids)

    def supports_inchannel_continuable_for_platform(self, platform: Any) -> bool:
        """Whether ONE fronted platform can host the flat continuable cron surface (D6
        gate). The scalar bit is the PRIMARY's only, so resolve the platform's own
        negotiated descriptor; fall back to the scalar when unavailable."""
        per_platform = self._negotiated_descriptor(str(getattr(platform, "value", platform) or ""))
        if per_platform is not None:
            return bool(getattr(per_platform, "supports_inchannel_continuable", False))
        return bool(self.supports_inchannel_continuable)

    async def on_interrupt(self, session_key: str, chat_id: str) -> None:
        """Bridge a connector-delivered /stop into the per-session interrupt path."""
        await self.interrupt_session_activity(session_key, chat_id)

    async def _on_passthrough(self, forward, buffer_id: Optional[str] = None) -> None:
        """Handle a connector-forwarded passthrough request. The connector answered the
        provider's latency-critical ACK at the edge, verified the signature and vaulted
        any shared-identity credential; the agent later acts via the token-less
        ``send_follow_up`` path. A Discord interaction becomes a normalized
        ``MessageEvent`` on the SAME agent path as chat; other forwards are logged and
        dropped. NEVER raises: a malformed forward must not kill the read loop."""
        try:
            platform = getattr(forward, "platform", "") or ""
            if platform == "discord":
                event = self._discord_interaction_to_event(forward)
                if event is not None:
                    self._capture_scope(event)
                    # A prompt-token component press is consumed (same gate as _on_inbound).
                    if await self._consume_prompt_response(event):
                        return
                    await self.handle_message(event)
                    return
            logger.info(
                "relay passthrough_forward dropped (no handler): platform=%s method=%s path=%s",
                platform, getattr(forward, "method", "?"), getattr(forward, "path", "?"),
            )
        except Exception:  # noqa: BLE001 - a bad forward must never break the reader
            logger.warning("relay passthrough_forward handling failed", exc_info=True)

    def _discord_interaction_to_event(self, forward):
        """Convert a forwarded Discord interaction body to a MessageEvent, or None for
        an unusable body (a PING is answered at the edge and never forwarded). The
        session source mirrors the connector's ``interactionSessionSource`` so the
        session key matches the one the follow-up capability was bound under."""
        try:
            payload = json.loads(bytes(getattr(forward, "body", b"")).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        # type 2 = APPLICATION_COMMAND; 3 = MESSAGE_COMPONENT; 5 = MODAL_SUBMIT.
        itype = payload.get("type")
        data = payload.get("data") or {}
        message_type = MessageType.TEXT
        if itype == 2:
            # Normalize to a leading-slash command string ("/name arg…"), the
            # shape the dispatcher and the connector's Slack slash lane expect.
            text = ("/" + str(data.get("name") or "")).rstrip("/") or ""
            if text:
                parts = [text] + self._render_interaction_options(data.get("options"))
                text = " ".join(parts).strip()
                message_type = MessageType.COMMAND
        elif itype == 3:
            text = str(data.get("custom_id") or "")
        else:
            text = ""
        member = payload.get("member") or {}
        user = (member.get("user") if isinstance(member, dict) else None) or payload.get("user") or {}
        if not isinstance(user, dict):
            user = {}
        guild_id = payload.get("guild_id")
        source = SessionSource(
            # The LOGICAL platform, not RELAY: session keys must match the connector's
            # capability binding (platform="discord"), /sethome must file under the
            # logical platform, and _capture_scope skips the generic "relay".
            platform=Platform.DISCORD,
            chat_id=str(payload.get("channel_id") or ""),
            # "group", not "channel": both the connector's capability binding and the
            # native Discord adapter key guild channels as "group".
            chat_type="group" if guild_id else "dm",
            user_id=str(user["id"]) if user.get("id") else None,
            user_name=str(user["username"]) if user.get("username") else None,
            scope_id=str(guild_id) if guild_id else None,
            message_id=str(payload.get("id")) if payload.get("id") else None,
            # Same upstream-trust marker the relay text lane stamps. Set locally, never
            # read off the wire (engages /sethome's via_relay guard).
            delivered_via_upstream_relay=True,
            # Profile routing (multiplex mode), mirroring _event_from_wire.
            # The HERMES profile this interaction is routed to (multiplex mode) — mirrors _event_from_wire's
            # profile stamping for plain relayed messages (#60586). Without this, a Team-Gateway's Discord
            # slash-command/button/modal always fell back to the legacy agent:main namespace even when the
            # connector resolved a specific profile for it.
            profile=getattr(forward, "profile", None),
        )
        event = MessageEvent(text=text, message_type=message_type, source=source)
        if itype == 3:
            # A component press whose custom_id is a Hermes prompt token
            # (hp1:<prompt_id>:<option_id>) becomes a STRUCTURED prompt answer;
            # foreign custom_ids keep the best-effort TEXT shape.
            decoded = self._decode_prompt_token(text)
            if decoded:
                prompt_id, option_id = decoded
                msg = payload.get("message") or {}
                prompt_message_id = str(msg["id"]) if isinstance(msg, dict) and msg.get("id") else None
                event.prompt_response = {
                    "prompt_id": prompt_id,
                    "option_id": option_id,
                    "prompt_message_id": prompt_message_id,
                }
                event.text = f"/{option_id}"
                event.message_type = MessageType.COMMAND
        return event

    @staticmethod
    def _decode_prompt_token(token: str):
        """Decode an hp1:<prompt_id>:<option_id> callback token, or None (mirrors the connector's promptCodec)."""
        parts = (token or "").split(":")
        if len(parts) != 3 or parts[0] != "hp1":
            return None
        if not _PROMPT_ID_RE.match(parts[1]) or not _PROMPT_ID_RE.match(parts[2]):
            return None
        return parts[1], parts[2]

    @staticmethod
    def _render_interaction_options(options) -> list:
        """Render Discord interaction options to text parts: scalars contribute their
        value (native ``f"/model {name}"`` shape); SUB_COMMAND (1) / SUB_COMMAND_GROUP
        (2) contribute their name then recurse into nested options."""
        parts: list = []
        if not isinstance(options, list):
            return parts
        for opt in options:
            if not isinstance(opt, dict):
                continue
            if opt.get("type") in (1, 2):
                sub_name = str(opt.get("name") or "").strip()
                if sub_name:
                    parts.append(sub_name)
                parts.extend(RelayAdapter._render_interaction_options(opt.get("options")))
            else:
                value = opt.get("value")
                if value is not None and str(value).strip():
                    parts.append(str(value).strip())
        return parts

    async def disconnect(self) -> None:
        # The runner wraps this call in wait_for(adapter disconnect budget). Monitor
        # teardown and go_idle eat into the transport's drain time, so measure from
        # the top and thread the REMAINDER down — otherwise teardown is cancelled
        # mid-drain and the transport's fail-pending loop is skipped (callers then
        # block on _OUTBOUND_TIMEOUT_S).
        from gateway.relay.ws_transport import _env_disconnect_budget_s
        _started = time.monotonic()
        _budget = _env_disconnect_budget_s()
        # Stop the revocation monitor first so it can't fire a spurious fatal
        # during/after a deliberate teardown.
        if self._revocation_monitor is not None:
            self._revocation_monitor.cancel()
            try:
                await asyncio.wait_for(
                    self._revocation_monitor, timeout=_RELAY_REVOCATION_MONITOR_TEARDOWN_TIMEOUT_S
                )
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001 - best-effort teardown
                pass
            self._revocation_monitor = None
        if self._transport is not None:
            # Ask the connector to flip this instance to buffered-only BEFORE tearing
            # down the socket, so inbound arriving while asleep buffers durably and
            # replays on reconnect. Best-effort: a transport without go_idle (the
            # stub) or a failed ack must not block shutdown. transport.disconnect()
            # runs in finally so an outer cancellation during go_idle still closes the
            # socket/supervisor; shield() keeps the teardown await itself from being
            # cancelled mid-flight.
            try:
                go_idle = getattr(self._transport, "go_idle", None)
                if callable(go_idle):
                    try:
                        result: Any = go_idle(timeout_s=_RELAY_GO_IDLE_ON_DISCONNECT_TIMEOUT_S)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:  # noqa: BLE001 - going-idle is an optimization, never blocks drain
                        logger.debug("relay going_idle failed during drain", exc_info=True)
            finally:
                try:
                    _remaining = max(0.0, _budget - (time.monotonic() - _started))
                    try:
                        _td = self._transport.disconnect(budget_s=_remaining)  # type: ignore[call-arg]
                    except TypeError:
                        # Transports without the budget_s keyword (stubs).
                        _td = self._transport.disconnect()
                    await asyncio.shield(_td)
                except Exception:  # noqa: BLE001 - teardown must not block outer cancel propagation
                    logger.debug("relay transport disconnect failed during drain", exc_info=True)

    async def go_dormant(self) -> bool:
        """Quiesce the relay for a scale-to-zero suspend. Unlike ``disconnect()`` this
        keeps the reconnect path armed so the gateway re-dials and drains its backlog
        on wake. A transport without ``go_dormant`` (the stub) is a no-op returning
        False. Deliberately does NOT stop the revocation monitor — dormancy is not a
        teardown."""
        go_dormant = getattr(self._transport, "go_dormant", None)
        if not callable(go_dormant):
            return False
        try:
            result: Any = go_dormant()
            return bool(await result) if asyncio.iscoroutine(result) else bool(result)
        except Exception:  # noqa: BLE001 - dormancy is best-effort, never blocks the idle path
            logger.debug("relay go_dormant failed", exc_info=True)
            return False

    def hold_redial(self) -> bool:
        """Park the transport's reconnect supervisor. False if it did not take.

        The caller suspends on the strength of this, so a swallowed failure here
        would report protection that was never installed.
        """
        return self._toggle_transport_redial("hold_redial")

    def release_redial(self) -> bool:
        """Undo hold_redial() when the suspend did not land."""
        return self._toggle_transport_redial("release_redial")

    def _toggle_transport_redial(self, method_name: str) -> bool:
        # Still never raises into the idle path, but a stub transport or a
        # throwing toggle now reports False instead of passing for success.
        method = getattr(self._transport, method_name, None)
        if not callable(method):
            return False
        try:
            method()
        except Exception:  # noqa: BLE001 - never blocks the idle path
            logger.debug("relay %s failed", method_name, exc_info=True)
            return False
        return True

    async def send_for_platform(
        self,
        logical_platform: Any,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send to an explicitly advertised logical platform over Relay. Scheduled and
        persisted-home deliveries have no fresh inbound event to populate
        ``_platform_by_chat``. The delivery resolver calls this only after
        ``fronts_platform`` succeeds; repeated here fail-closed."""
        platform_value = str(getattr(logical_platform, "value", logical_platform))
        if not self.fronts_platform(platform_value):
            return SendResult(success=False, error=f"relay does not front platform {platform_value}")
        _sfp_metadata = dict(metadata or {})
        # Gateway-internal interim marker (see send()): strip before the wire.
        _interim = bool(_sfp_metadata.pop("_interim_send", False))
        # The delivery resolver calls THIS method directly, bypassing send() — an
        # open native stream must absorb the turn-final here too.
        seal = await self._absorb_into_open_draft(chat_id, content, _sfp_metadata, _interim)
        if seal is not None:
            return seal
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        self._stamp_slack_unfurl(platform_value, _sfp_metadata)
        result = await self._transport.send_outbound(
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                # format_hints on the explicit-platform lane too: the cron brief
                # must render blocks exactly like an interactive send.
                "metadata": self._with_scope(
                    chat_id, self._with_format_hints_for_platform(platform_value, _sfp_metadata)
                ),
            },
            platform=platform_value,
        )
        return _send_result(result, raw_response=result)

    def _format_hints(
        self, descriptor: Optional[CapabilityDescriptor], platform: Optional[str]
    ) -> Optional[Dict[str, bool]]:
        """Block-formatting hints for one outbound text frame, or None. The CONNECTOR
        owns the platform API call, so the gateway only signals intent: stamped ONLY
        when (a) the DESTINATION platform's negotiated descriptor advertises
        ``supports_block_formatting`` (an old connector never receives dead metadata)
        and (b) the operator enabled ``platforms.relay.extra.<platform>.rich_blocks`` /
        ``markdown_blocks`` (both default OFF, ``_coerce_flag`` semantics).
        ``descriptor``/``platform`` are the DESTINATION's, never the scalar primary:
        gating on the primary both leaked hints onto platforms that never advertised
        the bit and suppressed them for ones that did."""
        if descriptor is None or not getattr(descriptor, "supports_block_formatting", False):
            return None
        try:
            knob_src = self._relay_platform_extra(str(platform or "").lower())
        except Exception:  # noqa: BLE001 - config shape is operator-owned
            return None
        hints = {knob: True for knob in ("rich_blocks", "markdown_blocks") if self._coerce_flag(knob_src.get(knob), False)}
        return hints or None

    def _stamp_format_hints(
        self,
        descriptor: Optional[CapabilityDescriptor],
        platform: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """``metadata`` with ``format_hints`` for the DESTINATION (descriptor, platform) stamped, if any."""
        hints = self._format_hints(descriptor, platform)
        if not hints:
            return metadata
        merged = dict(metadata or {})
        merged.setdefault("format_hints", hints)
        return merged

    def _with_format_hints_for_chat(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Hints for a chat-addressed send (chat's platform as seen inbound, else the primary)."""
        return self._stamp_format_hints(self._descriptor_for_chat(chat_id), self._chat_platform(chat_id), metadata)

    def _with_format_hints_for_platform(
        self, platform_value: str, metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Hints for an explicit-platform send (scheduled/persisted-home lane). Falls
        back to the scalar descriptor only when it IS that platform's — never stamp
        from another platform's capability bit."""
        descriptor = self._negotiated_descriptor(str(platform_value))
        if descriptor is None and self.descriptor.platform == str(platform_value):
            descriptor = self.descriptor
        return self._stamp_format_hints(descriptor, str(platform_value), metadata)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        send_metadata = dict(metadata or {})
        explicit_platform = send_metadata.pop("_relay_logical_platform", None)
        # Consumer-declared interim send (commentary, tail flush): NOT the turn-final,
        # so it must never trigger seal-interception (sealing the live stream with
        # interim text orphans the true final into a plain duplicate).
        # Gateway-internal marker; strip before the wire.
        _interim = bool(send_metadata.pop("_interim_send", False))
        # Seal-interception is checked BEFORE the explicit-platform branch: an open
        # stream absorbs the turn-final whichever door it arrives through.
        seal = await self._absorb_into_open_draft(chat_id, content, send_metadata, _interim)
        if seal is not None:
            return seal
        if explicit_platform:
            return await self.send_for_platform(
                explicit_platform, chat_id, content, reply_to=reply_to, metadata=send_metadata or None
            )
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        effective_reply_to = self._prepare_slack_egress(chat_id, reply_to, send_metadata)
        result = await self._outbound(
            chat_id,
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": effective_reply_to,
                "metadata": self._text_metadata(chat_id, send_metadata),
            },
        )
        # Auto-thread routing feedback: when the connector's auto-thread policy routed
        # this send into a thread it just created, the result carries thread_id (+
        # initial name). The conversation was keyed on the PARENT channel, so this is
        # the only place the gateway learns where the reply landed.
        try:
            _at_thread = result.get("thread_id")
            _at_name = result.get("auto_thread_name")
            if _at_thread and _at_name:
                self._auto_thread_by_chat[str(chat_id)] = (str(_at_thread), str(_at_name))
                if len(self._auto_thread_by_chat) > 256:
                    self._auto_thread_by_chat.pop(next(iter(self._auto_thread_by_chat)), None)
        except Exception:  # noqa: BLE001 - feedback capture must never break send
            pass
        # Wake the rename lane on EVERY send into this chat: "nowhere new" is an
        # answer it should get now rather than by outlasting a timeout.
        waiter = self._auto_thread_waiters.get(str(chat_id))
        if waiter is not None:
            waiter.set()
        return _send_result(result)

    def auto_thread_info_for_chat(self, chat_id: str) -> Optional[Tuple[str, str]]:
        """(thread_id, initial_name) of the connector-created auto-thread for the most
        recent send into *chat_id*, if any (semantic thread-rename lane)."""
        return self._auto_thread_by_chat.get(str(chat_id))

    async def wait_for_auto_thread_info(self, chat_id: str, timeout: float) -> Optional[Tuple[str, str]]:
        """``auto_thread_info_for_chat``, but willing to wait for the send. The rename
        lane asks as soon as the session is titled — a whole turn early. Waits for the
        next send into this chat, so a reply the connector didn't auto-thread reports
        its miss immediately; *timeout* is only a backstop for a turn that never sends."""
        info = self.auto_thread_info_for_chat(chat_id)
        if info is not None:
            return info
        key = str(chat_id)
        waiter = self._auto_thread_waiters.get(key)
        if waiter is None:
            waiter = asyncio.Event()
            self._auto_thread_waiters[key] = waiter
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            # Only the waiter we installed, and only if no later call replaced it; a
            # fired event must not make the next turn's wait return instantly.
            if self._auto_thread_waiters.get(key) is waiter:
                self._auto_thread_waiters.pop(key, None)
        return self.auto_thread_info_for_chat(chat_id)

    def _resolve_reply_to_for_send(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Suppress the synthetic-DM thread anchor for a Slack DM reply.

        The stream consumer sends a DM reply with ``reply_to`` = the triggering ts
        (its edit anchor); the connector maps a raw reply_to to a Slack thread_ts, so
        the reply would thread under the user's message and lose progressive edit
        streaming. Native ``_resolve_thread_ts`` drops that anchor only when
        ``reply_in_thread`` is off; mirror it: Slack DM + no real
        ``thread_id``/``thread_ts`` + flat mode ⇒ drop. In thread-per-message mode the
        triggering ts IS the thread anchor and the final reply's ONLY threading signal
        (dropping it unconditionally exiled finals to the DM root while progress
        stayed threaded). Removes an anchor, never adds one.

        It does NOT: * regress real-thread streaming — a real thread carries a distinct ``thread_id`` in
        metadata, so the guard leaves ``reply_to`` alone; * regress channel autoThread — a channel/group
        top-level reply carries ``thread_id`` (the message's own ts) in metadata when threading is on, so it
        is left alone; and a non-DM chat is never matched here. See #18859.
        """
        if reply_to is None:
            return None
        md = metadata or {}
        if (
            self._platform_by_chat.get(str(chat_id)) != _SLACK
            or self._chat_type_by_chat.get(str(chat_id)) != "dm"
            or md.get("thread_id")
            or md.get("thread_ts")
        ):
            return reply_to
        return reply_to if self._effective_reply_in_thread() else None

    def _apply_slack_thread_anchor(
        self,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Dict[str, Any],
        *,
        mirror_key: str = "reply_to_message_id",
    ) -> Optional[str]:
        """Resolve the outbound Slack thread anchor for ONE egress frame — the single
        choke point for text (``send``) and media (``_send_media``): (1) mode gate via
        ``_resolve_reply_to_for_send``; (2) when the anchor is dropped, strip the
        mirrored ``metadata.reply_to_message_id`` too, or the connector threads on it;
        (3) the connector's Slack sender THREADS ON METADATA ONLY (``threadTs()`` never
        reads the frame's ``reply_to``), so a surviving anchor is promoted into
        ``metadata.thread_id``. ``metadata`` is mutated in place."""
        effective_reply_to = self._resolve_reply_to_for_send(chat_id, reply_to, metadata)
        if effective_reply_to is None and reply_to is not None:
            metadata.pop(mirror_key, None)
        if (
            effective_reply_to is not None
            and self._platform_by_chat.get(str(chat_id)) == _SLACK
            and not (metadata.get("thread_id") or metadata.get("thread_ts"))
        ):
            metadata["thread_id"] = str(effective_reply_to)
        return effective_reply_to

    def _prepare_slack_egress(
        self, chat_id: str, reply_to: Optional[str], metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Text/media egress prep: Slack thread anchor (DM replies post flat at the DM
        root, native _resolve_thread_ts parity) + unfurl hints; mutates ``metadata``."""
        effective_reply_to = self._apply_slack_thread_anchor(chat_id, reply_to, metadata)
        self._stamp_slack_unfurl(self._chat_platform(chat_id), metadata)
        return effective_reply_to

    def _with_status_thread_anchor(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Copy ``metadata`` with the typing/status thread anchor applied. Slack's
        status line is THREAD-scoped and the typing lane's metadata carries no anchor
        for a top-level DM, so synthesize it from the per-chat inbound-ts cache. Shared
        by ``send_typing`` and ``stop_typing`` — the clear MUST target the same thread
        the heartbeat set or the status sticks until Slack's timeout."""
        md = dict(metadata or {})
        if (
            not (md.get("thread_id") or md.get("thread_ts"))
            and self._platform_by_chat.get(str(chat_id)) == _SLACK
            and self._chat_type_by_chat.get(str(chat_id)) == "dm"
        ):
            anchor = self._last_inbound_ts_by_chat.get(str(chat_id))
            if anchor:
                md["thread_id"] = anchor
        return md

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a relayed message through the connector-owned platform API."""
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        result = await self._outbound(
            chat_id,
            {
                "op": "edit",
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": self._text_metadata(chat_id, metadata),
            },
        )
        return SendResult(
            success=bool(result.get("success")), message_id=result.get("message_id") or message_id,
            error=result.get("error"),
        )

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a relayed message (the stream consumer's fresh-final cleanup). Gated
        on the descriptor advertising ``delete``: older connectors return False so
        cleanup degrades to leaving the preview in place."""
        if self._transport is None:
            return False
        if "delete" not in (self._descriptor_for_chat(str(chat_id)).supported_ops or ()):
            return False
        try:
            result = await self._outbound(
                chat_id,
                {
                    "op": "delete",
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "metadata": self._with_scope(chat_id, {}),
                },
            )
        except Exception:
            logger.debug("relay delete_message failed", exc_info=True)
            return False
        return bool(result.get("success"))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Egress a typing indicator (the base ``_keep_typing`` tick) as the ``typing``
        op. Best-effort and one-shot: Discord/Telegram indicators self-expire; Slack
        Assistant status persists, so ``stop_typing`` sends an explicit clear for
        Slack only."""
        if self._transport is None:
            return
        # Rich status parity: carry run.py's per-tool phrase as the frame's content
        # (rendered on assistant.threads.setStatus). Absent => omit content and the
        # connector uses its default heartbeat. NEVER send empty-string content here:
        # on Slack that is the CLEAR request.
        phrase = getattr(self, "_status_text", {}).get(str(chat_id))
        await self._typing_frame(chat_id, metadata, str(phrase) if phrase else None, "send_typing")

    async def stop_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Forward an explicit typing/status clear (empty ``content``) — Slack only:
        other relay senders have one-shot heartbeats, where an empty heartbeat would
        re-trigger typing at completion. A connector older than gateway-gateway #154
        hardcodes the typing status and would SET it on a clear frame — deploy the
        connector first."""
        if self._transport is None or self._platform_by_chat.get(str(chat_id)) != _SLACK:
            return
        await self._typing_frame(chat_id, metadata, "", "stop_typing")

    async def _typing_frame(
        self, chat_id: str, metadata: Optional[Dict[str, Any]], content: Optional[str], lane: str
    ) -> None:
        """One ``typing`` frame (``content`` None = omit; "" = Slack clear). Cosmetic: never raises."""
        # Thread anchor for the status surface. Slack's status line ("is thinking…" in the thread's replies
        # footer — works with plain chat:write, confirmed on native no-assistant bots) is THREAD-only: the
        # connector's typing case no-ops without a thread_ts. But the typing lane's metadata (base.py
        # _thread_metadata_for_source) has no anchor for a top-level DM — source.thread_id is None — so
        # every heartbeat was silently dropped. In thread-per-message mode the turn's thread root IS the
        # triggering message ts (run.py's synthetic root); synthesize it here from the per-chat inbound
        # cache, exactly like native send_typing resolves thread_ts from metadata.message_id. Flat mode
        # (reply_in_thread=false) keeps the no-anchor no-op: there is no thread and must not be one
        # (#18859).
        md = self._with_status_thread_anchor(chat_id, metadata)
        frame: Dict[str, Any] = {"op": "typing", "chat_id": chat_id, "metadata": self._with_scope(chat_id, md)}
        if content is not None:
            frame["content"] = content
        try:
            await self._outbound(chat_id, frame)
        except Exception:  # noqa: BLE001 - typing/status is cosmetic, never breaks a turn
            logger.debug("relay %s failed for %s", lane, chat_id, exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Op-gated so a legacy connector (which would only answer "unsupported op")
        # gets the same local fallback without a round trip.
        if self._transport is None or not self.descriptor.supports_op("get_chat_info"):
            return {"name": chat_id, "type": "dm"}
        return await self._transport.get_chat_info(chat_id)

    async def send_follow_up(
        self, session_key: str, kind: str, content: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send via a shared-identity capability bound to a session. The gateway never
        holds the credential: it names the session and the capability ``kind``; the
        connector resolves the value from its vault and egresses (enforcing tenant)."""
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        # `kind` is platform-prefixed ("discord.interaction_token"): tag the frame
        # with that platform when we front it; otherwise the connector's session
        # default routes it.
        prefix = kind.split(".", 1)[0] if kind and "." in kind else None
        follow_up_platform = prefix if prefix and self.fronts_platform(prefix) else None
        result = await self._transport.send_follow_up(
            {
                "op": "follow_up",
                "session_key": session_key,
                "kind": kind,
                "content": content,
                "metadata": metadata or {},
            },
            platform=follow_up_platform,
        )
        return _send_result(result)

    # ── Phase 2 media ─────────────────────────────────────────────────────

    def _get_media_client(self) -> Optional[RelayMediaClient]:
        """Lazily build the authenticated /relay/media client from the SAME dial URL and
        per-gateway creds the WS uses; None when unavailable (media lanes then degrade
        to their pre-media fallbacks)."""
        if self._media_client is not None:
            return self._media_client
        try:
            from gateway.relay import relay_connection_auth, relay_url
            from gateway.relay.media import media_base_url

            url = relay_url()
            gateway_id, secret = relay_connection_auth()
            if not url:
                return None
            client = RelayMediaClient(media_base_url(url), gateway_id, secret)
            if not client.enabled:
                return None
            self._media_client = client
            return client
        except Exception:  # noqa: BLE001 - media plumbing must never break the adapter
            logger.debug("relay media client init failed", exc_info=True)
            return None

    async def _send_media(
        self,
        chat_id: str,
        *,
        media_kind: str,
        source: str,
        source_is_path: bool,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """Egress one media object via the connector's ``send_media`` op. ``source`` is
        a LOCAL path (uploaded to /relay/media first — the connector cannot reach our
        filesystem) or an already-public URL (passed through). None when the lane is
        unavailable (op not advertised, no transport, upload failed, connector
        decline) so each caller falls back to its pre-media behaviour."""
        if self._transport is None or not self.descriptor.supports_op("send_media"):
            return None
        source_url = source
        if source_is_path:
            client = self._get_media_client()
            if client is None:
                return None
            uploaded = await client.upload(source, filename=filename)
            if not uploaded:
                return None
            source_url = uploaded
        # Same Slack thread-anchor contract as the text lane: media frames go through
        # the connector's Slack sender too (threadTs() reads metadata only).
        media_metadata: Dict[str, Any] = dict(metadata or {})
        effective_reply_to = self._prepare_slack_egress(chat_id, reply_to, media_metadata)
        action: Dict[str, Any] = {
            "op": "send_media",
            "chat_id": chat_id,
            "media_kind": media_kind,
            "source_url": source_url,
            "content": caption or "",
            "reply_to": effective_reply_to,
            "metadata": self._with_scope(chat_id, media_metadata),
        }
        if filename:
            action["filename"] = filename
        # A structured connector decline (size cap, platform rejection) is logged;
        # the caller's fallback still delivers the caption/notice.
        result = await self._gated_op(chat_id, action)
        if result is None:
            return None
        return SendResult(success=True, message_id=result.get("message_id"), raw_response=result)

    # Each media override tries the native ``send_media`` lane, then falls back to
    # the base adapter's text/URL behaviour when the lane is unavailable.
    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        result = await self._send_media(
            chat_id, media_kind="image", source=image_url, source_is_path=False,
            caption=caption, reply_to=reply_to, metadata=metadata,
        )
        return result if result is not None else await super().send_image(
            chat_id, image_url, caption=caption, reply_to=reply_to, metadata=metadata
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        result = await self._send_media(
            chat_id, media_kind="image", source=image_path, source_is_path=True,
            caption=caption, reply_to=reply_to, metadata=metadata,
        )
        return result if result is not None else await super().send_image_file(
            chat_id, image_path, caption=caption, reply_to=reply_to, metadata=metadata, **kwargs
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        result = await self._send_media(
            chat_id, media_kind="voice", source=audio_path, source_is_path=True,
            caption=caption, reply_to=reply_to, metadata=metadata,
        )
        return result if result is not None else await super().send_voice(
            chat_id, audio_path, caption=caption, reply_to=reply_to, metadata=metadata, **kwargs
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        result = await self._send_media(
            chat_id, media_kind="video", source=video_path, source_is_path=True,
            caption=caption, reply_to=reply_to, metadata=metadata,
        )
        return result if result is not None else await super().send_video(
            chat_id, video_path, caption=caption, reply_to=reply_to, metadata=metadata, **kwargs
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        result = await self._send_media(
            chat_id, media_kind="document", source=file_path, source_is_path=True,
            caption=caption, filename=file_name, reply_to=reply_to, metadata=metadata,
        )
        return result if result is not None else await super().send_document(
            chat_id, file_path, caption=caption, file_name=file_name, reply_to=reply_to, metadata=metadata, **kwargs
        )

    # ── Phase 3 interactive: prompt + react ──────────────────────────────

    def _mint_prompt(
        self, kind: str, state: Dict[str, Any], timeout_s: float = 3600.0
    ) -> str:
        """Register a pending prompt and return its id (``<owner nonce>.<8 hex>``).
        Expiry is enforced gateway-side on consumption (_pop_prompt); the wire's
        timeout_s is advisory. The nonce marks the minting process so a sibling
        gateway receiving the fanned-out answer stays quiet. Both segments use the
        connector codec's alphabet ([A-Za-z0-9_.-], <=32)."""
        prompt_id = f"{self._prompt_owner_nonce}.{secrets.token_hex(4)}"
        self._pending_prompts[prompt_id] = {**state, "kind": kind, "expires_at": time.time() + timeout_s}
        # Opportunistic sweep so abandoned prompts can't accumulate.
        now = time.time()
        for stale in [k for k, v in self._pending_prompts.items() if v.get("expires_at", 0) < now]:
            self._pending_prompts.pop(stale, None)
        return prompt_id

    def _minted_here(self, prompt_id: str) -> bool:
        """True when this process minted ``prompt_id``. Ids without a ``.`` segment
        predate the owner nonce (in-flight across an in-place upgrade) and are ours."""
        head, sep, _ = str(prompt_id).partition(".")
        return head == self._prompt_owner_nonce if sep else True

    def _pop_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Consume a pending prompt: one answer wins, expired entries miss."""
        state = self._pending_prompts.pop(str(prompt_id), None)
        if not state or state.get("expires_at", 0) < time.time():
            return None
        return state

    def _note_prompt_resolved(self, prompt_id: str) -> None:
        """Remember that this process answered ``prompt_id`` (bounded FIFO: a repeat is
        only interesting while a redelivery/double tap can arrive)."""
        self._resolved_prompts[str(prompt_id)] = time.time()
        while len(self._resolved_prompts) > _RESOLVED_PROMPT_MEMORY:
            self._resolved_prompts.popitem(last=False)

    async def _send_prompt(
        self,
        chat_id: str,
        *,
        prompt_kind: str,
        text: str,
        prompt_id: str,
        options: list,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[int] = None,
    ) -> Optional[SendResult]:
        """Egress one `prompt` op; None when the lane is unavailable (the caller falls
        back to its numbered-text base behaviour). Prompt metadata is forwarded
        VERBATIM: the threading mode is decided in exactly one place — run.py's
        _resolve_progress_thread_id (flat mode suppresses the synthetic self-anchor
        there; thread mode stamps the turn's thread)."""
        action: Dict[str, Any] = {
            "op": "prompt",
            "chat_id": chat_id,
            "content": text,
            "prompt_kind": prompt_kind,
            "prompt_id": prompt_id,
            "options": options,
            "reply_to": self._resolve_reply_to_for_send(chat_id, reply_to, metadata),
            "metadata": self._with_scope(chat_id, metadata),
        }
        if timeout_s is not None:
            action["timeout_s"] = int(timeout_s)
        result = await self._gated_op(chat_id, action)
        if result is None:
            return None
        return SendResult(success=True, message_id=result.get("message_id"), raw_response=result)

    async def _mint_and_send_prompt(
        self,
        kind: str,
        state: Dict[str, Any],
        chat_id: str,
        *,
        prompt_kind: str,
        text: str,
        options: list,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[SendResult]:
        """Register + egress a prompt; unregisters and returns None when the lane is unavailable."""
        prompt_id = self._mint_prompt(kind, {**state, "chat_id": str(chat_id)})
        result = await self._send_prompt(
            chat_id, prompt_kind=prompt_kind, text=text, prompt_id=prompt_id, options=options,
            metadata=metadata,
        )
        if result is None:
            self._pending_prompts.pop(prompt_id, None)
        return result

    _PROMPT_UNAVAILABLE = SendResult(success=False, error="relay prompt op unavailable")

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
        tool_use_id: str = "",
    ) -> SendResult:
        """Native-button exec approval over the relay (same choice set as native; the
        press resolves via tools.approval.resolve_gateway_approval). When the lane is
        unavailable the send FAILS (success=False) so run.py's button→text fallback runs."""
        options: list = [{"id": "once", "label": "Allow Once", "style": "primary"}]
        if not smart_denied and allow_session:
            options.append({"id": "session", "label": "Allow Session"})
            if allow_permanent:
                options.append({"id": "always", "label": "Always Allow"})
        options.append({"id": "deny", "label": "Deny", "style": "danger"})

        cmd_preview = command if len(command) <= 1500 else command[:1500] + "..."
        text = f"⚠️ **Command Approval Required**\n\n```\n{cmd_preview}\n```\nReason: {description}"
        if smart_denied:
            text += "\n\n**Smart DENY:** owner override applies to this one operation only."
        result = await self._mint_and_send_prompt(
            "exec_approval", {"session_key": session_key}, chat_id, prompt_kind="approval",
            text=text, options=options, metadata=metadata,
        )
        return result if result is not None else self._PROMPT_UNAVAILABLE

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Three-button slash-command confirmation over the relay (resolves via
        tools.slash_confirm.resolve; success=False falls back to text-intercept)."""
        options = [
            {"id": "once", "label": "Approve Once", "style": "primary"},
            {"id": "always", "label": "Always Approve"},
            {"id": "cancel", "label": "Cancel", "style": "danger"},
        ]
        result = await self._mint_and_send_prompt(
            "slash_confirm", {"session_key": session_key, "confirm_id": confirm_id}, chat_id,
            prompt_kind="approval", text=f"**{title}**\n\n{message}" if title else message,
            options=options, metadata=metadata,
        )
        return result if result is not None else self._PROMPT_UNAVAILABLE

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Native-button clarify over the relay. A press resolves with the CHOICE TEXT
        (never the option id); "Other" flips to text-capture. Option ids are
        positional (c0..cN / other) — choice text is arbitrary UTF-8 and would blow
        the 64-byte callback budget. Open-ended clarifies and unavailable lanes fall
        back to base."""
        if choices and self.descriptor.supports_op("prompt"):
            options = [{"id": f"c{i}", "label": str(choice)[:75]} for i, choice in enumerate(choices)]
            options.append({"id": "other", "label": "✏️ Other (type your answer)"})
            result = await self._mint_and_send_prompt(
                "clarify",
                {
                    "session_key": session_key,
                    "clarify_id": clarify_id,
                    "choices": [str(c) for c in choices],
                },
                chat_id,
                prompt_kind="clarify",
                text=f"❓ {question}",
                options=options,
                metadata=metadata,
            )
            if result is not None:
                return result
        return await super().send_clarify(
            chat_id, question, choices, clarify_id, session_key, metadata=metadata
        )

    async def _consume_prompt_response(self, event) -> bool:
        """Route an inbound prompt_response to its waiting primitive; True when the
        event was a prompt answer (consumed — never dispatched as chat). EVERY prompt
        answer is consumed, whoever owns it: a sibling's prompt (the connector fans the
        press to every gateway of the tenant; falling through produced a wall of
        "Unknown command"), a repeat answer (first one won), or our own expired/unknown
        prompt (answered with a short expiry notice — option ids are not commands)."""
        pr = getattr(event, "prompt_response", None)
        if not isinstance(pr, dict):
            return False
        prompt_id = str(pr.get("prompt_id") or "")
        option_id = str(pr.get("option_id") or "")
        if not prompt_id or not option_id:
            return False
        if not self._minted_here(prompt_id):
            logger.debug(
                "relay prompt_response %s (option=%s) belongs to another gateway instance — ignoring",
                prompt_id,
                option_id,
            )
            return True
        if prompt_id in self._resolved_prompts:
            logger.debug(
                "relay prompt_response %s (option=%s) already resolved — ignoring repeat", prompt_id, option_id
            )
            return True
        state = self._pop_prompt(prompt_id)
        if state is None:
            logger.info("relay prompt_response for unknown/expired prompt %s (option=%s)", prompt_id, option_id)
            await self._notify_prompt_expired(event)
            return True
        self._note_prompt_resolved(prompt_id)

        kind = state.get("kind")
        chat_id = str(state.get("chat_id") or getattr(event.source, "chat_id", ""))
        handler = _PROMPT_RESOLVERS.get(kind)
        try:
            if handler is None:
                logger.warning("relay prompt_response with unknown kind %r", kind)
            else:
                # Acks are fire-and-forget: we are ON the read loop here (see
                # _send_lifecycle_ack) and awaiting a send would self-deadlock.
                await handler(self, state, option_id, chat_id, self._prompt_reply_metadata(event))
        except Exception:  # noqa: BLE001 - a resolver failure must not kill the reader
            logger.warning("relay prompt_response resolution failed", exc_info=True)
        return True

    async def _resolve_exec_approval(self, state, option_id, chat_id, ack_meta) -> None:
        from tools.approval import resolve_gateway_approval

        choice = option_id if option_id in _EXEC_APPROVAL_LABELS else "deny"
        count = resolve_gateway_approval(str(state.get("session_key") or ""), choice)
        label = _EXEC_APPROVAL_LABELS[choice] if count else "⌛ Approval expired — no command was waiting."
        # In-channel ack preserves the audit trail the native edit gives (the
        # connector's prompt message can't be edited cross-platform yet).
        self._send_lifecycle_ack(chat_id, label, ack_meta)
        if count:
            self.resume_typing_for_chat(chat_id)

    async def _resolve_slash_confirm(self, state, option_id, chat_id, ack_meta) -> None:
        from tools import slash_confirm as slash_confirm_mod

        choice = option_id if option_id in _SLASH_CONFIRM_LABELS else "cancel"
        result_text = await slash_confirm_mod.resolve(
            str(state.get("session_key") or ""), str(state.get("confirm_id") or ""), choice
        )
        self._send_lifecycle_ack(chat_id, _SLASH_CONFIRM_LABELS[choice], ack_meta)
        if result_text:
            self._send_lifecycle_ack(chat_id, str(result_text), ack_meta)

    async def _resolve_clarify(self, state, option_id, chat_id, ack_meta) -> None:
        from tools.clarify_gateway import mark_awaiting_text, resolve_gateway_clarify

        clarify_id = str(state.get("clarify_id") or "")
        if option_id == "other":
            mark_awaiting_text(clarify_id)
            self._send_lifecycle_ack(chat_id, "✏️ Type your answer:", ack_meta)
            return
        choices = state.get("choices") or []
        try:
            idx = int(option_id[1:]) if option_id.startswith("c") else -1
        except ValueError:
            idx = -1
        if 0 <= idx < len(choices):
            resolve_gateway_clarify(clarify_id, str(choices[idx]))
            self._send_lifecycle_ack(chat_id, f"✅ {choices[idx]}", ack_meta)
        else:
            # Unmappable option: flip to text capture (never dead-end a clarify).
            mark_awaiting_text(clarify_id)

    def _send_lifecycle_ack(self, chat_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Fire-and-forget a prompt-lifecycle ack from read-loop context.
        _consume_prompt_response executes ON the transport read loop; an ``await
        self.send(...)`` there is a SELF-DEADLOCK (send() blocks on an outbound_result
        future only the read loop can resolve) — every button tap wedged the transport
        for the full outbound timeout. Acks are cosmetic, so they ride a background
        task; failures log at debug. The task ref is retained (asyncio holds tasks weakly)."""

        async def _ack() -> None:
            try:
                await self.send(chat_id, text, metadata=metadata)
            except Exception:  # noqa: BLE001 - ack is best-effort
                logger.debug("relay lifecycle ack failed", exc_info=True)

        task = asyncio.create_task(_ack(), name="relay-lifecycle-ack")
        self._lifecycle_ack_tasks.add(task)
        task.add_done_callback(self._lifecycle_ack_tasks.discard)

    async def _notify_prompt_expired(self, event) -> None:
        """Tell the presser their prompt is no longer waiting (owning gateway only, best-effort)."""
        chat_id = str(getattr(event.source, "chat_id", "") or "")
        if not chat_id:
            return
        self._send_lifecycle_ack(
            chat_id,
            "⌛ That prompt is no longer waiting for an answer. "
            "Send your reply as a normal message.",
            self._prompt_reply_metadata(event),
        )

    def _prompt_reply_metadata(self, event) -> Dict[str, Any]:
        """Thread metadata so prompt acks land where the prompt lives. Marked INTERIM:
        acks fire while the approval turn's OWN draft stream is open and carry only
        placement metadata, so send()'s single-open-stream fallback sealed the live
        draft with the ack text (frozen stream + duplicate final on every approval turn)."""
        meta: Dict[str, Any] = {"_interim_send": True}
        thread_id = getattr(event.source, "thread_id", None)
        if thread_id:
            meta["thread_id"] = str(thread_id)
        return meta

    # ── Phase 3 ack lifecycle (👀 → ✅/❌) ────────────────────────────────

    async def _react(
        self,
        chat_id: str,
        message_id: str,
        emoji: str,
        *,
        remove: bool = False,
    ) -> bool:
        """Egress one `react` op; best-effort (False on any failure, logged at debug)."""
        if not chat_id or not message_id:
            return False
        result = await self._gated_op(
            chat_id,
            {
                "op": "react",
                "chat_id": chat_id,
                "message_id": message_id,
                "emoji": emoji,
                "remove": remove,
                "metadata": self._with_scope(chat_id, None),
            },
            decline_level=None,
        )
        return result is not None

    async def on_processing_start(self, event) -> None:
        """Add the 👀 in-progress reaction (op-gated; silent no-op otherwise)."""
        message_id, chat_id = _event_ids(event)
        if message_id and chat_id:
            await self._react(str(chat_id), str(message_id), "👀")

    async def on_processing_complete(self, event, outcome) -> None:
        """Swap 👀 for ✅/❌ per outcome (op-gated; silent no-op otherwise)."""
        message_id, chat_id = _event_ids(event)
        if not (message_id and chat_id):
            return
        await self._react(str(chat_id), str(message_id), "👀", remove=True)
        if outcome == ProcessingOutcome.SUCCESS:
            await self._react(str(chat_id), str(message_id), "✅")
        elif outcome == ProcessingOutcome.FAILURE:
            await self._react(str(chat_id), str(message_id), "❌")

    # ── Phase 4 thread lifecycle ──────────────────────────────────────────

    async def create_handoff_thread(
        self,
        parent_chat_id: str,
        name: str,
    ) -> Optional[str]:
        """Create a thread/topic under ``parent_chat_id`` via the connector. One
        `thread_create` op covers Discord (channel thread), Telegram (forum topic) and
        Slack (named seed root message). None on any failure/unavailability so the
        handoff watcher falls back to the parent."""
        result = await self._gated_op(
            str(parent_chat_id),
            {
                "op": "thread_create",
                "chat_id": str(parent_chat_id),
                "thread_name": (str(name or "").strip() or "handoff")[:100],
                "metadata": self._with_scope(str(parent_chat_id), None),
            },
            decline_level=logging.INFO,
            subject=parent_chat_id,
        )
        if result is None:
            return None
        thread_id = result.get("thread_id") or result.get("message_id")
        return str(thread_id) if thread_id else None

    async def rename_thread(
        self,
        thread_id: str,
        name: str,
        *,
        only_if_current_name: Optional[str] = None,
        prefer_connector_created: bool = False,
        parent_chat_id: Optional[str] = None,
    ) -> bool:
        """Best-effort thread rename via the connector's `thread_rename` op. Prefer
        ``prefer_connector_created=True``: the CONNECTOR enforces the no-clobber guard
        from its own created-name memory, so the gateway need not reproduce the
        initial name byte-for-byte (any normalization drift silently declined every
        rename). ``only_if_current_name`` is the legacy string guard for older
        connectors. ``parent_chat_id`` defaults to the thread id (Telegram needs the
        containing chat; Discord ignores it)."""
        cleaned = " ".join(str(name or "").split()).strip()
        if not cleaned or not thread_id:
            return False
        chat_id = str(parent_chat_id or thread_id)
        action: Dict[str, Any] = {
            "op": "thread_rename",
            "chat_id": chat_id,
            "message_id": str(thread_id),
            "thread_name": cleaned[:100],
            "metadata": self._with_scope(chat_id, None),
        }
        if prefer_connector_created:
            action["only_if_connector_created"] = True
        elif only_if_current_name is not None:
            action["only_if_current_name"] = str(only_if_current_name)
        result = await self._gated_op(
            chat_id,
            action,
            decline_level=logging.INFO,
            subject=thread_id,
            platform=self._platform_by_chat.get(chat_id) or self._platform_by_chat.get(str(thread_id)),
        )
        return result is not None


# prompt kind -> resolver (order-independent: kinds are distinct keys).
_PROMPT_RESOLVERS = {
    "exec_approval": RelayAdapter._resolve_exec_approval,
    "slash_confirm": RelayAdapter._resolve_slash_confirm,
    "clarify": RelayAdapter._resolve_clarify,
}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import cast  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
