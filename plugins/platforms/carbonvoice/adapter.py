"""Carbon Voice platform adapter for Hermes Agent.

Architecture:
    Hermes  <──Socket.IO (primary)──   api.carbonvoice.app
    Hermes  ── REST poll fallback ──>  /v3/messages/recent
    Hermes  ── POST /v3/messages/start ──>  outbound replies

This module is the thin orchestrator that wires together:

    parse        — payload-shape helpers (pure)
    api          — REST client (CarbonVoiceAPI)
    transport    — Socket.IO + polling lifecycle (Transport)
    state        — disk-persisted cursor (Cursor)
    dedupe       — in-memory seen-message TTL cache (SeenCache)
    reactions    — visual ack on inbound (ReactionService)
    channels     — chat_type ("dm"/"group") + participant roster cache
    audit        — allowlist gate + ignored-sender audit log

No public webhook is required — the adapter holds an outbound Socket.IO
connection and polls /v3/messages/recent as a fallback. Cursor state is
persisted to disk so messages received while Hermes was offline are
processed on the next startup.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource

from .api import CarbonVoiceAPI
from .audit import AllowlistGate, IgnoredSenderLog, default_ignored_log_path
from .permits import ApprovalStore, parse_admin_command
from .channels import ChannelCache
from .conversations import ConversationTracker
from .constants import (
    DEFAULT_APPROVE_REACTION_ID,
    DEFAULT_BASE_URL,
    DEFAULT_POLL_INTERVAL_MS,
    DEFAULT_REJECT_REACTION_ID,
    DEFAULT_REVISIT_MAX_AGE_S,
    DEFAULT_STUCK_MAX_AGE_S,
    DEFAULT_WS_RETRY_MAX_MS,
    MAX_MESSAGE_LENGTH,
    STUCK_RETRY_DELAY_S,
)
from .dedupe import SeenCache
from .gate import MentionGate
from .parse import (
    bot_has_reacted,
    extract_attachments,
    extract_channel_id,
    extract_creator_id,
    extract_message_id,
    extract_share_link_id,
    extract_transcript,
    first_str,
    message_age_seconds,
    now_iso,
    now_utc,
    reactors_for,
)
# extract_transcript is also re-exported via parse for the parent-text path
# (now handled by ConversationTracker, but the import here is kept so
# extract_transcript stays available for any future inline use).
from .reactions import ReactionService
from .state import Cursor, default_state_path
from .transport import Transport

logger = logging.getLogger(__name__)


class CarbonVoiceAdapter(BasePlatformAdapter):
    """Hermes ↔ Carbon Voice bridge over Socket.IO + REST polling fallback."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    # Cap on tracked one-tap approval prompts (prompt msg_id → creator_id).
    # A flood of unknown senders can't grow this unbounded; oldest evict.
    _MAX_PENDING_PROMPTS = 200

    # Carbon Voice has no in-place message edit API — a "reply" is always a
    # new message. Declaring this False (like Signal / Weixin / WeCom) tells
    # the core's stream consumer NOT to attempt progressive edits, so it uses
    # the send-once path instead of editing a streamed bubble. Without it the
    # consumer keeps an editable-message assumption that, when a send fails
    # (e.g. CV returns 502), leaves "final delivery" unconfirmed and the core
    # re-sends the same response once per queued follow-up — the observed
    # "same message multiple times" duplication.
    SUPPORTS_MESSAGE_EDITING = False

    # Voice-out integration with Hermes core's auto-TTS pipeline.
    #
    # When core generates a TTS audio for the agent's reply and ships it
    # via ``send_voice`` → ``/v5/messages/audio``, Carbon Voice runs
    # server-side STT and renders the resulting message as a voice-memo
    # bubble with the transcript inline. That means the spoken text IS
    # the visible text — sending the same content again as a text bubble
    # is pure duplication.
    #
    # ``voice_out_carries_text = True`` tells Hermes core (see
    # ``gateway/platforms/base.py``'s ``_tts_caption_delivered`` check)
    # to suppress the follow-up text send when auto-TTS succeeded.
    # Conceptually it's the CV analog of Telegram's caption field on
    # voice messages — different mechanism (STT vs caption), same UX
    # contract (one bubble, text + audio together).
    #
    # The general delivery contract that honors this flag (suppressing
    # the follow-up text send after a successful play_tts, with a
    # completeness guard so truncated speech never drops content) is
    # PR #32655. Until it lands, the attribute is ignored and voice-out
    # delivers the audio memo plus a duplicate text bubble — degraded
    # UX, never lost content.
    voice_out_carries_text = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("carbonvoice"))
        extra = config.extra or {}
        pat: str = config.token or extra.get("pat") or ""
        base_url: str = (extra.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        poll_interval_s: float = (
            float(extra.get("poll_interval_ms") or DEFAULT_POLL_INTERVAL_MS) / 1000.0
        )
        ws_retry_max_s: float = (
            float(extra.get("ws_retry_max_ms") or DEFAULT_WS_RETRY_MAX_MS) / 1000.0
        )
        sp = extra.get("state_path")
        state_path: Path = Path(sp).expanduser() if sp else default_state_path()
        ilp = extra.get("ignored_senders_log")
        ignored_log_path: Path = (
            Path(ilp).expanduser() if ilp else default_ignored_log_path()
        )

        self._pat = pat
        self._creator_id: Optional[str] = extra.get("creator_id") or None
        self._self_user_id: Optional[str] = None
        self._mark_read_enabled: bool = not bool(extra.get("disable_mark_read"))
        # Voice-out: when true, every inbound MessageEvent is marked
        # ``MessageType.VOICE`` so Hermes core's auto-TTS pipeline
        # (``base.py:3493``) converts the agent's text reply to audio
        # and ships it via :meth:`send_voice` → ``/v5/messages/audio``.
        # Requires ``voice.auto_tts: true`` and a TTS provider in
        # ``config.yaml`` to actually fire — without those, marking
        # VOICE is a no-op (the gate's other conditions still fail).
        # Default ``False`` to preserve text-out for existing
        # deployments that haven't opted in.
        self._voice_out: bool = bool(extra.get("voice_out"))
        # Inbound multimodal (PR 7): per-attachment byte cap. CV's S3
        # URLs can hand back arbitrarily large files, and Hermes core's
        # vision / document pipeline pays per token for image bytes and
        # extracted text — a 50MB PDF blowing through the size limit
        # crashes the agent's API call. Default 10 MB matches what
        # Claude / OpenAI vision recommend; operators can raise it for
        # specialized use cases via ``CARBONVOICE_MAX_ATTACHMENT_MB``.
        self._max_attachment_bytes: int = int(
            extra.get("max_attachment_mb") or 10
        ) * 1024 * 1024
        # Stuck-message cutoff: a message with no transcript holds the cursor
        # (gets retried) only while younger than this. Past it we assume the
        # transcript will never arrive (image-only / system / failed STT) and
        # advance past it, so a single permanently-empty message can't pin the
        # cursor and re-feed the whole window on every poll/restart.
        self._stuck_max_age_s: float = float(
            extra.get("stuck_max_age_s")
            or os.environ.get("CARBONVOICE_STUCK_MAX_AGE_S")
            or DEFAULT_STUCK_MAX_AGE_S
        )
        # Revisit window for tag-lagged voice messages: a *voice* message in
        # a group channel gets its ``tagged_user_ids`` ~10–30s after creation
        # (Flutter applies the picker tags via the batch PUT only once STT
        # finishes). A gate rejection for "no mention" on such a message is
        # therefore provisional — we hold the cursor (stuck signal) while the
        # message is younger than this, so the next tick re-fetches and
        # re-evaluates with the by-then-populated array. Without the hold,
        # the tag-set ``message:updated`` is the LAST event that message ever
        # emits — one stale read (or one coalesced-away tick) and the cursor
        # is past it forever (observed live: tag landed at +30s, bot never
        # answered). Text messages carry tags at create time and never hold.
        self._revisit_max_age_s: float = float(
            extra.get("revisit_max_age_s")
            or os.environ.get("CARBONVOICE_REVISIT_MAX_AGE_S")
            or DEFAULT_REVISIT_MAX_AGE_S
        )
        # Outbound dedup: defense-in-depth against the core re-sending the
        # same response once per queued follow-up when delivery confirmation
        # is lost (e.g. CV 502 mid-stream). Keyed by channel → (text-hash,
        # monotonic ts); a repeat of the SAME text to the SAME channel within
        # the window is dropped. Independent of SUPPORTS_MESSAGE_EDITING, so
        # it also covers cores/paths we don't control. Window is short so a
        # user legitimately repeating themselves isn't blocked for long.
        self._send_dedup_window_s: float = float(
            extra.get("send_dedup_window_s")
            or os.environ.get("CARBONVOICE_SEND_DEDUP_WINDOW_S")
            or 90
        )
        self._last_sent: Dict[str, "tuple[str, float]"] = {}
        # Serialize message fetches. Every WS ``message:created`` /
        # ``message:updated`` event (and each reconnect) fires on_tick →
        # _fetch_missed_messages. A burst of events would otherwise run many
        # overlapping fetches over the SAME cursor window in parallel, each
        # re-processing the same messages — a key amplifier of the
        # duplicate-processing bursts. The lock makes fetches mutually
        # exclusive; _fetch_missed_messages coalesces overlapping ticks to a
        # single *trailing* re-fetch (``_tick_pending``) — never a plain
        # drop, because the event that fired mid-flight may announce a write
        # (e.g. the tag-resolution PUT) that the in-flight fetch's HTTP query
        # predates. Dropping it would lose the only re-fire that message
        # ever gets.
        self._fetch_lock = asyncio.Lock()
        self._tick_pending = False
        # One-shot delayed re-tick while something is stuck (no-transcript
        # or revisit-held messages). In WS mode polling is stopped, so
        # without this a held message would only retry when the *next*
        # unrelated event happens to arrive — potentially much later on a
        # quiet workspace.
        self._stuck_retry_task: Optional[asyncio.Task] = None

        self._api = CarbonVoiceAPI(pat, base_url) if pat and HTTPX_AVAILABLE else None
        self._cursor = Cursor(state_path)
        self._seen = SeenCache()
        self._transport = Transport(
            base_url=base_url,
            pat=pat,
            poll_interval_s=poll_interval_s,
            ws_retry_max_s=ws_retry_max_s,
            on_tick=self._fetch_missed_messages,
        )
        self._channels = ChannelCache(self._api) if self._api else None
        self._reactions = (
            ReactionService(
                self._api,
                reaction_id=extra.get("reaction_id"),
                enabled=not bool(extra.get("disable_ack_reaction")),
                pending_reaction_id=(
                    extra.get("pending_reaction_id")
                    or os.environ.get("CARBONVOICE_PENDING_REACTION_ID")
                    or None
                ),
            )
            if self._api
            else None
        )
        # Dynamic allow-list (Hermes core's PairingStore) + deny-by-default
        # gate. The owner is filled in at connect() from whoami.created_by.
        self._approvals = ApprovalStore()
        self._allowlist = AllowlistGate.from_env(self._approvals)
        self._gate = MentionGate.from_env()
        self._ignored_log = (
            IgnoredSenderLog(ignored_log_path, self._channels)
            if self._channels
            else None
        )

        # Interactive onboarding: the channel where the agent asks the owner
        # to approve unknown senders. ``home_channel`` falls back to the
        # legacy CARBONVOICE_HOME_CHANNEL env if not in ``extra``.
        self._home_channel: Optional[str] = (
            first_str(extra.get("home_channel"))
            or first_str(os.environ.get("CARBONVOICE_HOME_CHANNEL"))
        )
        # Per-process record of unauthorized senders we've prompted about:
        # ``user_id → {"channel": <where they wrote>, "notified_at": <monotonic>}``.
        # Rate-limits the owner prompt (and the "request sent" reply to the
        # sender) to once per cooldown, and remembers the channel so
        # /cv-allow-user can resolve their name. /cv-deny-user CLEARS the
        # entry (rather than silencing it) so a denied user can ask again
        # and the owner is re-notified — the add/remove cycle stays open.
        self._pending_approval: Dict[str, Dict[str, Any]] = {}
        self._approval_cooldown_s: int = int(
            extra.get("approval_notify_cooldown_s")
            or os.environ.get("CARBONVOICE_APPROVAL_COOLDOWN_S")
            or 1800  # 30 min
        )
        # One-tap owner approval: maps the bot's prompt message_id → the
        # creator_id it asks about, so when the owner reacts 👍/👎 on that
        # prompt we know who to approve/deny without them typing the id.
        # Mirrors cv-claude-channels' pendingPermissionMessages. Bounded by
        # _MAX_PENDING_PROMPTS so a flood of strangers can't grow it forever.
        self._pending_prompts: "OrderedDict[str, str]" = OrderedDict()
        self._approve_reaction_id: str = (
            extra.get("approve_reaction_id")
            or os.environ.get("CARBONVOICE_APPROVE_REACTION_ID")
            or DEFAULT_APPROVE_REACTION_ID
        )
        self._reject_reaction_id: str = (
            extra.get("reject_reaction_id")
            or os.environ.get("CARBONVOICE_REJECT_REACTION_ID")
            or DEFAULT_REJECT_REACTION_ID
        )

        # Per-thread reply anchors + parent-text cache + (eventually)
        # engagement / outbound tracking. See conversations.py and
        # DEVELOPMENT.md §7.5 for the design.
        self._tracker = ConversationTracker(self._api)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # ``is_reconnect`` (cold boot vs gateway-driven reconnect) needs no
        # special handling here: the disk cursor makes both paths identical —
        # we always catch up from the last processed message either way.
        if not self._pat or self._api is None:
            logger.error("carbonvoice: CARBONVOICE_PAT not set")
            return False

        # Credential-scoped lock: prevent two gateways (e.g. different
        # HERMES_HOME dirs) from claiming the same PAT at once. Same-pid
        # re-acquire is allowed, so reconnect retries don't self-block.
        if not self._acquire_platform_lock(
            "carbonvoice-pat", self._pat, "Carbon Voice PAT"
        ):
            return False

        await self._api.open()

        try:
            self._self_user_id, owner_id = await self._api.whoami()
        except Exception as exc:
            logger.error("carbonvoice: /whoami failed: %s", exc)
            await self._api.close()
            return False
        if not self._self_user_id:
            logger.error("carbonvoice: /whoami returned no user id")
            await self._api.close()
            return False
        # From here on every request also carries agent-id, so the backend
        # can attribute traffic to this specific agent account.
        self._api.set_agent_id(self._self_user_id)

        # Deny-by-default: the bot's creator (whoami.created_by) is the
        # owner — always authorized, and the seed from which they approve
        # everyone else via /cv-allow. Auto-detected so the security
        # default needs no manual setup.
        self._allowlist.set_owner(owner_id)
        if owner_id:
            logger.info("carbonvoice: owner is %s (auto-detected from created_by)", owner_id)
            # Mirror the owner into the dynamic allow-list (PairingStore).
            # Hermes core's own authorization checks the pairing store for
            # every platform but doesn't know about `created_by`, so without
            # this the owner could pass the plugin gate yet be blocked by
            # core. Idempotent.
            if self._approvals.approve(owner_id, "owner"):
                logger.info("carbonvoice: owner mirrored into pairing store")
            else:
                logger.warning(
                    "carbonvoice: could NOT mirror owner into pairing store "
                    "(PairingStore available=%s) — core may block the owner",
                    self._approvals.available,
                )
        if not self._allowlist.has_any_authorizer:
            logger.warning(
                "carbonvoice: deny-by-default is active but NO authorized "
                "users — whoami returned no owner and CARBONVOICE_ALLOWED_USERS "
                "is empty. The bot will ignore everyone. Set "
                "CARBONVOICE_ALLOWED_USERS to your user_guid, or "
                "CARBONVOICE_ALLOW_ALL_USERS=true to disable gating."
            )

        if self._reactions is not None:
            await self._reactions.discover()

        await self._cursor.load()

        try:
            await self._fetch_missed_messages()
        except Exception as exc:
            logger.warning("carbonvoice: initial catch-up failed: %s", exc)

        await self._transport.start()

        self._mark_connected()
        logger.info(
            "carbonvoice: connected as %s (mode=%s, state=%s)",
            self._self_user_id, self._transport.mode, self._cursor.path,
        )
        return True

    async def disconnect(self) -> None:
        if self._stuck_retry_task is not None and not self._stuck_retry_task.done():
            self._stuck_retry_task.cancel()
        self._stuck_retry_task = None
        await self._transport.stop()
        await self._cursor.stop()
        if self._api is not None:
            await self._api.close()
        self._release_platform_lock()
        self._mark_disconnected()

    # ── Outbound (Hermes → Carbon Voice) ─────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._api is None:
            return SendResult(success=False, error="adapter not connected")
        if not content or not content.strip():
            return SendResult(success=False, error="empty content")

        # Outbound dedup: drop an identical re-send to the same channel inside
        # the dedup window. The core re-sends the same "first response" once
        # per queued follow-up when streaming delivery wasn't confirmed (CV
        # 502s make this common); without this guard the user sees the same
        # reply many times. We report success (not failure) so the core
        # treats it as delivered and stops retrying. Keyed by an order-stable
        # hash of the exact text.
        dedup_key = chat_id or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if dedup_key:
            prev = self._last_sent.get(dedup_key)
            now_m = time.monotonic()
            if (
                prev is not None
                and prev[0] == content_hash
                and now_m - prev[1] < self._send_dedup_window_s
            ):
                logger.info(
                    "carbonvoice: suppressed duplicate send to %s "
                    "(same text within %.0fs dedup window)",
                    dedup_key, self._send_dedup_window_s,
                )
                return SendResult(success=True, message_id=None)

        # v5 transport: the resolved thread root is sent to the server as
        # ``reply_to_message_id`` (cv-api PR #277 renamed the old
        # ``thread_id`` input). The server resolves threading itself —
        # ``resolveRootParentMessageId`` roots whatever id we pass, so
        # sending the thread root keeps it the root and no client-side
        # reply-anchor lookup is required.
        #
        # ``thread_id`` priority (Hermes-side concept; the value becomes
        # the wire ``reply_to_message_id`` below):
        #   1. ``metadata['thread_id']`` — populated by Hermes core from
        #      ``SessionSource.thread_id`` for group messages.
        #   2. ``reply_to`` from the caller — used as a fallback when no
        #      thread context exists (DMs keep thread_id=None on
        #      ``SessionSource`` to preserve one-session-per-DM-pair).
        thread_id = (metadata or {}).get("thread_id") or reply_to

        try:
            data = await self._api.send_text_v5(
                conversation_id=chat_id,
                transcript=content,
                reply_to_message_id=thread_id,
            )
            msg_id = first_str(data.get("id"), data.get("message_id"))
            # Record for outbound dedup only on a real, successful send — a
            # failed send must NOT prime the dedup (else a legit retry of a
            # genuinely-undelivered message would be suppressed).
            if dedup_key:
                self._last_sent[dedup_key] = (content_hash, time.monotonic())
            return SendResult(success=True, message_id=msg_id, raw_response=data)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            body = exc.response.text if exc.response is not None else ""
            return SendResult(
                success=False,
                error=f"HTTP {status}: {body[:500]}",
                retryable=status in (408, 429, 500, 502, 503, 504),
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a voice memo via ``POST /v5/messages/audio`` (multipart).

        ``audio_path`` is a local audio file. CV transcribes it
        server-side and threads the resulting message via the resolved
        thread root sent as ``reply_to_message_id`` (same resolution
        rules as :meth:`send`).

        Parameter names match :class:`BasePlatformAdapter.send_voice` —
        Hermes core's media dispatch (``base.py:3640``) invokes us with
        the keyword ``audio_path=``, so renaming this would break the
        agent's "MEDIA:/foo.mp3 in reply" flow. ``caption`` is accepted
        for signature compatibility but currently ignored (CV's audio
        endpoint doesn't take a caption — the transcript IS the caption).
        """
        if self._api is None:
            return SendResult(success=False, error="adapter not connected")
        thread_id = (metadata or {}).get("thread_id") or reply_to
        try:
            data = await self._api.send_audio_v5(
                conversation_id=chat_id,
                audio_path=audio_path,
                reply_to_message_id=thread_id,
            )
            msg_id = first_str(data.get("id"), data.get("message_id"))
            return SendResult(success=True, message_id=msg_id, raw_response=data)
        except FileNotFoundError as exc:
            return SendResult(success=False, error=str(exc))
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            body = exc.response.text if exc.response is not None else ""
            return SendResult(
                success=False,
                error=f"HTTP {status}: {body[:500]}",
                retryable=status in (408, 429, 500, 502, 503, 504),
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Attach an image to the conversation.

        ``image_url`` accepts either a publicly-resolvable URL or a path
        to a local file. URL → single attachment with ``type:"link"``.
        Local file → 4-step signed-URL flow (see
        :meth:`_send_file_or_link`). ``caption`` becomes the transcript
        on the same bubble — agent text and image arrive together.
        """
        return await self._send_file_or_link(
            chat_id=chat_id,
            path_or_url=image_url,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Attach a local image file (``.jpg``, ``.png``, ``.webp``, ...).

        Hermes core's media dispatch wraps local image paths as
        ``file://...`` URIs and routes them through
        :meth:`BasePlatformAdapter.send_multiple_images`, whose default
        implementation calls :meth:`send_image_file` per item. Without
        this override the agent's "MEDIA:/foo.png" flow would fall back
        to "🖼️ Image: /foo.png" plain-text from the base class — useless
        on CV. Routes through the same signed-URL flow as
        :meth:`send_document`; the file just happens to be an image.
        """
        return await self._send_file_or_link(
            chat_id=chat_id,
            path_or_url=image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Attach a document (any non-image file) to the conversation.

        Same mechanics as :meth:`send_image` — both go through the
        ``type:"file"`` attachment shape on CV; the difference is purely
        in the Hermes core method dispatch. Use this for ``.md``, PDFs,
        archives, audio clips not meant as voice memos, etc. For voice
        memos (transcribed server-side) use :meth:`send_voice`.

        Parameter names match :class:`BasePlatformAdapter.send_document`
        so Hermes core's media dispatch (``base.py:3652``) reaches us
        with the right keywords. ``file_name``, if provided, overrides
        the on-disk basename when building the attachment payload — e.g.
        for renaming ``/tmp/tmpXYZ`` to ``report.md`` on the recipient
        side.
        """
        return await self._send_file_or_link(
            chat_id=chat_id,
            path_or_url=file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            metadata=metadata,
        )

    # ── Attachment flow (URL or local file) ─────────────────────────────
    #
    # Mirrors the Flutter client's pattern: the agent sees its message
    # appear in the conversation immediately with an "Initializing"
    # placeholder, while the actual S3 upload runs in the background and
    # flips the status to ``Uploaded`` (or ``Failed``) when it settles.
    #
    # URL inputs skip the upload entirely — they just attach the URL
    # with ``type:"link"`` since the file is already hosted somewhere
    # the recipient can fetch.

    @staticmethod
    def _is_url(path_or_url: str) -> bool:
        return path_or_url.startswith(("http://", "https://"))

    @staticmethod
    def _guess_mime(path: Path) -> str:
        """Best-effort MIME type from filename extension.

        ``mimetypes`` ships a tiny built-in DB plus the system's
        ``/etc/mime.types``. We add ``.md`` → ``text/markdown`` because
        the stdlib still classifies markdown as ``text/x-markdown`` on
        some platforms and ``None`` on others; ``text/markdown`` is the
        IANA-registered form (RFC 7763) and what the agent's tooling
        will actually produce.
        """
        if path.suffix.lower() == ".md":
            return "text/markdown"
        guessed, _ = mimetypes.guess_type(str(path))
        return guessed or "application/octet-stream"

    async def _send_file_or_link(
        self,
        *,
        chat_id: str,
        path_or_url: str,
        caption: Optional[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        file_name: Optional[str] = None,
    ) -> SendResult:
        if self._api is None:
            return SendResult(success=False, error="adapter not connected")
        if not path_or_url:
            return SendResult(success=False, error="attachment path/URL required")

        thread_id = (metadata or {}).get("thread_id") or reply_to
        caption_text = (caption or "").strip()

        try:
            if self._is_url(path_or_url):
                attachment = {"type": "link", "link": path_or_url}
                data = await self._create_attachment_message(
                    chat_id=chat_id,
                    thread_id=thread_id,
                    caption=caption_text,
                    attachment=attachment,
                )
                msg_id = first_str(data.get("id"), data.get("message_id"))
                return SendResult(success=True, message_id=msg_id, raw_response=data)

            # Local file: signed URL → message-create with Initializing →
            # background S3 PUT + status update.
            path = Path(path_or_url).expanduser()
            if not path.is_file():
                return SendResult(
                    success=False, error=f"file not found: {path}",
                )
            mime_type = self._guess_mime(path)
            # Caller may override the basename so a temp path like
            # ``/tmp/tmpXYZ`` shows up as ``report.md`` on the recipient.
            filename = file_name or path.name

            urls = await self._api.get_signed_upload_urls(
                [{"filename": filename, "mimetype": mime_type}],
            )
            if not urls or not urls[0].get("url"):
                return SendResult(
                    success=False,
                    error="signedurl: empty response from /v3/attachments/signedurl",
                )
            signed_url = urls[0]["url"]
            canonical_link = signed_url.split("?", 1)[0]

            attachment = {
                "type": "file",
                "link": canonical_link,
                "filename": filename,
                "mime_type": mime_type,
                "status": "Initializing",
                "percent_complete": 0,
            }
            try:
                attachment["length_in_bytes"] = path.stat().st_size
            except OSError:
                pass  # non-fatal; server tolerates missing size

            data = await self._create_attachment_message(
                chat_id=chat_id,
                thread_id=thread_id,
                caption=caption_text,
                attachment=attachment,
            )
            msg_id = first_str(data.get("id"), data.get("message_id"))

            # Find the just-created attachment id in the response so the
            # background task can flip its status when S3 settles. The
            # server returns ``attachments[]`` in the order we sent them,
            # so the first/only entry is ours.
            created_attachments = data.get("attachments") or []
            attachment_id: Optional[str] = None
            if created_attachments:
                first_att = created_attachments[0]
                if isinstance(first_att, dict):
                    attachment_id = first_str(
                        first_att.get("id"), first_att.get("_id"),
                    )

            if attachment_id:
                base_body = {
                    "type": "file",
                    "link": canonical_link,
                    "filename": filename,
                    "mime_type": mime_type,
                }
                # Fire-and-forget — survives this method returning.
                asyncio.create_task(
                    self._upload_attachment_in_background(
                        signed_url=signed_url,
                        file_path=str(path),
                        mime_type=mime_type,
                        message_id=msg_id or "",
                        attachment_id=attachment_id,
                        base_body=base_body,
                    )
                )
            else:
                logger.warning(
                    "carbonvoice: no attachment id in response for %s — "
                    "skipping background upload + status update (message "
                    "will show 'Initializing' indefinitely on the recipient)",
                    filename,
                )

            return SendResult(success=True, message_id=msg_id, raw_response=data)
        except FileNotFoundError as exc:
            return SendResult(success=False, error=str(exc))
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            body = exc.response.text if exc.response is not None else ""
            return SendResult(
                success=False,
                error=f"HTTP {status}: {body[:500]}",
                retryable=status in (408, 429, 500, 502, 503, 504),
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _create_attachment_message(
        self,
        *,
        chat_id: str,
        thread_id: Optional[str],
        caption: str,
        attachment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create the message that carries *attachment*.

        Routes based on whether the caller supplied a caption:
          - caption present → ``POST /v5/messages/text`` with
            ``transcript`` + ``attachments`` (the server requires
            ``transcript`` to be non-empty on this endpoint).
          - caption absent → ``POST /v5/messages/attachment``
            (attachment-only message, no transcript).
        """
        if caption:
            return await self._api.send_text_v5(
                conversation_id=chat_id,
                transcript=caption,
                reply_to_message_id=thread_id,
                attachments=[attachment],
            )
        return await self._api.send_attachment_v5(
            conversation_id=chat_id,
            attachments=[attachment],
            reply_to_message_id=thread_id,
        )

    async def _upload_attachment_in_background(
        self,
        *,
        signed_url: str,
        file_path: str,
        mime_type: str,
        message_id: str,
        attachment_id: str,
        base_body: Dict[str, Any],
    ) -> None:
        """Push the bytes to S3 then flip the attachment status.

        Runs detached from ``send_document``/``send_image`` so the agent
        gets ``SendResult(success=True)`` immediately — the recipient
        sees the message bubble appear with an ``Initializing``
        placeholder and the file fills in once S3 acks. Mirrors how the
        Flutter client behaves on send.

        On S3 failure we PUT ``status:"Failed"`` so the recipient's UI
        renders a clear error state rather than a perpetual spinner.
        Both branches are wrapped in try/except — a transient failure on
        the status-update PUT must not crash the gateway event loop.
        """
        try:
            await self._api.upload_to_s3(signed_url, file_path, mime_type)
        except Exception as exc:
            logger.warning(
                "carbonvoice: S3 upload failed for %s (msg=%s att=%s): %s",
                file_path, message_id, attachment_id, exc,
            )
            try:
                await self._api.update_attachment(
                    message_id,
                    attachment_id,
                    {**base_body, "status": "Failed", "percent_complete": 0},
                )
            except Exception as inner:
                logger.warning(
                    "carbonvoice: update_attachment(Failed) also failed for "
                    "%s: %s", attachment_id, inner,
                )
            return

        try:
            await self._api.update_attachment(
                message_id,
                attachment_id,
                {**base_body, "status": "Uploaded", "percent_complete": 100},
            )
            logger.info(
                "carbonvoice: attachment uploaded — msg=%s att=%s file=%s",
                message_id, attachment_id, file_path,
            )
        except Exception as exc:
            logger.warning(
                "carbonvoice: update_attachment(Uploaded) failed for %s: %s — "
                "S3 upload itself succeeded; recipient may see stale "
                "'Initializing' status",
                attachment_id, exc,
            )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "carbonvoice", "chat_id": chat_id}

    # ── Thread-context fetch (PR 4) ──────────────────────────────────────
    #
    # When the agent gets @mentioned in a group thread for the first time
    # (no Hermes session yet for that thread), we have no history to feed
    # the LLM — it sees one isolated message and has to guess at context.
    # We fetch the thread's prior messages via CV's REST API and prepend
    # them as a ``[Thread context …]`` block to the user's message so
    # the agent has the prior history from turn 1.
    #
    # The "no active session" guard means this only fires on the first
    # turn in any given thread; every subsequent turn rides on the session
    # history that Hermes core maintains in SQLite, so there is no
    # duplication.
    #
    # CV has no native "list messages in thread" endpoint today, so we
    # combine two calls — a lightweight channel index (just ids +
    # ``parent_message_id``) plus a batched ``by-ids`` fetch — to assemble
    # the thread's transcript. When cv-api adds a direct thread-listing
    # endpoint the workaround collapses to one call; see
    # ``api.list_channel_message_index`` for the full rationale.

    def _has_active_session_for_thread(
        self,
        channel_id: str,
        thread_id: str,
        user_id: str,
    ) -> bool:
        """Return True when a Hermes session already covers this thread.

        Uses ``build_session_key()`` as the single source of truth so the
        key respects ``group_sessions_per_user`` /
        ``thread_sessions_per_user`` exactly the way Hermes core does at
        message-routing time. A drift here would mean we'd inject thread
        context on a turn where Hermes already has session history,
        duplicating the parent in every prompt.
        """
        session_store = getattr(self, "_session_store", None)
        if not session_store:
            return False
        try:
            from gateway.session import build_session_key

            store_cfg = getattr(session_store, "config", None)
            gspu = (
                getattr(store_cfg, "group_sessions_per_user", True)
                if store_cfg
                else True
            )
            tspu = (
                getattr(store_cfg, "thread_sessions_per_user", False)
                if store_cfg
                else False
            )

            source = SessionSource(
                platform=Platform("carbonvoice"),
                chat_id=channel_id,
                chat_type="group",
                user_id=user_id,
                thread_id=thread_id,
            )
            session_key = build_session_key(
                source,
                group_sessions_per_user=gspu,
                thread_sessions_per_user=tspu,
            )

            ensure = getattr(session_store, "_ensure_loaded", None)
            if callable(ensure):
                ensure()
            entries = getattr(session_store, "_entries", None) or {}
            return session_key in entries
        except Exception:
            return False

    async def _fetch_thread_context(
        self,
        channel_id: str,
        thread_id: str,
        current_msg_id: str,
        *,
        limit: int = 30,
    ) -> str:
        """Return a formatted ``[Thread context …]`` prefix for *thread_id*.

        Returns ``""`` (empty string) on any failure or when the thread
        has no prior content — callers should treat empty as "nothing to
        prepend" and pass the original user text through unchanged.

        Steps:
          1. Cache hit via :meth:`ConversationTracker.get_cached_thread_context`.
          2. ``api.list_channel_message_index`` → ids + ``parent_message_id``.
          3. Client-side filter to thread (root + replies whose
             ``parent_message_id == thread_id``).
          4. ``api.get_messages_by_ids_v5`` for the last ``limit``
             transcripts in chronological order.
          5. Exclude the current triggering message (it will be delivered
             as the user message itself) and exclude our own prior bot
             replies (circular context — feeding them back creates an
             echo that the LLM tends to repeat).
          6. Format ``[thread parent] name: text`` for the root and
             ``name: text`` for replies, wrap in the standard delimiters,
             cache, return.
        """
        if self._api is None or not thread_id:
            return ""

        cached = self._tracker.get_cached_thread_context(thread_id)
        if cached is not None:
            return cached

        try:
            index = await self._api.list_channel_message_index(
                channel_id, limit=200, direction="older"
            )
        except Exception as exc:
            logger.debug(
                "carbonvoice: list_channel_message_index(%s) failed: %s",
                channel_id, exc,
            )
            return ""

        if not index:
            return ""

        # Pick out items in this thread: the root and its direct replies.
        # CV is flat (DEVELOPMENT.md §4) so a single equality check on
        # ``parent_message_id`` covers every sibling — no walk needed.
        thread_items = []
        for item in index:
            mid = first_str(
                item.get("message_id"), item.get("_id"), item.get("id"),
            )
            if not mid:
                continue
            parent = first_str(
                item.get("parent_message_id"),
                item.get("parent_message_guid"),
                item.get("thread_id"),
            )
            is_root = mid == thread_id
            is_sibling = parent == thread_id
            if not (is_root or is_sibling):
                continue
            if mid == current_msg_id:
                continue
            thread_items.append((mid, item, is_root))

        if not thread_items:
            # Cache the empty result so we don't refetch on every turn
            # in an otherwise empty thread.
            self._tracker.set_cached_thread_context(thread_id, "")
            return ""

        # Order chronologically. The index endpoint returns ``created_at``
        # as either ISO or epoch ms depending on call; sort lexically when
        # string and numerically when number — both give the right order.
        def _ts(entry):
            ts = entry[1].get("created_at") or entry[1].get("created") or 0
            return ts
        thread_items.sort(key=_ts)

        # Cap to ``limit`` most-recent so a long-running thread doesn't
        # blow the prompt budget. Keep the root if present so context is
        # anchored even when the tail is large.
        if len(thread_items) > limit:
            head = [t for t in thread_items if t[2]][:1]  # the root, if any
            tail = [t for t in thread_items if not t[2]][-(limit - len(head)):]
            thread_items = head + tail

        ids = [mid for mid, _, _ in thread_items]
        try:
            full = await self._api.get_messages_by_ids_v5(channel_id, ids)
        except Exception as exc:
            logger.debug(
                "carbonvoice: get_messages_by_ids_v5 for thread context failed: %s",
                exc,
            )
            return ""

        # Index by id so we can preserve our chronological order.
        full_by_id = {
            first_str(m.get("id"), m.get("message_id"), m.get("_id")): m
            for m in (full or [])
            if isinstance(m, dict)
        }

        parts = []
        for mid, item, is_root in thread_items:
            msg = full_by_id.get(mid)
            if not msg:
                continue
            text = (extract_transcript(msg) or "").strip()
            if not text:
                continue
            creator = extract_creator_id(msg) or item.get("creator_id") or ""
            # Skip our own prior bot replies — feeding them back as
            # "[bot]: …" creates a circular context the LLM tends to echo.
            # Keep the thread parent even when authored by the bot (e.g.
            # the thread was opened by a cron post we're now replying to).
            if (
                creator
                and self._self_user_id
                and creator == self._self_user_id
                and not is_root
            ):
                continue
            name = creator
            if creator and self._channels is not None:
                try:
                    name = await self._channels.resolve_name(channel_id, creator) or creator
                except Exception:
                    name = creator
            name = name or "unknown"
            prefix = "[thread parent] " if is_root else ""
            parts.append(f"{prefix}{name}: {text}")

        if not parts:
            self._tracker.set_cached_thread_context(thread_id, "")
            return ""

        content = (
            "[Thread context — prior messages in this thread "
            "(not yet in conversation history):]\n"
            + "\n".join(parts)
            + "\n[End of thread context]\n\n"
        )
        self._tracker.set_cached_thread_context(thread_id, content)
        # INFO so it shows up in default gateway.log — operators need to
        # see when context was injected to debug "why did the bot know
        # that?" / "why did the bot miss that?" questions without flipping
        # to DEBUG. Volume is bounded: fires at most once per thread per
        # TTL window (subsequent mentions in the same thread hit the
        # active-session guard and skip this method entirely).
        logger.info(
            "carbonvoice: thread context injected for %s — %d prior message(s), %d chars",
            thread_id, len(parts), len(content),
        )
        return content

    # ── Inbound processing ───────────────────────────────────────────────

    async def _fetch_missed_messages(self) -> None:
        if self._api is None:
            return

        # Coalesce overlapping ticks to a TRAILING re-fetch, never a drop.
        # A burst of WS events must not spawn parallel fetches over the same
        # cursor window (duplicate-processing amplifier) — but the event that
        # arrives mid-fetch may announce a write the in-flight HTTP query
        # predates (observed: the tag-resolution PUT fired while the
        # transcript-ready tick was still fetching; dropping that tick lost
        # the only re-fire the message ever gets). So a tick that finds the
        # lock held flags ``_tick_pending``; the lock holder loops one more
        # fetch per flag before releasing.
        if self._fetch_lock.locked():
            self._tick_pending = True
            logger.debug(
                "carbonvoice: fetch already in progress — queuing trailing re-fetch"
            )
            return
        async with self._fetch_lock:
            await self._fetch_missed_messages_locked()
            while self._tick_pending:
                self._tick_pending = False
                await self._fetch_missed_messages_locked()

    async def _fetch_missed_messages_locked(self) -> None:
        request_started_at = now_iso()

        if not self._cursor.last_seen_at:
            logger.info(
                "carbonvoice: first run, starting from %s", request_started_at
            )
            self._cursor.advance(request_started_at)
            return

        try:
            messages = await self._api.fetch_recent(self._cursor.last_seen_at)
        except Exception as exc:
            logger.warning("carbonvoice: /v3/messages/recent failed: %s", exc)
            return  # don't advance cursor — retry same window next tick

        # One-tap approval: resolve any owner 👍/👎 reactions on our pending
        # prompts. Done first (and best-effort) so an approval lands even if
        # the prompt isn't in this fetch window. Never blocks the main path.
        try:
            await self._check_pending_prompt_reactions(messages)
        except Exception as exc:
            logger.debug("carbonvoice: pending-prompt reaction check failed: %s", exc)

        messages.sort(key=lambda m: m.get("created_at") or "")

        # Track the first "stuck" message (transcript not ready yet —
        # `_process_message` returns None). We hold the cursor just before
        # it so the next poll re-fetches from there and retries, instead of
        # advancing past and risking a skip. Mirrors the Claude Code
        # Channel's stuck-message handling.
        first_stuck_idx: Optional[int] = None
        for i, msg in enumerate(messages):
            try:
                result = await self._process_message(msg)
            except Exception as exc:
                logger.error("carbonvoice: process_message error: %s", exc)
                continue
            if result is None and first_stuck_idx is None:
                first_stuck_idx = i

        # Advance the cursor as far as is safe:
        #   - no stuck messages → advance to the request start time
        #     (clock-safe; avoids missing concurrent writes mid-call).
        #   - some stuck → advance only to just before the first stuck
        #     message, leaving it (and everything after) for the next poll.
        #     Earlier already-dispatched messages are deduped by SeenCache
        #     if the shrunk window re-fetches them.
        #   - first message stuck (idx 0) or no usable timestamp → leave
        #     the cursor unchanged so the stuck message is retried.
        if first_stuck_idx is None:
            self._cursor.advance(request_started_at)
        elif first_stuck_idx > 0:
            prev_created = messages[first_stuck_idx - 1].get("created_at")
            if isinstance(prev_created, str) and prev_created:
                self._cursor.advance(prev_created)

        # Something is held (no-transcript stuck or revisit-held): schedule a
        # one-shot delayed re-tick so the retry doesn't depend on the next
        # unrelated WS event arriving. In WS mode polling is stopped, so on a
        # quiet workspace a held message would otherwise wait for the next
        # reconnect cycle (observed: tens of minutes). One task at a time —
        # each retry re-schedules itself via this same path while anything
        # remains held.
        if first_stuck_idx is not None:
            self._schedule_stuck_retry()

    def _schedule_stuck_retry(self) -> None:
        if self._stuck_retry_task is not None and not self._stuck_retry_task.done():
            return

        async def _retry() -> None:
            await asyncio.sleep(STUCK_RETRY_DELAY_S)
            try:
                await self._fetch_missed_messages()
            except Exception as exc:
                logger.debug("carbonvoice: stuck-retry tick failed: %s", exc)

        self._stuck_retry_task = asyncio.create_task(_retry())

    # ── Inbound multimodal (PR 7) ────────────────────────────────────────
    #
    # CV inbound payloads carry ``attachments[]`` whose ``link`` is the
    # canonical S3 URL — auth-gated, returns 403 to unauthenticated
    # requests. To consume them we resolve a signed S3 GET URL via
    # ``GET /attachments/signedurl/:_id`` (authenticated with our PAT),
    # download the bytes to ``IMAGE_CACHE_DIR``, and return local
    # filesystem paths for Hermes core to inject into the agent's
    # multimodal context (Claude vision sees the bytes inline).
    #
    # Scope for v1: ``image/*`` only. Other mime types (PDFs,
    # ``text/*``, binaries) are dropped with a WARNING because Hermes
    # core has no native document-extraction pipeline today. Without
    # one, the agent receives a ``file://...pdf`` path it can't
    # natively read — it reaches for ``read_file`` (returns binary
    # garbage), then ``terminal`` (asks the operator to approve
    # ``pdftotext`` / similar), then ``execute_code`` (tries Python
    # parsers that may not be installed). Net UX: the user gets a
    # permission prompt instead of an answer. Better to skip cleanly
    # and document the gap.
    #
    # Document support is queued for a follow-up PR that adds an
    # extraction pass (likely via ``pypdf`` + ``markdown`` / ``html``
    # parsers) and prepends the extracted text into the agent's
    # message context the same way thread context is prepended today.
    # Audio attachments live in ``audio_models[]``, not
    # ``attachments[]``; the transcript is already extracted via
    # :func:`extract_transcript`.

    async def _collect_inbound_media(
        self, msg: Dict[str, Any]
    ) -> "tuple[list[str], list[str], list[str]]":
        """Process inbound attachments and return three lists:

          - ``media_urls``  — local filesystem paths of downloaded image
            files (bare paths, no ``file://`` scheme), ready for
            ``MessageEvent.media_urls``
          - ``media_types`` — parallel list of mime types
          - ``link_urls``   — bare URLs from ``type:"link"``
            attachments (CV's link-sharing UI flow), to be prepended
            to the agent's message text so it sees them the same way
            it would see a URL the user typed inline

        ``type:"link"`` entries are not downloaded — they don't
        reference uploaded files, they're URLs to external resources.
        Threading them into the text channel lets the agent reach for
        its own browser / fetch tools the same way it does for URLs
        embedded in the transcript directly.
        """
        if self._api is None:
            return [], [], []

        attachments = extract_attachments(msg)
        if not attachments:
            return [], [], []

        # Import the cache dir constant from core so downloaded files
        # land in a root the media-delivery validator already allows.
        # Local import keeps this module gateway-free at import time
        # (CI imports the plugin without core).
        from gateway.platforms.base import IMAGE_CACHE_DIR

        media_urls: list[str] = []
        media_types: list[str] = []
        link_urls: list[str] = []

        for att in attachments:
            aid = att.get("_id") or ""
            mime = (att.get("mime_type") or "").lower()
            att_type = (att.get("type") or "").lower()
            link = att.get("link") or ""
            filename = att.get("filename") or aid or "attachment.bin"

            # CV's link attachment: the user picked "share a URL" in
            # the UI. ``link`` is the actual external URL (not an S3
            # path); ``mime_type`` is null. Surface the URL inline so
            # the agent can reach for its existing web tools just like
            # it would for a URL typed in the transcript directly.
            if att_type == "link":
                if link:
                    link_urls.append(link)
                    logger.info(
                        "carbonvoice: inbound link attachment surfaced "
                        "to agent — %s", link,
                    )
                else:
                    logger.warning(
                        "carbonvoice: skipping link attachment %s — "
                        "no link URL in payload", filename,
                    )
                continue

            if mime.startswith("image/"):
                target_dir = IMAGE_CACHE_DIR
            else:
                logger.warning(
                    "carbonvoice: skipping inbound attachment %s (%s) — "
                    "only image/* is wired in this plugin version "
                    "(document pipeline pending — see DEVELOPMENT.md §4)",
                    filename, mime or "no-mime",
                )
                continue

            if not aid:
                logger.warning(
                    "carbonvoice: skipping inbound attachment %s — "
                    "no attachment_id to resolve a signed URL",
                    filename,
                )
                continue

            try:
                local_path = await self._api.download_attachment(
                    aid,
                    target_dir,
                    filename=filename,
                    max_bytes=self._max_attachment_bytes,
                )
            except ValueError as exc:
                # Size cap hit.
                logger.warning(
                    "carbonvoice: skipping oversized inbound attachment %s: %s",
                    filename, exc,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "carbonvoice: failed to download inbound attachment "
                    "%s (%s): %s", filename, aid, exc,
                )
                continue

            # Pass the bare local path (NOT a ``file://`` URI) — that's
            # what every other built-in adapter puts in ``media_urls``
            # (Slack: ``media_urls.append(cached_path)``; Telegram:
            # ``event.media_urls = [cached_path]``), and what Hermes core's
            # native-image routing expects: ``build_native_content_parts``
            # (agent/image_routing.py) does ``Path(raw_path)`` directly, so a
            # ``file://`` prefix makes the path non-existent and the image is
            # silently dropped as "unreadable" (only the older text/vision
            # path tolerated the scheme — once image_input_mode resolves to
            # ``native`` for a vision model, the prefix breaks inbound images).
            media_urls.append(str(local_path))
            media_types.append(mime)
            logger.info(
                "carbonvoice: inbound attachment downloaded — "
                "att=%s mime=%s path=%s",
                aid, mime, local_path,
            )

        return media_urls, media_types, link_urls

    async def _fetch_forwarded_content(
        self, share_link_id: str, channel_id: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve a forwarded message's original content via its share link.

        Returns ``{"text": <forwarded block>, "media_urls": [...],
        "media_types": [...]}`` on success, or ``None`` when the content
        isn't retrievable *yet* — share-link fetch failed, the original's
        attachments are still uploading, or an image download failed. The
        caller treats None as the stuck signal (hold cursor, retry next
        poll) while the message is young, and degrades to a placeholder
        past the cutoff. Mirrors cv-claude-channels' share-link handling.

        The text block:

            [Forwarded message from <name>]
            <original transcript or "(no transcript)">
            [Attached link: ...]          ← link attachments, inline
            [Attachment foo.pdf — ...]    ← non-image files, noted only

        Image attachments are downloaded through the share-link-scoped
        signed-URL route (the bot may lack access to the original
        message's channel; the link itself authorizes) into the same
        IMAGE_CACHE_DIR as regular inbound images, and returned as bare
        local paths for ``MessageEvent.media_urls``.
        """
        try:
            share_link = await self._api.get_share_link(share_link_id)
        except Exception as exc:
            logger.warning(
                "carbonvoice: share-link fetch failed for %s: %s",
                share_link_id, exc,
            )
            return None
        shared = (share_link or {}).get("shared_message")
        if not isinstance(shared, dict):
            logger.warning(
                "carbonvoice: share link %s has no shared_message "
                "(revoked / expired / no access?)",
                share_link_id,
            )
            return None

        # Original sender: resolve against this channel's roster (cache
        # hit). The original author often isn't a member of the channel
        # the forward landed in — fall back to the raw id.
        sm_creator = extract_creator_id(shared)
        sender = ""
        if sm_creator and self._channels is not None:
            sender = await self._channels.resolve_name(
                channel_id, sm_creator
            ) or ""
        sender = sender or sm_creator or "unknown sender"

        from gateway.platforms.base import IMAGE_CACHE_DIR

        media_urls: list[str] = []
        media_types: list[str] = []
        att_lines: list[str] = []

        for att in extract_attachments(shared):
            aid = att.get("_id") or ""
            mime = (att.get("mime_type") or "").lower()
            att_type = (att.get("type") or "").lower()
            link = att.get("link") or ""
            filename = att.get("filename") or aid or "attachment.bin"
            status = (att.get("status") or "").lower()

            if att_type == "link":
                # Same inline-URL treatment as wrapper-level link
                # attachments — the agent fetches it with its web tools.
                if link:
                    att_lines.append(f"[Attached link: {link}]")
                continue
            if status == "failed":
                att_lines.append(
                    f"[Attachment {filename} — upload failed on the "
                    "original message]"
                )
                continue
            if status and status != "uploaded":
                # Initializing / Uploading — the original's file isn't on
                # S3 yet. Retry the whole forward.
                logger.info(
                    "carbonvoice: forwarded attachment %s still %s — "
                    "holding for retry", filename, status,
                )
                return None
            if not mime.startswith("image/"):
                # Same image-only scope as _collect_inbound_media (no
                # document pipeline in Hermes core yet) — but note the
                # file in the text so the agent knows it exists.
                att_lines.append(
                    f"[Attachment {filename} ({mime or 'unknown type'}) — "
                    "not imported: only image attachments are supported]"
                )
                logger.warning(
                    "carbonvoice: skipping forwarded attachment %s (%s) — "
                    "only image/* is wired (see DEVELOPMENT.md §4)",
                    filename, mime or "no-mime",
                )
                continue
            if not aid:
                continue
            try:
                local_path = await self._api.download_share_link_attachment(
                    share_link_id,
                    aid,
                    IMAGE_CACHE_DIR,
                    filename=filename,
                    max_bytes=self._max_attachment_bytes,
                )
            except ValueError as exc:
                # Size cap — permanent, don't hold the cursor for it.
                att_lines.append(
                    f"[Attachment {filename} — skipped: too large]"
                )
                logger.warning(
                    "carbonvoice: oversized forwarded attachment %s: %s",
                    filename, exc,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "carbonvoice: forwarded attachment download failed "
                    "%s (%s): %s — holding for retry",
                    filename, aid, exc,
                )
                return None
            media_urls.append(str(local_path))
            media_types.append(mime)
            logger.info(
                "carbonvoice: forwarded attachment downloaded — "
                "att=%s mime=%s path=%s", aid, mime, local_path,
            )

        block = (
            f"[Forwarded message from {sender}]\n"
            + (extract_transcript(shared) or "(no transcript)")
        )
        if att_lines:
            block += "\n" + "\n".join(att_lines)
        return {
            "text": block,
            "media_urls": media_urls,
            "media_types": media_types,
        }

    async def _process_message(self, msg: Dict[str, Any]) -> Optional[bool]:
        """Process one inbound message; return its disposition for the cursor.

          - ``True``  — dispatched to the agent.
          - ``False`` — skipped for good (self-loop, single-user restrict,
            not allowed, deduped, gate-rejected). Safe to advance past.
          - ``None``  — *stuck*: the transcript isn't ready yet (CV is
            still transcribing), or the message is a forward whose
            share-link content couldn't be resolved yet. The caller holds
            the cursor just *before* this message so the next poll
            re-fetches and retries it, instead of advancing past and
            risking a skip. Mirrors the Claude Code Channel's null-return
            contract.
        """
        message_id = extract_message_id(msg)
        if not message_id:
            return False

        channel_id = extract_channel_id(msg)
        if not channel_id:
            return False

        creator_id = extract_creator_id(msg)

        # Self-loop guard.
        if creator_id and self._self_user_id and creator_id == self._self_user_id:
            return False

        # Optional single-user restriction (acts before transcript check so
        # we don't waste cycles on transcripts we'll drop anyway).
        if self._creator_id and creator_id and creator_id != self._creator_id:
            return False

        # Dedup FIRST — before the allowlist gate. The same message_id can
        # arrive twice nearly simultaneously (socket event + poll fetch); if
        # the gate's unauthorized branch ran before this check, BOTH copies
        # would log "dropped", react, and prompt before either marked seen —
        # the observed double-drop-per-message burst from a spamming sender.
        # Marking happens at each terminal branch below; checking up front
        # makes a redundant copy a no-op. (Revisitable gate rejections are
        # deliberately NOT marked, so they still get re-evaluated — see the
        # mention-gate branch.)
        if self._seen.is_seen(message_id):
            return False

        # Allowlist gate — default is allow-all (see AllowlistGate docstring).
        # When the operator has configured a restriction, short-circuit
        # rejected senders here so we can log them with a resolved username
        # before Hermes core's parallel check drops them.
        if not self._allowlist.is_allowed(creator_id):
            logger.info(
                "carbonvoice: dropped message from unauthorized sender %s",
                creator_id,
            )
            # Deny-by-default onboarding: react ⁉️ on the sender's message
            # (silent "pending approval") and ask the owner in the home
            # channel to approve them (rate-limited per pending user).
            await self._maybe_notify_unauthorized(
                creator_id, channel_id, message_id
            )
            if self._ignored_log is not None and creator_id:
                self._ignored_log.record(creator_id, channel_id)
            # Mark THIS message seen so the poll loop doesn't re-process the
            # exact same unauthorized message every tick. Without this, a
            # not-yet-approved sender's message is re-evaluated on every poll
            # (worsened by 502 retries re-fetching the same window) — the
            # observed 2500×-"dropped unauthorized" burst. This does NOT lock
            # the *user* out: once the owner approves them, their NEW messages
            # pass the gate normally; only this specific already-reacted
            # message is suppressed (SeenCache TTL is short, so even it
            # re-evaluates later if still unapproved).
            self._seen.mark(message_id)
            return False

        # Two-phase transcript: empty means "not ready yet" (CV is still
        # transcribing). Return None — the *stuck* signal — so the poll
        # loop holds the cursor just before this message and retries it
        # next tick, rather than advancing past it. Don't mark seen.
        #
        # BUT only while the message is young. A message with no transcript
        # is "stuck" only transiently; some never get one (image-only,
        # system events, failed STT). If we held the cursor for those
        # forever, every poll/restart would re-fetch the whole window from
        # the pinned timestamp and re-feed already-processed messages — the
        # "cadena de mensajes" bug. Past CARBONVOICE_STUCK_MAX_AGE_S we stop
        # waiting and let it advance the cursor (return False, not None).
        transcript = extract_transcript(msg)
        if not transcript:
            # Forwards are the exception to the stuck-wait: a forward with
            # no typed comment never gets a transcript of its own — the
            # content lives behind the share link (fetched below). Only a
            # *voice* comment (is_text_message False) still waits for STT
            # like any voice message, and at the age cutoff it falls
            # through to forward processing (comment lost) instead of
            # being skipped (whole forward lost).
            share_link_hint = extract_share_link_id(msg)
            if not share_link_hint or msg.get("is_text_message") is False:
                age = message_age_seconds(msg, now_utc())
                if age is None or age <= self._stuck_max_age_s:
                    return None
                if not share_link_hint:
                    logger.info(
                        "carbonvoice: message %s has no transcript after %.0fs "
                        "(> %ss) — treating as permanently empty, advancing past it",
                        message_id, age, self._stuck_max_age_s,
                    )
                    self._seen.mark(message_id)
                    return False
                logger.info(
                    "carbonvoice: forward %s voice comment never transcribed "
                    "after %.0fs — proceeding with forwarded content only",
                    message_id, age,
                )

        # V5 source-of-truth enrichment. The socket / v3-poll push gives
        # us a V2-shaped payload that trails the v5 GET on async fields:
        # ``tagged_user_ids`` is empty here until a backend job resolves
        # the tag picker selection, and attachment metadata can lag the
        # same way. CV's v5 endpoint is the canonical post-resolution
        # state — the Flutter client follows the same "socket = signal,
        # REST = truth" pattern.
        #
        # We do the GET only here, after the cheap-reject gates above
        # (self-loop, allowlist, dedupe, empty-transcript), so empty
        # ``message:created`` events don't pay the HTTP. On fetch
        # failure we keep the V2 payload — defensive, so a transient
        # /v5 hiccup doesn't drop an otherwise-deliverable message.
        # The parse helpers (``extract_*``) prefer V5 fields when
        # present, so reassigning ``msg`` is enough — no further
        # downstream changes needed.
        if self._api is not None:
            try:
                enriched = await self._api.get_message_v5(message_id)
            except Exception as exc:
                logger.debug(
                    "carbonvoice: v5 enrichment failed for %s: %s — "
                    "continuing with v2 payload",
                    message_id, exc,
                )
                enriched = None
            if enriched:
                # Staleness guard: the v5 GET can race a write the push
                # payload already reflects (read-replica lag) — if the v2
                # copy has ``tagged_user_ids`` and the v5 copy doesn't,
                # keep the populated array rather than letting the
                # enrichment erase the mention.
                if not enriched.get("tagged_user_ids") and msg.get("tagged_user_ids"):
                    enriched["tagged_user_ids"] = msg["tagged_user_ids"]
                msg = enriched
                # Re-pull transcript from the (canonical) v5 payload —
                # usually the same string but keeps everything in one
                # shape after this point.
                transcript = extract_transcript(msg) or transcript

        # Server-side dedup (persistent, survives restarts). We put an ack
        # reaction on every *accepted* message, so a message already
        # carrying the bot's ack was already processed — skip it. This
        # complements the in-memory SeenCache, which is lost on restart
        # and expires after 5 min. Crucially it breaks the
        # ``use_last_updated`` re-capture loop: the ack reaction and the
        # bot's in-thread reply both bump ``updated_at``, so the poller
        # keeps re-fetching the same message; without a durable marker the
        # SeenCache eventually lapses and the agent re-answers the same
        # message (observed: one message dispatched 5× across a day of
        # restarts). We read ``reaction_summary`` from the canonical v5
        # payload above. Mark seen too so immediate re-polls skip without
        # paying another v5 GET. Mirrors the Claude Code Channel's
        # reaction-based ``isProcessed`` dedup.
        if (
            self._reactions is not None
            and self._reactions.enabled
            and self._self_user_id
            and bot_has_reacted(
                msg, self._self_user_id, self._reactions.reaction_id
            )
        ):
            logger.debug(
                "carbonvoice: skip %s — already acked by bot (server-side dedup)",
                message_id,
            )
            self._seen.mark(message_id)
            return False

        # Admin allow-list commands (/cv-allow, /cv-deny, /cv-list). Only the
        # OWNER may run these — a normally-approved user must not be able to
        # escalate by approving others. Handled here and NOT forwarded to the
        # agent.
        if self._allowlist.is_owner(creator_id):
            cmd = parse_admin_command(transcript)
            if cmd is not None:
                # Dedup BEFORE running the command. The command sends a reply
                # ("✅ Allowed …") which bumps updated_at and re-fires the
                # poll; if the durable ack isn't on the server yet (or the
                # SeenCache was lost to a restart), the re-fetched command
                # re-runs and re-replies — the observed 298×-spam bug. So we
                # (1) mark the in-memory SeenCache and (2) put the durable
                # server-side ack reaction *and wait for it* — BEFORE sending
                # the reply. ``ack_sync`` blocks until the marker is on the
                # server, so the re-fetch is guaranteed deduped by the
                # ``bot_has_reacted`` check above. ``approve``/``revoke`` are
                # idempotent too, so a stale in-flight copy is a harmless no-op.
                self._seen.mark(message_id)
                if self._reactions is not None:
                    await self._reactions.ack_sync(message_id)
                await self._handle_admin_command(channel_id, cmd)
                return False

        # Resolve chat_type before the mention gate so the gate can short-
        # circuit group messages without spinning up the rest of the
        # pipeline (visual ack, parent lookup, name resolution). The
        # channel cache makes the first message in each channel pay one
        # HTTP call; every subsequent message is free.
        chat_type = "dm"
        if self._channels is not None:
            chat_type = await self._channels.resolve_chat_type(channel_id)

        # Mention gate: in group channels, only respond when the agent
        # is @-mentioned (or the channel is configured to bypass). DMs
        # always pass. Evaluated before the visual ack so users in
        # non-mention scenarios don't see a phantom "I saw it" with no
        # follow-up reply.
        decision = self._gate.evaluate(
            msg=msg,
            chat_type=chat_type,
            channel_id=channel_id,
            self_user_id=self._self_user_id,
        )
        if not decision.process:
            logger.debug(
                "carbonvoice: skip message %s in %s — %s",
                message_id, channel_id, decision.reason,
            )
            # Revisitable rejection ("group without @-mention") of a *voice*
            # message: the verdict is provisional. Flutter applies picker
            # tags via the batch ``PUT /messages/:id/tagged-users`` only
            # after STT finishes (~10–30s post-create), so at this moment
            # ``tagged_user_ids`` may simply not be populated yet — or our
            # read raced the tag write (the tag-set ``message:updated`` tick
            # can fetch within ~100ms of the PUT and see a stale copy).
            # Returning False here advances the cursor past the message, and
            # since the tag PUT emits the LAST update that message ever
            # gets, the mention would be lost forever (observed live). So:
            # hold the cursor (stuck signal) while the message is young
            # enough for tags to still be in flight; the retry tick
            # re-fetches and re-evaluates. Text messages carry their tags on
            # the create body, so a missing mention there is final — no hold.
            if (
                decision.revisitable
                and msg.get("is_text_message") is False
            ):
                age = message_age_seconds(msg, now_utc())
                if age is not None and age <= self._revisit_max_age_s:
                    return None
            # Leave revisitable rejections out of the dedup cache so a
            # follow-up ``message:updated`` re-fire (e.g. cv-api emits
            # one after the async tag-resolution job populates
            # ``tagged_user_ids``) gets another shot at the gate. Stable
            # rejections (ignored channel, etc.) mark seen so we don't
            # re-evaluate them on every retry. See GateDecision docstring.
            if not decision.revisitable:
                self._seen.mark(message_id)
            return False

        # Forwarded message (share link): resolve the original message's
        # content BEFORE committing (mark-seen + ack) so a failed or
        # not-ready fetch can return None — the stuck signal — and the
        # cursor holds for a retry next poll. Mirrors cv-claude-channels'
        # retry-don't-skip contract for share links. Past the stuck cutoff
        # we degrade to a placeholder rather than pinning the cursor
        # forever (revoked/expired links never resolve).
        forwarded: Optional[Dict[str, Any]] = None
        share_link_id = extract_share_link_id(msg)
        if share_link_id and self._api is not None:
            forwarded = await self._fetch_forwarded_content(
                share_link_id, channel_id
            )
            if forwarded is None:
                age = message_age_seconds(msg, now_utc())
                if age is None or age <= self._stuck_max_age_s:
                    return None
                logger.warning(
                    "carbonvoice: forwarded content for %s (share link %s) "
                    "unavailable after %.0fs — delivering placeholder",
                    message_id, share_link_id, age,
                )
                forwarded = {
                    "text": "[Forwarded message — original content unavailable]",
                    "media_urls": [],
                    "media_types": [],
                }

        # Decision is "process" — commit to it. Marking seen here (rather
        # than before the gate) guarantees we only dedup messages we
        # actually dispatch; a re-fire with new metadata still gets a
        # fair gate evaluation up to this point.
        self._seen.mark(message_id)

        # Fire the visual ack first so the user sees feedback in <100ms,
        # well before the agent's reply (which can take 10+ seconds).
        if self._reactions is not None:
            self._reactions.ack(message_id)

        # Lane anchor: compute the thread root for this inbound message
        # and record it in the tracker so the next outbound reply threads
        # under the correct root. Carbon Voice enforces flat replies (see
        # DEVELOPMENT.md §4), so ``parent_message_id`` is always the true
        # root — no walking required. The tracker stores the anchor keyed
        # by ``thread_id``, and ``send()`` reads ``metadata['thread_id']``
        # populated by Hermes core from ``SessionSource.thread_id`` — so
        # concurrent threads in the same channel each resolve their own
        # anchor (closes the §7.6 latent bug end-to-end).
        parent = first_str(
            msg.get("parent_message_id"), msg.get("parent_message_guid")
        )
        thread_id = ConversationTracker.thread_id_of(msg)
        if thread_id:
            self._tracker.set_reply_anchor(thread_id, thread_id)

        # Resolve the sender's display name from the channel roster
        # (json_collaborators on GET /channel/{id}, cached). The old
        # GET /v3/users/{id} endpoint is dead (404), so the channel
        # collaborator list is the source of truth — and it's the same
        # payload we already fetched for chat_type above, so this is a
        # cache hit. Falls back to the raw guid when the sender isn't in
        # the list (shouldn't happen — you must be a collaborator to post).
        user_name = ""
        if creator_id and self._channels is not None:
            user_name = await self._channels.resolve_name(channel_id, creator_id) or ""
        if not user_name and creator_id:
            user_name = creator_id

        reply_to_text = await self._tracker.get_parent_text(parent)

        # Mentions now arrive structured in ``tagged_user_ids`` (see
        # parse.is_user_mentioned). The Flutter composer sends the
        # transcript as plain text — ``@Name`` without the guid — so
        # there is no inline ``@[name](guid)`` markup left to strip; pass
        # the transcript through as-is.
        clean_text = transcript

        # Forwarded message: the agent reads the original content first,
        # then the forwarder's comment (when there is one) — same layout
        # cv-claude-channels sends:
        #
        #     [Forwarded message from <original sender>]
        #     <original transcript / attachment lines>
        #
        #     [Forwarded by <user>]
        #     <comment>
        if forwarded is not None:
            if transcript:
                clean_text = (
                    f"{forwarded['text']}\n\n"
                    f"[Forwarded by {user_name}]\n{transcript}"
                )
            else:
                clean_text = forwarded["text"]

        # Session sharing in groups: pass the thread root as
        # ``SessionSource.thread_id`` so Hermes core composes a shared
        # session key (``agent:main:carbonvoice:group:<chat_id>:<thread_id>``)
        # and prefixes each user message with ``[sender name]`` for
        # multi-user attribution. DMs intentionally keep ``thread_id=None``:
        # a DM should remain one session per pair, not split per top-level
        # message inside the conversation.
        session_thread_id = thread_id if chat_type == "group" else None

        # Thread-context fetch (PR 4): when this is the first @mention in
        # a group thread (no Hermes session yet), pull the prior messages
        # so the agent has context from turn 1. Guard with the
        # "no active session" check so subsequent turns ride on Hermes'
        # SQLite session history without re-injecting the parent each
        # time. DMs skip the fetch — their single session already covers
        # the conversation, and there are no sibling participants whose
        # context we'd be missing.
        if (
            chat_type == "group"
            and session_thread_id
            and creator_id
            and not self._has_active_session_for_thread(
                channel_id, session_thread_id, creator_id,
            )
        ):
            context_prefix = await self._fetch_thread_context(
                channel_id=channel_id,
                thread_id=session_thread_id,
                current_msg_id=message_id,
            )
            if context_prefix:
                clean_text = context_prefix + clean_text

        source = SessionSource(
            platform=Platform("carbonvoice"),
            chat_id=channel_id,
            chat_name=f"cv:{channel_id[:8]}",
            chat_type=chat_type,
            user_id=creator_id or "",
            user_name=user_name or creator_id or "",
            message_id=message_id,
            thread_id=session_thread_id,
        )
        # Inbound multimodal (PR 7): pull any attached files into local
        # caches so Hermes core's vision pipeline can consume them. CV's
        # S3 URLs need auth, so we resolve a signed GET URL per file
        # attachment, download via that, and hand Hermes core a local
        # filesystem path in ``media_urls``. Image attachments are
        # routed to vision; ``type:"link"`` attachments (CV's link-
        # sharing UI) return their URLs in ``link_urls`` so we can
        # prepend them to the visible text — the agent then sees them
        # the same way it sees URLs typed inline, and uses its existing
        # browser / fetch tools to consume them. Anything else (PDFs,
        # binaries, …) is dropped with a WARNING.
        media_urls, media_types, link_urls = await self._collect_inbound_media(msg)

        # Images attached to the *forwarded* (original) message ride the
        # same vision pipeline as the wrapper's own attachments. Forwarded
        # images first — they're what the text block describes.
        if forwarded is not None and forwarded["media_urls"]:
            media_urls = list(forwarded["media_urls"]) + media_urls
            media_types = list(forwarded["media_types"]) + media_types

        # If CV's link-share UI was used, surface the URL(s) inline so
        # the agent can fetch them naturally. Prepending preserves the
        # user's own text right after, so the agent reads:
        #
        #     [Attached link: https://...]
        #     <user's actual message>
        if link_urls:
            link_prefix = "\n".join(
                f"[Attached link: {u}]" for u in link_urls
            )
            clean_text = f"{link_prefix}\n{clean_text}" if clean_text else link_prefix

        # Participant roster: give the agent the names of everyone in the
        # conversation (not just whoever is speaking) so it can address
        # people and attribute statements. Sourced from the channel
        # collaborator list (cache hit — same payload as chat_type), with
        # the bot itself excluded. Injected via ``channel_context``, which
        # Hermes core prepends once after the sender prefix and keeps in
        # history — unlike ``channel_prompt`` which resets per message and
        # would bust the prompt cache. We only inject when there are ≥2
        # other humans: in a 1:1 DM the sender's name already rides in the
        # system prompt (``SessionSource.user_name``), so a one-name roster
        # would be redundant noise.
        channel_context: Optional[str] = None
        if self._channels is not None:
            roster = await self._channels.get_roster(channel_id)
            others = sorted(
                n for g, n in roster.items() if g != self._self_user_id
            )
            if len(others) >= 2:
                channel_context = (
                    "[Participants in this conversation: "
                    + ", ".join(others)
                    + "]"
                )

        # Mark VOICE when ``CARBONVOICE_VOICE_OUT=true`` so Hermes core's
        # auto-TTS gate (``base.py:3493``) accepts this event for voice-
        # mode dispatch. CV doesn't distinguish text-typed vs voice-
        # transcribed at the outbound layer (everything ends up as
        # either a text bubble or a voice memo bubble), so applying
        # VOICE to every inbound is the right abstraction for a
        # voice-first platform — the operator opts in once and gets a
        # consistent symmetric experience.
        msg_type = MessageType.VOICE if self._voice_out else MessageType.TEXT
        event = MessageEvent(
            text=clean_text,
            message_type=msg_type,
            source=source,
            raw_message=msg,
            message_id=message_id,
            reply_to_message_id=parent,
            reply_to_text=reply_to_text,
            media_urls=media_urls,
            media_types=media_types,
        )
        # ``channel_context`` (participant roster) is a *newer* Hermes core
        # field on MessageEvent (added with the Discord channel-history
        # backfill). Set it only when this core supports it — passing it as a
        # ctor kwarg on an older core raises "unexpected keyword argument
        # 'channel_context'" and crashes every message. Setting the attribute
        # post-construction degrades gracefully: the roster is dropped on old
        # cores, everything else still works.
        if channel_context and hasattr(event, "channel_context"):
            event.channel_context = channel_context

        # Dispatch in a background task so processing one message can't block
        # the poll/WS loop while the agent thinks.
        asyncio.create_task(self._dispatch(event))
        return True

    # ── Interactive allow-list (deny-by-default onboarding) ──────────────

    async def _maybe_notify_unauthorized(
        self, creator_id: str, channel_id: str, message_id: str = ""
    ) -> None:
        """React to an unknown sender's message + ask the owner to approve.

        The sender gets a silent "pending approval" reaction (⁉️) on their
        message — NOT a text reply. A text reply clutters the channel and,
        worse, spammed every old conversation when we switched to
        deny-by-default (each re-flagged sender got a wall message). A
        reaction is unobtrusive and self-evidently "seen but not answered".

        The owner prompt (in the home channel) is rate-limited to once per
        ``approval_notify_cooldown_s`` per user so a persistent stranger
        doesn't spam the owner — but the owner IS re-notified after the
        cooldown (a single prompt could be missed). Always records the
        channel they wrote in (for name resolution on approval).
        """
        if not creator_id:
            return

        # No reaction on the sender's message. We used to react ⁉️ here, but
        # it was NOT cooldown-gated — a spamming stranger got one reaction per
        # message, which buried the owner in CV notifications. Mirroring
        # cv-claude-channels: an unknown sender's messages are dropped
        # silently; the only feedback is the owner prompt below (rate-limited)
        # and a one-time "you've been added" message to the sender once the
        # owner approves them (see _handle_admin_command's allow branch).
        now = time.monotonic()
        entry = self._pending_approval.get(creator_id)
        if entry is None:
            # notified_at=None means "never prompted" — distinct from a real
            # timestamp. (time.monotonic() can be small right after boot, so a
            # 0.0 sentinel would silence the FIRST prompt if a stranger wrote
            # within one cooldown of startup.)
            entry = {"channel": channel_id, "notified_at": None}
            self._pending_approval[creator_id] = entry
        else:
            # Keep the most recent channel for name resolution.
            entry["channel"] = channel_id or entry.get("channel") or ""
        # Cooldown gate: skip if we prompted recently (but always prompt the
        # first time, when notified_at is None).
        last = entry.get("notified_at")
        if last is not None and now - float(last) < self._approval_cooldown_s:
            return
        entry["notified_at"] = now

        # (A) prompt the owner in the home channel.
        if self._api is not None and self._home_channel:
            name = ""
            if self._channels is not None:
                try:
                    name = await self._channels.resolve_name(channel_id, creator_id) or ""
                except Exception:
                    name = ""
            who = f"{name} ({creator_id})" if name else creator_id
            text = (
                f"👤 {who} wants to talk to me but isn't authorized.\n"
                f"React 💯 to allow · 👎 to block — "
                f"or reply /cv-allow-user {creator_id}"
            )
            try:
                result = await self.send(self._home_channel, text)
                # Map the prompt message → the user it's about, so an owner
                # 👍/👎 reaction on it resolves the decision without typing
                # the id. (cv-claude-channels' pendingPermissionMessages.)
                prompt_id = getattr(result, "message_id", None)
                if prompt_id:
                    self._pending_prompts[prompt_id] = creator_id
                    while len(self._pending_prompts) > self._MAX_PENDING_PROMPTS:
                        self._pending_prompts.popitem(last=False)
                logger.info(
                    "carbonvoice: asked owner to approve %s in home channel "
                    "(prompt=%s, react 💯/👎)",
                    creator_id, prompt_id,
                )
            except Exception as exc:
                logger.warning(
                    "carbonvoice: failed to notify owner about %s: %s",
                    creator_id, exc,
                )
        elif self._api is not None:
            logger.info(
                "carbonvoice: unauthorized sender %s — no CARBONVOICE_HOME_CHANNEL "
                "configured, can't ask the owner to approve (set it to enable "
                "interactive onboarding)",
                creator_id,
            )

    async def _check_pending_prompt_reactions(
        self, polled: "list[Dict[str, Any]]"
    ) -> None:
        """Resolve owner 👍/👎 reactions on pending approval prompts.

        For each tracked prompt (prompt msg_id → creator_id), read the
        reactions on that prompt message and, if the OWNER reacted with the
        approve or reject reaction, apply the decision — no typed command.
        Mirrors cv-claude-channels' ``checkPendingPermissions``.

        Prompts already in the polled batch are read from it (free); any
        others are fetched by id (the owner's reaction won't necessarily
        bring the bot's own prompt into ``fetch_recent``). Only the owner's
        reaction counts — a stranger reacting 👍 on their own prompt must
        not self-approve.
        """
        if not self._pending_prompts or self._api is None:
            return
        owner = self._allowlist.owner_id
        if not owner:
            return  # without a known owner, nobody can authorize via reaction
        wanted = {self._approve_reaction_id, self._reject_reaction_id}

        by_id = {
            mid: m
            for m in polled
            if isinstance(m, dict) and (mid := extract_message_id(m))
        }
        # Snapshot keys — we mutate _pending_prompts as we resolve.
        for prompt_id in list(self._pending_prompts.keys()):
            creator_id = self._pending_prompts.get(prompt_id)
            if not creator_id:
                continue
            msg = by_id.get(prompt_id)
            if msg is None:
                try:
                    msg = await self._api.get_message_v5(prompt_id)
                except Exception:
                    msg = None
            if not isinstance(msg, dict):
                continue
            reactors = reactors_for(msg, wanted)
            if owner not in reactors:
                continue
            # Owner reacted. Approve takes precedence if both are present.
            approvers = reactors_for(msg, {self._approve_reaction_id})
            cmd = "allow" if owner in approvers else "deny"
            self._pending_prompts.pop(prompt_id, None)
            logger.info(
                "carbonvoice: owner reacted %s on prompt %s → %s %s",
                "💯" if cmd == "allow" else "👎", prompt_id, cmd, creator_id,
            )
            try:
                await self._handle_admin_command(
                    self._home_channel or "", (cmd, creator_id)
                )
            except Exception as exc:
                logger.warning(
                    "carbonvoice: failed to apply reaction verdict for %s: %s",
                    creator_id, exc,
                )

    def _drop_pending_prompts_for(self, creator_id: str) -> None:
        """Forget any pending approval prompts about *creator_id* (after a
        decision via either reaction or command), so a stale reaction on an
        old prompt can't re-trigger."""
        for pid in [
            p for p, c in self._pending_prompts.items() if c == creator_id
        ]:
            self._pending_prompts.pop(pid, None)

    async def _resolve_pending_name(self, user_id: str) -> str:
        """Display name of a pending user, from the channel they wrote in."""
        entry = self._pending_approval.get(user_id) or {}
        origin = entry.get("channel")
        if not origin or self._channels is None:
            return ""
        try:
            return await self._channels.resolve_name(origin, user_id) or ""
        except Exception:
            return ""

    async def _handle_admin_command(
        self, channel_id: str, cmd: "tuple[str, Optional[str]]"
    ) -> None:
        """Run an owner allow-list command and reply in *channel_id*."""
        action, arg = cmd
        reply: Optional[str] = None

        if action == "list":
            rows = self._approvals.list_approved()
            if not rows:
                reply = (
                    "No allowed users yet. "
                    "(The owner and CARBONVOICE_ALLOWED_USERS still apply.)"
                )
            else:
                lines = []
                for r in rows:
                    uid = r.get("user_id") or ""
                    nm = r.get("user_name") or ""
                    lines.append(f"• {nm} ({uid})" if nm else f"• {uid}")
                reply = "Allowed users:\n" + "\n".join(lines)

        elif action == "allow":
            if not arg:
                reply = "Usage: /cv-allow-user <user_guid>"
            else:
                # Resolve the name from the channel they originally wrote in
                # (saved in _pending_approval) — they're a stranger in the
                # home channel, so resolving there yields nothing.
                name = await self._resolve_pending_name(arg)
                # Grab their origin channel BEFORE popping the pending entry,
                # so we can tell them (in the channel they wrote in) that
                # they've been added.
                origin = (self._pending_approval.get(arg) or {}).get("channel")
                ok = self._approvals.approve(arg, name)
                self._pending_approval.pop(arg, None)
                self._drop_pending_prompts_for(arg)
                reply = (
                    f"✅ Allowed {name or arg}. They can talk to me now."
                    if ok
                    else f"⚠️ Couldn't approve {arg} — allow-list store unavailable."
                )
                # Tell the now-approved user (once) in the channel they wrote
                # in, so they know they can start talking. Best-effort.
                if ok and origin and self._api is not None:
                    # Greet them by first name when we resolved one; fall back
                    # to a plain greeting so we never send a dangling "Hey !".
                    greeting = (
                        f"Hey {name.split()[0]}! " if name and name.split() else "Hey! "
                    )
                    try:
                        await self.send(
                            origin,
                            f"✅ {greeting}You've been added to the allow-list — "
                            "you can talk to me now. Go ahead!",
                        )
                    except Exception as exc:
                        logger.debug(
                            "carbonvoice: failed to notify approved user %s: %s",
                            arg, exc,
                        )

        elif action == "deny":
            if not arg:
                reply = "Usage: /cv-deny-user <user_guid>"
            else:
                self._approvals.revoke(arg)  # drop if previously approved
                # Keep the pending entry but ARM its cooldown (notified_at=now)
                # instead of deleting it. Deleting reset the cooldown, so a
                # sender who spams messages got a NEW prompt within a second of
                # being denied — an endless deny→message→prompt→deny loop. By
                # arming the cooldown the add/remove cycle stays open (they can
                # ask again after the cooldown) without instant re-prompting.
                self._drop_pending_prompts_for(arg)
                ent = self._pending_approval.get(arg)
                if ent is None:
                    ent = {"channel": "", "notified_at": None}
                    self._pending_approval[arg] = ent
                ent["notified_at"] = time.monotonic()
                reply = (
                    f"🚫 {arg} denied — removed from the allow-list. "
                    "They can request access again later."
                )

        if reply and self._api is not None:
            try:
                await self.send(channel_id, reply)
            except Exception as exc:
                logger.warning("carbonvoice: failed to send admin reply: %s", exc)

    async def _dispatch(self, event: MessageEvent) -> None:
        try:
            await self.handle_message(event)
        except Exception as exc:
            logger.exception("carbonvoice: dispatch failed: %s", exc)
        finally:
            # Clear the unread badge once we've at least attempted handling.
            # On failure we still mark read — the operator sees the error in
            # logs; leaving the notification doesn't trigger a retry.
            if self._mark_read_enabled and self._api is not None:
                channel_id = event.source.chat_id
                msg_id = event.message_id
                if channel_id and msg_id:
                    try:
                        await self._api.mark_read(channel_id, msg_id)
                    except Exception as exc:
                        logger.debug(
                            "carbonvoice: mark_read(%s, %s) failed: %s",
                            channel_id, msg_id, exc,
                        )
