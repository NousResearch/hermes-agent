"""
Kakao i Open Builder ("channel chatbot" / skill-server) platform adapter
for Hermes Agent.

A bundled platform plugin that runs an aiohttp webhook server, accepts
Kakao "skill" requests (POSTed by the Open Builder bot engine whenever a
user messages the connected KakaoTalk channel), and relays them to/from
the agent via the standard ``BasePlatformAdapter`` interface.

Why this adapter looks different from every other bundled platform
--------------------------------------------------------------------

Every other Hermes platform adapter (Telegram, LINE, Discord, ...) has a
real *push* API: the adapter can call ``send()`` at any moment and the
platform delivers it. Kakao's Open Builder skill-server model has no such
API. Official references (Korean):

- Skill setup:   https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/make_skill
- Response JSON: https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format
- Event API (why proactive push is out of reach for this adapter):
  https://kakaobusiness.gitbook.io/main/tool/chatbot/main_notions/event-api

The model is **pure request-response, one reply per user utterance**:

1. A user messages the KakaoTalk channel. The bot engine matches a block
   (we use the fallback block, since hermes is the "brain" and doesn't
   need per-intent blocks) and POSTs a ``SkillRequest`` JSON body to our
   webhook.
2. We must answer with a ``SkillResponse`` JSON body **as the HTTP
   response to that same POST**, within Kakao's **5-second skill
   timeout**. There is no separate "send a message" call.
3. If the LLM needs longer than that, and *only* if the matched block has
   the callback option enabled, the request carries a one-shot
   ``callbackUrl`` (valid 5 minutes, usable exactly once). We can return
   ``{"version": "2.0", "useCallback": true}`` immediately and POST the
   real answer to ``callbackUrl`` later, still within the turn.
4. Once that HTTP response (or the callback POST) has gone out, the
   channel is closed. There is no way to message the user again until
   they send another utterance. Cron/proactive delivery is therefore
   **not supported** on this adapter: out-of-process delivery
   (``deliver: kakao`` cron jobs) fails with a descriptive error, and
   in-gateway ``send()`` calls that arrive after the turn's delivery
   slot is spent are held and served on the user's next utterance
   instead of being silently dropped.

Concretely, each inbound webhook holds one ``asyncio.Future`` open (a
"delivery slot") for the chat, and ``send()`` is the only way anything
fulfills it. This plays the same architectural role as LINE's single-use
reply token (see ``plugins/platforms/line/adapter.py``), except Kakao's
reply mechanism *is* the HTTP response itself (or the one-shot callback),
not a second API call.

Security: Kakao does not sign webhook requests (no HMAC / signature
header is provided by the platform). The Open Builder skill-registration
screen lets the skill owner configure an arbitrary custom HTTP header
that is sent with every request; we require that header to carry a
shared secret (``KAKAO_SKILL_SECRET``) and verify it in constant time.

Scope of this initial version: text in, text out. Kakao's SkillResponse
supports rich card/carousel outputs, but ``simpleText`` covers hermes'
actual output (LLM prose) and keeps the surface area small for the first
merge pass. Media sending (images/voice/video) is intentionally left to
the base-class defaults (unimplemented) rather than half-built — this
adapter does not claim capabilities it does not have.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.config import Platform


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Kakao's hard skill-server timeout ("skill timeout: 5sec" per Kakao's
# skill SLA). We budget less than that for our own sync
# attempt so there's slack for JSON (de)serialization and network jitter
# before Kakao's engine gives up on us entirely.
KAKAO_HARD_TIMEOUT_SECONDS = 5.0
# The sync budget must leave room for network latency between Kakao and
# this server on top of our own processing: behind a tunnel (cloudflared/
# ngrok) the round trip alone can eat ~0.5s, and a 4.5s budget then
# produces intermittent "스킬 서버 타임아웃 (1001)" errors on Kakao's side.
DEFAULT_SYNC_TIMEOUT_SECONDS = 4.0

# Kakao's docs disagree with themselves on how long the one-shot
# callbackUrl stays valid: the Open Builder UI and parts of the callback
# guide say "up to 5 minutes", while the same guide's error section says
# an expired token means "1분이내 요청 필요" and Kakao devtalk answers
# state "callbackUrl valid time: 1min". Observed behavior matches the
# 1-minute reading (POSTs after ~1min fail with "Invalid Callback
# token"), so we budget safely under one minute. Answers that miss this
# window are stashed and delivered on the user's next utterance instead
# (see ``_late_answers``).
DEFAULT_CALLBACK_TIMEOUT_SECONDS = 50.0

# SkillResponse limits, per Kakao's "answer JSON format" docs
# (https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format).
KAKAO_SIMPLETEXT_MAX_CHARS = 1000
KAKAO_SAFE_TEXT_CHARS = 900  # conservative chunk size, leaves headroom
KAKAO_MAX_OUTPUTS = 3  # SkillResponse.template.outputs: 1-3 items

# How long a held (undeliverable-at-the-time) answer stays eligible for
# delivery on the user's next utterance before being dropped as stale.
LATE_ANSWER_MAX_AGE_SECONDS = 1800.0
# Joined held answers are capped here; delivery further truncates to the
# 3-bubble SkillResponse limit.
HELD_ANSWER_MAX_CHARS = 6000
# How long the stash continuation keeps waiting for a straggling agent
# answer after the callback window has already been spent.
STASH_WAIT_SECONDS = 600.0
# Minimum gap between the useCallback ack and the callback POST -- a POST
# that races the ack can be rejected as an invalid token.
CALLBACK_SETTLE_SECONDS = 1.0
# The one-shot callbackUrl comes from the request body. Kakao issues them
# on *.kakao.com (observed: bot-api.kakao.com); anything else is treated
# as attacker-controlled and refused (SSRF guard).
CALLBACK_URL_ALLOWED_HOST_SUFFIX = ".kakao.com"

WEBHOOK_BODY_MAX_BYTES = 1_048_576  # 1 MiB -- skill requests are tiny JSON
DEFAULT_WEBHOOK_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/kakao/webhook"
DEFAULT_SECRET_HEADER = "X-Hermes-Kakao-Secret"

DEFAULT_UNAUTHORIZED_TEXT = "이 챗봇을 사용할 권한이 없습니다."
DEFAULT_BAD_REQUEST_TEXT = "요청을 처리할 수 없습니다."
DEFAULT_NO_CALLBACK_TIMEOUT_TEXT = (
    "답변 생성이 예상보다 오래 걸리고 있어요. 잠시 후 다시 시도해 주세요."
)
DEFAULT_LATE_ANSWER_NOTICE_TEXT = (
    "답변을 아직 만들고 있어요. 잠시 후 아래 버튼을 누르거나 아무 메시지나 "
    "보내 주시면 준비된 답변을 보여드릴게요."
)
DEFAULT_CALLBACK_WAITING_TEXT = "🤔 답변을 만들고 있어요. 잠시만 기다려 주세요."
# quickReply that lets the user tap (instead of type) to retrieve a held
# answer -- the tap is sent as a normal utterance, which pops the stash.
RETRIEVE_QUICK_REPLY = {
    "label": "답변 확인",
    "action": "message",
    "messageText": "답변 확인",
}
DEFAULT_INTERRUPTED_TEXT = "작업이 중단되었습니다."
DEFAULT_SUPERSEDED_TEXT = (
    "새 메시지가 도착해 이전 요청은 취소되었습니다."
)


# ---------------------------------------------------------------------------
# Markdown stripping -- Kakao's simpleText is plain text, no Markdown
# rendering. Same approach as the LINE adapter (URL-preserving), kept as
# an independent copy here so this plugin has zero cross-plugin imports.
# ---------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITAL_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_MD_CODE_INLINE_RE = re.compile(r"`([^`]+)`")
_MD_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)


def strip_markdown_preserving_urls(text: str) -> str:
    """Strip Markdown that Kakao's simpleText can't render, keeping URLs usable."""
    if not text:
        return text

    def _unfence(m: re.Match) -> str:
        return m.group(1).rstrip("\n")
    text = _MD_CODE_BLOCK_RE.sub(_unfence, text)
    text = _MD_CODE_INLINE_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITAL_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("• ", text)
    return text


def split_for_kakao(text: str, max_chars: int = KAKAO_SAFE_TEXT_CHARS) -> List[str]:
    """Split ``text`` into Kakao ``simpleText``-sized chunks.

    Returns at most ``KAKAO_MAX_OUTPUTS`` chunks (SkillResponse allows 1-3
    ``outputs``); longer text is truncated with an ellipsis on the final
    chunk since there is no follow-up call to deliver the remainder.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text
    while remaining and len(chunks) < KAKAO_MAX_OUTPUTS:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            remaining = ""
            break
        cut = remaining.rfind("\n\n", 0, max_chars)
        if cut < int(max_chars * 0.5):
            cut = remaining.rfind("\n", 0, max_chars)
        if cut < int(max_chars * 0.5):
            cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        if chunks:
            tail = chunks[-1]
            if len(tail) > max_chars - 1:
                tail = tail[: max_chars - 1]
            chunks[-1] = tail.rstrip() + "…"
        else:
            chunks.append(remaining[: max_chars - 1] + "…")
    return chunks


def build_skill_response(
    text: str, quick_replies: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Build a ``SkillResponse`` (version 2.0) with ``simpleText`` outputs."""
    plain = strip_markdown_preserving_urls(text or "")
    chunks = split_for_kakao(plain) or [""]
    outputs = [{"simpleText": {"text": c[:KAKAO_SIMPLETEXT_MAX_CHARS]}} for c in chunks]
    template: Dict[str, Any] = {"outputs": outputs}
    if quick_replies:
        template["quickReplies"] = quick_replies
    return {"version": "2.0", "template": template}


def build_callback_ack(waiting_text: Optional[str] = None) -> Dict[str, Any]:
    """The immediate reply when handing the turn off to the callback URL.

    Per Kakao's callback spec, on the ``useCallback`` path the ``template``
    field must be omitted entirely. ``data.text`` (optional) customizes the
    interim "thinking" bubble Kakao shows while waiting for the callback.
    """
    ack: Dict[str, Any] = {"version": "2.0", "useCallback": True}
    if waiting_text:
        ack["data"] = {"text": waiting_text}
    return ack


# ---------------------------------------------------------------------------
# Secret-header verification
# ---------------------------------------------------------------------------

def verify_shared_secret(header_value: Optional[str], expected_secret: str) -> bool:
    """Constant-time check of the shared-secret header.

    Kakao provides no request signature (there is no HMAC/signature header
    in the skill request spec) -- this shared-secret header is the only
    verification available, so it must not short-circuit on length or leak
    timing information.
    """
    if not expected_secret or not header_value:
        return False
    try:
        return hmac.compare_digest(str(header_value), str(expected_secret))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Busy-ack bypass -- messages the gateway sends mid-turn (queued/steered/
# interrupted notices) have nowhere to go in this adapter: there is only
# one delivery slot per turn, and burning it on a placeholder means the
# real answer is lost. We swallow them instead of erroring loudly.
# ---------------------------------------------------------------------------

_SYSTEM_BYPASS_PREFIXES = (
    "⚡ Interrupting",
    "⏳ Queued",
    "⏩ Steered",
    "💾",
)


def _is_system_bypass(content: str) -> bool:
    if not content:
        return False
    return any(content.startswith(p) for p in _SYSTEM_BYPASS_PREFIXES)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _csv_set(value: str) -> Set[str]:
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def _truthy_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _float_env_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _redact_user_id(chat_id: str) -> str:
    """Best-effort redaction for botUserKey values in logs."""
    if not chat_id:
        return chat_id
    if len(chat_id) <= 6:
        return "***"
    return f"{chat_id[:3]}...{chat_id[-3:]}"


def _allowed(chat_id: str, *, allow_all: bool, allowed_users: Set[str]) -> bool:
    if allow_all:
        return True
    return bool(chat_id) and chat_id in allowed_users


# ---------------------------------------------------------------------------
# Pending-turn bookkeeping -- the "delivery slot" for one open webhook
# request (or its callback extension).
# ---------------------------------------------------------------------------

@dataclass
class _PendingTurn:
    future: "asyncio.Future[str]"
    callback_url: Optional[str]
    created_at: float


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class KakaoAdapter(BasePlatformAdapter):
    """Kakao i Open Builder skill-server adapter."""

    # This adapter cannot push a message once a turn's delivery slot (HTTP
    # response or one-shot callback) has been used -- there is nowhere to
    # route a later background-completion notification. See module
    # docstring.
    supports_async_delivery = False

    def __init__(self, config, **kwargs):
        platform = Platform("kakao")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        # Credentials
        self.skill_secret = (
            os.getenv("KAKAO_SKILL_SECRET") or extra.get("skill_secret", "")
        )
        self.secret_header = (
            os.getenv("KAKAO_SECRET_HEADER")
            or extra.get("secret_header", DEFAULT_SECRET_HEADER)
        )

        # Webhook server
        self.webhook_host = os.getenv("KAKAO_HOST") or extra.get("host", "0.0.0.0")
        try:
            self.webhook_port = int(
                os.getenv("KAKAO_PORT") or extra.get("port", DEFAULT_WEBHOOK_PORT)
            )
        except (TypeError, ValueError):
            self.webhook_port = DEFAULT_WEBHOOK_PORT
        self.webhook_path = extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        self.public_base_url = (
            os.getenv("KAKAO_PUBLIC_URL") or extra.get("public_url", "") or ""
        ).rstrip("/")
        self.bot_id = os.getenv("KAKAO_BOT_ID") or extra.get("bot_id", "")

        # Allowlist (single list -- Kakao channel chats are always 1:1)
        self.allow_all = _truthy_env(
            "KAKAO_ALLOW_ALL_USERS", bool(extra.get("allow_all_users", False))
        )
        self.allowed_users = _csv_set(
            os.getenv("KAKAO_ALLOWED_USERS", "")
        ) | set(extra.get("allowed_users", []))

        # Timing budgets
        self.sync_timeout = _float_env_or(
            os.getenv("KAKAO_SYNC_TIMEOUT") or extra.get("sync_timeout"),
            DEFAULT_SYNC_TIMEOUT_SECONDS,
        )
        self.callback_timeout = _float_env_or(
            os.getenv("KAKAO_CALLBACK_TIMEOUT") or extra.get("callback_timeout"),
            DEFAULT_CALLBACK_TIMEOUT_SECONDS,
        )
        # However generous the operator is, the sync leg can never exceed
        # Kakao's hard 5s SLA (minus headroom for the response leg).
        self.sync_timeout = min(self.sync_timeout, KAKAO_HARD_TIMEOUT_SECONDS - 0.5)

        # User-overridable copy (env wins over extra, matching every other
        # config knob in this adapter).
        self.unauthorized_text = extra.get("unauthorized_text", DEFAULT_UNAUTHORIZED_TEXT)
        self.bad_request_text = extra.get("bad_request_text", DEFAULT_BAD_REQUEST_TEXT)
        self.no_callback_timeout_text = extra.get(
            "no_callback_timeout_text", DEFAULT_NO_CALLBACK_TIMEOUT_TEXT
        )
        self.superseded_text = extra.get("superseded_text", DEFAULT_SUPERSEDED_TEXT)
        self.interrupted_text = extra.get("interrupted_text", DEFAULT_INTERRUPTED_TEXT)
        self.late_answer_notice_text = (
            os.getenv("KAKAO_LATE_ANSWER_NOTICE_TEXT")
            or extra.get("late_answer_notice_text", DEFAULT_LATE_ANSWER_NOTICE_TEXT)
        )
        self.callback_waiting_text = (
            os.getenv("KAKAO_CALLBACK_WAITING_TEXT")
            or extra.get("callback_waiting_text", DEFAULT_CALLBACK_WAITING_TEXT)
        )

        # Runtime state
        self._app = None
        self._runner = None
        self._site = None
        # One open delivery slot per chat_id (botUserKey).
        self._pending_turns: Dict[str, _PendingTurn] = {}
        # Messages that could not be delivered inside their turn (answer
        # outlived the callback window, or the turn's slot was already
        # spent -- e.g. an approval ack answered the webhook and the agent
        # finished afterwards). Keyed by chat_id as (stashed_at, text);
        # served synchronously on that user's next utterance (which acts
        # as the retrieval tap -- same pattern as the LINE adapter's
        # slow-response postback button).
        self._late_answers: Dict[str, Tuple[float, str]] = {}
        # Strong references to fire-and-forget tasks (callback delivery,
        # stash continuations) so they can't be GC'd mid-flight and can be
        # cancelled on disconnect.
        self._bg_tasks: Set["asyncio.Task"] = set()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.skill_secret:
            self._set_fatal_error(
                "config_missing",
                "KAKAO_SKILL_SECRET must be set (shared secret verifying inbound "
                "skill requests -- Kakao does not sign webhooks)",
                retryable=False,
            )
            return False

        # Prevent two profiles from binding the same secret/webhook twice.
        # ``_acquire_platform_lock`` (base.py) sets a descriptive fatal error
        # itself on conflict, so we just propagate its bool result.
        import hashlib
        sec_hash = hashlib.sha256(self.skill_secret.encode()).hexdigest()[:16]
        if not self._acquire_platform_lock("kakao", sec_hash, "Kakao skill secret"):
            return False

        try:
            from aiohttp import web
        except ImportError:
            self._set_fatal_error(
                "missing_dep",
                "aiohttp is required for the Kakao adapter -- install with `pip install aiohttp`",
                retryable=False,
            )
            return False

        self._app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        self._app.router.add_post(self.webhook_path, self._handle_webhook)
        self._app.router.add_get(f"{self.webhook_path}/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        try:
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self.webhook_host, self.webhook_port)
            await self._site.start()
        except OSError as exc:
            self._set_fatal_error(
                "bind_failed",
                f"Could not bind Kakao webhook on {self.webhook_host}:{self.webhook_port}: {exc}",
                retryable=True,
            )
            return False

        self._mark_connected()
        logger.info(
            "Kakao: webhook listening on %s:%s%s%s",
            self.webhook_host,
            self.webhook_port,
            self.webhook_path,
            f" (public: {self.public_base_url})" if self.public_base_url else "",
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

        # Cancel any turns still waiting on a callback so their tasks don't
        # leak past shutdown.
        for turn in list(self._pending_turns.values()):
            if not turn.future.done():
                turn.future.cancel()
        self._pending_turns.clear()
        self._late_answers.clear()
        for task in list(self._bg_tasks):
            task.cancel()
        self._bg_tasks.clear()

        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                pass
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
        self._app = None

        self._release_platform_lock()

    # ------------------------------------------------------------------
    # Webhook handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request) -> Any:
        from aiohttp import web
        return web.json_response({"status": "ok", "platform": "kakao"})

    async def _handle_webhook(self, request) -> Any:
        from aiohttp import web

        try:
            body = await request.read()
        except Exception as exc:
            logger.debug("Kakao: read failed: %s", exc)
            return web.Response(status=400, text="bad request")
        if len(body) > WEBHOOK_BODY_MAX_BYTES:
            return web.Response(status=413, text="payload too large")

        header_value = request.headers.get(self.secret_header, "")
        if not verify_shared_secret(header_value, self.skill_secret):
            return web.Response(status=401, text="invalid secret")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.Response(status=400, text="bad json")
        if not isinstance(payload, dict):
            return web.Response(status=400, text="bad json")

        return web.json_response(await self._process_skill_request(payload))

    async def _process_skill_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_request = payload.get("userRequest") or {}
        utterance = user_request.get("utterance", "") or ""
        user = user_request.get("user") or {}
        chat_id = str(user.get("id", "") or "")
        callback_url = user_request.get("callbackUrl") or None

        if not chat_id:
            return build_skill_response(self.bad_request_text)

        # Advisory bot-id check. The console URL's bot id and the ids Kakao
        # puts on the wire (payload bot.id, callback-URL bot id) have been
        # observed to differ, so a mismatch warns rather than rejects --
        # the shared-secret header remains the authentication boundary.
        if self.bot_id:
            payload_bot_id = str((payload.get("bot") or {}).get("id") or "")
            if payload_bot_id and payload_bot_id != self.bot_id:
                logger.warning(
                    "Kakao: inbound bot.id %s does not match configured "
                    "KAKAO_BOT_ID %s", payload_bot_id, self.bot_id,
                )

        # No inbound dedup on purpose: unlike LINE (which documents webhook
        # re-delivery and ships a deduplicator), Kakao's skill engine was
        # not observed to re-POST a request -- a timeout simply fails that
        # user turn. Revisit if retries are ever observed in the wild.

        if not _allowed(chat_id, allow_all=self.allow_all, allowed_users=self.allowed_users):
            # Full key on purpose (warning level): operators need it verbatim
            # to build KAKAO_ALLOWED_USERS -- same approach as the LINE
            # adapter's unauthorized-source log. botUserKey is bot-scoped and
            # pseudonymous, so this does not leak a cross-service identity.
            logger.warning(
                "Kakao: rejecting unauthorized user %s -- add this botUserKey "
                "to KAKAO_ALLOWED_USERS to allow them.",
                chat_id,
            )
            return build_skill_response(self.unauthorized_text)

        # A previous turn left an undeliverable message behind (missed
        # callback window / slot already spent); this new utterance acts
        # as the retrieval tap for it.
        late = self._late_answers.pop(chat_id, None)
        if late is not None:
            stashed_at, text = late
            if time.time() - stashed_at <= LATE_ANSWER_MAX_AGE_SECONDS:
                logger.info(
                    "Kakao: delivering held answer to chat %s",
                    _redact_user_id(chat_id),
                )
                return build_skill_response(text)
            # Stale -- drop it and treat this as a normal utterance.

        future: "asyncio.Future[str]" = asyncio.get_running_loop().create_future()
        turn = _PendingTurn(future=future, callback_url=callback_url, created_at=time.time())

        # A second webhook for the same chat_id while a prior one is still
        # open supersedes it -- there's only one delivery slot per chat.
        old = self._pending_turns.get(chat_id)
        if old is not None and not old.future.done():
            old.future.cancel()
        self._pending_turns[chat_id] = turn

        source = self.build_source(
            chat_id=chat_id,
            chat_type="dm",
            user_id=chat_id,
            user_name=chat_id,
            chat_name=chat_id,
        )
        event = MessageEvent(
            text=utterance,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
        )

        # handle_message() spawns background processing and returns
        # quickly; the actual answer arrives later via send(), which
        # resolves ``future``.
        await self.handle_message(event)

        # Budget from request receipt, not from here -- Kakao's 5s clock
        # started when it sent the webhook, and allowlist checks /
        # handle_message dispatch above already spent part of it.
        remaining = max(0.05, self.sync_timeout - (time.time() - turn.created_at))
        try:
            content = await asyncio.wait_for(asyncio.shield(future), timeout=remaining)
        except asyncio.TimeoutError:
            if callback_url:
                self._spawn(
                    self._deliver_via_callback(
                        chat_id, callback_url, future, ack_at=time.time()
                    )
                )
                return build_callback_ack(self.callback_waiting_text)
            logger.warning(
                "Kakao: no callbackUrl on this block and the agent exceeded the "
                "%.1fs sync budget for chat %s -- the eventual answer will be "
                "held for the user's next message. Enable the callback option "
                "on the matched block in the Open Builder console to deliver "
                "slow answers in-turn.",
                self.sync_timeout,
                _redact_user_id(chat_id),
            )
            self._spawn(self._stash_late_answer(chat_id, future))
            return build_skill_response(self.no_callback_timeout_text)
        except asyncio.CancelledError:
            return build_skill_response(self.superseded_text)
        else:
            if self._pending_turns.get(chat_id) is turn:
                self._pending_turns.pop(chat_id, None)
            return build_skill_response(content)

    def _spawn(self, coro) -> "asyncio.Task":
        """create_task with a strong reference (GC guard) and cleanup."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def _deliver_via_callback(
        self,
        chat_id: str,
        callback_url: str,
        future: "asyncio.Future[str]",
        ack_at: float = 0.0,
    ) -> None:
        """Wait within the callback window, then POST the answer once.

        If the agent is still working when the window closes, the one-shot
        callback is spent on a "still working" notice (with a retrieval
        quick-reply button), and the eventual answer is stashed for the
        user's next utterance.
        """
        try:
            content = await asyncio.wait_for(asyncio.shield(future), timeout=self.callback_timeout)
        except asyncio.TimeoutError:
            logger.info(
                "Kakao: agent exceeded the %.0fs callback window for chat %s -- "
                "sending a still-working notice and stashing the answer for "
                "the next utterance.",
                self.callback_timeout,
                _redact_user_id(chat_id),
            )
            self._spawn(self._stash_late_answer(chat_id, future))
            await self._post_callback(
                chat_id,
                callback_url,
                self.late_answer_notice_text,
                ack_at=ack_at,
                quick_replies=[RETRIEVE_QUICK_REPLY],
            )
            return
        except asyncio.CancelledError:
            self._pop_turn_if_current(chat_id, future)
            return

        self._pop_turn_if_current(chat_id, future)
        status = await self._post_callback(chat_id, callback_url, content, ack_at=ack_at)
        if status is None or status >= 400:
            # The token was rejected (expired/invalid) -- don't lose the
            # answer; hand it to the user on their next message.
            self._hold_for_next_utterance(chat_id, content)

    def _pop_turn_if_current(self, chat_id: str, future: "asyncio.Future[str]") -> None:
        """Remove the chat's pending turn only if it is still *this* turn.

        A superseding utterance replaces the slot; an identity-unguarded
        pop from the older turn's cleanup would evict the newer turn.
        """
        cur = self._pending_turns.get(chat_id)
        if cur is not None and cur.future is future:
            self._pending_turns.pop(chat_id, None)

    async def _stash_late_answer(
        self, chat_id: str, future: "asyncio.Future[str]"
    ) -> None:
        """Continuation for answers that missed their delivery window."""
        try:
            content = await asyncio.wait_for(asyncio.shield(future), timeout=STASH_WAIT_SECONDS)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return
        finally:
            self._pop_turn_if_current(chat_id, future)
        self._hold_for_next_utterance(chat_id, content)

    def _hold_for_next_utterance(self, chat_id: str, content: str) -> None:
        """Keep an undeliverable message for this user's next utterance.

        Multiple held messages within the freshness window are joined so a
        follow-up send (e.g. the real answer after an approval ack) does
        not overwrite an earlier one.
        """
        now = time.time()
        # Opportunistic sweep so entries for users who never return don't
        # accumulate forever (cheap: only runs when something is stashed).
        stale = [k for k, (ts, _) in self._late_answers.items() if now - ts > LATE_ANSWER_MAX_AGE_SECONDS]
        for k in stale:
            del self._late_answers[k]
        prev = self._late_answers.get(chat_id)
        if prev is not None:
            content = prev[1] + "\n\n" + content
        self._late_answers[chat_id] = (now, content[:HELD_ANSWER_MAX_CHARS])

    async def _post_callback(
        self,
        chat_id: str,
        callback_url: str,
        content: str,
        ack_at: float = 0.0,
        quick_replies: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        """POST one SkillResponse to the one-shot callbackUrl."""
        # Give Kakao a beat to register the token after our useCallback ack;
        # a POST that races the ack can be rejected as an invalid token.
        if ack_at:
            since_ack = time.time() - ack_at
            if since_ack < CALLBACK_SETTLE_SECONDS:
                await asyncio.sleep(CALLBACK_SETTLE_SECONDS - since_ack)
        try:
            import aiohttp
            from yarl import URL

            # encoded=True: the callbackUrl embeds a one-time token; yarl's
            # default re-quoting can alter percent-encoded characters in it,
            # which Kakao rejects as "Invalid Callback token".
            target = URL(callback_url, encoded=True)
            # SSRF guard: the URL came from the request body. Only POST to
            # Kakao's own callback hosts, over https, and never follow a
            # redirect off them.
            host = target.host or ""
            if target.scheme != "https" or not (
                host == CALLBACK_URL_ALLOWED_HOST_SUFFIX.lstrip(".")
                or host.endswith(CALLBACK_URL_ALLOWED_HOST_SUFFIX)
            ):
                logger.warning(
                    "Kakao: refusing callback POST to non-Kakao URL "
                    "(scheme=%s host=%s) for chat %s",
                    target.scheme,
                    host,
                    _redact_user_id(chat_id),
                )
                return None

            timeout = aiohttp.ClientTimeout(total=15.0)
            body = build_skill_response(content, quick_replies=quick_replies)
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.post(target, json=body, allow_redirects=False) as resp:
                    if resp.status >= 400:
                        # Bounded read: don't materialize an arbitrarily
                        # large error body just to log 200 chars of it.
                        raw = await resp.content.read(4096)
                        text = raw.decode("utf-8", "replace")
                        logger.warning(
                            "Kakao: callback POST failed (%s) for chat %s "
                            "(%.1fs after ack): %s",
                            resp.status,
                            _redact_user_id(chat_id),
                            (time.time() - ack_at) if ack_at else -1.0,
                            text[:200],
                        )
                    return resp.status
        except Exception as exc:
            logger.warning("Kakao: callback POST error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Outbound (fulfills the one open delivery slot for the current turn)
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if _is_system_bypass(content):
            # No channel to deliver a mid-turn busy-ack on -- swallow it
            # rather than spend the one-shot delivery slot on a placeholder.
            return SendResult(success=True, message_id=None)

        turn = self._pending_turns.get(chat_id)
        if turn is None or turn.future.done():
            # The turn's single delivery slot is already spent (e.g. an
            # approval ack answered the webhook synchronously and the agent
            # finished afterwards), or there is no open turn at all. Kakao
            # has no push API, so hold the message and serve it when this
            # user next messages the channel.
            self._hold_for_next_utterance(chat_id, content)
            logger.info(
                "Kakao: no open skill request for chat %s -- holding the "
                "message for delivery on the user's next utterance (Kakao's "
                "request-response model has no push API).",
                _redact_user_id(chat_id),
            )
            return SendResult(success=True, message_id=None)
        turn.future.set_result(content)
        return SendResult(success=True, message_id=chat_id)

    async def interrupt_session_activity(self, session_key: str, chat_id: str) -> None:
        """Resolve the chat's open delivery slot so its webhook/callback
        waiters return promptly instead of dangling through both timeout
        windows (mirrors the LINE adapter's orphan-PENDING resolution)."""
        await super().interrupt_session_activity(session_key, chat_id)
        turn = self._pending_turns.pop(chat_id, None)
        if turn is not None and not turn.future.done():
            turn.future.set_result(self.interrupted_text)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        # No typing-indicator API in the skill-server model.
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Kakao Open Builder channel chats are always 1:1 between the user
        # and the channel -- there is no group/room concept in this model.
        return {"name": chat_id or "", "type": "dm"}

    def format_message(self, content: str) -> str:
        return strip_markdown_preserving_urls(content)


# ---------------------------------------------------------------------------
# Plugin entry-point hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Plugin gate: require the shared secret AND aiohttp at runtime."""
    if not os.getenv("KAKAO_SKILL_SECRET"):
        return False
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(os.getenv("KAKAO_SKILL_SECRET") or extra.get("skill_secret"))


def is_connected(config) -> bool:
    """Surface in ``hermes status`` even before the adapter is instantiated."""
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Auto-seed PlatformConfig.extra from env-only setups.

    Lets ``hermes status`` reflect a Kakao configuration that lives
    entirely in ``.env`` without a ``platforms.kakao`` block in
    ``config.yaml``. Mirrors the LINE plugin's pattern.
    """
    if not os.getenv("KAKAO_SKILL_SECRET"):
        return None
    seeded: Dict[str, Any] = {}
    if os.getenv("KAKAO_PORT"):
        try:
            seeded["port"] = int(os.environ["KAKAO_PORT"])
        except ValueError:
            pass
    if os.getenv("KAKAO_HOST"):
        seeded["host"] = os.environ["KAKAO_HOST"]
    if os.getenv("KAKAO_PUBLIC_URL"):
        seeded["public_url"] = os.environ["KAKAO_PUBLIC_URL"]
    if os.getenv("KAKAO_SECRET_HEADER"):
        seeded["secret_header"] = os.environ["KAKAO_SECRET_HEADER"]
    if os.getenv("KAKAO_BOT_ID"):
        seeded["bot_id"] = os.environ["KAKAO_BOT_ID"]
    return seeded or {}


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process delivery hook -- always fails, on purpose.

    Kakao's skill-server model has no push/notify API a cron job or
    out-of-process ``send_message`` call could use:
    every outbound message must answer a specific inbound webhook request
    that is already closed by the time a detached process could act. This
    hook exists (rather than being omitted) so ``deliver=kakao`` fails
    loudly with an explanation instead of a generic "no live adapter"
    error the operator would have to go dig up the reason for.
    """
    return {
        "error": (
            "Kakao does not support proactive/cron delivery: the Open Builder "
            "skill-server model can only answer a user's own message, within "
            "that message's request window."
        )
    }


def interactive_setup() -> None:
    """Minimal stdin wizard (reached via ``hermes gateway setup`` → Kakao)."""
    print()
    print("Kakao (Open Builder channel chatbot) setup")
    print("-------------------------------------------")
    print("1. Create a KakaoTalk channel and an Open Builder bot at i.kakao.com,")
    print("   connect the bot to the channel.")
    print("2. In Open Builder, create a Skill pointing at this webhook's public")
    print("   URL (<public-url>/kakao/webhook) and add a custom header carrying")
    print("   the same secret you enter below.")
    print("3. Attach the skill to the fallback block (bot response = 'Skill data'),")
    print("   and enable the callback option on that block if you want slow LLM")
    print("   answers to make it through Kakao's 5-second timeout.")
    print()

    try:
        from hermes_cli.config import get_env_var, set_env_var
    except ImportError:
        print("hermes_cli.config not available; set KAKAO_* vars manually in ~/.hermes/.env")
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_var(var) if callable(get_env_var) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            set_env_var(var, value)

    _prompt("KAKAO_SKILL_SECRET", "Shared secret (must match the custom header value in Open Builder)", secret=True)
    _prompt("KAKAO_SECRET_HEADER", f"Secret header name (blank = {DEFAULT_SECRET_HEADER})")
    _prompt("KAKAO_PUBLIC_URL", "Public HTTPS base URL (e.g. a cloudflared/ngrok tunnel)")
    _prompt("KAKAO_ALLOWED_USERS", "Allowed botUserKey values (comma-separated; blank=skip)")
    print(f"Done. Register the skill URL as <public-url>{DEFAULT_WEBHOOK_PATH} in Open Builder.")


def register(ctx) -> None:
    """Plugin entry point -- called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="kakao",
        label="Kakao",
        adapter_factory=lambda cfg: KakaoAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["KAKAO_SKILL_SECRET"],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        standalone_sender_fn=_standalone_send,
        allowed_users_env="KAKAO_ALLOWED_USERS",
        allow_all_env="KAKAO_ALLOW_ALL_USERS",
        # SkillResponse simpleText cap is 1000 chars; we chunk at 900.
        max_message_length=KAKAO_SAFE_TEXT_CHARS,
        emoji="🟡",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via a KakaoTalk channel (Kakao i Open Builder "
            "skill server). This is plain text only -- no Markdown rendering. "
            "Responses are capped at 3 short text bubbles (~900 characters "
            "each), so keep answers concise. You cannot send a message to this "
            "user except as a direct reply to their own message -- there is no "
            "proactive/notify capability on this platform."
        ),
    )
