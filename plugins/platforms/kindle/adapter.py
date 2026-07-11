"""Kindle Scribe platform adapter — DRAFT for review.

Bridges the "Hermes Agents Guide to the Galaxy" Scribe notebook into the Hermes
gateway as a first-class platform, so handwriting/typed notes reach the FULL
desktop Hermes agent (MoA + tools + memory) — exactly like the Telegram/Discord
bots. Modeled on plugins/platforms/sms/adapter.py.

Data flow:
    Kindle browser --(LAN)--> diary bridge (Node) --(localhost HTTP)--> THIS adapter
    THIS adapter --> MessageEvent --> gateway --> agent (MoA + `kindle` toolset)
    agent reply --> adapter.send() --> back to the pending bridge request --> Kindle

The adapter runs a tiny localhost ingest server. The diary bridge POSTs a note to
/ingest and gets the agent's reply back on the same request. Replies are correlated
to requests by `chat_id` (one Kindle = one session).

Env:
  KINDLE_INGEST_HOST      (default 127.0.0.1)  -- bind localhost; the bridge is same-box
  KINDLE_INGEST_PORT      (default 8793)
  KINDLE_INGEST_TOKEN     shared secret the bridge must send (X-Kindle-Token); required
                          unless KINDLE_INSECURE=true
  KINDLE_INSECURE         (true) skip the token — localhost dev only
  KINDLE_ALLOWED_USERS    comma-separated user ids allowed (e.g. "jeff")
  KINDLE_ALLOW_ALL_USERS  (true/false)
  KINDLE_HOME_CHANNEL     chat id for cron/home delivery
  KINDLE_REPLY_TIMEOUT    seconds to wait for the agent (default 360)

STATUS: non-streaming v1 (one request -> one reply). Streaming via send_draft is a
documented TODO below; the diary already streams to the Kindle, so we can add it
once the round-trip works.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_INGEST_HOST = "127.0.0.1"
DEFAULT_INGEST_PORT = 8793
DEFAULT_REPLY_TIMEOUT = 360
MAX_KINDLE_LENGTH = 8000  # e-ink page; plenty for a notebook reply

KINDLE_HOST_CONTEXT = (
    "[Kindle host context: This Kindle is a remote interface to Hermes running on the "
    "gateway host. You may use the host's configured tools, files, databases, and services. "
    "For requests that read, create, save, copy, or change host resources, perform the action "
    "with an available tool and verify its result before claiming success. Never invent a path "
    "or claim a resource was accessed or changed without a successful tool result.]"
)

KINDLE_INTENTS = {
    "summarize": (
        "Summarize the current page or note. Start with the answer in one short "
        "paragraph, then add only the most useful detail."
    ),
    "tasks": (
        "Extract tasks. Group them by owner, due date, and uncertainty. If a task "
        "is inferred from handwriting, label it inferred."
    ),
    "email": (
        "Draft a concise email from the note or marked page. Do not send it. Put "
        "the draft first, then a short note about assumptions."
    ),
    "workpaper": (
        "Create a workpaper-ready note: facts, evidence, open items, risks, and "
        "next action. Keep amounts, dates, and names exact."
    ),
}


def _clean_tag(value: Any) -> str:
    tag = str(value or "").strip().lstrip("#").lower()
    if not tag or len(tag) > 32:
        return ""
    return tag if all(ch.isalnum() or ch in "-_" for ch in tag) else ""


def _bridge_context(body: Dict[str, Any]) -> str:
    """Convert optional companion-bridge metadata into model-visible context.

    The adapter remains transport-only: the bridge owns OCR/UI/progress, and the
    agent owns tools. These hints make the contract explicit when the bridge has
    already classified the note or the visible artifact being annotated.
    """

    lines: list[str] = []
    intent = str(body.get("intent") or "").strip().lower()
    if intent in KINDLE_INTENTS:
        lines.append(f"Intent: {intent}. {KINDLE_INTENTS[intent]}")

    raw_tags = body.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [item.strip() for item in raw_tags.split(",")]
    if isinstance(raw_tags, list):
        tags = []
        seen = set()
        for raw in raw_tags:
            tag = _clean_tag(raw)
            if tag and tag not in seen:
                tags.append(tag)
                seen.add(tag)
        if tags:
            lines.append("Notebook tags: " + ", ".join(f"#{tag}" for tag in tags[:12]) + ".")

    source = str(body.get("source") or "").strip().lower()
    artifact_type = str(body.get("artifact_type") or "").strip().lower()
    if source in {"live-page", "artifact", "html"} or artifact_type == "html":
        lines.append(
            "Artifact display: the Kindle user is looking at a rendered HTML artifact. "
            "If you create or revise HTML, use the configured live-page/artifact tools "
            "so the visible HTML replaces the old page instead of merely describing the change."
        )

    raw_ocr = str(body.get("ocr_raw") or body.get("raw_transcription") or "").strip()
    cleaned_ocr = str(body.get("ocr_cleaned") or body.get("cleaned_transcription") or "").strip()
    if raw_ocr and cleaned_ocr and raw_ocr != cleaned_ocr:
        lines.append(
            "OCR uncertainty: raw handwriting transcription was "
            f"{raw_ocr!r}; cleaned transcription was {cleaned_ocr!r}. If a name, date, "
            "dollar amount, or command depends on the difference, state the uncertainty "
            "and likely alternatives."
        )
    elif raw_ocr:
        lines.append(
            "OCR uncertainty: handwriting may still contain ambiguous names, dates, "
            "dollar amounts, or arrows. Preserve uncertainty instead of over-committing."
        )

    if not lines:
        return ""
    return "[Kindle bridge context]\n" + "\n".join(lines) + "\n[/Kindle bridge context]"


def check_kindle_requirements() -> bool:
    """Adapter needs aiohttp for its ingest server."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return True


class KindleAdapter(BasePlatformAdapter):
    """Kindle Scribe <-> Hermes gateway adapter.

    Uses the dynamic Platform member path — Platform("kindle") — the same way
    the other plugin platforms (irc, line, ntfy) do, so no core enum edit is needed.
    """

    MAX_MESSAGE_LENGTH = MAX_KINDLE_LENGTH

    def __init__(self, config: PlatformConfig):
        platform = Platform("kindle")
        super().__init__(config=config, platform=platform)
        self._host: str = os.getenv("KINDLE_INGEST_HOST", DEFAULT_INGEST_HOST)
        self._port: int = int(os.getenv("KINDLE_INGEST_PORT", str(DEFAULT_INGEST_PORT)))
        self._token: str = os.getenv("KINDLE_INGEST_TOKEN", "").strip()
        self._insecure: bool = os.getenv("KINDLE_INSECURE", "").lower() == "true"
        self._reply_timeout: float = float(
            os.getenv("KINDLE_REPLY_TIMEOUT", str(DEFAULT_REPLY_TIMEOUT))
        )
        self._runner = None
        # chat_id -> asyncio.Future that adapter.send() fulfills with the reply text.
        self._pending: Dict[str, "asyncio.Future[str]"] = {}

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        from aiohttp import web

        if not self._token and not self._insecure:
            msg = (
                "[kindle] Refusing to start: KINDLE_INGEST_TOKEN not set. Set a shared "
                "secret the diary bridge presents, or KINDLE_INSECURE=true for localhost dev."
            )
            logger.error(msg)
            self._set_fatal_error("kindle_missing_token", msg, retryable=False)
            return False

        app = web.Application()
        app.router.add_post("/ingest", self._handle_ingest)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._running = True
        logger.info("[kindle] ingest server listening on %s:%d", self._host, self._port)
        return True

    async def disconnect(self) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._running = False
        logger.info("[kindle] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Deliver the agent's reply back to the pending ingest request."""
        fut = self._pending.get(chat_id)
        if fut is not None and not fut.done():
            # The gateway may call send() for transient streaming previews
            # (for example ``CLIENT ▉``) before the completed tool-assisted
            # answer. Final user-visible delivery is explicitly marked with
            # metadata.notify=True by BasePlatformAdapter. A request/response
            # bridge must keep waiting for that final send; resolving the
            # Future on a preview truncates every streamed or multi-turn reply.
            if not (metadata and metadata.get("notify") is True):
                # This request/response adapter does not display previews. A
                # successful result (even without a message id) tells the stream
                # consumer that this prefix was delivered and makes its fallback
                # final send contain only the unseen suffix. Explicitly decline
                # the preview so the consumer keeps the complete final response.
                return SendResult(success=False, error="streaming preview not supported")
            fut.set_result(content)
            return SendResult(success=True, message_id=chat_id)
        # No waiter (timed out, or a cron/home push). Nothing to deliver to.
        logger.warning("[kindle] send() for %s had no pending request", chat_id)
        return SendResult(success=False, error="no pending Kindle request")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    # ------------------------------------------------------------------
    # Ingest: the diary bridge POSTs a note here and awaits the reply
    # ------------------------------------------------------------------

    async def _handle_ingest(self, request) -> "aiohttp.web.Response":
        from aiohttp import web

        if not self._insecure:
            if request.headers.get("X-Kindle-Token", "") != self._token:
                return web.json_response({"error": "unauthorized"}, status=401)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)

        text = (body.get("text") or "").strip()
        # The diary already transcribes handwriting via the vision model; it sends
        # the transcription as `text`. `image_text` is an optional extra hint.
        image_text = (body.get("image_text") or "").strip()
        if image_text and image_text not in text:
            text = (text + "\n\n" + image_text).strip() if text else image_text
        user_id = (body.get("user") or "kindle").strip()
        chat_id = (body.get("chat_id") or user_id).strip()
        message_id = (body.get("message_id") or "").strip()

        if not text:
            return web.json_response({"error": "empty note"}, status=400)

        # The HTTP response is the delivery channel for exactly one agent turn.
        # BasePlatformAdapter queues a later message for an already-active chat,
        # so replacing this Future would route turn A's reply into request B and
        # strand request A. Make the synchronous bridge's concurrency contract
        # explicit: callers retry after the current turn finishes.
        pending = self._pending.get(chat_id)
        if pending is not None and not pending.done():
            return web.json_response(
                {"error": "a request for this chat is already in progress"},
                status=409,
            )

        source = self.build_source(
            chat_id=chat_id,
            chat_name="Kindle Scribe",
            chat_type="dm",
            user_id=user_id,
            user_name=user_id,
        )
        bridge_context = _bridge_context(body)
        context_parts = [KINDLE_HOST_CONTEXT]
        if bridge_context:
            context_parts.append(bridge_context)
        context_text = "\n\n".join(context_parts)

        event = MessageEvent(
            text=f"{context_text}\n\n{text}",
            message_type=MessageType.TEXT,
            source=source,
            raw_message=body,
            message_id=message_id,
        )

        loop = asyncio.get_event_loop()
        fut: "asyncio.Future[str]" = loop.create_future()
        self._pending[chat_id] = fut

        # Dispatch to the gateway (auth + session handled by base.handle_message,
        # which enforces KINDLE_ALLOWED_USERS). This spawns the agent.
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        try:
            reply = await asyncio.wait_for(fut, timeout=self._reply_timeout)
            return web.json_response({"reply": reply})
        except asyncio.TimeoutError:
            return web.json_response({"error": "agent timed out"}, status=504)
        except asyncio.CancelledError:
            return web.json_response({"error": "cancelled"}, status=503)
        finally:
            # Only remove our own waiter. This identity guard keeps cleanup
            # correct if the lifecycle changes to queues or retries later.
            if self._pending.get(chat_id) is fut:
                self._pending.pop(chat_id, None)


# ──────────────────────────────────────────────────────────────────────────
# Plugin glue (mirrors sms/adapter.py)
# ──────────────────────────────────────────────────────────────────────────


def _is_connected(config) -> bool:
    """Connected once the ingest server is meant to run. Presence of a token
    (or explicit insecure) is the minimal readiness signal."""
    return bool(os.getenv("KINDLE_INGEST_TOKEN") or os.getenv("KINDLE_INSECURE"))


def _build_adapter(config):
    return KindleAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="kindle",
        label="Kindle Scribe",
        adapter_factory=_build_adapter,
        check_fn=check_kindle_requirements,
        is_connected=_is_connected,
        required_env=[],  # token optional in dev; enforced in connect()
        install_hint="pip install aiohttp",
        allowed_users_env="KINDLE_ALLOWED_USERS",
        allow_all_env="KINDLE_ALLOW_ALL_USERS",
        cron_deliver_env_var="KINDLE_HOME_CHANNEL",
        max_message_length=MAX_KINDLE_LENGTH,
        pii_safe=False,  # this channel CAN reach firm tools — treat as sensitive
        emoji="✍️",  # ✍️
        allow_update_command=True,
    )


# TODO (session): streaming. Implement supports_draft_streaming()->True and
# send_draft(); replace the Future with an asyncio.Queue, have _handle_ingest
# return a StreamResponse that drains the queue, and have the diary bridge proxy
# that stream into its existing Kindle streaming protocol.
