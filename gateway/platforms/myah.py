"""
Myah web platform adapter.

Routes messages through the gateway's _handle_message() pipeline (unlike
the API server which bypasses it), giving Myah access to all 30 slash
commands, agent caching, and session management.

Registers HTTP endpoints on the shared aiohttp Application created by
the API server adapter.

Endpoints:
    POST /myah/v1/message          — dispatch a message or slash command
    GET  /myah/v1/events/{stream_id} — SSE event stream
    POST /myah/v1/confirm/{stream_id} — resolve pending approval
    GET  /myah/health              — health check

Requires: aiohttp (provided by gateway dependencies)
"""

import asyncio
import hmac
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]



from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.config import Platform, PlatformConfig

# ── Myah: attachments support ──────────────────────────────────────────────
import mimetypes as _myah_mimetypes
import os as _myah_os
from pathlib import Path as _myah_Path
try:
    import aiohttp as _myah_aiohttp
    _MYAH_AIOHTTP_AVAILABLE = True
except ImportError:
    _myah_aiohttp = None  # type: ignore[assignment]
    _MYAH_AIOHTTP_AVAILABLE = False
from gateway.platforms.base import (
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)

_MYAH_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # Myah: per-attachment cap (defense-in-depth)
_MYAH_PLATFORM_BASE_URL = _myah_os.environ.get('MYAH_PLATFORM_BASE_URL')  # Myah: platform URL
_MYAH_PLATFORM_BEARER = _myah_os.environ.get('MYAH_PLATFORM_BEARER')      # Myah: shared bearer
# ────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

_MAX_CONCURRENT_STREAMS = 20
_STREAM_TTL = 600  # seconds — orphaned streams cleaned up after this
_KEEPALIVE_INTERVAL = 30  # seconds between SSE keepalive comments


def check_myah_requirements() -> bool:
    """Check if Myah adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


# ── Myah: mime/extension helper ────────────────────────────────────────────
def _myah_ext(mime: str, filename: str, default: str) -> str:
    """Return a file extension (with leading dot) from MIME, filename, or default."""
    if filename and '.' in filename:
        ext = '.' + filename.rsplit('.', 1)[1].lower()
        if 2 <= len(ext) <= 6:
            return ext
    guessed = _myah_mimetypes.guess_extension(mime) if mime else None
    return guessed or default
# ────────────────────────────────────────────────────────────────────────────


class MyahAdapter(BasePlatformAdapter):
    """
    Gateway platform adapter for the Myah web frontend.

    Messages flow through the gateway's full _handle_message() pipeline,
    which provides slash command dispatch, session management, agent
    caching, voice transcription, image analysis — everything that
    Telegram/Discord/Slack adapters get automatically.

    Responses are delivered via SSE streams. Each POST /myah/v1/message
    returns a stream_id; the frontend subscribes to
    GET /myah/v1/events/{stream_id} for real-time events.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MYAH)
        self._auth_key: str = (config.extra or {}).get("auth_key", "")

        # ── Stream state ────────────────────────────────────────────────
        # stream_id → asyncio.Queue of SSE event dicts (None = sentinel)
        self._streams: Dict[str, asyncio.Queue] = {}
        # stream_id → creation timestamp (for TTL sweep)
        self._streams_created: Dict[str, float] = {}

        # ── Dual session mapping (Fix 1) ────────────────────────────────
        # The gateway calls adapter.send(chat_id=source.chat_id) where
        # chat_id is the RAW session_id from the source.  But the
        # approval system uses the FULL session_key (like
        # "agent:main:myah:dm:{session_id}").  We maintain two maps:
        #
        #   _session_streams  : session_key → stream_id  (for approvals)
        #   _chat_id_streams  : raw chat_id → stream_id  (for send/send_typing)
        self._session_streams: Dict[str, str] = {}
        self._chat_id_streams: Dict[str, str] = {}

        # stream_id → session_key (reverse lookup for confirm endpoint)
        self._stream_sessions: Dict[str, str] = {}

        # Pending secret captures: stream_id → { event: threading.Event, result: dict }
        self._pending_secrets: Dict[str, Dict] = {}

        # ── Thread safety (Fix 2) ──────────────────────────────────────
        # Captured in connect() so callbacks from the agent worker thread
        # can safely push events to asyncio.Queue via call_soon_threadsafe.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # ── Route registration state ──────────────────────────────────
        self._routes_registered = False
        # Register a pre-setup hook so our routes are added to the shared
        # aiohttp app BEFORE the API server calls runner.setup() (which
        # freezes the router and rejects new route additions).
        from gateway.platforms.api_server import register_pre_setup_hook
        register_pre_setup_hook(self._register_routes_on_app)

    # ── Auth ────────────────────────────────────────────────────────────

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """Validate Bearer token. Returns None if OK, 401 response on failure."""
        if not self._auth_key:
            return None  # No key configured — allow all

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._auth_key):
                return None

        return web.json_response(
            {"error": "Invalid or missing auth token"},
            status=401,
        )

    # ── SSE helpers ─────────────────────────────────────────────────────

    def _push_event(self, stream_id: str, event: Dict[str, Any]) -> None:
        """Thread-safe push of an event dict to a stream's queue.

        Uses call_soon_threadsafe (Fix 2) because the agent runs in a
        worker thread (via run_in_executor) and callbacks fire from
        that thread.  asyncio.Queue is NOT thread-safe for cross-thread
        put_nowait calls.
        """
        q = self._streams.get(stream_id)
        if q is None:
            return
        try:
            self._loop.call_soon_threadsafe(q.put_nowait, event)
        except RuntimeError:
            pass  # Event loop closed

    def _push_event_sync(self, stream_id: str, event: Dict[str, Any]) -> None:
        """Direct push — only safe from the event loop thread."""
        q = self._streams.get(stream_id)
        if q is None:
            return
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

    # ── HTTP endpoint handlers ──────────────────────────────────────────

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /myah/health — health check."""
        return web.json_response({
            "status": "ok" if self._running else "disconnected",
            "platform": "myah",
            "streams_active": len(self._streams),
        })

    async def _handle_message_endpoint(self, request: "web.Request") -> "web.Response":
        """POST /myah/v1/message — dispatch a message or slash command.

        Returns 202 with {stream_id} immediately.  The frontend subscribes
        to /myah/v1/events/{stream_id} for the response.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Enforce concurrency limit
        if len(self._streams) >= _MAX_CONCURRENT_STREAMS:
            return web.json_response(
                {"error": f"Too many concurrent streams (max {_MAX_CONCURRENT_STREAMS})"},
                status=429,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        message = (body.get("message") or "").strip()
        if not message:
            return web.json_response({"error": "Missing 'message' field"}, status=400)

        session_id = (body.get("session_id") or "").strip() or str(uuid.uuid4())
        user_name = body.get("user_name")
        user_id = body.get("user_id")
        chat_name = body.get("chat_name")

        # ── Myah: one-shot session-scoped model override (T3-932) ────
        # If the client supplies a 'model' field, apply it via
        # runner.set_session_override(...) BEFORE dispatching the message
        # so the gateway picks it up when constructing/resolving the agent.
        # This is the inline equivalent of calling
        # PUT /myah/api/sessions/{id}/model, useful for
        # "regenerate with different model" flows.
        _override_model = (body.get("model") or "").strip()
        # Optional 'provider' tag from the platform — when present it pins
        # the target provider so switch_model skips auto-detect. Needed for
        # OAuth-only providers (openai-codex, anthropic-claude-code) where
        # the env-var heuristic PROVIDER_REGISTRY uses would otherwise fall
        # back to OpenRouter on the very first message of a new chat.
        _override_provider = (body.get("provider") or "").strip()
        # ────────────────────────────────────────────────────────────

        # Create the SSE stream
        stream_id = f"myah_{uuid.uuid4().hex}"
        q: asyncio.Queue = asyncio.Queue()
        self._streams[stream_id] = q
        self._streams_created[stream_id] = time.time()

        # Build source and compute session key BEFORE spawning the task
        source = self.build_source(
            chat_id=session_id,
            chat_name=chat_name,
            chat_type="dm",
            user_id=user_id,
            user_name=user_name,
        )

        # Compute the full session key the same way the gateway does
        from gateway.session import build_session_key
        session_key = build_session_key(
            source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

        # ── Myah: apply one-shot model override if present ───────────
        runner = self.gateway_runner
        if _override_model and runner is not None:
            try:
                from hermes_cli.model_switch import switch_model
                # Load current config for switch_model context
                import yaml as _yaml
                from hermes_constants import get_hermes_home as _ghm
                _cfg_path = _ghm() / "config.yaml"
                _cfg = {}
                if _cfg_path.exists():
                    try:
                        _cfg = _yaml.safe_load(_cfg_path.read_text()) or {}
                    except Exception:
                        _cfg = {}
                _mc = _cfg.get("model", {}) if isinstance(_cfg.get("model"), dict) else {}
                _current_model = _mc.get("default") or (_cfg.get("model") if isinstance(_cfg.get("model"), str) else "")
                _current_provider = _mc.get("provider", "") or "openrouter"
                _current_base_url = _mc.get("base_url", "") or ""
                # Layer current session override on top if present
                _existing_override = runner.get_session_override(session_key) or {}
                if _existing_override:
                    _current_model = _existing_override.get('model', _current_model)
                    _current_provider = _existing_override.get('provider', _current_provider)
                    _current_base_url = _existing_override.get('base_url', _current_base_url)

                _result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: switch_model(
                        raw_input=_override_model,
                        current_provider=_current_provider,
                        current_model=_current_model,
                        current_base_url=_current_base_url,
                        current_api_key="",
                        is_global=False,
                        explicit_provider=_override_provider,  # Myah: pin provider when platform supplies it
                        user_providers=_cfg.get("providers"),
                        custom_providers=_cfg.get("custom_providers"),
                    ),
                )
                if getattr(_result, "success", False):
                    # set_session_override evicts the cached agent atomically.
                    runner.set_session_override(session_key, {
                        "model": _result.new_model,
                        "provider": _result.target_provider,
                        "api_key": getattr(_result, "api_key", "") or "",
                        "base_url": getattr(_result, "base_url", "") or "",
                        "api_mode": getattr(_result, "api_mode", "") or "",
                    })
                else:
                    logger.warning(
                        "[myah] one-shot model override failed: %s",
                        getattr(_result, "error_message", "unknown"),
                    )
            except Exception:
                logger.exception("[myah] one-shot model override error")
        # ────────────────────────────────────────────────────────────

        # Dual mapping (Fix 1): map both the raw chat_id and full session_key
        self._chat_id_streams[session_id] = stream_id
        self._session_streams[session_key] = stream_id
        self._stream_sessions[stream_id] = session_key

        # ── Myah: media attachments ingestion ──────────────────────────────
        attachments = body.get('attachments') or []
        _myah_media_urls: list = []
        _myah_media_types: list = []
        _myah_msg_type = MessageType.TEXT  # will be upgraded by mime routing below

        if attachments and not (_MYAH_PLATFORM_BASE_URL and _MYAH_PLATFORM_BEARER):
            return web.json_response(
                {'error': 'Adapter missing MYAH_PLATFORM_BASE_URL / MYAH_PLATFORM_BEARER env'},
                status=500,
            )

        if attachments:
            async def _fetch_one(att: dict) -> tuple:
                file_id = att.get('file_id')
                filename = att.get('filename') or 'attachment'
                declared_mime = (att.get('mime_type') or 'application/octet-stream').lower()
                declared_size = int(att.get('size') or 0)
                if not file_id:
                    raise ValueError(f"Attachment '{filename}' missing file_id")
                if declared_size > _MYAH_MAX_ATTACHMENT_BYTES:
                    raise ValueError(
                        f'{filename} exceeds {_MYAH_MAX_ATTACHMENT_BYTES // (1024*1024)} MB limit'
                    )
                fetch_url = (
                    f"{_MYAH_PLATFORM_BASE_URL.rstrip('/')}"
                    f"/api/v1/files/{file_id}/content"
                )
                timeout = _myah_aiohttp.ClientTimeout(total=30)
                async with _myah_aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.get(
                        fetch_url,
                        headers={'Authorization': f'Bearer {_MYAH_PLATFORM_BEARER}'},
                    ) as r:
                        if r.status != 200:
                            raise ValueError(
                                f'Platform returned {r.status} for {filename}'
                            )
                        raw = await r.read()
                if len(raw) > _MYAH_MAX_ATTACHMENT_BYTES:
                    raise ValueError(
                        f'{filename} body exceeds {_MYAH_MAX_ATTACHMENT_BYTES // (1024*1024)} MB'
                    )
                return raw, declared_mime, filename

            try:
                fetched = await asyncio.gather(
                    *(_fetch_one(a) for a in attachments), return_exceptions=False,
                )
            except Exception as _att_err:
                return web.json_response(
                    {'error': f'Attachment fetch failed: {_att_err}'},
                    status=502,
                )

            for _raw, _mime, _fname in fetched:
                if _mime.startswith('image/'):
                    _ext = _myah_ext(_mime, _fname, '.jpg')
                    _cached = cache_image_from_bytes(_raw, ext=_ext)
                    if _myah_msg_type == MessageType.TEXT:
                        _myah_msg_type = MessageType.PHOTO
                elif _mime.startswith('audio/'):
                    _ext = _myah_ext(_mime, _fname, '.ogg')
                    _cached = cache_audio_from_bytes(_raw, ext=_ext)
                    if _myah_msg_type == MessageType.TEXT:
                        _myah_msg_type = MessageType.VOICE
                else:
                    _cached = cache_document_from_bytes(_raw, _fname)
                    if _myah_msg_type == MessageType.TEXT:
                        _myah_msg_type = MessageType.DOCUMENT
                _myah_media_urls.append(_cached)
                _myah_media_types.append(_mime)
        # ────────────────────────────────────────────────────────────────────

        # Build the message event
        msg_type = MessageType.COMMAND if message.startswith('/') else _myah_msg_type  # Myah: upgraded by attachments
        event = MessageEvent(
            text=message,
            message_type=msg_type,
            source=source,
            message_id=stream_id,
            media_urls=_myah_media_urls,    # Myah: propagate attachments
            media_types=_myah_media_types,  # Myah: propagate attachments
        )

        # Dispatch in background — the gateway's handle_message spawns its
        # own background task, so we wrap to capture completion/failure.
        task = asyncio.create_task(self._dispatch_message(
            event, stream_id, session_id, session_key,
        ))
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {"stream_id": stream_id, "session_id": session_id},
            status=202,
        )

    async def _dispatch_message(
        self,
        event: MessageEvent,
        stream_id: str,
        chat_id: str,
        session_key: str,
    ) -> None:
        """Dispatch a message through the gateway pipeline and emit run events.

        The gateway's _handle_message() returns the final response text (or
        None if the adapter already sent it via send()).  We emit a
        run.completed event when done, or run.failed on error.
        """
        try:
            if not self._message_handler:
                self._push_event_sync(stream_id, {
                    "event": "run.failed",
                    "stream_id": stream_id,
                    "run_id": stream_id,
                    "timestamp": time.time(),
                    "error": "No message handler registered (gateway not ready)",
                })
                return

            # The gateway's handle_message() calls _process_message_background
            # which calls _message_handler (the GatewayRunner._handle_message).
            # That method returns the final response OR None if already sent.
            #
            # BUT: BasePlatformAdapter.handle_message() is what we should call
            # because it manages session locking, interrupt events, and pending
            # messages.  It spawns its own background task via
            # _process_message_background which calls _message_handler and then
            # adapter.send() with the response.
            #
            # So we call handle_message() and let the base class manage
            # the lifecycle.  Our send() method pushes events to the SSE stream.
            await self.handle_message(event)

            # Wait briefly for the background processing to start and complete.
            # The base handle_message() spawns a background task — we need to
            # let it finish before emitting run.completed.  We do this by
            # watching the _active_sessions dict: the session is removed when
            # processing completes.
            from gateway.session import build_session_key as _bsk
            _sk = _bsk(
                event.source,
                group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
                thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            )
            for _ in range(6000):  # Up to 10 minutes (100ms intervals)
                if _sk not in self._active_sessions:
                    break
                await asyncio.sleep(0.1)

        except Exception as exc:
            logger.exception("[myah] dispatch failed for stream %s", stream_id)
            self._push_event_sync(stream_id, {
                "event": "run.failed",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "error": str(exc),
            })
        finally:
            # Emit run.completed if no explicit failure was sent
            q = self._streams.get(stream_id)
            if q is not None:
                # Check if run.completed or run.failed was already emitted
                # by looking at stream state.  If the queue still exists and
                # we haven't sent a terminal event, send run.completed now.

                # ── Myah: per-message model attribution (T3-932) ──
                # Read the cached agent's model + provider so the
                # frontend can show "answered by X via Y" per message.
                _attribution_model = ""
                _attribution_provider = ""
                try:
                    runner = self.gateway_runner
                    if runner is not None:
                        attribution = runner.get_cached_agent_attribution(session_key)
                        if attribution is not None:
                            _attribution_model = attribution.get("model", "") or ""
                            _attribution_provider = attribution.get("provider", "") or ""
                        # Fallback to session override if cache was evicted mid-flight
                        if not _attribution_model:
                            _override = runner.get_session_override(session_key) or {}
                            _attribution_model = _override.get("model", "")
                            _attribution_provider = _override.get("provider", "")
                except Exception:
                    logger.debug("[myah] model attribution lookup failed", exc_info=True)
                # ────────────────────────────────────────────────
                self._push_event_sync(stream_id, {
                    "event": "run.completed",
                    "stream_id": stream_id,
                    "run_id": stream_id,
                    "timestamp": time.time(),
                    "model": _attribution_model,       # Myah: per-message attribution
                    "provider": _attribution_provider,  # Myah: per-message attribution
                })
                # Sentinel to close the SSE stream
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass

            # Unblock any pending secret capture (let callback thread handle cleanup)
            pending_secret = self._pending_secrets.get(stream_id)
            if pending_secret:
                pending_secret['event'].set()

            # Clean up dual mappings (Fix 1)
            self._chat_id_streams.pop(chat_id, None)
            self._session_streams.pop(session_key, None)
            self._stream_sessions.pop(stream_id, None)

    async def _handle_events_endpoint(self, request: "web.Request") -> "web.StreamResponse":
        """GET /myah/v1/events/{stream_id} — SSE event stream."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        stream_id = request.match_info["stream_id"]

        # Allow subscribing slightly before the stream is registered
        for _ in range(20):
            if stream_id in self._streams:
                break
            await asyncio.sleep(0.05)
        else:
            return web.json_response(
                {"error": f"Stream not found: {stream_id}"},
                status=404,
            )

        q = self._streams[stream_id]

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=_KEEPALIVE_INTERVAL)
                except asyncio.TimeoutError:
                    # Send keepalive comment to prevent connection timeout
                    await response.write(b": keepalive\n\n")
                    continue

                if event is None:
                    # Stream finished
                    await response.write(b": stream closed\n\n")
                    break

                payload = f"data: {json.dumps(event)}\n\n"
                await response.write(payload.encode())
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            logger.debug("[myah] SSE client disconnected: %s", stream_id)
        finally:
            # Clean up the stream
            self._streams.pop(stream_id, None)
            self._streams_created.pop(stream_id, None)

        return response

    async def _handle_confirm_endpoint(self, request: "web.Request") -> "web.Response":
        """POST /myah/v1/confirm/{stream_id} — resolve a pending approval.

        Body shape (Bug B fix — accept both queues):

        * Modern action confirmation (cron creation, plugin install, etc.):
          ``{"confirmation_id": "<uuid>", "choice": "approve|approve_session|deny"}``.
          Routes to ``resolve_action_confirmation`` (`_action_queues`).

        * Legacy terminal-command approval:
          ``{"choice": "approve|approve_session|deny"}``.  Routes to
          ``resolve_gateway_approval`` (`_gateway_queues`) using the
          session_key bound to the stream — preserved for backwards
          compatibility while the legacy approval flow is still
          referenced by ``send_exec_approval``.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        stream_id = request.match_info["stream_id"]
        session_key = self._stream_sessions.get(stream_id)
        if not session_key:
            return web.json_response(
                {"error": f"No active stream or session for stream_id={stream_id}"},
                status=404,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        choice = body.get("choice", "deny")
        if choice not in ("approve", "approve_session", "deny"):
            return web.json_response(
                {"error": "choice must be 'approve', 'approve_session', or 'deny'"},
                status=400,
            )

        # ── Myah: Bug B — dispatch on confirmation_id ───────────
        confirmation_id = body.get("confirmation_id")
        if confirmation_id:
            from tools.approval import resolve_action_confirmation

            ok = resolve_action_confirmation(confirmation_id, choice)
            if not ok:
                return web.json_response(
                    {
                        "error": (
                            f"No pending action confirmation matching "
                            f"confirmation_id={confirmation_id!r}"
                        )
                    },
                    status=404,
                )
            return web.json_response({"ok": True, "resolved": 1})
        # ────────────────────────────────────────────────────────

        from tools.approval import (
            resolve_gateway_approval,
            resolve_action_confirmation_by_session,
        )

        resolved = resolve_gateway_approval(session_key, choice)
        # ── Myah: fall back to action queue when frontend POSTs without confirmation_id ──
        # The Myah frontend's ``ConfirmationCard.svelte`` posts ``{run_id, choice}``
        # — it doesn't yet send ``confirmation_id``.  When the legacy queue
        # has nothing for this session_key (the common case for cron approvals),
        # resolve the oldest pending action confirmation belonging to the same
        # session.  Mirrors the dual-queue resolution in ``/approve`` /
        # ``/deny`` slash commands so HTTP and chat paths behave the same.
        if resolved == 0:
            resolved = resolve_action_confirmation_by_session(session_key, choice)
        # ────────────────────────────────────────────────────────────────────────────────
        if resolved == 0:
            return web.json_response(
                {"error": "No pending confirmation to resolve"},
                status=404,
            )

        return web.json_response({"ok": True, "resolved": resolved})

    def _secret_capture_callback(
        self, var_name: str, prompt: str, metadata=None, stream_id: str = '',
    ) -> dict:
        """Prompt the user for a secret via inline SSE card.

        Called from the agent worker thread.  Blocks until the user submits
        the value via POST /myah/v1/secret/{stream_id}, or until timeout.
        """
        import threading
        if not stream_id:
            return {
                'success': True,
                'skipped': True,
                'stored_as': var_name,
                'validated': False,
                'message': 'No stream for secret capture',
            }

        event = threading.Event()
        self._pending_secrets[stream_id] = {
            'event': event,
            'var_name': var_name,
            'result': None,
        }

        # Emit SSE event to frontend (thread-safe — we're in agent thread)
        meta = metadata or {}
        self._push_event(stream_id, {
            'event': 'secret.required',
            'stream_id': stream_id,
            'run_id': stream_id,
            'timestamp': time.time(),
            'var_name': var_name,
            'prompt': prompt,
            'help': meta.get('help', ''),
            'skill_name': meta.get('skill_name', ''),
        })

        # Block agent thread (same pattern as approval system)
        resolved = event.wait(timeout=120)

        pending = self._pending_secrets.pop(stream_id, None)
        if not resolved or not pending or not pending.get('result'):
            # Timeout or cancelled
            self._push_event(stream_id, {
                'event': 'secret.resolved',
                'stream_id': stream_id,
                'run_id': stream_id,
                'timestamp': time.time(),
                'var_name': var_name,
                'status': 'timeout',
            })
            return {
                'success': True,
                'skipped': True,
                'stored_as': var_name,
                'validated': False,
                'message': 'Secret setup timed out.',
            }

        result = pending['result']

        # Emit resolved event
        self._push_event(stream_id, {
            'event': 'secret.resolved',
            'stream_id': stream_id,
            'run_id': stream_id,
            'timestamp': time.time(),
            'var_name': var_name,
            'status': 'stored',
        })

        return result

    async def _handle_secret_endpoint(self, request: 'web.Request') -> 'web.Response':
        """POST /myah/v1/secret/{stream_id} — receive a secret value from the frontend."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        stream_id = request.match_info['stream_id']
        pending = self._pending_secrets.get(stream_id)
        if not pending:
            return web.json_response(
                {'error': 'No pending secret capture for this stream'}, status=404
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)

        var_name = body.get('var_name', '')
        value = body.get('value', '')

        if not value:
            return web.json_response({'error': 'value is required'}, status=400)
        if len(value) > 4096:
            return web.json_response({'error': 'value too long'}, status=400)
        if var_name != pending['var_name']:
            return web.json_response(
                {'error': f"var_name mismatch: expected {pending['var_name']}"}, status=400
            )

        # Write to .env using the same function the CLI uses
        try:
            from hermes_cli.config import save_env_value_secure
            result = save_env_value_secure(var_name, value)
            result['skipped'] = False
            result['message'] = 'Secret stored securely. The value was not exposed to the model.'
        except Exception as e:
            logger.error('[myah] Failed to save env value %s: %s', var_name, e)
            # Unblock the agent thread (leave result=None → callback treats as skip)
            pending['event'].set()
            return web.json_response({'error': f'Failed to store: {e}'}, status=500)

        # Unblock the agent thread
        pending['result'] = result
        pending['event'].set()

        return web.json_response({'ok': True, 'stored_as': var_name})

    # ── Myah: HTTP wrapper for auxiliary_client.call_llm ──────────────────
    _AUX_ALLOWED_TASKS = frozenset({
        'title_generation',
        'follow_up_generation',
    })

    async def _handle_aux_endpoint(self, request: 'web.Request') -> 'web.Response':
        """POST /myah/v1/aux/{task} — forward to auxiliary_client.call_llm."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        task = request.match_info.get('task', '')
        if task not in self._AUX_ALLOWED_TASKS:
            return web.json_response(
                {'error': f'unknown aux task: {task}'},
                status=400,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'invalid JSON'}, status=400)

        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            return web.json_response(
                {'error': 'messages field is required and must be a non-empty list'},
                status=400,
            )

        extra_body = {}
        if 'response_format' in body:
            extra_body['response_format'] = body['response_format']

        # ── Myah: aux router import for /myah/v1/aux/{task} ──────────────
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
        # ──────────────────────────────────────────────────────────────────
        try:
            response = await async_call_llm(
                task=task,
                messages=messages,
                temperature=body.get('temperature'),
                max_tokens=body.get('max_tokens'),
                extra_body=extra_body or None,
            )
        except Exception as e:
            return web.json_response({'error': str(e)}, status=502)

        usage_dict = {}
        if hasattr(response, 'usage') and response.usage is not None:
            usage_dict = {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0),
            }

        # Use extract_content_or_reasoning so reasoning-model responses
        # (DeepSeek-R1, gemini-2.5-flash with thinking, etc.) that put
        # output in `message.reasoning` instead of `message.content` still
        # surface a non-null content string. See e2e-output/report.md
        # ISSUE-002 — chat titles + follow-up chips fell back to empty
        # because the raw .content was None for these models.
        content = extract_content_or_reasoning(response)

        return web.json_response({
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': content,
                },
                'finish_reason': getattr(response.choices[0], 'finish_reason', 'stop'),
            }],
            'usage': usage_dict,
        })
    # ─────────────────────────────────────────────────────────────────────

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = 'dangerous command',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Emit a structured approval SSE event instead of plain text.

        Called by the gateway runner when a dangerous command requires
        approval.  Emits a tool.confirmation_required event so the
        frontend can render the structured ConfirmationCard UI.
        """
        stream_id = self._session_streams.get(session_key)
        if not stream_id or stream_id not in self._streams:
            return SendResult(success=False, error='No active stream')

        confirmation_id = uuid.uuid4().hex
        cmd_preview = command[:500] + '...' if len(command) > 500 else command

        self._push_event_sync(stream_id, {
            'event': 'tool.confirmation_required',
            'stream_id': stream_id,
            'run_id': stream_id,
            'timestamp': time.time(),
            'confirmation_id': confirmation_id,
            'action_type': 'exec_approval',
            'description': f'Command requires approval:\n{cmd_preview}\n\nReason: {description}',
            'options': ['approve', 'approve_session', 'deny'],
            'metadata': metadata or {},
        })

        return SendResult(success=True)

    # ── Myah: Bug A follow-on — structured action confirmation card ─────
    async def send_action_confirmation(
        self,
        session_key: str,
        payload: Dict[str, Any],
    ) -> SendResult:
        """Emit a ``tool.confirmation_required`` SSE event for a generic
        action confirmation (cron creation, plugin install, ...).

        Mirrors ``send_exec_approval`` but accepts the payload from
        ``tools/approval.py::request_action_confirmation`` directly so
        the frontend's ``ConfirmationCard`` renders an interactive
        Approve / Deny card with the same ``confirmation_id`` the agent
        is blocked on.  No text fallback — callers should fall back to
        ``adapter.send`` if this returns ``success=False`` so users
        without a live stream still see the prompt as plain text.

        Payload contract (from ``approval.py::request_action_confirmation``):
            ``type``           — always ``"tool.confirmation_required"``
            ``confirmation_id`` — uuid the gateway resolves against
            ``action_type``    — e.g. ``"cron_create"``
            ``description``    — one-line human-readable summary
            ``options``        — usually ``["approve", "approve_session", "deny"]``
            ``metadata``       — optional structured fields for the card
                                  (``schedule_display``, ``prompt_preview``, …)
        """
        stream_id = self._session_streams.get(session_key)
        if not stream_id or stream_id not in self._streams:
            return SendResult(success=False, error='No active stream')

        self._push_event_sync(stream_id, {
            'event': 'tool.confirmation_required',
            'stream_id': stream_id,
            'run_id': stream_id,
            'timestamp': time.time(),
            'confirmation_id': payload.get('confirmation_id', ''),
            'action_type': payload.get('action_type', 'confirmation'),
            'description': payload.get('description', ''),
            'options': payload.get('options') or ['approve', 'deny'],
            'metadata': payload.get('metadata') or {},
        })
        return SendResult(success=True)
    # ────────────────────────────────────────────────────────────────────

    # ── Structured callbacks for gateway runner ───────────────────────────

    def get_structured_callbacks(self, session_key: str) -> Optional[Dict]:
        """Return callbacks that push structured SSE events to the stream.

        Called by gateway/run.py before each agent turn.  If this returns
        a dict, the gateway uses these callbacks instead of the default
        text-based GatewayStreamConsumer.

        The callbacks fire from the agent's worker thread, so they use
        call_soon_threadsafe (Fix 2) to push events safely.

        tool_progress_callback receives these invocation patterns (Fix 3):
            ("tool.started", tool_name, preview, args_dict)
            ("tool.completed", tool_name, None, None, duration=float, is_error=bool)
            ("_thinking", first_line_text)
            ("reasoning.available", "_thinking", text_preview, None)
        """
        stream_id = self._session_streams.get(session_key)
        if not stream_id or stream_id not in self._streams:
            return None

        q = self._streams[stream_id]

        def _put(event_data: dict):
            """Thread-safe push from agent worker thread."""
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event_data)
            except RuntimeError:
                pass  # Loop closed

        def _stream_delta(text):
            if text is None:
                return  # Tool boundary signal — ignore for SSE
            _put({
                "event": "message.delta",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "delta": text,
            })

        def _tool_progress(*args, **kwargs):
            _put(self._format_tool_event(stream_id, args, kwargs))

        def _reasoning(text):
            if not text:
                return
            _put({
                "event": "reasoning.delta",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "text": text,
            })

        def _status(text):
            _put({
                "event": "status",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "text": text,
            })

        return {
            "stream_delta": _stream_delta,
            "tool_progress": _tool_progress,
            "reasoning": _reasoning,
            "status": _status,
        }

    @staticmethod
    def _format_tool_event(stream_id: str, args: tuple, kwargs: dict) -> dict:
        """Format tool_progress_callback arguments into an SSE event dict.

        Handles all four invocation patterns from run_agent.py (Fix 3).
        """
        if not args:
            return {
                "event": "status",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "text": "working",
            }

        event_type = args[0]

        if event_type == "tool.started" and len(args) >= 4:
            return {
                "event": "tool.started",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "tool": args[1],
                "call_id": args[1],
                "args": args[3] if isinstance(args[3], dict) else {},
                "preview": args[2] or "",
            }
        elif event_type == "tool.completed" and len(args) >= 2:
            return {
                "event": "tool.completed",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "tool": args[1],
                "call_id": args[1],
                "args": {},
                "result": "",
                "duration": kwargs.get("duration", 0),
                "error": kwargs.get("is_error", False),
            }
        elif event_type == "_thinking" and len(args) >= 2:
            return {
                "event": "reasoning.delta",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "text": args[1],
            }
        elif event_type == "reasoning.available" and len(args) >= 3:
            return {
                "event": "reasoning.available",
                "stream_id": stream_id,
                "run_id": stream_id,
                "timestamp": time.time(),
                "text": args[2] or "",
            }
        # Fallback for unknown event types
        return {
            "event": "status",
            "stream_id": stream_id,
            "run_id": stream_id,
            "timestamp": time.time(),
            "text": str(args[0]) if args else "unknown",
        }

    # ── Orphaned stream sweeper ─────────────────────────────────────────

    async def _sweep_orphaned_streams(self) -> None:
        """Periodically clean up streams that were never consumed."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            stale = [
                sid
                for sid, created_at in list(self._streams_created.items())
                if now - created_at > _STREAM_TTL
            ]
            for sid in stale:
                logger.debug("[myah] sweeping orphaned stream %s", sid)
                q = self._streams.pop(sid, None)
                self._streams_created.pop(sid, None)
                # Also clean up any lingering mappings
                session_key = self._stream_sessions.pop(sid, None)
                if session_key:
                    self._session_streams.pop(session_key, None)
                    try:
                        from tools.approval import unregister_gateway_notify
                        unregister_gateway_notify(session_key)
                    except Exception:
                        pass
                # Remove reverse chat_id mapping
                stale_chat_ids = [
                    cid for cid, s in self._chat_id_streams.items() if s == sid
                ]
                for cid in stale_chat_ids:
                    self._chat_id_streams.pop(cid, None)
                # Close the queue if anyone is listening
                if q is not None:
                    try:
                        q.put_nowait(None)
                    except Exception:
                        pass

    # ── BasePlatformAdapter interface ───────────────────────────────────

    # ── Myah: media streaming endpoint ─────────────────────────────────────────
    def _myah_allowed_media_roots(self) -> 'list[_myah_Path]':
        """Return the list of allowed cache directories for media streaming."""
        from hermes_constants import get_hermes_home, get_hermes_dir
        base = get_hermes_home()
        roots = [
            # ── Myah: root cache dir — allow any artifact the agent writes
            # directly under /data/.hermes/cache/. Without this, files the
            # agent creates via execute_code/terminal (which default to the
            # cache root, not a subdirectory) return 403 and never persist
            # back to the chat. T3-1001 dogfooding 2026-04-24.
            (base / 'cache').resolve(),
            # ────────────────────────────────────────────────────────────
            get_hermes_dir('cache/images', 'image_cache').resolve(),
            get_hermes_dir('cache/audio', 'audio_cache').resolve(),
            get_hermes_dir('cache/documents', 'document_cache').resolve(),
            get_hermes_dir('cache/screenshots', 'browser_screenshots').resolve(),
        ]
        return roots

    async def _handle_media_get(self, request: 'web.Request') -> 'web.Response':
        """GET /myah/v1/media?path=<path> — stream a cached media file.

        Security:
        - Bearer auth (same as other /myah/v1/* routes)
        - strict path resolution defeats symlink traversal
        - whitelist of allowed cache directories
        - no directory listing, no write, no delete
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        path_str = request.rel_url.query.get('path', '')
        if not path_str:
            return web.json_response({'error': "Missing 'path' query parameter"}, status=400)

        try:
            resolved = _myah_Path(path_str).resolve(strict=True)
        except (FileNotFoundError, RuntimeError, OSError):
            return web.json_response({'error': 'File not found'}, status=404)

        allowed_roots = self._myah_allowed_media_roots()

        def _is_subpath(child: '_myah_Path', parent: '_myah_Path') -> bool:
            try:
                child.relative_to(parent)
                return True
            except ValueError:
                return False

        if not any(_is_subpath(resolved, root) for root in allowed_roots):
            return web.json_response(
                {'error': 'Path not in an allowed cache directory'}, status=403,
            )

        mime, _ = _myah_mimetypes.guess_type(str(resolved))
        if not mime:
            mime = 'application/octet-stream'

        try:
            file_size = resolved.stat().st_size
        except OSError:
            return web.json_response({'error': 'File not accessible'}, status=404)

        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': mime,
                'Content-Length': str(file_size),
                'Cache-Control': 'private, max-age=300',
            },
        )
        await response.prepare(request)
        try:
            with resolved.open('rb') as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    await response.write(chunk)
        except Exception:
            pass  # client disconnected mid-stream
        await response.write_eof()
        return response
    # ────────────────────────────────────────────────────────────────────────────

    def _register_routes_on_app(self, app: "web.Application") -> None:
        """Pre-setup hook: add Myah routes to the shared aiohttp app.

        Called by the API server's connect() BEFORE runner.setup() freezes
        the router.  This is registered in __init__ via register_pre_setup_hook.
        """
        app["myah_adapter"] = self
        app.router.add_get("/myah/health", self._handle_health)
        app.router.add_post("/myah/v1/message", self._handle_message_endpoint)
        app.router.add_get("/myah/v1/events/{stream_id}", self._handle_events_endpoint)
        app.router.add_post("/myah/v1/confirm/{stream_id}", self._handle_confirm_endpoint)
        app.router.add_post("/myah/v1/secret/{stream_id}", self._handle_secret_endpoint)
        app.router.add_get("/myah/v1/media", self._handle_media_get)  # Myah: media endpoint
        # ── Myah: aux router HTTP wrapper ────────────────────────────────
        app.router.add_post('/myah/v1/aux/{task}', self._handle_aux_endpoint)
        # ─────────────────────────────────────────────────────────────────

        # ── Myah: runtime-control admin surface ──────────────────────────
        # Mounts /myah/v1/admin/* — the small set of admin operations that
        # MUST run in the gateway process because they touch GatewayRunner
        # state (session model overrides, cache eviction, busy-check, MCP
        # refresh). Everything else (file-system admin: SOUL, skills, plugins,
        # MCP CRUD, providers, reset) lives in the myah-admin DASHBOARD plugin
        # at agent/hermes/plugins/myah-admin/dashboard/plugin_api.py.
        from gateway.platforms.myah_runtime_admin import register_runtime_admin_routes
        register_runtime_admin_routes(
            app,
            runner=self.gateway_runner,
            auth_key=self._auth_key,
        )
        # ─────────────────────────────────────────────────────────────────

        self._routes_registered = True
        logger.info("[%s] Routes registered on shared aiohttp app (pre-setup hook)", self.name)

    async def connect(self) -> bool:
        """Finalize adapter connection after routes are registered.

        Routes are registered via the pre-setup hook (called during API server
        connect, before the router is frozen).  This method waits for the API
        server to be ready, then starts background tasks.

        If routes were already registered via the hook, we just need to verify
        and start.  If the API server hasn't connected yet, we return False
        and the gateway retries.
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        from gateway.platforms.api_server import get_shared_app
        app = get_shared_app()

        if not self._routes_registered:
            if app is None:
                logger.info(
                    "[%s] Shared aiohttp app not available yet. Will retry.",
                    self.name,
                )
                return False
            # API server is up but our pre-setup hook didn't fire (adapter
            # was created after API server connect).  Try registering routes
            # directly — this will only work if the router isn't frozen yet.
            try:
                self._register_routes_on_app(app)
            except RuntimeError as e:
                if "frozen" in str(e).lower():
                    logger.error(
                        "[%s] Cannot register routes: aiohttp router is frozen. "
                        "Ensure MYAH adapter is created BEFORE API_SERVER connect().",
                        self.name,
                    )
                    return False
                raise

        # Capture the event loop for thread-safe queue access (Fix 2)
        self._loop = asyncio.get_running_loop()

        # Start background sweep for orphaned streams
        sweep_task = asyncio.create_task(self._sweep_orphaned_streams())
        try:
            self._background_tasks.add(sweep_task)
        except TypeError:
            pass
        if hasattr(sweep_task, "add_done_callback"):
            sweep_task.add_done_callback(self._background_tasks.discard)

        self._mark_connected()
        logger.info("[%s] Myah adapter connected", self.name)
        return True

    async def disconnect(self) -> None:
        """Clean up all active streams and mappings."""
        self._mark_disconnected()

        # Close all active streams
        for stream_id, q in list(self._streams.items()):
            try:
                q.put_nowait(None)
            except Exception:
                pass

        # Unregister all approval callbacks
        from tools.approval import unregister_gateway_notify
        for session_key in list(self._session_streams.keys()):
            try:
                unregister_gateway_notify(session_key)
            except Exception:
                pass

        self._streams.clear()
        self._streams_created.clear()
        self._session_streams.clear()
        self._chat_id_streams.clear()
        self._stream_sessions.clear()

        logger.info("[%s] Myah adapter disconnected", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Push a response to the user.

        Routing depends on the **caller**:

        * **Cron deliveries** (``metadata.job_id`` is set — Bug D v4):
          ALWAYS go through the webhook so the platform persists the
          run to chat history.  SSE is *not* a persistence path —
          ``message.delta`` events without a real consumer are silently
          dropped, and even with a consumer the cron's payload would
          be concatenated into whatever assistant message the chat
          happens to be streaming for an unrelated turn.  The webhook
          (``/api/v1/processes/webhook/run-complete``) is the only
          surface that calls ``_inject_cron_output_to_chat`` and
          writes the cron run to the chat's message history.

        * **Live chat replies** (no ``metadata.job_id``):
          1. Push as ``message.delta`` onto the active SSE stream when
             a subscriber is bound to ``chat_id`` — the user sees
             tokens stream into the message they just sent.
          2. Fall back to ``"No active stream"`` failure when no
             subscriber exists.  These are normal gateway-runner
             responses; the runner has its own retry semantics, and
             the cron-only webhook would 400 on missing ``job_id``.

        Threading: invoked from the cron ``ThreadPoolExecutor`` worker
        via ``asyncio.run_coroutine_threadsafe(adapter.send(...),
        loop)`` (see ``cron/scheduler.py:_deliver_result``).  The body
        is a plain ``async def`` — the threading bridge happens in the
        caller, not here.  Never raises — the cron path relies on
        ``SendResult.success`` to decide between adapter delivery and
        the standalone fallback.

        Bug D-v4 history (2026-04-25): an earlier version of this
        method preferred SSE-first for ALL callers and only fell back
        to the webhook when ``_chat_id_streams.get(chat_id)`` was
        empty.  That broke "test it here" cron triggers — the
        triggering chat turn had a live stream, the cron's content
        was pushed onto that turn's SSE queue, ``send()`` returned
        success, and the webhook never fired.  The output was either
        appended into the unrelated assistant message buffer or
        dropped on the floor when the run completed.  Output existed
        on disk but never reached chat history.  See
        ``docs/superpowers/specs/2026-04-24-cron-origin-and-approval-design.md``.
        """
        meta = metadata or {}
        is_cron_delivery = bool(meta.get("job_id"))

        # ── Cron deliveries: always webhook (persistence path) ──
        if is_cron_delivery:
            # Optional live-preview: push the cron output to any active
            # SSE stream so the user gets a quick visual confirmation if
            # they happen to be watching this chat.  This is decoration
            # only — the platform's webhook handler is what writes the
            # cron run to chat history regardless of what we push here.
            stream_id = self._chat_id_streams.get(chat_id) if chat_id else None
            if stream_id and stream_id in self._streams:
                try:
                    self._push_event_sync(stream_id, {
                        "event": "message.delta",
                        "stream_id": stream_id,
                        "run_id": stream_id,
                        "timestamp": time.time(),
                        "delta": content,
                        "message_id": uuid.uuid4().hex[:12],
                    })
                except Exception as exc:  # noqa: BLE001 - best-effort preview
                    logger.debug(
                        f"Live-preview SSE push for cron delivery failed (non-fatal): {exc}"
                    )
            return await self._send_via_webhook(chat_id, content, metadata)
        # ────────────────────────────────────────────────────────

        # ── Live chat replies: SSE-first ────────────────────
        stream_id = self._chat_id_streams.get(chat_id) if chat_id else None
        if stream_id:
            q = self._streams.get(stream_id)
            if q is not None:
                msg_id = uuid.uuid4().hex[:12]
                self._push_event_sync(stream_id, {
                    "event": "message.delta",
                    "stream_id": stream_id,
                    "run_id": stream_id,
                    "timestamp": time.time(),
                    "delta": content,
                    "message_id": msg_id,
                })
                return SendResult(success=True, message_id=msg_id)

        # No live stream and not a cron delivery — preserve the legacy
        # ``No active stream`` failure shape.  The webhook is cron-only
        # (rejects payloads without ``user_id``/``job_id``), so attempting
        # it for a chat reply would always 400 and add log noise.
        return SendResult(
            success=False,
            error=f"No active stream for chat_id={chat_id}",
        )
        # ────────────────────────────────────────────────────

    # ── Myah: Bug D v3 — webhook delivery helper ────────────
    async def _send_via_webhook(
        self,
        chat_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        """POST cron output to the platform webhook receiver.

        Reads ``MYAH_PLATFORM_BASE_URL`` / ``MYAH_PLATFORM_BEARER`` /
        ``MYAH_USER_ID`` fresh from ``os.environ`` per call (the
        module-level cache at the top of this file is used by the
        attachment-fetch path which has different startup ordering).

        Payload shape pinned by ``platform/backend/open_webui/routers/
        processes.py:824-831`` (the ``/webhook/run-complete`` handler).

        **Only fires for cron deliveries.**  The platform's webhook
        endpoint is ``/api/v1/processes/webhook/run-complete`` — it's
        cron-specific and rejects payloads without ``user_id`` and
        ``job_id``.  We detect cron callers by the presence of
        ``metadata.job_id`` (populated by
        ``cron/scheduler.py::_build_myah_send_metadata``) and skip the
        webhook for non-cron chat replies that happen to hit a closed
        SSE stream — those should preserve the legacy
        ``"No active stream"`` failure so the gateway's standalone-send
        retry path can take over.
        """
        meta = metadata or {}

        # ── Skip for non-cron callers ──────────────────────
        # Live chat replies (gateway/run.py → adapter.send) carry
        # ``metadata={"thread_id": ...}`` or ``None`` — no cron context.
        # Attempting the webhook there always fails (platform 400s on
        # missing user_id/job_id) and adds noise to the logs.
        if not meta.get('job_id'):
            return SendResult(
                success=False,
                error=f"No active stream for chat_id={chat_id}",
            )

        base_url = _myah_os.environ.get('MYAH_PLATFORM_BASE_URL')
        bearer = _myah_os.environ.get('MYAH_PLATFORM_BEARER')
        if not (base_url and bearer):
            # No webhook env — preserve the legacy ``No active stream``
            # failure shape so existing callers (and existing tests)
            # don't see new behaviour when the platform isn't wired in.
            return SendResult(
                success=False,
                error=f"No active stream for chat_id={chat_id}",
            )

        # chat_id resolution: caller-supplied first, then origin metadata
        # (cron path passes deliver=origin → adapter receives the origin
        # chat_id via metadata enrichment in scheduler._deliver_result).
        origin = meta.get('origin') if isinstance(meta.get('origin'), dict) else {}
        resolved_chat_id = chat_id or (origin.get('chat_id') if origin else '') or ''
        if not resolved_chat_id:
            return SendResult(
                success=False,
                error="webhook fallback: no chat_id and no metadata.origin.chat_id",
            )

        user_id = _myah_os.environ.get('MYAH_USER_ID', '')
        payload = {
            'user_id': user_id,
            'job_id': meta.get('job_id') or '',
            'job_name': meta.get('job_name') or meta.get('job_id') or '',
            'chat_id': resolved_chat_id,
            'response': content,
            'status': meta.get('status') or 'ok',
            'ran_at': meta.get('ran_at') or '',
            'tool_calls_log': meta.get('tool_calls_log'),
        }
        url = f"{base_url.rstrip('/')}/api/v1/processes/webhook/run-complete"
        headers = {'Authorization': f'Bearer {bearer}'}

        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if 200 <= resp.status < 300:
                        msg_id = uuid.uuid4().hex[:12]
                        return SendResult(success=True, message_id=msg_id)
                    body_text = ''
                    try:
                        body_text = (await resp.text())[:200]
                    except Exception:
                        pass
                    logger.warning(
                        f"Myah webhook delivery failed: status={resp.status} "
                        f"url={url} chat_id={resolved_chat_id} body={body_text!r}"
                    )
                    self._maybe_breadcrumb(
                        f"webhook delivery non-2xx: {resp.status}",
                        url=url, chat_id=resolved_chat_id, status=resp.status,
                    )
                    return SendResult(
                        success=False,
                        error=f"webhook returned HTTP {resp.status}",
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Myah webhook delivery raised {type(exc).__name__}: {exc} "
                f"(url={url} chat_id={resolved_chat_id})"
            )
            self._maybe_breadcrumb(
                f"webhook delivery exception: {type(exc).__name__}",
                url=url, chat_id=resolved_chat_id, error=str(exc)[:200],
            )
            return SendResult(
                success=False,
                error=f"webhook error: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _maybe_breadcrumb(message: str, **data: Any) -> None:
        """Best-effort Sentry breadcrumb at warning level — never raises."""
        try:
            import sentry_sdk  # type: ignore[import-not-found]
            sentry_sdk.add_breadcrumb(
                category="myah.adapter",
                level="warning",
                message=message,
                data=data,
            )
        except Exception:  # noqa: BLE001 - breadcrumb is best-effort
            pass
    # ────────────────────────────────────────────────────────

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Push a typing/status indicator to the SSE stream."""
        stream_id = self._chat_id_streams.get(chat_id)
        if not stream_id:
            return

        self._push_event_sync(stream_id, {
            "event": "status",
            "stream_id": stream_id,
            "run_id": stream_id,
            "timestamp": time.time(),
            "status": "typing",
        })

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about this chat."""
        return {
            "name": "Myah Web",
            "type": "dm",
            "platform": "myah",
            "chat_id": chat_id,
        }
