"""NATS gateway adapter.

Registers one ``natsagent.Agent`` at ``agents.<agent>.<owner>.<name>`` and
routes inbound NATS Agent Protocol v0.2 prompts through the gateway's
normal ``MessageEvent`` pipeline. Streams responses back chunk-by-chunk
over the reply subject; the SDK owns terminator + heartbeat emission.

Protocol spec: ``../nats-agent-sdk-docs/core-protocol.md`` (v0.2).
Hermes architectural reference: ``docs/nats-gateway-design.md``.

Phase 4 scope (T4.1–T4.3): full inbound pipeline — session extraction,
attachment decoding, MessageEvent construction, keep-alive emission,
adapter-owned AIAgent + streaming delta pump for text prompts, and
reuse of the gateway's command dispatch for slash commands.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

try:
    import natsagent
    NATSAGENT_AVAILABLE = True
except ImportError:
    natsagent = None  # type: ignore[assignment]
    NATSAGENT_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cache_video_from_bytes,
)

if TYPE_CHECKING:
    from natsagent import Envelope, PromptStream

logger = logging.getLogger(__name__)


# Defaults per docs/nats-gateway-design.md §4.
DEFAULT_AGENT = "hermes"
DEFAULT_SESSION_DEFAULT = "default"
DEFAULT_HEARTBEAT_INTERVAL_S = 30
DEFAULT_MAX_PAYLOAD = "1MB"
DEFAULT_ATTACHMENTS_OK = True
DEFAULT_ACK_KEEPALIVE_INTERVAL_S = 20

# §6.6 recommends callers default to 60 s inactivity timeout. Keep the
# adapter's keep-alive cadence strictly below that so callers never trip
# on idle disconnects while the handler is silent mid-reasoning.
MAX_ACK_KEEPALIVE_INTERVAL_S = 60

# Matches the SDK's §2.1 size grammar — a number followed by B/KB/MB/GB.
# We pre-flight the value here so bad configs fail at startup, not during
# agent construction deep in the stack trace.
_MAX_PAYLOAD_RE = re.compile(r"^\s*\d+\s*(?:B|KB|MB|GB)\s*$", re.IGNORECASE)

# SDK's §2.2 subject-token grammar for the ``agent`` field. Owner/name are
# sanitized by the SDK (base64-url fallback for non-conforming tokens), so
# we only insist on non-empty there.
_AGENT_TOKEN_RE = re.compile(r"^[a-z0-9-]+$")

# Attachment extension → cache helper. Anything not matching an image /
# audio / video extension falls back to the document cache, which preserves
# the original filename and accepts arbitrary bytes (§5.2: "Agents interpret
# the bytes by extension or content sniff" — extension-only is compliant).
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# Handler-scoped current stream. Set by ``_on_prompt`` at entry and reset
# in its ``finally`` block; read by ``send()`` and the ``send_*`` helpers
# to reach the caller's own reply subject even when two prompts for the
# same ``envelope.session`` overlap.
#
# T5.0 (Phase 5) — this is the race-safe primary lookup. ``_active_streams``
# is kept as a compound-keyed ``(chat_id, id(stream)) → stream`` dict for
# diagnostics and for the narrow cases where a send is scheduled outside
# the handler's context (contextvars don't propagate through
# ``run_coroutine_threadsafe``). Phase 4's shortcoming #1 only manifested
# on the dict lookup — so once callers prefer the contextvar, concurrent
# prompts for the same chat_id land on their own streams regardless of
# dict state.
_current_stream: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "nats_current_stream", default=None
)


def check_nats_requirements() -> bool:
    """Return True iff the ``natsagent`` SDK is importable.

    Mirrors the ``check_*_requirements`` predicate every other adapter
    exposes for ``gateway.run._create_adapter`` to short-circuit when the
    dependency is missing.
    """
    return NATSAGENT_AVAILABLE


class NatsConfigError(ValueError):
    """Raised when ``PlatformConfig.extra`` for the NATS platform is invalid.

    Surfaced via ``_set_fatal_error(retryable=False)`` in
    :meth:`NatsAdapter.__init__` so the gateway fails fast with a
    readable message instead of crashing during ``connect()``.
    """


@dataclass(frozen=True)
class NatsAdapterSettings:
    """Parsed + validated NATS adapter configuration.

    Built from ``PlatformConfig.extra`` via :meth:`from_extra`. Frozen so
    no code path can mutate the resolved settings after ``__init__``.
    """

    servers: Optional[List[str]]
    context: Optional[str]
    agent: str
    owner: str
    name: str
    session_default: str
    heartbeat_interval_s: int
    max_payload: str
    attachments_ok: bool
    ack_keepalive_interval_s: int

    @classmethod
    def from_extra(cls, extra: Dict[str, Any]) -> "NatsAdapterSettings":
        """Parse ``config.extra`` into a validated settings object.

        Raises :class:`NatsConfigError` with an actionable message on any
        validation failure; never returns a partially-populated instance.
        """
        extra = extra or {}

        servers, context = _parse_transport(extra)

        agent = _require_token(
            extra.get("agent"),
            default=DEFAULT_AGENT,
            field_name="agent",
            pattern=_AGENT_TOKEN_RE,
        )
        owner = _require_token(
            extra.get("owner"),
            default=None,
            field_name="owner",
            pattern=None,
        )
        name = _require_token(
            extra.get("name"),
            default=None,
            field_name="name",
            pattern=None,
        )

        session_default = _optional_str(
            extra.get("session_default"),
            default=DEFAULT_SESSION_DEFAULT,
            field_name="session_default",
        )

        heartbeat_interval_s = _positive_int(
            extra.get("heartbeat_interval_s"),
            default=DEFAULT_HEARTBEAT_INTERVAL_S,
            field_name="heartbeat_interval_s",
        )

        max_payload = _optional_str(
            extra.get("max_payload"),
            default=DEFAULT_MAX_PAYLOAD,
            field_name="max_payload",
        )
        if not _MAX_PAYLOAD_RE.match(max_payload):
            raise NatsConfigError(
                f"NATS: 'max_payload' {max_payload!r} is not a valid size "
                f"(expected e.g. '1MB', '512KB', '4GB')"
            )

        attachments_ok = extra.get("attachments_ok", DEFAULT_ATTACHMENTS_OK)
        if not isinstance(attachments_ok, bool):
            raise NatsConfigError(
                f"NATS: 'attachments_ok' must be a boolean, got "
                f"{type(attachments_ok).__name__}"
            )

        ack_keepalive_interval_s = _positive_int(
            extra.get("ack_keepalive_interval_s"),
            default=DEFAULT_ACK_KEEPALIVE_INTERVAL_S,
            field_name="ack_keepalive_interval_s",
        )
        if ack_keepalive_interval_s >= MAX_ACK_KEEPALIVE_INTERVAL_S:
            raise NatsConfigError(
                f"NATS: 'ack_keepalive_interval_s' ({ack_keepalive_interval_s}) "
                f"must be < {MAX_ACK_KEEPALIVE_INTERVAL_S}s — protocol §6.6 "
                f"recommends callers default to 60 s inactivity timeout, so "
                f"keep-alive needs headroom below that"
            )

        return cls(
            servers=servers,
            context=context,
            agent=agent,
            owner=owner,
            name=name,
            session_default=session_default,
            heartbeat_interval_s=heartbeat_interval_s,
            max_payload=max_payload,
            attachments_ok=attachments_ok,
            ack_keepalive_interval_s=ack_keepalive_interval_s,
        )

    @property
    def identity(self) -> str:
        """Stable lock identity ``{agent}:{owner}:{name}``.

        Used by :meth:`NatsAdapter.connect` (Phase 3) to scope the
        ``acquire_scoped_lock`` call per design doc §5.
        """
        return f"{self.agent}:{self.owner}:{self.name}"


def _parse_transport(extra: Dict[str, Any]) -> tuple[Optional[List[str]], Optional[str]]:
    """Extract (servers, context) from extra, enforcing exactly-one."""
    raw_servers = extra.get("servers")
    raw_context = extra.get("context")

    servers: Optional[List[str]] = None
    context: Optional[str] = None

    has_servers = raw_servers not in (None, "", [])
    has_context = raw_context not in (None, "")

    if has_servers and has_context:
        raise NatsConfigError(
            "NATS: specify either 'servers' or 'context', not both"
        )
    if not has_servers and not has_context:
        raise NatsConfigError(
            "NATS: exactly one of 'servers' (list of URLs) or 'context' "
            "(nats CLI context name) is required"
        )

    if has_servers:
        if isinstance(raw_servers, str):
            candidates = [raw_servers]
        elif isinstance(raw_servers, (list, tuple)):
            candidates = list(raw_servers)
        else:
            raise NatsConfigError(
                f"NATS: 'servers' must be a string or list of strings, "
                f"got {type(raw_servers).__name__}"
            )
        servers = [str(s).strip() for s in candidates if str(s).strip()]
        if not servers:
            raise NatsConfigError(
                "NATS: 'servers' must contain at least one non-empty URL"
            )

    if has_context:
        if not isinstance(raw_context, str):
            raise NatsConfigError(
                f"NATS: 'context' must be a string, got "
                f"{type(raw_context).__name__}"
            )
        context = raw_context.strip()
        if not context:
            raise NatsConfigError("NATS: 'context' must be non-empty")

    return servers, context


def _require_token(
    value: Any,
    default: Optional[str],
    field_name: str,
    pattern: Optional[re.Pattern[str]],
) -> str:
    """Return a stripped non-empty token, applying ``default`` if unset.

    If ``pattern`` is given, fail fast when the supplied value doesn't
    match — used to catch invalid ``agent`` tokens before the SDK's own
    :class:`AgentSubject.new` surfaces the same error from deeper in the
    stack.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is None:
            raise NatsConfigError(f"NATS: '{field_name}' is required")
        value = default

    if not isinstance(value, str):
        raise NatsConfigError(
            f"NATS: '{field_name}' must be a string, got "
            f"{type(value).__name__}"
        )
    stripped = value.strip()
    if pattern is not None and not pattern.fullmatch(stripped):
        raise NatsConfigError(
            f"NATS: '{field_name}' {stripped!r} must match {pattern.pattern} "
            f"(protocol §2.2)"
        )
    return stripped


def _optional_str(value: Any, default: str, field_name: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise NatsConfigError(
            f"NATS: '{field_name}' must be a string, got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        return default
    return stripped


def _positive_int(value: Any, default: int, field_name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        # bool is a subclass of int — reject it explicitly to avoid silent
        # coercion of ``True`` to ``1``.
        raise NatsConfigError(
            f"NATS: '{field_name}' must be an integer, got bool"
        )
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise NatsConfigError(
            f"NATS: '{field_name}' must be an integer, got {value!r}"
        ) from exc
    if coerced <= 0:
        raise NatsConfigError(
            f"NATS: '{field_name}' must be positive, got {coerced}"
        )
    return coerced


class NatsAdapter(BasePlatformAdapter):
    """Gateway adapter for the NATS Agent Protocol v0.2.

    Phase 4 scope — settings parsing, connect/disconnect lifecycle, and
    the full inbound pipeline: ``_on_prompt`` reads ``envelope.session``,
    decodes attachments, starts a keep-alive task, dispatches slash
    commands through the gateway's command registry, and drives text
    prompts through an adapter-owned ``AIAgent`` with streaming deltas
    pumped onto the ``PromptStream``.
    """

    # NATS publishes each streaming chunk as a fresh ResponseChunk — there
    # is no "edit message" semantic on the wire (design doc §6.1), so the
    # default GatewayStreamConsumer (which progressively edits a single
    # platform message) is incompatible. This flag causes ``run.py`` to
    # skip consumer construction for NATS; streaming is instead wired
    # adapter-locally via ``_run_nats_agent``'s stream_delta_callback.
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.NATS)

        # Compound-keyed handle registry: ``(chat_id, id(stream)) → stream``.
        # Populated by ``_on_prompt`` on receipt and consulted by
        # ``send()`` / ``send_*`` helpers. Two prompts arriving with the
        # same ``envelope.session`` therefore each land on a distinct key rather
        # than overwriting each other (Phase 4 shortcoming #1 / T5.0).
        # The primary per-handler lookup is the ``_current_stream``
        # contextvar; this dict covers only the rare "send scheduled
        # outside my handler's context" fallback and diagnostics.
        self._active_streams: Dict[Tuple[str, int], Any] = {}
        self._nc: Optional[Any] = None
        self._agent: Optional[Any] = None
        self._settings: Optional[NatsAdapterSettings] = None

        # Shutdown signalling for in-flight prompt handlers. ``_on_prompt``
        # registers its own task here via ``asyncio.current_task()`` so
        # ``_teardown_handles`` can cancel-and-await every live invocation
        # before ``agent.stop()`` deregisters the micro-service endpoint
        # and ``nc.close()`` drops the connection. The ``Event`` itself is
        # loop-agnostic in Python 3.10+ (binds lazily at first
        # ``set()``/``wait()``) so constructing it in ``__init__`` — which
        # may run before any event loop exists — is safe.
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._in_flight_handlers: Set[asyncio.Task] = set()

        # Per-``envelope.session`` serialization. Concurrent ``_on_prompt``
        # invocations for the same chat_id queue on this lock so only one
        # handler is active per session at a time. Prevents the
        # stacking-race documented in the Phase 6 shortcomings:
        #   - ``register_gateway_notify(session_key, cb)`` is per-session
        #     overwrite — without this lock, handler B's notify cb
        #     replaces handler A's (different captured streams), so A's
        #     dangerous-command approvals would route to B's stream.
        #   - ``_current_stream`` contextvar doesn't propagate through
        #     ``asyncio.run_coroutine_threadsafe``, so the dict fallback
        #     in ``_resolve_stream`` is ambiguous when multiple
        #     ``(chat_id, *)`` entries exist.
        # Both concerns vanish when only one handler per chat_id runs at
        # a time. Distinct chat_ids still run in parallel. The lock dict
        # grows with every distinct session seen; ``_teardown_handles``
        # clears it on disconnect so adapter restarts reset the pool.
        self._session_locks: Dict[str, asyncio.Lock] = {}

        try:
            self._settings = NatsAdapterSettings.from_extra(config.extra or {})
        except NatsConfigError as exc:
            self._set_fatal_error(
                "nats_config_error",
                str(exc),
                retryable=False,
            )
            logger.error("[%s] %s", self.name, exc)

    # ------------------------------------------------------------------
    # Lifecycle (Phase 3)
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Open a NATS connection, register the agent, and start heartbeats.

        Sequence (design doc §9 "Gateway startup"):
          1. Acquire the machine-local scope lock ``nats:{agent}:{owner}:{name}``
             (§5) so two profiles on one host can't shadow each other's
             registrations.
          2. ``natsagent.connect(servers=... | context=...)`` — the SDK picks
             the right nats-py auth bundle. We pass through exactly one of
             ``servers`` / ``context``; :class:`NatsAdapterSettings`
             already enforced the xor at init time.
          3. Build the :class:`natsagent.Agent` with the resolved identity
             and §2.1 endpoint metadata (max_payload, attachments_ok) +
             §8.2 heartbeat cadence.
          4. Register the prompt handler (`self._on_prompt`) — mandatory
             per :meth:`natsagent.Agent.start`.
          5. ``agent.start()`` — registers the NATS micro service,
             advertises on ``$SRV.*`` discovery subjects, and spawns the
             heartbeat publisher task.

        Failures at any step roll back cleanly: the lock is released, any
        partially-constructed ``_agent``/``_nc`` handles are torn down,
        and a retryable fatal error is recorded so ``gateway/run.py``
        queues another attempt 30 s later.
        """
        if self.has_fatal_error and not self.fatal_error_retryable:
            # Config parsing in __init__ failed — nothing to recover.
            # Returning False here keeps the behavior gate deterministic
            # regardless of whether connect_all retried us by mistake.
            return False
        if self._settings is None:
            # Defensive — has_fatal_error should already be True in this
            # case, but guard so later code never dereferences None.
            return False
        if not NATSAGENT_AVAILABLE or natsagent is None:
            self._set_fatal_error(
                "nats_sdk_missing",
                "natsagent SDK not installed; run: pip install 'hermes-agent[nats]'",
                retryable=False,
            )
            return False

        settings = self._settings

        if not self._acquire_platform_lock(
            "nats",
            settings.identity,
            f"NATS agent identity {settings.identity}",
        ):
            # _acquire_platform_lock already set the fatal error and logged.
            return False

        try:
            # Reset shutdown signalling for this attempt so long-running
            # prompt handlers that gate streaming on
            # ``self._shutdown_event`` start from a clean slate when a
            # previous teardown set it.
            self._shutdown_event.clear()

            connect_kwargs: Dict[str, Any] = {}
            if settings.servers is not None:
                # natsagent.connect accepts list[str] directly; copy so the
                # SDK can't mutate our frozen-dataclass-owned list via
                # nats-py internals.
                connect_kwargs["servers"] = list(settings.servers)
            if settings.context is not None:
                connect_kwargs["context"] = settings.context

            self._nc = await natsagent.connect(**connect_kwargs)

            self._agent = natsagent.Agent(
                agent=settings.agent,
                owner=settings.owner,
                name=settings.name,
                nc=self._nc,
                heartbeat_interval_s=settings.heartbeat_interval_s,
                max_payload=settings.max_payload,
                attachments_ok=settings.attachments_ok,
                session=settings.session_default,
            )
            self._agent.on_prompt(self._on_prompt)
            await self._agent.start()

            self._mark_connected()
            logger.info(
                "[%s] Connected — registered as agents.%s.%s.%s "
                "(heartbeat=%ss, max_payload=%s, attachments_ok=%s)",
                self.name,
                settings.agent,
                settings.owner,
                settings.name,
                settings.heartbeat_interval_s,
                settings.max_payload,
                settings.attachments_ok,
            )
            return True

        except Exception as exc:
            # Best-effort teardown so the next retry starts from a clean
            # slate. _teardown_handles releases the lock too.
            await self._teardown_handles()
            self._set_fatal_error(
                "nats_connect_error",
                f"NATS connect failed: {exc}",
                retryable=True,
            )
            logger.error(
                "[%s] Failed to connect to NATS: %s",
                self.name,
                exc,
                exc_info=True,
            )
            return False

    async def disconnect(self) -> None:
        """Stop the agent, close the NATS client, and release the lock.

        Idempotent — safe to call after a failed ``connect()`` or twice in
        a row during gateway shutdown. Preserves any fatal error state so
        callers can still inspect ``fatal_error_message`` after shutdown.
        """
        await self._teardown_handles()
        self._mark_disconnected()
        logger.info("[%s] Disconnected from NATS", self.name)

    async def _teardown_handles(self) -> None:
        """Shared cleanup for both connect-failure and disconnect paths.

        Order matters (design doc §9 "Shutdown"):

          1. Signal shutdown + cancel in-flight ``_on_prompt`` tasks so
             they unwind any awaits on the live NATS connection *before*
             we deregister the service or drop the socket. Skipping this
             surfaces ``CancelledError`` / "connection closed" noise from
             handlers that were mid-``stream.send`` when shutdown fires.
          2. ``agent.stop()`` — deregisters the micro service endpoint
             and stops the heartbeat publisher. Runs while ``nc`` is
             still open so the heartbeat task's final iteration can
             cleanly bail out instead of racing the socket close.
          3. ``nc.close()`` — drops the underlying NATS connection.
          4. Release the scoped lock so the next gateway instance on
             this host can register the same identity.
        """
        # Step 1 — drain in-flight prompt handlers. Materialize the
        # pending list first: ``cancel()`` schedules CancelledError at
        # the task's next await point, and the handler's finally-block
        # mutates ``_in_flight_handlers`` via ``discard()`` — iterating
        # the live set would risk "set changed size during iteration".
        self._shutdown_event.set()
        pending = [t for t in self._in_flight_handlers if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            # return_exceptions=True so a CancelledError from one task
            # doesn't prevent us from awaiting the others — teardown
            # must be all-or-nothing-complete, never all-or-nothing-started.
            await asyncio.gather(*pending, return_exceptions=True)
        self._in_flight_handlers.clear()

        # Clear any lingering stream handles so a late send() fails fast
        # rather than publishing onto a socket that's about to close.
        self._active_streams.clear()
        # Drop the per-session lock pool so a subsequent ``connect()``
        # starts from a clean slate — Locks held by cancelled tasks
        # wouldn't release cleanly otherwise and could deadlock a retry.
        self._session_locks.clear()

        if self._agent is not None:
            try:
                await self._agent.stop()
            except Exception as exc:
                logger.warning(
                    "[%s] Error stopping natsagent.Agent: %s",
                    self.name,
                    exc,
                    exc_info=True,
                )
            finally:
                self._agent = None

        if self._nc is not None:
            try:
                await self._nc.close()
            except Exception as exc:
                logger.warning(
                    "[%s] Error closing NATS connection: %s",
                    self.name,
                    exc,
                    exc_info=True,
                )
            finally:
                self._nc = None

        self._release_platform_lock()

    # ------------------------------------------------------------------
    # Inbound prompt handler — Phase 4
    # ------------------------------------------------------------------

    async def _on_prompt(self, envelope: "Envelope", stream: "PromptStream") -> None:
        """Per-prompt entry point dispatched by :class:`natsagent.Agent`.

        Sequence (design doc §6.2):

          1. Read ``envelope.session`` → ``chat_id`` (spec §5.1 optional
             field; falls back to ``session_default`` when unset).
          2. Decode any ``attachments`` into the hermes media cache so the
             downstream agent can read them via local paths (§8.1). Done
             *before* the session lock so attachment errors fail fast
             without blocking another same-session handler.
          3. Start the keep-alive task BEFORE acquiring the session lock
             — a queued handler still needs to emit acks so the caller
             doesn't timeout waiting its turn (§6.4).
          4. Acquire the per-``chat_id`` session lock. Queues concurrent
             same-session handlers so only one runs at a time — which is
             what eliminates the notify-cb-overwrite + stream-resolution
             races documented as Phase 6 shortcomings.
          5. Inside the lock: register the stream, build the MessageEvent,
             dispatch. For slash commands, reuse the gateway runner's
             dispatch via ``self._message_handler``. For text prompts,
             run the adapter-owned :class:`AIAgent` via
             ``_run_text_prompt`` (§6.3).
          6. Always unwind the stream, keep-alive, and task-tracking in
             the ``finally`` block so the SDK's terminator runs on a
             clean slate whether we succeeded, raised, or got cancelled
             during shutdown.
        """
        task = asyncio.current_task()
        if task is not None:
            self._in_flight_handlers.add(task)

        chat_id = (envelope.session or "").strip() or self._session_default()
        keepalive_task: Optional[asyncio.Task] = None
        stream_key: Optional[Tuple[str, int]] = None

        # Bind the contextvar BEFORE the try so any send fired from
        # ``_unpack_envelope`` (attachment error paths) still reaches the
        # caller's stream; ``_context_token`` is reset unconditionally in
        # the finally block below.
        _context_token = _current_stream.set(stream)

        try:
            # Unpack envelope OUTSIDE the session lock so a malformed
            # attachment from handler B fails fast even while handler A
            # is still running — otherwise a bad attachment could block
            # behind a long-running earlier prompt for no reason.
            prompt_text, media_urls, media_types, message_type = self._unpack_envelope(envelope)

            # Keep-alive emission keeps callers (§6.6 recommends 60 s
            # inactivity timeout) from dropping the subscription while
            # we're queued behind an earlier same-session handler OR
            # while the model is silent mid-reasoning inside the lock.
            keepalive_task = asyncio.create_task(
                self._run_keepalive(stream),
                name=f"nats-keepalive-{chat_id}",
            )

            # Per-session serialization. ``setdefault`` is safe on a
            # single event loop — there's no await between the dict
            # lookup and the insert, so two coroutines for the same
            # chat_id can't both insert a fresh Lock. The acquire below
            # is the yield point; the second handler sees the first's
            # Lock in the dict and awaits on it.
            session_lock = self._session_locks.setdefault(chat_id, asyncio.Lock())
            async with session_lock:
                # Register under a compound key so overlapping prompts
                # with the same ``envelope.session`` still don't collide in the
                # registry — belt-and-braces defense since the lock
                # should prevent overlap in the first place.
                stream_key = (chat_id, id(stream))
                self._active_streams[stream_key] = stream

                source = self.build_source(
                    chat_id=chat_id,
                    chat_name=chat_id,
                    chat_type="dm",
                    user_id=chat_id,
                    user_name=chat_id,
                )
                is_command = self._looks_like_command(prompt_text)
                # ``_looks_like_command`` lstrips before matching so ``"  /help"``
                # is still classified as a command. ``MessageEvent.is_command``
                # / ``get_command()`` in base.py, by contrast, require the
                # literal ``/`` at index 0 — leading whitespace would cause the
                # gateway's command registry to miss the dispatch and fall
                # through to the agent path. Canonicalize the text here when
                # we've decided it's a command so the two heuristics agree.
                event_text = prompt_text.lstrip() if is_command else prompt_text
                event = MessageEvent(
                    text=event_text,
                    message_type=MessageType.COMMAND if is_command else message_type,
                    source=source,
                    media_urls=media_urls,
                    media_types=media_types,
                )

                if is_command:
                    await self._dispatch_command(event, stream)
                else:
                    await self._run_text_prompt(event, stream, chat_id)

        except asyncio.CancelledError:
            # Gateway shutdown cancelled this handler. Propagate so the
            # SDK's `_on_prompt_request` finally-block still emits the
            # terminator — CancelledError is a BaseException subclass
            # that's NOT caught by the SDK's broad `except Exception`.
            raise
        except Exception:
            logger.exception("[%s] NATS prompt handler failed", self.name)
            # Re-raise so the SDK responds with a 500 error frame + terminator
            # (agent.py:270-272, §9.3).
            raise
        finally:
            if keepalive_task is not None:
                keepalive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keepalive_task
            if stream_key is not None:
                # Compound-keyed pop — always safe regardless of which
                # other handler happens to share the ``chat_id``.
                self._active_streams.pop(stream_key, None)
            # Reset contextvar BEFORE discarding the task so any final
            # callback that runs on this task's context unwinds cleanly.
            _current_stream.reset(_context_token)
            if task is not None:
                # ``discard`` (not ``remove``) — ``_teardown_handles`` may
                # have already called ``clear()`` if cancellation landed
                # before this finally-block ran.
                self._in_flight_handlers.discard(task)

    # ------------------------------------------------------------------
    # Session extraction — §3 / §5.6
    # ------------------------------------------------------------------

    def _session_default(self) -> str:
        """Return the fallback session value from settings when the caller
        omits ``envelope.session``.

        Isolated as a method so the test-only path where ``_settings`` is
        None (fatal-init adapter) still has a deterministic answer rather
        than an ``AttributeError``.
        """
        if self._settings is not None:
            return self._settings.session_default
        return DEFAULT_SESSION_DEFAULT

    # ------------------------------------------------------------------
    # Attachment round-trip — §8.1
    # ------------------------------------------------------------------

    def _unpack_envelope(
        self, envelope: "Envelope"
    ) -> Tuple[str, List[str], List[str], MessageType]:
        """Decode an :class:`Envelope` into the MessageEvent fields.

        Returns ``(prompt_text, media_urls, media_types, message_type)``.

        Attachment-decode failures surface as ``RuntimeError`` and are
        caught by ``_on_prompt`` → re-raised, which the SDK converts to a
        500 error frame per §9.3. Per-attachment partial success is not
        attempted: a malformed attachment invalidates the whole prompt
        from the caller's perspective (they don't see a "half the files
        worked" response).
        """
        prompt_text = getattr(envelope, "prompt", "") or ""
        raw_attachments = getattr(envelope, "attachments", None) or []

        media_urls: List[str] = []
        media_types: List[str] = []
        first_message_type: Optional[MessageType] = None

        for idx, att in enumerate(raw_attachments):
            filename = getattr(att, "filename", "") or f"attachment_{idx}"
            try:
                data = att.to_bytes()
            except Exception as exc:
                raise RuntimeError(
                    f"NATS: attachment #{idx} ({filename!r}) base64 decode failed: {exc}"
                ) from exc

            ext = Path(filename).suffix.lower()
            try:
                if ext in _IMAGE_EXTS:
                    path = cache_image_from_bytes(data, ext=ext or ".jpg")
                    mtype = MessageType.PHOTO
                elif ext in _AUDIO_EXTS:
                    path = cache_audio_from_bytes(data, ext=ext or ".ogg")
                    mtype = MessageType.AUDIO
                elif ext in _VIDEO_EXTS:
                    path = cache_video_from_bytes(data, ext=ext or ".mp4")
                    mtype = MessageType.VIDEO
                else:
                    path = cache_document_from_bytes(data, filename=filename)
                    mtype = MessageType.DOCUMENT
            except ValueError as exc:
                # cache_image_from_bytes raises ValueError when the bytes
                # don't look like a real image — caller sent us HTML or
                # garbage with a .jpg extension. Surface as a protocol
                # error so the SDK returns 400.
                raise RuntimeError(
                    f"NATS: attachment #{idx} ({filename!r}) failed validation: {exc}"
                ) from exc

            media_urls.append(path)
            media_types.append(mtype.value)
            if first_message_type is None:
                first_message_type = mtype

        message_type = first_message_type or MessageType.TEXT
        return prompt_text, media_urls, media_types, message_type

    # ------------------------------------------------------------------
    # Media enrichment — §8.1
    # ------------------------------------------------------------------

    async def _enrich_event_with_media(self, event: MessageEvent) -> MessageEvent:
        """Fold ``event.media_urls`` into ``event.text`` for the agent.

        The gateway's default path (``GatewayRunner._handle_message``)
        performs two enrichment steps before the agent runs: inline
        vision pre-analysis for images via
        :meth:`GatewayRunner._enrich_message_with_vision`, and a
        descriptive context-note for documents. NATS bypasses
        ``_handle_message`` by design (§6.1 api_server-style adapter
        ownership), so we replicate both steps here to match every other
        messaging platform's behavior byte-for-byte on the adapter hot
        path.

        Image handling: calls ``vision_analyze`` inline and prepends the
        description using the same message template as
        ``_enrich_message_with_vision`` (run.py:8127). Analysis failures
        degrade to the "couldn't see it" fallback note pointing the
        agent at ``vision_analyze`` so it can retry itself — matching
        the gateway's error path. Each image costs one extra
        vision-model round-trip; the alternative (note-only) was
        considered in the Phase 8 first-pass fix and rejected because
        it diverges from the canonical gateway behavior for no real
        cost saving (vision pre-analysis is how every other adapter
        presents images to the agent).

        Document / audio / video handling: a bracketed path-note using
        the same shape as ``_handle_message``'s document block
        (run.py:3895) — the canonical behavior doesn't actually inline
        text bytes either (the "included below" wording there is
        misleading historical note; the code only writes the note, not
        the content). The agent can call ``read_file`` when it wants
        the content; matching that here keeps the user-facing contract
        identical to Telegram / Discord / Slack.
        """
        if not event.media_urls:
            return event

        image_paths: List[Tuple[int, str]] = []
        other_notes: List[str] = []
        for idx, path in enumerate(event.media_urls):
            mtype = event.media_types[idx] if idx < len(event.media_types) else MessageType.DOCUMENT.value
            if mtype == MessageType.PHOTO.value:
                image_paths.append((idx, path))
            elif mtype == MessageType.VOICE.value or mtype == MessageType.AUDIO.value:
                other_notes.append(
                    f"[The user attached an audio file at {path}. "
                    f"Use the transcription tool if you need its contents.]"
                )
            elif mtype == MessageType.VIDEO.value:
                other_notes.append(
                    f"[The user attached a video file at {path}.]"
                )
            else:
                basename = Path(path).name
                other_notes.append(
                    f"[The user sent a document: '{basename}'. "
                    f"The file is saved at: {path}. "
                    f"Ask the user what they'd like you to do with it, "
                    f"or call read_file if the file type is text-readable.]"
                )

        image_notes = await self._analyze_image_attachments([p for _, p in image_paths])

        prefix_parts = image_notes + other_notes
        prefix = "\n\n".join(prefix_parts)
        enriched_text = f"{prefix}\n\n{event.text}" if event.text else prefix

        return MessageEvent(
            text=enriched_text,
            message_type=event.message_type,
            source=event.source,
            media_urls=event.media_urls,
            media_types=event.media_types,
        )

    async def _analyze_image_attachments(self, image_paths: List[str]) -> List[str]:
        """Run ``vision_analyze`` on each image and return the description notes.

        Extracted from :meth:`_enrich_event_with_media` so tests can
        mock the per-image analysis independently of the overall routing
        (one place to stub the expensive call). Matches
        :meth:`GatewayRunner._enrich_message_with_vision` output
        verbatim — same analysis prompt, same "here's what I can see"
        template, same fallback wording on failure.
        """
        if not image_paths:
            return []
        # Local import keeps the module importable in test harnesses that
        # don't install the vision tool's dependencies (matches the
        # gateway's lazy-import in ``_enrich_message_with_vision``).
        from tools.vision_tools import vision_analyze_tool

        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        notes: List[str] = []
        for path in image_paths:
            try:
                result_json = await vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                )
                result = json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    notes.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    notes.append(
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as exc:
                logger.error(
                    "[%s] NATS vision enrichment failed for %s: %s",
                    self.name,
                    path,
                    exc,
                )
                notes.append(
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )
        return notes

    # ------------------------------------------------------------------
    # Keep-alive — §6.4
    # ------------------------------------------------------------------

    async def _run_keepalive(self, stream: Any) -> None:
        """Emit ``{type:status, data:"ack"}`` every ``ack_keepalive_interval_s``.

        MVP behavior: fixed tick regardless of handler activity (design
        doc §6.4). Protocol §6.6 recommends callers default to a 60 s
        inactivity timeout, and our settings validator keeps this cadence
        below that — the worst-case outcome here is occasional redundant
        acks, never a caller-side timeout.

        Stops cleanly on cancellation. We don't re-raise send failures
        because a dead stream is caught on the next ``stream.send`` call
        in the pump/command paths, where the handler is already on an
        error path.
        """
        interval = self._settings.ack_keepalive_interval_s if self._settings else DEFAULT_ACK_KEEPALIVE_INTERVAL_S
        chunk_factory = getattr(natsagent, "StatusChunk", None)
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            if self._shutdown_event.is_set():
                return
            try:
                chunk = chunk_factory(status="ack") if chunk_factory is not None else {"status": "ack"}
                await stream.send(chunk)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                # Don't escalate — the main handler will either finish
                # normally and hit the same error on its next send, or
                # already be unwinding. Log at debug so noisy reconnects
                # don't spam prod logs.
                logger.debug(
                    "[%s] NATS keep-alive send failed (stream likely closed): %s",
                    self.name,
                    exc,
                )
                return

    # ------------------------------------------------------------------
    # Slash-command dispatch — reuses gateway COMMAND_REGISTRY
    # ------------------------------------------------------------------

    def _looks_like_command(self, prompt: str) -> bool:
        """Heuristic: is this envelope a slash command?

        Conservative — we only treat a leading ``/`` followed by a word
        character as a command. Prompts that begin with ``/`` but encode
        a path (e.g. ``/var/log/foo``) would be misclassified; callers
        can defeat the heuristic by prefixing with whitespace or a
        space, which is documented in the design.
        """
        stripped = (prompt or "").lstrip()
        if not stripped.startswith("/"):
            return False
        if len(stripped) < 2:
            return False
        # Valid slash-command first chars: a-z A-Z 0-9 _ ; reject things
        # like "//" or "/var/log" which clearly aren't commands.
        head = stripped[1]
        if not (head.isalnum() or head == "_"):
            return False
        # Reject file paths — commands never contain "/" in the token.
        # Matches MessageEvent.get_command()'s behavior in base.py:746.
        first_token = stripped.split(None, 1)[0]
        if "/" in first_token[1:]:
            return False
        return True

    async def _dispatch_command(self, event: MessageEvent, stream: Any) -> None:
        """Route a slash command through the gateway's dispatch registry.

        Design doc §10: commands flow through the existing
        ``COMMAND_REGISTRY`` / ``command.dispatch()`` pipeline. The
        gateway runner sets ``_message_handler`` to
        ``GatewayRunner._handle_message``, which returns the rendered
        response string for recognized commands. We wrap the reply in a
        ``ResponseChunk`` and publish it on the prompt stream.

        If ``_message_handler`` is unset (standalone tests, misconfigured
        gateway), emit a short error instead of going silent — the caller
        deserves to know the command didn't run.
        """
        if self._message_handler is None:
            await self._send_text(stream, "NATS: gateway has no message handler wired; "
                                          "command not dispatched.")
            return

        try:
            response = await self._message_handler(event)
        except Exception as exc:
            logger.exception("[%s] command dispatch failed", self.name)
            await self._send_text(
                stream, f"[hermes] command failed: {exc}"
            )
            return

        if response:
            await self._send_text(stream, str(response))

    # ------------------------------------------------------------------
    # Text prompt — adapter-owned AIAgent + streaming pump
    # ------------------------------------------------------------------

    async def _run_text_prompt(
        self, event: MessageEvent, stream: Any, chat_id: str
    ) -> None:
        """Run a text prompt through an adapter-owned AIAgent.

        Deltas land via ``stream_delta_callback`` on the agent's worker
        thread; we forward them through an ``asyncio.Queue`` into a pump
        task that awaits ``stream.send`` on the event-loop thread (§6.3).

        Rationale for bypassing ``self._message_handler``: the default
        path constructs a ``GatewayStreamConsumer`` which edits a single
        platform message. NATS is publish-each-chunk — edits are
        meaningless on the wire. Building the agent here mirrors
        ``api_server.py``'s pattern (§6.1) while keeping slash commands
        on the gateway's dispatch path for registry consistency.

        The final assistant text (returned by ``run_conversation``) is
        the authoritative response; if streaming was disabled for any
        reason, it still lands on the caller via ``_send_text``. When
        streaming is live, the same text has already arrived via the
        pump — we detect that and skip the duplicate publish.

        ``event.media_urls`` is folded into the user message here (§8.1)
        via :meth:`_enrich_event_with_media`, which mirrors
        ``GatewayRunner._enrich_message_with_vision`` (inline pre-analysis
        of each image through the vision tool) plus the document/audio/
        video context-notes from ``GatewayRunner._handle_message``'s
        attachment block. Matching the canonical path byte-for-byte keeps
        the user-facing contract identical to every other platform.
        """
        event = await self._enrich_event_with_media(event)
        loop = asyncio.get_running_loop()
        delta_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        streamed_anything = threading.Event()

        def _delta_callback(text: Optional[str]) -> None:
            # The agent fires ``stream_delta_callback(None)`` to signal
            # CLI renderers to close a response box before tool use.
            # For NATS each chunk is its own publish, so None carries
            # no meaning on the wire — drop it.
            if not text:
                return
            streamed_anything.set()
            try:
                loop.call_soon_threadsafe(delta_queue.put_nowait, text)
            except RuntimeError:
                # Loop is closing — nothing we can do; drop the delta.
                pass

        pump_task = asyncio.create_task(
            self._pump_deltas(delta_queue, stream),
            name=f"nats-pump-{chat_id}",
        )

        try:
            result = await loop.run_in_executor(
                None,
                self._run_agent_sync,
                event,
                chat_id,
                _delta_callback,
                loop,
            )
        finally:
            # Signal end-of-stream and wait for the pump to drain so no
            # late deltas sneak out AFTER the SDK's terminator fires.
            await delta_queue.put(None)
            try:
                await asyncio.wait_for(pump_task, timeout=5.0)
            except asyncio.TimeoutError:
                pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pump_task

        final_text = _final_response_text(result)
        if final_text and not streamed_anything.is_set():
            # Streaming was disabled or the model produced its answer
            # entirely via non-streamed paths (e.g. tool-only turns that
            # finalize in the last message). Deliver the final text as a
            # single ResponseChunk so the caller gets the answer.
            await self._send_text(stream, final_text)

    async def _pump_deltas(self, queue: asyncio.Queue, stream: Any) -> None:
        """Drain deltas from ``queue`` and publish each as a ResponseChunk.

        Terminates when ``None`` is received (end-of-stream sentinel).
        Errors from ``stream.send`` surface as logs — by the time we're
        pumping, the handler is committed and the pump can't meaningfully
        escalate. A broken socket will also surface in
        ``_run_text_prompt``'s final-text fallback, which raises to the
        SDK.
        """
        chunk_factory = getattr(natsagent, "ResponseChunk", None)
        while True:
            delta = await queue.get()
            if delta is None:
                return
            try:
                chunk = chunk_factory(text=delta) if chunk_factory is not None else delta
                await stream.send(chunk)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    "[%s] NATS pump send failed (stream likely closed): %s",
                    self.name,
                    exc,
                )
                return

    def _run_agent_sync(
        self,
        event: MessageEvent,
        chat_id: str,
        stream_delta_callback: Any,
        loop: "asyncio.AbstractEventLoop",
    ) -> Any:
        """Build an :class:`AIAgent` and run one conversation turn synchronously.

        Called from ``run_in_executor`` so ``run_conversation`` (a
        long-running sync method) doesn't block the event loop. The
        returned ``result`` dict is consumed by ``_run_text_prompt`` to
        fall back to non-streamed delivery when the pump saw no deltas.

        ``loop`` is the adapter's event loop, captured in ``_run_text_prompt``
        and threaded through here so the approval notify callback can
        schedule ``request_interaction`` coroutines back onto it from this
        worker thread. ``asyncio.get_running_loop()`` inside the executor
        would raise (no loop on this thread).
        """
        from run_agent import AIAgent
        from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model, _load_gateway_config
        from hermes_cli.tools_config import _get_platform_tools
        from gateway.session import build_session_key
        from gateway.platforms.base import dispatch_approval_via_request_interaction
        from tools.approval import (
            register_gateway_notify,
            unregister_gateway_notify,
            set_current_session_key,
            reset_current_session_key,
        )

        user_config = _load_gateway_config()
        runtime_kwargs = _resolve_runtime_agent_kwargs()
        model = _resolve_gateway_model(user_config)
        enabled_toolsets = sorted(_get_platform_tools(user_config, Platform.NATS.value))

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

        # Load fallback provider chain so NATS matches Telegram/Discord
        # behavior when the primary provider errors mid-run.
        try:
            from gateway.run import GatewayRunner
            fallback_model = GatewayRunner._load_fallback_model()
        except Exception:
            fallback_model = None

        session_key = build_session_key(
            event.source,
            group_sessions_per_user=True,
        )

        # Best-effort session DB wiring so NATS conversations show up
        # under ``hermes sessions list`` alongside CLI/Telegram sessions.
        session_db = None
        try:
            from hermes_state import SessionDB
            session_db = SessionDB()
        except Exception as exc:
            logger.debug("[%s] SessionDB unavailable: %s", self.name, exc)

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            enabled_toolsets=enabled_toolsets,
            session_id=session_key,
            platform=Platform.NATS.value,
            user_id=event.source.user_id,
            gateway_session_key=session_key,
            stream_delta_callback=stream_delta_callback,
            session_db=session_db,
            fallback_model=fallback_model,
        )

        # Load prior history so multi-turn conversations over the same
        # session stay coherent.
        conversation_history: List[Dict[str, Any]] = []
        if session_db is not None:
            try:
                conversation_history = session_db.get_messages_as_conversation(session_key) or []
            except Exception as exc:
                logger.debug("[%s] loading history for %s failed: %s", self.name, session_key, exc)

        # Register an approval notify callback scoped to this session.
        # ``check_all_command_guards`` (tools/approval.py) fires it from the
        # agent's worker thread when a dangerous command hits; we schedule
        # ``request_interaction`` on ``loop`` and let the helper call
        # ``resolve_gateway_approval`` when the caller replies — unblocking
        # the agent thread waiting on the ApprovalEntry event.
        def _nats_approval_notify(approval_data: dict) -> None:
            # Capture the entry id SYNCHRONOUSLY — contextvars don't
            # propagate across ``asyncio.run_coroutine_threadsafe``, so
            # the scheduled coroutine can only see the id via closure.
            # Without this, ``resolve_gateway_approval`` falls back to
            # FIFO-oldest pop, which cross-routes choices for parallel
            # subagents hitting dangerous commands in the same session.
            from tools.approval import get_current_approval_entry_id
            captured_entry_id = get_current_approval_entry_id()

            try:
                dispatched = dispatch_approval_via_request_interaction(
                    self,
                    chat_id,
                    session_key,
                    approval_data,
                    loop,
                    timeout=_approval_timeout_from_config(),
                    entry_id=captured_entry_id,
                )
            except Exception as exc:
                logger.error(
                    "[%s] NATS approval notify failed (session=%s): %s",
                    self.name,
                    session_key,
                    exc,
                )
                dispatched = False
            if not dispatched:
                # Dispatch either found the adapter didn't override (cannot
                # happen for NATS — it always overrides) or scheduling on
                # ``loop`` raised (only during a shutdown race, when the loop
                # is already closed). Fail safe: resolve as "deny" right now
                # so the agent thread blocked on ``entry.event.wait()``
                # unblocks immediately instead of hanging for the full
                # ``gateway_timeout`` (default 300 s) before the framework
                # times out by itself. Pass the captured entry_id so we
                # resolve the specific entry, not FIFO-oldest.
                try:
                    from tools.approval import resolve_gateway_approval
                    resolve_gateway_approval(
                        session_key, "deny", entry_id=captured_entry_id,
                    )
                except Exception as exc:
                    logger.error(
                        "[%s] Fallback resolve_gateway_approval failed "
                        "(session=%s): %s",
                        self.name,
                        session_key,
                        exc,
                    )

        # Bind the approval session key on this worker thread's contextvar
        # so ``get_current_session_key`` inside ``check_all_command_guards``
        # finds the right session key — contextvars set on the main thread
        # don't propagate into ``run_in_executor`` (we don't go through
        # ``_run_in_executor_with_context``). Reset in the finally below.
        approval_token = set_current_session_key(session_key)
        register_gateway_notify(session_key, _nats_approval_notify)
        try:
            return agent.run_conversation(
                user_message=event.text,
                conversation_history=conversation_history,
                task_id=chat_id,
            )
        finally:
            unregister_gateway_notify(session_key)
            reset_current_session_key(approval_token)

    # ------------------------------------------------------------------
    # Outbound — publish a ResponseChunk on the stream for a given chat_id
    # ------------------------------------------------------------------

    def _resolve_stream(self, chat_id: str) -> Optional[Any]:
        """Return the PromptStream that ``chat_id`` should publish onto.

        Lookup order (T5.0):
          1. ``_current_stream`` contextvar — set by ``_on_prompt`` and
             inherited by every coroutine / executor thread spawned from
             that handler (``run_in_executor`` and ``asyncio.Task``
             default-copy the parent's context). This is the race-safe
             path: two concurrent prompts for the same ``envelope.session`` each
             see their own handler's stream here regardless of which
             overwrote the other in ``_active_streams``.
          2. ``_active_streams`` compound-key lookup by ``chat_id`` — the
             fallback for the narrow case where a send is scheduled
             outside the handler's context (e.g. via
             ``asyncio.run_coroutine_threadsafe`` from a worker thread
             whose context didn't propagate). Returns whichever stream
             happens to be registered first; ambiguous under concurrent
             same-``chat_id`` prompts, but the contextvar path handles
             those before this fallback runs.
        """
        ctx_stream = _current_stream.get()
        if ctx_stream is not None:
            return ctx_stream
        for (cid, _sid), stream in self._active_streams.items():
            if cid == chat_id:
                return stream
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Publish ``content`` as a :class:`ResponseChunk` on the prompt stream.

        NATS has no out-of-band delivery — every response lands on the
        caller's reply subject, which only stays open for the lifetime
        of the originating ``_on_prompt`` handler. If the stream for
        ``chat_id`` isn't registered we return a non-retryable
        ``SendResult`` rather than silently dropping the message; this
        surfaces logic bugs (tool firing after handler exit) instead of
        burying them.
        """
        stream = self._resolve_stream(chat_id)
        if stream is None:
            return SendResult(
                success=False,
                error=f"no active NATS stream for chat_id={chat_id}",
            )
        try:
            await self._send_text(stream, content)
        except Exception as exc:
            return SendResult(
                success=False,
                error=f"stream.send failed: {exc}",
                retryable=False,
            )
        return SendResult(success=True, message_id=uuid.uuid4().hex)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Publish an image as a :class:`ResponseChunk` with one attachment.

        Wraps the file in :meth:`Attachment.from_path` (base64 at the
        constructor, per the SDK's envelope.py) and sends one chunk
        carrying ``caption`` as ``text`` + the image in ``attachments``.
        The caller's NATS client sees it as the §6.3 keyed-object form
        ``{type: response, data: {text, attachments: [...]}}``.
        """
        return await self._send_attachment(
            chat_id=chat_id,
            file_path=image_path,
            caption=caption,
            kind="image",
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
        """Publish a generic file as a :class:`ResponseChunk` attachment.

        ``file_name`` overrides the filename that lands on the wire — useful
        for tool paths that stage downloads under a hash and want to
        present the original name to the caller.
        """
        return await self._send_attachment(
            chat_id=chat_id,
            file_path=file_path,
            caption=caption,
            kind="document",
            override_filename=file_name,
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
        """Publish audio as an attachment. v0.2 has no voice/audio distinction
        on the wire — the caller interprets by filename extension (§5.2).
        """
        return await self._send_attachment(
            chat_id=chat_id,
            file_path=audio_path,
            caption=caption,
            kind="voice",
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
        """Publish video as an attachment (same wire shape as images — §5.2)."""
        return await self._send_attachment(
            chat_id=chat_id,
            file_path=video_path,
            caption=caption,
            kind="video",
        )

    async def _send_attachment(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str],
        kind: str,
        override_filename: Optional[str] = None,
    ) -> SendResult:
        """Build a one-attachment :class:`ResponseChunk` and publish it.

        Centralized so the four ``send_*`` helpers share filename
        resolution, file-existence checks, the attachment construction,
        and the shared race-safe stream lookup. The ``kind`` parameter
        is used only for error messages — the v0.2 wire carries the
        attachment identically regardless of media type.
        """
        stream = self._resolve_stream(chat_id)
        if stream is None:
            return SendResult(
                success=False,
                error=f"no active NATS stream for chat_id={chat_id}",
            )

        path = Path(file_path)
        if not path.exists():
            return SendResult(
                success=False,
                error=f"{kind} path not found: {file_path}",
            )

        attachment_factory = getattr(natsagent, "Attachment", None)
        chunk_factory = getattr(natsagent, "ResponseChunk", None)
        if attachment_factory is None or chunk_factory is None:
            return SendResult(
                success=False,
                error="natsagent SDK missing Attachment / ResponseChunk",
            )

        try:
            if override_filename is not None:
                # from_path would pin ``path.name``; honor the caller's
                # explicit override by reading bytes then building via
                # from_bytes.
                attachment = attachment_factory.from_bytes(
                    override_filename, path.read_bytes()
                )
            else:
                attachment = attachment_factory.from_path(str(path))
        except Exception as exc:
            return SendResult(
                success=False,
                error=f"{kind} attachment build failed: {exc}",
            )

        try:
            chunk = chunk_factory(text=caption or "", attachments=[attachment])
            await stream.send(chunk)
        except Exception as exc:
            return SendResult(
                success=False,
                error=f"stream.send failed: {exc}",
                retryable=False,
            )
        return SendResult(success=True, message_id=uuid.uuid4().hex)

    async def _send_text(self, stream: Any, content: str) -> None:
        """Wrap a string in a ResponseChunk and publish it.

        Centralized so the command, pump, and fallback-delivery paths
        share one chunk construction — the ``ResponseChunk`` factory
        lookup is MagicMock-tolerant under test, where the module is
        patched in ``tests/gateway/conftest.py``.
        """
        chunk_factory = getattr(natsagent, "ResponseChunk", None)
        if chunk_factory is None:
            await stream.send(content)
            return
        await stream.send(chunk_factory(text=content))

    async def request_interaction(
        self,
        chat_id: str,
        prompt: str,
        *,
        kind: str,
        timeout: float,
    ) -> Optional[str]:
        """Ask the caller mid-stream and await their reply via ``stream.ask``.

        Protocol §7 round-trip: publishes a ``query`` chunk into the active
        prompt stream, allocates a fresh reply inbox, and blocks here until
        the caller publishes exactly one reply (or the timeout elapses).
        The underlying response stream stays open across the round-trip —
        the caller keeps iterating the prompt's async iterator while this
        handler awaits, so no keep-alive disruption is needed.

        Returns ``None`` on :class:`natsagent.QueryTimeout` or when no
        active stream can be resolved for ``chat_id``; the base-class
        contract (§7.2 of the design doc) asks adapters to distinguish
        "no answer" from "delivery failed" by raising only on programmer
        error. ``kind`` is accepted for the :class:`BasePlatformAdapter`
        signature but not wired into the query — v0.2 has no per-kind
        field on the wire, so the adapter just forwards the prompt text.
        """
        stream = self._resolve_stream(chat_id)
        if stream is None:
            logger.warning(
                "[%s] request_interaction: no active stream for chat_id=%s "
                "(kind=%s) — returning None so caller can fail safe",
                self.name,
                chat_id,
                kind,
            )
            return None

        query_timeout_cls = getattr(natsagent, "QueryTimeout", None)
        try:
            reply = await stream.ask(prompt, timeout=timeout)
        except Exception as exc:
            # Distinguish QueryTimeout ("caller stayed silent") from any
            # other ask() failure ("stream/socket problem"). The former is
            # a clean "no answer" per §7.3; the latter is a transport
            # error that the caller should fail safe on — both map to
            # None here, but we log at different levels so ops can tell
            # them apart.
            if query_timeout_cls is not None and isinstance(exc, query_timeout_cls):
                logger.info(
                    "[%s] request_interaction: caller did not reply within %ss "
                    "(chat_id=%s, kind=%s)",
                    self.name,
                    timeout,
                    chat_id,
                    kind,
                )
            else:
                logger.warning(
                    "[%s] request_interaction: stream.ask failed "
                    "(chat_id=%s, kind=%s): %s",
                    self.name,
                    chat_id,
                    kind,
                    exc,
                )
            return None

        reply_text = getattr(reply, "prompt", None)
        if isinstance(reply_text, str):
            return reply_text
        # Real SDK always returns an Envelope whose ``prompt`` is a string;
        # this branch only trips on mock substitutes. Return the str() so
        # downstream parsers have something to work with rather than None.
        return str(reply_text) if reply_text is not None else None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return minimal DM-style chat info for session-key construction.

        The NATS wire has no richer chat concept — every prompt is a
        direct request/reply, so ``chat_type="dm"`` is always the right
        answer (design doc §3). The name mirrors the caller-supplied
        ``envelope.session`` string, which is what ``build_session_key`` uses
        downstream to key sessions.
        """
        return {"name": chat_id, "type": "dm"}

    def format_message(self, content: str) -> str:
        """No-op: NATS carries plain-text chunks verbatim (§6.3).

        Override inherited from ``BasePlatformAdapter`` so the behavior
        is documented rather than accidental — any future platform
        formatter that assumes the base default is a no-op can still
        rely on that here.
        """
        return content


# ----------------------------------------------------------------------
# Module-level helpers (private; kept out of the class so tests can
# exercise them without constructing a full adapter).
# ----------------------------------------------------------------------


def _approval_timeout_from_config() -> float:
    """Return the gateway approval timeout in seconds from config.yaml.

    Mirrors the value used by ``tools/approval.py::check_all_command_guards``
    so the adapter's :meth:`request_interaction` round-trip and the agent
    thread's ``entry.event.wait()`` share the same deadline — avoids a
    stream.ask() that keeps waiting after the agent already timed out.
    Defaults to 300 s on any read failure so callers never hang forever.
    """
    try:
        from tools.approval import _get_approval_config  # noqa: WPS437 (private, but adapter-local)
        timeout = _get_approval_config().get("gateway_timeout", 300)
        return float(int(timeout))
    except Exception:
        return 300.0


def _final_response_text(result: Any) -> str:
    """Return the assistant's final text from a ``run_conversation`` result.

    Accepts both the dict shape (``{"final_response": ...}``) and the
    bare-string shape some code paths return. Empty / None results fold
    to an empty string so callers can safely do ``if final_text``.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        value = result.get("final_response")
        if isinstance(value, str):
            return value
    return ""
