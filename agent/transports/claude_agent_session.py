"""Synchronous facade over the async ``claude-agent-sdk`` client.

``ClaudeSDKClient`` is asyncio-only; Hermes' conversation loop is entirely
synchronous (AGENTS.md § Agent Loop).  This module owns the bridge: one
``ClaudeAgentSession`` per :class:`AIAgent`, holding one long-lived
``ClaudeSDKClient`` on a dedicated event-loop thread it created itself.

Two rules make the difference between this working and wedging:

* The client is constructed *and* connected on the owned loop, and every
  later call is submitted back to that same loop.  ``anyio`` memory streams
  and subprocess handles are bound to the loop that created them; touching
  them from another loop (or from a bare ``asyncio.run``) corrupts the
  transport.
* A turn drains the response stream past ``ResultMessage`` rather than
  stopping on it.  ``receive_response()`` returns the moment the result
  arrives, which drops trailing frames (late tool results, the post-result
  system events).  We keep pulling until the stream goes quiet.

Projection into Hermes' message shape deliberately does NOT happen here —
SDK messages are handed to the caller's thread through a queue so the
Hermes-side callbacks run on the turn thread, exactly like every other
runtime.  See :mod:`agent.claude_runtime`.

The public surface mirrors ``CodexAppServerSession``: ``ensure_started``,
``run_turn``, ``request_interrupt``, ``set_model``, ``close``.

``claude_agent_sdk`` is an optional extra and is imported lazily, so this
module imports cleanly without it.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# How long a turn keeps reading after ``ResultMessage`` before deciding the
# stream is quiet. The CLI emits the result before its trailing frames are
# flushed, so stopping on the result truncates them.
DEFAULT_TRAILING_DRAIN_SECONDS = 1.0

DEFAULT_START_TIMEOUT_SECONDS = 60.0
DEFAULT_TURN_TIMEOUT_SECONDS = 1800.0
DEFAULT_CLOSE_TIMEOUT_SECONDS = 15.0
# Bound on a control-plane round trip (interrupt / set_model). These go to a
# live subprocess; a hung one must not block the turn thread forever.
DEFAULT_CONTROL_TIMEOUT_SECONDS = 30.0


class ClaudeAgentSessionError(RuntimeError):
    """Raised for session-level failures (startup, teardown, wedged turn)."""


def is_result_message(message: Any) -> bool:
    """True when *message* is the SDK's terminal ``ResultMessage``.

    Matched by class name rather than ``isinstance``: this module must behave
    identically against the real optional extra and against the stand-in
    module the suite installs when the extra is absent.
    """
    return type(message).__name__ == "ResultMessage"


class ClaudeAgentSession:
    """One ``ClaudeSDKClient`` on one owned event-loop thread.

    Not thread-safe for concurrent turns — one caller drives it at a time,
    matching how ``run_conversation()`` is structured.  ``request_interrupt``
    and ``close`` are the exceptions and may be called from another thread.
    """

    def __init__(
        self,
        *,
        options_factory: Callable[[], Any],
        client_factory: Optional[Callable[..., Any]] = None,
        transport_factory: Optional[Callable[[Any], Any]] = None,
        start_timeout: float = DEFAULT_START_TIMEOUT_SECONDS,
        turn_timeout: float = DEFAULT_TURN_TIMEOUT_SECONDS,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
        trailing_drain_seconds: float = DEFAULT_TRAILING_DRAIN_SECONDS,
    ) -> None:
        self._options_factory = options_factory
        self._client_factory = client_factory
        self._transport_factory = transport_factory
        self._start_timeout = start_timeout
        self._turn_timeout = turn_timeout
        self._close_timeout = close_timeout
        self._trailing_drain_seconds = trailing_drain_seconds

        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._closed = False
        self._session_id: Optional[str] = None
        self._turn_count = 0
        # Temp CLAUDE_CONFIG_DIR holding a materialized resume, when this
        # session was started from a SessionStore-backed transcript.
        self._materialized: Any = None

    # ---------- introspection ----------

    @property
    def started(self) -> bool:
        return self._client is not None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def session_id(self) -> Optional[str]:
        """The SDK's own session id, once a turn has reported one.

        PR5 reads this to drive ``resume`` / the ``SessionStore`` mirror.
        """
        return self._session_id

    def note_session_id(self, session_id: Optional[str]) -> None:
        """Record the SDK session id observed by the caller's projector."""
        if session_id and session_id != self._session_id:
            self._session_id = str(session_id)

    # ---------- lifecycle ----------

    def ensure_started(self) -> None:
        """Spawn the loop thread and connect the client. Idempotent."""
        with self._lock:
            if self._closed:
                raise ClaudeAgentSessionError(
                    "Claude Agent SDK session is closed; build a new one."
                )
            if self._client is not None:
                return

            loop = asyncio.new_event_loop()
            ready = threading.Event()
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop, ready),
                name="hermes-claude-agent-sdk",
                daemon=True,
            )
            thread.start()
            if not ready.wait(self._start_timeout):
                loop.call_soon_threadsafe(loop.stop)
                raise ClaudeAgentSessionError(
                    "Claude Agent SDK event loop failed to start "
                    f"within {self._start_timeout:.0f}s."
                )
            self._loop = loop
            self._thread = thread

            try:
                self._client = self._submit(
                    self._connect(), timeout=self._start_timeout
                )
            except BaseException:
                # A half-started session must not linger: the loop thread and
                # any spawned CLI child are unreachable once __init__ returns
                # without a client.
                self._teardown_loop()
                raise
            logger.info(
                "claude_agent_sdk session connected (loop thread=%s)",
                thread.name,
            )

    def close(self) -> None:
        """Disconnect the client, stop the loop, join the thread. Idempotent.

        Safe from ``__del__`` / atexit: every step is independently guarded and
        joining is skipped when called from the loop thread itself.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None

        if client is not None:
            try:
                self._submit(client.disconnect(), timeout=self._close_timeout)
            except Exception:
                logger.debug("claude_agent_sdk disconnect failed", exc_info=True)
        if self._materialized is not None:
            # Ordered after disconnect: the child is pointed at this directory
            # and must exit before it is removed. It holds a copy of the
            # user's credentials, so it is not left behind on a failed close.
            try:
                self._submit(
                    self._cleanup_materialized(), timeout=self._close_timeout
                )
            except Exception:
                logger.debug("claude_agent_sdk resume cleanup failed", exc_info=True)
        self._teardown_loop()

    def __enter__(self) -> "ClaudeAgentSession":
        self.ensure_started()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        try:
            self.close()
        except Exception:
            pass

    # ---------- control plane ----------

    def request_interrupt(self) -> bool:
        """Ask the CLI to abort the active turn. Never raises.

        Only half of the interrupt: the caller must keep draining until the
        interrupted response's ``ResultMessage`` lands before issuing the next
        ``query()``. ``run_turn`` does that automatically — it does not return
        early on interrupt.
        """
        client = self._client
        if client is None or self._closed:
            return False
        try:
            self._submit(client.interrupt(), timeout=DEFAULT_CONTROL_TIMEOUT_SECONDS)
            return True
        except Exception:
            logger.debug("claude_agent_sdk interrupt failed", exc_info=True)
            return False

    def set_model(self, model: Optional[str]) -> bool:
        """Switch the model for subsequent turns. Never raises."""
        client = self._client
        if client is None or self._closed:
            return False
        try:
            self._submit(
                client.set_model(model), timeout=DEFAULT_CONTROL_TIMEOUT_SECONDS
            )
            return True
        except Exception:
            logger.debug("claude_agent_sdk set_model failed", exc_info=True)
            return False

    # ---------- per-turn ----------

    def run_turn(
        self,
        prompt: str,
        *,
        on_message: Callable[[Any], None],
        timeout: Optional[float] = None,
    ) -> int:
        """Send *prompt* and block until the response stream goes quiet.

        ``on_message`` is invoked once per SDK message, in arrival order, on
        the **calling** thread — Hermes' display/tool callbacks are
        thread-affine and must not run on the SDK's event loop.

        Returns the number of messages delivered.  Raises ``TimeoutError``
        when the turn blows its deadline (the caller should retire the
        session) and re-raises whatever the SDK raised otherwise.
        """
        self.ensure_started()
        inbox: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        future = asyncio.run_coroutine_threadsafe(
            self._pump_turn(prompt, inbox), self._require_loop()
        )
        self._turn_count += 1

        deadline = time.monotonic() + (timeout or self._turn_timeout)
        delivered = 0
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Claude Agent SDK turn exceeded "
                        f"{timeout or self._turn_timeout:.0f}s without completing."
                    )
                try:
                    kind, payload = inbox.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    continue
                if kind == "message":
                    delivered += 1
                    on_message(payload)
                    continue
                if kind == "error":
                    raise payload
                break
        except BaseException:
            future.cancel()
            raise
        # The pump has already signalled completion; this only surfaces an
        # exception raised after the last message was queued.
        future.result(timeout=self._close_timeout)
        return delivered

    # ---------- internals ----------

    def _run_loop(
        self, loop: asyncio.AbstractEventLoop, ready: threading.Event
    ) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.call_soon(ready.set)
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is None or loop.is_closed():
            raise ClaudeAgentSessionError(
                "Claude Agent SDK session has no running event loop."
            )
        return loop

    def _submit(self, coro: Any, *, timeout: float) -> Any:
        """Run *coro* on the owned loop and wait, bounded by *timeout*."""
        future = asyncio.run_coroutine_threadsafe(coro, self._require_loop())
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def _materialize(self, options: Any) -> Any:
        """Resolve a ``session_store`` + ``resume`` pair into real options.

        ``ClaudeSDKClient.connect()`` skips ``materialize_resume_session()``
        whenever a custom transport is supplied — the materialized options
        would never reach a transport that was already constructed, so the SDK
        does not bother.  Subscription mode *requires* the custom transport
        (it is the only thing that removes a stray ``ANTHROPIC_API_KEY`` from
        the child environment), so the materialization has to happen here
        instead, one step earlier: before the transport is built, so both the
        transport and the client see the same repointed options.

        What the SDK's helper does, and why none of it can be skipped: it
        loads the session from the store, writes it back out as JSONL into a
        temp directory laid out like ``~/.claude/`` (the CLI only knows how to
        resume from a local file), copies the caller's auth files into that
        directory with ``refreshToken`` redacted, materializes every subagent
        transcript ``list_subkeys()`` reports, and hands back the directory to
        point ``CLAUDE_CONFIG_DIR`` at.

        Returns ``(options, materialized_or_None)``.  When nothing needed
        materializing — no store, no resume, or an empty/unknown session — the
        options come back untouched and ``CLAUDE_CONFIG_DIR`` passes through
        to the child exactly as PR7 arranged.
        """
        if getattr(options, "session_store", None) is None:
            return options, None
        if not getattr(options, "resume", None) and not getattr(
            options, "continue_conversation", False
        ):
            return options, None
        from claude_agent_sdk._internal.session_resume import (
            apply_materialized_options,
            materialize_resume_session,
        )

        materialized = await materialize_resume_session(options)
        if materialized is None:
            return options, None
        # Repoints env["CLAUDE_CONFIG_DIR"] at the temp dir and rewrites
        # `resume` to the resolved id. Handing THESE options to the client (not
        # just to the transport) also keeps the transcript mirror correct: the
        # SDK resolves the batcher's projects_dir from `options.env`, so it
        # must see the same temp dir the subprocess writes into.
        return apply_materialized_options(options, materialized), materialized

    async def _connect(self) -> Any:
        factory = self._client_factory
        if factory is None:
            # Deferred so this module imports without the optional extra.
            from claude_agent_sdk import ClaudeSDKClient

            factory = ClaudeSDKClient
        options = self._options_factory()
        if self._transport_factory is not None:
            options, self._materialized = await self._materialize(options)
        kwargs: dict = {"options": options}
        if self._transport_factory is not None:
            # The transport owns the child environment; without one the SDK
            # spawns from a copy of os.environ, where a higher-precedence
            # credential would outrank the user's subscription.
            transport = self._transport_factory(options)
            if transport is not None:
                kwargs["transport"] = transport
        try:
            client = factory(**kwargs)
            await client.connect()
        except BaseException:
            # A failed connect leaves nobody to clean up the temp config dir,
            # and it contains credentials.
            await self._cleanup_materialized()
            raise
        return client

    async def _cleanup_materialized(self) -> None:
        """Remove the temp resume dir. Never raises.

        ``ClaudeSDKClient.disconnect()`` cleans up only what *it* materialized,
        and with a custom transport it materializes nothing — so the directory
        (which holds a copy of the user's credentials) is ours to remove.
        """
        materialized = self._materialized
        self._materialized = None
        if materialized is None:
            return
        try:
            await materialized.cleanup()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("claude_agent_sdk resume temp cleanup failed", exc_info=True)

    async def _pump_turn(
        self, prompt: str, inbox: "queue.Queue[tuple[str, Any]]"
    ) -> None:
        """Drive one query and forward every message to the caller's queue."""
        client = self._client
        if client is None:
            inbox.put(
                ("error", ClaudeAgentSessionError("Claude Agent SDK session is closed."))
            )
            return
        iterator = None
        try:
            await client.query(prompt)
            iterator = client.receive_messages().__aiter__()
            saw_result = False
            while True:
                try:
                    if saw_result:
                        # Past the result the stream never ends on its own, so
                        # quiet-for-a-beat is the only end-of-response signal.
                        message = await asyncio.wait_for(
                            iterator.__anext__(), self._trailing_drain_seconds
                        )
                    else:
                        message = await iterator.__anext__()
                except (StopAsyncIteration, asyncio.TimeoutError):
                    break
                if is_result_message(message):
                    saw_result = True
                inbox.put(("message", message))
            inbox.put(("done", None))
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller
            inbox.put(("error", exc))
        finally:
            if iterator is not None:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # pragma: no cover - generator teardown
                        logger.debug(
                            "claude_agent_sdk stream aclose failed", exc_info=True
                        )

    def _teardown_loop(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:  # pragma: no cover - already stopping
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._close_timeout)
            if thread.is_alive():  # pragma: no cover - wedged child
                logger.warning(
                    "claude_agent_sdk loop thread did not exit within %.0fs",
                    self._close_timeout,
                )


__all__ = [
    "ClaudeAgentSession",
    "ClaudeAgentSessionError",
    "DEFAULT_TRAILING_DRAIN_SECONDS",
    "is_result_message",
]
