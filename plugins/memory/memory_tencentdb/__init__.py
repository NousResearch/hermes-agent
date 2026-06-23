"""memory-tencentdb Memory Provider — MemoryProvider interface for Hermes.

Four-layer memory system (L0 conversation, L1 extraction, L2 scene blocks,
L3 persona synthesis) accessed via local Node.js Gateway sidecar.

The Gateway runs the memory-tencentdb Core engine (the same engine used by
the OpenClaw plugin) as an HTTP service. This provider translates Hermes
lifecycle events into Gateway API calls.

Config via environment variables:
  MEMORY_TENCENTDB_GATEWAY_HOST — Gateway host (default: 127.0.0.1)
  MEMORY_TENCENTDB_GATEWAY_PORT — Gateway port (default: 8420)
  MEMORY_TENCENTDB_GATEWAY_CMD  — Command to start the Gateway (optional; if
                                  unset, the provider auto-discovers
                                  ``src/gateway/server.ts`` next to the plugin
                                  checkout or under ``$HOME``)

The on-disk data directory (L0~L3 storage) is owned by the Gateway, not by
this provider. Point the Gateway at a custom location with ``TDAI_DATA_DIR``
(read directly by ``src/gateway/config.ts``); otherwise it falls back to
``~/.memory-tencentdb/memory-tdai`` (with legacy fallback to ``~/memory-tdai``
if it still exists). This provider no longer carries its own data-dir default
or env var — a single source of truth prevents the two layers from drifting
apart.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .client import MemoryTencentdbSdkClient
from .supervisor import GatewaySupervisor

logger = logging.getLogger(__name__)

# Circuit breaker: after N consecutive failures, pause API calls
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 60

# Gateway resurrect throttle: minimum seconds between two consecutive
# ensure_running() attempts triggered by in-flight request failures.
# Chosen smaller than _BREAKER_COOLDOWN_SECS so we can try to revive the
# Gateway *within* a breaker-open window (otherwise the breaker would mask
# the outage for a full minute before we'd even attempt recovery).
# Chosen larger than supervisor's HEALTH_CHECK_MAX_WAIT (30s) so a failed
# revive never overlaps with the next attempt.
_RECOVER_COOLDOWN_SECS = 15

# Background sync thread limits.
# _MAX_INFLIGHT_SYNCS caps concurrent capture threads: once reached we wait
# on the oldest one with _SYNC_JOIN_TIMEOUT_SECS before spawning a new one,
# so a hung Gateway can't cause unbounded thread growth.
_MAX_INFLIGHT_SYNCS = 4
_SYNC_JOIN_TIMEOUT_SECS = 5.0
# _SHUTDOWN_JOIN_TIMEOUT_SECS bounds how long shutdown will wait on *each*
# still-alive sync thread. Kept per-thread rather than global because one
# stuck thread shouldn't starve the rest.
_SHUTDOWN_JOIN_TIMEOUT_SECS = 5.0

# Watchdog: a daemon thread that periodically inspects the Gateway and
# resurrects it on death. This is the *only* mechanism that can recover from
# the "stuck-in-False" state where _gateway_available has been flipped to
# False (initial start failed or breaker-open path swallowed all errors) and
# every business request short-circuits before reaching the failure path that
# would otherwise call _try_recover_gateway().
#
# _WATCHDOG_INTERVAL_SECS controls the polling cadence. Kept smaller than
# _BREAKER_COOLDOWN_SECS so we can detect death and re-enable the provider
# well before the breaker would naturally expire.
# _WATCHDOG_SHUTDOWN_TIMEOUT_SECS bounds how long shutdown waits for the
# watchdog to exit cleanly; the thread is daemonized so a hang would not
# block interpreter exit, but a bounded join keeps logs orderly.
_WATCHDOG_INTERVAL_SECS = 10.0
_WATCHDOG_SHUTDOWN_TIMEOUT_SECS = 2.0

# Gateway networking defaults (kept here so is_available/initialize stay in sync)
_DEFAULT_GATEWAY_HOST = "127.0.0.1"
_DEFAULT_GATEWAY_PORT = 8420


def _resolve_gateway_port(default: int = _DEFAULT_GATEWAY_PORT) -> int:
    """Resolve MEMORY_TENCENTDB_GATEWAY_PORT with validation.

    Accepts surrounding whitespace. Falls back to ``default`` and logs a
    warning when the env var is unset, empty, not a valid integer, or
    outside the valid TCP port range (1..65535). This keeps ``is_available``
    exception-safe (required by the provider registration contract) and
    gives users a clear diagnostic instead of a raw ValueError stack.
    """
    raw = os.environ.get("MEMORY_TENCENTDB_GATEWAY_PORT")
    if raw is None or not raw.strip():
        return default
    try:
        port = int(raw.strip())
    except ValueError:
        logger.warning(
            "Invalid MEMORY_TENCENTDB_GATEWAY_PORT=%r (not an integer); "
            "falling back to default %d.",
            raw, default,
        )
        return default
    if not (1 <= port <= 65535):
        logger.warning(
            "MEMORY_TENCENTDB_GATEWAY_PORT=%d is out of range (1..65535); "
            "falling back to default %d.",
            port, default,
        )
        return default
    return port


def _resolve_gateway_host(default: str = _DEFAULT_GATEWAY_HOST) -> str:
    """Resolve MEMORY_TENCENTDB_GATEWAY_HOST, trimming whitespace."""
    raw = os.environ.get("MEMORY_TENCENTDB_GATEWAY_HOST")
    if raw is None:
        return default
    host = raw.strip()
    return host or default


# Candidate locations searched by _discover_gateway_cmd() when the user has not
# set MEMORY_TENCENTDB_GATEWAY_CMD. Order matters: in-tree checkout (next to
# this file) wins over ad-hoc clones in ``$HOME``.
_GATEWAY_DISCOVERY_RELATIVE_PATHS = (
    # hermes-plugin/memory/memory_tencentdb/__init__.py → plugin root
    Path("src") / "gateway" / "server.ts",
)
_GATEWAY_DISCOVERY_HOME_PATHS = (
    # New canonical install location (managed by install_hermes_memory_tencentdb.sh
    # and memory-tencentdb-ctl.sh): ~/.memory-tencentdb/tdai-memory-openclaw-plugin/...
    Path(".memory-tencentdb") / "tdai-memory-openclaw-plugin" / "src" / "gateway" / "server.ts",
    # Legacy locations (kept for backward compatibility with installations done
    # before the ~/.memory-tencentdb/ consolidation):
    Path("tdai-memory-openclaw-plugin") / "src" / "gateway" / "server.ts",
    Path(".hermes") / "plugins" / "tdai-memory-openclaw-plugin" / "src" / "gateway" / "server.ts",
)


def _discover_gateway_cmd() -> Optional[str]:
    """Best-effort fallback to locate the Node Gateway entry point.

    Called only when ``MEMORY_TENCENTDB_GATEWAY_CMD`` is unset, so that a fresh
    checkout works out-of-the-box without the user having to hand-craft an
    absolute launch command. Resolution order:

      1. ``<plugin-root>/src/gateway/server.ts`` (in-tree: this file lives at
         ``<plugin-root>/hermes-plugin/memory/memory_tencentdb/__init__.py``).
      2. Well-known paths under ``$HOME`` (preferred:
         ``~/.memory-tencentdb/tdai-memory-openclaw-plugin``; legacy:
         ``~/tdai-memory-openclaw-plugin`` and
         ``~/.hermes/plugins/tdai-memory-openclaw-plugin``).

    Returns a ready-to-``Popen`` command string wrapping a ``sh -c`` that
    ``cd``-s into the plugin root before exec-ing ``pnpm exec tsx
    src/gateway/server.ts``. The ``cd`` is required because ``tsx`` is
    installed under ``<plugin-root>/node_modules`` and Node's ESM resolver
    searches ``package.json`` from the cwd upward — if we launched ``tsx``
    with the hermes-agent cwd, resolution would fail with
    ``ERR_MODULE_NOT_FOUND``. Using ``sh -c`` keeps the supervisor's
    ``shlex.split`` + ``Popen(argv)`` contract intact (no ``shell=True``).

    Returns ``None`` if no ``server.ts`` candidate exists. The function never
    raises: supervisor-side validation will surface a friendly warning if the
    discovered path later fails to start.
    """
    import shlex

    here = Path(__file__).resolve()
    # hermes-plugin/memory/memory_tencentdb/__init__.py → parents[3] = plugin root
    plugin_root_candidates: List[Path] = []
    try:
        plugin_root_candidates.append(here.parents[3])
    except IndexError:  # pragma: no cover - defensive; __file__ depth is stable
        pass

    home_raw = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    home = Path(home_raw) if home_raw else None

    searched: List[Path] = []
    for root in plugin_root_candidates:
        for rel in _GATEWAY_DISCOVERY_RELATIVE_PATHS:
            searched.append(root / rel)
    if home is not None:
        for rel in _GATEWAY_DISCOVERY_HOME_PATHS:
            searched.append(home / rel)

    for candidate in searched:
        try:
            if candidate.is_file():
                # candidate = <plugin-root>/src/gateway/server.ts
                # -> parents[2] = <plugin-root>
                plugin_root = candidate.parents[2]
                logger.info(
                    "memory-tencentdb Gateway command auto-discovered: %s "
                    "(override with MEMORY_TENCENTDB_GATEWAY_CMD)",
                    candidate,
                )
                # shlex.quote guards against spaces / shell metachars in paths.
                # The inner command mirrors start-memory-tencentdb-gateway.sh:
                #   cd <plugin-root> && exec pnpm exec tsx src/gateway/server.ts
                inner = (
                    f"cd {shlex.quote(str(plugin_root))} && "
                    "exec pnpm exec tsx src/gateway/server.ts"
                )
                return f"sh -c {shlex.quote(inner)}"
        except OSError:  # pragma: no cover - e.g. permission errors on is_file
            continue

    logger.debug(
        "memory-tencentdb Gateway auto-discovery found no server.ts under: %s",
        ", ".join(str(p) for p in searched) or "<no candidates>",
    )
    return None


# Search tool limit bounds (shared by memory_search and conversation_search).
_DEFAULT_SEARCH_LIMIT = 5
_MAX_SEARCH_LIMIT = 20


def _coerce_limit(
    raw: Any,
    *,
    default: int = _DEFAULT_SEARCH_LIMIT,
    maximum: int = _MAX_SEARCH_LIMIT,
) -> int:
    """Coerce a tool-call ``limit`` arg into a valid int in ``[1, maximum]``.

    LLM tool calls don't always honor the JSON Schema ``type: integer``
    declaration — we regularly see strings ("10"), floats ("10.5"), None,
    or booleans. A bare ``int(x)`` either raises ValueError (string "abc",
    "10.5") or silently coerces True/False to 1/0, which would surface as
    a useless ``Tool call failed: invalid literal for int()`` back to the
    model. Instead we:

      * accept None / empty string -> return ``default``;
      * reject bool explicitly (bool is an ``int`` subclass in Python, and
        ``int(True) == 1`` is almost never what the caller meant);
      * accept int / float / numeric-looking strings via float() then int();
      * clamp the result to ``[1, maximum]``;
      * on any failure, log a warning and fall back to ``default``.
    """
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        logger.warning(
            "memory-tencentdb: ignoring non-numeric limit=%r (bool); "
            "falling back to default %d.",
            raw, default,
        )
        return default
    try:
        # float() handles int, float, and numeric strings uniformly;
        # int() then truncates toward zero.
        value = int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "memory-tencentdb: ignoring invalid limit=%r (not numeric); "
            "falling back to default %d.",
            raw, default,
        )
        return default
    if value < 1:
        return 1
    if value > maximum:
        return maximum
    return value


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

MEMORY_SEARCH_SCHEMA = {
    "name": "memory_tencentdb_memory_search",
    "description": (
        "Search through the user's long-term memories. Use this when you need to "
        "recall specific information about the user's preferences, past events, "
        "instructions, or context from previous conversations. Returns relevant "
        "memory records ranked by relevance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query describing what you want to recall about the user.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5, max: 20).",
            },
            "type": {
                "type": "string",
                "enum": ["persona", "episodic", "instruction"],
                "description": "Optional filter by memory type.",
            },
        },
        "required": ["query"],
    },
}

CONVERSATION_SEARCH_SCHEMA = {
    "name": "memory_tencentdb_conversation_search",
    "description": (
        "Search through past conversation history (raw dialogue records). "
        "Use when memory_tencentdb_memory_search doesn't have the information "
        "you need, or when you want to find specific past conversations or "
        "exact words the user said before."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query describing what conversation content you want to find.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of messages to return (default: 5, max: 20).",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MemoryTencentdbProvider(MemoryProvider):
    """memory-tencentdb four-layer memory via local Gateway sidecar."""

    def __init__(self):
        self._supervisor: Optional[GatewaySupervisor] = None
        self._client: Optional[MemoryTencentdbSdkClient] = None
        self._session_id = ""
        self._user_id = ""
        self._gateway_available = False
        self._initialized = False  # Track if initialize() has been called

        # Background sync threads.
        # We allow at most _MAX_INFLIGHT_SYNCS in-flight sync threads at any
        # time. Stuck threads (e.g. Gateway hung mid-capture) are tracked in
        # _active_syncs so shutdown can still join them and we never lose
        # references to spawned threads. _sync_lock guards both fields.
        self._sync_lock = threading.Lock()
        self._active_syncs: List[threading.Thread] = []

        # Circuit breaker
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

        # Gateway auto-resurrect state.
        # _recover_lock ensures only one thread at a time actually calls
        # supervisor.ensure_running() (which can block up to 30s). Other
        # threads that see a failure will try the lock non-blockingly and
        # fall through — they never wait, so recovery attempts never add
        # latency to business calls.
        # _last_recover_attempt gates how often we retry when revival keeps
        # failing (e.g. gateway binary missing, node not installed).
        # Initialized to -inf (rather than 0.0) because time.monotonic()'s
        # reference point is undefined — on some platforms (notably macOS)
        # it starts near zero at process start, which would make the
        # ``now - 0.0 < _RECOVER_COOLDOWN_SECS`` check swallow the very
        # first recovery attempt. Using -inf guarantees the first attempt
        # always passes the throttle.
        self._recover_lock = threading.Lock()
        self._last_recover_attempt = float("-inf")

        # Watchdog state.
        # The watchdog runs as a daemon thread that periodically (every
        # _WATCHDOG_INTERVAL_SECS) verifies the Gateway is alive and, on
        # failure, calls _try_recover_gateway(). This breaks the
        # "stuck-in-False" deadlock where business requests short-circuit on
        # _gateway_available == False and never reach the failure path that
        # would trigger recovery. _watchdog_stop is an Event so shutdown can
        # signal a clean exit without waiting a full polling interval.
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()

    # -- Properties -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "memory_tencentdb"

    # -- Circuit breaker ------------------------------------------------------

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "memory-tencentdb circuit breaker tripped after %d failures. Pausing for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    # -- Gateway auto-resurrect ----------------------------------------------

    def _try_recover_gateway(self, *, bypass_cooldown: bool = False) -> bool:
        """Best-effort: re-probe and, if needed, re-launch the Gateway.

        Called from the *failure* path of prefetch / sync_turn / handle_tool_call
        so a transient Gateway crash during an active Hermes session is not
        stuck behind the 60s circuit breaker. Also called from the watchdog
        thread (``bypass_cooldown=True``) which has its own cadence and must
        not be throttled by the request-driven 15s gate.

        Guarantees (do not break these without revisiting callers):
          * Never raises — exceptions are logged and swallowed.
          * Never blocks a losing thread: uses ``acquire(blocking=False)``.
            If another thread is already attempting recovery, we return
            ``False`` immediately.
          * Throttled by ``_RECOVER_COOLDOWN_SECS`` so a Gateway that
            refuses to start does not burn CPU on every failed request.
            The watchdog opts out of this throttle via ``bypass_cooldown``.
          * Refuses to run after ``shutdown()`` (detected via
            ``self._supervisor is None``) so we never resurrect a provider
            that the host has released.
          * On success: refreshes ``self._client`` / ``self._gateway_available``
            and resets the circuit breaker so the very next request isn't
            falsely blocked.
          * On failure: records the attempt timestamp; does NOT touch the
            circuit breaker (the caller already recorded a failure).
        """
        supervisor = self._supervisor
        if supervisor is None:
            # Either initialize() was never called, or shutdown() already ran.
            return False

        if not bypass_cooldown:
            now = time.monotonic()
            if now - self._last_recover_attempt < _RECOVER_COOLDOWN_SECS:
                return False

        if not self._recover_lock.acquire(blocking=False):
            # Another thread is already attempting recovery — let it work.
            return False

        try:
            # Re-check supervisor under the lock: shutdown() could have set it
            # to None between our first read and acquiring the lock.
            supervisor = self._supervisor
            if supervisor is None:
                return False

            # Double-check the cooldown under the lock too: another recovery
            # may have completed between our read and the acquire().
            if not bypass_cooldown:
                now = time.monotonic()
                if now - self._last_recover_attempt < _RECOVER_COOLDOWN_SECS:
                    return False

            # Fast path: maybe the Gateway is already back (someone else
            # restarted it, or it was a transient blip).
            if supervisor.is_running():
                logger.info(
                    "memory-tencentdb Gateway is reachable again; restoring provider state."
                )
                ok = True
            else:
                logger.warning(
                    "memory-tencentdb Gateway appears down; attempting to resurrect."
                )
                ok = supervisor.ensure_running()

            self._last_recover_attempt = time.monotonic()

            if ok:
                # Reattach the client (supervisor owns the authoritative one).
                self._client = supervisor.client
                self._gateway_available = True
                # Clear the breaker so the next request can proceed
                # immediately instead of being blocked by the 60s cooldown.
                self._consecutive_failures = 0
                self._breaker_open_until = 0.0
                logger.info("memory-tencentdb Gateway recovery succeeded.")
                return True

            logger.warning(
                "memory-tencentdb Gateway recovery failed; will retry no sooner than %ds.",
                _RECOVER_COOLDOWN_SECS,
            )
            return False
        except Exception as e:  # defensive: never propagate to caller
            self._last_recover_attempt = time.monotonic()
            logger.warning("memory-tencentdb Gateway recovery raised: %s", e)
            return False
        finally:
            self._recover_lock.release()

    # -- Watchdog & lazy probe -----------------------------------------------

    def _ensure_alive_for_request(self) -> bool:
        """Lazy probe used by the request short-circuit guards.

        Problem this solves: prefetch / sync_turn / handle_tool_call all
        return early when ``_gateway_available`` is False, which means a
        provider that failed to start (or was tripped by the 60s breaker
        and never re-enabled) can never recover via the request path —
        recovery only runs in the failure ``except`` branch, but the guard
        prevents requests from ever reaching that branch.

        This method gives the guards a way out: when the breaker is closed
        but ``_gateway_available`` is False, attempt a single recovery
        synchronously (subject to the same lock + cooldown as the failure
        path). On success the caller can proceed with the real request; on
        failure it returns the same empty / disabled response as before.

        Safe to call from any thread. Never raises. Returns the value of
        ``_gateway_available`` after the attempt.
        """
        if self._gateway_available:
            return True
        if self._is_breaker_open():
            # Breaker takes precedence: respect its 60s cooldown so we do
            # not turn every request into a Gateway-restart attempt during
            # an outage.
            return False
        # Try to bring the Gateway back. This is throttled by the same
        # 15s cooldown as the failure path, so a flood of requests won't
        # cause a recovery storm.
        self._try_recover_gateway()
        return self._gateway_available

    def _start_watchdog(self) -> None:
        """Start the background watchdog thread (idempotent).

        The watchdog is the only mechanism that can recover from the
        "Gateway dies while no requests are in flight" scenario. It also
        breaks the deadlock where _gateway_available is stuck False and
        every request short-circuits before triggering recovery.
        """
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="memory-tencentdb-watchdog",
        )
        self._watchdog_thread = thread
        thread.start()

    def _watchdog_loop(self) -> None:
        """Periodically verify Gateway health and resurrect on death.

        Runs until ``_watchdog_stop`` is set (by ``shutdown()``) or until
        the supervisor reference is dropped. Each iteration:

          1. Snapshot the supervisor reference. If None → exit (provider
             was shut down).
          2. Cheap path: if our own child PID is alive AND ``_gateway_available``
             is True, do nothing. Skips the HTTP round-trip in the common
             happy path.
          3. Otherwise, perform a real health check via supervisor.is_running().
             On success and ``_gateway_available`` is False (e.g. someone
             externally restarted the Gateway), reattach the client.
          4. On failure, call ``_try_recover_gateway(bypass_cooldown=True)``.
             The watchdog has its own pacing (``_WATCHDOG_INTERVAL_SECS``)
             so it must not be subject to the request-driven cooldown.

        All exceptions are logged and swallowed — the watchdog must never
        crash and leave the provider unsupervised.
        """
        logger.debug(
            "memory-tencentdb watchdog started (interval=%.1fs)",
            _WATCHDOG_INTERVAL_SECS,
        )
        while not self._watchdog_stop.wait(timeout=_WATCHDOG_INTERVAL_SECS):
            try:
                supervisor = self._supervisor
                if supervisor is None:
                    # Provider was shut down between ticks.
                    break

                # Cheap happy path: child is alive and we're already marked
                # available. Nothing to do.
                if self._gateway_available and supervisor.is_process_alive():
                    continue

                # Either we never marked available, the child died, or the
                # Gateway was started externally (no Popen handle but maybe
                # listening on the port). Do a real health check.
                healthy = False
                try:
                    healthy = supervisor.is_running()
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(
                        "memory-tencentdb watchdog health probe raised: %s", e,
                    )

                if healthy:
                    if not self._gateway_available:
                        # Externally revived (or first-time success after a
                        # bumpy start): reattach without re-spawning.
                        logger.info(
                            "memory-tencentdb watchdog: Gateway is reachable; "
                            "restoring provider state."
                        )
                        self._client = supervisor.client
                        self._gateway_available = True
                        self._consecutive_failures = 0
                        self._breaker_open_until = 0.0
                    continue

                # Truly down. Attempt resurrection, bypassing the request-path
                # cooldown — the watchdog itself enforces pacing.
                logger.warning(
                    "memory-tencentdb watchdog: Gateway unreachable; "
                    "attempting to resurrect."
                )
                self._try_recover_gateway(bypass_cooldown=True)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "memory-tencentdb watchdog iteration raised (continuing): %s", e,
                )

        logger.debug("memory-tencentdb watchdog exiting")

    def _stop_watchdog(self) -> None:
        """Signal the watchdog to exit and join briefly. Safe if not started."""
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        self._watchdog_thread = None
        if thread is None:
            return
        thread.join(timeout=_WATCHDOG_SHUTDOWN_TIMEOUT_SECS)
        if thread.is_alive():
            # Daemon thread, will not block interpreter exit; just log so
            # users can correlate with Gateway hangs in the health probe.
            logger.debug(
                "memory-tencentdb watchdog did not exit within %.1fs; "
                "abandoning (daemon).",
                _WATCHDOG_SHUTDOWN_TIMEOUT_SECS,
            )

    # -- Core lifecycle -------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the Gateway is configured or already running.

        Prefers local config checks (env vars) to avoid blocking network calls.
        Only falls back to health check when no env config is present.
        """
        # Fast path: env var configured → assume available (will verify in initialize)
        if os.environ.get("MEMORY_TENCENTDB_GATEWAY_CMD"):
            return True
        if os.environ.get("MEMORY_TENCENTDB_GATEWAY_PORT"):
            return True
        # Slow path: no env config, try a quick health check.
        # Use validated resolvers so a malformed env var never raises here
        # (is_available must never throw: it's called during provider
        # registration and an exception would break the whole plugin).
        host = _resolve_gateway_host()
        port = _resolve_gateway_port()
        client = MemoryTencentdbSdkClient(base_url=f"http://{host}:{port}", timeout=2)
        try:
            result = client.health(timeout=2)
            return result.get("status") in ("ok", "degraded")
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Start or connect to the Gateway sidecar.

        Gateway startup is performed in a background thread so that
        ``initialize()`` returns immediately and does not block the
        Hermes agent ``__init__`` (which would add up to 30 s latency
        before the first prompt is accepted).

        While the background thread is still running:
          * ``prefetch`` / ``sync_turn`` / ``handle_tool_call`` see
            ``_gateway_available == False`` and gracefully return empty
            results or no-ops — no data is lost because capture will
            succeed once the Gateway comes up and subsequent turns will
            work normally.
          * ``get_tool_schemas`` already returns schemas optimistically
            (gated on ``_initialized``, not ``_gateway_available``),
            so the tools appear in the LLM surface even before the
            Gateway is ready.
        """
        self._session_id = session_id
        self._user_id = kwargs.get("user_id", "default")

        host = _resolve_gateway_host()
        port = _resolve_gateway_port()
        # Priority: explicit env var → auto-discovery (in-tree / $HOME fallbacks).
        # Auto-discovery lets fresh checkouts work without manual CMD wiring;
        # it only runs when the env var is not set, so existing deployments
        # are unaffected.
        gateway_cmd = os.environ.get("MEMORY_TENCENTDB_GATEWAY_CMD") or _discover_gateway_cmd()

        self._supervisor = GatewaySupervisor(
            host=host,
            port=port,
            gateway_cmd=gateway_cmd,
        )

        # Mark as initialized immediately so tools are registered
        # (get_tool_schemas checks _initialized, not _gateway_available).
        self._initialized = True

        def _background_start():
            """Start / connect to the Gateway in the background."""
            try:
                available = self._supervisor.ensure_running()
                if available:
                    self._client = self._supervisor.client
                    self._gateway_available = True
                    logger.info(
                        "memory-tencentdb Gateway ready (background start, %s:%d)",
                        host, port,
                    )
                else:
                    logger.warning(
                        "memory-tencentdb Gateway not available after background start. "
                        "Memory features will be disabled until the Gateway is reachable. "
                        "Set MEMORY_TENCENTDB_GATEWAY_CMD to auto-start the Gateway, "
                        "or place the plugin checkout at ~/tdai-memory-openclaw-plugin "
                        "for auto-discovery."
                    )
            except Exception as e:
                logger.warning(
                    "memory-tencentdb background Gateway start failed (non-fatal): %s", e
                )

        # Fast path: if the Gateway is *already* running (e.g. started by
        # systemd, memory-tencentdb-ctl, or a previous session), skip the
        # thread overhead and attach synchronously. The health check takes
        # <100ms for a local Gateway, so this doesn't block meaningfully.
        if self._supervisor.is_running():
            self._client = self._supervisor.client
            self._gateway_available = True
            logger.info(
                "memory-tencentdb Gateway already running (%s:%d)",
                host, port,
            )
        else:
            # Gateway is not up yet — start it in the background.
            t = threading.Thread(
                target=_background_start, daemon=True,
                name="tdai-gateway-init",
            )
            t.start()

        # Start the watchdog regardless of the initial start outcome.
        # Even if _background_start fails (e.g. tdai binary missing on
        # first launch), the watchdog will keep retrying so a later
        # external fix (operator installs node, drops the plugin into
        # the discovery path, etc.) is picked up automatically without
        # requiring a hermes restart.
        self._start_watchdog()

    def system_prompt_block(self) -> str:
        if not self._gateway_available:
            return ""
        return (
            "# memory-tencentdb Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Four-layer memory system (L0→L1→L2→L3) with automatic conversation "
            "capture, structured memory extraction, scene blocks, and persona synthesis.\n"
            "Use memory_tencentdb_memory_search to find specific memories, "
            "memory_tencentdb_conversation_search to search raw conversation history."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Synchronous recall — fetch memories in real-time for the current turn."""
        if not query:
            return ""
        # Lazy probe before the short-circuit guard. If the Gateway died but
        # the breaker has not yet tripped (or has since cooled down), this
        # gives the request path a chance to revive it instead of silently
        # returning "" forever. See _ensure_alive_for_request() for the
        # guarantees and rationale.
        if not self._ensure_alive_for_request() or not self._client:
            return ""

        effective_session = session_id or self._session_id
        try:
            result = self._client.recall(
                query=query,
                session_key=effective_session,
                user_id=self._user_id,
            )
            context = result.get("context", "")
            self._record_success()
            if context:
                return f"## memory-tencentdb Memory\n{context}"
            return ""
        except Exception as e:
            self._record_failure()
            logger.debug("memory-tencentdb prefetch failed: %s", e)
            # Fire-and-forget attempt to bring the Gateway back for the next
            # call. Never blocks more than supervisor.ensure_running()'s own
            # timeout, and only one thread at a time actually does the work.
            self._try_recover_gateway()
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op — recall is done synchronously in prefetch()."""
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Gateway for capture (non-blocking).

        Threading model:
          * Each call spawns a daemon thread that performs one ``capture``.
          * ``_active_syncs`` retains references to all still-alive threads so
            they are never orphaned when a new sync starts.
          * If ``_MAX_INFLIGHT_SYNCS`` is reached (e.g. Gateway is hung),
            we wait on the oldest thread for ``_SYNC_JOIN_TIMEOUT_SECS`` before
            spawning a new one. If that thread is still alive afterwards we
            still spawn, but keep the stuck thread tracked so ``shutdown`` can
            try to reap it later.
          * All mutations of ``_active_syncs`` are serialized by
            ``_sync_lock`` so concurrent callers (future async entry points)
            cannot leak references via a read/modify/write race.
        """
        # Lazy probe — same rationale as prefetch(). Without this, a
        # provider stuck in the False/closed-breaker state would silently
        # drop every captured turn until the watchdog (or a manual
        # restart) revived it.
        if not self._ensure_alive_for_request() or not self._client:
            return

        effective_session = session_id or self._session_id
        client = self._client

        def _sync():
            try:
                client.capture(
                    user_content=user_content,
                    assistant_content=assistant_content,
                    session_key=effective_session,
                    user_id=self._user_id,
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("memory-tencentdb sync failed: %s", e)
                # Trigger recovery from a background thread — safe because
                # _try_recover_gateway itself is non-blocking under
                # contention and swallows all exceptions.
                self._try_recover_gateway()

        # Reap finished threads and, if at capacity, wait on the oldest one.
        # We pick the oldest non-finished candidate *outside* the lock so the
        # join() call doesn't hold _sync_lock (holding a lock across a
        # potentially slow join would serialize every incoming turn).
        oldest_to_join: Optional[threading.Thread] = None
        with self._sync_lock:
            self._active_syncs = [t for t in self._active_syncs if t.is_alive()]
            if len(self._active_syncs) >= _MAX_INFLIGHT_SYNCS:
                oldest_to_join = self._active_syncs[0]

        if oldest_to_join is not None:
            oldest_to_join.join(timeout=_SYNC_JOIN_TIMEOUT_SECS)
            if oldest_to_join.is_alive():
                logger.warning(
                    "memory-tencentdb sync backlog: oldest sync thread still "
                    "running after %.1fs; %d in-flight threads tracked. "
                    "Continuing with a new sync; Gateway may be hung.",
                    _SYNC_JOIN_TIMEOUT_SECS, len(self._active_syncs),
                )

        thread = threading.Thread(
            target=_sync, daemon=True, name="memory-tencentdb-sync",
        )
        with self._sync_lock:
            # Reap again in case the join above freed slots, then register.
            self._active_syncs = [t for t in self._active_syncs if t.is_alive()]
            self._active_syncs.append(thread)
        thread.start()

    def shutdown(self) -> None:
        """Clean shutdown — flush and release resources."""
        # Stop the watchdog FIRST so it does not race with shutdown by
        # spawning a fresh recovery attempt while we're tearing the
        # supervisor down. Idempotent + non-blocking-bounded.
        self._stop_watchdog()

        # Wait for every background sync thread we ever spawned (not just the
        # most recent one). Taking a snapshot under the lock first means new
        # calls to sync_turn during shutdown can't race with our iteration.
        with self._sync_lock:
            pending = list(self._active_syncs)
            self._active_syncs.clear()

        for t in pending:
            if not t.is_alive():
                continue
            t.join(timeout=_SHUTDOWN_JOIN_TIMEOUT_SECS)
            if t.is_alive():
                # Threads are daemon, so they won't block interpreter exit —
                # but log so users can correlate with Gateway issues.
                logger.warning(
                    "memory-tencentdb shutdown: sync thread %s still alive "
                    "after %.1fs; abandoning (daemon).",
                    t.name, _SHUTDOWN_JOIN_TIMEOUT_SECS,
                )

        # Send session end if Gateway is available
        if self._client and self._gateway_available:
            try:
                self._client.end_session(
                    session_key=self._session_id,
                    user_id=self._user_id,
                )
            except Exception as e:
                logger.debug("memory-tencentdb session end failed: %s", e)

        # Note: do NOT shut down the supervisor/Gateway here — it may serve
        # other sessions. The Gateway manages its own lifecycle.
        # We *do* drop our reference to the supervisor so any in-flight
        # _try_recover_gateway() call sees self._supervisor is None and
        # bails out instead of resurrecting a released provider.
        self._client = None
        self._gateway_available = False
        self._initialized = False
        self._supervisor = None

    # -- Tools ----------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Optimistically return tool schemas if Gateway is configured or running.
        # This is critical because MemoryManager.add_provider() calls
        # get_tool_schemas() BEFORE initialize() to build the _tool_to_provider
        # routing table. If we return [] here, tools won't be routable
        # even after initialize() succeeds (despite _refresh_tool_registration).
        if self._gateway_available or self._initialized:
            return [MEMORY_SEARCH_SCHEMA, CONVERSATION_SEARCH_SCHEMA]
        # Pre-init: check if Gateway is likely to be available
        if os.environ.get("MEMORY_TENCENTDB_GATEWAY_CMD") or os.environ.get("MEMORY_TENCENTDB_GATEWAY_PORT"):
            return [MEMORY_SEARCH_SCHEMA, CONVERSATION_SEARCH_SCHEMA]
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        # Lazy probe — gives tool-call path the same self-heal opportunity
        # as prefetch / sync_turn. Without this, an LLM-issued memory_search
        # call could see "Gateway is not connected" forever even after the
        # Gateway came back up, because nothing else would flip
        # _gateway_available back to True.
        self._ensure_alive_for_request()
        if not self._client:
            return json.dumps({
                "error": "memory-tencentdb Gateway is not connected. Memory search is temporarily unavailable.",
                "hint": "The Gateway may still be starting up. Try again in a moment.",
            })
        if self._is_breaker_open():
            return json.dumps({"error": "memory-tencentdb Gateway temporarily unavailable (circuit breaker open)."})

        try:
            if tool_name == "memory_tencentdb_memory_search":
                query = args.get("query", "")
                if not query:
                    return json.dumps({"error": "Missing required parameter: query"})
                result = self._client.search_memories(
                    query=query,
                    limit=_coerce_limit(args.get("limit")),
                    type_filter=args.get("type", ""),
                )
                self._record_success()
                return json.dumps(result)

            if tool_name == "memory_tencentdb_conversation_search":
                query = args.get("query", "")
                if not query:
                    return json.dumps({"error": "Missing required parameter: query"})
                result = self._client.search_conversations(
                    query=query,
                    limit=_coerce_limit(args.get("limit")),
                )
                self._record_success()
                return json.dumps(result)

            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            self._record_failure()
            # Same fire-and-forget recovery as prefetch(); the error
            # returned to the LLM below is unchanged.
            self._try_recover_gateway()
            return json.dumps({"error": f"Tool call failed: {e}"})

    # -- Optional hooks -------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to memory-tencentdb for indexing."""
        # TODO: Implement mirroring of Hermes builtin MEMORY.md/USER.md writes
        # to memory-tencentdb's recall index for conflict suppression and dedup.
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Trigger session-level flush on the Gateway."""
        if self._client and self._gateway_available:
            try:
                self._client.end_session(
                    session_key=self._session_id,
                    user_id=self._user_id,
                )
            except Exception as e:
                logger.debug("memory-tencentdb on_session_end failed: %s", e)

    # -- Config ---------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "gateway_cmd",
                "description": "Command to start the memory-tencentdb Gateway (e.g. 'node --import tsx /path/to/server.ts')",
                "env_var": "MEMORY_TENCENTDB_GATEWAY_CMD",
                "required": False,
            },
            {
                "key": "gateway_host",
                "description": "Gateway host",
                "default": "127.0.0.1",
                "env_var": "MEMORY_TENCENTDB_GATEWAY_HOST",
            },
            {
                "key": "gateway_port",
                "description": "Gateway port",
                "default": "8420",
                "env_var": "MEMORY_TENCENTDB_GATEWAY_PORT",
            },
            {
                "key": "llm_api_key",
                "description": "LLM API key (for Gateway's standalone LLM calls)",
                "secret": True,
                "required": True,
                "env_var": "MEMORY_TENCENTDB_LLM_API_KEY",
            },
            {
                "key": "llm_base_url",
                "description": "LLM API base URL",
                "default": "https://api.openai.com/v1",
                "env_var": "MEMORY_TENCENTDB_LLM_BASE_URL",
            },
            {
                "key": "llm_model",
                "description": "LLM model name",
                "default": "gpt-4o",
                "env_var": "MEMORY_TENCENTDB_LLM_MODEL",
            },
        ]


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register memory-tencentdb as a memory provider plugin."""
    ctx.register_memory_provider(MemoryTencentdbProvider())
