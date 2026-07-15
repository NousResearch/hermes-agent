"""Service-level orchestration for LSP clients.

The :class:`LSPService` is the bridge between the synchronous
file_operations layer and the async :class:`agent.lsp.client.LSPClient`.

Design choices:

- A **single asyncio event loop** runs in a background thread.  All
  client work happens on that loop.  Synchronous callers from
  ``tools/file_operations.py`` use :meth:`get_diagnostics_sync` to
  open + wait + drain in one blocking call.

- One client per ``(server_id, workspace_root)`` key.  Lazy spawn:
  the first request for a key spawns the client; subsequent requests
  re-use it.

- A **broken-set** records deterministic ``(server_id, workspace_root)``
  failures such as an unavailable spawn command. Transient startup/request
  failures use a short monotonic cooldown so the service can recover.

- A **delta baseline** map keeps "diagnostics-as-of-the-last-snapshot"
  per file.  ``snapshot_baseline()`` is called BEFORE a write; the
  next ``get_diagnostics_sync()`` returns only diagnostics that
  weren't in the baseline.  This is the lift from Claude Code's
  ``beforeFileEdited`` / ``getNewDiagnostics`` pattern, except wired
  to the local LSP layer instead of MCP IDE RPC.

The service is **off by default** — call :meth:`is_active` to check
whether it's actually doing anything.  When LSP is disabled in
config, when no git workspace can be detected, when all configured
servers are missing binaries and auto-install is off, ``is_active``
returns False and the file_operations layer falls through to the
in-process syntax check.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import math
import os
import threading
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from agent.lsp import eventlog
from agent.lsp.client import (
    DIAGNOSTICS_DOCUMENT_WAIT,
    LSPClient,
)
from agent.lsp.servers import (
    ServerContext,
    find_server_for_file,
    language_id_for,
)
from agent.lsp.workspace import (
    clear_cache,
    resolve_workspace_for_file,
)

logger = logging.getLogger("agent.lsp.manager")

DEFAULT_IDLE_TIMEOUT = 7200.0
DEFAULT_SWEEP_INTERVAL = 60.0
MAX_IDLE_TIMEOUT = 365 * 24 * 60 * 60
MAX_SWEEP_INTERVAL = 24 * 60 * 60
MAX_CLIENTS_PER_PROCESS = 64


@dataclass
class _ClientEntry:
    """One generation of one canonical ``(server, workspace)`` client."""

    key: Tuple[str, str]
    generation: int
    workspace_root: str
    state: str
    last_used: float
    client: Optional[LSPClient] = None
    leases: int = 0
    spawn_task: Optional[asyncio.Task] = None
    retire_task: Optional[asyncio.Task] = None
    pending_eviction: Optional[str] = None


@dataclass
class _ClientLease:
    """Generation-bound token that protects a client from retirement."""

    service: "LSPService"
    key: Tuple[str, str]
    generation: int
    client: LSPClient
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.service._release_lease(self)


def _lifecycle_config(lsp_cfg: Dict[str, Any]) -> Tuple[bool, float, float, int]:
    """Parse opt-in bounded lifecycle settings without truthy coercion.

    Invalid lifecycle configuration falls back to legacy process-lifetime
    retention.  A malformed resource policy must never disable LSP itself.
    """

    raw = lsp_cfg.get("lifecycle")
    defaults = (False, DEFAULT_IDLE_TIMEOUT, DEFAULT_SWEEP_INTERVAL, 0)
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        logger.warning("lsp.lifecycle must be a mapping; lifecycle remains disabled")
        return defaults

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        logger.warning("lsp.lifecycle.enabled must be true or false; lifecycle remains disabled")
        return defaults

    def finite_number(name: str, default: float, *, positive: bool, maximum: float) -> float:
        value = raw.get(name, default)
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid_type or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
        result = float(value)
        if (positive and result <= 0) or (not positive and result < 0) or result > maximum:
            comparator = "greater than zero" if positive else "non-negative"
            raise ValueError(f"{name} must be {comparator} and at most {maximum:g}")
        return result

    try:
        idle_timeout = finite_number(
            "idle_timeout_seconds",
            DEFAULT_IDLE_TIMEOUT,
            positive=False,
            maximum=MAX_IDLE_TIMEOUT,
        )
        sweep_interval = finite_number(
            "sweep_interval_seconds",
            DEFAULT_SWEEP_INTERVAL,
            positive=True,
            maximum=MAX_SWEEP_INTERVAL,
        )
        max_clients = raw.get("max_clients_per_process", 0)
        if (
            not isinstance(max_clients, int)
            or isinstance(max_clients, bool)
            or not 0 <= max_clients <= MAX_CLIENTS_PER_PROCESS
        ):
            raise ValueError(
                "max_clients_per_process must be an integer between "
                f"0 and {MAX_CLIENTS_PER_PROCESS}"
            )
    except (TypeError, ValueError) as exc:
        logger.warning("invalid lsp.lifecycle configuration (%s); lifecycle remains disabled", exc)
        return defaults
    return enabled, idle_timeout, sweep_interval, max_clients


class _BackgroundLoop:
    """A daemon thread that owns one asyncio event loop.

    Provides :meth:`run` for synchronous callers — submits a coroutine
    to the loop and blocks until it finishes (or a timeout fires).
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_forever,
            name="hermes-lsp-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_forever(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    def run(self, coro, *, timeout: Optional[float] = None) -> Any:
        """Submit a coroutine to the loop and block until done.

        Returns the coroutine's result, or raises its exception.
        """
        from agent.async_utils import safe_schedule_threadsafe
        if self._loop is None:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("background loop not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("background loop not running")
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None


class LSPService:
    """The process-wide LSP service.

    Created once via :meth:`create_from_config`; the
    :func:`agent.lsp.get_service` accessor manages the singleton.
    Most callers should use that accessor rather than constructing
    :class:`LSPService` directly.
    """

    # ------------------------------------------------------------------
    # construction + factory
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        enabled: bool,
        wait_mode: str,
        wait_timeout: float,
        install_strategy: str,
        binary_overrides: Optional[Dict[str, List[str]]] = None,
        env_overrides: Optional[Dict[str, Dict[str, str]]] = None,
        init_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        disabled_servers: Optional[List[str]] = None,
        lifecycle_enabled: bool = False,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        sweep_interval: float = DEFAULT_SWEEP_INTERVAL,
        max_clients: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])
        self._lifecycle_enabled = lifecycle_enabled
        self._idle_timeout = idle_timeout
        self._sweep_interval = sweep_interval
        self._max_clients = max_clients
        self._clock = clock

        # All registry transitions are guarded by this lock. Async spawn and
        # retirement work always happens after releasing it.
        self._state_lock = threading.RLock()
        self._service_state = "open"
        self._entries: Dict[Tuple[str, str], _ClientEntry] = {}
        self._broken: set = set()
        self._cooldowns: Dict[Tuple[str, str], float] = {}
        self._next_generation = 0
        self._admission_lock: Optional[asyncio.Lock] = None
        self._reaper_task: Optional[asyncio.Task] = None
        self._maintenance_tasks: set[asyncio.Task] = set()
        self._reap_count = 0
        self._capacity_eviction_count = 0
        self._overflow_count = 0

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write. ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()
            self._loop.run(self._initialize_async_state(), timeout=2.0)

    @classmethod
    def create_from_config(cls) -> Optional["LSPService"]:
        """Build a service from ``hermes_cli.config`` settings.

        Returns ``None`` if the config can't be loaded.  The service
        itself returns ``is_active()`` False when LSP is disabled.
        """
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP config load failed: %s", e)
            return None

        lsp_cfg = (cfg.get("lsp") or {}) if isinstance(cfg, dict) else {}
        if not isinstance(lsp_cfg, dict):
            lsp_cfg = {}

        enabled = bool(lsp_cfg.get("enabled", True))
        wait_mode = lsp_cfg.get("wait_mode", "document")
        wait_timeout = float(lsp_cfg.get("wait_timeout", DIAGNOSTICS_DOCUMENT_WAIT))
        install_strategy = lsp_cfg.get("install_strategy", "auto")
        servers_cfg = lsp_cfg.get("servers") or {}
        disabled = []
        binary_overrides: Dict[str, List[str]] = {}
        env_overrides: Dict[str, Dict[str, str]] = {}
        init_overrides: Dict[str, Dict[str, Any]] = {}
        if isinstance(servers_cfg, dict):
            for name, sub in servers_cfg.items():
                if not isinstance(sub, dict):
                    continue
                if sub.get("disabled"):
                    disabled.append(name)
                cmd = sub.get("command")
                if isinstance(cmd, list) and cmd:
                    binary_overrides[name] = cmd
                env = sub.get("env")
                if isinstance(env, dict):
                    env_overrides[name] = {k: str(v) for k, v in env.items()}
                init = sub.get("initialization_options")
                if isinstance(init, dict):
                    init_overrides[name] = init

        lifecycle_enabled, idle_timeout, sweep_interval, max_clients = _lifecycle_config(lsp_cfg)

        return cls(
            enabled=enabled,
            wait_mode=wait_mode,
            wait_timeout=wait_timeout,
            install_strategy=install_strategy,
            binary_overrides=binary_overrides,
            env_overrides=env_overrides,
            init_overrides=init_overrides,
            disabled_servers=disabled,
            lifecycle_enabled=lifecycle_enabled,
            idle_timeout=idle_timeout,
            sweep_interval=sweep_interval,
            max_clients=max_clients,
        )

    # ------------------------------------------------------------------
    # bounded client lifecycle
    # ------------------------------------------------------------------

    async def _initialize_async_state(self) -> None:
        self._admission_lock = asyncio.Lock()
        if self._lifecycle_enabled and (self._idle_timeout > 0 or self._max_clients > 0):
            self._reaper_task = asyncio.create_task(
                self._reaper_loop(), name="hermes-lsp-reaper"
            )

    async def _reaper_loop(self) -> None:
        """Sweep forever, but survive an individual sweep failure."""
        while True:
            await asyncio.sleep(self._sweep_interval)
            try:
                await self._reap_idle_clients()
                await self._converge_capacity()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("LSP lifecycle sweep failed; continuing: %s", exc)

    async def _reap_idle_clients(self) -> None:
        """Retire idle, unleased generations using monotonic age."""
        if not self._lifecycle_enabled or self._idle_timeout <= 0:
            return
        now = self._clock()
        retire_tasks: List[asyncio.Task] = []
        with self._state_lock:
            if self._service_state != "open":
                return
            for entry in list(self._entries.values()):
                if (
                    entry.state == "active"
                    and entry.leases == 0
                    and now - entry.last_used >= self._idle_timeout
                ):
                    task = self._begin_retirement_locked(entry, "idle")
                    if task is not None:
                        retire_tasks.append(task)
        if retire_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in retire_tasks),
                return_exceptions=True,
            )

    def _begin_retirement_locked(
        self, entry: _ClientEntry, reason: str
    ) -> Optional[asyncio.Task]:
        """Publish RETIRING atomically; perform shutdown outside the lock."""
        if entry.state == "retiring":
            return entry.retire_task
        if entry.state != "active" or entry.client is None:
            return None
        if entry.leases:
            entry.pending_eviction = reason
            return None
        entry.state = "retiring"
        entry.pending_eviction = reason
        task = asyncio.create_task(
            self._retire_entry(entry, reason),
            name=f"hermes-lsp-retire-{entry.generation}",
        )
        entry.retire_task = task
        return task

    async def _retire_entry(self, entry: _ClientEntry, reason: str) -> None:
        """Completion-idempotent retirement for one concrete generation."""
        client = entry.client
        error: Optional[BaseException] = None
        try:
            if client is not None:
                await client.shutdown()
        except BaseException as exc:  # noqa: BLE001
            error = exc
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            with self._state_lock:
                current = self._entries.get(entry.key)
                if current is entry:
                    self._entries.pop(entry.key, None)
                if reason == "idle":
                    self._reap_count += 1
                elif reason == "capacity":
                    self._capacity_eviction_count += 1
            level = logging.WARNING if error is not None else logging.INFO
            logger.log(
                level,
                "LSP lifecycle retired server=%s root=%s generation=%s "
                "reason=%s outcome=%s",
                entry.key[0],
                entry.workspace_root,
                entry.generation,
                reason,
                "error" if error is not None else "clean",
            )

    async def _retire_key_async(self, key: Tuple[str, str], reason: str) -> bool:
        """Request retirement without interrupting an active lease."""
        wait_task: Optional[asyncio.Task] = None
        with self._state_lock:
            entry = self._entries.get(key)
            if entry is None:
                return True
            if entry.state == "spawning":
                wait_task = entry.spawn_task
                if wait_task is not None:
                    wait_task.cancel()
            elif entry.state == "retiring":
                wait_task = entry.retire_task
            else:
                wait_task = self._begin_retirement_locked(entry, reason)
                if wait_task is None:
                    return False
        if wait_task is not None:
            await asyncio.gather(asyncio.shield(wait_task), return_exceptions=True)
        return True

    def _release_lease(self, lease: _ClientLease) -> None:
        retire_task: Optional[asyncio.Task] = None
        should_converge = False
        with self._state_lock:
            entry = self._entries.get(lease.key)
            if entry is None or entry.generation != lease.generation:
                return
            if entry.leases <= 0:
                logger.warning(
                    "LSP lifecycle lease underflow prevented for generation=%s",
                    lease.generation,
                )
                return
            entry.leases -= 1
            entry.last_used = self._clock()
            if entry.leases == 0 and entry.pending_eviction:
                retire_task = self._begin_retirement_locked(
                    entry, entry.pending_eviction
                )
            should_converge = (
                self._lifecycle_enabled
                and self._max_clients > 0
                and len(self._entries) > self._max_clients
            )
        if retire_task is not None:
            self._track_maintenance(retire_task)
        if should_converge:
            self._track_maintenance(
                asyncio.create_task(
                    self._converge_capacity(), name="hermes-lsp-capacity-converge"
                )
            )

    def _track_maintenance(self, task: asyncio.Task) -> None:
        self._maintenance_tasks.add(task)
        task.add_done_callback(self._maintenance_tasks.discard)

    def _select_lru_victim_locked(self) -> Optional[_ClientEntry]:
        candidates = [
            entry
            for entry in self._entries.values()
            if entry.state == "active" and entry.leases == 0
        ]
        return min(candidates, key=lambda entry: entry.last_used, default=None)

    async def _converge_capacity(self) -> None:
        if not self._lifecycle_enabled or self._max_clients <= 0:
            return
        admission_lock = self._admission_lock
        if admission_lock is None:
            return
        async with admission_lock:
            while True:
                with self._state_lock:
                    if (
                        self._service_state != "open"
                        or len(self._entries) <= self._max_clients
                    ):
                        return
                    victim = self._select_lru_victim_locked()
                    if victim is None:
                        return
                    retire_task = self._begin_retirement_locked(victim, "capacity")
                if retire_task is None:
                    return
                await asyncio.gather(
                    asyncio.shield(retire_task), return_exceptions=True
                )

    def _resolve_target(
        self, file_path: str, *, log_miss: bool = False
    ) -> Optional[Tuple[Any, Tuple[str, str], str]]:
        """Resolve one canonical registry key while preserving the LSP root."""
        srv = find_server_for_file(file_path)
        if srv is None or srv.server_id in self._disabled_servers:
            return None
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            if log_miss:
                eventlog.log_no_project_root(srv.server_id, file_path)
            return None
        try:
            per_server_root = srv.resolve_root(file_path, ws_root)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LSP root resolution failed for %s: %s", file_path, exc)
            return None
        if per_server_root is None:
            if log_miss:
                eventlog.log_disabled(
                    srv.server_id, file_path, "exclude marker hit (server gated off)"
                )
            return None
        workspace_root = os.path.abspath(per_server_root)
        canonical_root = os.path.normcase(os.path.realpath(workspace_root))
        return srv, (srv.server_id, canonical_root), workspace_root

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled and self.is_accepting_requests()

    def is_accepting_requests(self) -> bool:
        with self._state_lock:
            return self._service_state == "open"

    def begin_shutdown(self) -> bool:
        """Publish CLOSING synchronously before shutdown enters the loop."""
        with self._state_lock:
            if self._service_state != "open":
                return False
            self._service_state = "closing"
            return True

    def enabled_for(self, file_path: str) -> bool:
        """Return True when this file has an admissible LSP target."""
        if not self._enabled or not self.is_accepting_requests():
            return False
        target = self._resolve_target(file_path)
        if target is None:
            return False
        _, key, _ = target
        with self._state_lock:
            if key in self._broken:
                return False
            retry_after = self._cooldowns.get(key)
            if retry_after is not None:
                if self._clock() < retry_after:
                    return False
                self._cooldowns.pop(key, None)
        return True

    def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics for ``file_path`` as the delta baseline.

        Called BEFORE a write so the next ``get_diagnostics_sync()``
        can filter out pre-existing errors.  Best-effort — failures
        are silently swallowed so a flaky server can't break a write.

        Outer failures retire the affected generation and apply a short
        monotonic cooldown so subsequent edits skip it briefly instead of
        re-paying the timeout or permanently disabling the workspace.
        """
        if not self.enabled_for(file_path):
            return
        try:
            diags = self._loop.run(self._snapshot_async(file_path), timeout=8.0)
            self._delta_baseline[os.path.abspath(file_path)] = diags or []
        except Exception as e:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            self._delta_baseline[os.path.abspath(file_path)] = []

    def get_diagnostics_sync(
        self,
        file_path: str,
        *,
        delta: bool = True,
        timeout: Optional[float] = None,
        line_shift: Optional[Callable[[int], Optional[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously open ``file_path`` in the right server, wait for
        diagnostics, return them.

        If ``delta`` is True (default), the result is filtered against
        any baseline previously captured via :meth:`snapshot_baseline`.
        Diagnostics present in the baseline are removed so the caller
        only sees errors introduced by the current edit.

        When ``line_shift`` is provided, baseline diagnostics are
        remapped through it before the set-difference.  This handles
        the case where the edit deleted or inserted lines, causing
        pre-existing diagnostics below the edit point to surface at
        different line numbers in the post-edit snapshot — without
        the shift, they'd all look "introduced by this edit".  Pass
        a callable built by
        :func:`agent.lsp.range_shift.build_line_shift` (pre_text,
        post_text).  Omit when pre/post content isn't available;
        the unshifted comparison still catches diagnostics that
        didn't move.

        Returns an empty list when LSP is disabled, when no workspace
        can be detected, when no server matches, or when the server
        can't be spawned.  Never raises.
        """
        if not self.enabled_for(file_path):
            return []

        # Resolve server_id eagerly so we can emit structured logs even
        # when the request errors out below.
        srv = find_server_for_file(file_path)
        server_id = srv.server_id if srv else "?"

        try:
            t = timeout if timeout is not None else self._wait_timeout + 2.0
            diags = self._loop.run(self._open_and_wait_async(file_path), timeout=t) or []
        except asyncio.TimeoutError as e:
            eventlog.log_timeout(server_id, file_path)
            logger.debug("LSP diagnostics timeout for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []
        except Exception as e:  # noqa: BLE001
            eventlog.log_server_error(server_id, file_path, e)
            logger.debug("LSP diagnostics fetch failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []

        abs_path = os.path.abspath(file_path)
        if delta:
            baseline = self._delta_baseline.get(abs_path) or []
            if baseline:
                if line_shift is not None:
                    # Remap baseline diagnostics into post-edit
                    # coordinates so shifted-but-otherwise-identical
                    # entries hash equal under _diag_key.  Entries
                    # that mapped into a deleted region drop out
                    # silently — they no longer apply.
                    from agent.lsp.range_shift import shift_baseline
                    baseline = shift_baseline(baseline, line_shift)
                seen = {_diag_key(d) for d in baseline}
                diags = [d for d in diags if _diag_key(d) not in seen]
            # Roll baseline forward — next call returns deltas relative
            # to the just-emitted state, mirroring claude-code's
            # diagnosticTracking.
            try:
                fresh = self._loop.run(self._current_diags_async(file_path), timeout=2.0) or []
            except Exception:  # noqa: BLE001
                fresh = []
            if fresh:
                self._delta_baseline[abs_path] = fresh

        if diags:
            eventlog.log_diagnostics(server_id, file_path, len(diags))
        else:
            eventlog.log_clean(server_id, file_path)
        return diags

    def _mark_broken_for_file(self, file_path: str, exc: BaseException) -> None:
        """Retire a timed-out generation and apply a short retry cooldown.

        Request and cold-index timeouts are transient. They must not poison a
        healthy server/workspace pair for the remainder of the process.
        """
        target = self._resolve_target(file_path)
        if target is None:
            return
        srv, key, workspace_root = target
        with self._state_lock:
            first_failure = key not in self._cooldowns
            self._cooldowns[key] = self._clock() + 5.0
        try:
            self._loop.run(
                self._retire_key_async(key, "request-failure"), timeout=3.0
            )
        except Exception as cleanup_exc:  # noqa: BLE001
            logger.debug("LSP failed-generation cleanup for %s: %s", key, cleanup_exc)
        if first_failure:
            eventlog.log_spawn_failed(srv.server_id, workspace_root, exc)

    def shutdown(self) -> None:
        """Close admission, drain lifecycle work, and stop the loop."""
        if not self._enabled:
            with self._state_lock:
                self._service_state = "closed"
            return
        self.begin_shutdown()
        try:
            self._loop.run(self._shutdown_async(), timeout=15.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LSP shutdown did not complete cleanly: %s", exc)
        finally:
            self._loop.stop()
            with self._state_lock:
                self._service_state = "closed"
            clear_cache()

    # ------------------------------------------------------------------
    # async internals
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _leased_client(
        self, file_path: str, *, spawn: bool = True
    ) -> AsyncIterator[Optional[LSPClient]]:
        lease = await self._acquire_lease(file_path, spawn=spawn)
        try:
            yield lease.client if lease is not None else None
        finally:
            if lease is not None:
                lease.release()

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        async with self._leased_client(file_path) as client:
            if client is None:
                return []
            try:
                version = await client.open_file(
                    file_path, language_id=language_id_for(file_path)
                )
                await client.wait_for_diagnostics(
                    file_path, version, mode=self._wait_mode
                )
                return list(client.diagnostics_for(file_path))
            except Exception as exc:  # noqa: BLE001
                logger.debug("snapshot open/wait failed: %s", exc)
                return []

    async def _open_and_wait_async(self, file_path: str) -> List[Dict[str, Any]]:
        async with self._leased_client(file_path) as client:
            if client is None:
                return []
            try:
                version = await client.open_file(
                    file_path, language_id=language_id_for(file_path)
                )
                await client.save_file(file_path)
                await client.wait_for_diagnostics(
                    file_path, version, mode=self._wait_mode
                )
                return list(client.diagnostics_for(file_path))
            except Exception as exc:  # noqa: BLE001
                logger.debug("open/wait failed for %s: %s", file_path, exc)
                return []

    async def _current_diags_async(self, file_path: str) -> List[Dict[str, Any]]:
        async with self._leased_client(file_path, spawn=False) as client:
            if client is None:
                return []
            return list(client.diagnostics_for(file_path))

    async def _acquire_lease(
        self, file_path: str, *, spawn: bool
    ) -> Optional[_ClientLease]:
        target = self._resolve_target(file_path, log_miss=True)
        if target is None:
            return None
        srv, key, workspace_root = target

        while True:
            wait_task: Optional[asyncio.Task] = None
            with self._state_lock:
                if self._service_state != "open" or key in self._broken:
                    return None
                retry_after = self._cooldowns.get(key)
                if retry_after is not None:
                    if self._clock() < retry_after:
                        return None
                    self._cooldowns.pop(key, None)
                entry = self._entries.get(key)
                if entry is not None and entry.state == "active":
                    if entry.client is not None and entry.client.is_running:
                        entry.leases += 1
                        entry.last_used = self._clock()
                        eventlog.log_active(srv.server_id, workspace_root)
                        return _ClientLease(
                            self, key, entry.generation, entry.client
                        )
                    wait_task = self._begin_retirement_locked(entry, "crashed")
                    if wait_task is None:
                        # Another operation still owns this dead generation.
                        # Its final release will retire it; fall back rather
                        # than spinning the event loop.
                        return None
                elif entry is not None and entry.state == "spawning":
                    wait_task = entry.spawn_task
                elif entry is not None and entry.state == "retiring":
                    wait_task = entry.retire_task

            if wait_task is not None:
                await asyncio.gather(
                    asyncio.shield(wait_task), return_exceptions=True
                )
                continue
            if not spawn:
                return None
            lease = await self._reserve_spawn(srv, key, workspace_root)
            if lease is not None:
                return lease
            # A concurrent retirement may have completed while admission was
            # serialized. Re-evaluate canonical state before falling back.
            with self._state_lock:
                if self._service_state != "open" or key in self._broken:
                    return None
                retry_after = self._cooldowns.get(key)
                if retry_after is not None and self._clock() < retry_after:
                    return None

    async def _reserve_spawn(
        self, srv: Any, key: Tuple[str, str], workspace_root: str
    ) -> Optional[_ClientLease]:
        """Serialize capacity, retirement, spawn, and the first lease."""
        admission_lock = self._admission_lock
        if admission_lock is None:
            return None
        async with admission_lock:
            while True:
                wait_task: Optional[asyncio.Task] = None
                with self._state_lock:
                    if self._service_state != "open" or key in self._broken:
                        return None
                    retry_after = self._cooldowns.get(key)
                    if retry_after is not None:
                        if self._clock() < retry_after:
                            return None
                        self._cooldowns.pop(key, None)
                    existing = self._entries.get(key)
                    if existing is not None and existing.state == "active":
                        if existing.client is not None and existing.client.is_running:
                            existing.leases += 1
                            existing.last_used = self._clock()
                            return _ClientLease(
                                self, key, existing.generation, existing.client
                            )
                        wait_task = self._begin_retirement_locked(
                            existing, "crashed"
                        )
                        if wait_task is None:
                            return None
                    elif existing is not None and existing.state == "spawning":
                        wait_task = existing.spawn_task
                    elif existing is not None and existing.state == "retiring":
                        wait_task = existing.retire_task
                    else:
                        if (
                            self._lifecycle_enabled
                            and self._max_clients > 0
                            and len(self._entries) >= self._max_clients
                        ):
                            victim = self._select_lru_victim_locked()
                            if victim is not None:
                                wait_task = self._begin_retirement_locked(
                                    victim, "capacity"
                                )
                            else:
                                # A cancelled caller may leave its shielded
                                # spawn running. Count that reservation and
                                # wait for it (or a retirement) before
                                # admitting another distinct key. Only fully
                                # active, fully leased pools may overflow.
                                wait_task = next(
                                    (
                                        item.spawn_task or item.retire_task
                                        for item in self._entries.values()
                                        if item.state in {"spawning", "retiring"}
                                        and (item.spawn_task or item.retire_task)
                                    ),
                                    None,
                                )
                            if victim is None and wait_task is None:
                                self._overflow_count += 1
                                logger.info(
                                    "LSP lifecycle temporarily over capacity: "
                                    "all %s client slots are leased",
                                    self._max_clients,
                                )
                        if wait_task is None:
                            self._next_generation += 1
                            entry = _ClientEntry(
                                key=key,
                                generation=self._next_generation,
                                workspace_root=workspace_root,
                                state="spawning",
                                last_used=self._clock(),
                            )
                            wait_task = asyncio.create_task(
                                self._spawn_entry(entry, srv),
                                name=f"hermes-lsp-spawn-{entry.generation}",
                            )
                            entry.spawn_task = wait_task
                            self._entries[key] = entry
                if wait_task is None:
                    return None
                await asyncio.gather(
                    asyncio.shield(wait_task), return_exceptions=True
                )

    async def _spawn_entry(self, entry: _ClientEntry, srv: Any) -> Optional[LSPClient]:
        client: Optional[LSPClient] = None
        try:
            ctx = ServerContext(
                workspace_root=entry.workspace_root,
                install_strategy=self._install_strategy,
                binary_overrides=self._binary_overrides,
                env_overrides=self._env_overrides,
                init_overrides=self._init_overrides,
            )
            spec = srv.build_spawn(entry.workspace_root, ctx)
            if spec is None:
                eventlog.log_server_unavailable(srv.server_id, srv.server_id)
                with self._state_lock:
                    self._broken.add(entry.key)
                return None
            client = LSPClient(
                server_id=srv.server_id,
                workspace_root=spec.workspace_root,
                command=spec.command,
                env=spec.env,
                cwd=spec.cwd,
                initialization_options=spec.initialization_options,
                seed_diagnostics_on_first_push=(
                    spec.seed_diagnostics_on_first_push or srv.seed_first_push
                ),
            )
            await client.start()
            publish = False
            with self._state_lock:
                current = self._entries.get(entry.key)
                if (
                    current is entry
                    and entry.state == "spawning"
                    and self._service_state == "open"
                ):
                    entry.client = client
                    entry.state = "active"
                    entry.last_used = self._clock()
                    publish = True
            if not publish:
                await client.shutdown()
                return None
            eventlog.log_active(srv.server_id, entry.workspace_root)
            logger.info(
                "LSP lifecycle spawned server=%s root=%s generation=%s",
                srv.server_id,
                entry.workspace_root,
                entry.generation,
            )
            return client
        except asyncio.CancelledError:
            if client is not None:
                await asyncio.shield(client.shutdown())
            raise
        except BaseException as exc:  # noqa: BLE001
            eventlog.log_spawn_failed(srv.server_id, entry.workspace_root, exc)
            with self._state_lock:
                self._cooldowns[entry.key] = self._clock() + 5.0
            if client is not None:
                await client.shutdown()
            return None
        finally:
            with self._state_lock:
                current = self._entries.get(entry.key)
                if current is entry and entry.state == "spawning":
                    self._entries.pop(entry.key, None)

    async def _shutdown_async(self) -> None:
        reaper = self._reaper_task
        self._reaper_task = None
        if reaper is not None:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)

        with self._state_lock:
            retirement_tasks = {
                entry.retire_task
                for entry in self._entries.values()
                if entry.retire_task is not None
            }
            spawn_tasks = [
                entry.spawn_task
                for entry in self._entries.values()
                if entry.spawn_task is not None and not entry.spawn_task.done()
            ]
            maintenance = [
                task
                for task in self._maintenance_tasks
                if task not in retirement_tasks and not task.done()
            ]
        for task in spawn_tasks + maintenance:
            task.cancel()
        if spawn_tasks or maintenance:
            await asyncio.gather(
                *(spawn_tasks + maintenance), return_exceptions=True
            )

        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            with self._state_lock:
                leased = sum(entry.leases for entry in self._entries.values())
            if leased == 0 or asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.05)
        if leased:
            logger.warning(
                "LSP shutdown lease drain timed out with %s active lease(s); "
                "forcing bounded cleanup",
                leased,
            )

        retire_tasks: List[asyncio.Task] = []
        with self._state_lock:
            for entry in list(self._entries.values()):
                if entry.state == "retiring" and entry.retire_task is not None:
                    retire_tasks.append(entry.retire_task)
                    continue
                if entry.state == "active":
                    if entry.leases:
                        entry.leases = 0
                    task = self._begin_retirement_locked(entry, "shutdown")
                    if task is not None:
                        retire_tasks.append(task)
        if retire_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in retire_tasks),
                return_exceptions=True,
            )

        with self._state_lock:
            self._entries.clear()
            self._broken.clear()
            self._cooldowns.clear()
            self._maintenance_tasks.clear()
            self._service_state = "closed"

    # ------------------------------------------------------------------
    # status / introspection (used by ``hermes lsp status``)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a lifecycle-aware service snapshot for CLI/status JSON."""
        now = self._clock()
        with self._state_lock:
            clients = [
                {
                    "server_id": entry.key[0],
                    "workspace_root": entry.workspace_root,
                    "state": entry.state,
                    "running": bool(entry.client and entry.client.is_running),
                    "generation": entry.generation,
                    "leases": entry.leases,
                    "idle_seconds": max(0.0, now - entry.last_used),
                    "eviction_reason": entry.pending_eviction,
                }
                for entry in self._entries.values()
            ]
            broken = list(self._broken)
            counts = {
                state: sum(entry.state == state for entry in self._entries.values())
                for state in ("spawning", "active", "retiring")
            }
            service_state = self._service_state
        return {
            "enabled": self._enabled,
            "wait_mode": self._wait_mode,
            "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy,
            "clients": clients,
            "broken": broken,
            "disabled_servers": sorted(self._disabled_servers),
            "lifecycle": {
                "enabled": self._lifecycle_enabled,
                "service_state": service_state,
                "idle_timeout_seconds": self._idle_timeout,
                "sweep_interval_seconds": self._sweep_interval,
                "max_clients_per_process": self._max_clients,
                "reaper_running": bool(
                    self._reaper_task is not None and not self._reaper_task.done()
                ),
                "counts": counts,
                "reaped": self._reap_count,
                "capacity_evictions": self._capacity_eviction_count,
                "temporary_overflows": self._overflow_count,
            },
        }


def _diag_key(d: Dict[str, Any]) -> str:
    """Content equality key used for cross-edit delta filtering.

    Includes the diagnostic's position range — when used together
    with :func:`agent.lsp.range_shift.shift_baseline`, the baseline
    is line-shifted into post-edit coordinates BEFORE this key is
    computed, so identical-but-shifted diagnostics hash equal.  Two
    genuinely distinct diagnostics at different lines (e.g. the same
    error class introduced at a second site) hash differently and
    are surfaced as new.

    Mirrors :func:`agent.lsp.client._diagnostic_key`; intentionally
    identical so the two layers agree on diagnostic identity.
    """
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    code = d.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    return "\x00".join(
        [
            str(d.get("severity") or 1),
            str(code or ""),
            str(d.get("source") or ""),
            str(d.get("message") or "").strip(),
            f"{start.get('line', 0)}:{start.get('character', 0)}-{end.get('line', 0)}:{end.get('character', 0)}",
        ]
    )


__all__ = ["LSPService"]
