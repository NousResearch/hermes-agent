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

- A **broken-set** records ``(server_id, workspace_root)`` pairs that
  failed to spawn or initialize.  These are never retried for the
  life of the service.  Mirrors OpenCode's design.

- A **delta baseline** map keeps "diagnostics-as-of-the-last-snapshot"
  per file.  ``snapshot_baseline()`` is called BEFORE a write; the
  next ``get_diagnostics_sync()`` returns only diagnostics that
  weren't in the baseline.  This is the lift from Claude Code's
  ``beforeFileEdited`` / ``getNewDiagnostics`` pattern, except wired
  to the local LSP layer instead of MCP IDE RPC.

The service is enabled by default — call :meth:`is_active` to check
whether it is accepting work.  Per-file workspace/server gates still
fall through to the in-process syntax check when LSP cannot run.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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

DEFAULT_IDLE_TIMEOUT = 600  # seconds; servers idle for >10min get reaped
MIN_IDLE_TIMEOUT = 30  # floor for positive config values; 0 disables reaping
SHUTDOWN_WAIT_TIMEOUT = 10.0


def _client_key(srv: "ServerDef", root: str) -> tuple:
    """Cache key for the client serving ``root``: multi-root servers share one process per
    ``server_id``; everything else is keyed per resolved project root."""
    return (srv.server_id, "" if getattr(srv, "multi_root", False) else root)


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

    def submit(self, coro) -> Future:
        """Submit *coro* and return its thread-safe owner future."""
        from agent.async_utils import safe_schedule_threadsafe

        if self._loop is None:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("background loop not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("background loop not running")
        return fut

    def run(self, coro, *, timeout: Optional[float] = None) -> Any:
        """Submit a coroutine to the loop and block until done.

        Returns the coroutine's result, or raises its exception.
        """
        fut = self.submit(coro)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise

    def stop(self) -> bool:
        loop = self._loop
        if loop is None:
            return True
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                return False
        self._loop = None
        self._thread = None
        return True


@dataclass
class _ClientEntry:
    """One published client generation for a server/workspace key."""

    client: LSPClient
    generation: int
    leases: int = 0
    retiring: bool = False
    retire_reason: Optional[str] = None
    retirement_task: Optional[asyncio.Task] = None
    retirement_error: Optional[str] = None
    leases_drained: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.leases_drained.set()

    @property
    def workspace_folders(self) -> List[str]:
        """Proxy for main's direct-client attribute access (tests / status)."""
        return self.client.workspace_folders


def _task_returned_true(task: asyncio.Task) -> bool:
    """Return whether a completed lifecycle task confirmed cleanup."""
    if not task.done() or task.cancelled():
        return False
    try:
        return task.result() is True
    except Exception:  # noqa: BLE001
        return False


class _ClientLease:
    """Generation-bound ownership held across every awaited client use."""

    def __init__(
        self,
        service: "LSPService",
        key: Tuple[str, str],
        entry: _ClientEntry,
    ) -> None:
        self._service = service
        self._key = key
        self._entry = entry
        self._released = False

    @property
    def client(self) -> LSPClient:
        return self._entry.client

    @property
    def generation(self) -> int:
        return self._entry.generation

    async def __aenter__(self) -> LSPClient:
        return self.client

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._service._release_lease(self._key, self._entry)


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
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])
        self._idle_timeout = idle_timeout

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()

        # Per-(server_id, workspace_root) state
        self._clients: Dict[Tuple[str, str], _ClientEntry] = {}
        self._broken: set = set()
        self._spawning: Dict[Tuple[str, str], asyncio.Task] = {}
        self._generations: Dict[Tuple[str, str], int] = {}
        self._last_used: Dict[Tuple[str, str], float] = {}
        self._state_lock = threading.Lock()
        self._idle_reaper_task: Optional[asyncio.Task] = None
        self._shutdown_task: Optional[asyncio.Task] = None
        self._shutdown_future: Optional[Future] = None
        self._shutdown_state = "running" if self._enabled else "closed"
        self._shutdown_error: Optional[str] = None
        self._admitting = self._enabled
        self._clients_drained = not self._enabled
        self._loop_stopped = not self._enabled

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write.  ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

        if self._enabled and self._idle_timeout > 0:
            self._loop.run(self._start_idle_reaper(), timeout=2.0)

    @classmethod
    def create_from_config(cls) -> Optional["LSPService"]:
        """Build a service from ``hermes_cli.config`` settings.

        Returns ``None`` if the config can't be loaded.  The service
        itself returns ``is_active()`` False when LSP is disabled.
        """
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
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
        try:
            idle_timeout = float(lsp_cfg.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        except (TypeError, ValueError):
            idle_timeout = DEFAULT_IDLE_TIMEOUT
        if 0 < idle_timeout < MIN_IDLE_TIMEOUT:
            # Keep very small values from thrashing server indexes.  Active
            # operations are protected independently by generation leases.
            # Zero remains the explicit "disable reaping" value.
            idle_timeout = MIN_IDLE_TIMEOUT
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

        return cls(
            enabled=enabled,
            wait_mode=wait_mode,
            wait_timeout=wait_timeout,
            install_strategy=install_strategy,
            binary_overrides=binary_overrides,
            env_overrides=env_overrides,
            init_overrides=init_overrides,
            disabled_servers=disabled,
            idle_timeout=idle_timeout,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        with self._state_lock:
            return self._enabled and self._admitting

    def enabled_for(self, file_path: str) -> bool:
        """Return True iff LSP should run for this specific file.

        Gates on workspace detection (file or cwd inside a git worktree),
        on whether any registered server matches the extension, and
        on whether the (server_id, workspace_root) pair is in the
        broken-set from a previous spawn failure.

        Files in already-broken pairs return False so the file_operations
        layer skips the LSP path entirely — no spawn attempts, no
        timeout cost — until the service is restarted (``hermes lsp
        restart``) or the process exits.
        """
        if not self.is_active():
            return False
        srv = find_server_for_file(file_path)
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        ws_root, gated_in = resolve_workspace_for_file(file_path)
        if not (ws_root and gated_in):
            return False
        # Broken-set short-circuit.  Use the per-server root if we can
        # compute one cheaply; otherwise fall back to the workspace
        # root as the broken key (which is what _acquire_client would
        # have used anyway when it failed).
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        with self._state_lock:
            if (
                _client_key(srv, per_server_root) in self._broken
                or (srv.server_id, per_server_root) in self._broken
            ):
                return False
        return True

    def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics for ``file_path`` as the delta baseline.

        Called BEFORE a write so the next ``get_diagnostics_sync()``
        can filter out pre-existing errors.  Best-effort — failures
        are silently swallowed so a flaky server can't break a write.

        Outer timeouts (e.g. server hangs during initialize) mark the
        (server_id, workspace_root) pair as broken so subsequent edits
        skip it instantly instead of re-paying the timeout cost.
        """
        if not self.enabled_for(file_path):
            return
        try:
            # Outer join budget must exceed the inner wait budget or a
            # slow-but-alive server gets falsely marked broken.
            t = max(8.0, self._wait_timeout + 3.0)
            diags = self._loop.run(self._snapshot_async(file_path), timeout=t)
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
            diags = self._loop.run(self._open_and_wait_async(file_path), timeout=t)
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

        if diags is None:
            # The server is alive but never produced diagnostics for the
            # post-edit content within the wait budget (common for
            # tsserver on large projects).  Report "no data" rather than
            # whatever stale state is in the stores — surfacing the
            # previous edit's errors as if they were current is the
            # ghost-diagnostics bug.  The server is NOT marked broken:
            # slow is not dead, and the next edit may well succeed.
            eventlog.log_timeout(server_id, file_path, kind="fresh diagnostics")
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
        """Mark the (server_id, workspace_root) pair as broken so subsequent
        edits skip it instantly instead of re-paying timeout cost.

        Called when the outer ``_loop.run`` timeout cancels an in-flight
        spawn/initialize that the inner ``_acquire_client`` task was still
        holding open.  Without this, every subsequent write would re-enter
        the spawn path and re-pay the full ``snapshot_baseline``
        timeout (8s) until the binary is fixed.

        Also kills any orphan client process that survived the cancelled
        future, and emits a single eventlog WARNING so the user knows
        which server gave up.

        ``exc`` is whatever exception the outer wrapper caught — used
        only for logging, never re-raised.
        """
        srv = find_server_for_file(file_path)
        if srv is None:
            return
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            return
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        # Broken-set keys stay per-project (unnormalized) so one broken
        # project never poisons sibling projects on a shared multi-root
        # server; the client key is normalized separately below.
        broken_key = (srv.server_id, per_server_root)
        client_key = _client_key(srv, per_server_root)
        with self._state_lock:
            already_broken = broken_key in self._broken
            self._broken.add(broken_key)

        # Cancel an in-flight spawn and retire any published generation.
        # The retirement task itself is retained, so this bounded outer
        # wait cannot abandon cleanup when it times out.
        if not self._loop_stopped:
            try:
                self._loop.run(self._break_key_async(client_key), timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

        if not already_broken:
            eventlog.log_spawn_failed(srv.server_id, per_server_root, exc)

    def shutdown(self) -> bool:
        """Tear down all clients and stop the background loop.

        Returns ``True`` only after spawn tasks, request leases, client
        cleanup, and the background loop have all finished.  A timeout or
        cleanup failure leaves the loop alive so a later caller can observe
        the retained teardown task instead of publishing a replacement.
        """
        if not self._enabled:
            return True
        shutdown_future: Optional[Future] = None
        try:
            with self._state_lock:
                already_closed = self._shutdown_state == "closed"
                clients_drained = self._clients_drained
                if not already_closed:
                    # Close admission synchronously.  Even if the event loop is
                    # blocked inside a server installer/spawn and cannot start
                    # the teardown coroutine before our join timeout, no new
                    # request can acquire or publish a generation meanwhile.
                    self._admitting = False
                    if self._shutdown_state == "running":
                        self._shutdown_state = "closing"
                if not (already_closed or clients_drained):
                    shutdown_future = self._shutdown_future
                    if shutdown_future is None or shutdown_future.done():
                        # Store the cross-thread owner before waiting.  A caller
                        # timeout must not cancel the only queued teardown while
                        # build_spawn()/an installer is blocking the loop.
                        shutdown_future = self._loop.submit(self._shutdown_async())
                        self._shutdown_future = shutdown_future
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown scheduling error: %s", e)
            with self._state_lock:
                self._shutdown_state = "failed"
                self._shutdown_error = f"{type(e).__name__}: {e}"
            return False

        if already_closed or clients_drained:
            return self._finish_shutdown()
        assert shutdown_future is not None
        try:
            succeeded = bool(shutdown_future.result(timeout=SHUTDOWN_WAIT_TIMEOUT))
        except FutureTimeoutError:
            with self._state_lock:
                self._shutdown_error = "timed out waiting for retained teardown"
            return False
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
            with self._state_lock:
                self._shutdown_state = "failed"
                self._shutdown_error = f"{type(e).__name__}: {e}"
            return False
        if not succeeded:
            return False
        return self._finish_shutdown()

    def _finish_shutdown(self) -> bool:
        with self._state_lock:
            if self._loop_stopped:
                return True
        if not self._loop.stop():
            with self._state_lock:
                self._shutdown_state = "failed"
                self._shutdown_error = "background event loop did not stop"
            return False
        with self._state_lock:
            self._loop_stopped = True
            self._shutdown_state = "closed"
            self._shutdown_error = None
            self._shutdown_future = None
        clear_cache()
        return True

    def _get_shutdown_error(self) -> Optional[str]:
        """Return the in-process teardown error for singleton ownership."""
        with self._state_lock:
            return self._shutdown_error

    # ------------------------------------------------------------------
    # async internals
    # ------------------------------------------------------------------

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        lease = await self._acquire_client(file_path)
        if lease is None:
            return []
        client = lease.client
        try:
            try:
                version = await client.open_file(
                    file_path, language_id=language_id_for(file_path)
                )
                fresh = await client.wait_for_diagnostics(
                    file_path, version, mode=self._wait_mode
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("snapshot open/wait failed: %s", e)
                return []
            if not fresh:
                # No fresh data for the pre-edit content — an empty baseline
                # is safe: worst case the delta filter removes less, never
                # more.  Never seed the baseline from stale stores.
                return []
            return list(client.diagnostics_for(file_path, fresh_only=True))
        finally:
            lease.release()

    async def _open_and_wait_async(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Open + wait for FRESH diagnostics.

        Returns the fresh diagnostic list, or ``None`` when the server
        never produced post-change data within the wait budget.  The
        distinction matters: ``[]`` means "server checked the new
        content, it's clean", ``None`` means "no verdict" — the caller
        must not substitute stale data for either.
        """
        lease = await self._acquire_client(file_path)
        if lease is None:
            return None
        client = lease.client
        try:
            try:
                version = await client.open_file(
                    file_path, language_id=language_id_for(file_path)
                )
                await client.save_file(file_path)
                fresh = await client.wait_for_diagnostics(
                    file_path,
                    version,
                    mode=self._wait_mode,
                    timeout=self._wait_timeout,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("open/wait failed for %s: %s", file_path, e)
                return None
            if not fresh:
                return None
            return list(client.diagnostics_for(file_path, fresh_only=True))
        finally:
            lease.release()

    async def _current_diags_async(self, file_path: str) -> List[Dict[str, Any]]:
        ws, gated = resolve_workspace_for_file(file_path)
        srv = find_server_for_file(file_path)
        if not (ws and gated and srv):
            return []
        with self._state_lock:
            entry = self._clients.get(_client_key(srv, ws))
        if entry is None or entry.retiring:
            return []
        return list(entry.client.diagnostics_for(file_path, fresh_only=True))

    async def _acquire_client(self, file_path: str) -> Optional[_ClientLease]:
        """Return a lease on the current generation, spawning if needed.

        A retiring generation stays published until its cleanup completes.
        Callers wait on that retained retirement task before a replacement
        generation can be spawned for the same key.
        """
        srv = find_server_for_file(file_path)
        if srv is None:
            return None
        if srv.server_id in self._disabled_servers:
            eventlog.log_disabled(srv.server_id, file_path, "disabled in config")
            return None
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            eventlog.log_no_project_root(srv.server_id, file_path)
            return None
        per_server_root = srv.resolve_root(file_path, ws_root)
        if per_server_root is None:
            eventlog.log_disabled(
                srv.server_id, file_path, "exclude marker hit (server gated off)"
            )
            return None  # exclude marker hit, server gated off

        key = _client_key(srv, per_server_root)
        while True:
            retirement: Optional[asyncio.Task] = None
            spawning: Optional[asyncio.Task] = None
            lease: Optional[_ClientLease] = None
            with self._state_lock:
                if not self._admitting or key in self._broken:
                    return None
                entry = self._clients.get(key)
                if (
                    entry is not None
                    and not entry.retiring
                    and entry.client.is_running
                ):
                    entry.leases += 1
                    entry.leases_drained.clear()
                    self._last_used[key] = time.time()
                    lease = _ClientLease(self, key, entry)
                elif entry is not None:
                    retirement = self._begin_retirement_locked(
                        key, entry, "client no longer running"
                    )
                else:
                    spawning = self._spawning.get(key)
                    if spawning is None:
                        generation = self._generations.get(key, 0) + 1
                        self._generations[key] = generation
                        spawning = asyncio.create_task(
                            self._spawn_client(
                                srv,
                                key,
                                per_server_root,
                                generation,
                            )
                        )
                        self._spawning[key] = spawning

            if lease is not None:
                try:
                    if getattr(srv, "multi_root", False) and per_server_root not in (
                        lease.client.workspace_folders
                    ):
                        # Multi-root servers share one process; announce this
                        # root to it instead of spawning another client.
                        await lease.client.add_workspace_folder(per_server_root)
                    eventlog.log_active(srv.server_id, per_server_root)
                except BaseException:
                    lease.release()
                    raise
                return lease
            if retirement is not None:
                try:
                    retired = await asyncio.shield(retirement)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    return None
                if not retired:
                    return None
                continue
            assert spawning is not None
            try:
                await asyncio.shield(spawning)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                return None

    async def _spawn_client(
        self,
        srv,
        key: Tuple[str, str],
        per_server_root: str,
        generation: int,
    ) -> Optional[_ClientEntry]:
        """Build one generation and publish it only while admission is open."""
        client: Optional[LSPClient] = None
        try:
            with self._state_lock:
                if not self._admitting or key in self._broken:
                    return None
            ctx = ServerContext(
                workspace_root=per_server_root,
                install_strategy=self._install_strategy,
                binary_overrides=self._binary_overrides,
                env_overrides=self._env_overrides,
                init_overrides=self._init_overrides,
            )
            spec = srv.build_spawn(per_server_root, ctx)
            with self._state_lock:
                if not self._admitting or key in self._broken:
                    return None
            if spec is None:
                # ``build_spawn`` returns None when the binary can't be
                # located (auto-install disabled, manual-only server,
                # or install attempt failed).  Surface this once via
                # the structured logger so the user can act on it.
                eventlog.log_server_unavailable(srv.server_id, srv.server_id)
                with self._state_lock:
                    self._broken.add(key)
                return None
            client = LSPClient(
                server_id=srv.server_id,
                workspace_root=spec.workspace_root,
                command=spec.command,
                env=spec.env,
                cwd=spec.cwd,
                initialization_options=spec.initialization_options,
                seed_diagnostics_on_first_push=spec.seed_diagnostics_on_first_push or srv.seed_first_push,
            )
            await client.start()
            entry = _ClientEntry(client=client, generation=generation)
            with self._state_lock:
                publish = self._admitting and key not in self._broken
                if publish:
                    self._clients[key] = entry
                    self._last_used[key] = time.time()
            if not publish:
                await self._cleanup_unpublished_client(
                    key,
                    client,
                    generation,
                    "spawn completed after admission closed",
                )
                return None
            return entry
        except asyncio.CancelledError:
            if client is not None:
                await self._cleanup_unpublished_client(
                    key,
                    client,
                    generation,
                    "spawn cancelled",
                )
            raise
        except Exception as e:  # noqa: BLE001
            eventlog.log_spawn_failed(srv.server_id, per_server_root, e)
            with self._state_lock:
                self._broken.add(key)
            if client is not None:
                await self._cleanup_unpublished_client(
                    key,
                    client,
                    generation,
                    "spawn/initialize failed",
                )
            return None
        finally:
            with self._state_lock:
                if self._spawning.get(key) is asyncio.current_task():
                    self._spawning.pop(key, None)

    async def _cleanup_unpublished_client(
        self,
        key: Tuple[str, str],
        client: LSPClient,
        generation: int,
        reason: str,
    ) -> bool:
        """Clean a generation that never reached the active registry.

        A cleanup failure is still published as a retiring tombstone.  This
        makes service shutdown fail closed instead of forgetting a process
        that may still be alive merely because admission closed first.
        """
        entry = _ClientEntry(client=client, generation=generation)
        entry.retiring = True
        entry.retire_reason = reason
        with self._state_lock:
            published = key not in self._clients
            if published:
                self._clients[key] = entry
        try:
            await client.shutdown()
        except Exception as e:  # noqa: BLE001
            message = f"{type(e).__name__}: {e}"
            with self._state_lock:
                if self._clients.get(key) is entry:
                    entry.retirement_error = message
                self._broken.add(key)
            logger.warning(
                "LSP unpublished generation %s cleanup failed for %s/%s: %s",
                generation,
                key[0],
                key[1],
                message,
            )
            return False
        with self._state_lock:
            if published and self._clients.get(key) is entry:
                self._clients.pop(key, None)
                self._last_used.pop(key, None)
        return True

    def _release_lease(
        self,
        key: Tuple[str, str],
        entry: _ClientEntry,
    ) -> None:
        with self._state_lock:
            if entry.leases <= 0:
                return
            entry.leases -= 1
            if entry.leases == 0:
                entry.leases_drained.set()
            if self._clients.get(key) is entry and not entry.retiring:
                self._last_used[key] = time.time()

    def _begin_retirement_locked(
        self,
        key: Tuple[str, str],
        entry: _ClientEntry,
        reason: str,
    ) -> asyncio.Task:
        task = entry.retirement_task
        if task is not None and not task.done():
            return task
        if task is not None and _task_returned_true(task):
            return task
        if not entry.retiring:
            entry.retiring = True
            entry.retire_reason = reason
        elif entry.retire_reason is None:
            entry.retire_reason = reason
        task = asyncio.create_task(self._retire_entry(key, entry))
        entry.retirement_task = task
        return task

    async def _retire_entry(
        self,
        key: Tuple[str, str],
        entry: _ClientEntry,
    ) -> bool:
        await entry.leases_drained.wait()
        try:
            await entry.client.shutdown()
        except Exception as e:  # noqa: BLE001
            message = f"{type(e).__name__}: {e}"
            with self._state_lock:
                entry.retirement_error = message
                self._broken.add(key)
            logger.warning(
                "LSP generation %s cleanup failed for %s/%s: %s",
                entry.generation,
                key[0],
                key[1],
                message,
            )
            return False
        if entry.retire_reason == "idle timeout":
            # Clear the active-announcement key and emit the reap INFO while
            # this retiring entry is still published.  A replacement cannot
            # become visible until this task returns, so multi-key sweeps
            # cannot downgrade an early replacement to DEBUG reuse.
            eventlog.log_reaped([key], self._idle_timeout)
        with self._state_lock:
            entry.retirement_error = None
            if self._clients.get(key) is entry:
                self._clients.pop(key, None)
                self._last_used.pop(key, None)
        return True

    async def _break_key_async(self, key: Tuple[str, str]) -> None:
        with self._state_lock:
            spawning = self._spawning.get(key)
            entry = self._clients.get(key)
            retirement = (
                self._begin_retirement_locked(key, entry, "pair marked broken")
                if entry is not None
                else None
            )
        if spawning is not None:
            spawning.cancel()
        retained = [task for task in (spawning, retirement) if task is not None]
        if retained:
            await asyncio.gather(
                *(asyncio.shield(task) for task in retained),
                return_exceptions=True,
            )

    async def _start_idle_reaper(self) -> None:
        self._idle_reaper_task = asyncio.create_task(self._idle_reaper_loop())

    async def _idle_reaper_loop(self) -> None:
        interval = min(60.0, self._idle_timeout)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._reap_idle_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # A transient sweep error must not kill the reaper —
                # otherwise one bad shutdown permanently re-opens the
                # unbounded-accumulation leak this loop exists to fix.
                logger.debug("LSP idle reaper sweep error: %s", e)

    async def _reap_idle_once(self) -> None:
        cutoff = time.time() - self._idle_timeout
        with self._state_lock:
            idle_entries = [
                (key, entry)
                for key, entry in self._clients.items()
                if not entry.retiring and self._last_used.get(key, 0) < cutoff
            ]
            retirements = [
                self._begin_retirement_locked(key, entry, "idle timeout")
                for key, entry in idle_entries
            ]
        if retirements:
            await asyncio.gather(
                *(asyncio.shield(task) for task in retirements),
                return_exceptions=True,
            )

    async def _shutdown_async(self) -> bool:
        task = self._shutdown_task
        if task is None or (task.done() and not _task_returned_true(task)):
            task = asyncio.create_task(self._shutdown_impl())
            self._shutdown_task = task
        return bool(await asyncio.shield(task))

    async def _shutdown_impl(self) -> bool:
        with self._state_lock:
            self._admitting = False
            self._shutdown_state = "closing"
            self._shutdown_error = None

        reaper = self._idle_reaper_task
        self._idle_reaper_task = None
        if reaper is not None:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)

        with self._state_lock:
            spawning = list(self._spawning.values())
        for task in spawning:
            task.cancel()
        if spawning:
            await asyncio.gather(
                *(asyncio.shield(task) for task in spawning),
                return_exceptions=True,
            )

        with self._state_lock:
            retirements = [
                self._begin_retirement_locked(key, entry, "service shutdown")
                for key, entry in list(self._clients.items())
            ]
        results = []
        if retirements:
            results = await asyncio.gather(
                *(asyncio.shield(task) for task in retirements),
                return_exceptions=True,
            )

        with self._state_lock:
            failures = [result for result in results if result is not True]
            succeeded = not failures and not self._clients and not self._spawning
            self._clients_drained = succeeded
            if succeeded:
                self._shutdown_state = "closed"
                self._shutdown_error = None
            else:
                self._shutdown_state = "failed"
                if failures:
                    self._shutdown_error = "one or more client generations failed to retire"
                elif self._spawning:
                    self._shutdown_error = "one or more client generations are still spawning"
                else:
                    self._shutdown_error = "one or more client generations are still retiring"
            if succeeded:
                self._broken.clear()
                self._last_used.clear()
        return succeeded

    # ------------------------------------------------------------------
    # status / introspection (used by ``hermes lsp status``)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the service for the CLI status command."""
        with self._state_lock:
            clients = [
                {
                    "server_id": k[0],
                    # Multi-root servers share a blank key component; report
                    # the first announced workspace folder instead.
                    "workspace_root": k[1]
                    or (entry.client.workspace_folders or [""])[0],
                    "workspace_folders": list(entry.client.workspace_folders),
                    "state": entry.client.state,
                    "running": entry.client.is_running,
                }
                for k, entry in self._clients.items()
            ]
            broken = list(self._broken)
        return {
            "enabled": self._enabled,
            "wait_mode": self._wait_mode,
            "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy,
            "clients": clients,
            "broken": broken,
            "disabled_servers": sorted(self._disabled_servers),
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
