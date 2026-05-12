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

The service is **off by default** — call :meth:`is_active` to check
whether it's actually doing anything.  When LSP is disabled in
config, when no git workspace can be detected, when all configured
servers are missing binaries and auto-install is off, ``is_active``
returns False and the file_operations layer falls through to the
in-process syntax check.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Dict, List, Optional, Tuple

from agent.lsp import eventlog
from agent.lsp.client import (
    DIAGNOSTICS_DOCUMENT_WAIT,
    LSPClient,
    file_uri,
)
from agent.lsp.servers import (
    ServerContext,
    ServerDef,
    SpawnSpec,
    find_server_for_file,
    language_id_for,
)
from agent.lsp.workspace import (
    clear_cache,
    is_inside_workspace,
    resolve_workspace_for_file,
)

logger = logging.getLogger("agent.lsp.manager")

DEFAULT_IDLE_TIMEOUT = 600  # seconds; servers idle for >10min get reaped


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
        if self._loop is None:
            raise RuntimeError("background loop not started")
        fut: ConcurrentFuture = asyncio.run_coroutine_threadsafe(coro, self._loop)
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

    def schedule(self, coro) -> Optional[ConcurrentFuture]:
        """Submit a coroutine to run on the loop *without* blocking.

        Returns a ``concurrent.futures.Future`` that can be cancelled
        from the calling thread.  Use this for long-running background
        tasks (e.g. the idle reaper) whose result you don't await but
        whose lifecycle you do want to manage.  Returns ``None`` if
        the loop hasn't been started.
        """
        if self._loop is None:
            return None
        try:
            return asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # Loop closed between None-check and schedule.
            return None


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
        self._wait_mode = wait_mode if wait_mode in ("document", "full") else "document"
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
        self._clients: Dict[Tuple[str, str], LSPClient] = {}
        self._broken: set = set()
        self._spawning: Dict[Tuple[str, str], asyncio.Future] = {}
        self._last_used: Dict[Tuple[str, str], float] = {}
        self._state_lock = threading.Lock()

        # Delta baseline: opaque token → {"path": abs_path, "diags": [...], "created": ts}.
        # The token-keyed design (replacing the earlier path-keyed dict) makes
        # the baseline-to-lint relationship explicit and gateway-safe: each
        # snapshot_baseline() returns a fresh token, the caller threads it
        # through to get_diagnostics_sync(baseline_token=...), and there is
        # no implicit global state that two concurrent edits could collide
        # on.  ``_evict_stale_baselines`` reaps any tokens whose write path
        # crashed between snapshot and lint.
        self._delta_baseline: Dict[str, Dict[str, Any]] = {}
        self._baseline_lock = threading.Lock()

        # Idle-reaper handle.  When ``enabled`` is True we schedule a
        # background coroutine on ``self._loop`` that periodically
        # shuts down clients whose ``_last_used`` is older than
        # ``self._idle_timeout``.  Without this, a long-lived gateway
        # process accumulates one LSP subprocess per (language,
        # workspace) ever touched — pyright at ~200MB, gopls at ~80MB,
        # tsserver at ~150MB.  ``shutdown()`` cancels the handle.
        self._reaper_handle: Optional[ConcurrentFuture] = None
        if self._enabled:
            self._reaper_handle = self._loop.schedule(self._reaper_loop())

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

        # NOTE: fallback default mirrors hermes_cli.config.DEFAULT_CONFIG['lsp']
        # — opt-in for audit-compliance reasons.  If the user's config has no
        # ``lsp.enabled`` key at all, treat that as "user hasn't opted in".
        enabled = bool(lsp_cfg.get("enabled", False))
        wait_mode = lsp_cfg.get("wait_mode", "document")
        wait_timeout = float(lsp_cfg.get("wait_timeout", DIAGNOSTICS_DOCUMENT_WAIT))
        install_strategy = lsp_cfg.get("install_strategy", "manual")
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
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled

    def enabled_for(self, file_path: str) -> bool:
        """Return True iff LSP should run for this specific file.

        Gates on workspace detection (file or cwd inside a git worktree)
        and on whether any registered server matches the extension.
        """
        if not self._enabled:
            return False
        srv = find_server_for_file(file_path)
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        ws_root, gated_in = resolve_workspace_for_file(file_path)
        return bool(ws_root and gated_in)

    def snapshot_baseline(self, file_path: str) -> Optional[str]:
        """Snapshot current diagnostics for ``file_path`` as a delta baseline.

        Called BEFORE a write so the matching post-write
        ``get_diagnostics_sync(..., baseline_token=token)`` can filter
        out pre-existing errors.  Best-effort — failures are silently
        swallowed so a flaky server can't break a write.

        Returns an opaque token identifying this specific snapshot, or
        ``None`` if LSP is not active for this file.

        The token-keyed design (vs. the earlier path-keyed map) is
        load-bearing for two scenarios:

        * **Gateway concurrency.** Two chats editing the same path in
          different worktrees won't stomp each other's baselines.
        * **Recursive write paths.** ``patch_replace`` calls
          ``write_file`` (which snapshots + post-write-checks) and then
          runs its own post-write check; without token isolation the
          second check sees the rolled-forward baseline from the first
          and always reports clean.
        """
        if not self.enabled_for(file_path):
            return None
        token = uuid.uuid4().hex
        try:
            diags = self._loop.run(self._snapshot_async(file_path), timeout=8.0)
            with self._baseline_lock:
                self._delta_baseline[token] = {
                    "path": os.path.abspath(file_path),
                    "diags": diags or [],
                    "created": time.time(),
                }
        except Exception as e:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, e)
            # Empty baseline still gets a token — any post-edit diagnostic
            # is then considered "new" (safe default).
            with self._baseline_lock:
                self._delta_baseline[token] = {
                    "path": os.path.abspath(file_path),
                    "diags": [],
                    "created": time.time(),
                }
        # Bound the token map so a long-lived gateway doesn't leak entries
        # for any tokens the caller forgets to consume.
        self._evict_stale_baselines()
        return token

    def _evict_stale_baselines(self) -> None:
        """Drop baseline entries older than 5 minutes.

        Bounded fallback for callers that snapshot but never call
        ``get_diagnostics_sync`` with the token (e.g. a write that
        fails between snapshot and lint).  5 minutes is generous —
        legitimate write→lint cycles complete in well under a second.
        """
        cutoff = time.time() - 300.0
        with self._baseline_lock:
            stale = [
                tok for tok, entry in self._delta_baseline.items()
                if entry["created"] < cutoff
            ]
            for tok in stale:
                self._delta_baseline.pop(tok, None)

    def get_diagnostics_sync(
        self,
        file_path: str,
        *,
        delta: bool = True,
        timeout: Optional[float] = None,
        baseline_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously open ``file_path`` in the right server, wait for
        diagnostics, return them.

        If ``delta`` is True (default) and a ``baseline_token`` is
        supplied, the result is filtered against the baseline that
        ``snapshot_baseline()`` captured for that token.  Diagnostics
        present in the baseline are removed so the caller only sees
        errors introduced by the current edit.  The token is consumed
        (popped) on use so a follow-up call without a fresh snapshot
        won't reuse stale state.

        When ``baseline_token`` is ``None`` (no prior snapshot) the
        delta filter is a no-op — every diagnostic surfaces.  Callers
        wanting delta semantics MUST thread a token from a paired
        ``snapshot_baseline()`` call.

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
            return []
        except Exception as e:  # noqa: BLE001
            eventlog.log_server_error(server_id, file_path, e)
            logger.debug("LSP diagnostics fetch failed for %s: %s", file_path, e)
            return []

        if delta and baseline_token is not None:
            with self._baseline_lock:
                entry = self._delta_baseline.pop(baseline_token, None)
            if entry is not None:
                baseline_diags = entry.get("diags") or []
                if baseline_diags:
                    seen = {_diag_key(d) for d in baseline_diags}
                    diags = [d for d in diags if _diag_key(d) not in seen]

        if diags:
            eventlog.log_diagnostics(server_id, file_path, len(diags))
        else:
            eventlog.log_clean(server_id, file_path)
        return diags

    def shutdown(self) -> None:
        """Tear down all clients and stop the background loop."""
        if not self._enabled:
            return
        # Cancel the reaper first so it doesn't race with the client
        # teardown below.  ``ConcurrentFuture.cancel()`` is best-effort
        # while the coroutine is mid-sleep — the cancellation flag
        # propagates to ``asyncio.sleep`` which raises CancelledError.
        if self._reaper_handle is not None:
            self._reaper_handle.cancel()
            self._reaper_handle = None
        try:
            self._loop.run(self._shutdown_async(), timeout=10.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
        self._loop.stop()
        clear_cache()

    # ------------------------------------------------------------------
    # async internals
    # ------------------------------------------------------------------

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("snapshot open/wait failed: %s", e)
            return []
        self._last_used[(client.server_id, client.workspace_root)] = time.time()
        return list(client.diagnostics_for(file_path))

    async def _open_and_wait_async(self, file_path: str) -> List[Dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            await client.save_file(file_path)
            await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("open/wait failed for %s: %s", file_path, e)
            return []
        self._last_used[(client.server_id, client.workspace_root)] = time.time()
        return list(client.diagnostics_for(file_path))

    async def _current_diags_async(self, file_path: str) -> List[Dict[str, Any]]:
        ws, gated = resolve_workspace_for_file(file_path)
        srv = find_server_for_file(file_path)
        if not (ws and gated and srv):
            return []
        with self._state_lock:
            client = self._clients.get((srv.server_id, ws))
        if client is None:
            return []
        return list(client.diagnostics_for(file_path))

    async def _get_or_spawn(self, file_path: str) -> Optional[LSPClient]:
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

        key = (srv.server_id, per_server_root)
        if key in self._broken:
            return None
        with self._state_lock:
            client = self._clients.get(key)
            if client is not None and client.is_running:
                eventlog.log_active(srv.server_id, per_server_root)
                return client
            spawning = self._spawning.get(key)
        if spawning is not None:
            try:
                return await spawning
            except Exception:  # noqa: BLE001
                return None

        # Begin spawn
        loop = asyncio.get_running_loop()
        spawn_future: asyncio.Future = loop.create_future()
        with self._state_lock:
            self._spawning[key] = spawn_future
        try:
            ctx = ServerContext(
                workspace_root=per_server_root,
                install_strategy=self._install_strategy,
                binary_overrides=self._binary_overrides,
                env_overrides=self._env_overrides,
                init_overrides=self._init_overrides,
            )
            spec = srv.build_spawn(per_server_root, ctx)
            if spec is None:
                # ``build_spawn`` returns None when the binary can't be
                # located (auto-install disabled, manual-only server,
                # or install attempt failed).  Surface this once via
                # the structured logger so the user can act on it.
                eventlog.log_server_unavailable(srv.server_id, srv.server_id)
                self._broken.add(key)
                spawn_future.set_result(None)
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
            try:
                await client.start()
            except Exception as e:  # noqa: BLE001
                eventlog.log_spawn_failed(srv.server_id, per_server_root, e)
                self._broken.add(key)
                spawn_future.set_result(None)
                return None
            with self._state_lock:
                self._clients[key] = client
            self._last_used[key] = time.time()
            eventlog.log_active(srv.server_id, per_server_root)
            spawn_future.set_result(client)
            return client
        finally:
            with self._state_lock:
                self._spawning.pop(key, None)

    async def _reaper_loop(self) -> None:
        """Periodically shut down LSP clients idle past the timeout.

        Runs forever on the background event loop.  Each iteration:

        1. Sleep for ``idle_timeout / 2`` seconds (so a client just
           past the threshold gets reaped within one full timeout
           window, not two).
        2. Snapshot ``_last_used`` under the state lock.
        3. For each entry older than the cutoff, pop the client out
           of ``_clients`` and call ``client.shutdown()`` without
           holding the lock (shutdown can take a few seconds and we
           don't want to block in-flight requests).

        Cancelled by ``shutdown()``.  Any exception inside the loop
        is logged and swallowed — the reaper must never crash the
        background loop.
        """
        # Don't run faster than every 10s even if idle_timeout is very
        # low (test-only).  Keeps the loop from busy-spinning.
        interval = max(self._idle_timeout / 2.0, 10.0)
        while True:
            try:
                await asyncio.sleep(interval)
                await self._reap_idle()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug("LSP reaper iteration failed: %s", e)

    async def _reap_idle(self) -> None:
        """Reap clients idle past ``_idle_timeout`` seconds.

        Separated from ``_reaper_loop`` so tests can drive a single
        reap pass without waiting for the sleep interval.
        """
        cutoff = time.time() - self._idle_timeout
        with self._state_lock:
            stale_keys = [
                key for key, last in self._last_used.items()
                if last < cutoff
            ]
            stale_clients = []
            for key in stale_keys:
                client = self._clients.pop(key, None)
                self._last_used.pop(key, None)
                if client is not None:
                    stale_clients.append((key, client))
        # Drop the lock before calling client.shutdown() — it does
        # network I/O over stdio and can take a few seconds.
        for key, client in stale_clients:
            server_id, workspace_root = key
            try:
                await client.shutdown()
                logger.debug(
                    "LSP reaped idle client server=%s workspace=%s",
                    server_id, workspace_root,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "LSP reaper shutdown error for server=%s workspace=%s: %s",
                    server_id, workspace_root, e,
                )

    async def _shutdown_async(self) -> None:
        with self._state_lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self._broken.clear()
            self._last_used.clear()
        await asyncio.gather(
            *(c.shutdown() for c in clients),
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # status / introspection (used by ``hermes lsp status``)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the service for the CLI status command."""
        with self._state_lock:
            clients = [
                {
                    "server_id": k[0],
                    "workspace_root": k[1],
                    "state": c.state,
                    "running": c.is_running,
                }
                for k, c in self._clients.items()
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
    """Content equality key used for delta filtering.  Mirrors
    :func:`agent.lsp.client._diagnostic_key`."""
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
