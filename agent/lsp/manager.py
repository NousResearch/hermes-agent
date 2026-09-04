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
  re-use it.  Servers flagged ``multi_root`` (pyright) get ONE client
  per ``server_id``; further roots — typically sibling git worktrees —
  are attached to the running process via
  ``workspace/didChangeWorkspaceFolders`` instead of a new spawn.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.lsp import eventlog
from agent.lsp.client import DIAGNOSTICS_DOCUMENT_WAIT, LSPClient, _diagnostic_key as _diag_key
from agent.lsp.servers import ServerContext, ServerDef, find_server_for_file, language_id_for
from agent.lsp.workspace import clear_cache, resolve_workspace_for_file

logger = logging.getLogger("agent.lsp.manager")

DEFAULT_IDLE_TIMEOUT = 600  # seconds; servers idle for >10min get reaped
MIN_IDLE_TIMEOUT = 30  # floor for config values; must exceed any per-op wait budget

_Key = Tuple[str, str]
_Diags = List[Dict[str, Any]]


def _client_key(srv: ServerDef, root: str) -> _Key:
    """Cache key for the client serving ``root``: multi-root servers share one process per
    ``server_id``; everything else is keyed per resolved project root."""
    return (srv.server_id, "" if srv.multi_root else root)


class _BackgroundLoop:
    """A daemon thread owning one asyncio loop; :meth:`run` blocks on a coroutine."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_forever, name="hermes-lsp-loop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_forever(self) -> None:
        loop = self._loop = asyncio.new_event_loop()
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
        """Submit a coroutine to the loop and block for its result (or raise)."""
        from agent.async_utils import safe_schedule_threadsafe
        if self._loop is None:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("background loop not started")
        if (fut := safe_schedule_threadsafe(coro, self._loop)) is None:
            raise RuntimeError("background loop not running")
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise

    def stop(self) -> None:
        loop, self._loop = self._loop, None
        thread, self._thread = self._thread, None
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if thread is not None:
            thread.join(timeout=2.0)


class LSPService:
    """The process-wide LSP service; use :func:`agent.lsp.get_service` rather than constructing directly."""

    def __init__(
        self, *, enabled: bool, wait_mode: str, wait_timeout: float, install_strategy: str,
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
        self._clients: Dict[_Key, LSPClient] = {}
        self._broken: set = set()
        self._spawning: Dict[_Key, asyncio.Future] = {}
        self._last_used: Dict[_Key, float] = {}
        self._state_lock = threading.Lock()
        self._idle_reaper_task: Optional[asyncio.Task] = None
        # abs file path → diagnostics snapshot taken immediately before a write.
        self._delta_baseline: Dict[str, _Diags] = {}

        if self._enabled and self._idle_timeout > 0:
            self._loop.run(self._start_idle_reaper(), timeout=2.0)

    @classmethod
    def create_from_config(cls) -> Optional["LSPService"]:
        """Build a service from ``hermes_cli.config``; ``None`` if config can't load."""
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP config load failed: %s", e)
            return None
        lsp_cfg = cfg.get("lsp") if isinstance(cfg, dict) else None
        lsp_cfg = lsp_cfg if isinstance(lsp_cfg, dict) else {}
        try:
            idle_timeout = float(lsp_cfg.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        except (TypeError, ValueError):
            idle_timeout = DEFAULT_IDLE_TIMEOUT
        if 0 < idle_timeout < MIN_IDLE_TIMEOUT:
            # Below the per-op wait budget the reaper could kill a client mid-flight and the outer
            # timeout would then mark the pair broken for the process lifetime.  Clamp (0 still disables).
            idle_timeout = MIN_IDLE_TIMEOUT
        servers_cfg = lsp_cfg.get("servers") or {}
        servers = {n: c for n, c in servers_cfg.items() if isinstance(c, dict)} if isinstance(servers_cfg, dict) else {}
        return cls(
            enabled=bool(lsp_cfg.get("enabled", True)),
            wait_mode=lsp_cfg.get("wait_mode", "document"),
            wait_timeout=float(lsp_cfg.get("wait_timeout", DIAGNOSTICS_DOCUMENT_WAIT)),
            install_strategy=lsp_cfg.get("install_strategy", "auto"),
            binary_overrides={n: c["command"] for n, c in servers.items()
                              if isinstance(c.get("command"), list) and c["command"]},
            env_overrides={n: {k: str(v) for k, v in c["env"].items()} for n, c in servers.items()
                           if isinstance(c.get("env"), dict)},
            init_overrides={n: c["initialization_options"] for n, c in servers.items()
                            if isinstance(c.get("initialization_options"), dict)},
            disabled_servers=[n for n, c in servers.items() if c.get("disabled")],
            idle_timeout=idle_timeout,
        )

    # ---- public API ----

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled

    def _broken_key(self, srv: ServerDef, file_path: str) -> Optional[_Key]:
        """``(server_id, per-server root)`` broken-set key, or ``None`` when the file isn't gated in.

        Falls back to the workspace root when the per-server resolver fails —
        the same key ``_get_or_spawn`` would have used when it failed.
        """
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            return None
        try:
            return (srv.server_id, srv.resolve_root(file_path, ws_root) or ws_root)
        except Exception:  # noqa: BLE001
            return (srv.server_id, ws_root)

    def enabled_for(self, file_path: str) -> bool:
        """True iff LSP should run for this file: registered non-disabled server, git workspace,
        and pair not broken (a failed server costs nothing until ``hermes lsp restart`` / exit)."""
        srv = find_server_for_file(file_path) if self._enabled else None
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        key = self._broken_key(srv, file_path)
        return key is not None and key not in self._broken

    def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics for ``file_path`` as the delta baseline (call BEFORE a write).
        Best-effort: failures are swallowed so a flaky server can't break a write, but they mark the pair broken."""
        if not self.enabled_for(file_path):
            return
        try:
            # Outer budget must exceed the inner wait or a slow-but-alive server gets falsely marked broken.
            t = max(8.0, self._wait_timeout + 3.0)
            diags = self._loop.run(self._snapshot_async(file_path), timeout=t)
        except Exception as e:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            diags = []
        self._delta_baseline[os.path.abspath(file_path)] = diags or []

    def get_diagnostics_sync(
        self, file_path: str, *, delta: bool = True, timeout: Optional[float] = None,
        line_shift: Optional[Callable[[int], Optional[int]]] = None,
    ) -> _Diags:
        """Synchronously open ``file_path``, wait for diagnostics, return them.  Never raises.

        With ``delta`` (default) the result excludes the :meth:`snapshot_baseline`; ``line_shift`` (from
        :func:`agent.lsp.range_shift.build_line_shift`) remaps that baseline into post-edit coordinates
        first, so pre-existing diagnostics that merely moved don't look introduced by this edit.
        ``[]`` when LSP is disabled, nothing matches, or the server can't be spawned.
        """
        if not self.enabled_for(file_path):
            return []
        server_id = find_server_for_file(file_path).server_id  # enabled_for guarantees a match
        try:
            t = timeout if timeout is not None else self._wait_timeout + 2.0
            diags = self._loop.run(self._open_and_wait_async(file_path), timeout=t)
        except Exception as e:  # noqa: BLE001
            if isinstance(e, asyncio.TimeoutError):
                eventlog.log_timeout(server_id, file_path)
                logger.debug("LSP diagnostics timeout for %s: %s", file_path, e)
            else:
                eventlog.log_server_error(server_id, file_path, e)
                logger.debug("LSP diagnostics fetch failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []
        if diags is None:
            # Server alive but no verdict on the post-edit content in budget (common for tsserver on big
            # projects).  Report "no data" rather than stale stores — that would be the ghost-diagnostics
            # bug.  Not marked broken: slow is not dead.
            eventlog.log_timeout(server_id, file_path, kind="fresh diagnostics")
            return []
        if delta:
            diags = self._apply_delta(file_path, diags, line_shift)
        if diags:
            eventlog.log_diagnostics(server_id, file_path, len(diags))
        else:
            eventlog.log_clean(server_id, file_path)
        return diags

    def _apply_delta(self, file_path: str, diags: _Diags, line_shift: Optional[Callable[[int], Optional[int]]]) -> _Diags:
        """Drop diagnostics present in the pre-write baseline, then roll the baseline forward."""
        abs_path = os.path.abspath(file_path)
        baseline = self._delta_baseline.get(abs_path) or []
        if baseline:
            if line_shift is not None:
                # Entries that map into a deleted region drop out — they no longer apply.
                from agent.lsp.range_shift import shift_baseline
                baseline = shift_baseline(baseline, line_shift)
            seen = {_diag_key(d) for d in baseline}
            diags = [d for d in diags if _diag_key(d) not in seen]
        # Roll the baseline forward so the next call is a delta against this state.
        try:
            fresh = self._loop.run(self._current_diags_async(file_path), timeout=2.0) or []
        except Exception:  # noqa: BLE001
            fresh = []
        if fresh:
            self._delta_baseline[abs_path] = fresh
        return diags

    def _mark_broken_for_file(self, file_path: str, exc: BaseException) -> None:
        """Mark the file's ``(server_id, root)`` pair broken after an outer timeout/error.
        The outer ``_loop.run`` timeout cancels the in-flight spawn before ``_get_or_spawn`` could record
        the failure; without this every later write would re-pay the full timeout.  Also kills any
        half-initialized client and logs the failure once."""
        srv = find_server_for_file(file_path)
        key = self._broken_key(srv, file_path) if srv is not None else None
        if key is None:
            return
        already_broken = key in self._broken
        self._broken.add(key)

        # Kill any client we managed to spawn before the timeout.  The
        # cancelled future never reached the broken-set add inside
        # ``_get_or_spawn`` so the client may still be hanging in
        # ``_clients`` with a half-initialized state.
        ckey = _client_key(srv, per_server_root)
        with self._state_lock:
            client = self._clients.pop(ckey, None)
            self._last_used.pop(ckey, None)
        if client is not None:
            try:
                # Fire-and-forget shutdown — we're already on a slow path.
                self._loop.run(client.shutdown(), timeout=1.0)
            except Exception:  # noqa: BLE001
                pass
        if not already_broken:
            eventlog.log_spawn_failed(key[0], key[1], exc)

    def shutdown(self) -> None:
        """Tear down all clients and stop the background loop."""
        if not self._enabled:
            return
        try:
            self._loop.run(self._shutdown_async(), timeout=10.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
        self._loop.stop()
        clear_cache()

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the service for ``hermes lsp status``."""
        with self._state_lock:
            clients = [
                {"server_id": c.server_id, "workspace_root": c.workspace_root,
                 "workspace_folders": list(c.workspace_folders), "state": c.state, "running": c.is_running}
                for c in self._clients.values()
            ]
            broken = list(self._broken)
        return {
            "enabled": self._enabled, "wait_mode": self._wait_mode, "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy, "clients": clients, "broken": broken,
            "disabled_servers": sorted(self._disabled_servers),
        }

    # ---- async internals ----

    async def _snapshot_async(self, file_path: str) -> _Diags:
        # No fresh data for the pre-edit content → empty baseline.  Safe: the delta
        # filter then removes less, never more.  Never seed from stale stores.
        return await self._open_and_wait_async(file_path, snapshot=True) or []

    async def _open_and_wait_async(self, file_path: str, *, snapshot: bool = False) -> Optional[_Diags]:
        """Open + wait for FRESH diagnostics: ``[]`` = checked clean, ``None`` = no verdict in budget.

        Callers must not substitute stale data for either.  ``snapshot`` mode
        (pre-write baseline) skips didSave and uses the default wait budget.
        """
        client = await self._get_or_spawn(file_path)
        if client is None:
            return None
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            if not snapshot:
                await client.save_file(file_path)
            fresh = await client.wait_for_diagnostics(
                file_path, version, mode=self._wait_mode, timeout=None if snapshot else self._wait_timeout,
            )
        except Exception as e:  # noqa: BLE001
            if snapshot:
                logger.debug("snapshot open/wait failed: %s", e)
            else:
                logger.debug("open/wait failed for %s: %s", file_path, e)
            return None
        self._touch(client)
        return list(client.diagnostics_for(file_path, fresh_only=True)) if fresh else None

    async def _current_diags_async(self, file_path: str) -> _Diags:
        ws, gated = resolve_workspace_for_file(file_path)
        srv = find_server_for_file(file_path)
        if not (ws and gated and srv):
            return []
        with self._state_lock:
            client = self._clients.get(_client_key(srv, ws))
        if client is None:
            return []
        with self._state_lock:
            client = self._clients.get(_client_key(srv, root))
        return list(client.diagnostics_for(file_path, fresh_only=True)) if client else []

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

        if (srv.server_id, per_server_root) in self._broken:
            return None
        key = _client_key(srv, per_server_root)
        with self._state_lock:
            client = self._clients.get(key)
            if client is not None and client.is_running:
                self._last_used[key] = time.time()
                eventlog.log_active(srv.server_id, per_server_root)
            else:
                client = None
            spawning = self._spawning.get(key)
        if client is None and spawning is not None:
            try:
                client = await spawning
            except Exception:  # noqa: BLE001
                return None
        if client is not None:
            if srv.multi_root:
                await client.add_workspace_folder(per_server_root)
            return client

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
                self._broken.add((srv.server_id, per_server_root))
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
                self._broken.add((srv.server_id, per_server_root))
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

    @staticmethod
    async def _attach_root(srv: ServerDef, client: LSPClient, root: str) -> LSPClient:
        """Multi-root servers: announce ``root`` to the shared process instead of spawning another."""
        if srv.multi_root:
            await client.add_workspace_folder(root)
        return client

    async def _spawn_client(self, srv: ServerDef, root: str) -> Optional[LSPClient]:
        """Resolve the binary and start a client; ``None`` (after logging) when either fails."""
        ctx = ServerContext(
            workspace_root=root, install_strategy=self._install_strategy, binary_overrides=self._binary_overrides,
            env_overrides=self._env_overrides, init_overrides=self._init_overrides,
        )
        spec = srv.build_spawn(root, ctx)
        if spec is None:
            # Binary not locatable (auto-install off, manual-only, or install failed) — surface once.
            eventlog.log_server_unavailable(srv.server_id, srv.server_id)
            return None
        client = LSPClient(
            server_id=srv.server_id, workspace_root=spec.workspace_root, command=spec.command, env=spec.env,
            cwd=spec.cwd, initialization_options=spec.initialization_options,
            seed_diagnostics_on_first_push=spec.seed_diagnostics_on_first_push or srv.seed_first_push,
        )
        try:
            await client.start()
        except Exception as e:  # noqa: BLE001
            eventlog.log_spawn_failed(srv.server_id, root, e)
            return None
        return client

    def _touch(self, client: LSPClient) -> None:
        """Refresh the last-used timestamp for a client we just used.

        Guarded on membership so a reaped-mid-operation client can't
        resurrect an orphan ``_last_used`` entry after the reaper popped
        the key.  All writers and the reaper run on the background loop
        thread; the lock keeps this consistent with the reader anyway.
        """
        with self._state_lock:
            for key, c in self._clients.items():
                if c is client:
                    self._last_used[key] = time.time()

    async def _idle_reaper_loop(self) -> None:
        interval = min(60.0, self._idle_timeout)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._reap_idle_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # A transient sweep error must not kill the reaper, or the accumulation leak it fixes comes back.
                logger.debug("LSP idle reaper sweep error: %s", e)

    async def _reap_idle_once(self) -> None:
        cutoff = time.time() - self._idle_timeout
        with self._state_lock:
            idle_keys = [key for key in self._clients if self._last_used.get(key, 0) < cutoff]
            clients = [self._clients.pop(key) for key in idle_keys]
            for key in idle_keys:
                self._last_used.pop(key, None)
        if clients:
            eventlog.log_reaped([(c.server_id, c.workspace_root) for c in clients], self._idle_timeout)
            await asyncio.gather(*(client.shutdown() for client in clients), return_exceptions=True)

    async def _shutdown_async(self) -> None:
        if (reaper := self._idle_reaper_task) is not None:
            self._idle_reaper_task = None
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
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
                    "server_id": c.server_id,
                    "workspace_root": c.workspace_root,
                    "workspace_folders": list(c.workspace_folders),
                    "state": c.state,
                    "running": c.is_running,
                }
                for c in self._clients.values()
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


def _client_key(srv, root: str) -> Tuple[str, str]:
    """Cache key for the client serving ``root``.

    Multi-root servers share one process per ``server_id``; everything
    else is keyed per resolved project root.
    """
    return (srv.server_id, "" if srv.multi_root else root)


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
