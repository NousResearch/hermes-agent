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
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.lsp import eventlog
from agent.lsp.client import (
    DIAGNOSTICS_DOCUMENT_WAIT,
    LSPClient,
)
from agent.lsp.metrics import LSPMetrics
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

# Reasonable defaults.  All three are overridable via config.yaml.
DEFAULT_IDLE_TIMEOUT = 180.0  # seconds; clients idle > this are reap candidates
DEFAULT_REAPER_INTERVAL = 60.0  # seconds between idle sweeps
DEFAULT_TOTAL_LSP_CAP = 12  # hard cap on simultaneously-running LSP clients
DEFAULT_PER_SERVER_LIMITS: Dict[str, int] = {
    # Maps LSP server_id -> max concurrent clients.
    "pyright": 8,
    "typescript-language-server": 4,
    "vue-language-server": 2,
    "svelte-language-server": 2,
}


def _read_process_stats(server_ids: List[str]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Best-effort open-fd count, RSS for this process, and aggregate RSS of LSP children.

    Returns (fd_count_or_None, rss_bytes_or_None, lsp_child_rss_or_None).
    Uses psutil when available; falls back to /proc on Linux and
    ``psutil``-free macOS probes when not.  Always returns a tuple.
    Never raises — the metrics surface is best-effort.

    Per the user instruction, this counts numeric descriptor entries only:
    ``len(os.listdir('/proc/self/fd'))`` on Linux, and on macOS we walk
    ``lsof -p <pid>`` and count the lines whose FD column parses as a
    bare integer (regular FDs), not memory-mapped libraries or other
    rows lsof may report.
    """
    fd_count: Optional[int] = None
    rss_bytes: Optional[int] = None
    lsp_rss: Optional[int] = None
    pid = os.getpid()

    try:
        import psutil  # type: ignore[import-untyped]

        proc: Any = None
        try:
            proc = psutil.Process(pid)
            rss_bytes = int(proc.memory_info().rss)
        except Exception:  # noqa: BLE001
            rss_bytes = None
        if proc is not None:
            try:
                fd_count = int(proc.num_fds())
            except (AttributeError, OSError):
                fd_count = None
            try:
                lsp_rss_total = 0
                lsp_seen = False
                for child in proc.children(recursive=False):
                    cmdline = " ".join(child.cmdline() or [])
                    if not cmdline:
                        continue
                    if not any(sid in cmdline for sid in server_ids):
                        continue
                    lsp_seen = True
                    try:
                        lsp_rss_total += int(child.memory_info().rss)
                    except (psutil.NoSuchProcess, OSError):
                        pass
                lsp_rss = lsp_rss_total if lsp_seen else 0
            except (psutil.NoSuchProcess, OSError):
                lsp_rss = None
    except Exception:  # noqa: BLE001
        # psutil not installed; skip — callers will see None values.
        pass
    if fd_count is None and os.path.isdir("/proc/self/fd"):
        try:
            fd_count = len(os.listdir("/proc/self/fd"))
        except OSError:
            fd_count = None

    return fd_count, rss_bytes, lsp_rss


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
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        reaper_interval: float = DEFAULT_REAPER_INTERVAL,
        total_lsp_cap: int = DEFAULT_TOTAL_LSP_CAP,
        per_server_caps: Optional[Dict[str, int]] = None,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])
        self._idle_timeout = max(0.0, float(idle_timeout))
        self._reaper_interval = max(1.0, float(reaper_interval))
        self._total_lsp_cap = max(1, int(total_lsp_cap))
        self._per_server_caps: Dict[str, int] = dict(DEFAULT_PER_SERVER_LIMITS)
        if per_server_caps:
            for sid, cap in per_server_caps.items():
                try:
                    self._per_server_caps[sid] = max(0, int(cap))
                except (TypeError, ValueError):
                    continue

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()

        # Per-(server_id, workspace_root) state
        self._clients: Dict[Tuple[str, str], LSPClient] = {}
        self._broken: set = set()
        self._spawning: Dict[Tuple[str, str], asyncio.Future] = {}
        self._last_used: Dict[Tuple[str, str], float] = {}
        self._state_lock = threading.Lock()
        self._metrics = LSPMetrics()

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write.  ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

        # Reaper task — runs on the LSP background loop.  Started lazily
        # on the first call to ``_get_or_spawn`` so the service is
        # dormant until something actually needs an LSP client.
        self._reaper_task: Optional[asyncio.Task] = None

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
        try:
            idle_timeout = float(lsp_cfg.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        except (TypeError, ValueError):
            idle_timeout = DEFAULT_IDLE_TIMEOUT
        try:
            reaper_interval = float(
                lsp_cfg.get("reaper_interval", DEFAULT_REAPER_INTERVAL)
            )
        except (TypeError, ValueError):
            reaper_interval = DEFAULT_REAPER_INTERVAL
        try:
            total_lsp_cap = int(lsp_cfg.get("total_lsp_cap", DEFAULT_TOTAL_LSP_CAP))
        except (TypeError, ValueError):
            total_lsp_cap = DEFAULT_TOTAL_LSP_CAP
        raw_caps = lsp_cfg.get("per_server_caps") or {}
        per_server_caps: Dict[str, int] = {}
        if isinstance(raw_caps, dict):
            for sid, cap in raw_caps.items():
                try:
                    per_server_caps[str(sid)] = int(cap)
                except (TypeError, ValueError):
                    continue
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
            reaper_interval=reaper_interval,
            total_lsp_cap=total_lsp_cap,
            per_server_caps=per_server_caps,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled

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
        if not self._enabled:
            return False
        srv = find_server_for_file(file_path)
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        ws_root, gated_in = resolve_workspace_for_file(file_path)
        if not (ws_root and gated_in):
            return False
        # Broken-set short-circuit.  Use the per-server root if we can
        # compute one cheaply; otherwise fall back to the workspace
        # root as the broken key (which is what _get_or_spawn would
        # have used anyway when it failed).
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        if (srv.server_id, per_server_root) in self._broken:
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
        """Mark the (server_id, workspace_root) pair as broken so subsequent
        edits skip it instantly instead of re-paying timeout cost.

        Called when the outer ``_loop.run`` timeout cancels an in-flight
        spawn/initialize that the inner ``_get_or_spawn`` task was still
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
        key = (srv.server_id, per_server_root)
        already_broken = key in self._broken
        self._broken.add(key)

        # Kill any client we managed to spawn before the timeout.  The
        # cancelled future never reached the broken-set add inside
        # ``_get_or_spawn`` so the client may still be hanging in
        # ``_clients`` with a half-initialized state.
        with self._state_lock:
            client = self._clients.pop(key, None)
        if client is not None:
            try:
                # Fire-and-forget shutdown — give it a second to cleanup,
                # but don't block.  We're already on a slow path.
                self._loop.run(client.shutdown(), timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

        if not already_broken:
            eventlog.log_spawn_failed(srv.server_id, per_server_root, exc)

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

    # ------------------------------------------------------------------
    # async internals
    # ------------------------------------------------------------------

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        try:
            async with client.track_request():
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
            async with client.track_request():
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
        # Snapshot reads the cached store synchronously; mark inflight
        # so a concurrent cap-eviction can't tear down the underlying
        # process between our lookup and the read.
        async with client.track_request():
            return list(client.diagnostics_for(file_path))

    async def _get_or_spawn(self, file_path: str) -> Optional[LSPClient]:
        await self._maybe_start_reaper_async()
        await self._reap_idle_clients_async()

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
                self._last_used[key] = time.time()
                self._metrics.incr("reuses")
                eventlog.log_active(srv.server_id, per_server_root)
                return client
            spawning = self._spawning.get(key)
        if spawning is not None:
            try:
                return await spawning
            except Exception:  # noqa: BLE001
                return None

        # Enforce caps BEFORE spawning a new process.  If we're at the
        # per-server or total limit, try to evict an idle peer first.
        evicted = await self._enforce_caps(srv.server_id)
        if not evicted:
            # Cap held by active peers — refuse to spawn.  Callers will
            # retry on the next request; meanwhile the active request
            # completes normally.
            logger.debug(
                "LSP cap held for server=%s (active peers all busy); refusing spawn",
                srv.server_id,
            )
            eventlog.log_cap_blocked(srv.server_id)
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
                self._metrics.incr("spawn_failures")
                self._broken.add(key)
                spawn_future.set_result(None)
                return None
            with self._state_lock:
                self._clients[key] = client
            self._last_used[key] = time.time()
            self._metrics.incr("spawns")
            eventlog.log_active(srv.server_id, per_server_root)
            spawn_future.set_result(client)
            return client
        finally:
            with self._state_lock:
                self._spawning.pop(key, None)

    async def _enforce_caps(self, server_id: str) -> bool:
        """Enforce per-server and total LSP caps, evicting idle peers if needed.

        Returns True if a new client may be spawned (either because the
        cap is not yet hit, or because we successfully evicted an idle
        peer).  Returns False if the cap is held by active peers.
        """
        async def _evict_one_idle() -> bool:
            """Evict the LRU idle client.  Returns True if one was evicted."""
            with self._state_lock:
                # Eligible: in _clients, NOT spawning, inflight==0,
                # not currently being torn down.  Pick by oldest
                # ``_last_used`` so the LRU peer leaves first.
                candidates = [
                    (key, last)
                    for key, last in self._last_used.items()
                    if key in self._clients
                    and key not in self._spawning
                    and not self._clients[key].has_active_requests
                ]
                if not candidates:
                    return False
                victims = [
                    self._clients[k] for k, _ in sorted(candidates, key=lambda kv: kv[1])
                ]
                victim_keys = [
                    k for k, _ in sorted(candidates, key=lambda kv: kv[1])
                ]

            evictions = 0
            for client, key in zip(victims, victim_keys):
                with self._state_lock:
                    self._clients.pop(key, None)
                    self._last_used.pop(key, None)
                try:
                    await client.shutdown()
                    self._metrics.incr("shutdowns")
                    self._metrics.incr("reaps_cap")
                    eventlog.log_cap_eviction(key[0], key[1])
                    evictions += 1
                    break
                except Exception as e:  # noqa: BLE001
                    self._metrics.incr("shutdown_failures")
                    logger.debug("LSP cap eviction shutdown failed: %s", e)
            return evictions > 0

        with self._state_lock:
            total = len(self._clients)
            same_server = sum(
                1 for sid, _ in self._clients.keys() if sid == server_id
            )

        per_server_cap = self._per_server_caps.get(server_id, self._total_lsp_cap)

        def _under_cap(t: int, s: int) -> bool:
            return t < self._total_lsp_cap and s < per_server_cap

        if _under_cap(total, same_server):
            return True

        # At or over cap.  Evict idle peers one at a time, recomputing
        # counts after each eviction so we exit as soon as the cap has
        # slack.  Bound the loop so a flood of requests can't drive us
        # into unbounded eviction work.
        for _ in range(max(2, self._total_lsp_cap)):
            if not await _evict_one_idle():
                break
            with self._state_lock:
                total = len(self._clients)
                same_server = sum(
                    1 for sid, _ in self._clients.keys() if sid == server_id
                )
            if _under_cap(total, same_server):
                return True
        return False

    async def _maybe_start_reaper_async(self) -> None:
        """Start the periodic reaper task on first use, then leave it alone."""
        if self._reaper_task is not None and not self._reaper_task.done():
            return
        if self._idle_timeout <= 0:
            return  # Reaping disabled — no point starting the task.
        loop = asyncio.get_running_loop()
        self._reaper_task = loop.create_task(
            self._reaper_loop(),
            name="hermes-lsp-reaper",
        )

    async def _reaper_loop(self) -> None:
        """Run ``_reap_idle_clients_async`` forever on the configured interval."""
        try:
            while True:
                try:
                    await asyncio.sleep(self._reaper_interval)
                except asyncio.CancelledError:
                    raise
                try:
                    await self._reap_idle_clients_async()
                except Exception as e:  # noqa: BLE001
                    logger.debug("LSP reaper tick failed: %s", e)
        except asyncio.CancelledError:
            return

    async def _reap_idle_clients_async(self, *, now: Optional[float] = None) -> None:
        """Shutdown LSP clients that have been idle longer than the timeout.

        The service keeps one language server per ``(server_id, workspace)``.
        Long-lived gateway/cron processes can touch many temporary worktrees over
        time, so stale clients must be reaped or their stdio pipes accumulate
        until the process hits RLIMIT_NOFILE.

        Never evicts a client with ``inflight > 0`` — eviction only targets
        clients that are fully idle so no in-flight request is interrupted.
        """
        if self._idle_timeout <= 0:
            return

        cutoff = (time.time() if now is None else now) - self._idle_timeout
        stale_clients: List[LSPClient] = []
        active_blocked = 0

        with self._state_lock:
            stale_keys = [
                key
                for key, last_used in list(self._last_used.items())
                if last_used < cutoff
                and key in self._clients
                and key not in self._spawning
            ]
            for key in stale_keys:
                client = self._clients.get(key)
                if client is None:
                    continue
                if client.has_active_requests:
                    active_blocked += 1
                    continue
                self._clients.pop(key, None)
                self._last_used.pop(key, None)
                stale_clients.append(client)

        if active_blocked:
            self._metrics.incr("reaps_blocked_active", active_blocked)

        if not stale_clients:
            return

        results = await asyncio.gather(
            *(client.shutdown() for client in stale_clients),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                self._metrics.incr("shutdown_failures")
                logger.debug("LSP idle client shutdown failed: %s", result)
            else:
                self._metrics.incr("shutdowns")
        self._metrics.incr("reaps_idle", len(stale_clients))

    async def _reap_idle_clients_async(self, *, now: Optional[float] = None) -> None:  # noqa: F811
        """Shutdown LSP clients that have been idle longer than the timeout.

        The service keeps one language server per ``(server_id, workspace)``.
        Long-lived gateway/cron processes can touch many temporary worktrees over
        time, so stale clients must be reaped or their stdio pipes accumulate
        until the process hits RLIMIT_NOFILE.

        Never evicts a client with ``inflight > 0`` — eviction only targets
        clients that are fully idle so no in-flight request is interrupted.
        """
        if self._idle_timeout <= 0:
            return

        cutoff = (time.time() if now is None else now) - self._idle_timeout
        stale_clients: List[LSPClient] = []
        active_blocked = 0

        with self._state_lock:
            stale_keys = [
                key
                for key, last_used in list(self._last_used.items())
                if last_used < cutoff
                and key in self._clients
                and key not in self._spawning
            ]
            for key in stale_keys:
                client = self._clients.get(key)
                if client is None:
                    continue
                if client.has_active_requests:
                    active_blocked += 1
                    continue
                self._clients.pop(key, None)
                self._last_used.pop(key, None)
                stale_clients.append(client)

        if active_blocked:
            self._metrics.incr("reaps_blocked_active", active_blocked)

        if not stale_clients:
            return

        results = await asyncio.gather(
            *(client.shutdown() for client in stale_clients),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                self._metrics.incr("shutdown_failures")
                logger.debug("LSP idle client shutdown failed: %s", result)
            else:
                self._metrics.incr("shutdowns")
        self._metrics.incr("reaps_idle", len(stale_clients))

    async def _shutdown_async(self) -> None:
        # Cancel the periodic reaper first so it can't race with our
        # shutdown and try to call ``client.shutdown()`` on already-purged
        # clients during garbage collection.
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reaper_task = None

        with self._state_lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self._broken.clear()
            self._last_used.clear()
        if not clients:
            return
        await asyncio.gather(
            *(c.shutdown() for c in clients),
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # metrics (used by ``hermes lsp status`` and the gateway's status surface)
    # ------------------------------------------------------------------

    def snapshot_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of the service's runtime state.

        Includes:
        - ``metrics``: aggregate counters (spawns, reuses, reaps_idle,
          reaps_cap, reaps_blocked_active, shutdowns, shutdown_failures,
          spawn_failures)
        - ``clients``: list of {server_id, workspace_root, inflight,
          last_used, is_running} for every live client
        - ``caps``: total cap, per-server caps, current counts
        - ``idle_timeout_seconds``, ``reaper_interval_seconds``
        - ``process_fd_count``: best-effort open-fd count for the
          current process (requires psutil, else None)
        - ``process_rss_bytes``: best-effort RSS for the current process
          (requires psutil, else None)
        - ``lsp_child_rss_bytes``: sum of RSS for LSP child PIDs, best-effort
        """
        with self._state_lock:
            clients_payload = [
                {
                    "server_id": k[0],
                    "workspace_root": k[1],
                    "inflight": self._clients[k].inflight,
                    "last_used": self._last_used.get(k),
                    "is_running": self._clients[k].is_running,
                }
                for k in list(self._clients.keys())
            ]
            counts_per_server: Dict[str, int] = {}
            for k in self._clients.keys():
                counts_per_server[k[0]] = counts_per_server.get(k[0], 0) + 1
            live_server_ids = sorted({k[0] for k in self._clients.keys()})

        fd_count, rss_bytes, lsp_rss = _read_process_stats(live_server_ids)
        return {
            "metrics": self._metrics.as_dict(),
            "clients": clients_payload,
            "caps": {
                "total_lsp_cap": self._total_lsp_cap,
                "per_server_caps": dict(self._per_server_caps),
                "current_total": len(clients_payload),
                "current_per_server": counts_per_server,
            },
            "idle_timeout_seconds": self._idle_timeout,
            "reaper_interval_seconds": self._reaper_interval,
            "process_fd_count": fd_count,
            "process_rss_bytes": rss_bytes,
            "lsp_child_rss_bytes": lsp_rss,
        }

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
