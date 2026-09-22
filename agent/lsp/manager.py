"""Service-level orchestration for LSP clients.

:class:`LSPService` bridges the synchronous file_operations layer and the async
:class:`agent.lsp.client.LSPClient`: one asyncio loop in a background thread, one lazily
spawned client per ``(server_id, workspace_root)`` — servers flagged ``multi_root`` (pyright) get ONE
client per ``server_id`` and further roots (typically sibling git worktrees) are attached to the running
process via ``workspace/didChangeWorkspaceFolders`` — a **broken-set** of pairs that failed
to spawn/initialize (retried only after ``lsp.broken_retry_seconds``; never, by default), and a **delta baseline**
per file (``snapshot_baseline()`` runs BEFORE a write; the next ``get_diagnostics_sync()``
returns only diagnostics not in it).  Off unless config enables it.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.lsp import eventlog
from agent.lsp.client import DIAGNOSTICS_DOCUMENT_WAIT, LSPClient, _diagnostic_key as _diag_key
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, custom_servers, find_server_for_file, language_id_for
from agent.lsp.workspace import clear_cache, resolve_workspace_for_file

logger = logging.getLogger("agent.lsp.manager")

DEFAULT_IDLE_TIMEOUT = 600  # seconds; servers idle for >10min get reaped
_DELTA_BASELINE_CAP = 256  # per-file pre-write snapshots; paths never written again would otherwise live forever (#62950)
MIN_IDLE_TIMEOUT = 30  # floor for config values; must exceed any per-op wait budget
DEFAULT_MAX_HEAP_MB = 2048  # V8 old-space cap for Node-based servers; 0 = inherit parent env (#116446)
DEFAULT_CGROUP_MEMORY_MB = 4096  # memory.max (MB) for the dedicated LSP cgroup; 0 = separate cgroup, no cap

_Key = Tuple[str, str]
_Diags = List[Dict[str, Any]]


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def node_options_with_heap(base: Optional[str], max_heap_mb: int) -> str:
    """``NODE_OPTIONS`` for a spawned server: drop any inherited ``--max-old-space-size``
    (the gateway's own heap flag must not leak into server children, #116446), keep every
    other flag, then append the configured cap. Both token forms (``--flag=N`` and
    ``--flag N``) are stripped."""
    tokens: List[str] = []
    skip = False
    for tok in (base or "").split():
        if skip:
            skip = False
            continue
        if tok == "--max-old-space-size":
            skip = True  # separate-value form: the next token is the size
            continue
        if tok.startswith("--max-old-space-size="):
            continue
        tokens.append(tok)
    tokens.append(f"--max-old-space-size={max_heap_mb}")
    return " ".join(tokens)


def _parse_exclude_roots(value: Any) -> Optional[List[str]]:
    """Normalise ``lsp.exclude_roots``; ``None`` (fail closed) with a WARNING for a non-list value."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        eventlog.event_log.warning(
            "lsp.exclude_roots must be a list of glob patterns, e.g. ['~/big-monorepo', '/srv/*/vendor'] "
            "(got %s); LSP is skipped for every workspace until the key is fixed",
            type(value).__name__,
        )
        return None
    return [os.path.expanduser(p) for p in value if p]


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
        extra_servers: Optional[List[ServerDef]] = None,
        broken_retry_seconds: float = 0.0,
        warmup_timeout: float = 0.0,
        exclude_roots: Any = None,
        max_heap_mb: int = DEFAULT_MAX_HEAP_MB,
        cgroup_isolate: bool = True,
        cgroup_memory_mb: int = DEFAULT_CGROUP_MEMORY_MB,
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
        self._extra_servers: List[ServerDef] = list(extra_servers or [])
        self._broken_retry = max(0.0, broken_retry_seconds)
        self._warmup_timeout = max(0.0, warmup_timeout)
        # ``None`` = misconfigured → fail closed (every root excluded) until the user fixes the key: a
        # bare string here means "exclude that workspace", and excluding nothing would re-pay the stall it
        # was meant to avoid.
        self._exclude_roots: Optional[List[str]] = _parse_exclude_roots(exclude_roots)
        # V8 heap cap for spawned Node servers; 0 = leave the inherited env untouched.
        self._max_heap_mb = max(0, _int_or(max_heap_mb, DEFAULT_MAX_HEAP_MB))
        # cgroup v2 isolation: best-effort split so an LSP memory spike can't OOM
        # the gateway; falls back silently (one warning) where unavailable.
        self._cgroup_isolate = bool(cgroup_isolate)
        self._cgroup_memory_mb = max(0, _int_or(cgroup_memory_mb, DEFAULT_CGROUP_MEMORY_MB))

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()

        # Per-(server_id, workspace_root) state
        self._clients: Dict[_Key, LSPClient] = {}
        # (server_id, root) → monotonic deadline after which the pair may be retried (inf = lifetime).
        self._broken: Dict[_Key, float] = {}
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
            extra_servers=custom_servers(servers),
            broken_retry_seconds=_float_or(lsp_cfg.get("broken_retry_seconds"), 0.0),
            warmup_timeout=_float_or(lsp_cfg.get("warmup_timeout"), 0.0),
            exclude_roots=lsp_cfg.get("exclude_roots"),
            max_heap_mb=_int_or(lsp_cfg.get("max_heap_mb"), DEFAULT_MAX_HEAP_MB),
            cgroup_isolate=bool(lsp_cfg.get("cgroup_isolate", True)),
            cgroup_memory_mb=_int_or(lsp_cfg.get("cgroup_memory_mb"), DEFAULT_CGROUP_MEMORY_MB),
        )

    def _server_for(self, file_path: str) -> Optional[ServerDef]:
        """Config-declared servers first (they may claim an extension ahead of a built-in), then the registry."""
        extra = find_server_for_file(file_path, self._extra_servers) if self._extra_servers else None
        return extra or find_server_for_file(file_path)

    def handles_extension(self, ext: str) -> bool:
        """True iff a config-declared or built-in server claims ``ext`` (pre-write capture decision)."""
        return any(ext.lower() in s.extensions for s in (*self._extra_servers, *SERVERS))

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
        srv = self._server_for(file_path) if self._enabled else None
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        key = self._broken_key(srv, file_path)
        if key is None:
            return False
        if self._root_excluded(key[1]):
            eventlog.log_root_excluded(srv.server_id, key[1], file_path, invalid=self._exclude_roots is None)
            return False
        if self._is_broken(key):
            eventlog.log_skipped_broken(srv.server_id, key[1], file_path, retry_in=self._broken_retry_in(key))
            return False
        return True

    def _root_excluded(self, root: str) -> bool:
        """True iff ``root`` matches an ``lsp.exclude_roots`` glob (or the key is misconfigured)."""
        if self._exclude_roots is None:
            return True
        return any(fnmatch.fnmatchcase(root, pat) or fnmatch.fnmatchcase(root, pat.rstrip(os.sep) + os.sep + "*")
                   for pat in self._exclude_roots)

    def _is_broken(self, key: _Key) -> bool:
        """Broken-set membership; an entry whose retry deadline passed is dropped so the pair gets one more try."""
        deadline = self._broken.get(key)
        if deadline is None:
            return False
        if time.monotonic() >= deadline:
            self._broken.pop(key, None)
            return False
        return True

    def _broken_retry_in(self, key: _Key) -> Optional[float]:
        deadline = self._broken.get(key, math.inf)
        return None if math.isinf(deadline) else max(0.0, deadline - time.monotonic())

    def _mark_broken(self, key: _Key) -> None:
        self._broken[key] = time.monotonic() + self._broken_retry if self._broken_retry > 0 else math.inf

    def _wait_budget(self, file_path: str) -> float:
        """``wait_timeout`` for a root with a running client; ``max(wait_timeout, warmup_timeout)`` for a
        cold root, whose first request also pays the spawn/initialize and the server's initial program build."""
        if self._warmup_timeout <= self._wait_timeout:
            return self._wait_timeout
        srv = self._server_for(file_path)
        key = self._broken_key(srv, file_path) if srv is not None else None
        if key is None:
            return self._wait_timeout
        with self._state_lock:
            client = self._clients.get(_client_key(srv, key[1]))
        return self._wait_timeout if client is not None and client.is_running else self._warmup_timeout

    def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics for ``file_path`` as the delta baseline (call BEFORE a write).
        Best-effort: failures are swallowed so a flaky server can't break a write, but they mark the pair broken."""
        if not self.enabled_for(file_path):
            return
        try:
            # Outer join budget must exceed the inner wait or a slow-but-alive server gets falsely
            # marked broken; it is a ceiling only — the inner wait returns as soon as it completes.
            budget = self._wait_budget(file_path)
            t = max(DIAGNOSTICS_DOCUMENT_WAIT + 3.0, budget + 3.0)
            diags = self._loop.run(self._snapshot_async(file_path, budget), timeout=t)
        except Exception as e:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            diags = []
        self._set_delta_baseline(os.path.abspath(file_path), diags or [])

    def _set_delta_baseline(self, abs_path: str, diags: _Diags) -> None:
        """Store a baseline, refreshing recency (pop + reinsert) so eviction tracks write order.
        Callers run on arbitrary threads; the multi-step mutation needs the lock (callers don't hold it)."""
        with self._state_lock:
            self._delta_baseline.pop(abs_path, None)
            self._delta_baseline[abs_path] = diags
            while len(self._delta_baseline) > _DELTA_BASELINE_CAP:
                del self._delta_baseline[next(iter(self._delta_baseline))]

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
        server_id = self._server_for(file_path).server_id  # enabled_for guarantees a match
        try:
            budget = self._wait_budget(file_path)
            t = timeout if timeout is not None else budget + 2.0
            diags = self._loop.run(self._open_and_wait_async(file_path, budget=budget), timeout=t)
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
            self._set_delta_baseline(abs_path, fresh)
        return diags

    def _mark_broken_for_file(self, file_path: str, exc: BaseException) -> None:
        """Mark the file's ``(server_id, root)`` pair broken after an outer timeout/error.
        The outer ``_loop.run`` timeout cancels the in-flight spawn before ``_get_or_spawn`` could record
        the failure; without this every later write would re-pay the full timeout.  Also kills any
        half-initialized client and logs the failure once."""
        srv = self._server_for(file_path)
        key = self._broken_key(srv, file_path) if srv is not None else None
        if key is None:
            return
        already_broken = self._is_broken(key)
        self._mark_broken(key)
        ckey = _client_key(srv, key[1])
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
                 "workspace_folders": list(c.workspace_folders), "state": c.state, "running": c.is_running,
                 "cgroup_isolated": c.cgroup_isolated}
                for c in self._clients.values()
            ]
            broken = [key for key, deadline in self._broken.items() if time.monotonic() < deadline]
        return {
            "enabled": self._enabled, "wait_mode": self._wait_mode, "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy, "clients": clients, "broken": broken,
            "disabled_servers": sorted(self._disabled_servers),
            "broken_retry_seconds": self._broken_retry, "warmup_timeout": self._warmup_timeout,
            "exclude_roots": list(self._exclude_roots) if self._exclude_roots is not None else "INVALID",
            "max_heap_mb": self._max_heap_mb,
            "cgroup_isolate": self._cgroup_isolate,
            "cgroup_memory_mb": self._cgroup_memory_mb,
        }

    # ---- async internals ----

    async def _snapshot_async(self, file_path: str, budget: Optional[float] = None) -> _Diags:
        # No fresh data for the pre-edit content → empty baseline.  Safe: the delta
        # filter then removes less, never more.  Never seed from stale stores.
        return await self._open_and_wait_async(file_path, snapshot=True, budget=budget) or []

    async def _open_and_wait_async(
        self, file_path: str, *, snapshot: bool = False, budget: Optional[float] = None,
    ) -> Optional[_Diags]:
        """Open + wait for FRESH diagnostics: ``[]`` = checked clean, ``None`` = no verdict in budget.

        Callers must not substitute stale data for either.  ``snapshot`` mode (pre-write baseline)
        skips didSave; both modes wait at most ``budget`` (``lsp.wait_timeout`` unless the caller
        computed a cold-root warm-up budget via :meth:`_wait_budget`).
        """
        client = await self._get_or_spawn(file_path)
        if client is None:
            return None
        try:
            srv = self._server_for(file_path)
            version = await client.open_file(file_path, language_id=language_id_for(file_path, srv))
            if not snapshot:
                await client.save_file(file_path)
            fresh = await client.wait_for_diagnostics(
                file_path, version, mode=self._wait_mode,
                timeout=budget if budget is not None else self._wait_timeout,
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
        srv = self._server_for(file_path)
        if not (ws and gated and srv):
            return []
        # Same key _get_or_spawn() stored under: single-root servers live under their
        # resolved project root (a nested package.json), not the enclosing workspace.
        root = srv.resolve_root(file_path, ws)
        if root is None:
            return []
        with self._state_lock:
            client = self._clients.get(_client_key(srv, root))
        return list(client.diagnostics_for(file_path, fresh_only=True)) if client else []

    async def _get_or_spawn(self, file_path: str) -> Optional[LSPClient]:
        srv = self._server_for(file_path)
        if srv is None:
            return None
        if srv.server_id in self._disabled_servers:
            eventlog.log_disabled(srv.server_id, file_path, "disabled in config")
            return None
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            eventlog.log_no_project_root(srv.server_id, file_path)
            return None
        root = srv.resolve_root(file_path, ws_root)
        if root is None:
            eventlog.log_disabled(srv.server_id, file_path, "exclude marker hit (server gated off)")
            return None
        if self._is_broken((srv.server_id, root)):
            return None
        key = _client_key(srv, root)
        with self._state_lock:
            client = self._clients.get(key)
            if client is not None and client.is_running:
                self._last_used[key] = time.time()
            else:
                client = None
                spawning = self._spawning.get(key)
                owner = spawning is None
                if owner:
                    spawning = self._spawning[key] = asyncio.get_running_loop().create_future()
        if client is not None:
            eventlog.log_active(srv.server_id, root)
            return await self._attach_root(srv, client, root)
        if not owner:
            try:
                client = await spawning
            except Exception:  # noqa: BLE001
                return None
            return await self._attach_root(srv, client, root) if client is not None else None
        try:
            client = await self._spawn_client(srv, root)
            if client is None:
                self._mark_broken((srv.server_id, root))
            else:
                with self._state_lock:
                    self._clients[key] = client
                    self._last_used[key] = time.time()
                eventlog.log_active(srv.server_id, root)
            spawning.set_result(client)
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
        if self._max_heap_mb > 0:
            # Single choke point for every spawn: client merges env over os.environ, so this
            # NODE_OPTIONS wins over the inherited one. Non-Node servers ignore the variable.
            base = spec.env.get("NODE_OPTIONS", os.environ.get("NODE_OPTIONS", ""))
            spec.env = {**spec.env, "NODE_OPTIONS": node_options_with_heap(base, self._max_heap_mb)}
        client = LSPClient(
            server_id=srv.server_id, workspace_root=spec.workspace_root, command=spec.command, env=spec.env,
            cwd=spec.cwd, initialization_options=spec.initialization_options,
            seed_diagnostics_on_first_push=spec.seed_diagnostics_on_first_push or srv.seed_first_push,
            isolate_cgroup=self._cgroup_isolate, cgroup_memory_mb=self._cgroup_memory_mb,
        )
        try:
            await client.start()
        except Exception as e:  # noqa: BLE001
            eventlog.log_spawn_failed(srv.server_id, root, e)
            return None
        return client

    def _touch(self, client: LSPClient) -> None:
        """Refresh last-used; guarded on membership so a client reaped mid-operation can't resurrect its entry."""
        with self._state_lock:
            for key, c in self._clients.items():
                if c is client:
                    self._last_used[key] = time.time()

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
        # Externally deleted project roots (rm -rf, a worktree removed by another process) never go
        # idle from the server's point of view — tsserver keeps its multi-GiB heap for a tree that is gone.
        await self._detach_roots(lambda folder: not os.path.isdir(folder), reason="workspace root deleted")

    def release_workspace(self, workspace_root: str) -> int:
        """Shut down the clients serving ``workspace_root`` or any root beneath it; returns how many.

        Called by the worktree cleanup paths BEFORE ``git worktree remove`` so a gateway that outlives the
        session does not keep the language server (and its stdio pipes) alive for a tree that no longer
        exists.  Multi-root servers only drop the folder.  Idempotent; best-effort — never blocks removal.
        """
        if not self._enabled:
            return 0
        root = os.path.abspath(workspace_root)
        try:
            return self._loop.run(self._release_async(root), timeout=15.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP release of %s failed: %s", root, e)
            return 0

    async def _release_async(self, root: str) -> int:
        def _under(path: str) -> bool:
            return path == root or path.startswith(root + os.sep)

        # A spawn in flight would insert its client AFTER we detach; let it land first (same loop, so
        # once the future resolves the client is in ``_clients`` and the pass below sees it).
        with self._state_lock:
            pending = [fut for fut in self._spawning.values() if not fut.done()]
            self._broken = {key: deadline for key, deadline in self._broken.items() if not _under(key[1])}
            for path in [p for p in self._delta_baseline if _under(p)]:
                del self._delta_baseline[path]
        if pending:
            await asyncio.wait(pending, timeout=10.0)
        released = await self._detach_roots(_under, reason="workspace released")
        clear_cache()
        return released

    async def _detach_roots(self, is_gone: Callable[[str], bool], *, reason: str) -> int:
        """Shared teardown primitive for :meth:`release_workspace` and the reaper.

        Clients whose every workspace folder ``is_gone`` are detached under ``_state_lock`` and shut down;
        multi-root clients that still serve other folders only drop the gone ones.  Returns the number of
        clients shut down.
        """
        with self._state_lock:
            dead_keys = [key for key, c in self._clients.items() if all(map(is_gone, c.workspace_folders))]
            clients = [self._clients.pop(key) for key in dead_keys]
            for key in dead_keys:
                self._last_used.pop(key, None)
            trims = [(c, [f for f in c.workspace_folders if is_gone(f)]) for c in self._clients.values()]
        for client, folders in trims:
            for folder in folders:
                try:
                    await client.remove_workspace_folder(folder)
                except Exception as e:  # noqa: BLE001
                    logger.debug("LSP folder removal for %s failed: %s", folder, e)
        if clients:
            eventlog.log_released([(c.server_id, c.workspace_root) for c in clients], reason)
            await asyncio.gather(*(client.shutdown() for client in clients), return_exceptions=True)
        return len(clients)

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
        await asyncio.gather(*(c.shutdown() for c in clients), return_exceptions=True)


__all__ = ["LSPService"]
