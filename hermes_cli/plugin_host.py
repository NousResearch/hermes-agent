"""Parent side of the plugin host: one process per profile that runs its third-party plugins.

With ``plugins.isolation: host`` the loader hands a general Python plugin to :class:`PluginHost`
instead of importing it. The host process imports it and runs ``register(ctx)``; every ``ctx`` call
arrives here and is replayed on the plugin's real :class:`~hermes_cli.plugins.PluginContext`, with
the plugin's callables and provider objects swapped for proxies that call back into the host. From
then on the rest of Hermes sees ordinary registrations — the registry, hook dispatch, the ledger,
unload and ``hermes plugins list`` do not know the plugin lives elsewhere, and neither does the
plugin: its code is unchanged and never learns which process or tenant it serves.

Requests the host makes while Hermes is waiting on it (a tool handler calling ``ctx.dispatch_tool``)
run in the caller's context — same profile home, secret scope and session — via the wire's
``origin`` field; spontaneous ones (a ``spawn_task`` loop) run bound to this profile's home.

When the host dies every proxy raises :class:`PluginHostUnavailable`: tool calls return an error,
hooks fail like any raising callback, and Hermes itself keeps running. A fresh host is then started
and its plugins reloaded (bounded restarts; a plugin that killed the host while loading stays out).
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import inspect
import functools
import importlib
import itertools
import logging
import os
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hermes_cli.plugin_host_wire import (
    Channel, PluginHostUnavailable, PluginHostUnsupported, decode, encode, signature_from,
)

logger = logging.getLogger("hermes_cli.plugins")

_HOST_MODULE = "hermes_cli.plugin_host_child"
_START_TIMEOUT_SECS = 30.0
_SHUTDOWN_GRACE_SECS = 3.0
# A host that dies is restarted (its plugins reloaded) at most this many times per window; a plugin
# that kills the host while loading is not reloaded, so one bad plugin cannot crash-loop the rest.
_RESTART_BUDGET = 3
_RESTART_WINDOW_SECS = 600.0


class PluginHost:
    """One host process for one :class:`~hermes_cli.plugins.PluginManager` (one profile home)."""

    def __init__(self, manager: Any):
        self._manager = manager
        self._home = Path(manager.home_path)
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._channel: Optional[Channel] = None
        self._contexts: Dict[str, Any] = {}
        self._handles: Dict[int, Any] = {}
        self._handle_ids = itertools.count(1)
        self._base_context = self._build_base_context()
        self._loading: Optional[str] = None
        self._stopping = False
        self._deaths: list = []
        # Bumped per host process: a proxy made for an earlier process must not address this one
        # (object ids restart at 1, so a stale id would silently land on a different object).
        self._generation = 0
        self._releases: list = []
        self.info: Dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------------------------------------
    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @property
    def alive(self) -> bool:
        return (self._channel is not None and self._channel.closed_reason is None
                and self._proc is not None and self._proc.poll() is None)

    def _build_base_context(self) -> contextvars.Context:
        # A fresh context, not a copy of whichever caller created the host: requests with no
        # caller (a ``spawn_task`` loop) must not inherit that caller's session or approvals.
        context = contextvars.Context()

        def bind() -> None:
            from hermes_constants import set_hermes_home_override
            set_hermes_home_override(self._home)
            from agent.secret_scope import build_profile_secret_scope, is_multiplex_active, set_secret_scope
            if is_multiplex_active():
                set_secret_scope(build_profile_secret_scope(self._home), profile_home=str(self._home))

        context.run(bind)
        return context

    def _argv(self) -> list:
        from hermes_cli.plugin_isolation import host_launcher
        return [*host_launcher(), sys.executable, "-m", _HOST_MODULE]

    def _env(self) -> Dict[str, str]:
        from tools.environments.local import served_profile_child_env
        env = self._base_context.run(served_profile_child_env, target_home=self._home, inherit_credentials=True)
        repo_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(p for p in (repo_root, env.get("PYTHONPATH", "")) if p)
        env.setdefault("HERMES_PLUGIN_HOST_LOG_LEVEL", "WARNING")
        from hermes_cli.plugin_isolation import HOST_PROCESS_ENV
        env[HOST_PROCESS_ENV] = "1"
        return env

    def ensure_started(self) -> Channel:
        with self._lock:
            if self.alive:
                return self._channel  # type: ignore[return-value]
            self._stopping = False
            argv = self._argv()
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=self._env(), cwd=str(self._home),
                                    bufsize=0)
            threading.Thread(target=_pump_stderr, args=(proc, self._home.name), daemon=True,
                             name=f"plugin-host-{proc.pid}-stderr").start()
            self._generation += 1
            channel = Channel(proc.stdout, proc.stdin, self._handle,  # type: ignore[arg-type]
                              name=f"plugin-host[{proc.pid}]",
                              on_close=functools.partial(self._on_channel_close, self._generation),
                              context_for_origin=self._context_for_origin).start()
            self._proc, self._channel = proc, channel
            self._releases = []
        try:
            self.info = channel.call("hello", {}, timeout=_START_TIMEOUT_SECS)
        except Exception as exc:
            self.shutdown()
            raise PluginHostUnavailable(f"plugin host failed to start ({' '.join(argv)}): {exc}") from exc
        logger.info("Plugin host started for %s (pid %s)", self._home, proc.pid)
        return channel

    def _exit_reason(self) -> str:
        proc = self._proc
        code = proc.poll() if proc is not None else None
        if proc is not None and code is None:
            try:
                code = proc.wait(timeout=0.5)  # the pipe closes a moment before the exit status lands
            except subprocess.TimeoutExpired:
                pass
        return (f"the plugin host for this profile exited (code {code}); Hermes restarts it and reloads "
                f"its plugins automatically" if code is not None else "the plugin host is not running")

    def _on_channel_close(self, generation: int, reason: str) -> None:
        if self._stopping or generation != self._generation:
            return  # a deliberate shutdown, or a late close from a host already replaced
        culprit = self._loading
        keys = [k for k in self._contexts.copy() if k != culprit and k in self._manager._plugins]
        logger.warning("Plugin host for %s exited (%s)%s", self._home, self._exit_reason(),
                       f" while loading plugin '{culprit}'" if culprit else "")
        now = time.monotonic()
        self._deaths = [t for t in self._deaths if now - t < _RESTART_WINDOW_SECS] + [now]
        if keys:
            threading.Thread(target=self._restart, args=(keys,), daemon=True,
                             name="plugin-host-restart").start()

    def _restart_refused(self, what: str) -> bool:
        if len(self._deaths) <= _RESTART_BUDGET:
            return False
        logger.error("Plugin host for %s died %d times in %.0fs; not restarting it again (%s)",
                     self._home, len(self._deaths), _RESTART_WINDOW_SECS, what)
        return True

    def _restart(self, keys: list) -> None:
        if self._restart_refused("plugins: " + ", ".join(keys)):
            return
        time.sleep(0.5 * len(self._deaths))
        manager = self._manager
        for key in keys:
            loaded = manager._plugins.get(key)
            if loaded is None or not loaded.enabled:
                continue
            manager.unload(key)
            manager._load_plugin(loaded.manifest)

    def shutdown(self) -> None:
        self._stopping = True
        with self._lock:
            proc, channel = self._proc, self._channel
            self._proc = self._channel = None
        if channel is not None and channel.closed_reason is None:
            try:
                channel.call("shutdown", {}, timeout=_SHUTDOWN_GRACE_SECS)
            except Exception:
                pass
            channel.close("shutdown")
        if proc is not None:
            try:
                proc.wait(timeout=_SHUTDOWN_GRACE_SECS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_SHUTDOWN_GRACE_SECS)

    # -- loading ------------------------------------------------------------------------------------
    def load(self, manifest: Any, ctx: Any, *, module_name: Optional[str], entrypoint: bool) -> str:
        """Import ``manifest``'s plugin in the host and replay its registrations onto ``ctx``."""
        from hermes_cli.plugins import PluginContext, manifest_key
        channel = self.ensure_started()
        plugin_key = manifest_key(manifest)
        self._contexts[plugin_key] = ctx
        params = {
            "plugin_key": plugin_key, "plugin_id": ctx.plugin_id, "name": manifest.name,
            "path": manifest.path, "module_name": module_name, "entrypoint": entrypoint,
            "profile_name": self._base_context.copy().run(lambda: ctx.profile_name),
            "manifest": {k: getattr(manifest, k, None) for k in (
                "name", "version", "description", "author", "source", "path", "key", "kind",
                "skill_namespace")},
            "ctx_methods": sorted(n for n in dir(PluginContext) if not n.startswith("_")),
        }
        self._loading = plugin_key
        try:
            result = channel.call("load", params)
        except BaseException:
            self._contexts.pop(plugin_key, None)
            raise
        finally:
            self._loading = None
        ctx.on_unload(functools.partial(self._unload, plugin_key))
        return str((result or {}).get("module") or module_name or "")

    def load_instance(self, plugin_dir: Path, *, module_name: str, base_ref: str, capture: str,
                      ctx: Any = None, before_reload: Optional[Callable[[], None]] = None) -> Any:
        """Load a category plugin (memory provider, context engine, cron scheduler) in the host and
        return a proxy that is an instance of ``base_ref``. ``ctx`` receives its other registrations.

        The proxy outlives a host crash: its next use loads the plugin into the new host (after
        ``before_reload``, which drops the registrations the old load forwarded) and carries on."""
        load = functools.partial(self._load_instance_ref, Path(plugin_dir), module_name=module_name,
                                 base_ref=base_ref, capture=capture, ctx=ctx)
        ref = load()
        if ref is None:
            return None

        def reload() -> Dict[str, Any]:
            if self._restart_refused(f"{capture} plugin {Path(plugin_dir).name}"):
                raise PluginHostUnavailable(self._exit_reason())
            if before_reload is not None:
                before_reload()
            fresh = load()
            if fresh is None:
                raise PluginHostUnavailable(f"plugin {Path(plugin_dir).name!r} no longer loads in the plugin host")
            return fresh

        return self._object_proxy(ref, _import_ref(base_ref), reload=reload)

    def _load_instance_ref(self, plugin_dir: Path, *, module_name: str, base_ref: str, capture: str,
                           ctx: Any) -> Optional[Dict[str, Any]]:
        channel = self.ensure_started()
        plugin_key = f"{capture}:{Path(plugin_dir).name}"
        if ctx is not None:
            self._contexts[plugin_key] = ctx
        from hermes_cli.plugins import PluginContext
        params = {
            "plugin_key": plugin_key, "plugin_id": getattr(ctx, "plugin_id", Path(plugin_dir).name),
            "name": Path(plugin_dir).name, "path": str(plugin_dir), "module_name": module_name,
            "base": base_ref, "capture": capture, "forward": ctx is not None,
            "manifest": {"name": Path(plugin_dir).name, "path": str(plugin_dir), "source": "user"},
            "ctx_methods": sorted(n for n in dir(PluginContext) if not n.startswith("_")),
        }
        self._loading = plugin_key
        try:
            ref = channel.call("load_instance", params)
        finally:
            self._loading = None
        if ref is not None:
            ref["generation"] = self._generation
        return ref

    def config_schema(self, path: Path) -> Any:
        """Run a user memory provider's ``config_schema.py`` here and return its ``CONFIG_SCHEMA``."""
        self.ensure_started()
        return self._call("config_schema", {"path": str(path)})

    def profile_call(self, plugin_dir: str, module_name: str, profile: str, attr: str,
                     args: tuple, kwargs: dict) -> Any:
        """Run a model-provider profile's overridden method / callable field in the host."""
        self.ensure_started()
        return self._call("profile_call", {"path": plugin_dir, "module_name": module_name, "profile": profile,
                                           "attr": attr, "args": encode(list(args)), "kwargs": encode(kwargs)})

    def asgi_request(self, plugin_name: str, dashboard_dir: str, api_file: str, method: str, path: str,
                     query: str, headers: list, body: bytes) -> Dict[str, Any]:
        """One dashboard ``/api/plugins/<name>/`` request, served by the plugin's router in the host."""
        self.ensure_started()
        return self._call("asgi", {"plugin": plugin_name, "dashboard_dir": dashboard_dir, "api_file": api_file,
                                   "method": method, "path": path, "query": query,
                                   "headers": [list(h) for h in headers], "body": encode(body)})

    def _unload(self, plugin_key: str) -> None:
        self._contexts.pop(plugin_key, None)
        channel = self._channel
        if channel is None or channel.closed_reason is not None:
            return
        errors = (channel.call("unload", {"plugin_key": plugin_key}) or {}).get("errors") or []
        for error in errors:
            logger.warning("Plugin '%s' on_unload callback failed in the plugin host: %s", plugin_key, error)

    # -- parent -> host -----------------------------------------------------------------------------
    def _call(self, method: str, params: Dict[str, Any], *, generation: Optional[int] = None) -> Any:
        channel = self._channel
        try:
            if channel is None:
                raise PluginHostUnavailable("not started")
            if generation is not None and generation != self._generation:
                raise PluginHostUnavailable("this object belonged to a plugin host process that has exited")
            if self._releases:
                released, self._releases = self._releases, []
                channel.call("release", {"refs": released})
            return decode(channel.call(method, params), self._resolve_ref)
        except PluginHostUnavailable as exc:
            raise PluginHostUnavailable(self._exit_reason()) from exc

    def _release(self, generation: int, ref: int) -> None:
        """Finalizer of an object proxy (any thread, GC time): queue only; sent with the next call."""
        if generation == self._generation:
            self._releases.append(ref)

    def invoke(self, ref: int, args: tuple, kwargs: dict, *, generation: Optional[int] = None) -> Any:
        return self._call("invoke", {"ref": ref, "args": encode(list(args)), "kwargs": encode(kwargs)},
                          generation=generation)

    def obj_invoke(self, slot: "_ObjectSlot", method: str, args: tuple, kwargs: dict) -> Any:
        ref, generation = slot.live()
        return self._call("obj_invoke", {"ref": ref, "method": method, "args": encode(list(args)),
                                         "kwargs": encode(kwargs)}, generation=generation)

    def obj_getattr(self, slot: "_ObjectSlot", name: str) -> Any:
        ref, generation = slot.live()
        value = self._call("obj_getattr", {"ref": ref, "name": name}, generation=generation)
        if isinstance(value, dict) and set(value) == {"__missing__"}:
            raise AttributeError(name)
        return value

    def obj_setattr(self, slot: "_ObjectSlot", name: str, value: Any) -> None:
        ref, generation = slot.live()
        self._call("obj_setattr", {"ref": ref, "name": name, "value": encode(value)}, generation=generation)

    # -- host -> parent -----------------------------------------------------------------------------
    def _context_for_origin(self, origin: Optional[int]) -> contextvars.Context:
        channel = self._channel
        caller = channel.context_of(origin) if channel is not None else None
        return (caller or self._base_context).copy()

    def _handle(self, method: str, params: Dict[str, Any], _origin: Optional[int]) -> Any:
        if method == "ctx":
            return self._serve_ctx(params)
        if method == "facade":
            return self._serve_facade(params)
        if method == "dispose":
            handle = self._handles.pop(int(params["handle"]), None)
            if handle is not None:
                handle.dispose()
            return None
        raise ValueError(f"unknown plugin host request {method!r}")

    def _plugin_ctx(self, params: Dict[str, Any]) -> Any:
        ctx = self._contexts.get(str(params.get("plugin")))
        if ctx is None:
            raise LookupError(f"plugin {params.get('plugin')!r} is not loaded in this host")
        return ctx

    def _serve_ctx(self, params: Dict[str, Any]) -> Any:
        from hermes_cli.plugin_isolation import (
            HOST_OBJECT_BASES, HOST_SKIPPED_CTX_METHODS, HOST_UNSUPPORTED_CTX_METHODS,
        )
        ctx = self._plugin_ctx(params)
        method = str(params.get("method") or "")
        if (method.startswith("_") or method in HOST_UNSUPPORTED_CTX_METHODS
                or method in HOST_SKIPPED_CTX_METHODS or method in {"on_unload", "spawn_task"}):
            raise PluginHostUnsupported(f"ctx.{method}() cannot be called across the plugin host")
        base = _import_ref(HOST_OBJECT_BASES[method]) if method in HOST_OBJECT_BASES else None
        resolve = functools.partial(self._resolve_ref, base=base)
        args = decode(params.get("args") or [], resolve)
        kwargs = decode(params.get("kwargs") or {}, resolve)
        result = getattr(ctx, method)(*args, **kwargs)
        return self._encode_for_host(result)

    def _serve_facade(self, params: Dict[str, Any]) -> Any:
        from hermes_cli.plugin_isolation import HOST_REMOTE_FACADES
        ctx = self._plugin_ctx(params)
        facade, method = str(params.get("facade") or ""), str(params.get("method") or "")
        if facade not in HOST_REMOTE_FACADES or method.startswith("_"):
            raise PluginHostUnsupported(f"ctx.{facade}.{method} is not available in the plugin host")
        target = getattr(getattr(ctx, facade), method)
        if not callable(target):
            return self._encode_for_host(target)
        if params.get("probe"):
            return {"__method__": True, "async": inspect.iscoroutinefunction(target)}
        args = decode(params.get("args") or [], self._resolve_ref)
        kwargs = decode(params.get("kwargs") or {}, self._resolve_ref)
        result = target(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = _run_on_gateway_loop(result)
        return self._encode_for_host(result)

    def _encode_for_host(self, value: Any) -> Any:
        from hermes_cli.plugins_ledger import PluginRegistration

        def refs(obj: Any) -> Optional[dict]:
            if isinstance(obj, PluginRegistration):
                handle_id = next(self._handle_ids)
                self._handles[handle_id] = obj
                return {"__handle__": handle_id, "kind": obj.kind, "key": obj.key}
            return None

        return encode(value, refs)

    # -- proxies ------------------------------------------------------------------------------------
    def _resolve_ref(self, ref: Dict[str, Any], base: Optional[type] = None) -> Any:
        if "__callable__" in ref:
            return self._callable_proxy(ref)
        if "__object__" in ref:
            if base is None:
                raise PluginHostUnsupported(f"a {ref.get('type', 'plugin')} object cannot be passed here "
                                            "across the plugin host")
            return self._object_proxy(ref, base)
        raise PluginHostUnsupported("registration handles are owned by the plugin host")

    def _callable_proxy(self, ref: Dict[str, Any]) -> Callable[..., Any]:
        ref_id, generation = int(ref["__callable__"]), self._generation

        async def async_proxy(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(self.invoke, ref_id, args, kwargs, generation=generation)

        def sync_proxy(*args: Any, **kwargs: Any) -> Any:
            return self.invoke(ref_id, args, kwargs, generation=generation)

        proxy: Any = async_proxy if ref.get("async") else sync_proxy
        signature = signature_from(ref.get("sig"))
        if signature is not None:
            proxy.__signature__ = signature  # type: ignore[attr-defined]
        proxy.__name__ = str(ref.get("name") or "callback")
        proxy.__qualname__ = str(ref.get("qualname") or proxy.__name__)
        # Tool override policy and hook attribution key on the defining module.
        proxy.__module__ = str(ref.get("module") or __name__)
        proxy.__hermes_plugin_host_ref__ = ref_id  # type: ignore[attr-defined]
        return proxy

    def _object_proxy(self, ref: Dict[str, Any], base: type,
                      reload: Optional[Callable[[], Dict[str, Any]]] = None) -> Any:
        """An instance of ``base`` whose methods and public attributes live in the host. Attributes
        Hermes assigns go to the plugin's object when they can cross; live handles stay local."""
        host = self
        slot = _ObjectSlot(self, ref, reload)
        live = set(ref.get("live") or ())
        namespace: Dict[str, Any] = {}
        for name, meta in (ref.get("methods") or {}).items():
            namespace[name] = _method_proxy(host, slot, name, meta)

        def __getattribute__(self_: Any, name: str) -> Any:  # noqa: N807
            if name in live:
                return host.obj_getattr(slot, name)
            return object.__getattribute__(self_, name)

        def __getattr__(self_: Any, name: str) -> Any:  # noqa: N807 — set later inside the plugin
            if name.startswith("_"):
                raise AttributeError(name)
            try:
                return host.obj_getattr(slot, name)
            except PluginHostUnavailable as exc:  # ``hasattr``/``getattr(x, n, None)`` probes stay probes
                raise AttributeError(name) from exc

        def __setattr__(self_: Any, name: str, value: Any) -> None:  # noqa: N807
            if not name.startswith("_") and not _holds_live_handle(value):
                host.obj_setattr(slot, name, value)
                live.add(name)
                return
            live.discard(name)
            object.__setattr__(self_, name, value)

        namespace.update(__getattribute__=__getattribute__, __getattr__=__getattr__, __setattr__=__setattr__,
                         __module__=__name__, __hermes_plugin_host_ref__=slot,
                         __repr__=lambda self_: f"<plugin-host {ref.get('type')} #{slot.ref}>")
        cls = type(f"Hosted{ref.get('type') or base.__name__}", (base,), namespace)
        cls.__abstractmethods__ = frozenset()
        proxy = object.__new__(cls)
        slot.track(proxy)
        return proxy


class _ObjectSlot:
    """Which host object (id + host generation) a proxy addresses; re-resolved after a host restart
    when the proxy knows how to load its plugin again."""

    def __init__(self, host: PluginHost, ref: Dict[str, Any], reload: Optional[Callable[[], Dict[str, Any]]]):
        self._host, self._reload = host, reload
        self.ref, self.generation = int(ref["__object__"]), int(ref.get("generation") or host._generation)
        self._finalizer: Optional[weakref.finalize] = None
        self._proxy: Optional[weakref.ref] = None

    def track(self, proxy: Any) -> None:
        self._proxy = weakref.ref(proxy)
        self._finalizer = weakref.finalize(proxy, self._host._release, self.generation, self.ref)

    def live(self) -> tuple:
        if (self.generation != self._host._generation or not self._host.alive) and self._reload is not None:
            fresh = self._reload()
            if self._finalizer is not None:
                self._finalizer.detach()
            self.ref, self.generation = int(fresh["__object__"]), int(fresh["generation"])
            proxy = self._proxy() if self._proxy is not None else None
            if proxy is not None:
                self.track(proxy)
        return self.ref, self.generation


def _holds_live_handle(value: Any) -> bool:
    """True when ``value`` would reach the host only as an opaque placeholder (an agent, a client)."""
    found = False

    def refs(obj: Any) -> Optional[dict]:
        nonlocal found
        if callable(obj) or not (dataclasses.is_dataclass(obj) or hasattr(obj, "model_dump")):
            found = True
        return None

    encoded = encode(value, refs)
    return found or (isinstance(encoded, dict) and "__opaque__" in encoded)


def _run_on_gateway_loop(coro: Any) -> Any:
    """A facade coroutine (``platform_actions``) uses adapters bound to the gateway's event loop, so
    it runs there when a gateway is up; otherwise on a private loop. Always called off-loop (a
    channel thread), never from the loop itself."""
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    loop = getattr(runner, "_gateway_loop", None)
    if loop is not None and loop.is_running() and not loop.is_closed():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return asyncio.run(coro)


def _method_proxy(host: PluginHost, slot: _ObjectSlot, name: str, meta: Dict[str, Any]) -> Callable[..., Any]:
    async def async_method(self_: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(host.obj_invoke, slot, name, args, kwargs)

    def sync_method(self_: Any, *args: Any, **kwargs: Any) -> Any:
        return host.obj_invoke(slot, name, args, kwargs)

    method: Any = async_method if meta.get("async") else sync_method
    method.__name__ = method.__qualname__ = name
    signature = signature_from(meta.get("sig"))
    if signature is not None:  # callers introspect accepted kwargs; bound access drops ``self`` again
        self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        method.__signature__ = signature.replace(parameters=[self_param, *signature.parameters.values()])
    return method


def _import_ref(ref: str) -> type:
    module, attr = ref.split(":")
    return getattr(importlib.import_module(module), attr)


def _pump_stderr(proc: subprocess.Popen, label: str) -> None:
    """Plugin output and host logs go to the Hermes log, tagged with the host's profile."""
    stream = proc.stderr
    if stream is None:
        return
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        if line:
            logger.info("plugin host [%s]: %s", label, line)
