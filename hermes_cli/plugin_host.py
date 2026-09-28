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
        self._restarts: list = []
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
        context = contextvars.copy_context()

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
            channel = Channel(proc.stdout, proc.stdin, self._handle,  # type: ignore[arg-type]
                              name=f"plugin-host[{proc.pid}]",
                              on_close=self._on_channel_close,
                              context_for_origin=self._context_for_origin).start()
            self._proc, self._channel = proc, channel
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

    def _on_channel_close(self, reason: str) -> None:
        if self._stopping:
            return
        keys, culprit = [k for k in self._contexts if k != self._loading], self._loading
        logger.warning("Plugin host for %s exited (%s)%s", self._home, self._exit_reason(),
                       f" while loading plugin '{culprit}'" if culprit else "")
        if keys:
            threading.Thread(target=self._restart, args=(keys,), daemon=True,
                             name="plugin-host-restart").start()

    def _restart(self, keys: list) -> None:
        now = time.monotonic()
        self._restarts = [t for t in self._restarts if now - t < _RESTART_WINDOW_SECS]
        if len(self._restarts) >= _RESTART_BUDGET:
            logger.error("Plugin host for %s died %d times in %.0fs; not restarting it again (plugins: %s)",
                         self._home, len(self._restarts), _RESTART_WINDOW_SECS, ", ".join(keys))
            return
        self._restarts.append(now)
        time.sleep(0.5 * len(self._restarts))
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
                      ctx: Any = None) -> Any:
        """Load a category plugin (memory provider, context engine, cron scheduler) in the host and
        return a proxy that is an instance of ``base_ref``. ``ctx`` receives its other registrations."""
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
        if ref is None:
            return None
        return self._object_proxy(ref, _import_ref(base_ref))

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
    def _call(self, method: str, params: Dict[str, Any]) -> Any:
        channel = self._channel
        try:
            if channel is None:
                raise PluginHostUnavailable("not started")
            return decode(channel.call(method, params), self._resolve_ref)
        except PluginHostUnavailable as exc:
            raise PluginHostUnavailable(self._exit_reason()) from exc

    def invoke(self, ref: int, args: tuple, kwargs: dict) -> Any:
        return self._call("invoke", {"ref": ref, "args": encode(list(args)), "kwargs": encode(kwargs)})

    def obj_invoke(self, ref: int, method: str, args: tuple, kwargs: dict) -> Any:
        return self._call("obj_invoke", {"ref": ref, "method": method, "args": encode(list(args)),
                                         "kwargs": encode(kwargs)})

    def obj_getattr(self, ref: int, name: str) -> Any:
        return self._call("obj_getattr", {"ref": ref, "name": name})

    def obj_setattr(self, ref: int, name: str, value: Any) -> None:
        self._call("obj_setattr", {"ref": ref, "name": name, "value": encode(value)})

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
            return {"__method__": True}
        args = decode(params.get("args") or [], self._resolve_ref)
        kwargs = decode(params.get("kwargs") or {}, self._resolve_ref)
        result = target(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
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
        ref_id = int(ref["__callable__"])
        async def async_proxy(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(self.invoke, ref_id, args, kwargs)

        def sync_proxy(*args: Any, **kwargs: Any) -> Any:
            return self.invoke(ref_id, args, kwargs)

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

    def _object_proxy(self, ref: Dict[str, Any], base: type) -> Any:
        ref_id = int(ref["__object__"])
        host = self
        live = frozenset(ref.get("live") or ())
        namespace: Dict[str, Any] = {}
        for name, meta in (ref.get("methods") or {}).items():
            namespace[name] = _method_proxy(host, ref_id, name, meta)
        for name, value in (ref.get("static") or {}).items():
            decoded = decode(value, self._resolve_ref)
            namespace[name] = staticmethod(decoded) if callable(decoded) else decoded

        def __getattribute__(self_: Any, name: str) -> Any:  # noqa: N807
            if name in live:
                return host.obj_getattr(ref_id, name)
            return object.__getattribute__(self_, name)

        def __setattr__(self_: Any, name: str, value: Any) -> None:  # noqa: N807
            if name in live:
                host.obj_setattr(ref_id, name, value)
            else:
                object.__setattr__(self_, name, value)

        namespace.update(__getattribute__=__getattribute__, __setattr__=__setattr__,
                         __module__=__name__, __hermes_plugin_host_ref__=ref_id,
                         __repr__=lambda self_: f"<plugin-host {ref.get('type')} #{ref_id}>")
        cls = type(f"Hosted{ref.get('type') or base.__name__}", (base,), namespace)
        cls.__abstractmethods__ = frozenset()
        return object.__new__(cls)


def _method_proxy(host: PluginHost, ref_id: int, name: str, meta: Dict[str, Any]) -> Callable[..., Any]:
    async def async_method(self_: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(host.obj_invoke, ref_id, name, args, kwargs)

    def sync_method(self_: Any, *args: Any, **kwargs: Any) -> Any:
        return host.obj_invoke(ref_id, name, args, kwargs)

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
