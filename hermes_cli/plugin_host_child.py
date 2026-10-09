"""The plugin host process: imports third-party plugins and serves their callbacks to Hermes.

Started by :class:`hermes_cli.plugin_host.PluginHost` as ``python -m hermes_cli.plugin_host_child``
(optionally under ``plugins.host.launcher``, e.g. a sandbox runner) with the profile's environment.
Plugins get a :class:`RemotePluginContext` that looks like ``PluginContext``: registrations and
queries are forwarded to Hermes, callables and provider objects stay here and are invoked by
reference. Nothing in this module is imported by the Hermes process itself.

stdin/stdout carry the protocol, so the real file descriptors are duplicated for the channel and
fd 0/1 are pointed at /dev/null and stderr before any plugin code runs: a stray ``print()`` in a
plugin lands in the host log instead of corrupting the wire.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import importlib.metadata
import importlib.util
import inspect
import itertools
import logging
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_cli.plugin_host_wire import (
    Channel, Opaque, PROTOCOL_VERSION, PluginHostUnsupported, bind_serving_request, decode, describe_signature,
    encode, is_async_callable, serving_request,
)

logger = logging.getLogger("hermes_cli.plugin_host_child")

_ENTRY_POINTS_GROUP = "hermes_agent.plugins"


class RemoteRegistration:
    """Child-side twin of ``PluginRegistration``: ``dispose()`` releases the Hermes-side entry."""

    def __init__(self, runtime: "HostRuntime", handle_id: int, kind: str, key: str):
        self._runtime, self._id, self.kind, self.key = runtime, handle_id, kind, key
        self._disposed = False

    @property
    def active(self) -> bool:
        return not self._disposed

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._runtime.channel.call("dispose", {"handle": self._id})


class RemoteFacade:
    """``ctx.state`` / ``ctx.llm`` / ...: every method call runs on the Hermes-side facade."""

    def __init__(self, runtime: "HostRuntime", plugin_key: str, name: str):
        self._runtime, self._plugin_key, self._name = runtime, plugin_key, name
        self._methods: Dict[str, bool] = {}  # method name -> is a coroutine function in Hermes

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        if attr not in self._methods:
            value = self._runtime.facade_call(self._plugin_key, self._name, attr, _probe=True)
            if not (isinstance(value, dict) and value.get("__method__")):
                return value  # a plain attribute (``ctx.state.data_dir``): read fresh every time
            self._methods[attr] = bool(value.get("async"))
        call = functools.partial(self._runtime.facade_call, self._plugin_key, self._name, attr)
        if not self._methods[attr]:
            return call

        async def remote(*args: Any, **kwargs: Any) -> Any:  # ``await ctx.llm.acomplete(...)``
            return await asyncio.to_thread(call, *args, **kwargs)

        remote.__name__ = attr
        return remote


class RemotePluginContext:
    """What ``register(ctx)`` receives inside the host. Mirrors ``PluginContext``'s public surface as
    announced by Hermes at load, so ``hasattr(ctx, "register_x")`` answers the same as in-process."""

    def __init__(self, runtime: "HostRuntime", plugin_key: str, info: Dict[str, Any]):
        self._runtime, self._plugin_key = runtime, plugin_key
        self._methods = set(info.get("ctx_methods") or ())
        self._facades: Dict[str, RemoteFacade] = {}
        self.manifest = types.SimpleNamespace(**(info.get("manifest") or {}))
        self.plugin_id = info.get("plugin_id") or plugin_key
        self.profile_name = info.get("profile_name")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        from hermes_cli.plugin_isolation import (
            HOST_REMOTE_FACADES, HOST_SKIPPED_CTX_METHODS, HOST_UNSUPPORTED_CTX_METHODS,
        )
        if name in HOST_UNSUPPORTED_CTX_METHODS:
            def unsupported(*_args: Any, **_kwargs: Any) -> Any:
                raise PluginHostUnsupported(
                    f"ctx.{name}() cannot run in the plugin host: {HOST_UNSUPPORTED_CTX_METHODS[name]}")
            return unsupported
        if name in HOST_SKIPPED_CTX_METHODS:
            def skipped(*_args: Any, **_kwargs: Any) -> None:
                logger.warning("Plugin '%s': ctx.%s() is skipped in the plugin host (%s)",
                               self.manifest.name, name, HOST_SKIPPED_CTX_METHODS[name])
                return None
            return skipped
        if name in HOST_REMOTE_FACADES:
            return self._facades.setdefault(name, RemoteFacade(self._runtime, self._plugin_key, name))
        if name not in self._methods:
            raise AttributeError(f"'PluginContext' object has no attribute {name!r}")
        return functools.partial(self._runtime.ctx_call, self._plugin_key, name)

    def on_unload(self, callback: Callable[[], Any]) -> None:
        if not callable(callback):
            raise TypeError("on_unload() callback must be callable")
        self._runtime.unload_callbacks.setdefault(self._plugin_key, []).append(callback)

    def spawn_task(self, coro: Any, *, name: Optional[str] = None) -> Any:
        return self._runtime.spawn(coro, name=name)


class HostRuntime:
    """Reference tables, the asyncio loop for async plugin code, and the request handler."""

    def __init__(self, reader, writer):
        self.refs: Dict[int, Any] = {}
        self.owners: Dict[int, str] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.unload_callbacks: Dict[str, List[Callable[[], Any]]] = {}
        self.modules: Dict[str, str] = {}
        self.asgi_apps: Dict[str, Any] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.instance_modules: Dict[str, Any] = {}
        self.stopped = threading.Event()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="plugin-host-loop", daemon=True).start()
        self.channel = Channel(reader, writer, self.handle, name="plugin-host",
                               on_close=lambda _reason: self.stopped.set())

    # -- references -----------------------------------------------------------------------------
    def _remember(self, plugin_key: str, value: Any) -> int:
        with self._lock:
            ref = next(self._ids)
            self.refs[ref] = value
            self.owners[ref] = plugin_key
        return ref

    def ref_for(self, plugin_key: str, value: Any, *, allow_objects: bool) -> Optional[dict]:
        if isinstance(value, type):
            return None
        if callable(value) and not (allow_objects and _is_provider_object(value)):
            return {"__callable__": self._remember(plugin_key, value), "async": is_async_callable(value),
                    "sig": describe_signature(value), "name": getattr(value, "__name__", "callback"),
                    "qualname": getattr(value, "__qualname__", None),
                    "module": getattr(value, "__module__", None)}
        if allow_objects and _is_provider_object(value):
            return self._describe_object(plugin_key, value)
        return None

    def _describe_object(self, plugin_key: str, obj: Any) -> dict:
        """Method table + attribute plan for a provider object; every method runs here, in the host."""
        methods: Dict[str, Any] = {}
        instance = sorted(k for k in vars(obj) if not k.startswith("_")) if hasattr(obj, "__dict__") else []
        live: List[str] = list(instance)
        for name in dir(obj):
            if name.startswith("_") or name in instance:
                continue
            raw = inspect.getattr_static(obj, name, None)
            if isinstance(raw, property) or type(raw).__name__ == "cached_property":
                live.append(name)
                continue
            value = getattr(obj, name, None)
            if callable(value):
                methods[name] = {"async": is_async_callable(value), "sig": describe_signature(value)}
            else:
                # Class-level defaults too: an engine that sets ``context_length`` only later in
                # ``update_model()`` must not read as the base-class default forever in Hermes.
                live.append(name)
        return {"__object__": self._remember(plugin_key, obj), "methods": methods, "live": live,
                "type": type(obj).__name__}

    def resolve_from_parent(self, ref: dict) -> Any:
        if "__handle__" in ref:
            return RemoteRegistration(self, int(ref["__handle__"]), str(ref.get("kind") or ""),
                                      str(ref.get("key") or ""))
        return Opaque("callable" if "__callable__" in ref else "object")

    # -- outgoing ---------------------------------------------------------------------------------
    def _send_call(self, method: str, plugin_key: str, payload: Dict[str, Any], *, allow_objects: bool) -> Any:
        refs = functools.partial(self.ref_for, plugin_key, allow_objects=allow_objects)
        payload = {**payload, "plugin": plugin_key,
                   "args": encode(list(payload.pop("args", ())), refs),
                   "kwargs": encode(dict(payload.pop("kwargs", {})), refs)}
        return decode(self.channel.call(method, payload), self.resolve_from_parent)

    def ctx_call(self, plugin_key: str, method: str, *args: Any, **kwargs: Any) -> Any:
        from hermes_cli.plugin_isolation import HOST_OBJECT_BASES
        return self._send_call("ctx", plugin_key, {"method": method, "args": args, "kwargs": kwargs},
                               allow_objects=method in HOST_OBJECT_BASES)

    def facade_call(self, plugin_key: str, facade: str, method: str, *args: Any, _probe: bool = False,
                    **kwargs: Any) -> Any:
        return self._send_call("facade", plugin_key, {"facade": facade, "method": method, "probe": _probe,
                                                      "args": args, "kwargs": kwargs}, allow_objects=False)

    def spawn(self, coro: Any, *, name: Optional[str] = None) -> Any:
        if not inspect.iscoroutine(coro):
            raise TypeError("spawn_task() requires a coroutine object")
        return asyncio.run_coroutine_threadsafe(_on_behalf_of(coro, serving_request()), self.loop)

    # -- incoming ---------------------------------------------------------------------------------
    def handle(self, method: str, params: Dict[str, Any], _origin: Optional[int]) -> Any:
        handler = _HANDLERS.get(method)
        if handler is None:
            raise ValueError(f"unknown plugin host request {method!r}")
        return handler(self, params)

    def _run(self, result: Any) -> Any:
        """Await plugin coroutines on the host loop, still attributed to the request being served
        (their ``ctx`` calls then run in that caller's session, not a bare context)."""
        if inspect.isawaitable(result):
            return asyncio.run_coroutine_threadsafe(_on_behalf_of(result, serving_request()), self.loop).result()
        return result

    def _encode_result(self, plugin_key: str, value: Any) -> Any:
        return encode(value, functools.partial(self.ref_for, plugin_key, allow_objects=False))

    def _decode_args(self, params: Dict[str, Any]):
        return decode(params.get("args") or [], None), decode(params.get("kwargs") or {}, None)

    def op_hello(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"protocol": PROTOCOL_VERSION, "pid": os.getpid(), "python": sys.version.split()[0]}

    def op_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        plugin_key = str(params["plugin_key"])
        target = _import_plugin(params)
        register = target if callable(target) and not isinstance(target, types.ModuleType) \
            else getattr(target, "register", None)
        if not callable(register):
            raise AttributeError(f"Plugin '{params.get('name')}' has no register() function")
        module_name = getattr(target, "__name__", None) or getattr(register, "__module__", None)
        self.modules[plugin_key] = str(module_name or "")
        self._run(register(RemotePluginContext(self, plugin_key, params)))
        return {"module": module_name}

    def op_load_instance(self, params: Dict[str, Any]) -> Any:
        """Category plugins (memory provider, context engine, cron scheduler): import the directory,
        run ``register(ctx)`` capturing the one ``capture`` registration, else instantiate the first
        subclass of ``base``. Other registrations reach Hermes only when it bound a ctx for them."""
        plugin_key = str(params["plugin_key"])
        # Hermes asks for a fresh instance many times (agent cache, doctor, dashboard); like the
        # in-process loader, the module body runs once per host and only the instance is new.
        module = self.instance_modules.get(str(params["path"]))
        if module is None:
            module = self.instance_modules[str(params["path"])] = _import_plugin(params)
        self.modules[plugin_key] = module.__name__
        base = _import_base(str(params["base"]))
        capture = str(params["capture"])
        captured: List[Any] = []
        register = getattr(module, "register", None)
        if callable(register):
            ctx = _CapturingContext(self, plugin_key, params, capture, base, captured,
                                    forward=bool(params.get("forward")))
            try:
                self._run(register(ctx))
            except Exception as exc:
                if not captured:
                    logger.debug("register() failed for %s: %s", params.get("name"), exc)
        if not captured:
            for attr in dir(module):
                value = getattr(module, attr, None)
                if isinstance(value, type) and issubclass(value, base) and value is not base:
                    try:
                        captured.append(value())
                        break
                    except Exception:
                        continue
        if not captured:
            return None
        return self._describe_object(plugin_key, captured[0])

    def op_profile_call(self, params: Dict[str, Any]) -> Any:
        """Run one overridden method (or callable field) of a model-provider profile, loading the
        plugin on first use in this host process. Addressed by name, so it survives host restarts."""
        profiles = self.profiles.get(str(params["path"]))
        if profiles is None:
            profiles = self.profiles[str(params["path"])] = {
                p.name: p for p in capture_profiles(str(params["path"]), str(params["module_name"]))}
        target = getattr(profiles[str(params["profile"])], str(params["attr"]))
        args, kwargs = self._decode_args(params)
        return encode(self._run(target(*args, **kwargs)))

    def op_asgi(self, params: Dict[str, Any]) -> Any:
        """Serve one dashboard API request with the plugin's FastAPI ``router`` (loaded once)."""
        api_path = Path(str(params["dashboard_dir"])) / str(params["api_file"])
        app = self.asgi_apps.get(str(api_path))
        if app is None:
            from fastapi import FastAPI
            module_name = f"hermes_dashboard_plugin_{params['plugin']}"
            spec = importlib.util.spec_from_file_location(module_name, api_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load dashboard api {api_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            router = getattr(module, "router", None)
            if router is None:
                raise AttributeError(f"dashboard api {api_path.name} has no 'router'")
            app = FastAPI()
            app.include_router(router)
            self.asgi_apps[str(api_path)] = app

        async def request() -> Dict[str, Any]:
            import httpx
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://plugin-host") as client:
                response = await client.request(
                    str(params["method"]), str(params["path"]), params=str(params.get("query") or ""),
                    headers=[tuple(h) for h in params.get("headers") or []],
                    content=decode(params.get("body")) or b"")
            return {"status": response.status_code, "headers": list(response.headers.multi_items()),
                    "body": response.content}

        return encode(self._run(request()))

    def op_invoke(self, params: Dict[str, Any]) -> Any:
        ref = int(params["ref"])
        fn = self.refs.get(ref)
        if fn is None:
            raise LookupError(f"plugin host callable {ref} was released")
        args, kwargs = self._decode_args(params)
        return self._encode_result(self.owners.get(ref, ""), self._run(fn(*args, **kwargs)))

    def op_obj_invoke(self, params: Dict[str, Any]) -> Any:
        ref = int(params["ref"])
        obj = self.refs.get(ref)
        if obj is None:
            raise LookupError(f"plugin host object {ref} was released")
        method = str(params["method"])
        if method.startswith("_"):
            raise AttributeError(method)
        args, kwargs = self._decode_args(params)
        return self._encode_result(self.owners.get(ref, ""), self._run(getattr(obj, method)(*args, **kwargs)))

    def op_obj_getattr(self, params: Dict[str, Any]) -> Any:
        obj = self.refs.get(int(params["ref"]))
        name = str(params["name"])
        if obj is None:
            raise LookupError(f"plugin host object {params['ref']} was released")
        if name.startswith("_") or not hasattr(obj, name):
            return {"__missing__": name}
        return self._encode_result(self.owners.get(int(params["ref"]), ""), getattr(obj, name))

    def op_release(self, params: Dict[str, Any]) -> None:
        """Drop objects whose Hermes-side proxies were garbage-collected."""
        with self._lock:
            for ref in params.get("refs") or ():
                self.refs.pop(int(ref), None)
                self.owners.pop(int(ref), None)

    def op_config_schema(self, params: Dict[str, Any]) -> Any:
        """A memory provider's ``config_schema.py`` ``CONFIG_SCHEMA`` (user code: it runs here)."""
        path = Path(str(params["path"]))
        spec = importlib.util.spec_from_file_location(f"_hermes_memory_config_schema.{path.parent.name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return encode(getattr(module, "CONFIG_SCHEMA", None))

    def op_obj_setattr(self, params: Dict[str, Any]) -> None:
        obj = self.refs.get(int(params["ref"]))
        name = str(params["name"])
        if obj is None or name.startswith("_"):
            raise AttributeError(name)
        setattr(obj, name, decode(params.get("value"), None))

    def op_unload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        plugin_key = str(params["plugin_key"])
        errors = []
        for callback in reversed(self.unload_callbacks.pop(plugin_key, [])):
            try:
                self._run(callback())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        with self._lock:
            for ref in [r for r, owner in self.owners.items() if owner == plugin_key]:
                self.refs.pop(ref, None)
                self.owners.pop(ref, None)
        module_name = self.modules.pop(plugin_key, "")
        if module_name:
            for name in [m for m in sys.modules if m == module_name or m.startswith(module_name + ".")]:
                sys.modules.pop(name, None)
        return {"errors": errors}

    def op_shutdown(self, _params: Dict[str, Any]) -> None:
        for plugin_key in list(self.unload_callbacks):
            self.op_unload({"plugin_key": plugin_key})
        self.stopped.set()


_HANDLERS: Dict[str, Callable[[HostRuntime, Dict[str, Any]], Any]] = {
    "hello": HostRuntime.op_hello, "load": HostRuntime.op_load, "load_instance": HostRuntime.op_load_instance,
    "asgi": HostRuntime.op_asgi, "profile_call": HostRuntime.op_profile_call,
    "invoke": HostRuntime.op_invoke,
    "obj_invoke": HostRuntime.op_obj_invoke, "obj_getattr": HostRuntime.op_obj_getattr,
    "obj_setattr": HostRuntime.op_obj_setattr, "unload": HostRuntime.op_unload,
    "release": HostRuntime.op_release, "config_schema": HostRuntime.op_config_schema,
    "shutdown": HostRuntime.op_shutdown,
}


class _CapturingContext(RemotePluginContext):
    """``register(ctx)`` for a category plugin: the category's own registration is kept here; the rest
    forward to Hermes when it bound a context (memory providers), else are no-ops."""

    def __init__(self, runtime: HostRuntime, plugin_key: str, info: Dict[str, Any], capture: str,
                 base: type, captured: List[Any], *, forward: bool):
        super().__init__(runtime, plugin_key, info)
        self._capture, self._base, self._captured, self._forward = capture, base, captured, forward

    def __getattr__(self, name: str) -> Any:
        if name == self._capture:
            def capture(obj: Any, *_args: Any, **_kwargs: Any) -> None:
                if isinstance(obj, self._base) and not self._captured:
                    self._captured.append(obj)
            return capture
        if not self._forward and (name.startswith("register_") or name in {"subscribe"}):
            return lambda *_args, **_kwargs: None
        return super().__getattr__(name)


def _import_base(ref: str) -> type:
    module, attr = ref.split(":")
    return getattr(importlib.import_module(module), attr)


def _is_provider_object(value: Any) -> bool:
    """A plugin-defined instance (provider/engine), as opposed to a function or bound method."""
    return not isinstance(value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType,
                                  functools.partial, type)) and type(value).__module__ != "builtins"


async def _on_behalf_of(awaitable: Any, origin: Optional[int]) -> Any:
    bind_serving_request(origin)  # this task's context only; tasks it spawns inherit it
    return await awaitable


def _import_plugin(params: Dict[str, Any]) -> Any:
    """Import a directory plugin under the module name Hermes assigned, or resolve an entry point."""
    if params.get("entrypoint"):
        for ep in importlib.metadata.entry_points().select(group=_ENTRY_POINTS_GROUP):
            if ep.name == params.get("name"):
                return ep.load()
        raise ImportError(f"Entry point '{params.get('name')}' not found in group '{_ENTRY_POINTS_GROUP}'")
    plugin_dir = Path(str(params["path"]))
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"No __init__.py in {plugin_dir}")
    module_name = str(params["module_name"])
    parts = module_name.split(".")[:-1]
    for i in range(1, len(parts) + 1):  # synthetic parent packages (hermes_plugins, _hermes_user_memory)
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            ns_pkg = types.ModuleType(parent)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = parent
            sys.modules[parent] = ns_pkg
    spec = importlib.util.spec_from_file_location(module_name, init_file,
                                                  submodule_search_locations=[str(plugin_dir)])
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {init_file}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name
    module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in [m for m in sys.modules if m == module_name or m.startswith(module_name + ".")]:
            sys.modules.pop(name, None)
        raise
    return module


def _take_protocol_streams():
    """Duplicate stdin/stdout for the channel, then point fd 0 at /dev/null and fd 1 at stderr."""
    reader = os.fdopen(os.dup(0), "rb")  # windows-footgun: ok — binary protocol stream
    writer = os.fdopen(os.dup(1), "wb")  # windows-footgun: ok — binary protocol stream
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, encoding="utf-8-sig")  # noqa: SIM115 — lives for the process
    sys.stdout = sys.stderr
    return reader, writer


def capture_profiles(path: str, module_name: str) -> List[Any]:
    """Import a model-provider plugin and return the profiles it passes to ``register_provider``."""
    import providers
    from providers.base import ProviderProfile
    providers.list_providers()  # bundled + known profiles register before the capture window
    captured: List[Any] = []
    original = providers.register_provider
    providers.register_provider = captured.append  # type: ignore[assignment]
    try:
        _import_plugin({"path": path, "module_name": module_name, "name": Path(path).name})
    finally:
        providers.register_provider = original  # type: ignore[assignment]
    for profile in captured:
        if type(profile).create_client is not ProviderProfile.create_client:
            raise PluginHostUnsupported(
                f"provider profile {getattr(profile, 'name', '?')!r} overrides create_client(), which returns "
                "a live SDK client; it runs only in-process")
    return captured


def describe_profile(profile: Any) -> Dict[str, Any]:
    """Profile data (instance fields) plus the methods and callable fields that must run here."""
    from providers.base import ProviderProfile
    fields: Dict[str, Any] = {}
    calls: Dict[str, Any] = {}
    for name, value in vars(profile).items():
        if name.startswith("_"):
            continue
        if callable(value):
            calls[name] = {"async": is_async_callable(value), "sig": describe_signature(value), "field": True}
        else:
            fields[name] = encode(value)
    for name in dir(type(profile)):
        if name.startswith("_") or name in fields or name in calls:
            continue
        attr = getattr(type(profile), name)
        if attr is getattr(ProviderProfile, name, object()):
            continue
        if isinstance(inspect.getattr_static(profile, name), property) or not callable(attr):
            fields[name] = encode(getattr(profile, name))
            continue
        calls[name] = {"async": is_async_callable(attr), "sig": describe_signature(getattr(profile, name)),
                       "field": False}
    return {"name": profile.name, "type": type(profile).__name__, "fields": fields, "calls": calls}


def extract_profiles_main(path: str, module_name: str) -> int:
    """``--extract-profiles``: one-shot, credential-free; prints the profiles as JSON on stdout."""
    import json
    reader, writer = _take_protocol_streams()
    del reader
    import hermes_bootstrap  # noqa: F401
    try:
        payload = {"profiles": [describe_profile(p) for p in capture_profiles(path, module_name)]}
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    writer.write(json.dumps(payload).encode("utf-8"))
    writer.flush()
    return 0


def main() -> int:
    reader, writer = _take_protocol_streams()
    import hermes_bootstrap  # noqa: F401 — PM dependency environments, like every Hermes entry point
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("HERMES_PLUGIN_HOST_LOG_LEVEL", "WARNING"),
                        format="%(levelname)s %(name)s: %(message)s")
    runtime = HostRuntime(reader, writer)
    runtime.channel.start()
    runtime.stopped.wait()
    runtime.channel.close("host shutdown")
    return 0


if __name__ == "__main__":
    code = (extract_profiles_main(sys.argv[2], sys.argv[3]) if sys.argv[1:2] == ["--extract-profiles"]
            else main())
    sys.stderr.flush()
    # A plugin thread stuck in a handler must not keep the host alive once Hermes has gone.
    os._exit(code)
