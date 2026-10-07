"""Model-provider plugins under ``plugins.isolation: host``.

Model-provider discovery runs while ``hermes_cli.config`` / ``hermes_cli.auth`` are still importing,
so it must not start the plugin host (building the host's environment imports those same modules).
Profile DATA therefore comes from a one-shot, credential-free extraction
(``plugin_host_child --extract-profiles``), cached under the profile home by the plugin's file
fingerprint. Overridden methods and callable fields run later in the profile's plugin host, addressed
by (plugin dir, profile name, attribute) so they survive host restarts.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

from hermes_cli.plugin_host_wire import PluginHostUnavailable, decode, signature_from

logger = logging.getLogger(__name__)

_EXTRACT_TIMEOUT_SECS = 60.0
_CACHE_VERSION = 1


def load_hosted_profiles(plugin_dir: Path, module_name: str) -> List[Any]:
    """ProviderProfile proxies for a model-provider plugin whose code runs in the plugin host."""
    from providers.base import ProviderProfile
    payload = _cached_extraction(Path(plugin_dir), module_name)
    if payload.get("error"):
        raise PluginHostUnavailable(str(payload["error"]))
    return [_profile_proxy(ProviderProfile, str(plugin_dir), module_name, entry)
            for entry in payload.get("profiles") or []]


def _fingerprint(plugin_dir: Path) -> str:
    digest = hashlib.sha256(f"{_CACHE_VERSION}|{sys.executable}".encode())
    try:
        from hermes_cli import __version__
        digest.update(str(__version__).encode())
    except Exception:
        pass
    for path in sorted(plugin_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        stat = path.stat()
        digest.update(f"{path.relative_to(plugin_dir)}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _cached_extraction(plugin_dir: Path, module_name: str) -> Dict[str, Any]:
    from hermes_constants import get_hermes_home
    # Keyed by the full path too: a user and a project plugin may share a directory name.
    path_key = hashlib.sha256(str(plugin_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    cache = get_hermes_home() / "cache" / "plugin_host" / "model-providers" / f"{plugin_dir.name}-{path_key}.json"
    fingerprint = _fingerprint(plugin_dir)
    try:
        cached = json.loads(cache.read_text(encoding="utf-8-sig"))
        if cached.get("fingerprint") == fingerprint and cached.get("module_name") == module_name:
            return cached["payload"]
    except (OSError, ValueError, KeyError):
        pass
    payload = _extract(plugin_dir, module_name)
    if not payload.get("error"):
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint, "module_name": module_name,
                                   "payload": payload}), encoding="utf-8")
        os.replace(tmp, cache)
    return payload


def _extract(plugin_dir: Path, module_name: str) -> Dict[str, Any]:
    """Import the plugin in a throwaway host process with no credentials and read its profiles."""
    from hermes_cli.plugin_isolation import HOST_PROCESS_ENV, host_launcher
    from hermes_constants import get_hermes_home
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT") if key in os.environ}
    # Windows resolves Path.home() from USERPROFILE, never HOME: without it the child cannot start.
    env.update(HOME=str(Path.home()), USERPROFILE=str(Path.home()), HERMES_HOME=str(get_hermes_home()),
               PYTHONPATH=repo_root,
               **{HOST_PROCESS_ENV: "1"})
    argv = [*host_launcher(), sys.executable, "-m", "hermes_cli.plugin_host_child", "--extract-profiles",
            str(plugin_dir), module_name]
    try:
        done = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, env=env,
                              timeout=_EXTRACT_TIMEOUT_SECS, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"profile extraction failed: {exc}"}
    try:
        return json.loads(done.stdout.decode("utf-8"))
    except ValueError:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        return {"error": f"profile extraction exited {done.returncode}: {' | '.join(tail)}"}


def _host_call(plugin_dir: str, module_name: str, profile: str, attr: str) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        from hermes_cli.plugin_isolation import user_plugin_host
        host = user_plugin_host()
        if host is None:
            raise PluginHostUnavailable("plugins.isolation is no longer 'host'; restart Hermes")
        return host.profile_call(plugin_dir, module_name, profile, attr, args, kwargs)
    return call


def _profile_proxy(base: type, plugin_dir: str, module_name: str, entry: Dict[str, Any]) -> Any:
    name = str(entry["name"])
    namespace: Dict[str, Any] = {"__module__": __name__,
                                 "__repr__": lambda self_: f"<plugin-host profile {name!r}>"}
    fields = {key: decode(value) for key, value in (entry.get("fields") or {}).items()}
    for attr, meta in (entry.get("calls") or {}).items():
        call = _host_call(plugin_dir, module_name, name, attr)
        if meta.get("async"):
            sync_call = call

            async def call(*args: Any, _sync: Callable[..., Any] = sync_call, **kwargs: Any) -> Any:
                return await asyncio.to_thread(_sync, *args, **kwargs)
        signature = signature_from(meta.get("sig"))
        if signature is not None:
            call.__signature__ = signature  # type: ignore[attr-defined]
        call.__name__ = call.__qualname__ = attr
        if meta.get("field"):
            fields[attr] = call
        else:
            namespace[attr] = _as_method(call)
    cls = type(f"Hosted{entry.get('type') or base.__name__}", (base,), namespace)
    profile = object.__new__(cls)
    for key, value in fields.items():
        object.__setattr__(profile, key, value)
    return profile


def _as_method(call: Callable[..., Any]) -> Callable[..., Any]:
    if asyncio.iscoroutinefunction(call):
        async def method(self_: Any, *args: Any, **kwargs: Any) -> Any:
            return await call(*args, **kwargs)
    else:
        def method(self_: Any, *args: Any, **kwargs: Any) -> Any:
            return call(*args, **kwargs)
    method.__name__ = method.__qualname__ = call.__name__
    signature = getattr(call, "__signature__", None)
    if signature is not None:  # callers introspect accepted kwargs; bound access drops ``self`` again
        self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        method.__signature__ = signature.replace(  # type: ignore[attr-defined]
            parameters=[self_param, *signature.parameters.values()])
    return method
