"""Host-side contract for external memory providers (profile plugin directories and pip entry points).
Dormant for bundled providers, which keep origin/main's plain-import paths in plugins.memory,
web_server_memory, memory_providers and memory_oauth; the sole provider path once they leave core.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from plugins.memory.config_schema import STORAGE_HONCHO_HOST_BLOCK, ProviderConfigSchema
from plugins.package_generation import PackageSnapshot, capture_package, load_package_generation

logger = logging.getLogger(__name__)

OAUTH_HOOKS = ("start_loopback_flow_background", "get_flow_status")
STATE_DETAIL = {"idle": "", "pending": "Waiting for browser consent",
                "connected": "Connected", "error": "Authorization did not complete"}
_DESKTOP_KINDS = {"boolean": "bool", "integer": "number"}
# Resolved config_schema.py path -> (source digest, schema): a same-size reinstall keeps the mtime.
_SCHEMA_CACHE: Dict[str, Tuple[bytes, ProviderConfigSchema]] = {}
_pending: Dict[Tuple[str, str], Any] = {}  # (home, provider) -> runner of a flow that is still pending


class CompanionError(RuntimeError):
    """A companion file could not be loaded or run."""


def for_provider(name: str) -> Optional["ExternalProvider"]:
    """The external provider called ``name``; None for bundled providers and unknown names."""
    from plugins.memory import _is_bundled, find_provider_dir, find_provider_entry_point

    directory = find_provider_dir(name)
    if directory is not None:
        return None if _is_bundled(directory) else ExternalProvider(name, directory)
    return ExternalProvider(name) if find_provider_entry_point(name) is not None else None


def generation(provider_dir: Path) -> Tuple[str, PackageSnapshot]:
    """Module name and frozen sources of one package generation, keyed by home and source bytes,
    so a session keeps its module across an update while new loads get the new code."""
    from hermes_cli.plugin_installation import plugin_installation_lock
    from hermes_constants import hermes_home_key
    from plugins.memory import _module_name

    with plugin_installation_lock():
        snapshot = capture_package(provider_dir)
    owner = hashlib.sha256(hermes_home_key().encode()).hexdigest()[:16]
    return f"{_module_name(provider_dir, provider_dir.name)}__home_{owner}__generation_{snapshot.digest}", snapshot


def normalize_status(raw: Any) -> Dict[str, Any]:
    """Reduce a hook's dict to state, connected and auth; provider strings never cross."""
    data = raw if isinstance(raw, dict) else {}
    state = data.get("state") if data.get("state") in STATE_DETAIL else "error"
    status: Dict[str, Any] = {"state": state, "detail": STATE_DETAIL[state]}
    if data.get("connected") is True:
        status["connected"] = True
    if "auth" in data:
        status["auth"] = data["auth"] if data["auth"] in ("oauth", "apikey") else None
    return status


def load_oauth_companion(snapshot: PackageSnapshot):
    """Import ``oauth_flow`` from a captured package without executing its ``__init__``."""
    name = f"_hermes_memory_companions_{snapshot.digest}"
    load_package_generation(name, snapshot, execute_init=False, exact_files=frozenset({"oauth_flow"}))
    module = importlib.import_module(name + ".oauth_flow")
    if not all(callable(getattr(module, hook, None)) for hook in OAUTH_HOOKS):
        raise TypeError("Invalid OAuth companion")
    return module


class _ThreadRunner:
    """Hooks that accept ``hermes_home`` run in this process; the launcher gets a thread under the owner's scope."""

    alive = True

    def __init__(self, module, home: Path):
        self.module, self.home = module, str(home)

    def request(self, *, start: bool):
        from agent.memory_provider import spawn_context_thread

        if start:
            spawn_context_thread(self.module.start_loopback_flow_background, name=f"memory-oauth-{self.module.__name__}",
                                 kwargs={"hermes_home": self.home}).start()
        return self.module.get_flow_status(hermes_home=self.home)

    def close(self):
        pass


def _hooks_take_hermes_home(companion: Path) -> bool:
    """Both hooks are module-level ``def``s with a ``hermes_home`` parameter, read from the source:
    importing the companion here would run its module body with the launch environment."""
    defs = {node.name: node.args for node in ast.parse(companion.read_bytes(), str(companion)).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return all(hook in defs and any(a.arg == "hermes_home" for a in (*defs[hook].posonlyargs, *defs[hook].args,
                                                                      *defs[hook].kwonlyargs)) for hook in OAUTH_HOOKS)


def _runner(home: Path, companion: Path):
    snapshot = capture_package(companion.parent)
    if _hooks_take_hermes_home(companion):
        return _ThreadRunner(load_oauth_companion(snapshot), home)
    from hermes_cli.memory_oauth_process import OAuthChild

    return OAuthChild(home, snapshot.directory)


def shutdown_oauth() -> None:
    while _pending:
        _pending.popitem()[1].close()


def _desktop_field(field: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """A raw-schema row in the Desktop's shape: its kind names, string values, write-only secrets."""
    from hermes_cli.web_server_memory import _field_is_set, _field_value

    kind = field["kind"]
    value = "" if kind == "secret" else _field_value(field, data)
    if isinstance(value, bool):
        value = "true" if value else "false"
    return {**{k: field[k] for k in ("key", "label", "description", "placeholder", "required", "options",
                                      "url", "when", "minimum", "maximum", "step")},
            "kind": _DESKTOP_KINDS.get(kind, kind), "default": "" if kind == "secret" else field["default"],
            "value": str(value), "is_set": _field_is_set(field, data), "info": "", "inline": False, "group": ""}


def _reject_unknown(values: Dict[str, Any], keys) -> None:
    if set(values) - set(keys):
        raise ValueError("Submission contains unknown memory provider fields")


@dataclass
class ExternalProvider:
    name: str
    directory: Optional[Path] = None

    def companion(self, filename: str) -> Optional[Path]:
        """``<package dir>/<filename>``: config_schema.py or oauth_flow.py, read from disk, never imported as a package."""
        path = self.directory / filename if self.directory is not None else None
        return path if path is not None and path.is_file() else None

    def load(self):
        from plugins.memory import load_memory_provider

        try:
            return load_memory_provider(self.name)
        except Exception:
            logger.debug("Failed to load memory provider %s", self.name, exc_info=True)
            return None

    def declared_schema(self) -> Optional[ProviderConfigSchema]:
        """``CONFIG_SCHEMA`` of the package's ``config_schema.py``, reloaded whenever its bytes change."""
        path = self.companion("config_schema.py")
        if path is None:
            return None
        key, source = str(path.resolve()), path.read_bytes()
        fingerprint = hashlib.sha256(source).digest()
        cached = _SCHEMA_CACHE.get(key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        try:
            module = importlib.util.module_from_spec(
                importlib.util.spec_from_file_location(f"_hermes_memory_config_schema.{self.name}", path))
            exec(compile(source, str(path), "exec"), module.__dict__)
            schema = getattr(module, "CONFIG_SCHEMA", None)
        except Exception:
            schema = None
            logger.exception("failed to load config schema for memory provider %r", self.name)
        if schema is None:
            _SCHEMA_CACHE.pop(key, None)
        else:
            _SCHEMA_CACHE[key] = (fingerprint, schema)
        return schema

    def host_modules(self, declared: ProviderConfigSchema):
        """``client`` and ``oauth`` of this package, imported from the generation the runtime loads, for a
        ``honcho_host_block`` schema; None for other storage. The bundled namespace is never assumed."""
        from hermes_cli.web_routers.memory_providers import HonchoModules
        from plugins.memory import import_provider_package

        if declared.storage != STORAGE_HONCHO_HOST_BLOCK:
            return None
        package = import_provider_package(self.directory) if self.directory is not None else None
        if package is None:
            raise CompanionError(f"memory provider {self.name} did not import")
        return HonchoModules(*(importlib.import_module(f"{package.__name__}.{m}") for m in ("client", "oauth")))

    def describe(self) -> Dict[str, Any]:
        """The Desktop config form: the declared schema when the package ships one, else the raw
        ``get_config_schema()`` rows, with ``requires_full_form`` when the provider saves itself."""
        from hermes_cli.plugin_installation import management_scope
        from hermes_cli.web_routers import memory_providers as router
        from hermes_cli.web_server_memory import _normalize_memory_provider_schema, _read_memory_provider_existing_values

        with management_scope():
            declared = self.declared_schema()
            if declared is not None:
                return router._declared_provider_payload(declared, modules=self.host_modules(declared))
            provider = self.load()
            fields = _normalize_memory_provider_schema(self.name, provider) if provider is not None else []
            data = _read_memory_provider_existing_values(self.name) if fields else {}
            partial = router._memory_provider_native_writer(provider) is None if fields else True
        return {"name": self.name, "label": self.name.replace("_", " ").replace("-", " ").title(), "docs_url": "",
                "fields": [_desktop_field(field, data) for field in fields],
                **({"capabilities": router._capabilities(partial_saves=partial)} if fields else {})}

    def save(self, values: Dict[str, Any]) -> bool:
        """Persist a Desktop submission without selecting the provider; False when there is nothing to configure.
        Declared schemas save partially into host storage; raw schemas go through the provider's ``save_config``
        or the host's ``memory.<name>`` merge, missing fields filled from stored values then defaults."""
        from hermes_cli.plugin_installation import management_scope
        from hermes_cli.web_routers import memory_providers as router
        from hermes_cli.web_server_memory import _normalize_memory_provider_schema

        with management_scope():
            declared = self.declared_schema()
            if declared is not None:
                _reject_unknown(values, [field.key for field in declared.fields])
                router._update_memory_provider_config(
                    declared, {k: router._stringify_submitted(v) for k, v in values.items()}, activate=False,
                    modules=self.host_modules(declared))
                return True
            provider = self.load()
            fields = _normalize_memory_provider_schema(self.name, provider) if provider is not None else []
            if not fields:
                return False
            _reject_unknown(values, [field["key"] for field in fields])
            router._write_memory_provider_config_values(self.name, provider, values)
            return True

    def oauth(self, home: Path, *, start: bool) -> Optional[Dict[str, Any]]:
        """Run the companion's start or status hook for ``home``; None when the package ships none.
        A pending flow keeps its runner, and its code, until the hook reports a terminal state or the host
        shuts down; a repeated start reads status, and a status call that raises reports ``error`` once."""
        from hermes_cli.plugin_installation import management_scope

        companion = self.companion("oauth_flow.py")
        if companion is None:
            return None
        key = (str(home), self.name)
        with management_scope():
            pinned = _pending.get(key)
            try:
                runner = pinned or _runner(home, companion)
            except Exception as exc:
                logger.debug("OAuth companion for %s failed to load", self.name, exc_info=True)
                raise CompanionError(str(exc)) from exc
            try:
                status = normalize_status(runner.request(start=start and pinned is None))
            except Exception as exc:
                logger.debug("OAuth companion for %s failed", self.name, exc_info=True)
                if pinned is not None and runner.alive:
                    return normalize_status({"state": "error"})
                _pending.pop(key, None)
                runner.close()
                raise CompanionError(str(exc)) from exc
            if status["state"] == "pending":
                _pending[key] = runner
            else:
                _pending.pop(key, None)
                runner.close()
            return status
