"""Host-owned NeMo-Flow telemetry bridge.

This module translates Hermes' generic observer hook payloads into NeMo-Flow
runtime events. It is intentionally not a Hermes plugin: plugin hooks remain
the public extension bus, while this adapter is the candidate canonical
telemetry service for the host itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal
import tomllib

logger = logging.getLogger(__name__)

_INIT_FAILED = object()
_runtime_lock = threading.RLock()
_runtime: "_Runtime | object | None" = None
_enabled_cache: bool | None = None
_DEFAULT_PLUGINS_TOML = Path(__file__).with_name("nemo_flow_plugins.toml")

_HOOKS = {
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "pre_tool_call",
    "post_tool_call",
    "pre_approval_request",
    "post_approval_response",
    "subagent_start",
    "subagent_stop",
}

ErrorPolicy = Literal["log", "ignore"]
TelemetryObserver = Callable[[dict[str, Any]], None]

NEMO_FLOW_TELEMETRY_SCHEMA_VERSION = "nemo_flow.telemetry.v1"

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "proxy_authorization",
    "cookie",
    "secret",
    "token",
    "password",
)

_SAFE_TOKEN_ACCOUNTING_KEYS = {
    "tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "audio_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
}


@dataclass(slots=True)
class _NoopSubscription:
    name: str
    reason: str
    _active: bool = True

    def deregister(self) -> bool:
        self._active = False
        return False

    def __enter__(self) -> "_NoopSubscription":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.deregister()


@dataclass(slots=True)
class _Settings:
    max_payload_chars: int = 50000
    plugins_toml_path: str = ""
    discover_plugins_toml: bool = False
    atof_enabled: bool = False
    atof_output_directory: str = ""
    atof_filename: str = "hermes-atof.jsonl"
    atof_mode: str = "append"
    atif_enabled: bool = False
    atif_output_directory: str = ""
    atif_filename_template: str = "hermes-atif-{session_id}.json"
    atif_agent_name: str = "Hermes Agent"
    atif_agent_version: str = "0.13.0"


@dataclass(slots=True)
class _PluginTomlConfig:
    value: dict[str, Any]
    user_supplied: bool


@dataclass(slots=True)
class _PluginInitResult:
    initialized: bool
    owns_exporters: bool


def record_hook(hook_name: str, kwargs: dict[str, Any]) -> None:
    """Record one Hermes hook event into NeMo-Flow when telemetry is enabled."""
    if hook_name not in _HOOKS or not _enabled():
        return

    runtime = _get_runtime()
    if runtime is None:
        return

    try:
        runtime.handle_hook(hook_name, kwargs)
    except Exception as exc:
        logger.debug(
            "NeMo-Flow telemetry hook '%s' failed: %s", hook_name, exc, exc_info=True
        )


def is_enabled() -> bool:
    """Return whether Hermes is configured to mirror hooks into NeMo-Flow."""
    return _enabled()


def observer_available() -> bool:
    """Return whether the stable NeMo-Flow telemetry observer API is available."""
    if not _enabled():
        return False
    runtime = _get_runtime()
    if runtime is None:
        return False
    telemetry_v1 = _get_telemetry_v1(runtime)
    return callable(getattr(telemetry_v1, "register_observer", None))


def register_observer(
    name: str,
    callback: TelemetryObserver,
    *,
    error_policy: ErrorPolicy = "log",
) -> Any:
    """Register a stable telemetry observer when NeMo-Flow supports it.

    Hermes currently supports the published ``nemo-flow==0.2.0rc3`` wheel, which
    does not yet expose ``nemo_flow.telemetry_v1``. In that case this returns a
    no-op subscription so callers can adopt the facade before the NeMo-Flow API
    lands in a released wheel.
    """
    if error_policy not in {"log", "ignore"}:
        raise ValueError("error_policy must be one of: log, ignore")
    if not _enabled():
        return _NoopSubscription(name=name, reason="disabled")

    runtime = _get_runtime()
    if runtime is None:
        return _NoopSubscription(name=name, reason="nemo-flow-unavailable")

    telemetry_v1 = _get_telemetry_v1(runtime)
    register = getattr(telemetry_v1, "register_observer", None)
    if not callable(register):
        return _NoopSubscription(name=name, reason="telemetry-v1-unavailable")

    return register(name, callback, error_policy=error_policy)


def observer(
    name: str,
    callback: TelemetryObserver,
    *,
    error_policy: ErrorPolicy = "log",
) -> Any:
    """Context-manager alias for :func:`register_observer`."""
    return register_observer(name, callback, error_policy=error_policy)


def reset_for_tests() -> None:
    """Reset cached adapter state."""
    global _runtime, _enabled_cache
    with _runtime_lock:
        if isinstance(_runtime, _Runtime) and _runtime._plugin_initialized:
            _clear_plugin_configuration(_runtime.nemo_flow)
        _runtime = None
        _enabled_cache = None


def _enabled() -> bool:
    raw = os.getenv("HERMES_NEMO_FLOW_TELEMETRY")
    if raw is not None:
        return _truthy(raw)

    global _enabled_cache
    if _enabled_cache is not None:
        return _enabled_cache

    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config()
        _enabled_cache = bool(
            cfg_get(config, "telemetry", "nemo_flow", "enabled", default=False)
        )
    except Exception:
        _enabled_cache = False
    return _enabled_cache


def _get_runtime() -> "_Runtime | None":
    global _runtime
    with _runtime_lock:
        if _runtime is _INIT_FAILED:
            return None
        if isinstance(_runtime, _Runtime):
            return _runtime
        try:
            import nemo_flow
        except Exception as exc:
            logger.debug("NeMo-Flow telemetry disabled: import failed: %s", exc)
            _runtime = _INIT_FAILED
            return None
        try:
            _runtime = _Runtime(nemo_flow=nemo_flow, settings=_load_settings())
            _runtime.configure_exporters()
        except Exception as exc:
            logger.debug(
                "NeMo-Flow telemetry disabled: init failed: %s", exc, exc_info=True
            )
            _runtime = _INIT_FAILED
            return None
        return _runtime


def _get_telemetry_v1(runtime: "_Runtime") -> Any:
    telemetry_v1 = getattr(runtime.nemo_flow, "telemetry_v1", None)
    if telemetry_v1 is not None:
        return telemetry_v1
    try:
        from nemo_flow import telemetry_v1 as imported_telemetry_v1

        return imported_telemetry_v1
    except Exception:
        return None


def _load_settings() -> _Settings:
    settings = _Settings()
    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config()
    except Exception:
        config = {}

    def get(*keys: str, default: Any = None) -> Any:
        try:
            from hermes_cli.config import cfg_get

            return cfg_get(config, "telemetry", "nemo_flow", *keys, default=default)
        except Exception:
            return default

    settings.max_payload_chars = _int(
        os.getenv("HERMES_NEMO_FLOW_MAX_PAYLOAD_CHARS")
        or get("max_payload_chars", default=settings.max_payload_chars),
        settings.max_payload_chars,
    )
    settings.plugins_toml_path = str(
        os.getenv("HERMES_NEMO_FLOW_PLUGINS_TOML")
        or get("plugins_toml_path", default=settings.plugins_toml_path)
        or settings.plugins_toml_path
    )
    discover_plugins_toml = os.getenv("HERMES_NEMO_FLOW_DISCOVER_PLUGINS_TOML")
    settings.discover_plugins_toml = (
        _truthy(discover_plugins_toml)
        if discover_plugins_toml is not None
        else bool(
            get(
                "discover_plugins_toml",
                default=settings.discover_plugins_toml,
            )
        )
    )

    atof_dir = os.getenv("HERMES_NEMO_FLOW_ATOF_DIR") or str(
        get("atof", "output_directory", default="") or ""
    )
    settings.atof_output_directory = atof_dir
    settings.atof_enabled = bool(atof_dir) or bool(
        get("atof", "enabled", default=False)
    )
    settings.atof_filename = str(
        os.getenv("HERMES_NEMO_FLOW_ATOF_FILENAME")
        or get("atof", "filename", default=settings.atof_filename)
        or settings.atof_filename
    )
    settings.atof_mode = str(
        get("atof", "mode", default=settings.atof_mode) or settings.atof_mode
    ).lower()

    atif_dir = os.getenv("HERMES_NEMO_FLOW_ATIF_DIR") or str(
        get("atif", "output_directory", default="") or ""
    )
    settings.atif_output_directory = atif_dir
    settings.atif_enabled = bool(atif_dir) or bool(
        get("atif", "enabled", default=False)
    )
    settings.atif_filename_template = str(
        os.getenv("HERMES_NEMO_FLOW_ATIF_FILENAME_TEMPLATE")
        or get("atif", "filename_template", default=settings.atif_filename_template)
        or settings.atif_filename_template
    )
    settings.atif_agent_name = str(
        get("atif", "agent_name", default=settings.atif_agent_name)
        or settings.atif_agent_name
    )
    settings.atif_agent_version = str(
        get("atif", "agent_version", default=settings.atif_agent_version)
        or settings.atif_agent_version
    )
    return settings


def _load_plugins_toml(settings: _Settings) -> _PluginTomlConfig | None:
    paths = _plugins_toml_paths(settings)
    if not paths:
        return None

    merged: dict[str, Any] = {}
    loaded = False
    user_supplied = False
    for path, is_user_supplied in paths:
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                value = tomllib.load(handle)
            _validate_component_kinds(path, value)
            merged = _merge_plugin_toml(merged, value)
            loaded = True
            user_supplied = user_supplied or is_user_supplied
        except Exception as exc:
            logger.debug(
                "NeMo-Flow plugins.toml skipped at %s: %s",
                path,
                exc,
                exc_info=True,
            )
    return _PluginTomlConfig(merged, user_supplied=user_supplied) if loaded else None


def _plugins_toml_paths(settings: _Settings) -> list[tuple[Path, bool]]:
    paths = [(_DEFAULT_PLUGINS_TOML, False)]
    if settings.plugins_toml_path:
        paths.append((Path(settings.plugins_toml_path).expanduser(), True))
        return paths
    if not settings.discover_plugins_toml:
        return paths

    paths.append((Path("/etc/nemo-flow/plugins.toml"), True))
    project_path = _nearest_project_plugins_toml(Path.cwd())
    if project_path is not None:
        paths.append((project_path, True))
    paths.append((_user_plugins_toml_path(), True))
    return paths


def _nearest_project_plugins_toml(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        path = candidate / ".nemo-flow" / "plugins.toml"
        if path.exists():
            return path
    return None


def _user_plugins_toml_path() -> Path:
    xdg_config = os.getenv("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config).expanduser() / "nemo-flow" / "plugins.toml"
    return Path.home() / ".config" / "nemo-flow" / "plugins.toml"


def _validate_component_kinds(path: Path, value: dict[str, Any]) -> None:
    seen: set[str] = set()
    components = value.get("components")
    if not isinstance(components, list):
        return
    for component in components:
        if not isinstance(component, dict):
            continue
        kind = component.get("kind")
        if not isinstance(kind, str):
            continue
        if kind in seen:
            raise ValueError(
                f"duplicate plugin component kind in {path}: {kind}; declare "
                "each kind once per plugins.toml"
            )
        seen.add(kind)


def _merge_plugin_toml(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if key == "components" and isinstance(value, list):
            merged[key] = _merge_components(merged.get(key), value)
        else:
            merged[key] = _merge_values(merged.get(key), value)
    return merged


def _merge_components(left: Any, right: list[Any]) -> list[Any]:
    components = list(left) if isinstance(left, list) else []
    by_kind = {
        component.get("kind"): index
        for index, component in enumerate(components)
        if isinstance(component, dict) and isinstance(component.get("kind"), str)
    }
    for component in right:
        if not isinstance(component, dict) or not isinstance(
            component.get("kind"), str
        ):
            components.append(component)
            continue
        index = by_kind.get(component["kind"])
        if index is None:
            by_kind[component["kind"]] = len(components)
            components.append(component)
        else:
            components[index] = _merge_values(components[index], component)
    return components


def _merge_values(left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        merged = dict(left)
        for key, value in right.items():
            merged[key] = _merge_values(merged.get(key), value)
        return merged
    return right


def _plugin_config_has_enabled_exporter(config: dict[str, Any]) -> bool:
    components = config.get("components")
    if not isinstance(components, list):
        return False

    for component in components:
        if not isinstance(component, dict) or component.get("kind") != "observability":
            continue
        if component.get("enabled") is False:
            continue
        component_config = component.get("config")
        if not isinstance(component_config, dict):
            continue
        for exporter_name in ("atof", "atif", "opentelemetry", "openinference"):
            exporter_config = component_config.get(exporter_name)
            if isinstance(exporter_config, dict) and _truthy(
                exporter_config.get("enabled")
            ):
                return True
    return False


def _run_async(coro: Any) -> Any:
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=target, name="hermes-nemo-flow-plugin-init")
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _clear_plugin_configuration(nemo_flow: Any) -> None:
    plugin_module = getattr(nemo_flow, "plugin", None)
    if plugin_module is None:
        try:
            from nemo_flow import plugin as plugin_module
        except Exception:
            return
    clear = getattr(plugin_module, "clear", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            pass


@dataclass(slots=True)
class _AtifState:
    exporter: Any
    subscriber_name: str


class _Runtime:
    def __init__(self, *, nemo_flow: Any, settings: _Settings) -> None:
        self.nemo_flow = nemo_flow
        self.settings = settings
        self._session_scopes: dict[str, Any] = {}
        self._turn_scopes: dict[str, Any] = {}
        self._api_spans: dict[str, Any] = {}
        self._tool_spans: dict[str, Any] = {}
        self._subagent_scopes: dict[str, Any] = {}
        self._atif: dict[str, _AtifState] = {}
        self._atof_exporter: Any = None
        self._lock = threading.RLock()
        self._plugin_initialized = False
        self._plugin_configured = False

    def configure_exporters(self) -> None:
        plugin_result = self._configure_plugin_exporters()
        self._plugin_initialized = plugin_result.initialized
        self._plugin_configured = plugin_result.owns_exporters
        if self._plugin_configured:
            return

        if not self.settings.atof_enabled:
            return

        config = self.nemo_flow.AtofExporterConfig()
        if self.settings.atof_output_directory:
            output_dir = Path(self.settings.atof_output_directory).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            config.output_directory = str(output_dir)
        if self.settings.atof_filename:
            config.filename = self.settings.atof_filename
        if self.settings.atof_mode == "overwrite":
            config.mode = self.nemo_flow.AtofExporterMode.Overwrite
        else:
            config.mode = self.nemo_flow.AtofExporterMode.Append

        try:
            self._atof_exporter = self.nemo_flow.AtofExporter(config)
            self._atof_exporter.register("hermes-nemo-flow-atof")
        except Exception as exc:
            self._atof_exporter = None
            logger.debug("NeMo-Flow ATOF exporter setup failed: %s", exc, exc_info=True)

    def _configure_plugin_exporters(self) -> _PluginInitResult:
        plugins_toml = _load_plugins_toml(self.settings)
        if plugins_toml is None:
            return _PluginInitResult(initialized=False, owns_exporters=False)

        plugin_module = getattr(self.nemo_flow, "plugin", None)
        if plugin_module is None:
            try:
                from nemo_flow import plugin as plugin_module
            except Exception as exc:
                logger.debug(
                    "NeMo-Flow plugins.toml ignored: plugin API unavailable: %s",
                    exc,
                    exc_info=True,
                )
                return _PluginInitResult(initialized=False, owns_exporters=False)

        try:
            _run_async(plugin_module.initialize(plugins_toml.value))
        except Exception as exc:
            logger.debug(
                "NeMo-Flow plugins.toml initialization failed: %s",
                exc,
                exc_info=True,
            )
            return _PluginInitResult(initialized=False, owns_exporters=False)

        return _PluginInitResult(
            initialized=True,
            owns_exporters=plugins_toml.user_supplied
            or _plugin_config_has_enabled_exporter(plugins_toml.value),
        )

    def handle_hook(self, hook_name: str, kwargs: dict[str, Any]) -> None:
        with self._lock:
            if hook_name == "on_session_start":
                handle = self._ensure_session(kwargs)
                self._event("hermes.session.start", kwargs, handle=handle)
            elif hook_name == "on_session_end":
                self._event(
                    "hermes.turn.end", kwargs, handle=self._parent_handle(kwargs)
                )
                self._close_turn(kwargs)
                self._flush()
            elif hook_name == "on_session_finalize":
                self._event(
                    "hermes.session.finalize",
                    kwargs,
                    handle=self._session_handle(kwargs),
                )
                self._close_session(kwargs)
            elif hook_name == "on_session_reset":
                handle = self._ensure_session(kwargs)
                self._event("hermes.session.reset", kwargs, handle=handle)
            elif hook_name == "pre_api_request":
                self._pre_api_request(kwargs)
            elif hook_name == "post_api_request":
                self._post_api_request(kwargs)
            elif hook_name == "api_request_error":
                self._api_request_error(kwargs)
            elif hook_name == "pre_tool_call":
                self._pre_tool_call(kwargs)
            elif hook_name == "post_tool_call":
                self._post_tool_call(kwargs)
            elif hook_name == "pre_approval_request":
                self._event(
                    "hermes.approval.request",
                    kwargs,
                    handle=self._parent_handle(kwargs),
                )
            elif hook_name == "post_approval_response":
                self._event(
                    "hermes.approval.response",
                    kwargs,
                    handle=self._parent_handle(kwargs),
                )
            elif hook_name == "subagent_start":
                self._subagent_start(kwargs)
            elif hook_name == "subagent_stop":
                self._subagent_stop(kwargs)

    def _pre_api_request(self, kwargs: dict[str, Any]) -> None:
        handle = self._parent_handle(kwargs)
        request = (
            kwargs.get("request") if isinstance(kwargs.get("request"), dict) else {}
        )
        body = request.get("body", request) if isinstance(request, dict) else request
        llm_request = self.nemo_flow.LLMRequest(
            {}, _jsonable(body, self.settings.max_payload_chars)
        )
        span = self.nemo_flow.llm.call(
            str(kwargs.get("provider") or kwargs.get("model") or "hermes-llm"),
            llm_request,
            handle=handle,
            model_name=str(kwargs.get("model") or ""),
            metadata=self._metadata(kwargs, status="started"),
            timestamp=_timestamp(kwargs.get("started_at")),
        )
        self._api_spans[self._api_key(kwargs)] = span

    def _post_api_request(self, kwargs: dict[str, Any]) -> None:
        span = self._api_spans.pop(self._api_key(kwargs), None)
        if span is None:
            self._event(
                "hermes.llm.unmatched_end", kwargs, handle=self._parent_handle(kwargs)
            )
            return
        self.nemo_flow.llm.call_end(
            span,
            _jsonable(kwargs.get("response") or {}, self.settings.max_payload_chars),
            metadata=self._metadata(kwargs, status="ok"),
            timestamp=_timestamp(kwargs.get("ended_at")),
        )

    def _api_request_error(self, kwargs: dict[str, Any]) -> None:
        span = self._api_spans.pop(self._api_key(kwargs), None)
        payload = {
            "error": kwargs.get("error"),
            "status_code": kwargs.get("status_code"),
            "retry_count": kwargs.get("retry_count"),
            "max_retries": kwargs.get("max_retries"),
            "retryable": kwargs.get("retryable"),
            "reason": kwargs.get("reason"),
        }
        if span is None:
            self._event(
                "hermes.llm.error",
                {**kwargs, "error_payload": payload},
                handle=self._parent_handle(kwargs),
            )
            return
        self.nemo_flow.llm.call_end(
            span,
            _jsonable(payload, self.settings.max_payload_chars),
            metadata=self._metadata(kwargs, status="error"),
            timestamp=_timestamp(kwargs.get("ended_at")),
        )

    def _pre_tool_call(self, kwargs: dict[str, Any]) -> None:
        span = self.nemo_flow.tools.call(
            str(kwargs.get("tool_name") or "unknown_tool"),
            _jsonable(kwargs.get("args") or {}, self.settings.max_payload_chars),
            handle=self._parent_handle(kwargs),
            metadata=self._metadata(kwargs, status="started"),
            tool_call_id=str(kwargs.get("tool_call_id") or "") or None,
            timestamp=_timestamp(kwargs.get("started_at")),
        )
        self._tool_spans[self._tool_key(kwargs)] = span

    def _post_tool_call(self, kwargs: dict[str, Any]) -> None:
        span = self._tool_spans.pop(self._tool_key(kwargs), None)
        if span is None:
            self._event(
                "hermes.tool.unmatched_end", kwargs, handle=self._parent_handle(kwargs)
            )
            return
        self.nemo_flow.tools.call_end(
            span,
            _jsonable(
                _parse_json_string(kwargs.get("result")),
                self.settings.max_payload_chars,
            ),
            metadata=self._metadata(kwargs, status=str(kwargs.get("status") or "ok")),
            timestamp=_timestamp(kwargs.get("ended_at")),
        )

    def _subagent_start(self, kwargs: dict[str, Any]) -> None:
        parent = self._parent_handle({
            "session_id": kwargs.get("parent_session_id"),
            "turn_id": kwargs.get("parent_turn_id"),
            "task_id": kwargs.get("parent_subagent_id"),
        })
        child_key = self._subagent_key(kwargs)
        handle = self.nemo_flow.scope.push(
            f"hermes-subagent-{child_key}",
            self.nemo_flow.ScopeType.Agent,
            handle=parent,
            metadata=self._metadata(kwargs, status="started"),
            input=_jsonable(
                {"goal": kwargs.get("task_goal"), "role": kwargs.get("child_role")},
                self.settings.max_payload_chars,
            ),
        )
        self._subagent_scopes[child_key] = handle

    def _subagent_stop(self, kwargs: dict[str, Any]) -> None:
        child_key = self._subagent_key(kwargs)
        handle = self._subagent_scopes.pop(child_key, None)
        if handle is None:
            self._event(
                "hermes.subagent.unmatched_stop",
                kwargs,
                handle=self._parent_handle(kwargs),
            )
            return
        self._pop_scope(
            handle,
            output={
                "summary": kwargs.get("child_summary"),
                "status": kwargs.get("child_status"),
                "error": kwargs.get("error"),
            },
            timestamp=None,
        )

    def _ensure_session(self, kwargs: dict[str, Any]) -> Any:
        key = _session_key(kwargs)
        handle = self._session_scopes.get(key)
        if handle is not None:
            self._ensure_atif(key, kwargs)
            return handle

        self._ensure_atif(key, kwargs)
        handle = self.nemo_flow.scope.push(
            f"hermes-session-{key}",
            self.nemo_flow.ScopeType.Agent,
            metadata=self._metadata(kwargs, status="started"),
            input={"session_id": key, "platform": _str(kwargs.get("platform"))},
        )
        self._session_scopes[key] = handle
        return handle

    def _ensure_turn(self, kwargs: dict[str, Any]) -> Any:
        turn_id = _str(kwargs.get("turn_id"))
        if not turn_id:
            return self._ensure_session(kwargs)
        key = f"{_session_key(kwargs)}:{turn_id}"
        handle = self._turn_scopes.get(key)
        if handle is not None:
            return handle

        handle = self.nemo_flow.scope.push(
            f"hermes-turn-{_safe_filename(turn_id)}",
            self.nemo_flow.ScopeType.Function,
            handle=self._ensure_session(kwargs),
            metadata=self._metadata(kwargs, status="started"),
            input={"turn_id": turn_id, "task_id": _str(kwargs.get("task_id"))},
        )
        self._turn_scopes[key] = handle
        return handle

    def _parent_handle(self, kwargs: dict[str, Any]) -> Any:
        return (
            self._ensure_turn(kwargs)
            if kwargs.get("turn_id")
            else self._ensure_session(kwargs)
        )

    def _session_handle(self, kwargs: dict[str, Any]) -> Any | None:
        return self._session_scopes.get(_session_key(kwargs))

    def _close_turn(self, kwargs: dict[str, Any]) -> None:
        turn_id = _str(kwargs.get("turn_id"))
        if not turn_id:
            return
        handle = self._turn_scopes.pop(f"{_session_key(kwargs)}:{turn_id}", None)
        if handle is not None:
            self._pop_scope(
                handle,
                output={
                    "completed": kwargs.get("completed"),
                    "interrupted": kwargs.get("interrupted"),
                },
                timestamp=_timestamp(kwargs.get("ended_at")),
            )

    def _close_session(self, kwargs: dict[str, Any]) -> None:
        key = _session_key(kwargs)
        for turn_key, turn_handle in list(self._turn_scopes.items()):
            if turn_key.startswith(f"{key}:"):
                self._turn_scopes.pop(turn_key, None)
                self._pop_scope(turn_handle, output={"session_finalized": True})
        handle = self._session_scopes.pop(key, None)
        if handle is not None:
            self._pop_scope(
                handle,
                output={
                    "reason": kwargs.get("reason"),
                    "completed": kwargs.get("completed"),
                    "interrupted": kwargs.get("interrupted"),
                },
                timestamp=_timestamp(kwargs.get("ended_at")),
            )
        self._export_atif(key)
        self._flush()

    def _pop_scope(
        self, handle: Any, *, output: Any = None, timestamp: datetime | None = None
    ) -> None:
        try:
            self.nemo_flow.scope.pop(
                handle,
                output=_jsonable(output, self.settings.max_payload_chars),
                timestamp=timestamp,
            )
        except Exception as exc:
            logger.debug("NeMo-Flow scope pop failed: %s", exc, exc_info=True)

    def _event(
        self, name: str, kwargs: dict[str, Any], *, handle: Any | None = None
    ) -> None:
        self.nemo_flow.scope.event(
            name,
            handle=handle,
            data=_jsonable(_event_data(kwargs), self.settings.max_payload_chars),
            metadata=self._metadata(kwargs),
            timestamp=_timestamp(kwargs.get("started_at") or kwargs.get("ended_at")),
        )

    def _metadata(
        self, kwargs: dict[str, Any], *, status: str | None = None
    ) -> dict[str, Any]:
        metadata = {
            "source": "hermes-agent",
            "telemetry_schema_version": kwargs.get("telemetry_schema_version"),
            "session_id": kwargs.get("session_id")
            or kwargs.get("parent_session_id")
            or kwargs.get("child_session_id"),
            "task_id": kwargs.get("task_id"),
            "turn_id": kwargs.get("turn_id") or kwargs.get("parent_turn_id"),
            "api_request_id": kwargs.get("api_request_id"),
            "tool_call_id": kwargs.get("tool_call_id")
            or kwargs.get("parent_tool_call_id"),
            "model": kwargs.get("model"),
            "provider": kwargs.get("provider"),
            "base_url": kwargs.get("base_url"),
            "api_mode": kwargs.get("api_mode"),
            "api_call_count": kwargs.get("api_call_count"),
            "api_duration": kwargs.get("api_duration"),
            "started_at": kwargs.get("started_at"),
            "ended_at": kwargs.get("ended_at"),
            "finish_reason": kwargs.get("finish_reason"),
            "response_model": kwargs.get("response_model"),
            "usage": kwargs.get("usage"),
            "assistant_content_chars": kwargs.get("assistant_content_chars"),
            "assistant_tool_call_count": kwargs.get("assistant_tool_call_count"),
            "duration_ms": kwargs.get("duration_ms"),
            "platform": kwargs.get("platform"),
            "status": status or kwargs.get("status"),
            "error_type": kwargs.get("error_type"),
            "error_message": kwargs.get("error_message"),
            "reason": kwargs.get("reason"),
            "choice": kwargs.get("choice"),
            "parent_session_id": kwargs.get("parent_session_id"),
            "child_session_id": kwargs.get("child_session_id"),
            "parent_turn_id": kwargs.get("parent_turn_id"),
            "parent_tool_call_id": kwargs.get("parent_tool_call_id"),
            "parent_subagent_id": kwargs.get("parent_subagent_id"),
            "subagent_id": kwargs.get("subagent_id"),
            "child_role": kwargs.get("child_role"),
            "depth": kwargs.get("depth"),
            "child_status": kwargs.get("child_status"),
            "exit_reason": kwargs.get("exit_reason"),
            "tokens": kwargs.get("tokens"),
        }
        return _jsonable(
            {k: v for k, v in metadata.items() if v not in (None, "")},
            self.settings.max_payload_chars,
        )

    def _ensure_atif(self, session_key: str, kwargs: dict[str, Any]) -> None:
        if self._plugin_configured or not self.settings.atif_enabled:
            return
        if session_key in self._atif:
            return
        exporter = self.nemo_flow.AtifExporter(
            session_key,
            self.settings.atif_agent_name,
            self.settings.atif_agent_version,
            model_name=str(kwargs.get("model") or "unknown"),
            extra={"source": "hermes-agent"},
        )
        subscriber_name = f"hermes-nemo-flow-atif-{_safe_filename(session_key)}"
        exporter.register(subscriber_name)
        self._atif[session_key] = _AtifState(
            exporter=exporter, subscriber_name=subscriber_name
        )

    def _export_atif(self, session_key: str) -> None:
        state = self._atif.pop(session_key, None)
        if state is None:
            return
        try:
            if self.settings.atif_output_directory:
                output_dir = Path(self.settings.atif_output_directory).expanduser()
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = self.settings.atif_filename_template.format(
                    session_id=_safe_filename(session_key)
                )
                (output_dir / filename).write_text(
                    state.exporter.export_json(), encoding="utf-8"
                )
        finally:
            try:
                state.exporter.deregister(state.subscriber_name)
            except Exception:
                pass

    def _flush(self) -> None:
        if self._atof_exporter is not None:
            try:
                self._atof_exporter.force_flush()
            except Exception:
                pass

    def _api_key(self, kwargs: dict[str, Any]) -> str:
        api_request_id = _str(kwargs.get("api_request_id"))
        if api_request_id:
            return api_request_id
        return f"{_session_key(kwargs)}:{kwargs.get('api_call_count') or 'api'}"

    def _tool_key(self, kwargs: dict[str, Any]) -> str:
        tool_call_id = _str(kwargs.get("tool_call_id"))
        if tool_call_id:
            return tool_call_id
        return f"{_session_key(kwargs)}:{kwargs.get('turn_id') or ''}:{kwargs.get('tool_name') or 'tool'}"

    def _subagent_key(self, kwargs: dict[str, Any]) -> str:
        return _str(
            kwargs.get("subagent_id")
            or kwargs.get("child_session_id")
            or kwargs.get("task_index")
            or "unknown"
        )


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _session_key(kwargs: dict[str, Any]) -> str:
    for key in (
        "session_id",
        "parent_session_id",
        "child_session_id",
        "session_key",
        "task_id",
    ):
        value = _str(kwargs.get(key)).strip()
        if value:
            return value
    return f"thread-{threading.get_ident()}"


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _event_data(kwargs: dict[str, Any]) -> dict[str, Any]:
    omitted = {"request", "response", "args", "result"}
    return {k: v for k, v in kwargs.items() if k not in omitted}


def _parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _jsonable(value: Any, max_chars: int, *, depth: int = 0) -> Any:
    if depth > 8:
        return f"<{type(value).__name__} depth limit>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return (
            value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"
        )
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 200:
                out["_truncated_items"] = len(value) - 200
                break
            key_str = str(key)
            out[key_str] = (
                "<redacted>"
                if _sensitive_key(key_str)
                else _jsonable(item, max_chars, depth=depth + 1)
            )
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = [_jsonable(item, max_chars, depth=depth + 1) for item in seq[:200]]
        if len(seq) > 200:
            out.append({"_truncated_items": len(seq) - 200})
        return out
    if isinstance(value, SimpleNamespace):
        return _jsonable(vars(value), max_chars, depth=depth + 1)
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"), max_chars, depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            public = {
                k: v for k, v in vars(value).items() if not str(k).startswith("_")
            }
            return _jsonable(public, max_chars, depth=depth + 1)
        except Exception:
            pass
    return str(value)[:max_chars]


def _sensitive_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in _SAFE_TOKEN_ACCOUNTING_KEYS:
        return False
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return safe or "session"
