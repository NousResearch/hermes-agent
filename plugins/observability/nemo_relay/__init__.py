"""nemo_relay — optional Hermes plugin for NeMo Relay observability."""

from __future__ import annotations

import atexit
import asyncio
import inspect
import json
import logging
import os
import threading
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agent import relay_runtime

logger = logging.getLogger(__name__)

_INIT_FAILED = object()
_LOCK = threading.RLock()
_RUNTIMES: dict[str, "_Runtime | object"] = {}
_SESSION_INITIALIZER_NAME = "hermes.nemo_relay.rich_observability"


@dataclass
class _SessionState:
    session_id: str
    relay_session: relay_runtime.RelaySession | None = None
    handle: Any = None
    atif_exporter: Any = None
    atif_subscriber_name: str = ""
    is_embedded_subagent: bool = False
    parent_session_id: str = ""


@dataclass
class _SubagentContext:
    parent_session_id: str
    metadata: dict[str, Any]


@dataclass
class _Settings:
    plugins_toml_path: str = ""
    plugins_config: dict[str, Any] | None = None
    dynamic_plugins: list[dict[str, Any]] = field(default_factory=list)
    atof_enabled: bool = False
    atof_output_directory: str = ""
    atof_filename: str = "hermes-atof.jsonl"
    atof_mode: str = "append"
    atif_enabled: bool = False
    atif_output_directory: str = ""
    atif_filename_template: str = "hermes-atif-{session_id}.json"
    atif_subagent_export_mode: str = "embedded"
    atif_agent_name: str = "Hermes Agent"
    atif_agent_version: str = "unknown"
    atif_model_name: str = "unknown"


class _Runtime:
    def __init__(
        self,
        nemo_relay: Any,
        settings: _Settings,
        host: relay_runtime.RelayRuntime,
    ) -> None:
        self.nemo_relay = nemo_relay
        self.settings = settings
        self.host = host
        self._sessions_lock = threading.RLock()
        self.sessions: dict[str, _SessionState] = {}
        self.subagent_contexts: dict[str, _SubagentContext] = {}
        self.atof_exporter: Any = None
        self._atof_subscriber_name = f"hermes.nemo_relay.atof.{self.host.runtime_id}"
        self._plugin_activation: Any = None
        self._shutdown_registered = False
        self._plugin_config_initialized = self._configure_plugins_toml()
        self._plugin_config_needs_reinit = False
        if not self._plugin_config_initialized:
            self._activate_direct_fallbacks()

    def _configure_plugins_toml(self) -> bool:
        if not self.settings.plugins_config:
            return False
        plugin_mod = getattr(self.nemo_relay, "plugin", None)
        if plugin_mod is None:
            return False
        plugin_config = _static_plugin_config(self.settings.plugins_config)
        if self.settings.dynamic_plugins:
            activate_dynamic = getattr(plugin_mod, "activate_dynamic_plugins", None)
            if callable(activate_dynamic):
                try:
                    self._ensure_plugin_config_output_dirs(plugin_config)
                    self._plugin_activation = _resolve_awaitable(
                        activate_dynamic(plugin_config, self.settings.dynamic_plugins)
                    )
                    self._ensure_shutdown_registered()
                    return True
                except Exception as exc:
                    logger.warning(
                        "NeMo Relay dynamic plugin activation failed; continuing with static "
                        "observability only: %s",
                        exc,
                    )
            else:
                logger.warning(
                    "NeMo Relay dynamic plugins require a binding that exposes "
                    "plugin.activate_dynamic_plugins (available in NeMo Relay 0.6+). "
                    "Continuing with static observability only."
                )
        initialize = getattr(plugin_mod, "initialize", None)
        if not callable(initialize):
            return False
        try:
            self._ensure_plugin_config_output_dirs(plugin_config)
            _resolve_awaitable(initialize(plugin_config))
            return True
        except Exception as exc:
            logger.debug("NeMo Relay plugins.toml init failed: %s", exc, exc_info=True)
            return False

    def _ensure_shutdown_registered(self) -> None:
        if self._shutdown_registered:
            return
        atexit.register(self.shutdown)
        self._shutdown_registered = True

    def _clear_plugins_toml(self) -> None:
        if not self._plugin_config_initialized:
            return
        failures: list[str] = []
        if self._plugin_activation is not None:
            activation = self._plugin_activation
            try:
                _flush_relay_subscribers(self.nemo_relay)
            except Exception as exc:
                failures.append(f"subscriber flush failed: {exc}")

            close = getattr(activation, "close", None)
            if callable(close):
                try:
                    _resolve_awaitable(close())
                except Exception as exc:
                    failures.append(f"dynamic plugin activation close failed: {exc}")
                finally:
                    # Retain the owned activation through the complete close
                    # attempt. The binding transitions it to a terminal state
                    # before its awaitable resolves, including error results.
                    self._plugin_activation = None
                    self._plugin_config_initialized = False
                    self._plugin_config_needs_reinit = bool(self.settings.plugins_config)
            else:
                failures.append("dynamic plugin activation has no close method")
        else:
            try:
                plugin_mod = getattr(self.nemo_relay, "plugin", None)
                clear = getattr(plugin_mod, "clear", None)
                if callable(clear):
                    _resolve_awaitable(clear())
            except Exception as exc:
                failures.append(f"static plugin configuration clear failed: {exc}")
            finally:
                self._plugin_config_initialized = False
                self._plugin_config_needs_reinit = bool(self.settings.plugins_config)

        if failures:
            raise RuntimeError("; ".join(failures))

    def _activate_direct_fallbacks(self) -> None:
        self._plugin_config_needs_reinit = False
        self._configure_atof()

    def _maybe_reinitialize_plugins_toml(self) -> None:
        if not self._plugin_config_needs_reinit or self._plugin_config_initialized:
            return
        self._plugin_config_initialized = self._configure_plugins_toml()
        if not self._plugin_config_initialized:
            self._activate_direct_fallbacks()
            return
        self._clear_atof()
        self._plugin_config_needs_reinit = False

    def _plugins_toml_owns_exporter(self, exporter_name: str) -> bool:
        return self._plugin_config_initialized and _observability_exporter_enabled(
            self.settings.plugins_config,
            exporter_name,
        )

    def _ensure_plugin_config_output_dirs(self, config: dict[str, Any]) -> None:
        for component in config.get("components", []):
            if not isinstance(component, dict):
                continue
            if component.get("kind") != "observability":
                continue
            if component.get("enabled") is False:
                continue
            component_config = component.get("config")
            if not isinstance(component_config, dict):
                continue
            for exporter_name in ("atof", "atif"):
                exporter_config = component_config.get(exporter_name)
                if not isinstance(exporter_config, dict):
                    continue
                output_directory = exporter_config.get("output_directory")
                if isinstance(output_directory, str) and output_directory.strip():
                    Path(output_directory).mkdir(parents=True, exist_ok=True)

    def _configure_atof(self) -> None:
        if not self.settings.atof_enabled or self.atof_exporter is not None:
            return
        config = self.nemo_relay.AtofExporterConfig()
        if self.settings.atof_output_directory:
            Path(self.settings.atof_output_directory).mkdir(parents=True, exist_ok=True)
            config.output_directory = self.settings.atof_output_directory
        config.filename = self.settings.atof_filename
        if self.settings.atof_mode.lower() == "overwrite":
            config.mode = self.nemo_relay.AtofExporterMode.Overwrite
        else:
            config.mode = self.nemo_relay.AtofExporterMode.Append
        self.atof_exporter = self.nemo_relay.AtofExporter(config)
        self.atof_exporter.register(self._atof_subscriber_name)

    def _clear_atof(self) -> None:
        if self.atof_exporter is None:
            return
        deregister = getattr(self.atof_exporter, "deregister", None)
        if callable(deregister):
            try:
                deregister(self._atof_subscriber_name)
            except Exception:
                logger.debug("NeMo Relay ATOF deregister failed", exc_info=True)
        self.atof_exporter = None

    def prepare_session(self, kwargs: dict[str, Any]) -> _SessionState:
        """Register per-session subscribers without opening the core scope."""
        session_id = _session_id(kwargs)
        with self._sessions_lock:
            self._maybe_reinitialize_plugins_toml()
            state = self.sessions.get(session_id)
            if state is not None:
                return state

            state = _SessionState(session_id=session_id)
            if self.settings.atif_enabled and not self._plugins_toml_owns_exporter("atif"):
                state.atif_exporter = self.nemo_relay.AtifExporter(
                    session_id,
                    self.settings.atif_agent_name,
                    self.settings.atif_agent_version,
                    model_name=str(kwargs.get("model") or self.settings.atif_model_name),
                    extra={
                        "source": "hermes-agent",
                        "plugin": "observability/nemo_relay",
                    },
                )
                state.atif_subscriber_name = (
                    f"hermes.nemo_relay.atif.{self.host.runtime_id}.{session_id}"
                )
                state.atif_exporter.register(state.atif_subscriber_name)
            self.sessions[session_id] = state
            return state

    def ensure_session(self, kwargs: dict[str, Any]) -> _SessionState:
        state = self.prepare_session(kwargs)
        if state.relay_session is not None:
            return state

        rich_metadata = _metadata(kwargs)
        with self._sessions_lock:
            subagent_context = self.subagent_contexts.get(state.session_id)
        if subagent_context is not None:
            rich_metadata = {**rich_metadata, **subagent_context.metadata}
        relay_session = self.host.ensure_session(
            kwargs,
            data={"session_id": state.session_id},
            metadata=rich_metadata,
        )
        if relay_session is None:
            raise RuntimeError("Hermes core Relay session is unavailable")
        state.relay_session = relay_session
        state.handle = relay_session.handle
        if subagent_context is not None:
            state.is_embedded_subagent = True
            state.parent_session_id = subagent_context.parent_session_id
        return state

    def run_in_session(
        self,
        state: _SessionState,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if state.relay_session is None:
            raise RuntimeError("Hermes core Relay session is unavailable")
        return self.host.run_in_session(
            state.relay_session,
            callback,
            *args,
            **kwargs,
        )

    def export_atif(self, state: _SessionState) -> None:
        if not self.settings.atif_enabled or state.atif_exporter is None:
            return
        if state.is_embedded_subagent and self.settings.atif_subagent_export_mode != "all":
            return
        output_dir = self.settings.atif_output_directory
        if not output_dir:
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = self.settings.atif_filename_template.format(session_id=state.session_id)
        Path(output_dir, filename).write_text(state.atif_exporter.export_json(), encoding="utf-8")

    def close_session(
        self,
        kwargs: dict[str, Any],
        *,
        close_host: bool = True,
    ) -> None:
        session_id = _session_id(kwargs)
        with self._sessions_lock:
            self.subagent_contexts.pop(session_id, None)
            state = self.sessions.pop(session_id, None)
        if state is None:
            return
        failures: list[str] = []
        if close_host:
            try:
                self.host.close_session(kwargs)
            except Exception as exc:
                failures.append(f"core session close failed: {exc}")
        try:
            self.export_atif(state)
        except Exception as exc:
            failures.append(f"ATIF export failed: {exc}")
        if state.atif_exporter is not None and state.atif_subscriber_name:
            try:
                state.atif_exporter.deregister(state.atif_subscriber_name)
            except Exception as exc:
                failures.append(f"ATIF deregister failed: {exc}")
        with self._sessions_lock:
            if (
                self._plugin_config_initialized
                and self._plugin_activation is None
                and not self.sessions
            ):
                try:
                    self._clear_plugins_toml()
                except Exception as exc:
                    failures.append(f"plugin configuration clear failed: {exc}")
            elif (
                self.settings.plugins_config
                and self._plugin_activation is None
                and not self.sessions
            ):
                self._plugin_config_needs_reinit = True
        if failures:
            logger.warning(
                "NeMo Relay session %s teardown completed with errors: %s",
                session_id,
                "; ".join(failures),
            )

    def shutdown(self) -> None:
        """Close active sessions and the process-lifetime plugin activation."""
        failures: list[str] = []
        with self._sessions_lock:
            session_ids = list(self.sessions)
        for session_id in session_ids:
            try:
                self.close_session({"session_id": session_id, "reason": "runtime_shutdown"})
            except Exception as exc:
                failures.append(f"session {session_id} close failed: {exc}")
        if self._plugin_config_initialized:
            try:
                self._clear_plugins_toml()
            except Exception as exc:
                failures.append(f"plugin runtime close failed: {exc}")
        self._clear_atof()
        if self._shutdown_registered and self._plugin_activation is None:
            atexit.unregister(self.shutdown)
            self._shutdown_registered = False
        if failures:
            logger.warning(
                "NeMo Relay runtime shutdown completed with errors: %s",
                "; ".join(failures),
            )

    def mark(self, name: str, kwargs: dict[str, Any]) -> None:
        state = self.ensure_session(kwargs)
        self.run_in_session(
            state,
            self.nemo_relay.scope.event,
            name,
            handle=state.handle,
            data=_jsonable(kwargs),
            metadata=_metadata(kwargs),
        )

    def mark_subagent_start(self, kwargs: dict[str, Any]) -> None:
        parent_state = self.ensure_session(kwargs)
        metadata = _metadata(kwargs)
        child_session_id = _child_session_id(kwargs)
        if child_session_id:
            with self._sessions_lock:
                self.subagent_contexts[child_session_id] = _SubagentContext(
                    parent_session_id=parent_state.session_id,
                    metadata=_subagent_child_metadata(kwargs, metadata),
                )
        self.run_in_session(
            parent_state,
            self.nemo_relay.scope.event,
            "hermes.subagent.start",
            handle=parent_state.handle,
            data=_jsonable(kwargs),
            metadata=metadata,
        )

    def mark_subagent_stop(self, kwargs: dict[str, Any]) -> None:
        child_session_id = _child_session_id(kwargs)
        if child_session_id:
            self.close_session(
                {"session_id": child_session_id},
                close_host=False,
            )
            with self._sessions_lock:
                self.subagent_contexts.pop(child_session_id, None)
        self.mark("hermes.subagent.stop", kwargs)

def register(ctx) -> None:
    relay_runtime.SESSION_COORDINATOR.register_session_initializer(
        _SESSION_INITIALIZER_NAME,
        _prepare_core_session,
    )
    # Activate dynamic plugins before Hermes installs the managed execution
    # boundaries that invoke their interceptors.
    if _load_settings().dynamic_plugins:
        _get_runtime()
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_approval_request", on_pre_approval_request)
    ctx.register_hook("post_approval_response", on_post_approval_response)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)


def on_session_start(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.ensure_session(kwargs))


def on_session_end(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: (runtime.mark("hermes.session.end", kwargs), runtime.export_atif(runtime.ensure_session(kwargs))))


def on_session_finalize(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.close_session(kwargs, close_host=False))


def on_session_reset(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.close_session(kwargs, close_host=False))


def on_pre_llm_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.turn.start", kwargs))


def on_post_llm_call(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.turn.end", kwargs))


def on_pre_approval_request(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.approval.request", kwargs))


def on_post_approval_response(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark("hermes.approval.response", kwargs))


def on_subagent_start(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark_subagent_start(kwargs))


def on_subagent_stop(**kwargs: Any) -> None:
    runtime = _get_runtime()
    if runtime is not None:
        _safe(lambda: runtime.mark_subagent_stop(kwargs))


def _prepare_core_session(
    host: relay_runtime.RelayRuntime,
    context: dict[str, Any],
) -> None:
    """Register rich subscribers before core creates the conversation scope."""
    runtime = _get_runtime(
        profile_key=str(context.get("profile_key") or host.profile_key),
        host=host,
    )
    if runtime is not None:
        runtime.prepare_session(context)


def _get_runtime(
    *,
    profile_key: str | None = None,
    host: relay_runtime.RelayRuntime | None = None,
) -> Optional[_Runtime]:
    profile_key = profile_key or relay_runtime.current_profile_key()
    with _LOCK:
        runtime = _RUNTIMES.get(profile_key)
        if runtime is _INIT_FAILED:
            return None
        if isinstance(runtime, _Runtime):
            if host is None or runtime.host is host:
                return runtime
            runtime.shutdown()
            _RUNTIMES.pop(profile_key, None)
        try:
            resolved_host = host or relay_runtime.get_runtime(profile_key=profile_key)
            if resolved_host is None:
                raise RuntimeError("Hermes core Relay runtime is unavailable")
            runtime = _Runtime(
                nemo_relay=resolved_host.relay,
                settings=_load_settings(),
                host=resolved_host,
            )
        except Exception as exc:
            logger.debug("NeMo Relay plugin disabled: init failed: %s", exc, exc_info=True)
            _RUNTIMES[profile_key] = _INIT_FAILED
            return None
        _RUNTIMES[profile_key] = runtime
        return runtime


def _load_settings() -> _Settings:
    plugins_toml_path = _env("HERMES_NEMO_RELAY_PLUGINS_TOML")
    plugins_config = _load_plugins_config(plugins_toml_path)
    return _Settings(
        plugins_toml_path=plugins_toml_path,
        plugins_config=plugins_config,
        dynamic_plugins=_dynamic_plugin_specs(plugins_config, plugins_toml_path),
        atof_enabled=_env_bool("HERMES_NEMO_RELAY_ATOF_ENABLED"),
        atof_output_directory=_env("HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY"),
        atof_filename=_env("HERMES_NEMO_RELAY_ATOF_FILENAME") or "hermes-atof.jsonl",
        atof_mode=_env("HERMES_NEMO_RELAY_ATOF_MODE") or "append",
        atif_enabled=_env_bool("HERMES_NEMO_RELAY_ATIF_ENABLED"),
        atif_output_directory=_env("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY"),
        atif_filename_template=_env("HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE") or "hermes-atif-{session_id}.json",
        atif_subagent_export_mode=_atif_subagent_export_mode(),
        atif_agent_name=_env("HERMES_NEMO_RELAY_ATIF_AGENT_NAME") or "Hermes Agent",
        atif_agent_version=_env("HERMES_NEMO_RELAY_ATIF_AGENT_VERSION") or "unknown",
        atif_model_name=_env("HERMES_NEMO_RELAY_ATIF_MODEL_NAME") or "unknown",
    )


def _static_plugin_config(plugins_config: dict[str, Any]) -> dict[str, Any]:
    """Return Relay's base config without embedding- or gateway-host fields."""
    return {
        key: value
        for key, value in plugins_config.items()
        if key not in {"dynamic_plugins", "plugins"}
    }


def _dynamic_plugin_specs(
    plugins_config: dict[str, Any] | None,
    plugins_toml_path: str = "",
) -> list[dict[str, Any]]:
    if not isinstance(plugins_config, dict):
        return []

    raw_specs = plugins_config.get("dynamic_plugins")
    plugins_section = plugins_config.get("plugins")
    if plugins_section is not None:
        if not isinstance(plugins_section, dict):
            logger.error(
                "Invalid NeMo Relay plugins config: expected [plugins] to be an object; "
                "no dynamic plugins will be activated. Continuing with static "
                "observability only."
            )
            return []
        if plugins_section:
            logger.error(
                "Hermes cannot activate Relay gateway [[plugins.dynamic]] records because "
                "the Python binding does not expose the CLI lifecycle resolver for "
                "enablement, trust policy, and worker environments. Use Hermes-owned "
                "[[dynamic_plugins]] activation specs instead; no dynamic plugins will be "
                "activated. Continuing with static observability only."
            )
            return []
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        logger.warning(
            "Ignoring invalid NeMo Relay dynamic_plugins config: expected an array of plugin specs"
        )
        return []

    specs: list[dict[str, Any]] = []
    invalid = False
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            logger.warning(
                "Invalid NeMo Relay dynamic_plugins[%d]: expected an object", index
            )
            invalid = True
            continue
        plugin_id = raw_spec.get("plugin_id")
        kind = raw_spec.get("kind")
        manifest_ref = raw_spec.get("manifest_ref")
        config = raw_spec.get("config", {})
        environment_ref = raw_spec.get("environment_ref")
        if not isinstance(plugin_id, str) or not plugin_id.strip():
            logger.warning(
                "Invalid NeMo Relay dynamic_plugins[%d]: plugin_id is required", index
            )
            invalid = True
            continue
        if kind not in {"rust_dynamic", "worker"}:
            logger.warning(
                "Invalid NeMo Relay dynamic_plugins[%d]: kind must be rust_dynamic or worker",
                index,
            )
            invalid = True
            continue
        if not isinstance(manifest_ref, str) or not manifest_ref.strip():
            logger.warning(
                "Invalid NeMo Relay dynamic_plugins[%d]: manifest_ref is required", index
            )
            invalid = True
            continue
        if not isinstance(config, dict):
            logger.warning(
                "Invalid NeMo Relay dynamic_plugins[%d]: config must be an object", index
            )
            invalid = True
            continue
        if environment_ref is not None and (
            not isinstance(environment_ref, str) or not environment_ref.strip()
        ):
            logger.warning(
                "Invalid NeMo Relay dynamic_plugins[%d]: environment_ref must be a "
                "non-empty string",
                index,
            )
            invalid = True
            continue
        spec: dict[str, Any] = {
            "plugin_id": plugin_id.strip(),
            "kind": kind,
            "manifest_ref": _config_relative_path(manifest_ref.strip(), plugins_toml_path),
            "config": config,
        }
        if environment_ref is not None:
            spec["environment_ref"] = _config_relative_path(
                environment_ref.strip(), plugins_toml_path
            )
        specs.append(spec)
    if invalid:
        logger.error(
            "NeMo Relay dynamic plugin configuration is invalid; no dynamic plugins "
            "will be activated. Continuing with static observability only."
        )
        return []
    return specs


def _config_relative_path(value: str, plugins_toml_path: str) -> str:
    """Resolve a plugin path relative to its physical ``plugins.toml`` file."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    config_path = Path(plugins_toml_path) if plugins_toml_path else Path.cwd() / "plugins.toml"
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return os.path.abspath(config_path.parent / path)


def _flush_relay_subscribers(nemo_relay: Any) -> None:
    subscribers = getattr(nemo_relay, "subscribers", None)
    flush = getattr(subscribers, "flush", None)
    if callable(flush):
        flush()


def _load_plugins_config(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("NeMo Relay plugins.toml load failed: %s", exc, exc_info=True)
        return None


def _enabled_component_config(
    plugins_config: dict[str, Any] | None,
    kind: str,
) -> dict[str, Any] | None:
    if not isinstance(plugins_config, dict):
        return None
    components = plugins_config.get("components")
    if not isinstance(components, list):
        return None
    for component in components:
        if not isinstance(component, dict):
            continue
        if component.get("kind") != kind or not component.get("enabled", True):
            continue
        config = component.get("config")
        return config if isinstance(config, dict) else {}
    return None


def _observability_exporter_enabled(
    plugins_config: dict[str, Any] | None,
    exporter_name: str,
) -> bool:
    observability_config = _enabled_component_config(plugins_config, "observability")
    if not isinstance(observability_config, dict):
        return False
    exporter_config = observability_config.get(exporter_name)
    if not isinstance(exporter_config, dict):
        return False
    return exporter_config.get("enabled", True) is not False


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _atif_subagent_export_mode() -> str:
    mode = _env("HERMES_NEMO_RELAY_ATIF_SUBAGENT_EXPORT_MODE").lower()
    return "all" if mode == "all" else "embedded"


def _env_bool(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("session_id") or kwargs.get("parent_session_id") or "default")


def _child_session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("child_session_id") or "")


def _subagent_child_metadata(
    kwargs: dict[str, Any],
    parent_metadata: dict[str, Any],
) -> dict[str, Any]:
    child_session_id = _child_session_id(kwargs)
    metadata = {
        "session_id": child_session_id,
        "trajectory_id": child_session_id,
        "nemo_relay_scope_role": "subagent",
    }
    for target, source in (
        ("subagent_id", "child_subagent_id"),
        ("child_session_id", "child_session_id"),
        ("child_subagent_id", "child_subagent_id"),
        ("child_role", "child_role"),
        ("parent_session_id", "parent_session_id"),
        ("parent_turn_id", "parent_turn_id"),
        ("parent_subagent_id", "parent_subagent_id"),
        ("parent_trajectory_id", "parent_trajectory_id"),
        ("telemetry_schema_version", "telemetry_schema_version"),
    ):
        value = parent_metadata.get(source)
        if value is not None:
            metadata[target] = value
    return metadata


def _metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "telemetry_schema_version",
        "session_id",
        "platform",
        "task_id",
        "turn_id",
        "api_request_id",
        "tool_call_id",
        "parent_session_id",
        "parent_turn_id",
        "parent_subagent_id",
        "child_session_id",
        "child_subagent_id",
        "child_role",
        "child_status",
        "provider",
        "model",
        "api_mode",
        "status",
        "reason",
    )
    metadata = {
        key: _jsonable(kwargs[key])
        for key in keys
        if key in kwargs and kwargs[key] is not None
    }
    if "session_id" in metadata:
        metadata.setdefault("trajectory_id", metadata["session_id"])
    if "parent_session_id" in metadata:
        metadata.setdefault("parent_trajectory_id", metadata["parent_session_id"])
    if "child_session_id" in metadata:
        metadata.setdefault("child_trajectory_id", metadata["child_session_id"])
    return metadata


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        if hasattr(value, "model_dump"):
            return _jsonable(value.model_dump(mode="json"))
    except Exception:
        pass
    try:
        if hasattr(value, "__dict__"):
            return _jsonable(vars(value))
    except Exception:
        pass
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _safe(fn) -> None:
    try:
        fn()
    except Exception as exc:
        logger.debug("NeMo Relay hook handling failed: %s", exc, exc_info=True)


def _resolve_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(value)
        except BaseException as exc:  # pragma: no cover - re-raised below
            error["exc"] = exc

    thread = threading.Thread(
        target=_runner,
        name="hermes-nemo-relay-awaitable",
        daemon=True,
    )
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def reset_for_tests() -> None:
    relay_runtime.SESSION_COORDINATOR.unregister_session_initializer(
        _SESSION_INITIALIZER_NAME
    )
    with _LOCK:
        runtimes = list(_RUNTIMES.values())
        _RUNTIMES.clear()
    for runtime in runtimes:
        if isinstance(runtime, _Runtime):
            runtime.shutdown()
