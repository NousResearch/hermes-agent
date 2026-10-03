"""Opt-in required plugin authority at the core Kanban transition boundary.

Only the explicit allow decision admits a transition. Optional lifecycle observers
remain independent and keep their existing best-effort delivery semantics.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
import logging
import threading
from typing import Any

from hermes_cli.config_effective import load_user_config_effective

logger = logging.getLogger(__name__)
_TIMEOUT_SECONDS = 2.0
_CONFIG_KEY = "required_transition_admission_plugin"


def _configured_plugin_enabled(config: dict, provider: str, loaded: Any) -> bool:
    plugins = config.get("plugins", {})
    if not isinstance(plugins, dict):
        return False
    disabled = plugins.get("disabled", [])
    if not isinstance(disabled, list) or not all(isinstance(name, str) for name in disabled):
        return False
    names = {provider, loaded.manifest.name}
    if names.intersection(disabled):
        return False
    if loaded.manifest.source == "bundled" and loaded.manifest.kind == "backend":
        return True
    enabled = plugins.get("enabled")
    return isinstance(enabled, list) and bool(names.intersection(enabled))


@dataclass(frozen=True)
class Admission:
    provider: str | None
    binding: tuple[Any, object] | None = None
    loaded: Any = None
    status: str | None = None
    run_id: int | None = None


def admit(action: str, task_id: str, *, status: str, run_id: int | None,
          force: bool, previous: Admission | None = None) -> Admission | None:
    """No live DB handle or handoff content crosses the plugin boundary.

    A missing config key is the backwards-compatible opt-out; a present but
    malformed key, provider load failure, timeout, or invalid response denies.
    """
    try:
        config = load_user_config_effective(fail_closed=True)
        kanban = config.get("kanban", {})
        if not isinstance(kanban, dict):
            raise ValueError("invalid kanban config")
        if _CONFIG_KEY not in kanban:
            if previous is not None and previous.provider is not None:
                raise ValueError("required admission removed during transition")
            return Admission(None)
        provider = kanban[_CONFIG_KEY]
        if not isinstance(provider, str) or not provider.strip() or provider != provider.strip():
            raise ValueError("invalid required admission provider")
        if previous is not None and previous.provider not in (None, provider):
            raise ValueError("required admission provider changed")
        from hermes_cli import plugins
        manager = plugins.get_plugin_manager()
        manager.discover_and_load()
        with manager._discovery_lock:
            loaded = manager._plugins.get(provider)
            binding = manager._kanban_transition_admissions.get(provider)
            if (loaded is None or not loaded.enabled or loaded.error or binding is None
                    or not callable(binding[0]) or not _configured_plugin_enabled(config, provider, loaded)):
                raise ValueError("required admission provider unavailable")
            if previous is not None and previous.provider is not None:
                if (previous.loaded is not loaded or previous.binding is not binding
                        or previous.status != status or previous.run_id != run_id):
                    raise ValueError("required admission binding or run changed")
                return previous
            callback = binding[0]
            # Reuse the callback slot only after the previous invocation actually
            # returned; a timed-out plugin cannot accumulate unbounded workers.
            with manager._kanban_admission_lock:
                if provider in manager._kanban_admission_pending:
                    raise TimeoutError("required admission provider still running")
                manager._kanban_admission_pending.add(provider)
            response: list[Any] = []
            context = contextvars.copy_context()

            def call() -> None:
                try:
                    response.append(context.run(
                        callback, action=action, task_id=task_id, status=status,
                        run_id=run_id, force=force,
                    ))
                except BaseException as exc:
                    response.append(exc)
                finally:
                    with manager._kanban_admission_lock:
                        manager._kanban_admission_pending.discard(provider)

            worker = threading.Thread(target=call, daemon=True, name="kanban-admission")
            worker.start()
            worker.join(_TIMEOUT_SECONDS)
            if worker.is_alive() or len(response) != 1 or isinstance(response[0], BaseException):
                raise ValueError("required admission callback failed")
            # Registration drift (including a callback that re-registers itself)
            # never turns a prior acknowledgement into authority.
            if manager._plugins.get(provider) is not loaded or manager._kanban_transition_admissions.get(provider) is not binding:
                raise ValueError("required admission registration changed")
            current = load_user_config_effective(fail_closed=True)
            if (not isinstance(current.get("kanban"), dict)
                    or current["kanban"].get(_CONFIG_KEY) != provider
                    or not _configured_plugin_enabled(current, provider, loaded)):
                raise ValueError("required admission config changed")
            if (type(response[0]) is not dict or set(response[0]) != {"allow"}
                    or type(response[0]["allow"]) is not bool or response[0]["allow"] is not True):
                raise ValueError("required admission rejected or malformed")
            return Admission(provider, binding, loaded, status, run_id)
    except Exception:
        logger.warning("Required Kanban transition admission denied (%s)", action)
        return None
