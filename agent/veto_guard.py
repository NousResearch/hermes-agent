"""Veto policy guard for Hermes tool calls.

The registry-level check in ``tools.registry`` uses this module as a small
adapter around the optional ``veto`` Python SDK. The bundled ``plugins/veto``
plugin also calls into this module from the ``pre_tool_call`` hook so users can
enable Veto through Hermes' existing plugin UX.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_VALID_MODES = {"strict", "log", "shadow"}
_VALID_VALIDATION_MODES = {"local", "cloud"}

_process_enabled = False
_client_lock = threading.RLock()
_client_key: Optional[tuple[Any, ...]] = None
_client: Any = None
_thread_state = threading.local()
_main_loop: Optional[asyncio.AbstractEventLoop] = None
_main_loop_lock = threading.Lock()
_worker_thread_local = threading.local()


@dataclass(frozen=True)
class VetoSettings:
    enabled: bool
    config_dir: Path
    mode: str
    validation_mode: str
    fail_open: bool
    api_key: Optional[str]
    endpoint: Optional[str]
    log_level: str
    excluded_tools: frozenset[str]


@dataclass(frozen=True)
class VetoCheck:
    allowed: bool
    message: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None
    rule_id: Optional[str] = None


def enable_for_process() -> None:
    """Enable registry-level Veto checks for this process.

    The bundled plugin calls this during registration. Users can also enable
    the guard without the plugin by setting ``veto.enabled: true`` in
    ``config.yaml`` or ``HERMES_VETO_ENABLED=1``.
    """
    global _process_enabled
    _process_enabled = True


def disable_for_process() -> None:
    """Disable process-forced Veto checks.

    Intended for tests and explicit teardown only; config/env may still enable
    the guard.
    """
    global _process_enabled
    _process_enabled = False


def reset_for_tests() -> None:
    """Clear cached SDK state and thread-local precheck data."""
    global _client, _client_key, _process_enabled, _main_loop
    with _client_lock:
        _client = None
        _client_key = None
    _process_enabled = False
    _thread_state.prechecked = set()
    if _main_loop is not None and not _main_loop.is_closed():
        _main_loop.close()
    _main_loop = None


def check_tool_call(
    tool_name: str,
    args: Mapping[str, Any] | None,
    *,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    skip_veto_guard: bool = False,
) -> VetoCheck:
    """Check one Hermes tool call against Veto policy."""
    if skip_veto_guard:
        return VetoCheck(allowed=True)

    settings = _load_settings()
    if not settings.enabled or tool_name in settings.excluded_tools:
        return VetoCheck(allowed=True)

    args_dict = dict(args or {})
    if _consume_prechecked(tool_name, args_dict):
        return VetoCheck(allowed=True)

    return _run_veto_check(
        settings,
        tool_name,
        args_dict,
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        mark_allowed=False,
    )


def precheck_tool_call(
    tool_name: str,
    args: Mapping[str, Any] | None,
    *,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
) -> VetoCheck:
    """Run Veto from a Hermes ``pre_tool_call`` hook.

    If the call is allowed, the signature is marked as already checked so the
    immediately following registry dispatch does not call Veto twice.
    """
    settings = _load_settings(force_enabled=True)
    if tool_name in settings.excluded_tools:
        return VetoCheck(allowed=True)

    return _run_veto_check(
        settings,
        tool_name,
        dict(args or {}),
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        mark_allowed=True,
    )


def status_text() -> str:
    settings = _load_settings()
    status = "enabled" if settings.enabled else "disabled"
    source = "config/env/plugin" if settings.enabled else "not enabled"
    return (
        f"Veto guard is {status} ({source}).\n"
        f"Config dir: {settings.config_dir}\n"
        f"Validation mode: {settings.validation_mode}\n"
        f"SDK installed: {'yes' if _sdk_available() else 'no'}\n"
        f"Fail open: {'yes' if settings.fail_open else 'no'}"
    )


def _load_settings(*, force_enabled: bool = False) -> VetoSettings:
    config = _load_hermes_config()
    veto_cfg = config.get("veto") if isinstance(config.get("veto"), dict) else {}

    env_enabled = _parse_bool(os.environ.get("HERMES_VETO_ENABLED"))
    config_enabled = _parse_bool(veto_cfg.get("enabled"))
    enabled = bool(
        force_enabled
        or _process_enabled
        or (env_enabled if env_enabled is not None else config_enabled)
    )

    raw_config_dir = (
        os.environ.get("HERMES_VETO_CONFIG_DIR")
        or _string(veto_cfg.get("config_dir"))
        or str(_default_config_dir())
    )
    config_dir = Path(raw_config_dir).expanduser()

    mode = (
        os.environ.get("HERMES_VETO_MODE")
        or _string(veto_cfg.get("mode"))
        or "strict"
    )
    if mode not in _VALID_MODES:
        mode = "strict"

    validation_mode = (
        os.environ.get("HERMES_VETO_VALIDATION_MODE")
        or _string(veto_cfg.get("validation_mode"))
        or "local"
    )
    if validation_mode not in _VALID_VALIDATION_MODES:
        validation_mode = "local"

    fail_open = _parse_bool(os.environ.get("HERMES_VETO_FAIL_OPEN"))
    if fail_open is None:
        fail_open = _parse_bool(veto_cfg.get("fail_open"))
    if fail_open is None:
        fail_open = False

    api_key_env = _string(veto_cfg.get("api_key_env")) or "VETO_API_KEY"
    endpoint_env = _string(veto_cfg.get("endpoint_env")) or "VETO_ENDPOINT"
    api_key = os.environ.get(api_key_env) or _string(veto_cfg.get("api_key"))
    endpoint = os.environ.get(endpoint_env) or _string(veto_cfg.get("endpoint"))
    log_level = _string(veto_cfg.get("log_level")) or "warn"
    excluded_tools = frozenset(
        item for item in veto_cfg.get("excluded_tools", []) if isinstance(item, str)
    )

    return VetoSettings(
        enabled=enabled,
        config_dir=config_dir,
        mode=mode,
        validation_mode=validation_mode,
        fail_open=fail_open,
        api_key=api_key,
        endpoint=endpoint,
        log_level=log_level,
        excluded_tools=excluded_tools,
    )


def _load_hermes_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        loaded = load_config_readonly()
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.debug("Could not load Hermes config for Veto guard: %s", exc)
        return {}


def _default_config_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "veto"
    except Exception:
        return Path.home() / ".hermes" / "veto"


def _string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    return None


def _run_veto_check(
    settings: VetoSettings,
    tool_name: str,
    args: dict[str, Any],
    *,
    task_id: str,
    session_id: str,
    tool_call_id: str,
    mark_allowed: bool,
) -> VetoCheck:
    try:
        veto = _get_veto_client(settings)
        result = _run_coro(
            veto.guard(
                tool_name,
                args,
                session_id=session_id or task_id or None,
                agent_id="hermes-agent",
            )
        )
    except Exception as exc:
        logger.exception("Veto guard failed for tool %s", tool_name)
        if settings.fail_open:
            return VetoCheck(allowed=True, reason=str(exc))
        return VetoCheck(
            allowed=False,
            decision="guard_error",
            reason=str(exc),
            message=(
                "Veto guard failed before tool execution. "
                "Set veto.fail_open: true only if you intentionally want Hermes "
                "to continue when policy validation is unavailable."
            ),
        )

    decision = getattr(result, "decision", None)
    reason = getattr(result, "reason", None)
    rule_id = getattr(result, "rule_id", None)
    if decision == "allow":
        if mark_allowed:
            _mark_prechecked(tool_name, args)
        return VetoCheck(allowed=True, decision=decision, reason=reason, rule_id=rule_id)

    message = _format_block_message(tool_name, decision, reason, rule_id, tool_call_id)
    return VetoCheck(
        allowed=False,
        decision=str(decision or "deny"),
        reason=reason if isinstance(reason, str) else None,
        rule_id=rule_id if isinstance(rule_id, str) else None,
        message=message,
    )


def _get_veto_client(settings: VetoSettings) -> Any:
    global _client, _client_key
    key = (
        str(settings.config_dir),
        settings.mode,
        settings.validation_mode,
        settings.api_key,
        settings.endpoint,
        settings.log_level,
        settings.config_dir.exists(),
    )
    with _client_lock:
        if _client is not None and _client_key == key:
            return _client
        _client = _create_veto_client(settings)
        _client_key = key
        return _client


def _create_veto_client(settings: VetoSettings) -> Any:
    try:
        from veto import Veto, VetoOptions
    except Exception as exc:
        raise RuntimeError(
            "Python package 'veto' is required. Install it with `pip install veto` "
            "or `uv pip install veto`."
        ) from exc

    if settings.validation_mode == "cloud" or settings.config_dir.exists():
        return _run_coro(
            Veto.init(
                VetoOptions(
                    config_dir=str(settings.config_dir),
                    mode=settings.mode,
                    validation_mode=settings.validation_mode,
                    api_key=settings.api_key,
                    base_url=settings.endpoint,
                    log_level=settings.log_level,
                )
            )
        )

    return Veto.from_rules(
        rules=_load_bundled_rules(),
        mode=settings.mode,
        api_key=settings.api_key,
        endpoint=settings.endpoint,
        log_level=settings.log_level,
    )


def _load_bundled_rules() -> list[dict[str, Any]]:
    rules_dir = Path(__file__).resolve().parent.parent / "plugins" / "veto" / "defaults" / "rules"
    rules: list[dict[str, Any]] = []
    for path in sorted(rules_dir.glob("*.y*ml")):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            rules.extend(rule for rule in data["rules"] if isinstance(rule, dict))
    return rules


def _format_block_message(
    tool_name: str,
    decision: Any,
    reason: Any,
    rule_id: Any,
    tool_call_id: str,
) -> str:
    label = "requires approval" if decision == "require_approval" else "blocked"
    parts = [f"Veto {label} {tool_name}"]
    if isinstance(reason, str) and reason.strip():
        parts.append(reason.strip())
    if isinstance(rule_id, str) and rule_id:
        parts.append(f"rule={rule_id}")
    if tool_call_id:
        parts.append(f"call={tool_call_id}")
    return ": ".join(parts)


def _signature(tool_name: str, args: Mapping[str, Any]) -> str:
    return json.dumps(
        {"tool": tool_name, "args": args},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _prechecked_set() -> set[str]:
    existing = getattr(_thread_state, "prechecked", None)
    if not isinstance(existing, set):
        existing = set()
        _thread_state.prechecked = existing
    return existing


def _mark_prechecked(tool_name: str, args: Mapping[str, Any]) -> None:
    _prechecked_set().add(_signature(tool_name, args))


def _consume_prechecked(tool_name: str, args: Mapping[str, Any]) -> bool:
    sig = _signature(tool_name, args)
    prechecked = _prechecked_set()
    if sig not in prechecked:
        return False
    prechecked.remove(sig)
    return True


def _sdk_available() -> bool:
    try:
        import veto  # noqa: F401

        return True
    except Exception:
        return False


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    global _main_loop
    if threading.current_thread() is threading.main_thread():
        with _main_loop_lock:
            if _main_loop is None or _main_loop.is_closed():
                _main_loop = asyncio.new_event_loop()
            return _main_loop

    loop = getattr(_worker_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _worker_thread_local.loop = loop
    return loop


def _run_coro(coro: Any) -> Any:
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result(timeout=300)

    loop = _get_thread_loop()
    return loop.run_until_complete(coro)
