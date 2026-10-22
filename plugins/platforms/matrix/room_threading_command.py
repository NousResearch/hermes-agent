"""Persist chat-controlled room threading through the standard config writer."""

import asyncio
import logging
from typing import TYPE_CHECKING

from agent.i18n import t
from plugins.platforms.matrix.threading_policy import parse_room_auto_thread_overrides, should_create_thread

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger(__name__)
_MODES = {"on": True, "off": False, "default": None}
_MATRIX_PATHS = (
    ("matrix",), ("platforms", "matrix"),
    ("gateway", "matrix"), ("gateway", "platforms", "matrix"),
)


class ManagedThreadingError(RuntimeError):
    pass


def _matrix_sections(config: dict):
    for path in _MATRIX_PATHS:
        section = config
        for key in path:
            section = section.get(key) if isinstance(section, dict) else None
        if isinstance(section, dict):
            yield section


def _check_managed() -> None:
    from hermes_cli.config import is_managed
    from hermes_cli.managed_scope import load_managed_config

    if is_managed():
        raise ManagedThreadingError
    for section in _matrix_sections(load_managed_config() or {}):
        for settings in (section, section.get("extra", {})):
            if not isinstance(settings, dict) or "room_overrides" not in settings:
                continue
            # The gateway resolves this mapping as one setting across alternate YAML layouts.
            raise ManagedThreadingError


def _room_overrides_for_write(raw: dict) -> dict:
    from gateway.config import Platform, PlatformConfig
    from gateway.config_loader import (
        bridge_platform_shared_keys, load_legacy_gateway_json, merge_platform_sections, snapshot_authored_extra,
    )
    from hermes_constants import get_hermes_home

    gateway = raw.get("gateway", {})
    gateway_platforms = gateway.get("platforms") if isinstance(gateway, dict) else None
    data = load_legacy_gateway_json(get_hermes_home())
    platforms = merge_platform_sections(raw, gateway, data)
    authored = snapshot_authored_extra(merge_platform_sections(raw, gateway, {}))
    # Reuse loader precedence for write-back, without copying managed pins or expanded values.
    bridge_platform_shared_keys(
        raw, gateway_platforms, data, platforms, [Platform.MATRIX], authored=authored)
    extra = PlatformConfig.from_dict(platforms.get("matrix", {})).extra
    overrides = extra.get("room_overrides", {})
    parse_room_auto_thread_overrides(overrides)
    return {room: dict(settings) for room, settings in overrides.items()}


def _persist(adapter: "MatrixAdapter", room_id: str, mode: str) -> None:
    from gateway.config import Platform, load_gateway_config
    from hermes_cli.config import (
        _CONFIG_LOCK, atomic_config_replace, get_config_path, require_readable_config_before_write,
    )

    with _CONFIG_LOCK:
        _check_managed()
        config_path = get_config_path()
        raw = require_readable_config_before_write(config_path)
        effective = load_gateway_config().platforms[Platform.MATRIX].extra.get("room_overrides", {})
        parsed = parse_room_auto_thread_overrides(effective)
        overrides = _room_overrides_for_write(raw)
        if mode == "default":
            overrides.pop(room_id, None)
            parsed.pop(room_id, None)
        else:
            overrides[room_id] = {"auto_thread": _MODES[mode]}
            parsed[room_id] = mode == "on"
        # Explicit platform extra wins over all accepted root/nested Matrix YAML forms.
        section = raw
        for key in ("platforms", "matrix", "extra"):
            section = section.setdefault(key, {})
            if not isinstance(section, dict):
                raise ValueError(f"Cannot save room threading: {key} must be a mapping")
        section["room_overrides"] = overrides
        atomic_config_replace(config_path, raw)
        adapter.config.extra["room_overrides"] = {
            room: {"auto_thread": flag} for room, flag in parsed.items()
        }
        adapter._room_auto_thread_overrides = parsed


async def configure_room_threading(adapter: "MatrixAdapter", room_id: str, mode: str) -> str:
    if mode not in {"", "status", *_MODES}:
        return t("threading.usage")
    if mode in _MODES:
        try:
            await asyncio.to_thread(_persist, adapter, room_id, mode)
        except ManagedThreadingError:
            return t("threading.managed", room_id=room_id)
        except (OSError, ValueError, RuntimeError):
            logger.exception("Matrix: could not persist threading for %s", room_id)
            return t("threading.save_failed", room_id=room_id)
    is_dm = await adapter._is_dm_room(room_id)
    automatic = should_create_thread(adapter, room_id, is_dm, False)
    mentioned = should_create_thread(adapter, room_id, is_dm, True)
    state = "on" if automatic else ("mentions" if mentioned else "off")
    policy = "override" if room_id in adapter._room_auto_thread_overrides else "default"
    reply = t("threading.status", room_id=room_id, state=t(f"threading.{state}"),
              policy=t(f"threading.{policy}"))
    return reply + ("\n" + t("threading.saved") if mode in _MODES else "")
