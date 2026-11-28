"""One fresh plugin rediscovery on the ``hermes send`` path after a load-timeout.

The 10s hang-guard and the discard of a late ``register_platform()`` stay
untouched. This module only retries discovery when the first miss is
specifically a plugin-load timeout for the requested platform — never by
un-abandoning the timed-out registration.
"""

from __future__ import annotations

import json
import sys
from typing import Callable

from hermes_cli.plugins_loader import plugin_load_timeout_override, _resolve_plugin_load_timeout

_NO_ADAPTER_PREFIX = "No live adapter for platform '"
_TIMEOUT_NEEDLE = "load timed out"
SEND_PLUGIN_LOAD_TIMEOUT_EXIT = 2
RETRY_MULTIPLIER = 3
RETRY_CAP_SECS = 60.0


def retry_load_timeout_seconds(resolved: float) -> float:
    """Per-retry deadline: ``min(3 * resolved, 60)``. ``0`` stays disabled."""
    if resolved <= 0:
        return resolved
    return min(RETRY_MULTIPLIER * resolved, RETRY_CAP_SECS)


def _platform_from_target(target: str) -> str:
    return target.split(":", 1)[0].strip().lower()


def _payload(result: str) -> dict:
    try:
        data = json.loads(result) if result else {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def is_no_live_adapter(payload: dict, platform: str) -> bool:
    err = str(payload.get("error") or "")
    return f"{_NO_ADAPTER_PREFIX}{platform}'" in err


def plugin_owns_platform(info: dict, platform: str) -> bool:
    plat = platform.lower()
    name = str(info.get("name") or "").lower()
    key = str(info.get("key") or "").lower()
    kind = str(info.get("kind") or "").lower()
    aliases = {plat, f"{plat}-platform"}
    if name in aliases or key in aliases:
        return True
    if kind == "platform" and (name == plat or name.startswith(f"{plat}-") or name.endswith(f"-{plat}")):
        return True
    return False


def plugin_load_timed_out_for_platform(platform: str) -> bool:
    from hermes_cli.plugins import get_plugin_manager

    for info in get_plugin_manager().list_plugins():
        err = str(info.get("error") or "")
        if _TIMEOUT_NEEDLE in err and plugin_owns_platform(info, platform):
            return True
    return False


def rediscover_plugins_for_send_retry() -> None:
    from hermes_cli.plugins import discover_plugins

    retry_timeout = retry_load_timeout_seconds(_resolve_plugin_load_timeout())
    with plugin_load_timeout_override(retry_timeout):
        discover_plugins(force=True)


def maybe_retry_send_after_plugin_load_timeout(
    target: str,
    result: str,
    send_again: Callable[[], str],
) -> tuple[str, int | None]:
    """Retry send once after a load-timeout miss. Return ``(result, exit_or_None)``.

    ``exit_or_None`` is 2 only on a double timeout miss (stderr already printed).
    Any other outcome leaves the caller on today's ``_emit_result`` path.
    """
    platform = _platform_from_target(target)
    if not platform:
        return result, None
    payload = _payload(result)
    if not is_no_live_adapter(payload, platform):
        return result, None
    if not plugin_load_timed_out_for_platform(platform):
        return result, None
    rediscover_plugins_for_send_retry()
    result2 = send_again()
    payload2 = _payload(result2)
    if is_no_live_adapter(payload2, platform) and plugin_load_timed_out_for_platform(platform):
        print(f"plugin_load_timeout platform={platform} attempts=2", file=sys.stderr)
        return result2, SEND_PLUGIN_LOAD_TIMEOUT_EXIT
    return result2, None
