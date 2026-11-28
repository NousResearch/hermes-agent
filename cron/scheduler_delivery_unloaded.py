"""Cron destinations whose platform plugin is not loaded in the served profile: a home channel saved in
config.yaml that ``all``/``origin`` cannot reach, and a platform that left core, whose ref is read with the
primary's parser (else a generic ``chat[:thread]`` split) and named with its install command.

Split out of ``cron.scheduler_delivery``, which reaches it late-bound (``_unloaded``); this module reaches
the delivery seam the same way, so monkeypatching either defining module works.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("cron.scheduler")


# Core-owned platforms a saved home channel may name without any plugin registering them (beside the
# delivery module's known and non-push origin platforms).
_CORE_HOME_PLATFORMS = frozenset({"local", "relay", "msgraph_webhook"})


def _saved_home_channel(platform_name: str) -> dict:
    """The ``home_channel`` block config.yaml saves for *platform_name* (``chat_id``, optional ``thread_id``),
    read raw like :func:`_unloaded_home_platforms`: a platform nothing registers here has no gateway-config
    home, but the address the profile saved is still its home. ``{}`` when none is saved."""
    try:
        blocks = (_sched.load_config() or {}).get("platforms") or {}
    except Exception:
        logger.debug("cron: config.yaml unreadable for saved home channels", exc_info=True)
        return {}
    block = next((b for name, b in blocks.items() if str(name).lower() == platform_name.lower()), None)
    home = block.get("home_channel") if isinstance(block, dict) else None
    return home if isinstance(home, dict) and home.get("chat_id") else {}


def _unloaded_home_platforms() -> list[str]:
    """Platforms with a home chat saved in config.yaml (what ``/sethome`` writes) that no loaded plugin
    registers in this profile (not installed, not enabled, failed to load). Read raw, because
    ``load_gateway_config`` drops a block whose platform nothing registers, so ``origin``/``all`` would
    otherwise skip that home without a word."""
    from gateway.platform_registry import platform_registry
    try:
        blocks = (_sched.load_config() or {}).get("platforms") or {}
    except Exception:
        logger.debug("cron: config.yaml unreadable for saved home channels", exc_info=True)
        return []
    saved = sorted(
        str(name).lower() for name, block in blocks.items()
        if isinstance(block, dict) and isinstance(block.get("home_channel"), dict)
        and block["home_channel"].get("chat_id"))
    list(_delivery._iter_home_target_platforms())  # discovers plugins so the registry lookups below are final
    core_homes = _delivery._KNOWN_DELIVERY_PLATFORMS | _delivery._NON_PUSH_ORIGIN_PLATFORMS | _CORE_HOME_PLATFORMS
    return [n for n in saved if n not in core_homes and platform_registry.get(n) is None]


def _unloaded_home_error(platform_name: str) -> str:
    """Job-result text for a saved home channel skipped because its platform plugin is not loaded."""
    from hermes_cli.left_core_migration import platform_install_hint
    remedy = platform_install_hint(platform_name) or (
        f". Install or enable the plugin that provides '{platform_name}' (`hermes plugins list`)")
    return (f"{platform_name} home channel skipped: no loaded plugin provides the platform{remedy}, "
            f"or remove platforms.{platform_name}.home_channel from config.yaml")


def _note_unloaded_homes(names, skipped: Optional[list]) -> None:
    for name in names:
        if skipped is not None and (error := _unloaded_home_error(name)) not in skipped:
            skipped.append(error)


def moved_platform_hint(platform_name: str) -> str:
    """``". <label> moved out of core into the '<plugin>' plugin: install it with `<command>`"`` for a
    platform that left core whose plugin is not loaded, else ``""``: a route saved while core shipped it
    names the install instead of failing as an unknown target."""
    from gateway.platform_registry import platform_registry
    from hermes_cli.left_core_migration import platform_install_hint
    name = platform_name.strip().lower()
    return "" if platform_registry.get(name) else platform_install_hint(name)


def _primary_registration(platform_key: str):
    """The primary's loaded ``PlatformEntry`` for *platform_key*, or ``None``: a moved platform's ref syntax."""
    from gateway.platform_registry import platform_registry
    from hermes_constants import get_default_hermes_root, hermes_home_key
    return platform_registry.snapshot_registration(platform_key, scope=hermes_home_key(get_default_hermes_root()))[0]


def _moved_platform_ref(platform_key: str, ref: str) -> tuple[str, Optional[str]]:
    """``(chat_id, thread_id)`` of a moved platform's ref when its plugin is not loaded here: the
    primary's already-loaded plugin parser when it reads the ref, else core's generic
    ``chat[:thread]`` split. Only an exact route match on the result sends anything."""
    entry = _primary_registration(platform_key)
    parse = entry.parse_target_ref_fn if entry is not None else None
    parsed = None
    if parse is not None:
        try:
            parsed = parse(ref)
        except Exception:
            logger.debug("Primary's target parser failed for %s", platform_key, exc_info=True)
    if isinstance(parsed, tuple) and len(parsed) == 2 and parsed[0]:
        return str(parsed[0]), str(parsed[1]) if parsed[1] else None
    chat_id, _, thread_id = ref.partition(":")
    return chat_id, thread_id or None


# Late-bound (see module docstring): imported LAST, after both modules define their names.
from cron import scheduler as _sched
from cron import scheduler_delivery as _delivery
