"""Per-profile view of ``rabbit_cli.models``' module-level catalog slots.

Several caches on the facade (curated OpenRouter list, reasoning-capability catalogs and their
once-per-process guards) hold values derived from ONE profile's config, ``.env`` and ``<home>/cache``
files. In a multiplexed gateway every turn runs under a RABBIT_HOME override, so a single module slot
would hand the launch profile's value to every other profile. Under an override the slot is read and
written per home key (routed profiles start cold, never from the launch profile's warmed value);
without one the module attribute stays the slot, so single-profile behaviour and the tests that reset
``models._X = None`` are untouched. Same shape as ``tools.approval._permanent_set``.
"""

from __future__ import annotations

from typing import Any

from rabbit_constants import get_rabbit_home_override, rabbit_home_key

_SLOTS_BY_HOME: dict[tuple[str, str], Any] = {}


def profile_slot_get(module: Any, attr: str, default: Any = None) -> Any:
    """``module.<attr>`` for the active profile; ``default`` is a routed profile's cold value."""
    if get_rabbit_home_override() is None:
        return getattr(module, attr)
    return _SLOTS_BY_HOME.get((rabbit_home_key(), attr), default)


def profile_slot_set(module: Any, attr: str, value: Any) -> None:
    if get_rabbit_home_override() is None:
        setattr(module, attr, value)
    else:
        _SLOTS_BY_HOME[(rabbit_home_key(), attr)] = value
