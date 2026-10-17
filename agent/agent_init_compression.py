"""Config helpers for the ``compression`` section, lifted out of ``agent_init.py``.

``agent_init.py`` is over its file-line cap and may only shrink, so the section's small
coercions and the per-surface merge live here: they take plain dicts and need nothing from
the module that calls them.
"""

from __future__ import annotations

from typing import Any


def _cfg_flag(cfg: dict[str, Any], key: str, default: bool) -> bool:
    """Legacy string-set truthiness used by the ``compression`` section."""
    return str(cfg.get(key, default)).lower() in {"true", "1", "yes"}


def _cfg_dict(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """``cfg[key]`` if it is a mapping, else ``{}`` (malformed sections are ignored)."""
    section = cfg.get(key, {})
    return section if isinstance(section, dict) else {}


def _platform_override_key(platform: Any) -> str:
    """Normalise a surface name for the ``compression.platform_overrides`` lookup.

    ``init_agent`` hands ``platform`` in as a plain string ("telegram", "local"), but callers
    and tests also pass a ``Platform`` enum member, so read ``.value`` when present and fold
    case: one spelling resolves either way.
    """
    if platform is None:
        return ""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _merge_platform_overrides(cfg: dict[str, Any], platform: Any) -> dict[str, Any]:
    """Merge ``compression.platform_overrides.<platform>`` over the global section, per key.

    Opt-in and additive: with no platform, or no block for that surface, ``cfg`` is returned
    unchanged — so a profile that never writes the key behaves exactly as before. A block that
    sets one key inherits the rest from the global section, and a non-mapping block is ignored
    like any other malformed section. Returns a shallow copy; the loaded config is never mutated.
    """
    key = _platform_override_key(platform)
    if not key:
        return cfg
    overrides = _cfg_dict(cfg, "platform_overrides")
    block = overrides.get(key)
    if block is None:
        # The config file is hand-written, so match keys case-insensitively (exact hit first).
        for name, value in overrides.items():
            if isinstance(value, dict) and str(name).strip().lower() == key:
                block = value
                break
    if not isinstance(block, dict) or not block:
        return cfg
    merged = dict(cfg)
    merged.update(block)
    return merged
