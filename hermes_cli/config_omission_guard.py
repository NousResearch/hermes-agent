"""Which settings a config write would delete by leaving them out.

Split out of :mod:`hermes_cli.config` (code-health size ratchet); ``_write_config_state``
late-imports it to refuse an incomplete ``atomic_config_write``.
"""

from __future__ import annotations

from typing import Any


def _omitted_config_paths(
    existing: dict[str, Any], proposed: dict[str, Any], prefix: tuple[str, ...] = (),
) -> list[str]:
    """Mapping paths that *proposed* would delete by omission from *existing*.

    The round-trip writer recurses through mappings, so the completeness check must recurse too:
    checking top-level key counts still lets a partial ``plugins: {...}`` payload erase sibling
    settings inside that section. Replacing a non-empty mapping with a scalar/list is likewise a
    deletion of that mapping's children and is reported at the mapping path.
    """
    omitted: list[str] = []
    for key, old_value in existing.items():
        path = (*prefix, str(key))
        if key not in proposed:
            omitted.append(".".join(path))
            continue
        new_value = proposed[key]
        if isinstance(old_value, dict):
            if isinstance(new_value, dict):
                omitted.extend(_omitted_config_paths(old_value, new_value, path))
            elif old_value:
                omitted.append(".".join(path))
    return omitted
