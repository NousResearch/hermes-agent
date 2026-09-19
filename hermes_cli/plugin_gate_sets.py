"""Shared plugin enablement resolution for the web server's trust gates.

``_dashboard_plugin_search_dirs`` scans BOTH the launch home and the hermes root when the
process itself is profile-scoped (#87197), so a plugin installed in the root is discovered,
its assets are served and its API router is mounted — while ``plugins.enabled`` /
``plugins.disabled`` were read from the process home alone, whose config.yaml (a fresh
profile's) has no ``plugins:`` section. The two halves of one trust decision then disagree:
discovery says the plugin exists, the gate 404s it (``Plugin not found``). Resolving the
sets from the SAME homes discovery scanned keeps them in lockstep, with the launch home
authoritative and the deny-list always winning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Set, Tuple

from hermes_constants import get_default_hermes_root, get_process_hermes_home


def _name_set_from_config(config_path: Path, keys: Tuple[str, str]) -> Set[str]:
    """``plugins.<key>`` read directly from one config.yaml; empty on missing/failure."""
    try:
        import yaml

        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        section = data.get(keys[0]) if isinstance(data, dict) else None
        value = section.get(keys[1], []) if isinstance(section, dict) else None
        return set(value) if isinstance(value, list) else set()
    except Exception:
        return set()


def plugin_enable_sets() -> Tuple[Set[str], Set[str]]:
    """``(enabled, disabled)`` covering every home plugin discovery scans.

    Mirrors ``_dashboard_plugin_search_dirs``: the launch home first, then the hermes root
    when the process is profile-scoped. The deny-list unions across homes (an explicit
    disable in either location always wins); the allow-list unions as well so a root-enabled
    plugin keeps working in a profile process whose own config.yaml predates it. On a root
    process the second home is the same directory and this degenerates to the legacy
    single-home read.
    """
    homes = [get_process_hermes_home()]
    root = get_default_hermes_root()
    if root.resolve(strict=False) != homes[0].resolve(strict=False):
        homes.append(root)

    enabled: Set[str] = set()
    disabled: Set[str] = set()
    for home in homes:
        enabled |= _name_set_from_config(home / "config.yaml", ("plugins", "enabled"))
        disabled |= _name_set_from_config(home / "config.yaml", ("plugins", "disabled"))

    if not enabled and not disabled:
        # Nothing readable anywhere: preserve the historical fail-closed read of the
        # process home so a broken config still blocks rather than opens the gate.
        from hermes_cli.plugins_cmd import _config_name_set

        return _config_name_set("plugins", "enabled"), _config_name_set("plugins", "disabled")
    return enabled, disabled
