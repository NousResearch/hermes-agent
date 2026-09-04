"""Which plugins are enabled, per profile — pm's read of the plugins
config (order-preserving for the incumbent-wins tiebreak).

pm needs two things the plugins_cmd helpers don't give: EVERY profile's
enabled list (the union is per-install, cross-profile) and the list
ORDER (config order = enable recency; enabling appends). Writes go
through the same config.yaml the plugins CLI owns — pm never invents a
second authority for enabled state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _profiles_root() -> Path:
    # Profile roots are derived from get_default_hermes_root() — the ONE
    # authority for "where do profiles live" (hermes_constants): it
    # returns HERMES_HOME directly for custom roots (Docker /opt/data,
    # non-default local roots) and <root> for profile-mode HERMES_HOME,
    # on both standard and custom layouts. Hardcoding Path.home()/
    # .hermes/profiles silently omits custom-root profiles — their
    # enabled dep plugins never join the union and bisect disable
    # decisions never write back to their config.
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "profiles"


def _enabled_list_for_home(home: Path) -> list[str]:
    """plugins.enabled for ONE hermes home, ORDER-PRESERVING."""
    try:
        import yaml

        config_path = home / "config.yaml"
        if not config_path.is_file():
            return []
        with config_path.open(encoding="utf-8-sig") as f:
            config = yaml.safe_load(f) or {}
        plugins_cfg = config.get("plugins") or {}
        if not isinstance(plugins_cfg, dict):
            return []
        enabled = plugins_cfg.get("enabled")
        if not isinstance(enabled, list):
            return []
        out: list[str] = []
        for name in enabled:
            if isinstance(name, str) and name and name not in out:
                out.append(name)
        return out
    except Exception:
        return []


def _all_homes() -> list[Path]:
    """The default home + every profile home (the union's scope)."""
    homes: list[Path] = []
    try:
        from hermes_constants import get_default_hermes_root

        homes.append(get_default_hermes_root())
    except Exception:
        pass
    try:
        root = _profiles_root()
        if root.is_dir():
            for profile in sorted(root.iterdir(), key=str):
                if profile.is_dir():
                    homes.append(profile)
    except OSError:
        pass
    return homes


def enabled_plugins_ordered() -> dict[Path, list[str]]:
    """plugins_dir → ordered enabled list, per home. Keyed by the
    PLUGINS DIR (where the member dirs live), not the home itself.

    The ACTIVE MEMORY PROVIDER joins its home's list: providers install
    via ``memory.provider`` (mnemosyne's documented path), not via
    plugins.enabled — without this, a provider's dep plugin never joins
    the union. The provider rides LAST (newest — the bisect's
    incumbent-wins tiebreak disables it before older plugins)."""
    out: dict[Path, list[str]] = {}
    for home in _all_homes():
        enabled = _enabled_list_for_home(home)
        provider = _active_memory_provider(home)
        if provider and provider not in enabled:
            enabled = enabled + [provider]
        if enabled:
            out[home / "plugins"] = enabled
    return out


def _active_memory_provider(home: Path) -> Optional[str]:
    """The home's ``memory.provider`` config key, when set and its plugin
    dir exists (a provider name with no installed dir is not a member)."""
    try:
        import yaml

        config_path = home / "config.yaml"
        if not config_path.is_file():
            return None
        with config_path.open(encoding="utf-8-sig") as f:
            config = yaml.safe_load(f) or {}
        provider = (config.get("memory") or {}).get("provider")
        if not isinstance(provider, str) or not provider.strip():
            return None
        name = provider.strip()
        if (home / "plugins" / name).is_dir():
            return name
        return None
    except Exception:
        return None


def disable_plugins(names: list[str]) -> dict[str, list[str]]:
    """Remove names from EVERY home's enabled list (a bisect decision
    names the plugin, not the profile — disable where it's enabled).
    Returns per-home what was removed. Writes via yaml round-trip of
    the same config.yaml the plugins CLI owns."""
    removed: dict[str, list[str]] = {}
    if not names:
        return removed
    name_set = set(names)
    import yaml

    for home in _all_homes():
        config_path = home / "config.yaml"
        if not config_path.is_file():
            continue
        try:
            with config_path.open(encoding="utf-8-sig") as f:
                config = yaml.safe_load(f) or {}
            plugins_cfg = config.get("plugins")
            if not isinstance(plugins_cfg, dict):
                continue
            enabled = plugins_cfg.get("enabled")
            if not isinstance(enabled, list):
                continue
            kept = [n for n in enabled if not (isinstance(n, str) and n in name_set)]
            hit = [n for n in enabled if isinstance(n, str) and n in name_set]
            if not hit:
                continue
            plugins_cfg["enabled"] = kept
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            removed[str(home)] = hit
        except Exception:
            continue
    return removed
