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
from typing import Any, Optional


def _profiles_root() -> Path:
    # Plugin discovery and dependency publication must use the same home root.
    from hermes_cli.runtime_paths import dependency_home_root

    return dependency_home_root() / "profiles"


def _read_home_config(home: Path) -> Optional[dict[str, Any]]:
    """Read a selection once; only an absent file means an unknown home.

    An unreadable selection must not shrink the next dependency generation.
    Empty YAML is an explicit empty configuration, as in the CLI loader.
    """
    config_path = home / "config.yaml"
    try:
        text = config_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read plugin selection: {config_path}") from exc

    # Missing YAML support is a broken runtime, not an empty plugin selection.
    import utils
    from ruamel.yaml.error import YAMLError

    try:
        config = utils.fast_safe_load(text)
    except YAMLError as exc:
        raise ValueError(f"could not parse plugin selection: {config_path}") from exc
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    for section in ("plugins", "memory"):
        if config.get(section) is not None and not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping: {config_path}")
    plugins = config.get("plugins") or {}
    for key in ("enabled", "disabled"):
        names = plugins.get(key)
        if names is not None and (not isinstance(names, list)
                                  or any(not isinstance(name, str) for name in names)):
            raise ValueError(f"plugins.{key} must be a list of names: {config_path}")
    provider = (config.get("memory") or {}).get("provider")
    if provider is not None and not isinstance(provider, str):
        raise ValueError(f"memory.provider must be a name: {config_path}")
    return config


def _enabled_from_config(config: dict[str, Any]) -> list[str]:
    """plugins.enabled from an already-parsed config, ORDER-PRESERVING."""
    plugins_cfg = config.get("plugins")
    if not isinstance(plugins_cfg, dict):
        return []
    enabled = plugins_cfg.get("enabled")
    if not isinstance(enabled, list):
        return []
    disabled = plugins_cfg.get("disabled", [])
    disabled = set(disabled) if isinstance(disabled, list) else set()
    out: list[str] = []
    for name in enabled:
        if (isinstance(name, str) and name and name not in out
                and name not in disabled and name.rsplit("/", 1)[-1] not in disabled):
            out.append(name)
    return out


def _is_directory(path: Path) -> bool:
    import stat

    try:
        return stat.S_ISDIR(path.stat().st_mode)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"could not inspect plugin directory: {path}") from exc


def _all_homes() -> list[Path]:
    """Enumerate the complete union or refuse; a partial scan cannot remove members."""
    from hermes_cli.runtime_paths import dependency_home_root

    homes = [dependency_home_root()]
    root = _profiles_root()
    try:
        profiles = sorted(root.iterdir(), key=str)
    except FileNotFoundError:
        return homes
    except OSError as exc:
        raise ValueError(f"could not enumerate profiles: {root}") from exc
    homes.extend(profile for profile in profiles if _is_directory(profile))
    return homes


def enabled_plugins_ordered(*, proposed_home=None, enabled=None, disabled=None) -> dict[Path, list[str]]:
    """plugins_dir → ordered enabled list, per home. Keyed by the
    PLUGINS DIR (where the member dirs live), not the home itself.

    The ACTIVE MEMORY PROVIDER joins its home's list: providers install
    via ``memory.provider`` (mnemosyne's documented path), not via
    plugins.enabled — without this, a provider's dep plugin never joins
    the union. Admission refuses a conflicting candidate without changing
    the active environment or disabling an existing provider."""
    out: dict[Path, list[str]] = {}
    for home in _all_homes():
        # ONE parse per home feeds both queries (enabled + provider).
        config = _read_home_config(home)
        config = config or {}
        if proposed_home is not None and home.resolve() == Path(proposed_home).resolve():
            config = {**config, "plugins": {"enabled": list(enabled or ()), "disabled": list(disabled or ())}}
        names = _enabled_from_config(config)
        provider = _provider_from_config(home, config)
        if provider and provider not in names:
            names.append(provider)
        if names:
            out[home / "plugins"] = names
    return out


def _provider_from_config(home: Path, config: dict[str, Any]) -> Optional[str]:
    """The ``memory.provider`` key of an already-parsed config, when its
    plugin dir exists (no dir = not a member)."""
    provider = (config.get("memory") or {}).get("provider")
    if not provider or not provider.strip():
        return None
    name = provider.strip()
    return name if _is_directory(home / "plugins" / name) else None


def disable_plugins(names: list[str]) -> dict[str, list[str]]:
    """Remove names from EVERY home's enabled list (an operator or
    caller decision names the plugin, not the profile — disable where
    it's enabled). There is NO automatic bisect in pm today; this is
    the explicit write-back path. Returns per-home what was removed.

    Writes go through utils.atomic_roundtrip_yaml_update — the same
    atomic, comment-preserving round-trip writer the plugins CLI's
    config path uses — pointed at that home's config.yaml (explicit
    home scope; pm never derives the target from ambient state). A
    write failure RAISES: a disable that didn't land must never be
    reported as removed. An EXISTING home config that can't be parsed
    also raises — silently skipping it would report success while the
    plugin stays enabled in that home.
    """
    removed: dict[str, list[str]] = {}
    if not names:
        return removed
    name_set = set(names)

    for home in _all_homes():
        config_path = home / "config.yaml"
        config = _read_home_config(home)
        if config is None:
            continue
        plugins_cfg = config.get("plugins")
        if not isinstance(plugins_cfg, dict):
            continue
        enabled = plugins_cfg.get("enabled")
        if not isinstance(enabled, list):
            continue
        hit = [n for n in enabled if isinstance(n, str) and n in name_set]
        if not hit:
            continue
        kept = [n for n in enabled if not (isinstance(n, str) and n in name_set)]
        import utils

        utils.atomic_roundtrip_yaml_update(config_path, "plugins.enabled", kept)
        removed[str(home)] = hit
    return removed
