"""Read every profile's enabled plugins in config order for the shared union.

Plugin admission owns writes; discovery never edits a profile's selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import logging

LOG = logging.getLogger(__name__)


def _profiles_root() -> Path:
    # Plugin discovery and dependency publication must use the same home root.
    from pm.environments import dependency_home_root

    return dependency_home_root() / "profiles"


def read_home_selection(home: Path) -> Optional[dict[str, Any]]:
    """The plugin/memory selection a home's config.yaml declares (None: no config yet).

    The public reader for anything that must agree with what PM installs for that home.
    An unreadable selection raises rather than shrinking the next dependency generation;
    empty YAML is an explicit empty configuration, as in the CLI loader.
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
    if provider is False:
        # Legacy "unset" written by older config writers. The runtime reader treats falsy as
        # no-provider, so rejecting it here bricks `hermes pm repair` — the only tool that
        # could heal such a config. The parsed config is returned untouched; every consumer
        # already coerces falsy to no-provider.
        provider = None
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


def dependency_homes() -> list[Path]:
    """Every home whose selection feeds the shared venv: the default home plus each LIVE profile.
    Enumerates the complete union or refuses; a partial scan cannot remove members.

    Live means what ``hermes profile`` lists (hermes_constants): a valid id carrying an identity
    marker and no tombstone. Staging dirs (``.work.staging-*``), deleted profiles and stray
    marker-less dirs must not put plugins into the shared environment.
    """
    from hermes_constants import PROFILE_ID_RE, named_profile_is_live
    from pm.environments import dependency_home_root

    homes = [dependency_home_root()]
    root = _profiles_root()
    try:
        profiles = sorted(root.iterdir(), key=str)
    except FileNotFoundError:
        return homes
    except OSError as exc:
        raise ValueError(f"could not enumerate profiles: {root}") from exc
    homes.extend(profile for profile in profiles
                 if _is_directory(profile) and profile.name != "default"
                 and PROFILE_ID_RE.match(profile.name) and named_profile_is_live(profile))
    return homes


def enabled_plugins_ordered(*, proposed_home=None, enabled=None, disabled=None,
                            installing: Path | None = None,
                            skip_invalid_secondary: bool = False) -> dict[Path, list[str]]:
    """plugins_dir → ordered enabled list, per home. Keyed by the
    PLUGINS DIR (where the member dirs live), not the home itself.

    The ACTIVE MEMORY PROVIDER joins its home's list: providers install
    via ``memory.provider`` (mnemosyne's documented path), not via
    plugins.enabled — without this, a provider's dep plugin never joins
    the union. Admission refuses a conflicting candidate without changing
    the active environment or disabling an existing provider."""
    out: dict[Path, list[str]] = {}
    homes = dependency_homes()
    for home in homes:
        # ONE parse per home feeds both queries (enabled + provider).
        try:
            config = read_home_selection(home)
        except ValueError as exc:
            if not skip_invalid_secondary or home == homes[0]:
                raise
            LOG.warning("Skipping broken secondary profile %s during dependency verification: %s", home, exc)
            continue
        config = config or {}
        if proposed_home is not None and home.resolve() == Path(proposed_home).resolve():
            config = {**config, "plugins": {"enabled": list(enabled or ()), "disabled": list(disabled or ())}}
        names = _enabled_from_config(config)
        provider = _provider_from_config(home, config, installing=installing)
        if provider and provider not in names:
            names.append(provider)
        if names:
            out[home / "plugins"] = names
    return out


def _provider_from_config(home: Path, config: dict[str, Any], *, installing: Path | None = None) -> Optional[str]:
    """The ``memory.provider`` key of an already-parsed config, when its
    plugin dir exists (no dir = not a member)."""
    provider = (config.get("memory") or {}).get("provider")
    if not provider or not provider.strip():
        return None
    name = provider.strip()
    return name if (_is_directory(home / "plugins" / name) or
                    (installing is not None and (home / "plugins" / name).resolve() == installing.resolve())) else None
