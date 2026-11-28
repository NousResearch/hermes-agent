"""Read every profile's enabled plugins in config order for the shared union.

Plugin admission owns writes; discovery never edits a profile's selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import logging

LOG = logging.getLogger(__name__)

# plugins.<RECORD> maps each plugin an update disabled to {enabled, reason}: what to undo once it fits.
RECORD = "disabled_by_update"


def _profiles_root() -> Path:
    # Plugin discovery and dependency publication must use the same home root.
    from pm.environments import dependency_home_root

    return dependency_home_root() / "profiles"


def _selection_readers(config_path: Path):
    """``(version(path), read(path))`` for a home's selection. PM's dependency runtime reads the
    LOCAL file it publishes to and compare-and-swaps (``pm.publication``): it has neither the
    remote config backend's client nor the plane credential, so a remote deployment never fails
    PM closed there. Every other process reads through the selected config backend; a file
    deployment never asks whether this is PM's runtime (the selector is all it needs)."""
    import os

    from hermes_cli import config_backend

    if os.environ.get(config_backend.BACKEND_ENV, "").strip().lower() in ("", "file"):
        return config_backend.config_version, config_backend.read_config_doc
    from pm.package import InstallError
    from pm.runtime import is_runtime
    try:
        in_runtime = is_runtime()
    except InstallError:  # an application process whose packaged PM runtime is damaged: not PM's runtime
        in_runtime = False
    if not in_runtime:
        return config_backend.config_version, config_backend.read_config_doc
    backend, home = config_backend.local_file_layer(config_path)
    return (lambda _path: backend.version(home)), (lambda _path: backend.read_user_doc(home))


def read_home_selection(home: Path) -> Optional[dict[str, Any]]:
    """The plugin/memory selection a home's config.yaml declares (None: no config yet).

    The public reader for anything that must agree with what PM installs for that home.
    An unreadable selection raises rather than shrinking the next dependency generation;
    empty YAML is an explicit empty configuration, as in the CLI loader.
    """
    config_path = home / "config.yaml"
    config_version, read_config_doc = _selection_readers(config_path)
    # Stat before the YAML stack: a home without config needs none (PM's own runtime tree has none).
    # Only a missing file means "no config"; a symlink loop or permission error still refuses.
    try:
        config_version(config_path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"could not read plugin selection: {config_path}") from exc
    # Missing YAML support is a broken runtime, not an empty plugin selection.
    from ruamel.yaml.error import YAMLError
    try:
        config = read_config_doc(config_path)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read plugin selection: {config_path}") from exc
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


_SELECTION_KEYS = ("plugins", "memory")  # the only sections ``enabled_plugins_ordered`` reads
_SELECTION_CACHE_VERSION = 1


def _selection_cache_path() -> Path:
    from pm.environments import dependency_home_root

    return dependency_home_root() / "cache" / "plugin-selection.json"


def _load_selection_cache() -> dict[str, Any]:
    import json

    try:
        data = json.loads(_selection_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    homes = data.get("homes") if isinstance(data, dict) and data.get("version") == _SELECTION_CACHE_VERSION else None
    return homes if isinstance(homes, dict) else {}


def _store_selection_cache(homes: dict[str, Any]) -> None:
    import json
    import os
    import tempfile

    path = _selection_cache_path()
    try:
        payload = json.dumps({"version": _SELECTION_CACHE_VERSION, "homes": homes})
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
    except (OSError, TypeError, ValueError):
        pass  # an unwritable or non-JSON selection just costs the parse next time


def _read_selection_cached(home: Path) -> Optional[dict[str, Any]]:
    """``read_home_selection(home)``, without the YAML parse when config.yaml is byte-for-byte the file
    a previous launch already parsed.

    Every launch checks the venv against EVERY profile's selection, and PM's runtime parses YAML in pure
    Python (~30 ms per config), so nine profiles cost a quarter of a second each time. The file signature
    is ``config_backend.config_version`` (mtime, size, inode and ctime: an in-place rewrite that keeps
    mtime and size still changes it). Only a successful read is stored, trimmed to the sections
    this module reads; a remote config backend, an unreadable signature or any cache problem takes the
    full read, which is also what raises for a broken config."""
    import os

    from hermes_cli import config_backend

    if os.environ.get(config_backend.BACKEND_ENV, "").strip().lower() not in ("", "file"):
        return read_home_selection(home)
    try:
        signature = list(config_backend.config_version(home / "config.yaml"))
    except OSError:
        return read_home_selection(home)  # no config.yaml, or a stat error that read_home_selection reports
    homes = _load_selection_cache()
    entry = homes.get(str(home))
    if isinstance(entry, dict) and entry.get("signature") == signature and isinstance(entry.get("selection"), dict):
        return entry["selection"]
    config = read_home_selection(home)
    if config is not None:
        homes[str(home)] = {"signature": signature, "selection": {k: config[k] for k in _SELECTION_KEYS if k in config}}
        _store_selection_cache(homes)
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
            config = _read_selection_cached(home)
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
    plugin dir exists (no dir = not a member) and plugins.disabled does not park it."""
    provider = (config.get("memory") or {}).get("provider")
    if not provider or not provider.strip():
        return None
    name = provider.strip()
    plugin_dir = home / "plugins" / name
    if installing is not None and plugin_dir.resolve() == installing.resolve():
        return name
    if not _is_directory(plugin_dir) or _parked(plugin_dir, config):
        return None
    return name


def _parked(plugin_dir: Path, config: dict[str, Any]) -> bool:
    """Whether plugins.disabled names this provider plugin. The provider loader honours the same
    spellings (plugins.memory._explicitly_disabled), so a parked provider is never loaded and
    installing its dependencies could only fail the generation for nothing."""
    disabled = (config.get("plugins") or {}).get("disabled") or []
    return bool(plugin_spellings(plugin_dir) & set(disabled))


def plugin_spellings(plugin_dir: Path, key: str | None = None) -> set[str]:
    """Every name a plugins list may use for this plugin: the selection key and its bare leaf, the
    directory name and the manifest ``name:``. Read from the plugin itself, so it holds in any home."""
    from pm.plugin_declarations import native_manifest_file, read_native_manifest

    names = {plugin_dir.name} | ({key, key.rsplit("/", 1)[-1]} if key else set())
    try:
        manifest_file = native_manifest_file(plugin_dir)
        if manifest_file is not None:
            names.add(str(read_native_manifest(manifest_file).get("name") or ""))
    except ValueError:
        pass  # no readable manifest: the other spellings still hold
    names.discard("")
    return names
