"""Public-name catalogs for shared metrics: the only non-enum names a package may carry.

Every set here is something Nous itself publishes (slash-command registry, bundled and optional
skills, optional-mcps/ and plugin-catalog/ entries, built-in auxiliary tasks, shipped locales).
A name outside its set is reported as ``custom`` so user-defined identities never leave the machine.
Loaders are cached: the catalogs only change with the installed Hermes version.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

CUSTOM = "custom"


def _skill_dir_names(root: Path) -> frozenset[str]:
    try:
        return frozenset(p.parent.name.lower() for p in root.rglob("SKILL.md"))
    except OSError:
        return frozenset()


def _yaml_stems(root: Path) -> frozenset[str]:
    try:
        return frozenset(p.stem.lower() for p in root.glob("*.yaml"))
    except OSError:
        return frozenset()


@functools.cache
def slash_command_names() -> frozenset[str]:
    from hermes_cli.commands import COMMAND_REGISTRY

    return frozenset(command.name for command in COMMAND_REGISTRY)


@functools.cache
def bundled_skill_names() -> frozenset[str]:
    from hermes_constants import get_bundled_skills_dir, get_optional_skills_dir

    return _skill_dir_names(get_bundled_skills_dir(_REPO_ROOT / "skills")) | _skill_dir_names(
        get_optional_skills_dir(_REPO_ROOT / "optional-skills")
    )


@functools.cache
def mcp_catalog_names() -> frozenset[str]:
    from hermes_constants import get_optional_mcps_dir

    root = get_optional_mcps_dir(_REPO_ROOT / "optional-mcps")
    try:
        return frozenset(p.name.lower() for p in root.iterdir() if p.is_dir())
    except OSError:
        return frozenset()


@functools.cache
def plugin_catalog_names() -> frozenset[str]:
    return _yaml_stems(_REPO_ROOT / "plugin-catalog")


@functools.cache
def aux_task_names() -> frozenset[str]:
    from hermes_cli.main_provider_setup import _AUX_TASKS

    return frozenset(key for key, _name, _desc in _AUX_TASKS)


@functools.cache
def display_languages() -> frozenset[str]:
    return _yaml_stems(_REPO_ROOT / "locales")


@functools.cache
def provider_names() -> frozenset[str]:
    """Provider ids Hermes itself ships: built-in auth rows, overlays, alias tables and the
    in-tree ``plugins/model-providers`` profiles. Never the live registries (``PROVIDER_REGISTRY``,
    picker labels): ``$HERMES_HOME`` and pip provider plugins add their user-chosen names there."""
    import providers
    from hermes_cli.auth import _PROVIDER_ALIASES, BUILTIN_PROVIDER_IDS
    from hermes_cli.models_catalog_static import _PROVIDER_ALIASES as _CATALOG_ALIASES
    from hermes_cli.providers import ALIASES, HERMES_OVERLAYS

    providers.list_providers()  # runs discovery; bundled profiles land in the process-wide layer
    bundled = {
        alias for name, profile in providers._REGISTRY.items() if providers._SOURCES.get(name) == "bundled"
        for alias in (name, *profile.aliases)
    }
    return BUILTIN_PROVIDER_IDS.union(
        HERMES_OVERLAYS, ALIASES, _PROVIDER_ALIASES, _CATALOG_ALIASES, bundled, ("openrouter", CUSTOM)
    )


@functools.cache
def user_named_model_providers() -> frozenset[str]:
    """Providers whose model ids the user names: custom endpoints and loopback servers."""
    from urllib.parse import urlparse

    from hermes_cli.auth import PROVIDER_REGISTRY

    loopback = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    return frozenset(
        name for name, config in PROVIDER_REGISTRY.items()
        if urlparse(str(getattr(config, "inference_base_url", "") or "")).hostname in loopback
    ) | {CUSTOM}


@functools.cache
def custom_provider_aliases() -> frozenset[str]:
    """Provider ids Hermes routes through the generic ``custom`` provider (``ollama``, ``vllm``,
    ``llamacpp``...): shipped names, but the server and its model ids are the user's own."""
    from hermes_cli import auth, models, providers

    tables = (providers.ALIASES, getattr(auth, "_PROVIDER_ALIASES", {}), getattr(models, "_PROVIDER_ALIASES", {}))
    return frozenset(alias for table in tables for alias, canon in table.items() if canon == CUSTOM)


# ---- v4 gateway ----
@functools.cache
def bundled_platform_names() -> frozenset[str]:
    """Messaging platforms Hermes ships as ``plugins/platforms/<name>`` (the dir is the registered name)."""
    root = _REPO_ROOT / "plugins" / "platforms"
    try:
        return frozenset(p.name.lower() for p in root.iterdir() if (p / "plugin.yaml").is_file())
    except OSError:
        return frozenset()


@functools.cache
def catalog_platform_names() -> frozenset[str]:
    """``plugin-catalog/`` entries whose category is ``platform`` (in-tree only: never a network fetch)."""
    from hermes_cli.plugin_catalog import CATALOG_TIERS, load_catalog

    return frozenset(e.name for e in load_catalog() if e.category == "platform" and e.tier in CATALOG_TIERS)


def platform_metric_name(raw: object, core: frozenset[str]) -> str:
    """A messaging platform's public name: a core or bundled platform's own name, the catalog entry
    name of the installed catalog plugin that registered it, else ``plugin``. ``core`` is the
    contract's static platform vocabulary."""
    name = _norm(getattr(raw, "value", raw))
    if name in core or name in _safe(bundled_platform_names):
        return name
    if not name:
        return "plugin"
    from hermes_constants import get_hermes_home

    return _catalog_platform_owner(str(get_hermes_home()), name) or "plugin"


@functools.lru_cache(maxsize=64)
def _catalog_platform_owner(home: str, platform: str) -> str | None:
    """The catalog entry that installed the plugin registering ``platform`` in this profile. The
    plugin is located from where its adapter factory's code lives (not a name it could claim), and
    the catalog install is proven only by the installer-owned ``.install-metadata.json`` record,
    never by anything inside the plugin tree, so a URL install cannot claim a catalog name."""
    try:
        import inspect

        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins_provenance import read_sidecar_rows

        entry = platform_registry.get(platform)
        if entry is None or getattr(entry, "source", "") != "plugin":
            return None
        plugins_dir = (Path(home) / "plugins").resolve()
        code_path = Path(inspect.getfile(entry.adapter_factory)).resolve()
        if not code_path.is_relative_to(plugins_dir) or code_path.parent == plugins_dir:
            return None
        row = read_sidecar_rows(plugins_dir).get(code_path.relative_to(plugins_dir).parts[0])
        row = row if isinstance(row, dict) else {}
        block = row.get("catalog")
        name = _norm(block.get("name") if isinstance(block, dict) else row.get("catalog_name"))
        return name if name in _safe(catalog_platform_names) else None
    except Exception:
        logger.debug("Shared-metrics platform provenance unavailable", exc_info=True)
        return None


def _norm(value: object) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _safe(loader) -> frozenset[str]:
    try:
        return loader()
    except Exception:
        logger.debug("Shared-metrics catalog %s unavailable", loader.__name__, exc_info=True)
        return frozenset()


def slash_command_metric_name(raw: object) -> str:
    """Canonical registry name for a command or alias; skill and plugin commands stay anonymous."""
    name = _norm(raw).lstrip("/").split(" ", 1)[0]
    if not name:
        return "unknown"
    try:
        from hermes_cli.commands import resolve_command

        command = resolve_command(name)
    except Exception:
        command = None
    if command is not None and command.name in _safe(slash_command_names):
        return command.name
    try:
        from agent.skill_commands import resolve_skill_command_key

        if resolve_skill_command_key(name):
            return "skill"
        from hermes_cli.plugins import get_plugin_command_handler

        if get_plugin_command_handler(name) is not None:
            return "plugin"
    except Exception:
        pass
    return "unknown"


def skill_metric_name(raw: object) -> str:
    name = _norm(raw)
    return name if name in _safe(bundled_skill_names) else CUSTOM


_EXTENSION_CATALOGS = {
    "skill": bundled_skill_names,
    "mcp_server": mcp_catalog_names,
    "plugin": plugin_catalog_names,
}


def extension_metric_name(kind: str, raw: object) -> str:
    """A catalog entry's public name, else ``custom`` (URL/local installs of anything)."""
    loader = _EXTENSION_CATALOGS.get(kind)
    name = _norm(raw).rsplit("/", 1)[-1]
    return name if loader is not None and name in _safe(loader) else CUSTOM


def aux_task_metric_name(raw: object) -> str:
    name = _norm(raw)
    if not name:
        return "none"
    return name if name in _safe(aux_task_names) else "other"


def provider_metric_name(raw: object) -> str:
    """A shipped provider id; user-named providers (``custom:<name>``, unknown ids) and the local
    server aliases of ``custom`` read ``custom``."""
    from .shared_metrics_contract import PROVIDER_IDENTIFIER_MAX_LENGTH, _metric_identifier

    name = _metric_identifier(raw, max_length=PROVIDER_IDENTIFIER_MAX_LENGTH)
    if name == "unknown":
        return name
    if name.startswith(CUSTOM) or name in _safe(custom_provider_aliases):
        return CUSTOM
    return name if name in _safe(provider_names) or _models_dev_provider(name) else CUSTOM


@functools.lru_cache(maxsize=256)
def _models_dev_provider(name: str) -> bool:
    """A public models.dev provider id, from the local cache only (never a network call)."""
    try:
        from agent.models_dev import get_provider_info

        return get_provider_info(name, allow_network=False) is not None
    except Exception:
        return False


def model_metric_name(raw: object, provider: str, *, max_length: int) -> str:
    """The model id for a shipped remote provider; ``custom`` when the user names it (custom
    endpoint, loopback server), the provider is unknown (nothing proves the id is public), or it
    looks like a filesystem path or URL."""
    from .shared_metrics_contract import _metric_identifier

    if provider in _safe(user_named_model_providers):
        return CUSTOM
    model = _metric_identifier(raw, max_length=max_length)
    if model == "unknown":
        return model
    if provider == "unknown" or "://" in model or ":/" in model or model.endswith((".gguf", ".bin", ".safetensors")):
        return CUSTOM
    return model


def display_language_metric_name(raw: object) -> str:
    name = _norm(raw) or "en"
    return name if name in _safe(display_languages) else "other"
