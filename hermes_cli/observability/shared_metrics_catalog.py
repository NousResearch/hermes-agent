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
    """Provider ids Hermes itself ships (auth registry, overlays, model catalog, aliases)."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import _KNOWN_PROVIDER_NAMES
    from hermes_cli.providers import ALIASES, HERMES_OVERLAYS

    return frozenset(PROVIDER_REGISTRY) | frozenset(HERMES_OVERLAYS) | frozenset(_KNOWN_PROVIDER_NAMES) | frozenset(ALIASES)


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
    """A shipped provider id; user-named providers (``custom:<name>``, unknown ids) read ``custom``."""
    from .shared_metrics_contract import PROVIDER_IDENTIFIER_MAX_LENGTH, _metric_identifier

    name = _metric_identifier(raw, max_length=PROVIDER_IDENTIFIER_MAX_LENGTH)
    if name == "unknown":
        return name
    if name.startswith(CUSTOM):
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
    endpoint, loopback server) or it looks like a filesystem path or URL."""
    from .shared_metrics_contract import _metric_identifier

    if provider in _safe(user_named_model_providers):
        return CUSTOM
    model = _metric_identifier(raw, max_length=max_length)
    if "://" in model or ":/" in model or model.endswith((".gguf", ".bin", ".safetensors")):
        return CUSTOM
    return model


def display_language_metric_name(raw: object) -> str:
    name = _norm(raw) or "en"
    return name if name in _safe(display_languages) else "other"
