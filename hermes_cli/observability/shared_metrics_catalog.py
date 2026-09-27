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


def display_language_metric_name(raw: object) -> str:
    name = _norm(raw) or "en"
    return name if name in _safe(display_languages) else "other"
