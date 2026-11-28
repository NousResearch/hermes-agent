"""Config keys the runtime reads that the DEFAULT_CONFIG walk in ``_validate_config_key`` cannot see.

``_validate_config_key`` asks here only AFTER the schema walk refused a key, so nothing here can
override a schema verdict; it only vouches for an exact path (and what is below it) a reader uses.
Four owners supply them:

* Core selection keys (``browser.cloud_provider``, ``<category>.use_gateway``, ``stt.provider``):
  :func:`tools.tool_backend_helpers.selection_config_keys`, built from ``read_selection``'s tables.
* Core-owned unseeded keys outside that table, each listed with its reader: exact leaves in
  :data:`CORE_RUNTIME_KEYS`, open mappings in :data:`CORE_RUNTIME_SUBTREES`.
* MoA preset fields under any preset name (:func:`hermes_cli.moa_config.is_preset_field_path`). Only a
  preset's own fields and its aggregator slot fields qualify, not arbitrary descendants of ``moa.presets``.
* ACTIVE plugins: a ``config_keys`` list in ``plugin.yaml`` (or ``extensions.com.nousresearch.hermes``
  in a portable ``plugin.json``) names core-section paths the plugin's code reads, e.g.
  ``memory.retaindb``. Plugins are found and gated by the loader's own discovery
  (``plugins_discovery``), plus the memory provider ``memory.provider`` selects. A declared path is
  at least two segments deep, is ignored when DEFAULT_CONFIG already defines it, and covers itself
  and everything below it.

Nothing is cached: the lookup runs only for a key the schema refused, and a plugin installed or
enabled in a long-running process must count on the next ``config get/set``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

# Core-owned keys read at runtime but deliberately absent from DEFAULT_CONFIG -> their reader. An entry
# vouches for that exact path only, because its value is a scalar or list: a mapping typed below it
# (``display.tool_progress.typo``) is never read. ``*`` is one user-chosen segment (a task, a provider).
CORE_RUNTIME_KEYS: dict[str, str] = {
    "computer_use.backend": "plugins/computer_use/__init__.py configured_backend_name() (unset = cua)",
    "agent.reasoning_effort": "hermes_constants.py / agent/auxiliary_task_config.py",
    "auxiliary.*.fallback_chain": "agent/auxiliary_client.py (per-task fallback chain)",
    "auxiliary.*.max_concurrency": "agent/auxiliary_client.py (per-task concurrency cap)",
    "code_execution.timeout": "tools/code_execution_tool.py",
    "code_execution.max_tool_calls": "tools/code_execution_tool.py",
    "dashboard.ssh_isolated_idle_grace_s": "hermes_cli/web_server.py",
    "delegation.worktree_isolation": "tools/delegate_tool_config.py",
    "display.tool_progress": "gateway/display_config.py (per-platform tier default)",
    "display.cleanup_progress": "gateway/run_turn.py",
    "skills.creation_nudge_interval": "agent/agent_init.py",
    "stt.prompt": "tools/transcription_cloud.py / transcription_command.py",
    "terminal.docker_persist_across_processes": "tools/terminal_tool_backends.py",
    "terminal.docker_orphan_reaper": "tools/terminal_tool_backends.py",
    "terminal.lifetime_seconds": "tools/terminal_scope.py",
    "tts.speed": "tools/tts_tool_openai.py and siblings (global fallback speed)",
    "tts.*.speed": "tools/tts_tool_*.py (per-provider speed)",
    "tts.openai.base_url": "tools/tts_tool_openai.py",
    "worktree": "cli.py (hermes -w default)",
    "worktree_sync": "hermes_cli/cli_commands_mixin.py",
    "unauthorized_dm_decline_message": "gateway/config.py (top-level form)",
}

# Core-owned keys whose value is an open mapping: the path and everything below it is read.
CORE_RUNTIME_SUBTREES: dict[str, str] = {
    "dashboard.kanban": "plugins/kanban/dashboard/plugin_api.py (bundled dashboard tab preferences)",
    "agent.personalities": "hermes_cli/personality.py (user-named personalities)",
    "skills.config": "agent/skill_utils.py (settings SKILL.md frontmatter declares)",
    "tool_budget": "tools/budget_config.py (legacy spelling of the tool_output section)",
}


def _matches(key: str, path: str, *, subtree: bool) -> bool:
    parts, pattern = key.split("."), path.split(".")
    if len(parts) < len(pattern) or (not subtree and len(parts) != len(pattern)):
        return False
    return all(p == "*" or p == k for p, k in zip(pattern, parts))


def _covered(key: str, paths: Iterable[str], *, subtree: bool = True) -> bool:
    return any(_matches(key, path, subtree=subtree) for path in paths)


def _in_default_config(path: str) -> bool:
    from hermes_cli.config import DEFAULT_CONFIG

    node = DEFAULT_CONFIG
    for segment in path.split("."):
        if not isinstance(node, dict) or segment not in node:
            return False
        node = node[segment]
    return True


def _selected_memory_provider_manifest():
    """Manifest of the memory provider ``memory.provider`` selects. Memory providers are ``exclusive``
    plugins: the general gate never enables them, the selection does."""
    from hermes_cli.config import cfg_get, load_config
    from hermes_cli.plugins_manifest import parse_manifest_file

    name = cfg_get(load_config(), "memory", "provider")
    if not isinstance(name, str) or not name.strip():
        return None
    from plugins.memory import _explicitly_disabled, find_provider_dir

    provider_dir = find_provider_dir(name.strip())
    # The memory loader refuses a provider parked in ``plugins.disabled``; its keys are not live either.
    if provider_dir is None or _explicitly_disabled(name.strip(), provider_dir):
        return None
    manifest_file = next((f for f in (provider_dir / "plugin.yaml", provider_dir / "plugin.yml") if f.exists()), None)
    return parse_manifest_file(manifest_file, provider_dir, "memory", "memory") if manifest_file else None


def _active_manifests() -> Iterator:
    """Manifests of the plugins the loader would activate, by its own discovery, precedence and gate."""
    from hermes_cli.plugins_discovery import (
        _get_disabled_plugins, _get_enabled_plugins, collect_directory_manifests, discover_entrypoint_manifests,
        gate_manifest, resolve_manifest_winners,
    )
    from hermes_cli.plugins_manifest import manifest_key, requires_hermes_error

    manifests = collect_directory_manifests()
    # Same merge as PluginManager._discover_and_load_inner: a directory install keeps its identity.
    directory_keys = {manifest_key(m) for m in manifests}
    manifests += [m for m in discover_entrypoint_manifests() if manifest_key(m) not in directory_keys]
    disabled, enabled = _get_disabled_plugins(), _get_enabled_plugins()
    for manifest in resolve_manifest_winners(manifests).values():
        verdict = gate_manifest(manifest, disabled, enabled)
        if (verdict.action != "placeholder" or verdict.enabled) and requires_hermes_error(manifest) is None:
            yield manifest
    memory = _selected_memory_provider_manifest()
    if memory is not None:
        yield memory


def plugin_declared_config_keys() -> frozenset[str]:
    """``config_keys`` declared by active plugins, minus paths DEFAULT_CONFIG already defines."""
    keys = set()
    for manifest in _active_manifests():
        for path in manifest.config_keys:
            if _in_default_config(path):
                logger.debug("config_keys: ignoring %s from plugin %s (a DEFAULT_CONFIG key)", path, manifest.name)
            else:
                keys.add(path)
    return frozenset(keys)


def is_runtime_read_key(parts: list[str]) -> bool:
    """True when the split key ``parts`` is a MoA preset field, a core selection key, a core runtime key,
    or sits at/under a path an active plugin declares. Call only for a key the schema walk refused.

    The MoA rule needs the segments (a preset name may contain a dot); the other owners match the
    dotted join."""
    from hermes_cli.moa_config import is_preset_field_path
    from tools.tool_backend_helpers import selection_config_keys

    if is_preset_field_path(parts):
        return True
    dotted = ".".join(parts)
    if dotted in selection_config_keys() or _covered(dotted, CORE_RUNTIME_KEYS, subtree=False):
        return True
    if _covered(dotted, CORE_RUNTIME_SUBTREES):
        return True
    return _covered(dotted, plugin_declared_config_keys())
