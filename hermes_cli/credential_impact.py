"""Read-only credential dependency map.

Answers "what breaks if I rotate or remove this env var?" by cross-
referencing the provider registry, auxiliary-task config, MCP server
config, and plugin/platform manifests (``requires_env`` / ``optional_env``)
for a given env-var NAME.

This is a diagnostic surface only — it never mutates any store. It
deliberately duplicates a small amount of directory-walking logic from
``hermes_cli.plugins`` instead of driving ``PluginManager.discover_and_load``,
because that path imports and executes plugin modules; this module must
stay side-effect-free so it is safe to run before deciding whether to
rotate or remove a credential.

Secrecy contract (same as ``credential_lifecycle.py``): no function here
logs, prints, or returns a credential VALUE — only var NAMES and config
PATHS.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

__all__ = ["CredentialImpact", "compute_impact", "credentials_command"]

# Plugin categories with their own discovery system (see plugins.py
# module docstring) — mirrors the skip-list PluginManager._scan_directory
# call sites use for the top-level bundled scan.
_OWN_DISCOVERY_CATEGORIES = frozenset(
    {"memory", "context_engine", "platforms", "model-providers"}
)


@dataclass
class CredentialImpact:
    """Cross-referenced consumers of a single env-var NAME."""

    var: str
    declared_in_env: bool = False
    providers: List[str] = field(default_factory=list)
    auxiliary_tasks: List[str] = field(default_factory=list)
    mcp_servers: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    plugins: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.declared_in_env
            or self.providers
            or self.auxiliary_tasks
            or self.mcp_servers
            or self.platforms
            or self.plugins
        )


def _providers_for_var(var: str) -> List[str]:
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:
        return []
    hits: List[str] = []
    for provider_id, cfg in PROVIDER_REGISTRY.items():
        try:
            if var in (cfg.api_key_env_vars or ()):
                hits.append(provider_id)
        except Exception:
            continue
    return sorted(hits)


def _auxiliary_tasks_for_providers(
    providers: List[str], config: Dict[str, Any]
) -> List[str]:
    """Auxiliary tasks whose resolved provider is in ``providers``.

    A task without its own ``provider`` override falls back to the main
    ``model.provider`` (mirrors ``agent.auxiliary_client`` resolution).
    """
    if not providers:
        return []
    aux = config.get("auxiliary")
    if not isinstance(aux, dict):
        return []
    provider_set = set(providers)
    main_provider = str((config.get("model") or {}).get("provider") or "").strip().lower()
    hits: List[str] = []
    for task, task_cfg in aux.items():
        if not isinstance(task_cfg, dict):
            continue
        task_provider = str(task_cfg.get("provider") or main_provider).strip().lower()
        if task_provider in provider_set:
            hits.append(task)
    return sorted(hits)


def _mcp_servers_for_var(var: str, config: Dict[str, Any]) -> List[str]:
    """MCP servers whose ``env:`` block declares this var NAME.

    Only catches explicit ``env:`` bindings — a server that inherits the
    var through the safe-env passthrough (see ``tools.mcp_tool._build_safe_env``)
    without naming it is not detectable from config alone.
    """
    mcp_cfg = config.get("mcp")
    servers = mcp_cfg.get("servers") if isinstance(mcp_cfg, dict) else None
    if not isinstance(servers, dict):
        return []
    hits: List[str] = []
    for name, server_cfg in servers.items():
        if not isinstance(server_cfg, dict):
            continue
        env = server_cfg.get("env")
        if isinstance(env, dict) and var in env:
            hits.append(str(name))
    return sorted(hits)


def _manifest_env_names(data: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for section in ("requires_env", "optional_env"):
        entries = data.get(section)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.append(entry["name"])
    return names


def _scan_manifest_dir(
    path: Path, *, skip_top_level: bool = False
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Walk ``plugin.yaml`` manifests under *path* (flat or one category deep).

    Returns ``(key, kind, raw_manifest_dict)`` tuples so callers can
    inspect ``requires_env``/``optional_env`` directly — mirrors the flat
    vs. category layout ``PluginManager._scan_directory`` supports, without
    depending on that private method or on ``PluginManifest`` (which drops
    ``optional_env``).
    """
    from utils import fast_safe_load

    results: List[Tuple[str, str, Dict[str, Any]]] = []
    if not path.is_dir():
        return results

    def _read(manifest_file: Path) -> Dict[str, Any]:
        try:
            return fast_safe_load(manifest_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        if skip_top_level and child.name in _OWN_DISCOVERY_CATEGORIES:
            continue
        manifest_file = child / "plugin.yaml"
        if not manifest_file.exists():
            manifest_file = child / "plugin.yml"
        if manifest_file.exists():
            data = _read(manifest_file)
            kind = str(data.get("kind", "standalone") or "standalone").strip().lower()
            key = str(data.get("name", child.name))
            results.append((key, kind, data))
            continue
        # No manifest at this level — treat as a category namespace and
        # look one level deeper (mirrors PluginManager._scan_directory).
        for grandchild in sorted(child.iterdir()):
            if not grandchild.is_dir():
                continue
            gc_manifest = grandchild / "plugin.yaml"
            if not gc_manifest.exists():
                gc_manifest = grandchild / "plugin.yml"
            if not gc_manifest.exists():
                continue
            data = _read(gc_manifest)
            kind = str(data.get("kind", "standalone") or "standalone").strip().lower()
            key = f"{child.name}/{grandchild.name}"
            results.append((key, kind, data))
    return results


def _platforms_and_plugins_for_var(var: str) -> Tuple[List[str], List[str]]:
    from hermes_cli.plugins import get_bundled_plugins_dir
    from hermes_constants import get_hermes_home

    platforms: set = set()
    plugins: set = set()

    repo_plugins = get_bundled_plugins_dir()
    scans = [
        _scan_manifest_dir(repo_plugins, skip_top_level=True),
        _scan_manifest_dir(repo_plugins / "platforms"),
        _scan_manifest_dir(get_hermes_home() / "plugins"),
    ]
    for manifests in scans:
        for key, kind, data in manifests:
            if var not in _manifest_env_names(data):
                continue
            if kind == "platform":
                platforms.add(key)
            else:
                plugins.add(key)
    return sorted(platforms), sorted(plugins)


def compute_impact(var: str) -> CredentialImpact:
    """Compute the read-only dependency map for env-var ``var``."""
    from hermes_cli.config import load_config_readonly, load_env

    var = var.strip()
    config = load_config_readonly()
    providers = _providers_for_var(var)
    auxiliary_tasks = _auxiliary_tasks_for_providers(providers, config)
    mcp_servers = _mcp_servers_for_var(var, config)
    platforms, plugins = _platforms_and_plugins_for_var(var)
    declared_in_env = var in load_env()

    return CredentialImpact(
        var=var,
        declared_in_env=declared_in_env,
        providers=providers,
        auxiliary_tasks=auxiliary_tasks,
        mcp_servers=mcp_servers,
        platforms=platforms,
        plugins=plugins,
    )


def _format_human(impact: CredentialImpact) -> str:
    lines = [f"Credential impact map: {impact.var}"]
    lines.append(f"  declared in .env: {'yes' if impact.declared_in_env else 'no'}")

    def _section(title: str, items: List[str]) -> None:
        lines.append(f"  {title} ({len(items)}):")
        if items:
            lines.extend(f"    - {item}" for item in items)
        else:
            lines.append("    (none)")

    _section("providers", impact.providers)
    _section("auxiliary tasks", impact.auxiliary_tasks)
    _section("mcp servers", impact.mcp_servers)
    _section("platforms", impact.platforms)
    _section("plugins", impact.plugins)

    if impact.is_empty:
        lines.append("")
        lines.append(
            "  no declared consumer found for this name — could be unused, "
            "a typo, or a dependency read dynamically from os.environ "
            "(not covered by static manifest/config scanning)."
        )
    return "\n".join(lines)


def _format_json(impact: CredentialImpact) -> str:
    import json

    return json.dumps(
        {
            "var": impact.var,
            "declared_in_env": impact.declared_in_env,
            "providers": impact.providers,
            "auxiliary_tasks": impact.auxiliary_tasks,
            "mcp_servers": impact.mcp_servers,
            "platforms": impact.platforms,
            "plugins": impact.plugins,
        },
        indent=2,
    )


def credentials_command(args: argparse.Namespace) -> int:
    """``hermes credentials`` dispatcher."""
    action = getattr(args, "credentials_action", None)
    if action != "impact":
        print(
            "usage: hermes credentials impact <VAR_NAME> [--json]",
            file=sys.stderr,
        )
        return 1

    var = (getattr(args, "var", "") or "").strip()
    if not var:
        print("error: VAR_NAME is required", file=sys.stderr)
        return 1

    impact = compute_impact(var)
    if getattr(args, "json", False):
        print(_format_json(impact))
    else:
        print(_format_human(impact))
    return 0
