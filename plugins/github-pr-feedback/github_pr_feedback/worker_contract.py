"""Read worker opt-in state without loading another profile's plugins or secrets."""
import json
from pathlib import Path

import yaml

from hermes_cli.managed_scope import apply_managed_overlay


_PLUGIN_NAME = "github-pr-feedback"
_REQUIRED_HOOKS = frozenset({"pre_tool_call", "pre_kanban_complete"})


def configured_assignees(policy):
    names = {policy.assignee or ""}
    names.update(rule.assignee for rule in (*policy.assignee_rules, *policy.routing_rules))
    if policy.local_ci_audit is not None:
        names.add(policy.local_ci_audit.assignee)
    names.update(item.assignee for item in policy.merge_policies())
    if policy.repair_steward is not None:
        names.add(policy.repair_steward.assignee)
    for maintenance in policy.release_policies():
        names.add(maintenance.assignee)
        names.update(lane.assignee for lane in maintenance.lanes)
    return names


def _declared_hooks(plugin_dir: Path) -> set[str] | None:
    """Read a plugin manifest without importing the plugin package."""
    for filename, loader in (("plugin.yaml", yaml.safe_load), ("plugin.yml", yaml.safe_load),
                             ("plugin.json", json.loads)):
        manifest = plugin_dir / filename
        try:
            data = loader(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(data, dict) or data.get("name") != _PLUGIN_NAME:
            return None
        hooks = data.get("provides_hooks")
        if not isinstance(hooks, list) or not all(isinstance(item, str) for item in hooks):
            return None
        return set(hooks)
    return None


def _resolved_declared_hooks(home: Path, project_root: Path | None = None) -> set[str] | None:
    """Resolve the worker plugin contract from manifests only.

    A profile-local manifest wins over the trusted bundled manifest, matching the
    plugin discovery precedence without executing profile-owned code.
    """
    user_plugins = home / "plugins"
    candidates = []
    if project_root is not None:
        candidates.append(Path(project_root) / ".hermes" / "plugins" / _PLUGIN_NAME)
    candidates.append(user_plugins / _PLUGIN_NAME)
    try:
        categories = tuple(path for path in user_plugins.iterdir() if path.is_dir())
    except OSError:
        categories = ()
    candidates.extend(category / _PLUGIN_NAME for category in categories)
    for candidate in candidates:
        if candidate.exists():
            return _declared_hooks(candidate)
    return _declared_hooks(Path(__file__).resolve().parents[1])


def worker_contract_enabled(
    root: Path, assignee: str, *, project_root: Path | None = None
) -> bool:
    if not assignee or assignee in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in assignee):
        return False
    home = root if assignee == "default" else root / "profiles" / assignee
    try:
        config = yaml.safe_load((home / "config.yaml").read_text())
    except (OSError, UnicodeError, yaml.YAMLError):
        return False
    config = apply_managed_overlay(config) if isinstance(config, dict) else {}
    plugins = config.get("plugins")
    if not isinstance(plugins, dict):
        return False
    enabled, disabled = plugins.get("enabled"), plugins.get("disabled", [])
    if disabled is None:
        disabled = []
    if (not isinstance(enabled, list) or not all(isinstance(item, str) for item in enabled)
            or _PLUGIN_NAME not in enabled
            or not isinstance(disabled, list)
            or not all(isinstance(item, str) for item in disabled)
            or _PLUGIN_NAME in disabled):
        return False

    hooks = _resolved_declared_hooks(home, project_root)
    return hooks is not None and _REQUIRED_HOOKS <= hooks
