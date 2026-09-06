"""Read worker opt-in state without loading another profile's plugins or secrets."""
from pathlib import Path
import shutil
import tempfile

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


def worker_contract_enabled(root: Path, assignee: str) -> bool:
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

    manager = None
    try:
        from hermes_cli.plugins import PluginManager

        with tempfile.TemporaryDirectory(prefix="worker-contract-") as probe_dir:
            probe_home = Path(probe_dir)
            shutil.copy2(home / "config.yaml", probe_home / "config.yaml")
            user_plugins = home / "plugins"
            if user_plugins.is_dir():
                shutil.copytree(user_plugins, probe_home / "plugins", symlinks=True)
            manager = PluginManager(scope_key=str(probe_home))
            manager.discover_and_load()
            selected = manager._plugins.get(_PLUGIN_NAME)
            return bool(
                selected is not None
                and selected.enabled
                and selected.error is None
                and _REQUIRED_HOOKS <= set(selected.hooks_registered)
            )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    finally:
        if manager is not None:
            manager.unload()
