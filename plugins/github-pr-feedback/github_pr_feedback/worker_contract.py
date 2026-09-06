"""Read worker opt-in state without loading another profile's plugins or secrets."""
from pathlib import Path

import yaml

from hermes_cli.managed_scope import apply_managed_overlay


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
    return (isinstance(enabled, list) and all(isinstance(item, str) for item in enabled)
            and "github-pr-feedback" in enabled
            and isinstance(disabled, list) and all(isinstance(item, str) for item in disabled)
            and "github-pr-feedback" not in disabled)
