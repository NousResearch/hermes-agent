"""Read worker opt-in state without loading another profile's plugins or secrets."""
import importlib.metadata
from pathlib import Path

import yaml

from hermes_cli.config import _expand_env_vars
from hermes_cli.managed_scope import apply_managed_overlay
from utils import env_var_enabled


_PLUGIN_NAME = "github-pr-feedback"
_PLUGIN_ENTRY_POINT_GROUP = "hermes_agent.plugins"
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
    """Read a native plugin manifest without importing the plugin package."""
    for filename in ("plugin.yaml", "plugin.yml"):
        manifest = plugin_dir / filename
        try:
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(data, dict) or data.get("name") != _PLUGIN_NAME:
            return None
        hooks = data.get("provides_hooks")
        if not isinstance(hooks, list) or not all(isinstance(item, str) for item in hooks):
            return None
        return set(hooks)
    return None


def _runtime_manifest_present(plugin_dir: Path) -> bool:
    """Match runtime discovery: malformed YAML is dropped, other overrides win."""
    if (plugin_dir / "plugin.json").is_file():
        return True
    for filename in ("plugin.yaml", "plugin.yml"):
        manifest = plugin_dir / filename
        if not manifest.is_file():
            continue
        try:
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            return False
        return isinstance(data, dict) and data.get("name") == _PLUGIN_NAME
    return False


def _resolved_declared_hooks(
    home: Path, project_root: Path | None = None
) -> tuple[set[str] | None, str]:
    """Resolve the worker plugin contract from manifests only.

    A profile-local manifest wins over the trusted bundled manifest, matching the
    plugin discovery precedence without executing profile-owned code.
    """
    user_plugins = home / "plugins"
    candidates: list[tuple[Path, str]] = []
    if project_root is not None and env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        candidates.append((
            Path(project_root) / ".hermes" / "plugins" / _PLUGIN_NAME,
            _PLUGIN_NAME,
        ))
        try:
            project_categories = tuple(
                path for path in (Path(project_root) / ".hermes" / "plugins").iterdir()
                if path.is_dir()
            )
        except OSError:
            project_categories = ()
        candidates.extend(
            (category / _PLUGIN_NAME, f"{category.name}/{_PLUGIN_NAME}")
            for category in project_categories
        )
    candidates.append((user_plugins / _PLUGIN_NAME, _PLUGIN_NAME))
    try:
        categories = tuple(path for path in user_plugins.iterdir() if path.is_dir())
    except OSError:
        categories = ()
    candidates.extend(
        (category / _PLUGIN_NAME, f"{category.name}/{_PLUGIN_NAME}")
        for category in categories
    )
    for candidate, key in candidates:
        if candidate.exists() and _runtime_manifest_present(candidate):
            # Override manifests are untrusted declarations. Do not claim the
            # worker is protected unless the bundled artifact is the one in use;
            # importing override code here would execute it during doctor checks.
            return None, key
    return _declared_hooks(Path(__file__).resolve().parents[1]), _PLUGIN_NAME


def _entrypoint_override_present(selected_key: str = _PLUGIN_NAME) -> bool:
    """Return whether an installed plugin wins the selected manifest key."""
    try:
        entry_points = importlib.metadata.entry_points()
        if hasattr(entry_points, "select"):
            candidates = entry_points.select(group=_PLUGIN_ENTRY_POINT_GROUP)
        elif isinstance(entry_points, dict):
            candidates = entry_points.get(_PLUGIN_ENTRY_POINT_GROUP, ())
        else:
            candidates = (
                entry_point for entry_point in entry_points
                if getattr(entry_point, "group", None) == _PLUGIN_ENTRY_POINT_GROUP
            )
        return any(getattr(entry_point, "name", None) == selected_key for entry_point in candidates)
    except Exception:
        # A metadata failure must not make doctor trust the bundled hooks.
        return True


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
    config = (
        apply_managed_overlay(_expand_env_vars(config))
        if isinstance(config, dict)
        else {}
    )
    plugins = config.get("plugins")
    if not isinstance(plugins, dict):
        return False
    enabled, disabled = plugins.get("enabled"), plugins.get("disabled", [])
    if disabled is None:
        disabled = []
    if (not isinstance(enabled, list) or not all(isinstance(item, str) for item in enabled)
            or not isinstance(disabled, list)
            or not all(isinstance(item, str) for item in disabled)
    ):
        return False

    hooks, key = _resolved_declared_hooks(home, project_root)
    manifest_names = {_PLUGIN_NAME, key}
    if not manifest_names.intersection(enabled) or manifest_names.intersection(disabled):
        return False
    if key in enabled and _entrypoint_override_present(key):
        return False
    return hooks is not None and _REQUIRED_HOOKS <= hooks
