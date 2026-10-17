"""Configuration validation for session-scoped ACP workspace routing."""

from typing import Any


def validate_acp_workspace(config: dict[str, Any], issues: list[Any]) -> None:
    """Validate the optional session-scoped ACP SSH workspace route."""
    from hermes_cli.config import _issue
    acp = config.get("acp")
    if acp is None:
        return
    if not isinstance(acp, dict):
        _issue(issues, "error", f"acp must be a dict, got {type(acp).__name__}",
               "Use: acp:\n  workspace: {}")
        return
    workspace = acp.get("workspace")
    if workspace in (None, {}):
        return
    if not isinstance(workspace, dict):
        _issue(issues, "error", f"acp.workspace must be a dict, got {type(workspace).__name__}",
               "Configure backend, host, user, port, key, and sync under acp.workspace")
        return
    if workspace.get("backend") != "ssh":
        _issue(issues, "error", f"acp.workspace.backend must be 'ssh', got {workspace.get('backend')!r}",
               "Set acp.workspace.backend to ssh")
    for key in ("host", "user"):
        value = workspace.get(key)
        if not isinstance(value, str) or not value.strip():
            _issue(issues, "error", f"acp.workspace.{key} must be a non-empty string",
                   f"Set acp.workspace.{key}")
    port = workspace.get("port", 22)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        _issue(issues, "error", f"acp.workspace.port must be an integer from 1 to 65535, got {port!r}",
               "Set acp.workspace.port to the SSH port (usually 22)")
    key = workspace.get("key", "")
    if not isinstance(key, str):
        _issue(issues, "error", f"acp.workspace.key must be a string, got {type(key).__name__}",
               "Set acp.workspace.key to an SSH private-key path or omit it")
    sync = workspace.get("sync", False)
    if not isinstance(sync, bool):
        _issue(issues, "error", f"acp.workspace.sync must be true or false, got {sync!r}",
               "Set acp.workspace.sync to false to use the remote workspace in place")
