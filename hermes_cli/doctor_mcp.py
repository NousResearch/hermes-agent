"""Static diagnostics for config.yaml MCP servers; live handshakes belong to doctor --live."""

import os

from hermes_cli.config import load_config
from hermes_cli.doctor_report import (
    Finding, _fail_and_issue, check_info, check_ok, check_warn, doctor_check,
)
from hermes_constants import display_hermes_home
from tools.mcp_tool_common import _parse_boolish
from tools.mcp_tool_config import (
    _ENV_VAR_PATTERN, _build_safe_env, _interpolate_env_vars, _resolve_stdio_command,
)
from tools.mcp_tool_errors import InvalidMcpUrlError, _validate_remote_mcp_url


def _problem(name, field: str, detail: str, issues: list[str]) -> None:
    location = f"mcp_servers.{name}" + (f".{field}" if field else "")
    _fail_and_issue(
        f"MCP server '{name}': {detail}", "",
        f"Fix {location} in {display_hermes_home()}/config.yaml — {detail}.", issues,
    )


def _string_mapping(name: str, field: str, value, issues: list[str]) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        _problem(name, field, f"'{field}' is not a mapping", issues)
        return False
    if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str)
           for k, v in value.items()):
        label = "header" if field == "headers" else "env"
        _problem(name, field, f"invalid {label} name/value; expected strings", issues)
        return False
    return True


def _unresolved_fields(value, path: str) -> list[str]:
    if isinstance(value, str):
        return [path] if _ENV_VAR_PATTERN.search(value) else []
    if isinstance(value, dict):
        return [p for key, item in value.items() for p in _unresolved_fields(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in _unresolved_fields(item, f"{path}[{i}]")]
    return []


def _check_http(name: str, entry: dict, issues: list[str]) -> bool:
    valid = _string_mapping(name, "headers", entry.get("headers"), issues)
    try:
        _validate_remote_mcp_url(name, entry.get("url"))
    except InvalidMcpUrlError:
        # Validator exceptions can contain credentials from the URL. Report the
        # field to fix without echoing configured values into terminal/log output.
        _problem(name, "url", "invalid url; expected an http(s) URL with a host", issues)
        valid = False
    if valid:
        check_ok(f"MCP server '{name}' (http)", "url and headers look valid")
    return valid


def _check_stdio(name: str, entry: dict, issues: list[str]) -> bool:
    valid = True
    args = entry.get("args", [])
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        _problem(name, "args", "invalid 'args'; expected a list of strings", issues)
        valid = False
    if not _string_mapping(name, "env", entry.get("env"), issues):
        return False
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        _problem(name, "command", "'command' must be a non-empty string", issues)
        return False

    try:
        child_env = _build_safe_env(entry.get("env"))
    except Exception:
        _problem(name, "env", "could not build child environment", issues)
        return False
    try:
        resolved, _ = _resolve_stdio_command(command, child_env)
    except Exception:
        _problem(name, "command", "could not resolve command; check command and PATH", issues)
        return False
    if os.sep not in resolved and not os.path.isabs(resolved):
        _problem(name, "command", "command not found on PATH; install it or set an absolute path", issues)
        valid = False
    elif not os.path.isfile(resolved):
        _problem(name, "command", "command not found; configure an existing executable file", issues)
        valid = False
    elif os.name != "nt" and not os.access(resolved, os.X_OK):
        _problem(name, "command", "command is not executable", issues)
        valid = False

    empty = sorted(k for k in (entry.get("env") or {}) if not child_env.get(k, "").strip())
    if empty:
        check_warn(f"MCP server '{name}' (stdio): empty env value(s)", ", ".join(empty))
        issues.append(f"Check mcp_servers.{name}.env: {', '.join(empty)}; empty values may be intentional.")
        valid = False
    if valid:
        check_ok(f"MCP server '{name}' (stdio)", "command resolves to an executable")
    return valid


def _check_server(name: str, entry: dict, issues: list[str]) -> bool:
    if not _parse_boolish(entry.get("enabled", True), default=True):
        check_info(f"MCP server '{name}': disabled; launch checks skipped")
        return True

    from hermes_cli.mcp_security import validate_mcp_server_entry
    if validate_mcp_server_entry(name, entry):
        check_warn(f"MCP server '{name}': blocked by MCP security validation")
        issues.append(f"Review mcp_servers.{name}; see the MCP Server Security section.")
        return False

    has_url, has_command = "url" in entry, "command" in entry
    if not has_url and not has_command:
        _problem(name, "", "no transport configured; needs 'command' (stdio) or 'url' (http)", issues)
        return False
    if has_url and has_command:
        check_warn(f"MCP server '{name}': both 'url' and 'command' set",
                   "HTTP transport is used; 'command' is ignored")

    entry = _interpolate_env_vars(entry)
    fields = ("url", "headers") if has_url else ("command", "args", "env")
    unresolved = [p for field in fields for p in _unresolved_fields(entry.get(field), field)]
    if unresolved:
        _problem(name, "", "unresolved env reference(s) in " + ", ".join(unresolved), issues)
        check_info(f"Set the referenced variables in {display_hermes_home()}/.env.")
        return False
    return (_check_http if has_url else _check_stdio)(name, entry, issues)


@doctor_check()
def _check_mcp_servers(should_fix: bool, f: Finding) -> None:
    from utils import env_var_enabled
    if env_var_enabled("HERMES_SAFE_MODE"):
        check_info("MCP servers disabled by safe mode; launch checks skipped")
        return
    try:
        servers = load_config().get("mcp_servers")
    except Exception:
        _fail_and_issue("Could not read mcp_servers configuration", "",
                        f"Check {display_hermes_home()}/config.yaml is readable and valid YAML.",
                        f.manual_issues)
        return
    if servers is None:
        servers = {}
    if not isinstance(servers, dict):
        _fail_and_issue("mcp_servers: malformed configuration", "expected a mapping",
                        "Fix mcp_servers in config.yaml — it must be a mapping of server names to settings.",
                        f.manual_issues)
        return
    if not servers:
        check_info("No MCP servers configured in config.yaml")
        return

    # The runtime loader also discovers/imports portable plugins. Keep this
    # section static by applying its interpolation and launch helpers directly
    # to the configured entries, before any runtime filtering can hide them.
    for name, entry in sorted(servers.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not name.strip():
            _problem(name, "", "invalid name; expected a non-empty string", f.manual_issues)
        elif not isinstance(entry, dict):
            _problem(name, "", "malformed entry; expected a mapping", f.manual_issues)
        else:
            try:
                _check_server(name, entry, f.manual_issues)
            except Exception:
                # An unexpected per-server failure must not hide the remaining
                # servers, abort doctor, or echo an exception containing secrets.
                _problem(name, "", "preflight failed; check the server configuration", f.manual_issues)
    check_info("Run `hermes doctor --live` to start configured servers and list their tools")
