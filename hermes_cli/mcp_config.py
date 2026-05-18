"""
MCP Server Management CLI — ``hermes mcp`` subcommand.

Implements ``hermes mcp add/remove/list/test/configure`` for interactive
MCP server lifecycle management (issue #690 Phase 2).

Relies on tools/mcp_tool.py for connection/discovery and keeps
configuration in ~/.hermes/config.yaml under the ``mcp_servers`` key.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.config import (
    cfg_get,
    load_config,
    save_config,
    get_env_value,
    save_env_value,
    get_hermes_home,  # noqa: F401 — used by test mocks
)
from hermes_cli.colors import Colors, color
from hermes_constants import display_hermes_home
from tools.mcp_tool import _ENV_VAR_PATTERN

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_MCP_PRESETS: Dict[str, Dict[str, Any]] = {
    "codex": {
        "command": "codex",
        "args": ["mcp-server"],
    },
}


# ─── UI Helpers ───────────────────────────────────────────────────────────────

def _info(text: str):
    print(color(f"  {text}", Colors.DIM))

def _success(text: str):
    print(color(f"  ✓ {text}", Colors.GREEN))

def _warning(text: str):
    print(color(f"  ⚠ {text}", Colors.YELLOW))

def _error(text: str):
    print(color(f"  ✗ {text}", Colors.RED))


def _confirm(question: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    try:
        val = input(color(f"  {question} [{default_str}]: ", Colors.YELLOW)).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return default
    if not val:
        return default
    return val in {"y", "yes"}


def _prompt(question: str, *, password: bool = False, default: str = "") -> str:
    from hermes_cli.cli_output import prompt as _shared_prompt
    return _shared_prompt(question, default=default, password=password)


# ─── Config Helpers ───────────────────────────────────────────────────────────

def _get_mcp_servers(config: Optional[dict] = None) -> Dict[str, dict]:
    """Return the ``mcp_servers`` dict from config, or empty dict."""
    if config is None:
        config = load_config()
    servers = config.get("mcp_servers")
    if not servers or not isinstance(servers, dict):
        return {}
    return servers


def _save_mcp_server(name: str, server_config: dict):
    """Add or update a server entry in config.yaml."""
    config = load_config()
    config.setdefault("mcp_servers", {})[name] = server_config
    save_config(config)


def _remove_mcp_server(name: str) -> bool:
    """Remove a server from config.yaml.  Returns True if it existed."""
    config = load_config()
    servers = config.get("mcp_servers", {})
    if name not in servers:
        return False
    del servers[name]
    if not servers:
        config.pop("mcp_servers", None)
    save_config(config)
    return True


def _env_key_for_server(name: str) -> str:
    """Convert server name to an env-var key like ``MCP_MYSERVER_API_KEY``."""
    return f"MCP_{name.upper().replace('-', '_')}_API_KEY"


def _parse_env_assignments(raw_env: Optional[List[str]]) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` strings from CLI args into an env dict."""
    parsed: Dict[str, str] = {}
    for item in raw_env or []:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid --env value '{text}' (expected KEY=VALUE)")
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env value '{text}' (missing variable name)")
        if not _ENV_VAR_NAME_RE.match(key):
            raise ValueError(f"Invalid --env variable name '{key}'")
        parsed[key] = value
    return parsed


def _apply_mcp_preset(
    name: str,
    *,
    preset_name: Optional[str],
    url: Optional[str],
    command: Optional[str],
    cmd_args: List[str],
    server_config: Dict[str, Any],
) -> tuple[Optional[str], Optional[str], List[str], bool]:
    """Apply a known MCP preset when transport details were omitted."""
    if not preset_name:
        return url, command, cmd_args, False

    preset = _MCP_PRESETS.get(preset_name)
    if not preset:
        raise ValueError(f"Unknown MCP preset: {preset_name}")

    if url or command:
        return url, command, cmd_args, False

    url = preset.get("url")
    command = preset.get("command")
    cmd_args = list(preset.get("args") or [])

    if url:
        server_config["url"] = url
    if command:
        server_config["command"] = command
    if cmd_args:
        server_config["args"] = cmd_args

    return url, command, cmd_args, True


# ─── Discovery (temporary connect) ───────────────────────────────────────────

def _probe_single_server(
    name: str, config: dict, connect_timeout: float = 30
) -> List[Tuple[str, str]]:
    """Temporarily connect to one MCP server, list its tools, disconnect.

    Returns list of ``(tool_name, description)`` tuples.
    Raises on connection failure.
    """
    from tools.mcp_tool import (
        _ensure_mcp_loop,
        _run_on_mcp_loop,
        _connect_server,
        _stop_mcp_loop,
    )

    _ensure_mcp_loop()

    tools_found: List[Tuple[str, str]] = []

    async def _probe():
        server = await asyncio.wait_for(
            _connect_server(name, config), timeout=connect_timeout
        )
        for t in server._tools:
            desc = getattr(t, "description", "") or ""
            # Truncate long descriptions for display
            if len(desc) > 80:
                desc = desc[:77] + "..."
            tools_found.append((t.name, desc))
        await server.shutdown()

    try:
        _run_on_mcp_loop(_probe(), timeout=connect_timeout + 10)
    except BaseException as exc:
        raise _unwrap_exception_group(exc) from None
    finally:
        _stop_mcp_loop()

    return tools_found


def _unwrap_exception_group(exc: BaseException) -> Exception:
    """Extract the root-cause exception from anyio TaskGroup wrappers.

    The MCP SDK uses anyio task groups, which wrap errors in
    ``BaseExceptionGroup`` / ``ExceptionGroup``.  This makes error
    messages opaque ("unhandled errors in a TaskGroup").  We unwrap
    to surface the real cause (e.g. "401 Unauthorized").
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    # Return a plain Exception so callers can catch normally
    if isinstance(exc, Exception):
        return exc
    return RuntimeError(str(exc))


# ─── hermes mcp add ──────────────────────────────────────────────────────────

def cmd_mcp_add(args):
    """Add a new MCP server with discovery-first tool selection."""
    name = args.name
    url = getattr(args, "url", None)
    # Read from `mcp_command` (set by --command via explicit dest) — see
    # mcp_add_p.add_argument("--command", dest="mcp_command", ...) in
    # hermes_cli/main.py for why the dest is renamed.
    command = getattr(args, "mcp_command", None)
    cmd_args = getattr(args, "args", None) or []
    auth_type = getattr(args, "auth", None)
    preset_name = getattr(args, "preset", None)
    raw_env = getattr(args, "env", None)

    server_config: Dict[str, Any] = {}
    try:
        explicit_env = _parse_env_assignments(raw_env)
        url, command, cmd_args, _preset_applied = _apply_mcp_preset(
            name,
            preset_name=preset_name,
            url=url,
            command=command,
            cmd_args=list(cmd_args),
            server_config=server_config,
        )
    except ValueError as exc:
        _error(str(exc))
        return

    if url and explicit_env:
        _error("--env is only supported for stdio MCP servers (--command or stdio presets)")
        return

    # Validate transport
    if not url and not command:
        _error("Must specify --url <endpoint>, --command <cmd>, or --preset <name>")
        _info("Examples:")
        _info('  hermes mcp add ink --url "https://mcp.ml.ink/mcp"')
        _info('  hermes mcp add github --command npx --args @modelcontextprotocol/server-github')
        _info('  hermes mcp add myserver --preset mypreset')
        return

    # Check if server already exists
    existing = _get_mcp_servers()
    if name in existing:
        if not _confirm(f"Server '{name}' already exists. Overwrite?", default=False):
            _info("Cancelled.")
            return

    # Build initial config
    if url:
        server_config["url"] = url
    else:
        server_config["command"] = command
        if cmd_args:
            server_config["args"] = cmd_args
        if explicit_env:
            server_config["env"] = explicit_env


    # ── Authentication ────────────────────────────────────────────────

    if url and auth_type == "oauth":
        print()
        _info(f"Starting OAuth flow for '{name}'...")
        oauth_ok = False
        try:
            from tools.mcp_oauth_manager import get_manager
            oauth_auth = get_manager().get_or_build_provider(name, url, None)
            if oauth_auth:
                server_config["auth"] = "oauth"
                _success("OAuth configured (tokens will be acquired on first connection)")
                oauth_ok=True
            else:
                _warning("OAuth setup failed — MCP SDK auth module not available")
        except Exception as exc:
            _warning(f"OAuth error: {exc}")

        if not oauth_ok:
            _info("This server may not support OAuth.")
            if _confirm("Continue without authentication?", default=True):
                # Don't store auth: oauth — server doesn't support it
                pass
            else:
                _info("Cancelled.")
                return

    elif url:
        # Prompt for API key / Bearer token for HTTP servers
        print()
        _info(f"Connecting to {url}")
        needs_auth = _confirm("Does this server require authentication?", default=True)
        if needs_auth:
            if auth_type == "header" or not auth_type:
                env_key = _env_key_for_server(name)
                existing_key = get_env_value(env_key)
                if existing_key:
                    _success(f"{env_key}: already configured")
                    api_key = existing_key
                else:
                    api_key = _prompt("API key / Bearer token", password=True)
                    if api_key:
                        save_env_value(env_key, api_key)
                        _success(f"Saved to {display_hermes_home()}/.env as {env_key}")

                # Set header with env var interpolation
                if api_key or existing_key:
                    server_config["headers"] = {
                        "Authorization": f"Bearer ${{{env_key}}}"
                    }

    # ── Discovery: connect and list tools ─────────────────────────────

    print()
    print(color(f"  Connecting to '{name}'...", Colors.CYAN))

    try:
        tools = _probe_single_server(name, server_config)
    except Exception as exc:
        _error(f"Failed to connect: {exc}")
        if _confirm("Save config anyway (you can test later)?", default=False):
            server_config["enabled"] = False
            _save_mcp_server(name, server_config)
            _success(f"Saved '{name}' to config (disabled)")
            _info("Fix the issue, then: hermes mcp test " + name)
        return

    if not tools:
        _warning("Server connected but reported no tools.")
        if _confirm("Save config anyway?", default=True):
            _save_mcp_server(name, server_config)
            _success(f"Saved '{name}' to config")
        return

    # ── Tool selection ────────────────────────────────────────────────

    print()
    _success(f"Connected! Found {len(tools)} tool(s) from '{name}':")
    print()
    for tool_name, desc in tools:
        short = desc[:60] + "..." if len(desc) > 60 else desc
        print(f"    {color(tool_name, Colors.GREEN):40s} {short}")
    print()

    # Ask: enable all, select, or cancel
    try:
        choice = input(
            color(f"  Enable all {len(tools)} tools? [Y/n/select]: ", Colors.YELLOW)
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        _info("Cancelled.")
        return

    if choice in {"n", "no"}:
        _info("Cancelled — server not saved.")
        return

    if choice in {"s", "select"}:
        # Interactive tool selection
        from hermes_cli.curses_ui import curses_checklist

        labels = [f"{t[0]}  —  {t[1]}" for t in tools]
        pre_selected = set(range(len(tools)))

        chosen = curses_checklist(
            f"Select tools for '{name}'",
            labels,
            pre_selected,
        )

        if not chosen:
            _info("No tools selected — server not saved.")
            return

        chosen_names = [tools[i][0] for i in sorted(chosen)]
        server_config.setdefault("tools", {})["include"] = chosen_names

        tool_count = len(chosen_names)
        total = len(tools)
    else:
        # Enable all (no filter needed — default behaviour)
        tool_count = len(tools)
        total = len(tools)

    # ── Save ──────────────────────────────────────────────────────────

    server_config["enabled"] = True
    _save_mcp_server(name, server_config)

    print()
    _success(f"Saved '{name}' to {display_hermes_home()}/config.yaml ({tool_count}/{total} tools enabled)")
    _info("Start a new session to use these tools.")


# ─── hermes mcp remove ───────────────────────────────────────────────────────

def cmd_mcp_remove(args):
    """Remove an MCP server from config."""
    name = args.name
    existing = _get_mcp_servers()

    if name not in existing:
        _error(f"Server '{name}' not found in config.")
        servers = list(existing.keys())
        if servers:
            _info(f"Available servers: {', '.join(servers)}")
        return

    if not _confirm(f"Remove server '{name}'?", default=True):
        _info("Cancelled.")
        return

    _remove_mcp_server(name)
    _success(f"Removed '{name}' from config")

    # Clean up OAuth tokens if they exist — route through MCPOAuthManager so
    # any provider instance cached in the current process (e.g. from an
    # earlier `hermes mcp test` in the same session) is evicted too.
    try:
        from tools.mcp_oauth_manager import get_manager
        get_manager().remove(name)
        _success("Cleaned up OAuth tokens")
    except Exception:
        pass


# ─── hermes mcp list ──────────────────────────────────────────────────────────

def cmd_mcp_list(args=None):
    """List all configured MCP servers."""
    servers = _get_mcp_servers()

    if not servers:
        print()
        _info("No MCP servers configured.")
        print()
        _info("Add one with:")
        _info('  hermes mcp add <name> --url <endpoint>')
        _info('  hermes mcp add <name> --command <cmd> --args <args...>')
        print()
        return

    print()
    print(color("  MCP Servers:", Colors.CYAN + Colors.BOLD))
    print()

    # Table header
    print(f"  {'Name':<16} {'Transport':<30} {'Tools':<12} {'Status':<10}")
    print(f"  {'─' * 16} {'─' * 30} {'─' * 12} {'─' * 10}")

    for name, cfg in servers.items():
        # Transport info
        if "url" in cfg:
            url = cfg["url"]
            # Truncate long URLs
            if len(url) > 28:
                url = url[:25] + "..."
            transport = url
        elif "command" in cfg:
            cmd = cfg["command"]
            cmd_args = cfg.get("args", [])
            if isinstance(cmd_args, list) and cmd_args:
                transport = f"{cmd} {' '.join(str(a) for a in cmd_args[:2])}"
            else:
                transport = cmd
            if len(transport) > 28:
                transport = transport[:25] + "..."
        else:
            transport = "?"

        # Tool count
        tools_cfg = cfg.get("tools", {})
        if isinstance(tools_cfg, dict):
            include = tools_cfg.get("include")
            exclude = tools_cfg.get("exclude")
            if include and isinstance(include, list):
                tools_str = f"{len(include)} selected"
            elif exclude and isinstance(exclude, list):
                tools_str = f"-{len(exclude)} excluded"
            else:
                tools_str = "all"
        else:
            tools_str = "all"

        # Enabled status
        enabled = cfg.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() in {"true", "1", "yes"}
        status = color("✓ enabled", Colors.GREEN) if enabled else color("✗ disabled", Colors.DIM)

        print(f"  {name:<16} {transport:<30} {tools_str:<12} {status}")

    print()


# ─── hermes mcp test ──────────────────────────────────────────────────────────

def cmd_mcp_test(args):
    """Test connection to an MCP server."""
    name = args.name
    servers = _get_mcp_servers()

    if name not in servers:
        _error(f"Server '{name}' not found in config.")
        available = list(servers.keys())
        if available:
            _info(f"Available: {', '.join(available)}")
        return

    cfg = servers[name]
    print()
    print(color(f"  Testing '{name}'...", Colors.CYAN))

    # Show transport info
    if "url" in cfg:
        _info(f"Transport: HTTP → {cfg['url']}")
    else:
        cmd = cfg.get("command", "?")
        _info(f"Transport: stdio → {cmd}")

    # Show auth info (masked)
    auth_type = cfg.get("auth", "")
    headers = cfg.get("headers", {})
    if auth_type == "oauth":
        _info("Auth: OAuth 2.1 PKCE")
    elif headers:
        for k, v in headers.items():
            if isinstance(v, str) and ("key" in k.lower() or "auth" in k.lower()):
                # Mask the value
                resolved = _ENV_VAR_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), v)
                if len(resolved) > 8:
                    masked = resolved[:4] + "***" + resolved[-4:]
                else:
                    masked = "***"
                print(f"    {k}: {masked}")
    else:
        _info("Auth: none")

    # Attempt connection
    start = time.monotonic()
    try:
        tools = _probe_single_server(name, cfg)
        elapsed_ms = (time.monotonic() - start) * 1000
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        _error(f"Connection failed ({elapsed_ms:.0f}ms): {exc}")
        return

    _success(f"Connected ({elapsed_ms:.0f}ms)")
    _success(f"Tools discovered: {len(tools)}")

    if tools:
        print()
        for tool_name, desc in tools:
            short = desc[:55] + "..." if len(desc) > 55 else desc
            print(f"    {color(tool_name, Colors.GREEN):36s} {short}")
    print()


# ─── hermes mcp login ────────────────────────────────────────────────────────

def cmd_mcp_login(args):
    """Force re-authentication for an OAuth-based MCP server.

    Deletes cached tokens (both on disk and in the running process's
    MCPOAuthManager cache) and triggers a fresh OAuth flow via the
    existing probe path.

    Use this when:
      - Tokens are stuck in a bad state (server revoked, refresh token
        consumed by an external process, etc.)
      - You want to re-authenticate to change scopes or account
      - A tool call returned ``needs_reauth: true``
    """
    name = args.name
    servers = _get_mcp_servers()

    if name not in servers:
        _error(f"Server '{name}' not found in config.")
        if servers:
            _info(f"Available servers: {', '.join(servers)}")
        return

    server_config = servers[name]
    url = server_config.get("url")
    if not url:
        _error(f"Server '{name}' has no URL — not an OAuth-capable server")
        return
    if server_config.get("auth") != "oauth":
        _error(f"Server '{name}' is not configured for OAuth (auth={server_config.get('auth')})")
        _info("Use `hermes mcp remove` + `hermes mcp add` to reconfigure auth.")
        return

    # Wipe both disk and in-memory cache so the next probe forces a fresh
    # OAuth flow.
    try:
        from tools.mcp_oauth_manager import get_manager
        mgr = get_manager()
        mgr.remove(name)
    except Exception as exc:
        _warning(f"Could not clear existing OAuth state: {exc}")

    print()
    _info(f"Starting OAuth flow for '{name}'...")

    # Probe triggers the OAuth flow (browser redirect + callback capture).
    try:
        tools = _probe_single_server(name, server_config)
        if tools:
            _success(f"Authenticated — {len(tools)} tool(s) available")
        else:
            _success("Authenticated (server reported no tools)")
    except Exception as exc:
        _error(f"Authentication failed: {exc}")


# ─── hermes mcp configure ────────────────────────────────────────────────────

def cmd_mcp_configure(args):
    """Reconfigure which tools are enabled for an existing MCP server."""
    import sys as _sys
    if not _sys.stdin.isatty():
        print("Error: 'hermes mcp configure' requires an interactive terminal.", file=_sys.stderr)
        _sys.exit(1)
    name = args.name
    servers = _get_mcp_servers()

    if name not in servers:
        _error(f"Server '{name}' not found in config.")
        available = list(servers.keys())
        if available:
            _info(f"Available: {', '.join(available)}")
        return

    cfg = servers[name]

    # Discover all available tools
    print()
    print(color(f"  Connecting to '{name}' to discover tools...", Colors.CYAN))

    try:
        all_tools = _probe_single_server(name, cfg)
    except Exception as exc:
        _error(f"Failed to connect: {exc}")
        return

    if not all_tools:
        _warning("Server reports no tools.")
        return

    # Determine which are currently enabled
    tools_cfg = cfg.get("tools", {})
    if isinstance(tools_cfg, dict):
        include = tools_cfg.get("include")
        exclude = tools_cfg.get("exclude")
    else:
        include = None
        exclude = None

    tool_names = [t[0] for t in all_tools]

    if include and isinstance(include, list):
        include_set = set(include)
        pre_selected = {
            i for i, tn in enumerate(tool_names) if tn in include_set
        }
    elif exclude and isinstance(exclude, list):
        exclude_set = set(exclude)
        pre_selected = {
            i for i, tn in enumerate(tool_names) if tn not in exclude_set
        }
    else:
        pre_selected = set(range(len(all_tools)))

    currently = len(pre_selected)
    total = len(all_tools)
    _info(f"Currently {currently}/{total} tools enabled for '{name}'.")
    print()

    # Interactive checklist
    from hermes_cli.curses_ui import curses_checklist

    labels = [f"{t[0]}  —  {t[1]}" for t in all_tools]

    chosen = curses_checklist(
        f"Select tools for '{name}'",
        labels,
        pre_selected,
    )

    if chosen == pre_selected:
        _info("No changes made.")
        return

    # Update config
    config = load_config()
    server_entry = cfg_get(config, "mcp_servers", name, default={})

    if len(chosen) == total:
        # All selected → remove include/exclude (register all)
        server_entry.pop("tools", None)
    else:
        chosen_names = [tool_names[i] for i in sorted(chosen)]
        server_entry.setdefault("tools", {})
        server_entry["tools"]["include"] = chosen_names
        server_entry["tools"].pop("exclude", None)

    config.setdefault("mcp_servers", {})[name] = server_entry
    save_config(config)

    new_count = len(chosen)
    _success(f"Updated config: {new_count}/{total} tools enabled")
    _info("Start a new session for changes to take effect.")


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def mcp_command(args):
    """Main dispatcher for ``hermes mcp`` subcommands."""
    action = getattr(args, "mcp_action", None)

    if action == "serve":
        from mcp_serve import run_mcp_server
        run_mcp_server(verbose=getattr(args, "verbose", False))
        return

    handlers = {
        "add": cmd_mcp_add,
        "remove": cmd_mcp_remove,
        "rm": cmd_mcp_remove,
        "list": cmd_mcp_list,
        "ls": cmd_mcp_list,
        "test": cmd_mcp_test,
        "configure": cmd_mcp_configure,
        "config": cmd_mcp_configure,
        "login": cmd_mcp_login,
    }

    handler = handlers.get(action)
    if handler:
        handler(args)
    else:
        # No subcommand — show list
        cmd_mcp_list()
        print(color("  Commands:", Colors.CYAN))
        _info("hermes mcp serve                              Run as MCP server")
        _info("hermes mcp add <name> --url <endpoint>        Add an MCP server")
        _info("hermes mcp add <name> --command <cmd>         Add a stdio server")
        _info("hermes mcp add <name> --preset <preset>       Add from a known preset")
        _info("hermes mcp remove <name>                      Remove a server")
        _info("hermes mcp list                               List servers")
        _info("hermes mcp test <name>                        Test connection")
        _info("hermes mcp configure <name>                   Toggle tools")
        _info("hermes mcp login <name>                       Re-authenticate OAuth")
        print()
