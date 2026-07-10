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
from hermes_cli.mcp_security import validate_mcp_server_entry
from tools.mcp_tool import _ENV_VAR_PATTERN, _env_ref_name

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


def _save_mcp_server(name: str, server_config: dict) -> bool:
    """Add or update a server entry in config.yaml.

    Returns False when a high-signal exfiltration-shaped stdio command is
    rejected. MCP stdio servers are user-chosen local commands, so this blocks
    shell+egress payloads rather than whitelisting command families.
    """
    issues = validate_mcp_server_entry(name, server_config)
    if issues:
        for issue in issues:
            _warning(issue)
        _warning(f"Server '{name}' was NOT saved due to suspicious configuration.")
        return False
    config = load_config()
    config.setdefault("mcp_servers", {})[name] = server_config
    save_config(config)
    return True


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


def _replace_mcp_servers(servers: Dict[str, dict]) -> Tuple[bool, List[str]]:
    """Replace the WHOLE ``mcp_servers`` map in config.yaml.

    Unlike ``_save_mcp_server`` (per-key upsert), this sets the entire map so
    the GUI's mcp.json editor can delete servers, drop an ``enabled: false``
    flag (re-enable), or remove nested fields and have those *removals* land on
    disk.  A plain ``/api/config`` deep-merge can only add/override keys, never
    delete them — which is why edits appeared to succeed but the old entry
    survived (see MCP tab persistence bug).

    Every entry is validated up front; on any suspicious command/args the whole
    save is rejected (returns ``(False, issues)``) so a bad paste can't be
    partially applied.  An empty map removes the key entirely.
    """
    issues: List[str] = []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            issues.append(f"Server '{name}': expected an object")
            continue
        issues.extend(validate_mcp_server_entry(name, cfg))

    if issues:
        return False, issues

    config = load_config()
    if servers:
        config["mcp_servers"] = dict(servers)
    else:
        config.pop("mcp_servers", None)
    save_config(config)
    return True, []


def _env_key_for_server(name: str) -> str:
    """Convert server name to an env-var key like ``MCP_MYSERVER_API_KEY``."""
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", name.upper()).strip("_")
    return f"MCP_{suffix}_API_KEY"


def _strip_bearer_prefix(token: str) -> str:
    """Strip a leading ``Bearer `` from a pasted token.

    The header template stores ``Authorization: Bearer ${MCP_X_API_KEY}``, so
    if a user pastes a token that already includes the ``Bearer `` prefix the
    server receives ``Bearer Bearer <jwt>`` → 401. Normalize on save. (#37792)
    """
    if not isinstance(token, str):
        return token
    stripped = token.strip()
    if stripped[:7].lower() == "bearer ":
        return stripped[7:].strip()
    return stripped


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

def _resolve_mcp_server_config(config: dict) -> dict:
    """Resolve ``${ENV}`` placeholders in a server config before connecting.

    Mirrors ``_load_mcp_config()`` in ``tools/mcp_tool.py``: load
    ``~/.hermes/.env`` into ``os.environ`` and recursively interpolate any
    ``${VAR}`` placeholders. The CLI builds header templates like
    ``Authorization: Bearer ${MCP_X_API_KEY}`` but the probe path never
    resolved them, so the discovery probe sent the literal placeholder and
    auth-requiring servers (e.g. n8n) returned 401 — while runtime tool
    loading worked because it interpolates. (#37792)
    """
    from tools.mcp_tool import _interpolate_env_vars

    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv()
    except Exception:  # pragma: no cover — defensive
        pass
    return _interpolate_env_vars(config)


def _probe_single_server(
    name: str, config: dict, connect_timeout: Optional[float] = None, *, details: Optional[dict] = None
) -> List[Tuple[str, str]]:
    """Temporarily connect to one MCP server, list its tools, disconnect.

    Returns list of ``(tool_name, description)`` tuples.
    Raises on connection failure.

    ``details``: optional dict the probe fills with extra capability counts
    (``prompts``, ``resources``) — an out-param so the return shape stays
    stable for existing CLI callers.
    """
    issues = validate_mcp_server_entry(name, config)
    if issues:
        raise ValueError("; ".join(issues))

    from tools.mcp_tool import (
        _ensure_mcp_loop,
        _run_on_mcp_loop,
        _connect_server,
        _stop_mcp_loop_if_idle,
        _parse_boolish,
    )

    config = _resolve_mcp_server_config(config)
    if connect_timeout is None:
        raw_timeout = config.get("connect_timeout", 30)
        try:
            connect_timeout = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            connect_timeout = 30.0

    _ensure_mcp_loop()
    tools_found: List[Tuple[str, str]] = []

    async def _probe():
        server = await asyncio.wait_for(
            _connect_server(name, config), timeout=connect_timeout
        )
        try:
            for t in server._tools:
                desc = getattr(t, "description", "") or ""
                # Truncate long descriptions for display
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                tools_found.append((t.name, desc))
            if details is not None:
                # Gate the capability probes exactly like runtime utility-tool
                # registration (tools.mcp_tool._select_utility_schemas):
                #   1. honour the user's tools.prompts / tools.resources config
                #   2. only call a family the server actually advertises.
                # Without this the "Test server" probe fired prompts/list and
                # resources/list at every server unconditionally — so a server
                # that rejects those methods (e.g. Unreal's MCP server, which
                # answers "Call to unknown method 'prompts/list'") logged a hard
                # error, and setting tools.prompts: false did NOT suppress it.
                tools_filter = config.get("tools") or {}
                prompts_enabled = _parse_boolish(
                    tools_filter.get("prompts"), default=True
                )
                resources_enabled = _parse_boolish(
                    tools_filter.get("resources"), default=True
                )
                advertised_caps = getattr(
                    getattr(server, "initialize_result", None),
                    "capabilities",
                    None,
                )

                def _advertises(cap_attr: str) -> bool:
                    # When no capability info was captured (legacy fixtures /
                    # older servers) preserve the old always-try behaviour.
                    if advertised_caps is None:
                        return True
                    return getattr(advertised_caps, cap_attr, None) is not None

                # Capability probes are best-effort: servers without the
                # capability raise, which just means "0".
                if prompts_enabled and _advertises("prompts"):
                    try:
                        result = await server.session.list_prompts()
                        details["prompts"] = len(result.prompts)
                    except Exception:
                        pass
                if resources_enabled and _advertises("resources"):
                    try:
                        result = await server.session.list_resources()
                        details["resources"] = len(result.resources)
                    except Exception:
                        pass
        finally:
            await server.shutdown()

    try:
        _run_on_mcp_loop(_probe(), timeout=connect_timeout + 10)
    except BaseException as exc:
        raise _unwrap_exception_group(exc) from None
    finally:
        _stop_mcp_loop_if_idle()

    return tools_found


def _oauth_tokens_present(name: str) -> bool:
    """Return True if an OAuth token file exists on disk for ``name``.

    Used after ``hermes mcp login`` to distinguish a genuine authentication
    from a probe that succeeded only because the server allowed
    initialize/tools-list without auth (so no token was ever acquired).
    """
    try:
        from tools.mcp_oauth import HermesTokenStorage
        return HermesTokenStorage(name).has_cached_tokens()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Could not check OAuth tokens for '%s': %s", name, exc)
        # Be permissive on unexpected errors: don't block a real success.
        return True


def _mcp_oauth_config(config: dict) -> dict:
    oauth_config = config.get("oauth")
    return oauth_config if isinstance(oauth_config, dict) else {}


def _has_preregistered_oauth_client(config: dict) -> bool:
    return bool(_mcp_oauth_config(config).get("client_id"))


def _force_oauth_login(name: str, config: dict) -> bool:
    """Run an explicit OAuth flow for pre-registered MCP clients.

    Some MCP servers allow initialize/tools-list without auth, so probing the
    server is not enough to start the SDK's 401-driven OAuth flow.
    """
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool import _ensure_mcp_loop, _run_on_mcp_loop, _stop_mcp_loop

    url = config.get("url")
    oauth_config = _mcp_oauth_config(config)
    timeout = float(oauth_config.get("timeout", 300))

    _ensure_mcp_loop()
    try:
        return bool(
            _run_on_mcp_loop(
                get_manager().force_login(name, url, oauth_config),
                timeout=timeout + 10,
            )
        )
    finally:
        _stop_mcp_loop()


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
    if cmd_args and cmd_args[0] == "--":
        cmd_args = cmd_args[1:]
    auth_type = getattr(args, "auth", None)
    preset_name = getattr(args, "preset", None)
    raw_env = getattr(args, "env", None)
    raw_connect_timeout = getattr(args, "connect_timeout", None)

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
    if raw_connect_timeout is not None:
        server_config["connect_timeout"] = raw_connect_timeout

    issues = validate_mcp_server_entry(name, server_config)
    if issues:
        for issue in issues:
            _warning(issue)
        _warning(f"Server '{name}' was NOT saved due to suspicious configuration.")
        return

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
                        api_key = _strip_bearer_prefix(api_key)
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
            if _save_mcp_server(name, server_config):
                _success(f"Saved '{name}' to config (disabled)")
                _info("Fix the issue, then: hermes mcp test " + name)
        return

    if not tools:
        _warning("Server connected but reported no tools.")
        if _confirm("Save config anyway?", default=True):
            if _save_mcp_server(name, server_config):
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
    if _save_mcp_server(name, server_config):
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
                # Mask the value (accepts ${VAR} and Cursor-style ${env:VAR})
                resolved = _ENV_VAR_PATTERN.sub(lambda m: os.getenv(_env_ref_name(m.group(1)), ""), v)
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

def _reauth_oauth_server(name: str, server_config: dict) -> bool:
    """Force a fresh OAuth flow for one server. Returns True on success.

    Wipes cached OAuth state (disk + in-process MCPOAuthManager cache),
    re-probes to trigger the browser flow, and verifies a token actually
    landed before reporting success. Shared by ``hermes mcp login`` and
    ``hermes mcp reauth`` so both behave identically for a single server.
    """
    url = server_config.get("url")
    if not url:
        _error(f"Server '{name}' has no URL — not an OAuth-capable server")
        return False
    if server_config.get("auth") != "oauth":
        _error(f"Server '{name}' is not configured for OAuth (auth={server_config.get('auth')})")
        _info("Use `hermes mcp remove` + `hermes mcp add` to reconfigure auth.")
        return False

    # Wipe both disk and in-memory cache so the next probe forces a fresh
    # OAuth flow.
    try:
        from tools.mcp_oauth_manager import get_manager
        get_manager().remove(name)
    except Exception as exc:
        _warning(f"Could not clear existing OAuth state: {exc}")

    print()
    _info(f"Starting OAuth flow for '{name}'...")

    # Probe triggers the OAuth flow (browser redirect + callback capture).
    # Honor the server's configured connect_timeout so a human has enough
    # time to complete the browser sign-in; the 30s default is too tight for
    # an interactive OAuth round-trip. Floor at 315s — the OAuth callback
    # window (300s in mcp_oauth) plus headroom — matching the GUI re-auth
    # path in web_server.py so CLI and dashboard behave identically.
    try:
        _login_connect_timeout = server_config.get("connect_timeout")
        try:
            _login_connect_timeout = float(_login_connect_timeout)
        except (TypeError, ValueError):
            _login_connect_timeout = 0.0
        _login_connect_timeout = max(_login_connect_timeout, 315.0)
        tools = _probe_single_server(
            name, server_config, connect_timeout=_login_connect_timeout
        )
        # A clean probe is NOT proof of authentication. Some MCP servers
        # (notably Google's official Drive server) serve initialize +
        # tools/list WITHOUT auth, so the probe lists tools even when the
        # OAuth flow never completed — e.g. dynamic client registration
        # 400'd because the provider doesn't support RFC 7591. Reporting
        # "Authenticated — N tools" in that case is a false success: every
        # real tool call later hangs until timeout because there's no token.
        # Verify a token actually landed on disk before claiming success.
        if not _oauth_tokens_present(name):
            _warning(
                "Server responded, but no OAuth token was obtained — "
                "authentication did not complete."
            )
            print()
            _info(
                "Some providers (e.g. Google Drive, Atlassian) do not support "
                "automatic client registration. For those you must create an "
                "OAuth client yourself and add its credentials to config.yaml:"
            )
            print()
            print(color("    mcp_servers:", Colors.DIM))
            print(color(f"      {name}:", Colors.DIM))
            print(color(f"        url: {url}", Colors.DIM))
            print(color("        auth: oauth", Colors.DIM))
            print(color("        oauth:", Colors.DIM))
            print(color("          client_id: \"<your-oauth-client-id>\"", Colors.DIM))
            print(color("          client_secret: \"<your-oauth-client-secret>\"", Colors.DIM))
            print()
            _info("Then re-run `hermes mcp login " + name + "`.")
            return False
        if tools:
            _success(f"Authenticated — {len(tools)} tool(s) available")
        else:
            _success("Authenticated (server reported no tools)")
        return True
    except Exception as exc:
        _error(f"Authentication failed: {exc}")
        return False


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

    _reauth_oauth_server(name, servers[name])


def cmd_mcp_reauth(args):
    """Re-authenticate one OAuth MCP server, or all of them sequentially.

    ``hermes mcp reauth <name>`` re-auths a single server (same as ``login``).
    ``hermes mcp reauth --all`` discovers every ``auth: oauth`` server in
    config and re-auths them ONE AT A TIME.

    Serial-by-design: a human can only complete one browser OAuth flow at a
    time, so re-authing all servers concurrently would open N tabs at once
    and N-1 would time out. This is the self-service fix for the recurring
    stale-client ritual in GH#36767 (and avoids the startup popup storm when
    several servers go stale at once).
    """
    servers = _get_mcp_servers()
    do_all = getattr(args, "all", False)
    name = getattr(args, "name", None)

    if do_all:
        oauth_servers = [
            (n, c) for n, c in servers.items()
            if c.get("auth") == "oauth" and c.get("url")
        ]
        if not oauth_servers:
            _info("No OAuth-based MCP servers found in config.")
            return
        print()
        _info(f"Re-authenticating {len(oauth_servers)} OAuth server(s) one at a time...")
        succeeded = 0
        for n, c in oauth_servers:
            print()
            print(color(f"  ── {n} ──", Colors.CYAN + Colors.BOLD))
            if _reauth_oauth_server(n, c):
                succeeded += 1
        print()
        _success(f"Re-authenticated {succeeded}/{len(oauth_servers)} server(s)")
        return

    if not name:
        _error("Specify a server name, or use --all to re-auth every OAuth server.")
        _info("Usage: hermes mcp reauth <name>   |   hermes mcp reauth --all")
        return
    if name not in servers:
        _error(f"Server '{name}' not found in config.")
        if servers:
            _info(f"Available servers: {', '.join(servers)}")
        return

    _reauth_oauth_server(name, servers[name])


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

    # Catalog subcommands live in mcp_picker / mcp_catalog. Import lazily so
    # the original `mcp_config` module stays import-cheap.
    if action == "picker":
        from hermes_cli.mcp_picker import run_picker
        run_picker()
        return
    if action == "catalog":
        from hermes_cli.mcp_picker import show_catalog
        show_catalog()
        return
    if action == "install":
        from hermes_cli.mcp_picker import install_by_name
        import sys as _sys
        rc = install_by_name(getattr(args, "identifier", "") or "")
        if rc:
            _sys.exit(rc)
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
        "reauth": cmd_mcp_reauth,
    }

    handler = handlers.get(action)
    if handler:
        handler(args)
    else:
        # No subcommand — drop the user into the catalog picker. This is the
        # "try enabling and it flows you into setup" UX matching `hermes plugin`.
        from hermes_cli.mcp_picker import run_picker
        run_picker()
        print(color("  Commands:", Colors.CYAN))
        _info("hermes mcp                                    Open the catalog picker (default)")
        _info("hermes mcp catalog                            List Nous-approved MCPs")
        _info("hermes mcp install <name>                     Install a catalog MCP")
        _info("hermes mcp serve                              Run as MCP server")
        _info("hermes mcp add <name> --url <endpoint>        Add a custom MCP server")
        _info("hermes mcp add <name> --command <cmd>         Add a stdio server")
        _info("hermes mcp add <name> --preset <preset>       Add from a known preset")
        _info("hermes mcp remove <name>                      Remove a server")
        _info("hermes mcp list                               List configured servers")
        _info("hermes mcp test <name>                        Test connection")
        _info("hermes mcp configure <name>                   Toggle tools")
        _info("hermes mcp login <name>                       Re-authenticate OAuth")
        _info("hermes mcp reauth <name> | --all              Re-auth one or all OAuth servers")
        print()
