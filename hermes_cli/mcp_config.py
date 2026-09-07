"""MCP Server Management CLI — ``hermes mcp`` subcommand."""

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
from tools.mcp_tool_config import _ENV_VAR_PATTERN
from tools.mcp_tool_common import _env_ref_name

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_MCP_PRESETS: Dict[str, Dict[str, Any]] = {
    "codex": {"command": "codex", "args": ["mcp-server"]},
}


def _info(text: str): print(color(f"  {text}", Colors.DIM))
def _success(text: str): print(color(f"  ✓ {text}", Colors.GREEN))
def _warning(text: str): print(color(f"  ⚠ {text}", Colors.YELLOW))
def _error(text: str): print(color(f"  ✗ {text}", Colors.RED))


def _confirm(question: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    try:
        val = input(color(f"  {question} [{default_str}]: ", Colors.YELLOW)).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return default
    return val in {"y", "yes"} if val else default


def _print_tools(tools: List[Tuple[str, str]], width: int, desc_max: int) -> None:
    for tool_name, desc in tools:
        short = desc[:desc_max] + "..." if len(desc) > desc_max else desc
        print(f"    {color(tool_name, Colors.GREEN):{width}s} {short}")


def _get_mcp_servers(config: Optional[dict] = None) -> Dict[str, dict]:
    """Return the ``mcp_servers`` dict from config, or empty dict."""
    if config is None:
        config = load_config()
    servers = config.get("mcp_servers")
    return servers if servers and isinstance(servers, dict) else {}


def _tool_filters(cfg: dict) -> Tuple[Optional[list], Optional[list]]:
    """Return the ``(include, exclude)`` tool lists from a server config (non-empty lists only)."""
    tools_cfg = cfg.get("tools", {})
    if not isinstance(tools_cfg, dict):
        return None, None
    include, exclude = tools_cfg.get("include"), tools_cfg.get("exclude")
    return (
        include if include and isinstance(include, list) else None,
        exclude if exclude and isinstance(exclude, list) else None)


def _save_mcp_server(name: str, server_config: dict) -> bool:
    """Add or update a server entry in config.yaml.

    Returns False when a high-signal exfiltration-shaped stdio command is rejected (shell+egress
    payloads are blocked rather than whitelisting command families).
    """
    if not _validate_or_warn(name, server_config):
        return False
    config = load_config()
    config.setdefault("mcp_servers", {})[name] = server_config
    save_config(config)
    return True


def _validate_or_warn(name: str, server_config: dict) -> bool:
    """Print every suspicious-config issue as a warning; True when the entry is clean."""
    issues = validate_mcp_server_entry(name, server_config)
    for issue in issues:
        _warning(issue)
    if issues:
        _warning(f"Server '{name}' was NOT saved due to suspicious configuration.")
    return not issues


def _lookup_server(
    name: str, servers: Dict[str, dict], available_label: str = "Available servers"
) -> Optional[dict]:
    """Return the named server config, or print the not-found hint and return None."""
    if name in servers:
        return servers[name]
    _error(f"Server '{name}' not found in config.")
    if servers:
        _info(f"{available_label}: {', '.join(servers)}")
    return None


def _remove_mcp_server(name: str) -> bool:
    """Remove a server from config.yaml.  Returns True if it existed."""
    config = load_config()
    servers = config.get("mcp_servers") or {}
    if name not in servers:
        return False
    del servers[name]
    if not servers:
        config.pop("mcp_servers", None)
    save_config(config)
    return True


def _replace_mcp_servers(servers: Dict[str, dict]) -> Tuple[bool, List[str]]:
    """Replace the WHOLE ``mcp_servers`` map in config.yaml.

    Every entry is validated up front; any suspicious entry rejects the whole save (``(False,
    issues)``) so a bad paste can't be partially applied. An empty map removes the key entirely.
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

    The header template already stores ``Authorization: Bearer ${MCP_X_API_KEY}``; a token pasted
    with its own prefix would send ``Bearer Bearer <jwt>`` and get a 401.

    Normalize on save. (#37792)
    """
    if not isinstance(token, str):
        return token
    stripped = token.strip()
    if stripped[:7].lower() == "bearer ":
        return stripped[7:].strip()
    return stripped


def _bearer_auth_headers(name: str) -> Dict[str, str]:
    """Build the persisted Authorization header template for a named MCP server.

    The secret lives in the profile's ``.env``; CLI and Dashboard share this so they produce
    byte-equivalent config.
    """
    return {"Authorization": f"Bearer ${{{_env_key_for_server(name)}}}"}


def _save_bearer_auth_token(name: str, token: str) -> Dict[str, str]:
    """Persist a normalized Bearer token to ``.env`` and return the header template for config.yaml."""
    normalized = _strip_bearer_prefix(token)
    if not normalized or normalized.lower() == "bearer":
        raise ValueError("Bearer token is required")
    save_env_value(_env_key_for_server(name), normalized)
    return _bearer_auth_headers(name)


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
    server_config: Dict[str, Any]) -> tuple[Optional[str], Optional[str], List[str], bool]:
    """Apply a known MCP preset when transport details were omitted."""
    if not preset_name:
        return url, command, cmd_args, False
    preset = _MCP_PRESETS.get(preset_name)
    if not preset:
        raise ValueError(f"Unknown MCP preset: {preset_name}")
    if url or command:
        return url, command, cmd_args, False
    url, command = preset.get("url"), preset.get("command")
    cmd_args = list(preset.get("args") or [])
    if url:
        server_config["url"] = url
    if command:
        server_config["command"] = command
    if cmd_args:
        server_config["args"] = cmd_args
    return url, command, cmd_args, True


def _resolve_mcp_server_config(config: dict) -> dict:
    """Resolve ``${ENV}`` placeholders in a server config before connecting.

    Mirrors ``_load_mcp_config()`` in ``tools/mcp_tool.py``; without it the discovery probe sent
    literal placeholders in header templates and auth-requiring servers returned 401.

    Mirrors ``_load_mcp_config()`` in ``tools/mcp_tool.py``: load ``~/.hermes/.env`` into ``os.environ`` and
    recursively interpolate any ``${VAR}`` placeholders. The CLI builds header templates like
    ``Authorization: Bearer ${MCP_X_API_KEY}`` but the probe path never resolved them, so the discovery
    probe sent the literal placeholder and auth-requiring servers (e.g. n8n) returned 401 — while runtime
    tool loading worked because it interpolates. (#37792)
    """
    from tools.mcp_tool_config import _interpolate_env_vars
    from agent.secret_scope import current_secret_scope

    if current_secret_scope() is None:
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

    Returns ``(tool_name, description)`` tuples; raises on connection failure. ``details`` is an
    out-param filled with ``schema_chars``/``prompts``/``resources`` so the return shape stays stable.
    """
    issues = validate_mcp_server_entry(name, config)
    if issues:
        raise ValueError("; ".join(issues))

    from tools.mcp_tool_loop import _ensure_mcp_loop, _run_on_mcp_loop
    from tools.mcp_tool_discovery import _connect_server
    from tools.mcp_tool_lifecycle import _stop_mcp_loop_if_idle
    from tools.mcp_tool_common import _parse_boolish

    config = _resolve_mcp_server_config(config)
    if connect_timeout is None:
        try:
            connect_timeout = max(1.0, float(config.get("connect_timeout", 30)))
        except (TypeError, ValueError):
            connect_timeout = 30.0

    _ensure_mcp_loop()
    tools_found: List[Tuple[str, str]] = []

    async def _probe():
        server = await asyncio.wait_for(_connect_server(name, config), timeout=connect_timeout)
        try:
            for t in server._tools:
                desc = getattr(t, "description", "") or ""
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                tools_found.append((t.name, desc))
            if details is not None:
                # Per-tool registry-schema sizes (the SAME converted schema the agent registers) so
                # the desktop can estimate per-call token cost. Best-effort, absent on failure.
                try:
                    import json as _json
                    from tools.mcp_tool_schema import _convert_mcp_schema

                    details["schema_chars"] = {
                        t.name: len(_json.dumps(_convert_mcp_schema(name, t), separators=(",", ":"), default=str))
                        for t in server._tools
                    }
                except Exception:  # pragma: no cover — display-only extra
                    pass
                # Gate capability probes like runtime registration (_select_utility_schemas):
                # honour tools.prompts / tools.resources config AND only call a family the server
                # advertises — some servers hard-error on unknown prompts/list.
                tools_filter = config.get("tools") or {}
                advertised_caps = getattr(getattr(server, "initialize_result", None), "capabilities", None)

                def _wanted(cap: str) -> bool:
                    # No capability info captured (legacy fixtures / older servers) => always try.
                    if not _parse_boolish(tools_filter.get(cap), default=True):
                        return False
                    return advertised_caps is None or getattr(advertised_caps, cap, None) is not None

                # Best-effort: servers without the capability raise, which just means "0".
                if _wanted("prompts"):
                    try:
                        details["prompts"] = len((await server.session.list_prompts()).prompts)
                    except Exception:
                        pass
                if _wanted("resources"):
                    try:
                        details["resources"] = len((await server.session.list_resources()).resources)
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
    """True if an OAuth token file exists for ``name`` (a clean probe alone is not proof of auth)."""
    try:
        from tools.mcp_oauth import HermesTokenStorage
        return HermesTokenStorage(name).has_cached_tokens()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Could not check OAuth tokens for '%s': %s", name, exc)
        return True  # permissive: don't block a real success


def _unwrap_exception_group(exc: BaseException) -> Exception:
    """Extract the root cause from anyio ``ExceptionGroup`` wrappers so e.g. "401 Unauthorized" surfaces."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc if isinstance(exc, Exception) else RuntimeError(str(exc))


def _configure_http_auth(
    name: str, url: str, auth_type: Optional[str], server_config: Dict[str, Any]
) -> bool:
    """OAuth or Bearer-token setup for an HTTP server. False when the user cancelled."""
    print()
    if auth_type == "oauth":
        _info(f"Starting OAuth flow for '{name}'...")
        oauth_ok = False
        try:
            from tools.mcp_oauth_manager import get_manager
            if get_manager().get_or_build_provider(name, url, server_config.get("oauth")):
                server_config["auth"] = "oauth"
                _success("OAuth configured (tokens will be acquired on first connection)")
                oauth_ok = True
            else:
                _warning("OAuth setup failed — MCP SDK auth module not available")
        except Exception as exc:
            _warning(f"OAuth error: {exc}")
        if not oauth_ok:
            _info("This server may not support OAuth.")
            if not _confirm("Continue without authentication?", default=True):
                _info("Cancelled.")
                return False
        return True

    _info(f"Connecting to {url}")
    needs_auth = _confirm("Does this server require authentication?", default=True)
    if needs_auth and (auth_type == "header" or not auth_type):
        env_key = _env_key_for_server(name)
        if get_env_value(env_key):
            _success(f"{env_key}: already configured")
            server_config["headers"] = _bearer_auth_headers(name)
        else:
            from hermes_cli.cli_output import prompt
            api_key = prompt("API key / Bearer token", default="", password=True)
            if api_key:
                server_config["headers"] = _save_bearer_auth_token(name, api_key)
                _success(f"Saved to {display_hermes_home()}/.env as {env_key}")
    return True


def _choose_tools(name: str, tools: List[Tuple[str, str]], server_config: Dict[str, Any]) -> Optional[int]:
    """Ask enable-all / select / cancel; returns the enabled-tool count or None when cancelled."""
    print()
    _success(f"Connected! Found {len(tools)} tool(s) from '{name}':")
    print()
    _print_tools(tools, 40, 60)
    print()
    try:
        choice = input(
            color(f"  Enable all {len(tools)} tools? [Y/n/select]: ", Colors.YELLOW)
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        _info("Cancelled.")
        return None
    if choice in {"n", "no"}:
        _info("Cancelled — server not saved.")
        return None
    if choice not in {"s", "select"}:
        return len(tools)
    from hermes_cli.curses_ui import curses_checklist

    labels = [f"{t[0]}  —  {t[1]}" for t in tools]
    chosen = curses_checklist(f"Select tools for '{name}'", labels, set(range(len(tools))))
    if not chosen:
        _info("No tools selected — server not saved.")
        return None
    chosen_names = [tools[i][0] for i in sorted(chosen)]
    server_config.setdefault("tools", {})["include"] = chosen_names
    return len(chosen_names)


def cmd_mcp_add(args):
    """Add a new MCP server with discovery-first tool selection."""
    name = args.name
    url = getattr(args, "url", None)
    # --command uses dest="mcp_command" (see hermes_cli/main.py for why the dest is renamed).
    command = getattr(args, "mcp_command", None)
    cmd_args = getattr(args, "args", None) or []
    if cmd_args and cmd_args[0] == "--":
        cmd_args = cmd_args[1:]
    auth_type = getattr(args, "auth", None)
    raw_connect_timeout = getattr(args, "connect_timeout", None)

    server_config: Dict[str, Any] = {}
    try:
        explicit_env = _parse_env_assignments(getattr(args, "env", None))
        url, command, cmd_args, _preset_applied = _apply_mcp_preset(
            name, preset_name=getattr(args, "preset", None), url=url, command=command,
            cmd_args=list(cmd_args), server_config=server_config)
    except ValueError as exc:
        _error(str(exc))
        return

    if url and explicit_env:
        _error("--env is only supported for stdio MCP servers (--command or stdio presets)")
        return
    if not url and not command:
        _error("Must specify --url <endpoint>, --command <cmd>, or --preset <name>")
        _info("Examples:")
        _info('  hermes mcp add ink --url "https://mcp.ml.ink/mcp"')
        _info('  hermes mcp add github --command npx --args @modelcontextprotocol/server-github')
        _info('  hermes mcp add myserver --preset mypreset')
        return

    if name in _get_mcp_servers() and not _confirm(
        f"Server '{name}' already exists. Overwrite?", default=False
    ):
        _info("Cancelled.")
        return

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

    if not _validate_or_warn(name, server_config):
        return
    if url and not _configure_http_auth(name, url, auth_type, server_config):
        return

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
        if _confirm("Save config anyway?", default=True) and _save_mcp_server(name, server_config):
            _success(f"Saved '{name}' to config")
        return

    tool_count = _choose_tools(name, tools, server_config)
    if tool_count is None:
        return
    server_config["enabled"] = True
    if _save_mcp_server(name, server_config):
        print()
        _success(
            f"Saved '{name}' to {display_hermes_home()}/config.yaml ({tool_count}/{len(tools)} tools enabled)"
        )
        _info("Start a new session to use these tools.")


def cmd_mcp_remove(args):
    """Remove an MCP server from config."""
    name = args.name
    if _lookup_server(name, _get_mcp_servers()) is None:
        return
    if not _confirm(f"Remove server '{name}'?", default=True):
        _info("Cancelled.")
        return
    _remove_mcp_server(name)
    _success(f"Removed '{name}' from config")
    # Route OAuth cleanup through MCPOAuthManager so any provider cached in this process (e.g. from
    # an earlier `hermes mcp test`) is evicted too.
    try:
        from tools.mcp_oauth_manager import get_manager
        get_manager().remove(name)
        _success("Cleaned up OAuth tokens")
    except Exception:
        pass


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
    print(f"  {'Name':<16} {'Transport':<30} {'Tools':<12} {'Status':<10}")
    print(f"  {'─' * 16} {'─' * 30} {'─' * 12} {'─' * 10}")

    for name, cfg in servers.items():
        if "url" in cfg:
            transport = cfg["url"]
        elif "command" in cfg:
            transport = cfg["command"]
            cmd_args = cfg.get("args", [])
            if isinstance(cmd_args, list) and cmd_args:
                transport = f"{transport} {' '.join(str(a) for a in cmd_args[:2])}"
        else:
            transport = "?"
        if len(transport) > 28:
            transport = transport[:25] + "..."

        include, exclude = _tool_filters(cfg)
        if include:
            tools_str = f"{len(include)} selected"
        elif exclude:
            tools_str = f"-{len(exclude)} excluded"
        else:
            tools_str = "all"

        enabled = cfg.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() in {"true", "1", "yes"}
        status = color("✓ enabled", Colors.GREEN) if enabled else color("✗ disabled", Colors.DIM)
        print(f"  {name:<16} {transport:<30} {tools_str:<12} {status}")
    print()


def cmd_mcp_test(args):
    """Test connection to an MCP server."""
    name = args.name
    cfg = _lookup_server(name, _get_mcp_servers(), "Available")
    if cfg is None:
        return
    print()
    print(color(f"  Testing '{name}'...", Colors.CYAN))
    if "url" in cfg:
        _info(f"Transport: HTTP → {cfg['url']}")
    else:
        _info(f"Transport: stdio → {cfg.get('command', '?')}")

    headers = cfg.get("headers", {})
    if cfg.get("auth", "") == "oauth":
        _info("Auth: OAuth 2.1 PKCE")
    elif headers:
        for k, v in headers.items():
            if isinstance(v, str) and ("key" in k.lower() or "auth" in k.lower()):
                # Mask the value (accepts ${VAR} and Cursor-style ${env:VAR})
                resolved = _ENV_VAR_PATTERN.sub(lambda m: os.getenv(_env_ref_name(m.group(1)), ""), v)
                masked = resolved[:4] + "***" + resolved[-4:] if len(resolved) > 8 else "***"
                print(f"    {k}: {masked}")
    else:
        _info("Auth: none")

    start = time.monotonic()
    try:
        tools = _probe_single_server(name, cfg)
    except Exception as exc:
        _error(f"Connection failed ({(time.monotonic() - start) * 1000:.0f}ms): {exc}")
        return
    _success(f"Connected ({(time.monotonic() - start) * 1000:.0f}ms)")
    _success(f"Tools discovered: {len(tools)}")
    if tools:
        print()
        _print_tools(tools, 36, 55)
    print()


def _reauth_oauth_server(name: str, server_config: dict) -> bool:
    """Force a fresh OAuth flow for one server. Returns True on success.

    Wipes cached OAuth state (disk + in-process MCPOAuthManager cache), re-probes to trigger the
    browser flow, and verifies a token actually landed. Shared by ``login`` and ``reauth``.
    """
    url = server_config.get("url")
    if not url:
        _error(f"Server '{name}' has no URL — not an OAuth-capable server")
        return False
    if server_config.get("auth") != "oauth":
        _error(f"Server '{name}' is not configured for OAuth (auth={server_config.get('auth')})")
        _info("Use `hermes mcp remove` + `hermes mcp add` to reconfigure auth.")
        return False

    try:
        from tools.mcp_oauth_manager import get_manager
        get_manager().remove(name)
    except Exception as exc:
        _warning(f"Could not clear existing OAuth state: {exc}")

    print()
    _info(f"Starting OAuth flow for '{name}'...")

    # The probe triggers the OAuth flow (browser redirect + callback capture). Honor the configured
    # connect_timeout, floored at 315s (the 300s OAuth callback window + headroom) — matching the GUI
    # re-auth path in web_server.py. force_interactive_oauth: `hermes mcp login` is explicitly
    # user-initiated even when stdin isn't a TTY (desktop / agent-spawned terminals), where
    # _is_interactive() alone would refuse to open a browser.
    try:
        from tools.mcp_oauth import force_interactive_oauth

        try:
            _login_connect_timeout = float(server_config.get("connect_timeout"))
        except (TypeError, ValueError):
            _login_connect_timeout = 0.0
        with force_interactive_oauth():
            tools = _probe_single_server(
                name, server_config, connect_timeout=max(_login_connect_timeout, 315.0)
            )
        # A clean probe is NOT proof of authentication: some servers (e.g. Google Drive) serve
        # initialize + tools/list without auth, so the flow may have failed (e.g. DCR 400 for
        # providers without RFC 7591) while the probe still lists tools. Verify a token landed.
        if not _oauth_tokens_present(name):
            _warning("Server responded, but no OAuth token was obtained — authentication did not complete.")
            print()
            _info(
                "Some providers (e.g. Google Drive, Atlassian) do not support "
                "automatic client registration. For those you must create an "
                "OAuth client yourself and add its credentials to config.yaml:"
            )
            print()
            for line in (
                "mcp_servers:", f"  {name}:", f"    url: {url}", "    auth: oauth", "    oauth:",
                '      client_id: "<your-oauth-client-id>"',
                '      client_secret: "<your-oauth-client-secret>"',
            ):
                print(color(f"    {line}", Colors.DIM))
            print()
            _info("Then re-run `hermes mcp login " + name + "`.")
            return False
        if tools:
            _success(f"Authenticated — {len(tools)} tool(s) available")
        else:
            _success("Authenticated (server reported no tools)")
        return True
    except Exception as exc:
        try:
            from tools.mcp_oauth import humanize_oauth_registration_error
            humanized = humanize_oauth_registration_error(name, exc, server_url=url)
        except Exception:
            humanized = None
        _error(f"Authentication failed: {humanized or exc}")
        return False


def cmd_mcp_login(args):
    """Force re-authentication for an OAuth-based MCP server (wipes cached tokens, re-runs the flow)."""
    cfg = _lookup_server(args.name, _get_mcp_servers())
    if cfg is not None:
        _reauth_oauth_server(args.name, cfg)


def cmd_mcp_reauth(args):
    """Re-authenticate one OAuth MCP server, or all of them sequentially.

    Serial-by-design: a human can only complete one browser OAuth flow at a time.

    This is the self-service fix for the recurring stale-client ritual in GH#36767 (and avoids the startup
    popup storm when several servers go stale at once).
    """
    servers = _get_mcp_servers()
    name = getattr(args, "name", None)
    if getattr(args, "all", False):
        oauth_servers = [(n, c) for n, c in servers.items() if c.get("auth") == "oauth" and c.get("url")]
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
    cfg = _lookup_server(name, servers)
    if cfg is not None:
        _reauth_oauth_server(name, cfg)


def _rebuild_exclude_list(
    name: str, exclude: list, tool_names: List[str], chosen: set, matches_name_filter
) -> List[str]:
    """New ``tools.exclude`` for an exclude-mode entry after a checklist edit.

    Stays in exclude mode rather than demoting the user's dynamic filter to a frozen include list:
    newly-unchecked tools are appended as literal excludes, re-checked tools drop their literal
    entries, and glob patterns are preserved (they keep excluding future vendor tools by design).
    """
    old_exclude = [str(p) for p in exclude]
    glob_entries = [p for p in old_exclude if "*" in p or "?" in p or "[" in p]
    literal_entries = {p for p in old_exclude if p not in glob_entries}
    unchecked = {tool_names[i] for i in range(len(tool_names)) if i not in chosen}
    checked = {tool_names[i] for i in chosen}
    new_literals = (literal_entries - checked) | {
        tn for tn in unchecked if not matches_name_filter(tn, set(old_exclude))}
    # A re-checked tool still matched by a kept glob can't be enabled without dropping the glob —
    # surface that instead of silently ignoring the click or silently freezing the config.
    glob_shadowed = sorted(
        tn for tn in checked if glob_entries and matches_name_filter(tn, set(glob_entries))
    )
    if glob_shadowed:
        _warning(
            f"{len(glob_shadowed)} re-enabled tool(s) still match glob "
            f"exclude pattern(s) {glob_entries} and stay excluded: "
            f"{', '.join(glob_shadowed[:5])}"
            f"{' ...' if len(glob_shadowed) > 5 else ''}. Remove the "
            f"pattern from mcp_servers.{name}.tools.exclude in "
            "config.yaml to enable them."
        )
    return glob_entries + sorted(new_literals)


def cmd_mcp_configure(args):
    """Reconfigure which tools are enabled for an existing MCP server."""
    import sys as _sys
    if not _sys.stdin.isatty():
        print("Error: 'hermes mcp configure' requires an interactive terminal.", file=_sys.stderr)
        _sys.exit(1)
    name = args.name
    cfg = _lookup_server(name, _get_mcp_servers(), "Available")
    if cfg is None:
        return

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

    include, exclude = _tool_filters(cfg)
    tool_names = [t[0] for t in all_tools]
    total = len(all_tools)

    # Same matching semantics as runtime registration (tools/mcp_tool.py): exact names or globs.
    try:
        from tools.mcp_tool_schema import matches_name_filter
    except ImportError:  # pragma: no cover — defensive fallback
        def matches_name_filter(tool_name, patterns):
            return tool_name in patterns

    patterns = {str(p) for p in (include or exclude or [])}
    if patterns:
        pre_selected = {
            i for i, tn in enumerate(tool_names) if matches_name_filter(tn, patterns) == bool(include)
        }
    else:
        pre_selected = set(range(total))

    _info(f"Currently {len(pre_selected)}/{total} tools enabled for '{name}'.")
    print()

    from hermes_cli.curses_ui import curses_checklist

    labels = [f"{t[0]}  —  {t[1]}" for t in all_tools]
    chosen = curses_checklist(f"Select tools for '{name}'", labels, pre_selected)
    if chosen == pre_selected:
        _info("No changes made.")
        return

    config = load_config()
    server_entry = cfg_get(config, "mcp_servers", name, default={})
    exclude_mode = bool(exclude) and not include

    if len(chosen) == total and not exclude_mode:
        server_entry.pop("tools", None)  # all selected → register all
    elif exclude_mode:
        new_exclude = _rebuild_exclude_list(name, exclude, tool_names, chosen, matches_name_filter)
        if not new_exclude:
            server_entry.pop("tools", None)
        else:
            server_entry.setdefault("tools", {})
            server_entry["tools"]["exclude"] = new_exclude
            server_entry["tools"].pop("include", None)
    else:
        server_entry.setdefault("tools", {})
        server_entry["tools"]["include"] = [tool_names[i] for i in sorted(chosen)]
        server_entry["tools"].pop("exclude", None)

    config.setdefault("mcp_servers", {})[name] = server_entry
    save_config(config)
    _success(f"Updated config: {len(chosen)}/{total} tools enabled")
    _info("Start a new session for changes to take effect.")


_MCP_USAGE = (
    "hermes mcp                                    Open the catalog picker (default)",
    "hermes mcp catalog                            List Nous-approved MCPs",
    "hermes mcp install <name>                     Install a catalog MCP",
    "hermes mcp serve                              Run as MCP server",
    "hermes mcp add <name> --url <endpoint>        Add a custom MCP server",
    "hermes mcp add <name> --command <cmd>         Add a stdio server",
    "hermes mcp add <name> --preset <preset>       Add from a known preset",
    "hermes mcp remove <name>                      Remove a server",
    "hermes mcp list                               List configured servers",
    "hermes mcp test <name>                        Test connection",
    "hermes mcp configure <name>                   Toggle tools",
    "hermes mcp login <name>                       Re-authenticate OAuth",
    "hermes mcp reauth <name> | --all              Re-auth one or all OAuth servers",
)


def mcp_command(args):
    """Main dispatcher for ``hermes mcp`` subcommands."""
    action = getattr(args, "mcp_action", None)
    if action == "serve":
        from mcp_serve import run_mcp_server
        run_mcp_server(verbose=getattr(args, "verbose", False))
        return
    if action in ("picker", "catalog", "install"):
        # Catalog subcommands live in mcp_picker / mcp_catalog; import lazily to keep this module cheap.
        from hermes_cli import mcp_picker

        if action == "picker":
            mcp_picker.run_picker()
        elif action == "catalog":
            mcp_picker.show_catalog()
        else:
            import sys as _sys
            rc = mcp_picker.install_by_name(getattr(args, "identifier", "") or "")
            if rc:
                _sys.exit(rc)
        return
    handler = {
        "add": cmd_mcp_add, "remove": cmd_mcp_remove, "rm": cmd_mcp_remove, "list": cmd_mcp_list,
        "ls": cmd_mcp_list, "test": cmd_mcp_test, "configure": cmd_mcp_configure,
        "config": cmd_mcp_configure, "login": cmd_mcp_login, "reauth": cmd_mcp_reauth,
    }.get(action)
    if handler:
        handler(args)
        return
    # No subcommand — drop the user into the catalog picker (same UX as `hermes plugin`).
    from hermes_cli.mcp_picker import run_picker
    run_picker()
    print(color("  Commands:", Colors.CYAN))
    for line in _MCP_USAGE:
        _info(line)
    print()
