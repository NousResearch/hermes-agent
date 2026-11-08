"""``hermes utcp`` — manage native UTCP tool providers.

Configuration lives in ``config.yaml`` under ``utcp``::

    utcp:
      enabled: true
      providers:
        - name: weather
          call_template_type: http
          url: https://weather.example.com/.well-known/utcp
          variables:
            API_KEY: env:WEATHER_UTCP_API_KEY

Commands mirror ``hermes mcp``: add/remove/list/status/tools/refresh.  All
network work is delegated to ``tools/utcp_tool.py``; this module only reads
configuration and formats output.  Secrets are never printed — only the names
of the environment variables a provider consumes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config, save_config


# ─── UI helpers ──────────────────────────────────────────────────────────────

def _info(text: str) -> None:
    print(color(f"  {text}", Colors.DIM))


def _success(text: str) -> None:
    print(color(f"  ✓ {text}", Colors.GREEN))


def _warning(text: str) -> None:
    print(color(f"  ⚠ {text}", Colors.YELLOW))


def _error(text: str) -> None:
    print(color(f"  ✗ {text}", Colors.RED))


def _header(text: str) -> None:
    print(color(text, Colors.CYAN))


def _utcp_section(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if config is None:
        config = load_config() or {}
    section = config.get("utcp")
    return section if isinstance(section, dict) else {}


def _providers(section: dict[str, Any]) -> list[dict[str, Any]]:
    providers = section.get("providers")
    if not isinstance(providers, (list, tuple)):
        return []
    return [entry for entry in providers if isinstance(entry, dict)]


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_utcp_list(args) -> int:
    """List configured providers (config view, no network)."""
    from tools.utcp_tool import load_utcp_config

    section = _utcp_section()
    settings, providers, problems = load_utcp_config(section)
    if not settings.enabled:
        _warning("UTCP support is disabled (utcp.enabled=false).")
        _info("Enable it with: hermes utcp add <name> --url <manual-url>")
        return 0
    if not providers and not problems:
        _info("No UTCP providers configured.")
        _info("Add one with: hermes utcp add <name> --url <manual-url>")
        return 0

    _header(f"UTCP providers ({len(providers)})")
    for provider in providers:
        state = "enabled" if provider.enabled else "disabled"
        print(
            f"  {color(provider.name, Colors.BOLD)}  "
            f"{provider.call_template_type}  {provider.template.get('url')}  "
            f"[{state}]"
        )
        if provider.env_refs:
            mapping = ", ".join(f"{name}=env:{env}" for name, env in sorted(provider.env_refs.items()))
            _info(f"variables: {mapping}")
            missing = provider.missing_env()
            if missing:
                _warning(f"unset environment variable(s): {', '.join(missing)}")
        if provider.timeout:
            _info(f"timeout: {provider.timeout:g}s  connect: {provider.connect_timeout:g}s")
    for problem in problems:
        _error(f"{problem.name}: {problem.message}")
    return 1 if problems else 0


def cmd_utcp_status(args) -> int:
    """Show live provider status (runs discovery, bounded)."""
    from tools.utcp_tool import discover_utcp_tools, get_utcp_status, load_utcp_settings

    section = _utcp_section()
    if not load_utcp_settings(section).enabled:
        _warning("UTCP support is disabled (utcp.enabled=false).")
        return 0

    try:
        discover_utcp_tools()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        _error(f"discovery failed: {exc}")
        return 1

    rows = get_utcp_status()
    if not rows:
        _info("No UTCP providers discovered.")
        return 0

    _header("UTCP status")
    failures = 0
    for row in rows:
        connected = row.get("connected")
        glyph = color("✓", Colors.GREEN) if connected else color("✗", Colors.RED)
        name = row.get("name") or "<unknown>"
        print(
            f"  {glyph} {color(str(name), Colors.BOLD)}  "
            f"{row.get('transport') or '-'}  "
            f"{row.get('registered', 0)}/{row.get('tools', 0)} tools registered"
        )
        if row.get("url"):
            _info(str(row["url"]))
        if row.get("discovery_ms") is not None:
            _info(f"discovery: {row['discovery_ms']:.0f} ms")
        if row.get("variables"):
            _info(f"credential vars: {', '.join(row['variables'])}")
        if row.get("error"):
            _warning(str(row["error"]))
        if not connected:
            failures += 1
    return 1 if failures else 0


def cmd_utcp_tools(args) -> int:
    """List discovered UTCP tools registered with the agent."""
    from tools.utcp_tool import discover_utcp_tools, get_utcp_tools, load_utcp_settings

    section = _utcp_section()
    if not load_utcp_settings(section).enabled:
        _warning("UTCP support is disabled (utcp.enabled=false).")
        return 0
    try:
        discover_utcp_tools()
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        _error(f"discovery failed: {exc}")
        return 1

    provider = getattr(args, "provider", None)
    rows = get_utcp_tools(provider)
    if not rows:
        _info("No UTCP tools discovered." + (f" (provider: {provider})" if provider else ""))
        return 0
    _header(f"UTCP tools ({len(rows)})")
    for row in rows:
        print(f"  {color(row['registry_name'], Colors.BOLD)}")
        _info(f"provider={row['provider']} tool={row['tool']} utcp={row['utcp_name']}")
        if row.get("description"):
            _info(row["description"][:160])
    return 0


def cmd_utcp_refresh(args) -> int:
    """Re-discover one provider (or all) and re-register its tools."""
    from tools.utcp_tool import refresh_utcp_provider

    provider = getattr(args, "provider", None)
    try:
        result = refresh_utcp_provider(provider)
    except Exception as exc:
        _error(str(exc))
        return 1
    if not result:
        _info("Nothing to refresh.")
        return 0
    for name, tools in sorted(result.items()):
        if tools:
            _success(f"{name}: {len(tools)} tool(s) refreshed")
        else:
            _warning(f"{name}: no tools registered (see `hermes utcp status`)")
    return 0


def cmd_utcp_add(args) -> int:
    """Add (or replace) a provider in config.yaml."""
    from tools.utcp_tool import UtcpConfigError, load_utcp_settings, validate_provider_entry

    name = (getattr(args, "name", "") or "").strip()
    url = (getattr(args, "url", "") or "").strip()
    if not name or not url:
        _error("usage: hermes utcp add <name> --url <manual-url>")
        return 2

    entry: dict[str, Any] = {
        "name": name,
        "call_template_type": (getattr(args, "transport", None) or "http"),
        "url": url,
        "enabled": True,
    }
    headers = getattr(args, "header", None) or []
    if headers:
        parsed_headers: dict[str, str] = {}
        for item in headers:
            if "=" not in item:
                _error(f"--header must be KEY=VALUE, got {item!r}")
                return 2
            key, value = item.split("=", 1)
            parsed_headers[key.strip()] = value
        entry["headers"] = parsed_headers
    variables = getattr(args, "variable", None) or []
    if variables:
        parsed_vars: dict[str, str] = {}
        for item in variables:
            if "=" not in item:
                _error(f"--variable must be NAME=env:ENV_VAR, got {item!r}")
                return 2
            key, value = item.split("=", 1)
            value = value.strip()
            parsed_vars[key.strip()] = value if value.startswith("env:") else f"env:{value}"
        entry["variables"] = parsed_vars
    if getattr(args, "timeout", None) is not None:
        entry["timeout"] = args.timeout

    config = load_config() or {}
    section = _utcp_section(config)
    settings = load_utcp_settings({**section, "enabled": True})

    # Validate before writing so a bad --url/--variable never lands in config.
    try:
        provider = validate_provider_entry(entry, settings)
    except UtcpConfigError as exc:
        _error(str(exc))
        return 1

    providers = [p for p in _providers(section) if p.get("name") != provider.name]
    providers.append(entry)
    section = dict(section)
    section["enabled"] = True
    section["providers"] = providers
    config["utcp"] = section
    save_config(config)

    _success(f"UTCP provider '{provider.name}' saved to config.yaml")
    _info(f"transport: {provider.call_template_type}  url: {provider.template.get('url')}")
    if provider.env_refs:
        mapping = ", ".join(f"{name}=env:{env}" for name, env in sorted(provider.env_refs.items()))
        _info(f"variables: {mapping} — set them in ~/.hermes/.env")
        missing = provider.missing_env()
        if missing:
            _warning(f"still unset: {', '.join(missing)} (discovery will fail until they are set)")
    _info("Verify with: hermes utcp status")
    return 0


def cmd_utcp_remove(args) -> int:
    """Remove a provider from config.yaml."""
    name = (getattr(args, "name", "") or "").strip()
    if not name:
        _error("usage: hermes utcp remove <name>")
        return 2
    config = load_config() or {}
    section = _utcp_section(config)
    providers = _providers(section)
    remaining = [p for p in providers if p.get("name") != name]
    if len(remaining) == len(providers):
        _warning(f"No UTCP provider named '{name}'.")
        return 1
    section = dict(section)
    section["providers"] = remaining
    config["utcp"] = section
    save_config(config)
    _success(f"Removed UTCP provider '{name}'")
    return 0


def utcp_command(args) -> int:
    """Dispatcher for ``hermes utcp <action>``; returns a process exit code."""
    action = getattr(args, "utcp_action", None)
    handlers = {
        "list": cmd_utcp_list,
        "ls": cmd_utcp_list,
        "status": cmd_utcp_status,
        "tools": cmd_utcp_tools,
        "refresh": cmd_utcp_refresh,
        "add": cmd_utcp_add,
        "remove": cmd_utcp_remove,
        "rm": cmd_utcp_remove,
    }
    handler = handlers.get(action) if action else None
    if handler is None:
        # No action or unknown action: show configuration + help text.
        print(color("UTCP (Universal Tool Calling Protocol) providers", Colors.CYAN))
        _info("hermes utcp list                       List configured providers")
        _info("hermes utcp status                     Show live provider status")
        _info("hermes utcp tools [provider]           List discovered tools")
        _info("hermes utcp refresh [provider]         Re-run discovery")
        _info("hermes utcp add <name> --url <url>     Add or update a provider")
        _info("hermes utcp remove <name>              Remove a provider")
        if action:
            _error(f"unknown action '{action}'")
            return 2
        return 0
    return handler(args) or 0
