"""MCP picker — interactive `hermes mcp picker` (also the default `hermes mcp`)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional

from hermes_cli.colors import Colors
from hermes_cli.cli_output import prompt_yes_no
from hermes_cli.curses_ui import curses_single_select
from hermes_cli.mcp_catalog import (
    CatalogEntry, CatalogError, catalog_diagnostics, install_entry, is_enabled, is_installed,
    list_catalog, installed_servers, remove_server, server_enabled, uninstall_entry, _say,
)
from hermes_cli.config import load_config, save_config

_STATUS_NOT_INSTALLED = "available"
_STATUS_DISABLED = "installed (disabled)"
_STATUS_ENABLED = "enabled"
_STATUS_CUSTOM_ENABLED = "custom — enabled"
_STATUS_CUSTOM_DISABLED = "custom — disabled"


@dataclass
class _Row:
    """A picker row. ``entry`` is set for catalog rows; custom MCPs carry only name/description/status."""

    name: str
    description: str
    status: str
    entry: Optional[CatalogEntry] = None  # None for non-catalog (custom) rows

    @property
    def is_custom(self) -> bool:
        return self.entry is None


def _build_rows() -> List[_Row]:
    """Return catalog rows + any custom (non-catalog) MCPs found in config."""
    catalog_entries = list_catalog()
    catalog_names = {e.name for e in catalog_entries}
    servers = installed_servers()

    rows: List[_Row] = []
    for entry in catalog_entries:
        cfg = servers.get(entry.name)
        if entry.name not in servers:
            status = _STATUS_NOT_INSTALLED
        elif cfg and server_enabled(cfg):
            status = _STATUS_ENABLED
        else:
            status = _STATUS_DISABLED
        rows.append(_Row(entry.name, entry.description, status, entry))
    # Custom (non-catalog) MCPs: the transport URL/command doubles as the description.
    for name, cfg in sorted(servers.items()):
        if name in catalog_names:
            continue
        status = _STATUS_CUSTOM_ENABLED if server_enabled(cfg) else _STATUS_CUSTOM_DISABLED
        rows.append(_Row(name, str(cfg.get("url") or cfg.get("command") or "(no transport)"), status))
    return rows


def _format_row(row: _Row) -> str:
    return f"{row.name:<18} {row.status:<24} {row.description}"



def _enable_disable(name: str, *, enable: bool) -> None:
    cfg = load_config()
    servers = cfg.get("mcp_servers") or {}
    server = servers.get(name)
    if not server:
        _say(f"  '{name}' is not installed.", Colors.RED)
        return
    server["enabled"] = enable
    cfg["mcp_servers"] = servers
    save_config(cfg)
    _say(
        f"  ✓ '{name}' {'enabled' if enable else 'disabled'}. "
        "Start a new Hermes session for changes to take effect."
    )


def _configure_tools(name: str) -> None:
    """Open the tool selection checklist for an already-installed MCP."""
    from argparse import Namespace
    from hermes_cli.mcp_config import cmd_mcp_configure

    cmd_mcp_configure(Namespace(name=name))


def _remove_custom(name: str) -> None:
    """Remove a non-catalog MCP entry from config.yaml."""
    if not is_installed(name):
        _say(f"  '{name}' is not configured.", Colors.RED)
        return
    if not prompt_yes_no(f"Remove '{name}' from mcp_servers?", default=False):
        return
    remove_server(name)
    _say(f"  ✓ Removed '{name}'")


def _install(entry: CatalogEntry, verb: str) -> bool:
    """Install *entry*, printing (not raising) a CatalogError. True on success."""
    try:
        install_entry(entry, enable=True)
    except CatalogError as exc:
        _say(f"  ✗ {verb} failed: {exc}", Colors.RED)
        return False
    return True


def _uninstall(name: str) -> None:
    if not prompt_yes_no(f"Uninstall '{name}'?", default=False):
        return
    if uninstall_entry(name):
        _say(
            f"  ✓ Uninstalled '{name}'. "
            "Credentials in .env preserved — delete manually if no longer needed."
        )
    else:
        _say(f"  '{name}' was not installed", Colors.DIM)


def _run_submenu(title: str, actions: list) -> None:
    """Show a single-select of ``(label, callback)`` pairs and run the picked callback."""
    choice = curses_single_select(title, [label for label, _ in actions])
    if choice is not None:
        actions[choice][1]()


def _handle_row(row: _Row) -> None:
    """Act on the picked row based on its current status."""
    if row.entry and not is_installed(row.name):
        _install(row.entry, "install")
        return
    if row.entry and not is_enabled(row.name):
        _enable_disable(row.name, enable=True)
        return
    if row.is_custom:
        enabled = is_enabled(row.name)
        _run_submenu(f"Action for '{row.name}' (custom)", [
            ("Configure tools (probe server + re-pick)", lambda: _configure_tools(row.name)),
            ("Enable" if not enabled else "Disable",
             lambda: _enable_disable(row.name, enable=not is_enabled(row.name))),
            ("Remove from config", lambda: _remove_custom(row.name))])
        return
    # Catalog row, installed + enabled
    print()
    _say(f"  '{row.name}' is already enabled.", Colors.DIM)
    _run_submenu(f"Action for '{row.name}'", [
        ("Configure tools (probe server + re-pick)", lambda: _configure_tools(row.name)),
        ("Disable (keep config, stop loading on next session)",
         lambda: _enable_disable(row.name, enable=False)),
        ("Uninstall (remove config and any cloned files)", lambda: _uninstall(row.name)),
        ("Reinstall (re-clone, re-prompt for credentials)",
         lambda: _install(row.entry, "reinstall"))])


def _print_rows_text(rows: List[_Row]) -> None:
    """Plain-text catalog dump: `hermes mcp catalog` output and the non-curses fallback."""
    print()
    if not rows:
        _say("  No MCPs in the catalog or configured.", Colors.DIM)
        print()
        return

    _say("  MCP Catalog + configured servers:", Colors.CYAN + Colors.BOLD)
    print()
    print(f"  {'Name':<18} {'Status':<24} Description")
    print(f"  {'-' * 18} {'-' * 24} {'-' * 11}")
    for row in rows:
        print(f"  {_format_row(row)}")
    print()
    _say("  Install: hermes mcp install <name>    Picker: hermes mcp", Colors.DIM)
    # Manifest-version warnings: the user's Hermes is too old to install everything listed.
    future = [d for d in catalog_diagnostics() if d[1] == "future_manifest"]
    if future:
        print()
        for name, _, _msg in future:
            _say(f"  ⚠ '{name}' requires a newer Hermes — run `hermes update` to install this entry.", Colors.YELLOW)
        print()
    print()


def show_catalog() -> None:
    """`hermes mcp catalog` — print the curated list + custom servers, no interaction."""
    _print_rows_text(_build_rows())


def run_picker() -> None:
    """`hermes mcp picker` (and default `hermes mcp`) — interactive selector; re-renders after each
    action until ESC/q."""
    while True:
        rows = _build_rows()
        if not rows or not sys.stdin.isatty():
            _print_rows_text(rows)  # non-interactive: degrade to the text dump
            return
        idx = curses_single_select(
            "MCP Catalog  —  ↑↓ navigate  ENTER act on entry  ESC/q quit",
            [_format_row(r) for r in rows],
        )
        if idx is None:
            return
        _handle_row(rows[idx])


def install_by_name(identifier: str) -> int:
    """`hermes mcp install <name>` — non-interactive entry-point."""
    from hermes_cli.mcp_catalog import get_entry

    entry = get_entry(identifier)
    if entry is None:
        _say(
            f"  ✗ '{identifier}' is not in the catalog. Run `hermes mcp catalog` to see available entries.",
            Colors.RED,
        )
        return 1
    return 0 if _install(entry, "install") else 1
