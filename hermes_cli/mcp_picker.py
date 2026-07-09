"""MCP picker — interactive `hermes mcp picker` (also the default `hermes mcp`).

Lists every catalog entry plus any custom MCP servers the user has added via
``hermes mcp add``, lets them pick one, and routes to install / enable /
disable / uninstall / configure-tools flows.

Mirrors the `hermes plugin` picker UX: arrow keys to navigate, ENTER on a row
to act on it. The action depends on current status:

  not installed (catalog)   → install  (clone/bootstrap if needed, prompt for creds)
  installed / disabled      → enable
  installed / enabled       → submenu: configure tools / disable / uninstall / reinstall
  custom (non-catalog)      → submenu: configure tools / enable / disable / remove

The picker loops until the user hits ESC/q so they can manage multiple
entries in one session.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional

from hermes_cli.colors import Colors, color
from hermes_cli.cli_output import prompt_yes_no
from hermes_cli.curses_ui import curses_single_select
from hermes_cli.mcp_catalog import (
    CatalogEntry,
    CatalogError,
    catalog_diagnostics,
    install_entry,
    is_enabled,
    is_installed,
    list_catalog,
    installed_servers,
    uninstall_entry,
)
from hermes_cli.config import load_config, save_config


# ─── Status badges ────────────────────────────────────────────────────────────

_STATUS_NOT_INSTALLED = "available"
_STATUS_DISABLED = "installed (disabled)"
_STATUS_ENABLED = "enabled"
_STATUS_CUSTOM_ENABLED = "custom — enabled"
_STATUS_CUSTOM_DISABLED = "custom — disabled"


# ─── Row model — unifies catalog and custom entries ──────────────────────────


@dataclass
class _Row:
    """A row in the picker. ``entry`` is set for catalog rows; for custom
    user-added MCPs only ``name`` + ``description`` + status are populated."""

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

    rows: List[_Row] = []
    for entry in catalog_entries:
        if not is_installed(entry.name):
            status = _STATUS_NOT_INSTALLED
        elif is_enabled(entry.name):
            status = _STATUS_ENABLED
        else:
            status = _STATUS_DISABLED
        rows.append(
            _Row(
                name=entry.name,
                description=entry.description,
                status=status,
                entry=entry,
            )
        )

    # Custom MCPs the user added directly (not in the catalog)
    for name, cfg in sorted(installed_servers().items()):
        if name in catalog_names:
            continue
        enabled = cfg.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() in {"true", "1", "yes"}
        status = _STATUS_CUSTOM_ENABLED if enabled else _STATUS_CUSTOM_DISABLED
        # Use the transport URL/command as the "description" for custom rows
        desc = cfg.get("url") or cfg.get("command") or "(no transport)"
        rows.append(_Row(name=name, description=str(desc), status=status))

    return rows


def _format_row(row: _Row) -> str:
    return f"{row.name:<18} {row.status:<24} {row.description}"


# ─── Actions ──────────────────────────────────────────────────────────────────


def _enable_disable(name: str, *, enable: bool) -> None:
    cfg = load_config()
    servers = cfg.get("mcp_servers") or {}
    server = servers.get(name)
    if not server:
        print(color(f"  '{name}' is not installed.", Colors.RED))
        return
    server["enabled"] = enable
    cfg["mcp_servers"] = servers
    save_config(cfg)
    print(color(
        f"  ✓ '{name}' {'enabled' if enable else 'disabled'}. "
        "Start a new IX Agency session for changes to take effect.",
        Colors.GREEN,
    ))


def _configure_tools(name: str) -> None:
    """Open the tool selection checklist for an already-installed MCP.

    Delegates to the existing ``cmd_mcp_configure`` flow which probes the
    server, displays a checklist, and writes ``tools.include``.
    """
    import argparse
    from hermes_cli.mcp_config import cmd_mcp_configure

    cmd_mcp_configure(argparse.Namespace(name=name))


def _remove_custom(name: str) -> None:
    """Remove a non-catalog MCP entry from config.yaml."""
    cfg = load_config()
    servers = cfg.get("mcp_servers") or {}
    if name not in servers:
        print(color(f"  '{name}' is not configured.", Colors.RED))
        return
    if not prompt_yes_no(f"Remove '{name}' from mcp_servers?", default=False):
        return
    del servers[name]
    if not servers:
        cfg.pop("mcp_servers", None)
    else:
        cfg["mcp_servers"] = servers
    save_config(cfg)
    print(color(f"  ✓ Removed '{name}'", Colors.GREEN))


def _handle_row(row: _Row) -> None:
    """Act on the picked row based on its current status."""
    # === Catalog row, not yet installed ===
    if row.entry and not is_installed(row.name):
        try:
            install_entry(row.entry, enable=True)
        except CatalogError as exc:
            print(color(f"  ✗ install failed: {exc}", Colors.RED))
        return

    # === Catalog row, installed but disabled ===
    if row.entry and not is_enabled(row.name):
        _enable_disable(row.name, enable=True)
        return

    # === Catalog row, installed + enabled OR custom row ===
    if row.is_custom:
        # Custom (non-catalog) row submenu
        actions = [
            "Configure tools (probe server + re-pick)",
            "Enable" if not is_enabled(row.name) else "Disable",
            "Remove from config",
        ]
        choice = curses_single_select(f"Action for '{row.name}' (custom)", actions)
        if choice is None:
            return
        if choice == 0:
            _configure_tools(row.name)
        elif choice == 1:
            _enable_disable(row.name, enable=not is_enabled(row.name))
        elif choice == 2:
            _remove_custom(row.name)
        return

    # Catalog row, installed + enabled
    print()
    print(color(f"  '{row.name}' is already enabled.", Colors.DIM))
    actions = [
        "Configure tools (probe server + re-pick)",
        "Disable (keep config, stop loading on next session)",
        "Uninstall (remove config and any cloned files)",
        "Reinstall (re-clone, re-prompt for credentials)",
    ]
    choice = curses_single_select(f"Action for '{row.name}'", actions)
    if choice is None:
        return
    if choice == 0:
        _configure_tools(row.name)
    elif choice == 1:
        _enable_disable(row.name, enable=False)
    elif choice == 2:
        if prompt_yes_no(f"Uninstall '{row.name}'?", default=False):
            if uninstall_entry(row.name):
                print(color(
                    f"  ✓ Uninstalled '{row.name}'. "
                    "Credentials in .env preserved — delete manually if no longer needed.",
                    Colors.GREEN,
                ))
            else:
                print(color(f"  '{row.name}' was not installed", Colors.DIM))
    elif choice == 3:
        try:
            assert row.entry is not None
            install_entry(row.entry, enable=True)
        except CatalogError as exc:
            print(color(f"  ✗ reinstall failed: {exc}", Colors.RED))


# ─── Output / entry points ────────────────────────────────────────────────────


def _print_rows_text(rows: List[_Row]) -> None:
    """Plain-text catalog dump used as a fallback when curses can't run, and
    as the default output of `hermes mcp catalog`."""
    if not rows:
        print()
        print(color("  No MCPs in the catalog or configured.", Colors.DIM))
        print()
        return

    print()
    print(color("  MCP Catalog + configured servers:", Colors.CYAN + Colors.BOLD))
    print()
    print(f"  {'Name':<18} {'Status':<24} Description")
    print(f"  {'-' * 18} {'-' * 24} {'-' * 11}")
    for row in rows:
        print(f"  {_format_row(row)}")
    print()
    print(color(
        "  Install: hermes mcp install <name>    Picker: hermes mcp",
        Colors.DIM,
    ))

    # Surface manifest-version warnings so users know when their IX Agency is
    # too old to install everything in the catalog.
    diags = catalog_diagnostics()
    future = [d for d in diags if d[1] == "future_manifest"]
    if future:
        print()
        for name, _, msg in future:
            print(color(
                f"  ⚠ '{name}' requires a newer IX Agency — run `hermes update` "
                "to install this entry.",
                Colors.YELLOW,
            ))
        print()
    print()


def show_catalog() -> None:
    """`hermes mcp catalog` — print the curated list + custom servers, no interaction."""
    _print_rows_text(_build_rows())


def run_picker() -> None:
    """`hermes mcp picker` (and default `hermes mcp`) — interactive selector.

    Loops until the user hits ESC/q. After each action the picker re-renders
    so the user can manage several entries in one session.
    """
    if not sys.stdin.isatty():
        # Non-interactive shell: degrade to the text dump rather than failing.
        _print_rows_text(_build_rows())
        return

    while True:
        rows = _build_rows()
        if not rows:
            _print_rows_text(rows)
            return

        labels = [_format_row(r) for r in rows]
        idx = curses_single_select(
            "MCP Catalog  —  ↑↓ navigate  ENTER act on entry  ESC/q quit",
            labels,
        )
        if idx is None:
            return
        _handle_row(rows[idx])


def install_by_name(identifier: str) -> int:
    """`hermes mcp install <name>` — non-interactive entry-point.

    Returns 0 on success, non-zero on failure (so the CLI can propagate
    exit codes).
    """
    from hermes_cli.mcp_catalog import get_entry

    entry = get_entry(identifier)
    if entry is None:
        print(color(
            f"  ✗ '{identifier}' is not in the catalog. "
            "Run `hermes mcp catalog` to see available entries.",
            Colors.RED,
        ))
        return 1
    try:
        install_entry(entry, enable=True)
    except CatalogError as exc:
        print(color(f"  ✗ install failed: {exc}", Colors.RED))
        return 1
    return 0
