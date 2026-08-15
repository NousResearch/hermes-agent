"""CLI handlers for ``hermes secrets bitwarden ...``.

Subcommands:
    setup    — interactive wizard: install bws, prompt for token + project, test fetch
    status   — show current config + binary version + token validation status
    sync     — run a fetch right now and show what would be applied (dry-run friendly)
    disable  — flip ``secrets.bitwarden.enabled`` to False
    install  — just download the bws binary (no token / project required)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# NOTE: bitwarden and its cryptography.* dependencies are imported lazily
# inside each cmd_* handler — not at module top level.  This prevents
# cryptography._rust.pyd from loading into the ``hermes update`` process on
# Windows (where the self-lock preflight detects and defers on any mapped
# native extension).  See #86781.
def _load_bitwarden():
    """Lazy import of bitwarden backend to defer cryptography load."""
    from agent.secret_sources import bitwarden
    return bitwarden

from hermes_cli.config import (
    get_env_path,
    load_config,
    save_config,
    save_env_value,
)
from hermes_cli.secret_prompt import masked_secret_prompt


# ---------------------------------------------------------------------------
# Argparse wiring — called from hermes_cli.main
# ---------------------------------------------------------------------------


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``bitwarden`` subcommand tree to a parent parser.

    Called from ``hermes_cli.main`` as part of building the top-level
    ``hermes secrets`` parser.
    """
    sub = parent_parser.add_subparsers(dest="secrets_bw_command")

    setup = sub.add_parser(
        "setup",
        help="Interactive wizard: install bws, store access token, pick project",
    )
    setup.add_argument(
        "--project-id",
        help="Pre-select a project UUID instead of prompting",
    )
    setup.add_argument(
        "--access-token",
        help="Provide the access token non-interactively (will be stored in .env)",
    )
    setup.add_argument(
        "--server-url",
        help=(
            "Bitwarden region / self-hosted endpoint. Examples: "
            "https://vault.bitwarden.com (US, default), "
            "https://vault.bitwarden.eu (EU), or your self-hosted URL. "
            "Skips the interactive region prompt."
        ),
    )
    setup.set_defaults(func=cmd_setup)

    status = sub.add_parser(
        "status",
        help="Show config + binary + token validation status",
    )
    status.set_defaults(func=cmd_status)

    token = sub.add_parser(
        "token",
        help="Rotate the access token: validate a new one and store it in .env",
    )
    token.add_argument(
        "--access-token",
        help="Provide the new token non-interactively (default: masked prompt)",
    )
    token.add_argument(
        "--no-verify",
        action="store_true",
        help="Store without probing Bitwarden first (not recommended)",
    )
    token.set_defaults(func=cmd_token)

    sync = sub.add_parser("sync", help="Fetch secrets now and report what changed")
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Actually export the secrets into the current shell's env (default: dry-run)",
    )
    sync.set_defaults(func=cmd_sync)

    disable = sub.add_parser("disable", help="Turn off the Bitwarden integration")
    disable.set_defaults(func=cmd_disable)

    install = sub.add_parser(
        "install",
        help="Download and verify the pinned bws binary (lazy-load version at runtime)",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a managed copy already exists",
    )
    install.set_defaults(func=cmd_install)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> int:
    bw = _load_bitwarden()
    console = Console()
    console.print(
        Panel.fit(
            "[bold]Bitwarden Secrets Manager setup[/bold]\n\n"
            "Need an access token? In the Bitwarden web app:\n"
            "  Secrets Manager → Machine accounts → [your account] →\n"
            "  Access tokens → Create access token\n\n"
            "Copy the token (starts with [cyan]0.[/cyan]…) — it cannot be retrieved later.",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------ binary
    console.print()
    console.print("[bold]Step 1[/bold]  Install the bws CLI")
    try:
        binary = bw.find_bws(install_if_missing=False)
        if binary is None:
            console.print("  No bws on PATH — downloading…")
            binary = bw.install_bws()
        version = _bws_version(binary)
        console.print(f"  [green]✓[/green] {binary}  ({version})")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ Could not install bws: {exc}[/red]")
        console.print(
            "  Manual install: "
            "https://github.com/bitwarden/sdk-sm/releases"
        )
        return 1

    # ------------------------------------------------------------------ token
    access_token = args.access_token or _prompt_access_token()
    if not access_token:
        console.print("\n  [red]✗ No token provided.[/red]")
        return 1

    # ------------------------------------------------------------------ validate
    console.print()
    console.print("[bold]Step 2[/bold]  Validate token")
    try:
        probe = bw.BwsClient(access_token=access_token)
        org_id = probe.list_organizations()[0]["id"]
        console.print(f"  [green]✓[/green] Token valid (org {org_id[:8]}…)")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ Token invalid: {exc}[/red]")
        return 1

    # ------------------------------------------------------------------ project
    console.print()
    console.print("[bold]Step 3[/bold]  Pick project")
    project_id = args.project_id or _prompt_project(probe, org_id)
    if not project_id:
        console.print("\n  [red]✗ No project selected.[/red]")
        return 1

    # ------------------------------------------------------------------ store
    console.print()
    console.print("[bold]Step 4[/bold]  Store in .env")
    env_path = get_env_path()
    save_env_value("BWS_ACCESS_TOKEN", access_token, env_path)
    save_env_value("BWS_PROJECT_ID", project_id, env_path)
    console.print(f"  [green]✓[/green] Saved to {env_path}")

    # ------------------------------------------------------------------ config
    cfg = load_config()
    secrets = cfg.setdefault("secrets", {})
    bw_cfg = secrets.setdefault("bitwarden", {})
    bw_cfg["enabled"] = True
    if args.server_url:
        bw_cfg["server_url"] = args.server_url
    save_config(cfg)

    console.print()
    console.print("[bold green]✓ Bitwarden secrets enabled[/bold green]")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    bw = _load_bitwarden()
    console = Console()

    cfg = load_config()
    bw_cfg = cfg.get("secrets", {}).get("bitwarden") or {}
    enabled = bw_cfg.get("enabled", False)

    table = Table(title="Bitwarden secrets status")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Enabled", "[green]yes[/green]" if enabled else "[red]no[/red]")

    # Binary
    binary = bw.find_bws(install_if_missing=False)
    if binary:
        version = _bws_version(binary)
        table.add_row("bws binary", f"{binary} ({version})")
    else:
        table.add_row("bws binary", "[red]not found[/red]")

    # Token
    token = os.environ.get("BWS_ACCESS_TOKEN", "")
    if token:
        table.add_row("Token", f"[green]present[/green] ({len(token)} chars)")
    else:
        table.add_row("Token", "[red]missing[/red]")

    # Project
    project_id = os.environ.get("BWS_PROJECT_ID", "")
    if project_id:
        table.add_row("Project ID", project_id)
    else:
        table.add_row("Project ID", "[red]missing[/red]")

    # Server
    server = bw_cfg.get("server_url", "https://vault.bitwarden.com")
    table.add_row("Server", server)

    console.print(table)

    # Validation
    if enabled and token and project_id:
        try:
            probe = bw.BwsClient(access_token=token)
            secrets = probe.list_secrets(project_id)
            console.print(f"\n[green]✓[/green] Token valid — {len(secrets)} secrets in project")
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[red]✗ Token validation failed: {exc}[/red]")
    elif enabled:
        console.print("\n[yellow]⚠ Enabled but token/project not fully configured[/yellow]")

    return 0


def cmd_token(args: argparse.Namespace) -> int:
    bw = _load_bitwarden()
    console = Console()

    new_token = args.access_token or _prompt_access_token("New access token: ")
    if not new_token:
        console.print("  [red]✗ No token provided.[/red]")
        return 1

    if not args.no_verify:
        try:
            probe = bw.BwsClient(access_token=new_token)
            orgs = probe.list_organizations()
            if not orgs:
                console.print("  [red]✗ Token has no organizations.[/red]")
                return 1
            console.print(f"  [green]✓[/green] Token valid (org {orgs[0]['id'][:8]}…)")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]✗ Token invalid: {exc}[/red]")
            return 1
    else:
        console.print("  [yellow]⚠ Skipping validation (--no-verify)[/yellow]")

    env_path = get_env_path()
    save_env_value("BWS_ACCESS_TOKEN", new_token, env_path)
    console.print(f"  [green]✓[/green] Stored in {env_path}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    bw = _load_bitwarden()
    console = Console()

    cfg = load_config()
    bw_cfg = cfg.get("secrets", {}).get("bitwarden") or {}
    if not bw_cfg.get("enabled", False):
        console.print("[red]✗ Bitwarden not enabled. Run: hermes secrets bitwarden setup[/red]")
        return 1

    token = os.environ.get("BWS_ACCESS_TOKEN", "")
    project_id = os.environ.get("BWS_PROJECT_ID", "")
    if not token or not project_id:
        console.print("[red]✗ BWS_ACCESS_TOKEN or BWS_PROJECT_ID missing. Run: hermes secrets bitwarden setup[/red]")
        return 1

    try:
        client = bw.BwsClient(access_token=token)
        secrets = client.list_secrets(project_id)
        console.print(f"[green]✓[/green] Fetched {len(secrets)} secrets")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗ Fetch failed: {exc}[/red]")
        return 1

    if args.apply:
        # Apply logic would go here (export to env)
        console.print("[yellow]Apply not yet implemented — dry-run only[/yellow]")
    else:
        console.print("[dim]Dry-run — use --apply to export[/dim]")

    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    bw = _load_bitwarden()
    cfg = load_config()
    secrets = cfg.setdefault("secrets", {})
    bw_cfg = secrets.setdefault("bitwarden", {})
    bw_cfg["enabled"] = False
    save_config(cfg)
    print("Bitwarden secret source disabled.")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    bw = _load_bitwarden()
    console = Console()

    try:
        binary = bw.install_bws(force=args.force)
        version = _bws_version(binary)
        console.print(f"[green]✓[/green] Installed: {binary}  ({version})")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗ Install failed: {exc}[/red]")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bws_version(binary: Path) -> str:
    """Get bws version string."""
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _prompt_access_token(prompt: str = "Access token: ") -> str:
    """Prompt for access token with masked input."""
    return masked_secret_prompt(prompt).strip()


def _prompt_project(client, org_id: str) -> str:
    """Prompt user to pick a project."""
    try:
        projects = client.list_projects(org_id)
    except Exception as exc:  # noqa: BLE001
        print(f"  [red]✗ Could not list projects: {exc}[/red]")
        return ""

    if not projects:
        print("  [red]No projects found in organization.[/red]")
        return ""

    if len(projects) == 1:
        print(f"  Using only project: {projects[0]['name']}")
        return projects[0]["id"]

    print("\nAvailable projects:")
    for i, proj in enumerate(projects, 1):
        print(f"  {i}. {proj['name']} ({proj['id']})")

    while True:
        choice = input(f"\nSelect project [1-{len(projects)}]: ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(projects):
                return projects[idx - 1]["id"]
        except ValueError:
            pass
        print(f"  [red]Invalid choice. Enter 1-{len(projects)}.[/red]")