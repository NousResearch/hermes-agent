"""CLI handlers for ``hermes secrets bitwarden ...``.

Subcommands:
    setup    — interactive wizard: install bws, prompt for token + project, test fetch
    status   — show current config + binary version + last fetch outcome
    sync     — run a fetch right now and show what would be applied (dry-run friendly)
    disable  — flip ``secrets.bitwarden.enabled`` to False
    install  — just download the bws binary (no token / project required)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import bitwarden as bw
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

    status = sub.add_parser("status", help="Show config + binary + last fetch")
    status.set_defaults(func=cmd_status)

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
        help=f"Download and verify the pinned bws binary (v{bw._BWS_VERSION})",
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

    # -- non-interactive guard --
    if not sys.stdin.isatty():
        missing = []
        if not (args.access_token and args.access_token.strip()):
            missing.append("--access-token")
        if not (args.server_url and args.server_url.strip()):
            # Also accept BWS_SERVER_URL env var as non-interactive substitute
            if not os.environ.get("BWS_SERVER_URL", "").strip():
                missing.append("--server-url")
        if not (args.project_id and args.project_id.strip()):
            missing.append("--project-id")
        if missing:
            console.print(
                f"  [red]Non-interactive mode (no TTY) requires all setup flags.[/red]\n"
                f"  Missing: {', '.join(missing)}\n\n"
                "  Usage:\n"
                "    hermes secrets bitwarden setup \\\n"
                "      --access-token '0.xxx' \\\n"
                "      --server-url 'https://vault.bitwarden.com' \\\n"
                "      --project-id 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'"
            )
            return 1

    # ------------------------------------------------------------------- token
    console.print()
    console.print("[bold]Step 2[/bold]  Provide your access token")
    cfg = load_config()
    secrets_cfg = (cfg.setdefault("secrets", {})
                     .setdefault("bitwarden", {}))
    token_env = secrets_cfg.get("access_token_env", "BWS_ACCESS_TOKEN")

    token = (args.access_token or "").strip()
    if not token:
        token = masked_secret_prompt(f"  Paste access token ({token_env}): ").strip()
    if not token:
        console.print("  [red]Empty token, aborting.[/red]")
        return 1
    if not token.startswith("0."):
        console.print(
            "  [yellow]Warning: token doesn't start with '0.' — usually that means "
            "you pasted something other than a BSM access token.  Continuing anyway.[/yellow]"
        )

    save_env_value(token_env, token)
    os.environ[token_env] = token  # so the test fetch below sees it
    console.print(f"  [green]✓[/green] stored in {get_env_path()} as {token_env}")

    # ------------------------------------------------------------------ region
    console.print()
    console.print("[bold]Step 3[/bold]  Pick a Bitwarden region")
    server_url = _resolve_server_url(args, secrets_cfg, console)
    if server_url is None:
        return 1
    if server_url:
        console.print(f"  [green]✓[/green] using {server_url}")
    else:
        console.print(
            "  [green]✓[/green] using bws default "
            "(US Cloud, https://vault.bitwarden.com)"
        )

    # ------------------------------------------------------------------- project
    if args.project_id and args.project_id.strip():
        project_id = args.project_id.strip()
    else:
        console.print()
        console.print("[bold]Step 4[/bold]  Pick a project")
        project_id = ""
        projects = _list_projects(binary, token, console, server_url=server_url)
        if projects is None:
            return 1
        if not projects:
            console.print("  [yellow]No projects visible to this machine account.[/yellow]")
            console.print(
                "  In the Bitwarden web app, open the machine account → Projects tab "
                "and grant it access to at least one project."
            )
            return 1

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Name")
        table.add_column("ID", style="dim")
        for i, p in enumerate(projects, 1):
            table.add_row(str(i), p.get("name", "?"), p.get("id", "?"))
        console.print(table)

        while True:
            choice = console.input(f"  Select project [1-{len(projects)}]: ").strip()
            if not choice:
                continue
            try:
                idx = int(choice)
            except ValueError:
                console.print("  [red]Enter a number.[/red]")
                continue
            if 1 <= idx <= len(projects):
                project_id = projects[idx - 1]["id"]
                break
            console.print(f"  [red]Out of range — pick 1-{len(projects)}.[/red]")

    # ------------------------------------------------------------------- test
    console.print()
    step_num = 5 if not (args.project_id and args.project_id.strip()) else 4
    console.print(f"[bold]Step {step_num}[/bold]  Test fetch")
    try:
        secrets, warnings = bw.fetch_bitwarden_secrets(
            access_token=token,
            project_id=project_id,
            binary=binary,
            use_cache=False,
            server_url=server_url,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ Fetch failed: {exc}[/red]")
        return 1

    if not secrets:
        console.print("  [yellow]Fetch succeeded but the project has no secrets.[/yellow]")
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Name", style="cyan")
        table.add_column("Status")
        for key in sorted(secrets):
            if key == token_env:
                status = "[dim]bootstrap token — never overrides itself[/dim]"
            elif os.environ.get(key):
                status = "[yellow]already set in env (will be overwritten)[/yellow]"
            else:
                status = "[green]new[/green]"
            table.add_row(key, status)
        console.print(table)
    for w in warnings:
        console.print(f"  [yellow]warning:[/yellow] {w}")

    # ------------------------------------------------------------------- save
    secrets_cfg["enabled"] = True
    secrets_cfg["project_id"] = project_id
    secrets_cfg["server_url"] = server_url
    secrets_cfg.setdefault("access_token_env", token_env)
    secrets_cfg.setdefault("cache_ttl_seconds", 300)
    secrets_cfg.setdefault("override_existing", True)
    secrets_cfg.setdefault("auto_install", True)
    save_config(cfg)

    console.print()
    console.print(
        "[green]✓ Bitwarden Secrets Manager is enabled.[/green]  "
        "Secrets will be pulled at the start of every IX Agency process."
    )
    console.print(
        "  Status:  [cyan]hermes secrets bitwarden status[/cyan]\n"
        "  Refresh: [cyan]hermes secrets bitwarden sync[/cyan]\n"
        "  Disable: [cyan]hermes secrets bitwarden disable[/cyan]"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    bw_cfg = (cfg.get("secrets") or {}).get("bitwarden") or {}

    enabled = bool(bw_cfg.get("enabled"))
    token_env = bw_cfg.get("access_token_env", "BWS_ACCESS_TOKEN")
    project_id = bw_cfg.get("project_id", "")
    server_url = str(bw_cfg.get("server_url", "") or "").strip()
    token_set = bool(os.environ.get(token_env))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled",         _yn(enabled))
    table.add_row("Token env var",   token_env)
    table.add_row("Token in env",    _yn(token_set))
    table.add_row("Project ID",      project_id or "[dim](unset)[/dim]")
    table.add_row(
        "Server URL",
        server_url or "[dim]default (US Cloud, https://vault.bitwarden.com)[/dim]",
    )
    table.add_row("Override existing", _yn(bool(bw_cfg.get("override_existing", False))))
    table.add_row("Cache TTL (s)",   str(bw_cfg.get("cache_ttl_seconds", 300)))
    table.add_row("Auto-install",    _yn(bool(bw_cfg.get("auto_install", True))))

    binary = bw.find_bws(install_if_missing=False)
    if binary:
        table.add_row("bws binary",  f"{binary} ({_bws_version(binary)})")
    else:
        table.add_row("bws binary",  "[yellow]not installed[/yellow]")

    console.print(Panel(table, title="Bitwarden Secrets Manager", border_style="cyan"))

    if not enabled:
        console.print("\n  Run [cyan]hermes secrets bitwarden setup[/cyan] to enable.")
        return 0
    if not token_set:
        console.print(
            f"\n  [yellow]Enabled but {token_env} is not set — IX Agency will skip BSM "
            "and warn on next startup.[/yellow]"
        )
    if not project_id:
        console.print(
            "\n  [yellow]Enabled but no project_id — nothing to fetch.[/yellow]"
        )
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    bw_cfg = (cfg.get("secrets") or {}).get("bitwarden") or {}
    if not bw_cfg.get("enabled"):
        console.print(
            "[yellow]Bitwarden integration is disabled.  Run "
            "`hermes secrets bitwarden setup` first.[/yellow]"
        )
        return 1

    token_env = bw_cfg.get("access_token_env", "BWS_ACCESS_TOKEN")
    token = os.environ.get(token_env, "").strip()
    if not token:
        console.print(f"[red]{token_env} is not set.[/red]")
        return 1

    project_id = bw_cfg.get("project_id", "")
    if not project_id:
        console.print("[red]No project_id configured.[/red]")
        return 1

    server_url = str(bw_cfg.get("server_url", "") or "").strip()

    try:
        secrets, warnings = bw.fetch_bitwarden_secrets(
            access_token=token,
            project_id=project_id,
            use_cache=False,
            server_url=server_url,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fetch failed: {exc}[/red]")
        return 1

    if not secrets:
        console.print("[yellow]No secrets in project.[/yellow]")
        return 0

    override = bool(bw_cfg.get("override_existing", False)) or args.apply
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Action")
    applied = 0
    for key in sorted(secrets):
        if key == token_env:
            table.add_row(key, "[dim]skip (bootstrap token)[/dim]")
            continue
        already = bool(os.environ.get(key))
        if already and not override:
            table.add_row(key, "[dim]skip (already set)[/dim]")
            continue
        if args.apply:
            os.environ[key] = secrets[key]
            applied += 1
            table.add_row(key, "[green]exported[/green]" + (" (overrode)" if already else ""))
        else:
            table.add_row(key, "[green]would export[/green]" + (" (overrides)" if already else ""))

    console.print(table)
    for w in warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")

    if not args.apply:
        console.print(
            "\n  This was a dry-run — secrets are picked up automatically on the "
            "next [cyan]hermes[/cyan] invocation.  Re-run with [cyan]--apply[/cyan] "
            "to export into the current shell instead."
        )
    else:
        console.print(f"\n  [green]Exported {applied} secret(s) into current process.[/green]")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    bw_cfg = (cfg.setdefault("secrets", {})
                .setdefault("bitwarden", {}))
    bw_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]Disabled.[/green]  Bitwarden secrets will NOT be pulled on the next "
        "IX Agency invocation.\n"
        "  Your access token is left in .env — remove it manually if you also want "
        "to revoke the credential."
    )
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    console = Console()
    try:
        path = bw.install_bws(force=bool(args.force))
        console.print(f"[green]✓[/green] {path}  ({_bws_version(path)})")
        return 0
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Install failed: {exc}[/red]")
        return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yn(b: bool) -> str:
    return "[green]yes[/green]" if b else "[dim]no[/dim]"


def _bws_version(binary: Path) -> str:
    try:
        res = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return (res.stdout or res.stderr).strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "version unknown"


def _list_projects(
    binary: Path, token: str, console: Console, *, server_url: str = ""
) -> Optional[List[dict]]:
    """Call ``bws project list`` and return the parsed list, or None on failure."""
    env = os.environ.copy()
    env["BWS_ACCESS_TOKEN"] = token
    env.setdefault("NO_COLOR", "1")
    if server_url:
        env["BWS_SERVER_URL"] = server_url
    try:
        res = subprocess.run(
            [str(binary), "project", "list", "--output", "json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        console.print(f"  [red]Couldn't list projects: {exc}[/red]")
        return None

    if res.returncode != 0:
        err = (res.stderr or res.stdout).strip()[:300]
        console.print(f"  [red]bws project list failed: {err}[/red]")
        lowered = err.lower()
        if "invalid_client" in lowered or "400 bad request" in lowered:
            console.print(
                "  [yellow]'invalid_client' from the US identity endpoint usually "
                "means the token is for a different Bitwarden region.  Re-run "
                "[cyan]hermes secrets bitwarden setup[/cyan] and pick EU or "
                "self-hosted at the region prompt, or set [cyan]secrets.bitwarden."
                "server_url[/cyan] in config.yaml.[/yellow]"
            )
        elif "authorization" in lowered or "invalid" in lowered:
            console.print(
                "  [yellow]This usually means the access token is wrong or revoked. "
                "Double-check it in the Bitwarden web app.[/yellow]"
            )
        return None

    try:
        data = json.loads(res.stdout or "[]")
    except json.JSONDecodeError as exc:
        console.print(f"  [red]bws returned non-JSON: {exc}[/red]")
        return None
    if not isinstance(data, list):
        return []
    return [p for p in data if isinstance(p, dict) and p.get("id")]


# Canonical Bitwarden region endpoints.  Keep in sync with what Bitwarden
# publishes — these are stable but if a third region appears, add it here
# and to the prompt below.
_REGION_PRESETS = [
    ("US Cloud  (https://vault.bitwarden.com — bws default)", ""),
    ("EU Cloud  (https://vault.bitwarden.eu)", "https://vault.bitwarden.eu"),
]


def _resolve_server_url(
    args: argparse.Namespace,
    secrets_cfg: dict,
    console: Console,
) -> Optional[str]:
    """Pick a Bitwarden server URL for setup.

    Resolution order:
      1. ``--server-url`` CLI flag (non-interactive)
      2. ``BWS_SERVER_URL`` env var (so users running with that already set
         in their shell don't have to re-enter it)
      3. Existing ``secrets.bitwarden.server_url`` value (for re-runs)
      4. Interactive menu: US / EU / self-hosted

    Returns the chosen URL as a string (empty string = bws default,
    i.e. US Cloud).  Returns None if the user aborted with an empty
    custom URL.
    """
    if args.server_url and args.server_url.strip():
        return args.server_url.strip()

    env_url = os.environ.get("BWS_SERVER_URL", "").strip()
    if env_url:
        console.print(
            f"  Detected [cyan]BWS_SERVER_URL[/cyan]={env_url} in your shell — using it."
        )
        return env_url

    existing = str(secrets_cfg.get("server_url", "") or "").strip()
    if existing:
        console.print(
            f"  Existing config: [cyan]{existing}[/cyan]. "
            "Press Enter to keep, or pick a different option below."
        )

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="cyan", width=4)
    table.add_column("Region / endpoint")
    for i, (label, _url) in enumerate(_REGION_PRESETS, 1):
        table.add_row(str(i), label)
    table.add_row(str(len(_REGION_PRESETS) + 1), "Self-hosted / custom URL")
    console.print(table)

    custom_idx = len(_REGION_PRESETS) + 1
    while True:
        prompt = f"  Select region [1-{custom_idx}]"
        if existing:
            prompt += " (Enter to keep current)"
        prompt += ": "
        choice = console.input(prompt).strip()
        if not choice:
            if existing:
                return existing
            console.print("  [red]Enter a number.[/red]")
            continue
        try:
            idx = int(choice)
        except ValueError:
            console.print("  [red]Enter a number.[/red]")
            continue
        if 1 <= idx <= len(_REGION_PRESETS):
            return _REGION_PRESETS[idx - 1][1]
        if idx == custom_idx:
            custom = console.input(
                "  Enter your Bitwarden server URL "
                "(e.g. https://vault.example.com): "
            ).strip()
            if not custom:
                console.print("  [red]Empty URL, aborting.[/red]")
                return None
            if not custom.startswith(("http://", "https://")):
                console.print(
                    "  [yellow]Warning: URL doesn't start with http:// or "
                    "https:// — bws may reject it.[/yellow]"
                )
            return custom
        console.print(f"  [red]Out of range — pick 1-{custom_idx}.[/red]")
