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
        "Secrets will be pulled at the start of every Hermes process."
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
            f"\n  [yellow]Enabled but {token_env} is not set — Hermes will skip BSM "
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
        "Hermes invocation.\n"
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


# ===========================================================================
# 1Password Service Account — CLI handlers
# ===========================================================================


def register_onepassword_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``onepassword`` subcommand tree to ``hermes secrets``."""
    sub = parent_parser.add_subparsers(dest="secrets_op_command")

    setup = sub.add_parser(
        "setup",
        help="Interactive setup: verify SDK, test auth, enable",
    )
    setup.set_defaults(func=cmd_onepassword_setup)

    status = sub.add_parser(
        "status",
        help="Show config + SDK availability + token presence",
    )
    status.set_defaults(func=cmd_onepassword_status)

    sync = sub.add_parser(
        "sync",
        help="Fetch secrets from 1Password and show what would be applied",
    )
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Actually export secrets into os.environ",
    )
    sync.set_defaults(func=cmd_onepassword_sync)

    disable = sub.add_parser(
        "disable",
        help="Turn off the 1Password integration",
    )
    disable.set_defaults(func=cmd_onepassword_disable)

    list_vaults = sub.add_parser(
        "list-vaults",
        help="List vaults the service account can access",
    )
    list_vaults.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
    )
    list_vaults.set_defaults(func=cmd_onepassword_list_vaults)


def cmd_onepassword_setup(args: argparse.Namespace) -> int:
    """Interactive wizard for 1Password integration setup."""
    import getpass
    import sys as _sys

    console = Console()
    cfg = load_config()
    op_cfg = (cfg.setdefault("secrets", {})
                 .setdefault("onepassword", {}))
    token_env = op_cfg.get("token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    vault = op_cfg.get("vault", "")

    console.print(
        Panel.fit(
            "[bold]1Password Service Account setup[/bold]\n\n"
            "This backend uses the [cyan]onepassword-sdk[/cyan] Python package\n"
            "instead of the ``op`` CLI daemon.  The SDK authenticates directly\n"
            "via the 1Password REST API — no background daemon needed.\n\n"
            "To get a service account token:\n"
            "  1Password web → Developers → Service Accounts →\n"
            "  Create service account → grant vault access → copy token\n\n"
            "The token (starts with [cyan]ops_[/cyan]) is stored in .env\n"
            "and used by the Python SDK to authenticate.",
            border_style="green",
        )
    )

    # Step 1: Check SDK
    console.print()
    console.print("[bold]Step 1[/bold]  Check onepassword-sdk")
    try:
        from agent.secret_sources.onepassword import apply_onepassword_secrets  # noqa: F401
        console.print("  [green]✓[/green] onepassword-sdk is installed")
    except ImportError:
        console.print(
            "  [red]✗[/red] onepassword-sdk not found. Install:\n"
            f"    uv pip install --python {_sys.executable} onepassword-sdk"
        )
        return 1

    # Step 2: Token
    console.print()
    console.print(
        f"[bold]Step 2[/bold]  Provide your service account token "
        f"({token_env})"
    )
    existing_token = os.environ.get(token_env, "").strip()
    if existing_token:
        console.print(
            f"  [green]✓[/green] {token_env} is already set "
            f"({len(existing_token)} chars)"
        )
        token = existing_token
    else:
        token = getpass.getpass(f"  Paste token ({token_env}): ").strip()
        if not token:
            console.print("  [red]Empty token, aborting.[/red]")
            return 1
        save_env_value(token_env, token)
        os.environ[token_env] = token
        console.print(
            f"  [green]✓[/green] stored in {get_env_path()} as {token_env}"
        )

    # Step 3: Choose mapping mode
    console.print()
    console.print("[bold]Step 3[/bold]  Choose mapping mode")
    console.print(
        "  [1] Auto-discovery — scan a vault and map credential fields "
        "→ env vars automatically"
    )
    console.print(
        "  [2] Explicit mapping — list each env var → op:// reference "
        "in config.yaml"
    )
    console.print("  [3] Both — explicit + auto-discovery (explicit wins on conflicts)")

    choice = console.input("  Pick 1-3 [1]: ").strip() or "1"
    if choice not in ("1", "2", "3"):
        console.print("  [red]Invalid choice.[/red]")
        return 1

    auto_discover = choice in ("1", "3")

    # Step 4: Vault selection (for auto-discover mode)
    selected_vault = ""
    if auto_discover:
        console.print()
        console.print("[bold]Step 4[/bold]  Pick a vault")
        selected_vault = _pick_onepassword_vault(token, token_env, vault, console)
        if selected_vault is None:
            return 1

    # Step 5: Test fetch
    console.print()
    console.print(
        f"[bold]Step {'5' if auto_discover else '4'}[/bold]  Test fetch"
    )
    try:
        from agent.secret_sources.onepassword import apply_onepassword_secrets
    except ImportError:
        console.print("  [red]SDK import failed unexpectedly.[/red]")
        return 1

    env_refs: dict = {}
    result = apply_onepassword_secrets(
        enabled=True,
        token_env=token_env,
        vault=selected_vault,
        override_existing=True,
        cache_ttl_seconds=0,  # Bypass cache for setup test
        auto_discover=auto_discover,
        env_refs=env_refs,
    )

    if result.error:
        console.print(f"  [red]✗ Fetch failed: {result.error}[/red]")
        return 1

    if not result.secrets:
        console.print(
            "  [yellow]No secrets found.[/yellow]  "
            "Check that the vault contains items with credential fields "
            "(Concealed, Password, or API Credential types)."
        )
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Env Var", style="cyan")
        table.add_column("Status")
        for key in sorted(result.secrets):
            if os.environ.get(key):
                status = (
                    "[yellow]already set (will be overwritten)[/yellow]"
                )
            else:
                status = "[green]new[/green]"
            table.add_row(key, status)
        console.print(table)
    for w in result.warnings:
        console.print(f"  [yellow]warning:[/yellow] {w}")

    # Save config
    op_cfg["enabled"] = True
    op_cfg["token_env"] = token_env
    if selected_vault:
        op_cfg["vault"] = selected_vault
    op_cfg["auto_discover"] = auto_discover
    op_cfg.setdefault("override_existing", True)
    op_cfg.setdefault("cache_ttl_seconds", 300)
    save_config(cfg)

    console.print()
    console.print(
        "[green]✓ 1Password integration is enabled.[/green]  "
        "Secrets will be pulled at the start of every Hermes process."
    )
    console.print(
        "  Status:  [cyan]hermes secrets onepassword status[/cyan]\n"
        "  Sync:    [cyan]hermes secrets onepassword sync[/cyan]\n"
        "  Disable: [cyan]hermes secrets onepassword disable[/cyan]"
    )
    return 0


def cmd_onepassword_status(args: argparse.Namespace) -> int:
    """Show 1Password integration status."""
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}

    enabled = bool(op_cfg.get("enabled"))
    token_env = op_cfg.get("token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    vault = op_cfg.get("vault", "")
    token_set = bool(os.environ.get(token_env))

    def _yn(val: bool) -> str:
        return "[green]yes[/green]" if val else "[red]no[/red]"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled",        _yn(enabled))
    table.add_row("Token env var",  token_env)
    table.add_row("Token in env",   _yn(token_set))
    table.add_row("Vault",          vault or "(not set)")
    table.add_row("Auto-discover",  _yn(bool(op_cfg.get("auto_discover", False))))
    table.add_row("Explicit env",   str(len(op_cfg.get("env") or {})))
    table.add_row("Override",       _yn(bool(op_cfg.get("override_existing", False))))
    table.add_row("Cache TTL (s)",  str(op_cfg.get("cache_ttl_seconds", 300)))

    sdk_ok = False
    try:
        from agent.secret_sources.onepassword import apply_onepassword_secrets  # noqa: F401
        sdk_ok = True
    except ImportError:
        pass
    table.add_row("SDK installed",  _yn(sdk_ok))

    console.print(
        Panel(table, title="1Password Service Account", border_style="green")
    )

    if not enabled:
        console.print(
            "\n  Run [cyan]hermes secrets onepassword setup[/cyan] to enable."
        )
        return 0
    if not token_set:
        console.print(
            f"\n  [yellow]Enabled but {token_env} is not set — "
            "Hermes will skip 1Password on next startup.[/yellow]"
        )
    if not vault and not op_cfg.get("env"):
        console.print(
            "\n  [yellow]Enabled but no vault or env mapping configured "
            "— nothing to fetch.[/yellow]"
        )
    return 0


def cmd_onepassword_sync(args: argparse.Namespace) -> int:
    """Fetch 1Password secrets now."""
    import sys as _sys

    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}

    if not op_cfg.get("enabled"):
        console.print(
            "[yellow]1Password integration is disabled. Run "
            "`hermes secrets onepassword setup` first.[/yellow]"
        )
        return 1

    token_env = op_cfg.get("token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    token = os.environ.get(token_env, "").strip()
    if not token:
        console.print(f"[red]{token_env} is not set.[/red]")
        return 1

    try:
        from agent.secret_sources.onepassword import apply_onepassword_secrets
    except ImportError:
        console.print(
            "[red]onepassword-sdk not installed.[/red] Install with:\n"
            f"  uv pip install --python {_sys.executable} onepassword-sdk"
        )
        return 1

    result = apply_onepassword_secrets(
        enabled=True,
        token_env=token_env,
        vault=op_cfg.get("vault", ""),
        override_existing=True,
        cache_ttl_seconds=0,  # Bypass cache for explicit sync
        auto_discover=bool(op_cfg.get("auto_discover", False)),
        env_refs=op_cfg.get("env") or {},
    )

    if result.error:
        console.print(f"[red]{result.error}[/red]")
        return 1

    table = Table(show_header=True, header_style="bold")
    table.add_column("Env Var", style="cyan")
    table.add_column("Action")
    for name in sorted(result.secrets):
        if name in result.applied:
            table.add_row(name, "[green]exported[/green]")
        elif name in result.skipped:
            table.add_row(name, "[dim]skipped[/dim]")
    console.print(table)
    for w in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")

    if not args.apply:
        console.print(
            "\n  This was a dry-run. Secrets are picked up automatically "
            "on the next [cyan]hermes[/cyan] invocation. "
            "Re-run with [cyan]--apply[/cyan] to export now."
        )
    else:
        console.print(
            f"\n  [green]Exported {len(result.applied)} secret(s).[/green]"
        )
    return 0


def cmd_onepassword_disable(args: argparse.Namespace) -> int:
    """Disable the 1Password integration."""
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.setdefault("secrets", {}).setdefault("onepassword", {}))
    op_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]✓ 1Password integration disabled.[/green]  "
        "Re-enable with [cyan]hermes secrets onepassword setup[/cyan]."
    )
    return 0


def cmd_onepassword_list_vaults(args: argparse.Namespace) -> int:
    """List vaults accessible to the service account."""
    import asyncio
    import sys as _sys

    console = Console()
    token = os.environ.get("OP_SERVICE_ACCOUNT_TOKEN", "").strip()
    if not token:
        console.print(
            "[red]OP_SERVICE_ACCOUNT_TOKEN is not set.[/red]"
        )
        return 1

    try:
        from onepassword import Client
        from onepassword.types import VaultGetParams
    except ImportError:
        console.print(
            "[red]onepassword-sdk not installed.[/red]"
        )
        return 1

    async def _list() -> list:
        client = await Client.authenticate(
            auth=token,
            integration_name="Hermes Agent (CLI)",
            integration_version="1.0.0",
        )
        vaults = await client.vaults.list()
        result = []
        for v in vaults:
            full = await client.vaults.get(
                vault_id=v.id, vault_params=VaultGetParams()
            )
            result.append({
                "id": full.id,
                "name": full.title,
                "description": full.description,
                "item_count": full.active_item_count,
            })
        return result

    try:
        vaults = asyncio.run(_list())
    except Exception as exc:
        console.print(f"[red]Error listing vaults: {exc}[/red]")
        return 1

    if args.format == "json":
        console.print(json.dumps(vaults, indent=2))
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Name", style="cyan")
        table.add_column("Items", justify="right")
        table.add_column("ID", style="dim")
        for v in vaults:
            table.add_row(v["name"], str(v["item_count"]), v["id"])
        console.print(table)
        console.print(f"\n  {len(vaults)} vault(s) accessible.")
    return 0


def _pick_onepassword_vault(
    token: str,
    token_env: str,
    default: str,
    console: Console,
) -> Optional[str]:
    """Let the user pick a vault interactively."""
    import asyncio

    try:
        from onepassword import Client
        from onepassword.types import VaultGetParams
    except ImportError:
        console.print("  [red]onepassword-sdk not installed.[/red]")
        return None

    async def _list() -> list:
        client = await Client.authenticate(
            auth=token,
            integration_name="Hermes Agent Setup",
            integration_version="1.0.0",
        )
        vaults = await client.vaults.list()
        result = []
        for v in vaults:
            full = await client.vaults.get(
                vault_id=v.id, vault_params=VaultGetParams()
            )
            result.append({
                "id": full.id,
                "name": full.title,
                "item_count": full.active_item_count,
            })
        return result

    try:
        vaults = asyncio.run(_list())
    except Exception as exc:
        console.print(f"  [red]Failed to list vaults: {exc}[/red]")
        return None

    if not vaults:
        console.print("  [yellow]No vaults accessible with this token.[/yellow]")
        return None

    console.print(f"  Found {len(vaults)} vault(s):")
    for i, v in enumerate(vaults):
        marker = " ← default" if v["name"] == default else ""
        console.print(
            f"    [{i + 1}] {v['name']} ({v['item_count']} items){marker}"
        )

    choice_s = console.input(
        f"  Pick 1-{len(vaults)}"
        + (f" [{1}]: " if vaults else ": ")
    ).strip()

    idx = 1
    if choice_s:
        try:
            idx = int(choice_s)
        except ValueError:
            pass

    if 1 <= idx <= len(vaults):
        return vaults[idx - 1]["name"]

    return vaults[0]["name"] if vaults else None
