"""CLI handlers for ``hermes secrets infisical ...``."""

from __future__ import annotations

import argparse
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import infisical
from hermes_cli.config import (
    get_env_path,
    load_config,
    save_config,
    save_env_value,
)
from hermes_cli.secret_prompt import masked_secret_prompt


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``infisical`` subcommand tree to a parent parser."""
    sub = parent_parser.add_subparsers(dest="secrets_infisical_command")

    setup = sub.add_parser(
        "setup",
        help="Interactive wizard: store Universal Auth credentials and test fetch",
    )
    setup.add_argument(
        "--client-id",
        help="Universal Auth client ID (will be stored in .env)",
    )
    setup.add_argument(
        "--client-secret",
        help="Universal Auth client secret (will be stored in .env)",
    )
    setup.add_argument("--project-id", help="Infisical project UUID")
    setup.add_argument(
        "--api-url",
        default="",
        help=f"Infisical API URL (default: {infisical.DEFAULT_API_URL})",
    )
    setup.add_argument("--env", default="", help="Infisical environment slug")
    setup.add_argument("--path", default="", help="Secret path to sync")
    setup.add_argument(
        "--organization-slug",
        default="",
        help="Optional organization slug for Universal Auth",
    )
    setup.set_defaults(func=cmd_setup)

    status = sub.add_parser("status", help="Show config + credential presence")
    status.set_defaults(func=cmd_status)

    sync = sub.add_parser("sync", help="Fetch secrets now and report what changed")
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Actually export the secrets into the current process env",
    )
    sync.set_defaults(func=cmd_sync)

    disable = sub.add_parser("disable", help="Turn off the Infisical integration")
    disable.set_defaults(func=cmd_disable)


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]Infisical setup[/bold]\n\n"
            "Create a Machine Identity with Universal Auth in Infisical, "
            "grant it read access to the project/environment/path Hermes "
            "should sync, then paste the client ID and client secret here.",
            border_style="cyan",
        )
    )

    cfg = load_config()
    secrets_cfg = cfg.setdefault("secrets", {}).setdefault("infisical", {})
    client_id_env = secrets_cfg.get("client_id_env", "INFISICAL_CLIENT_ID")
    client_secret_env = secrets_cfg.get(
        "client_secret_env", "INFISICAL_CLIENT_SECRET"
    )

    console.print()
    console.print("[bold]Step 1[/bold]  Provide Universal Auth credentials")
    client_id = (args.client_id or "").strip()
    if not client_id:
        client_id = console.input(f"  Client ID ({client_id_env}): ").strip()
    if not client_id:
        console.print("  [red]Empty client ID, aborting.[/red]")
        return 1

    client_secret = (args.client_secret or "").strip()
    if not client_secret:
        client_secret = masked_secret_prompt(
            f"  Client secret ({client_secret_env}): "
        ).strip()
    if not client_secret:
        console.print("  [red]Empty client secret, aborting.[/red]")
        return 1

    save_env_value(client_id_env, client_id)
    save_env_value(client_secret_env, client_secret)
    os.environ[client_id_env] = client_id
    os.environ[client_secret_env] = client_secret
    console.print(
        f"  [green]✓[/green] stored bootstrap credentials in {get_env_path()}"
    )

    console.print()
    console.print("[bold]Step 2[/bold]  Select Infisical project and path")
    project_id = (args.project_id or "").strip()
    if not project_id:
        project_id = str(secrets_cfg.get("project_id", "") or "").strip()
    if not project_id:
        project_id = console.input("  Project ID: ").strip()
    if not project_id:
        console.print("  [red]Empty project ID, aborting.[/red]")
        return 1

    api_url = (
        args.api_url
        or secrets_cfg.get("api_url")
        or os.environ.get("INFISICAL_API_URL")
        or infisical.DEFAULT_API_URL
    )
    api_url = str(api_url).strip()
    environment = (
        args.env
        or secrets_cfg.get("env")
        or "prod"
    )
    environment = str(environment).strip()
    secret_path = (
        args.path
        or secrets_cfg.get("path")
        or "/"
    )
    secret_path = str(secret_path).strip()
    organization_slug = (
        args.organization_slug
        or secrets_cfg.get("organization_slug")
        or ""
    )
    organization_slug = str(organization_slug).strip()

    console.print(f"  API URL:     [cyan]{api_url}[/cyan]")
    console.print(f"  Project ID:  [cyan]{project_id}[/cyan]")
    console.print(f"  Environment: [cyan]{environment}[/cyan]")
    console.print(f"  Path:        [cyan]{secret_path}[/cyan]")
    if organization_slug:
        console.print(f"  Org slug:    [cyan]{organization_slug}[/cyan]")

    console.print()
    console.print("[bold]Step 3[/bold]  Test fetch")
    try:
        secrets, warnings = infisical.fetch_infisical_secrets(
            client_id=client_id,
            client_secret=client_secret,
            project_id=project_id,
            environment=environment,
            secret_path=secret_path,
            api_url=api_url,
            organization_slug=organization_slug,
            use_cache=False,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ Fetch failed: {exc}[/red]")
        return 1

    _print_secret_preview(console, secrets, warnings, bootstrap_names={
        client_id_env,
        client_secret_env,
        secrets_cfg.get("project_id_env", "INFISICAL_PROJECT_ID"),
    })

    secrets_cfg["enabled"] = True
    secrets_cfg["api_url"] = api_url
    secrets_cfg["client_id_env"] = client_id_env
    secrets_cfg["client_secret_env"] = client_secret_env
    secrets_cfg["project_id"] = project_id
    secrets_cfg.setdefault("project_id_env", "INFISICAL_PROJECT_ID")
    secrets_cfg["env"] = environment
    secrets_cfg["path"] = secret_path
    secrets_cfg["organization_slug"] = organization_slug
    secrets_cfg.setdefault("cache_ttl_seconds", 300)
    secrets_cfg.setdefault("override_existing", True)
    secrets_cfg.setdefault("recursive", False)
    secrets_cfg.setdefault("include_imports", True)
    secrets_cfg.setdefault("expand_secret_references", True)
    save_config(cfg)

    console.print()
    console.print(
        "[green]✓ Infisical is enabled.[/green]  Secrets will be pulled at "
        "the start of every Hermes process."
    )
    console.print(
        "  Status:  [cyan]hermes secrets infisical status[/cyan]\n"
        "  Refresh: [cyan]hermes secrets infisical sync[/cyan]\n"
        "  Disable: [cyan]hermes secrets infisical disable[/cyan]"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    inf_cfg = (cfg.get("secrets") or {}).get("infisical") or {}

    enabled = bool(inf_cfg.get("enabled"))
    client_id_env = inf_cfg.get("client_id_env", "INFISICAL_CLIENT_ID")
    client_secret_env = inf_cfg.get(
        "client_secret_env", "INFISICAL_CLIENT_SECRET"
    )
    project_id = str(inf_cfg.get("project_id", "") or "").strip()
    project_id_env = inf_cfg.get("project_id_env", "INFISICAL_PROJECT_ID")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled", _yn(enabled))
    table.add_row("API URL", inf_cfg.get("api_url", infisical.DEFAULT_API_URL))
    table.add_row("Client ID env", client_id_env)
    table.add_row("Client ID in env", _yn(bool(os.environ.get(client_id_env))))
    table.add_row("Client secret env", client_secret_env)
    table.add_row(
        "Client secret in env",
        _yn(bool(os.environ.get(client_secret_env))),
    )
    table.add_row("Project ID", project_id or f"[dim]({project_id_env})[/dim]")
    table.add_row("Project ID env", project_id_env)
    table.add_row("Project ID in env", _yn(bool(os.environ.get(project_id_env))))
    table.add_row("Environment", str(inf_cfg.get("env", "prod") or "prod"))
    table.add_row("Path", str(inf_cfg.get("path", "/") or "/"))
    table.add_row(
        "Organization slug",
        str(inf_cfg.get("organization_slug", "") or "") or "[dim](unset)[/dim]",
    )
    table.add_row(
        "Override existing",
        _yn(bool(inf_cfg.get("override_existing", True))),
    )
    table.add_row("Cache TTL (s)", str(inf_cfg.get("cache_ttl_seconds", 300)))

    console.print(Panel(table, title="Infisical", border_style="cyan"))

    if not enabled:
        console.print("\n  Run [cyan]hermes secrets infisical setup[/cyan] to enable.")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    inf_cfg = (cfg.get("secrets") or {}).get("infisical") or {}
    if not inf_cfg.get("enabled"):
        console.print(
            "[yellow]Infisical integration is disabled. Run "
            "`hermes secrets infisical setup` first.[/yellow]"
        )
        return 1

    client_id_env = inf_cfg.get("client_id_env", "INFISICAL_CLIENT_ID")
    client_secret_env = inf_cfg.get(
        "client_secret_env", "INFISICAL_CLIENT_SECRET"
    )
    client_id = os.environ.get(client_id_env, "").strip()
    client_secret = os.environ.get(client_secret_env, "").strip()
    if not client_id:
        console.print(f"[red]{client_id_env} is not set.[/red]")
        return 1
    if not client_secret:
        console.print(f"[red]{client_secret_env} is not set.[/red]")
        return 1

    project_id_env = inf_cfg.get("project_id_env", "INFISICAL_PROJECT_ID")
    project_id = str(inf_cfg.get("project_id", "") or "").strip()
    if not project_id:
        project_id = os.environ.get(project_id_env, "").strip()
    if not project_id:
        console.print(f"[red]No project_id configured and {project_id_env} is not set.[/red]")
        return 1

    try:
        secrets, warnings = infisical.fetch_infisical_secrets(
            client_id=client_id,
            client_secret=client_secret,
            project_id=project_id,
            environment=str(inf_cfg.get("env", "prod") or "prod"),
            secret_path=str(inf_cfg.get("path", "/") or "/"),
            api_url=str(
                inf_cfg.get("api_url")
                or os.environ.get("INFISICAL_API_URL")
                or infisical.DEFAULT_API_URL
            ),
            organization_slug=str(inf_cfg.get("organization_slug", "") or ""),
            use_cache=False,
            recursive=bool(inf_cfg.get("recursive", False)),
            include_imports=bool(inf_cfg.get("include_imports", True)),
            expand_secret_references=bool(
                inf_cfg.get("expand_secret_references", True)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fetch failed: {exc}[/red]")
        return 1

    _print_sync_actions(console, args, inf_cfg, secrets, warnings, {
        client_id_env,
        client_secret_env,
        project_id_env,
    })
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    inf_cfg = cfg.setdefault("secrets", {}).setdefault("infisical", {})
    inf_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]Disabled.[/green] Infisical secrets will NOT be pulled on "
        "the next Hermes invocation.\n"
        "  Bootstrap credentials are left in .env; remove or revoke them "
        "manually if needed."
    )
    return 0


def _print_secret_preview(
    console: Console,
    secrets: dict[str, str],
    warnings: list[str],
    *,
    bootstrap_names: set[str],
) -> None:
    if not secrets:
        console.print("  [yellow]Fetch succeeded but this path has no secrets.[/yellow]")
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Name", style="cyan")
        table.add_column("Status")
        for key in sorted(secrets):
            if key in bootstrap_names:
                status = "[dim]bootstrap credential — never overrides itself[/dim]"
            elif os.environ.get(key):
                status = "[yellow]already set in env (will be overwritten)[/yellow]"
            else:
                status = "[green]new[/green]"
            table.add_row(key, status)
        console.print(table)
    for warning in warnings:
        console.print(f"  [yellow]warning:[/yellow] {warning}")


def _print_sync_actions(
    console: Console,
    args: argparse.Namespace,
    inf_cfg: dict,
    secrets: dict[str, str],
    warnings: list[str],
    bootstrap_names: set[str],
) -> None:
    if not secrets:
        console.print("[yellow]No secrets found.[/yellow]")
        return

    override = bool(inf_cfg.get("override_existing", True))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Action")
    applied = 0
    for key in sorted(secrets):
        if key in bootstrap_names:
            table.add_row(key, "[dim]skip (bootstrap credential)[/dim]")
            continue
        already = bool(os.environ.get(key))
        if already and not override:
            table.add_row(key, "[dim]skip (already set)[/dim]")
            continue
        if args.apply:
            os.environ[key] = secrets[key]
            applied += 1
            table.add_row(
                key,
                "[green]exported[/green]" + (" (overrode)" if already else ""),
            )
        else:
            table.add_row(
                key,
                "[green]would export[/green]" + (" (overrides)" if already else ""),
            )

    console.print(table)
    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    if not args.apply:
        console.print(
            "\n  This was a dry-run — secrets are picked up automatically on the "
            "next [cyan]hermes[/cyan] invocation. Re-run with [cyan]--apply[/cyan] "
            "to export into the current process instead."
        )
    else:
        console.print(f"\n  [green]Exported {applied} secret(s) into current process.[/green]")


def _yn(value: bool) -> str:
    return "[green]yes[/green]" if value else "[dim]no[/dim]"
