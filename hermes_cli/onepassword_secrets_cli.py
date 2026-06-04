"""CLI handlers for ``hermes secrets onepassword ...``."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import onepassword as opsec
from hermes_cli.config import get_env_path, load_config, save_config, save_env_value
from hermes_cli.secret_prompt import masked_secret_prompt


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    sub = parent_parser.add_subparsers(dest="secrets_op_command")

    setup = sub.add_parser(
        "setup",
        help="Configure 1Password CLI secret references",
    )
    setup.add_argument(
        "--service-account-token",
        help="Provide the 1Password service account token non-interactively",
    )
    setup.add_argument(
        "--token-env",
        default=None,
        help="Env var holding the service account token (default: OP_SERVICE_ACCOUNT_TOKEN)",
    )
    setup.add_argument(
        "--map",
        action="append",
        default=[],
        type=_parse_mapping,
        metavar="ENV_VAR=op://vault/item/field",
        help="Add/update a secret reference mapping. Repeatable.",
    )
    setup.add_argument(
        "--skip-test",
        action="store_true",
        help="Save config without validating refs with `op read` first",
    )
    setup.add_argument(
        "--no-enable",
        action="store_true",
        help="Save config but leave secrets.onepassword.enabled false",
    )
    setup.set_defaults(func=cmd_setup)

    status = sub.add_parser("status", help="Show config + op binary status")
    status.set_defaults(func=cmd_status)

    sync = sub.add_parser("sync", help="Resolve references now and report what changed")
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Actually export the secrets into the current process env (default: dry-run)",
    )
    sync.set_defaults(func=cmd_sync)

    disable = sub.add_parser("disable", help="Turn off the 1Password integration")
    disable.set_defaults(func=cmd_disable)


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]1Password secrets setup[/bold]\n\n"
            "Uses 1Password CLI secret references, e.g.\n"
            "  [cyan]op://Hermes/TELEGRAM_BOT_TOKEN/password[/cyan]\n\n"
            "For unattended gateway use, use a 1Password Service Account token.",
            border_style="cyan",
        )
    )

    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    token_env = args.token_env or op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")

    token = (args.service_account_token or "").strip()
    if not token and not os.environ.get(token_env):
        token = masked_secret_prompt(f"  Paste service account token ({token_env}): ").strip()
    if token:
        save_env_value(token_env, token)
        os.environ[token_env] = token
        console.print(f"  [green]✓[/green] stored token in {get_env_path()} as {token_env}")
    elif os.environ.get(token_env):
        console.print(f"  [green]✓[/green] using existing {token_env} from environment/.env")
    else:
        console.print(f"  [red]{token_env} is not set.[/red]")
        return 1

    mapping = dict(op_cfg.get("mapping") or {})
    for key, ref in args.map:
        mapping[key] = ref

    if not mapping:
        console.print(
            "  [yellow]No mappings configured yet. Add them with "
            "--map ENV_VAR=op://Vault/Item/password.[/yellow]"
        )
        console.print("  [yellow]Saved disabled config so startup stays quiet.[/yellow]")

    op_cfg["service_account_token_env"] = token_env
    op_cfg["mapping"] = mapping
    op_cfg.setdefault("override_existing", True)
    op_cfg.setdefault("cache_ttl_seconds", 300)
    op_cfg["enabled"] = bool(mapping) and not bool(args.no_enable)

    if op_cfg["enabled"] and not bool(args.skip_test):
        console.print("\n[bold]Test fetch[/bold]")
        try:
            secrets, warnings = opsec.fetch_onepassword_secrets(
                service_account_token=os.environ[token_env],
                mapping=mapping,
                cache_ttl_seconds=float(op_cfg.get("cache_ttl_seconds", 300)),
                use_cache=False,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]✗ Fetch failed: {exc}[/red]")
            console.print("  Config was not enabled. Fix the refs or re-run with --skip-test.")
            return 1

        for warning in warnings:
            console.print(f"  [yellow]warning:[/yellow] {warning}")
        if not secrets:
            console.print("  [red]✗ No secrets resolved.[/red]")
            console.print("  Config was not enabled. Add at least one valid --map entry.")
            return 1
        console.print(f"  [green]✓[/green] resolved {len(secrets)} secret(s)")

    save_config(cfg)

    console.print("\n[green]✓ 1Password config saved.[/green]")
    _print_mapping(console, mapping)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}
    enabled = bool(op_cfg.get("enabled"))
    token_env = op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    token_set = bool(os.environ.get(token_env))
    mapping = op_cfg.get("mapping") or {}

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled", _yn(enabled))
    table.add_row("Token env var", token_env)
    table.add_row("Token in env", _yn(token_set))
    table.add_row("Mappings", str(len(mapping)))
    table.add_row("Override existing", _yn(bool(op_cfg.get("override_existing", False))))
    table.add_row("Cache TTL (s)", str(op_cfg.get("cache_ttl_seconds", 300)))

    binary = opsec.find_op()
    if binary:
        table.add_row("op binary", f"{binary} ({_op_version(binary)})")
    else:
        table.add_row("op binary", "[yellow]not installed[/yellow]")

    console.print(Panel(table, title="1Password", border_style="cyan"))
    if mapping:
        _print_mapping(console, mapping)
    else:
        console.print("\n  Add mappings with [cyan]hermes secrets onepassword setup --map ENV=op://...[/cyan]")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}
    if not op_cfg.get("enabled"):
        console.print("[yellow]1Password integration is disabled. Run setup first.[/yellow]")
        return 1

    token_env = op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    token = os.environ.get(token_env, "").strip()
    if not token:
        console.print(f"[red]{token_env} is not set.[/red]")
        return 1

    mapping = op_cfg.get("mapping") or {}
    try:
        secrets, warnings = opsec.fetch_onepassword_secrets(
            service_account_token=token,
            mapping=mapping,
            use_cache=False,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fetch failed: {exc}[/red]")
        return 1

    if not secrets:
        console.print("[yellow]No secrets resolved.[/yellow]")
        for w in warnings:
            console.print(f"[yellow]warning:[/yellow] {w}")
        return 0

    override = bool(op_cfg.get("override_existing", False)) or args.apply
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
    if args.apply:
        console.print(f"\n  [green]Exported {applied} secret(s) into current process.[/green]")
    else:
        console.print("\n  Dry-run. Hermes applies these automatically on next process startup.")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    op_cfg["enabled"] = False
    save_config(cfg)
    console.print("[green]Disabled.[/green] 1Password secrets will not be pulled on next startup.")
    return 0


def _parse_mapping(entry: str) -> tuple[str, str]:
    if "=" not in entry:
        raise argparse.ArgumentTypeError("mapping must be ENV_VAR=op://Vault/Item/field")
    key, ref = entry.split("=", 1)
    key = key.strip()
    ref = ref.strip()
    if not key or not ref:
        raise argparse.ArgumentTypeError("mapping must be ENV_VAR=op://Vault/Item/field")
    if not opsec._is_valid_env_name(key):  # type: ignore[attr-defined]
        raise argparse.ArgumentTypeError(f"invalid environment variable name: {key}")
    if not ref.startswith("op://"):
        raise argparse.ArgumentTypeError("1Password reference must start with op://")
    return key, ref


def _print_mapping(console: Console, mapping: Dict[str, str]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Env var", style="cyan")
    table.add_column("1Password reference")
    for key, ref in sorted(mapping.items()):
        table.add_row(str(key), str(ref))
    console.print(table)


def _op_version(binary: Path) -> str:
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


def _yn(value: bool) -> str:
    return "[green]yes[/green]" if value else "[dim]no[/dim]"
