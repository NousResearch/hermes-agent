"""CLI handlers for ``hermes secrets onepassword ...``."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
from pathlib import Path
from typing import Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import onepassword as op_source
from hermes_cli.config import load_config, save_config

_DEFAULT_ENV_FILE = "~/.hermes/secrets/1password.env"
_EXAMPLE_LINES = """# Hermes + 1Password secret references
#
# Replace placeholders with your actual 1Password secret references.
# Keep op:// references here, not plaintext API keys.
#
# Example format:
#   OPENROUTER_API_KEY=op://Private/OpenRouter API Key/credential
#
# Uncomment and replace these placeholders:
# OPENROUTER_API_KEY=op://<vault>/<item-openrouter>/<field>
# RUNWARE_API_KEY=op://<vault>/<item-runware>/<field>
# FAL_KEY=op://<vault>/<item-fal>/<field>
"""


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``onepassword`` subcommand tree to a parent parser."""
    sub = parent_parser.add_subparsers(dest="secrets_op_command")

    setup = sub.add_parser(
        "setup",
        help="Configure Hermes to pull secrets from 1Password CLI references",
    )
    setup.add_argument(
        "--env-file",
        default=_DEFAULT_ENV_FILE,
        help="Env-style file containing NAME=op://vault/item/field references",
    )
    setup.add_argument(
        "--service-account-token-env",
        default="OP_SERVICE_ACCOUNT_TOKEN",
        help="Env var that may hold a 1Password Service Account token",
    )
    setup.add_argument(
        "--op-path",
        default="",
        help="Explicit path to the op binary (default: resolve from PATH)",
    )
    setup.add_argument(
        "--no-create-env-file",
        action="store_true",
        help="Do not create the references file if it is missing",
    )
    setup.set_defaults(func=cmd_setup)

    status = sub.add_parser("status", help="Show 1Password config and readiness")
    status.set_defaults(func=cmd_status)

    sync = sub.add_parser("sync", help="Resolve references now and report actions")
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Export resolved secrets into this process (default: dry-run)",
    )
    sync.set_defaults(func=cmd_sync)

    disable = sub.add_parser("disable", help="Turn off the 1Password integration")
    disable.set_defaults(func=cmd_disable)


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]1Password Secrets setup[/bold]\n\n"
            "Hermes will read API keys from 1Password via [cyan]op read[/cyan].\n"
            "The references file stores only [cyan]op://...[/cyan] references, "
            "not plaintext secrets.",
            border_style="cyan",
        )
    )

    binary = op_source.find_op(op_path=args.op_path)
    if binary:
        console.print(f"  [green]✓[/green] op binary: {binary} ({_op_version(binary)})")
    else:
        console.print(
            "  [yellow]op binary not found.[/yellow] Install 1Password CLI or "
            "pass --op-path. Setup can still write config."
        )

    env_file = _resolve_env_file(args.env_file)
    if not env_file.exists() and not args.no_create_env_file:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(_EXAMPLE_LINES, encoding="utf-8")
        os.chmod(env_file, stat.S_IRUSR | stat.S_IWUSR)
        console.print(f"  [green]✓[/green] created references file: {env_file}")
    elif env_file.exists():
        try:
            os.chmod(env_file, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        console.print(f"  [green]✓[/green] references file: {env_file}")
    else:
        console.print(f"  [yellow]references file missing:[/yellow] {env_file}")

    cfg = load_config()
    secrets_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    secrets_cfg["enabled"] = True
    secrets_cfg["env_file"] = args.env_file
    secrets_cfg["service_account_token_env"] = args.service_account_token_env
    secrets_cfg["op_path"] = args.op_path
    secrets_cfg.setdefault("override_existing", True)
    secrets_cfg.setdefault("cache_ttl_seconds", 300)
    save_config(cfg)

    console.print()
    console.print(
        "[green]✓ 1Password secret source is enabled.[/green]\n"
        "  Edit the references file with real op:// references, then run:\n"
        "  [cyan]hermes secrets onepassword sync[/cyan]"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}

    enabled = bool(op_cfg.get("enabled"))
    env_file_cfg = op_cfg.get("env_file", _DEFAULT_ENV_FILE)
    env_file = _resolve_env_file(env_file_cfg)
    service_env = op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    op_path = str(op_cfg.get("op_path", "") or "")
    binary = op_source.find_op(op_path=op_path)
    ref_count, warn_count = _reference_counts(env_file)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled", _yn(enabled))
    table.add_row("References file", str(env_file))
    table.add_row("References file exists", _yn(env_file.exists()))
    table.add_row("op:// references", str(ref_count))
    table.add_row("Reference warnings", str(warn_count))
    table.add_row("Service token env var", service_env)
    table.add_row("Service token in env", _yn(bool(os.environ.get(service_env))))
    table.add_row("Override existing", _yn(bool(op_cfg.get("override_existing", False))))
    table.add_row("Cache TTL (s)", str(op_cfg.get("cache_ttl_seconds", 300)))
    table.add_row("Configured op path", op_path or "[dim](PATH)[/dim]")
    table.add_row("op binary", f"{binary} ({_op_version(binary)})" if binary else "[yellow]not found[/yellow]")
    console.print(Panel(table, title="1Password Secrets", border_style="cyan"))

    if not enabled:
        console.print("\n  Run [cyan]hermes secrets onepassword setup[/cyan] to enable.")
    elif not binary:
        console.print("\n  [yellow]Enabled but op is not available.[/yellow]")
    elif not env_file.exists():
        console.print("\n  [yellow]Enabled but the references file does not exist.[/yellow]")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}
    if not op_cfg.get("enabled"):
        console.print(
            "[yellow]1Password integration is disabled. Run "
            "`hermes secrets onepassword setup` first.[/yellow]"
        )
        return 1

    try:
        secrets, warnings = op_source.fetch_onepassword_secrets(
            env_file=op_cfg.get("env_file", _DEFAULT_ENV_FILE),
            op_path=str(op_cfg.get("op_path", "") or ""),
            cache_ttl_seconds=float(op_cfg.get("cache_ttl_seconds", 300)),
            use_cache=False,
        )
    except RuntimeError as exc:
        console.print(f"[red]Fetch failed: {exc}[/red]")
        return 1

    if not secrets:
        console.print("[yellow]No secrets resolved.[/yellow]")
        for w in warnings:
            console.print(f"[yellow]warning:[/yellow] {w}")
        return 0

    override = bool(op_cfg.get("override_existing", False)) or args.apply
    service_env = op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Action")
    applied = 0
    for key in sorted(secrets):
        if key == service_env:
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
        console.print(
            "\n  This was a dry-run — secrets are picked up automatically on the "
            "next [cyan]hermes[/cyan] invocation. Re-run with [cyan]--apply[/cyan] "
            "to export into the current process instead."
        )
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    op_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]Disabled.[/green]  1Password secrets will NOT be pulled on the next "
        "Hermes invocation. The references file is left untouched."
    )
    return 0


def _resolve_env_file(path: str) -> Path:
    raw = str(path or _DEFAULT_ENV_FILE).strip()
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except Exception:  # noqa: BLE001 — CLI status should stay best-effort
        home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    if raw == "~/.hermes" or raw.startswith("~/.hermes/"):
        suffix = raw.removeprefix("~/.hermes").lstrip("/")
        return home / suffix
    if raw.startswith("~"):
        return Path(raw).expanduser()
    resolved = Path(raw)
    if resolved.is_absolute():
        return resolved
    return home / resolved


def _reference_counts(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    try:
        references, warnings = op_source.parse_reference_file(path)
    except RuntimeError:
        return 0, 1
    return len(references), len(warnings)


def _op_version(binary: Path) -> str:
    try:
        res = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
        if res.returncode == 0:
            return (res.stdout or res.stderr).strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "version unknown"


def _yn(value: bool) -> str:
    return "[green]yes[/green]" if value else "[dim]no[/dim]"
