"""CLI handlers for ``hermes secrets onepassword ...``.

Subcommands:
    setup    — configure one or more ENV=op://... references and test them
    status   — show current config + op availability
    sync     — resolve configured references now and show what would be applied
    set      — add/update one ENV_VAR -> op:// reference
    remove   — remove one configured ENV_VAR mapping
    disable  — flip ``secrets.onepassword.enabled`` to False
"""

from __future__ import annotations

import argparse
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import onepassword as op_secret
from hermes_cli.config import load_config, save_config


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``onepassword`` subcommand tree to a parent parser."""

    sub = parent_parser.add_subparsers(dest="secrets_op_command")

    setup = sub.add_parser(
        "setup",
        help="Configure ENV_VAR=op://... references and enable 1Password",
    )
    setup.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="ENV_VAR=op://vault/item/field",
        help="Add an env-var mapping. May be repeated.",
    )
    setup.add_argument(
        "--account",
        default="",
        help="Optional 1Password account shorthand/email for `op read --account`.",
    )
    setup.add_argument(
        "--token-env",
        default=None,
        help="Env var holding an optional 1Password service-account token.",
    )
    setup.add_argument(
        "--skip-test",
        action="store_true",
        help="Save config without resolving references immediately.",
    )
    setup.set_defaults(func=cmd_setup)

    status = sub.add_parser("status", help="Show config + op availability")
    status.set_defaults(func=cmd_status)

    sync = sub.add_parser("sync", help="Resolve references now and report actions")
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Actually export resolved values into this process (default: dry-run)",
    )
    sync.set_defaults(func=cmd_sync)

    set_cmd = sub.add_parser("set", help="Add or update one ENV_VAR reference")
    set_cmd.add_argument("env_var", help="Environment variable to populate")
    set_cmd.add_argument("reference", help="1Password reference: op://vault/item/field")
    set_cmd.add_argument("--enable", action="store_true", help="Also enable the integration")
    set_cmd.set_defaults(func=cmd_set)

    remove = sub.add_parser("remove", aliases=["rm"], help="Remove one ENV_VAR reference")
    remove.add_argument("env_var", help="Environment variable mapping to remove")
    remove.set_defaults(func=cmd_remove)

    disable = sub.add_parser("disable", help="Turn off the 1Password integration")
    disable.set_defaults(func=cmd_disable)


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]1Password secret references setup[/bold]\n\n"
            "Hermes reads configured [cyan]op://[/cyan] references with the official "
            "[cyan]op[/cyan] CLI at startup. Sign in with [cyan]op signin[/cyan] "
            "or set [cyan]OP_SERVICE_ACCOUNT_TOKEN[/cyan] before enabling.",
            border_style="cyan",
        )
    )

    if not args.skip_test:
        binary = op_secret.find_op()
        if binary is None:
            console.print(
                "[red]op CLI is not installed or not on PATH.[/red]\n"
                "Install it from https://developer.1password.com/docs/cli/get-started/ "
                "and run this command again, or pass --skip-test to save config only."
            )
            return 1
        console.print(f"[green]✓[/green] op CLI found at {binary}")
    else:
        binary = None

    refs = _parse_maps(args.map, console)
    if refs is None:
        return 1

    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    op_cfg["env"] = _merged_refs(op_cfg)
    existing = op_cfg["env"]
    existing.update(refs)
    if args.account:
        op_cfg["account"] = args.account.strip()
    if args.token_env:
        op_cfg["service_account_token_env"] = args.token_env.strip() or "OP_SERVICE_ACCOUNT_TOKEN"
    else:
        op_cfg.setdefault("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    op_cfg.setdefault("cache_ttl_seconds", 300)
    op_cfg.setdefault("override_existing", True)

    if not existing:
        console.print(
            "[yellow]No references configured yet.[/yellow] Add one with:\n"
            "  [cyan]hermes secrets onepassword set OPENAI_API_KEY "
            "op://Private/OpenAI/credential --enable[/cyan]"
        )
        return 1

    if not args.skip_test:
        try:
            secrets, warnings = op_secret.fetch_onepassword_secrets(
                references=existing,
                binary=binary,
                account=str(op_cfg.get("account", "") or ""),
                service_account_token_env=str(
                    op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")
                ),
                use_cache=False,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Test fetch failed: {exc}[/red]")
            return 1
        _print_resolved_table(
            console,
            secrets,
            warnings,
            apply=False,
            override_existing=bool(op_cfg.get("override_existing", True)),
            token_env=str(op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")),
        )

    op_cfg["enabled"] = True
    save_config(cfg)
    console.print("\n[green]✓ 1Password secret references are enabled.[/green]")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}
    refs = _merged_refs(op_cfg)
    token_env = op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled", _yn(bool(op_cfg.get("enabled"))))
    table.add_row("op binary", str(op_secret.find_op() or "[yellow]not found[/yellow]"))
    table.add_row("Account", str(op_cfg.get("account", "") or "[dim](default)[/dim]"))
    table.add_row("Token env var", token_env)
    table.add_row("Token in env", _yn(bool(os.environ.get(token_env))))
    table.add_row("References", str(len(refs) if isinstance(refs, dict) else 0))
    table.add_row("Override existing", _yn(bool(op_cfg.get("override_existing", True))))
    table.add_row("Cache TTL (s)", str(op_cfg.get("cache_ttl_seconds", 300)))
    console.print(Panel(table, title="1Password", border_style="cyan"))

    if refs and isinstance(refs, dict):
        ref_table = Table(show_header=True, header_style="bold")
        ref_table.add_column("Env var", style="cyan")
        ref_table.add_column("1Password reference", style="dim")
        for key, ref in sorted(refs.items()):
            ref_table.add_row(str(key), str(ref))
        console.print(ref_table)
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}
    if not op_cfg.get("enabled"):
        console.print(
            "[yellow]1Password integration is disabled. Run "
            "`hermes secrets onepassword setup --map ENV=op://...` first.[/yellow]"
        )
        return 1

    refs = _merged_refs(op_cfg)
    if not refs:
        console.print("[red]No 1Password references configured.[/red]")
        return 1

    token_env = op_cfg.get("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    override = bool(op_cfg.get("override_existing", True))
    if args.apply:
        result = op_secret.apply_onepassword_secrets(
            enabled=True,
            references=refs,
            override_existing=override,
            cache_ttl_seconds=_coerce_cache_ttl(op_cfg.get("cache_ttl_seconds", 300)),
            account=str(op_cfg.get("account", "") or ""),
            service_account_token_env=str(token_env),
        )
        _print_apply_result_table(console, result)
        if result.error:
            console.print(f"[red]Fetch failed: {result.error}[/red]")
            return 1
        console.print(
            f"\n[green]Exported {len(result.applied)} secret(s) into current process.[/green]"
        )
    else:
        try:
            secrets, warnings = op_secret.fetch_onepassword_secrets(
                references=refs,
                account=str(op_cfg.get("account", "") or ""),
                service_account_token_env=str(token_env),
                use_cache=False,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Fetch failed: {exc}[/red]")
            return 1
        _print_resolved_table(
            console,
            secrets,
            warnings,
            apply=False,
            override_existing=override,
            token_env=str(token_env),
        )
        console.print(
            "\nThis was a dry-run — secrets are picked up automatically on the next "
            "[cyan]hermes[/cyan] invocation. Re-run with [cyan]--apply[/cyan] "
            "to export into this process."
        )
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    console = Console()
    valid, warnings = op_secret._validate_references({args.env_var: args.reference})
    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if not valid:
        return 1

    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    op_cfg["env"] = _merged_refs(op_cfg)
    env_var, reference = next(iter(valid.items()))
    op_cfg["env"][env_var] = reference
    op_cfg.setdefault("cache_ttl_seconds", 300)
    op_cfg.setdefault("override_existing", True)
    op_cfg.setdefault("service_account_token_env", "OP_SERVICE_ACCOUNT_TOKEN")
    if args.enable:
        op_cfg["enabled"] = True
    save_config(cfg)
    console.print(f"[green]✓[/green] mapped {env_var} → {reference}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    op_cfg["env"] = _merged_refs(op_cfg)
    refs = op_cfg["env"]
    legacy_refs = op_cfg.get("references") if isinstance(op_cfg.get("references"), dict) else {}
    if args.env_var not in refs and args.env_var not in legacy_refs:
        console.print(f"[yellow]{args.env_var} is not mapped.[/yellow]")
        return 0
    refs.pop(args.env_var, None)
    if isinstance(legacy_refs, dict):
        legacy_refs.pop(args.env_var, None)
    save_config(cfg)
    console.print(f"[green]✓[/green] removed {args.env_var}")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    op_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]Disabled.[/green]  1Password references will not be resolved on "
        "the next Hermes invocation."
    )
    return 0


def _merged_refs(op_cfg: dict) -> dict[str, str]:
    """Return env mappings, including legacy ``references`` entries."""

    refs: dict[str, str] = {}
    legacy = op_cfg.get("references")
    current = op_cfg.get("env")
    if isinstance(legacy, dict):
        refs.update({str(k): str(v) for k, v in legacy.items()})
    if isinstance(current, dict):
        refs.update({str(k): str(v) for k, v in current.items()})
    return refs


def _parse_maps(values: list[str], console: Console) -> dict[str, str] | None:
    refs: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            console.print(f"[red]Invalid --map value {value!r}; expected ENV_VAR=op://...[/red]")
            return None
        key, ref = value.split("=", 1)
        refs[key.strip()] = ref.strip()
    valid, warnings = op_secret._validate_references(refs)
    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if refs and not valid:
        return None
    return valid


def _print_resolved_table(
    console: Console,
    secrets: dict[str, str],
    warnings: list[str],
    *,
    apply: bool,
    override_existing: bool = True,
    token_env: str = "OP_SERVICE_ACCOUNT_TOKEN",
) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Action")
    for key in sorted(secrets):
        if key == token_env:
            table.add_row(key, "[dim]skip (bootstrap token)[/dim]")
            continue
        already = bool(os.environ.get(key))
        if already and not override_existing:
            table.add_row(key, "[dim]skip (already set)[/dim]")
            continue
        if apply:
            action = "[green]exported[/green]" + (" (overrode)" if already else "")
        else:
            action = "[green]would export[/green]" + (" (overrides)" if already else "")
        table.add_row(key, action)
    console.print(table)
    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def _print_apply_result_table(console: Console, result: op_secret.FetchResult) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Action")
    for key in sorted(result.applied):
        table.add_row(key, "[green]exported[/green]")
    for key in sorted(result.skipped):
        table.add_row(key, "[dim]skip[/dim]")
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def _coerce_cache_ttl(value, default: float = 300) -> float:  # noqa: ANN001
    try:
        ttl = float(value)
    except (TypeError, ValueError):
        return default
    if ttl < 0:
        return default
    return ttl


def _yn(value: bool) -> str:
    return "[green]yes[/green]" if value else "[dim]no[/dim]"
