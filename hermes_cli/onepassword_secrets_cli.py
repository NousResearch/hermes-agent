"""CLI handlers for ``hermes secrets onepassword ...``.

Subcommands:
    setup    — verify the op CLI, set account / token env var, enable
    status   — show config + op binary + auth + configured references
    set      — map an env var to an ``op://…`` reference
    remove   — drop a mapping
    sync     — resolve references now and show what would be applied (dry-run)
    disable  — flip ``secrets.onepassword.enabled`` to False

Unlike Bitwarden, the ``op`` binary is NOT auto-installed: 1Password publishes
the CLI through OS package managers and signed installers, so Hermes expects
an already-installed, already-authenticated ``op`` and never downloads one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import onepassword as op_src
from hermes_cli.config import (
    get_env_path,
    load_config,
    save_config,
    save_env_value,
)
from hermes_cli.secret_prompt import masked_secret_prompt

_DEFAULT_TOKEN_ENV = "OP_SERVICE_ACCOUNT_TOKEN"
_DOCS_URL = "https://developer.1password.com/docs/cli/get-started/"


# ---------------------------------------------------------------------------
# Argparse wiring — called from hermes_cli.main
# ---------------------------------------------------------------------------


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``onepassword`` subcommand tree to a parent parser."""
    sub = parent_parser.add_subparsers(dest="secrets_op_command")

    setup = sub.add_parser(
        "setup",
        help="Verify the op CLI, set account / token env var, and enable",
    )
    setup.add_argument(
        "--account",
        help="1Password account shorthand or sign-in address (op --account)",
    )
    setup.add_argument(
        "--token-env",
        help=f"Env var holding a service-account token (default {_DEFAULT_TOKEN_ENV})",
    )
    setup.add_argument(
        "--token",
        help="Service-account token to store in .env non-interactively",
    )
    setup.add_argument(
        "--binary-path",
        help="Absolute path to the op binary (skips PATH lookup)",
    )
    setup.set_defaults(func=cmd_setup)

    status = sub.add_parser("status", help="Show config + op binary + references")
    status.set_defaults(func=cmd_status)

    token = sub.add_parser(
        "token",
        help="Rotate the service-account token: validate and store it in .env",
    )
    token.add_argument(
        "--token",
        help="Provide the new token non-interactively (default: masked prompt)",
    )
    token.add_argument(
        "--no-verify",
        action="store_true",
        help="Store without probing 1Password first (not recommended)",
    )
    token.set_defaults(func=cmd_token)

    set_p = sub.add_parser("set", help="Map an env var to an op:// reference")
    set_p.add_argument("env_var", help="Environment variable name, e.g. OPENAI_API_KEY")
    set_p.add_argument("reference", help="1Password reference, e.g. op://Private/OpenAI/api key")
    set_p.set_defaults(func=cmd_set)

    remove = sub.add_parser("remove", help="Remove an env-var → reference mapping")
    remove.add_argument("env_var", help="Environment variable name to unmap")
    remove.set_defaults(func=cmd_remove)

    sync = sub.add_parser("sync", help="Resolve references now and report what changed")
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Actually export resolved values into the current shell (default: dry-run)",
    )
    sync.set_defaults(func=cmd_sync)

    disable = sub.add_parser("disable", help="Turn off the 1Password integration")
    disable.set_defaults(func=cmd_disable)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]1Password secret source setup[/bold]\n\n"
            "Hermes resolves [cyan]op://vault/item/field[/cyan] references through your\n"
            "already-installed, already-authenticated 1Password CLI (`op`).\n\n"
            f"Don't have it yet? Install + sign in: [cyan]{_DOCS_URL}[/cyan]",
            border_style="cyan",
        )
    )

    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})

    # ------------------------------------------------------------------ binary
    console.print()
    console.print("[bold]Step 1[/bold]  Locate the op CLI")
    binary_path = (args.binary_path or op_cfg.get("binary_path", "") or "").strip()
    binary = op_src.find_op(binary_path)
    if binary is None:
        if binary_path:
            console.print(f"  [red]✗ {binary_path} is not an executable op binary.[/red]")
        else:
            console.print("  [red]✗ op not found on PATH.[/red]")
        console.print(f"  Install the 1Password CLI: {_DOCS_URL}")
        return 1
    console.print(f"  [green]✓[/green] {binary}  ({_op_version(binary)})")
    if binary_path:
        op_cfg["binary_path"] = binary_path

    # ----------------------------------------------------------------- account
    if args.account and args.account.strip():
        op_cfg["account"] = args.account.strip()
        console.print(f"  Account: [cyan]{op_cfg['account']}[/cyan]")

    # ------------------------------------------------------------------- token
    console.print()
    console.print("[bold]Step 2[/bold]  Authentication")
    token_env = (args.token_env or op_cfg.get("service_account_token_env")
                 or _DEFAULT_TOKEN_ENV).strip()
    op_cfg["service_account_token_env"] = token_env

    token = (args.token or "").strip()
    if token:
        save_env_value(token_env, token)
        os.environ[token_env] = token
        console.print(f"  [green]✓[/green] service-account token stored in "
                      f"{get_env_path()} as {token_env}")
    elif os.environ.get(token_env):
        console.print(f"  [green]✓[/green] using service-account token from {token_env}")
    else:
        who = _op_whoami(binary, op_cfg.get("account", ""))
        if who:
            console.print(f"  [green]✓[/green] using existing op session ({who})")
        else:
            console.print(
                "  [yellow]No service-account token and no active op session "
                "detected.[/yellow]\n"
                "  Either run [cyan]op signin[/cyan] (desktop/interactive) or set a "
                f"service-account token in {token_env}, then re-run status."
            )

    # ----------------------------------------------------------------- enable
    op_cfg["enabled"] = True
    op_cfg.setdefault("env", {})
    op_cfg.setdefault("cache_ttl_seconds", 300)
    op_cfg.setdefault("override_existing", True)
    save_config(cfg)

    console.print()
    console.print("[green]✓ 1Password secret source is enabled.[/green]")
    console.print(
        "  Map credentials:  [cyan]hermes secrets onepassword set OPENAI_API_KEY "
        "\"op://Private/OpenAI/api key\"[/cyan]\n"
        "  Preview:          [cyan]hermes secrets onepassword sync[/cyan]\n"
        "  Status:           [cyan]hermes secrets onepassword status[/cyan]"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}

    enabled = bool(op_cfg.get("enabled"))
    account = str(op_cfg.get("account", "") or "").strip()
    token_env = op_cfg.get("service_account_token_env", _DEFAULT_TOKEN_ENV)
    binary_path = str(op_cfg.get("binary_path", "") or "").strip()
    references = op_cfg.get("env") if isinstance(op_cfg.get("env"), dict) else {}
    token_set = bool(os.environ.get(token_env))

    binary = op_src.find_op(binary_path)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled", _yn(enabled))
    table.add_row("Account", account or "[dim]default[/dim]")
    table.add_row("Token env var", token_env)
    table.add_row("Token in env", _yn(token_set))
    table.add_row("Override existing", _yn(bool(op_cfg.get("override_existing", True))))
    table.add_row("Cache TTL (s)", str(op_cfg.get("cache_ttl_seconds", 300)))
    if binary:
        table.add_row("op binary", f"{binary} ({_op_version(binary)})")
    else:
        table.add_row("op binary", "[yellow]not found[/yellow]")
    table.add_row("References", str(len(references)))

    console.print(Panel(table, title="1Password secret source", border_style="cyan"))

    if references:
        ref_table = Table(show_header=True, header_style="bold")
        ref_table.add_column("Env var", style="cyan")
        ref_table.add_column("Reference")
        for name in sorted(references):
            ref_table.add_row(name, str(references[name]))
        console.print(ref_table)

    if not enabled:
        console.print("\n  Run [cyan]hermes secrets onepassword setup[/cyan] to enable.")
        return 0
    if binary and not token_set:
        who = _op_whoami(binary, account)
        if who:
            console.print(f"\n  [green]Active op session:[/green] {who}")
        else:
            console.print(
                f"\n  [yellow]No active op session and {token_env} is unset — "
                "Hermes will warn and skip 1Password on next startup.[/yellow]"
            )
    if not references:
        console.print(
            "\n  [yellow]No references mapped yet.[/yellow]  Add one: "
            "[cyan]hermes secrets onepassword set ENV_VAR \"op://…\"[/cyan]"
        )
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    console = Console()
    # Reuse the backend validator so the CLI and startup paths agree on what a
    # valid reference is — and store the *validated/stripped* value, not the
    # raw arg (so trailing whitespace never lands in config.yaml).
    valid, warnings = op_src._validate_references({args.env_var: args.reference})
    if args.env_var not in valid:
        for w in warnings:
            console.print(f"[red]{w}[/red]")
        return 1

    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    env_map = op_cfg.get("env")
    if not isinstance(env_map, dict):
        env_map = {}
        op_cfg["env"] = env_map
    env_map[args.env_var] = valid[args.env_var]
    save_config(cfg)
    console.print(
        f"[green]✓[/green] mapped [cyan]{args.env_var}[/cyan] → "
        f"{valid[args.env_var]}"
    )
    if not op_cfg.get("enabled"):
        console.print(
            "  [yellow]Note: the integration is disabled — run "
            "[cyan]hermes secrets onepassword setup[/cyan] to turn it on.[/yellow]"
        )
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    env_map = op_cfg.get("env")
    if not isinstance(env_map, dict) or args.env_var not in env_map:
        console.print(f"[yellow]{args.env_var} is not mapped.[/yellow]")
        return 1
    del env_map[args.env_var]
    save_config(cfg)
    console.print(f"[green]✓[/green] removed mapping for [cyan]{args.env_var}[/cyan]")
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    """Rotate the 1Password service-account token without the full setup flow.

    Prompts for (or accepts via ``--token``) a new service-account token,
    verifies it with ``op whoami`` (unless ``--no-verify``), and only then
    persists it to .env — so a bad paste never bricks the working token.
    """
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}
    token_env = op_cfg.get("service_account_token_env", _DEFAULT_TOKEN_ENV)
    account = str(op_cfg.get("account", "") or "").strip()
    binary_path = str(op_cfg.get("binary_path", "") or "").strip()

    token = (args.token or "").strip()
    if not token:
        if not sys.stdin.isatty():
            console.print("[red]No TTY — pass the token with --token.[/red]")
            return 1
        console.print(
            "Create a new service-account token at "
            "https://my.1password.com → Developer → Service Accounts.\n"
        )
        token = masked_secret_prompt(f"Paste new token ({token_env}): ").strip()
    if not token:
        console.print("[red]Empty token, aborting.[/red]")
        return 1

    if not args.no_verify:
        binary = op_src.find_op(binary_path)
        if binary is None:
            console.print(
                f"[red]op CLI not found — install it ({_DOCS_URL}) or "
                "re-run with --no-verify to store anyway.[/red]"
            )
            return 1
        console.print("Verifying with `op whoami`…")
        who = _op_whoami(binary, account, token_value=token)
        if who is None:
            console.print(
                "[red]✗ New token was rejected by op — nothing was changed.[/red]"
            )
            return 1
        console.print(f"[green]✓ Token accepted[/green] ({who}).")

    save_env_value(token_env, token)
    os.environ[token_env] = token
    # Cached resolutions are keyed on the previous token's fingerprint;
    # drop them so the next startup resolves fresh with the new credential.
    op_src.clear_caches()
    console.print(
        f"[green]✓[/green] stored in {get_env_path()} as {token_env}.  "
        "Takes effect on the next Hermes invocation."
    )
    if not op_cfg.get("enabled"):
        console.print(
            "[yellow]Note: the 1Password integration is currently disabled — "
            "run `hermes secrets onepassword setup` to turn it on.[/yellow]"
        )
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = (cfg.get("secrets") or {}).get("onepassword") or {}
    if not op_cfg.get("enabled"):
        console.print(
            "[yellow]1Password integration is disabled.  Run "
            "`hermes secrets onepassword setup` first.[/yellow]"
        )
        return 1

    references = op_cfg.get("env") if isinstance(op_cfg.get("env"), dict) else {}
    if not references:
        console.print(
            "[yellow]No op:// references configured.  Add one with "
            "`hermes secrets onepassword set ENV_VAR \"op://…\"`.[/yellow]"
        )
        return 0

    account = str(op_cfg.get("account", "") or "").strip()
    token_env = op_cfg.get("service_account_token_env", _DEFAULT_TOKEN_ENV)
    binary_path = str(op_cfg.get("binary_path", "") or "").strip()

    # --apply delegates to the same code path startup uses, so the skip /
    # override / token-guard policy lives in exactly one place.
    if args.apply:
        result = op_src.apply_onepassword_secrets(
            enabled=True,
            env=references,
            account=account,
            service_account_token_env=token_env,
            binary_path=binary_path,
            override_existing=bool(op_cfg.get("override_existing", True)),
            cache_ttl_seconds=0,  # an explicit sync always resolves fresh
        )
        if result.error:
            console.print(f"[red]{result.error}[/red]")
            return 1
        table = Table(show_header=True, header_style="bold")
        table.add_column("Env var", style="cyan")
        table.add_column("Action")
        for name in sorted(result.applied):
            table.add_row(name, "[green]exported[/green]")
        for name in sorted(result.skipped):
            table.add_row(name, "[dim]skipped (already set / token var)[/dim]")
        console.print(table)
        for w in result.warnings:
            console.print(f"[yellow]warning:[/yellow] {w}")
        console.print(
            f"\n  [green]Exported {len(result.applied)} secret(s) into current "
            "process.[/green]"
        )
        return 0

    # Dry-run: resolve fresh (no cache) and preview, mutating nothing.
    try:
        secrets, warnings = op_src.fetch_onepassword_secrets(
            references=references,
            account=account,
            token_env=token_env,
            binary_path=binary_path,
            use_cache=False,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    override = bool(op_cfg.get("override_existing", True))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Env var", style="cyan")
    table.add_column("Action")
    for name in sorted(references):
        if name == token_env:
            table.add_row(name, "[dim]skip (token var)[/dim]")
        elif name not in secrets:
            table.add_row(name, "[red]unresolved (see warnings)[/red]")
        elif os.environ.get(name) and not override:
            table.add_row(name, "[dim]skip (already set)[/dim]")
        else:
            already = bool(os.environ.get(name))
            table.add_row(
                name,
                "[green]would export[/green]" + (" (overrides)" if already else ""),
            )
    console.print(table)
    for w in warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    console.print(
        "\n  This was a dry-run — references resolve automatically on the next "
        "[cyan]hermes[/cyan] invocation.  Re-run with [cyan]--apply[/cyan] to export "
        "into the current shell instead."
    )
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = cfg.setdefault("secrets", {}).setdefault("onepassword", {})
    op_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]Disabled.[/green]  1Password references will NOT be resolved on the "
        "next Hermes invocation.\n"
        "  Your reference mappings are left in config.yaml — remove them with "
        "[cyan]hermes secrets onepassword remove ENV_VAR[/cyan] if you no longer "
        "need them."
    )
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yn(b: bool) -> str:
    return "[green]yes[/green]" if b else "[dim]no[/dim]"


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


def _op_whoami(
    binary: Path, account: str, *, token_value: str = ""
) -> Optional[str]:
    """Return a short identity string if op is authenticated, else None.

    ``token_value``, when given, is passed to the child as
    ``OP_SERVICE_ACCOUNT_TOKEN`` so a candidate token can be probed
    without touching the caller's environment.
    """
    cmd = [str(binary), "whoami"]
    if account:
        cmd += ["--account", account]
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    if token_value:
        env["OP_SERVICE_ACCOUNT_TOKEN"] = token_value
    try:
        res = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    out = (res.stdout or "").strip()
    return out.replace("\n", " ")[:120] or "authenticated"
