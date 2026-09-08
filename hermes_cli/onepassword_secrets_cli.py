"""CLI handlers for ``hermes secrets onepassword ...``.

Unlike Bitwarden, the ``op`` binary is NOT auto-installed: 1Password publishes the CLI through OS
package managers and signed installers, so Hermes expects an already-installed, already-
authenticated ``op`` and never downloads one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from agent.secret_sources import onepassword as op_src
from hermes_cli._secrets_common import (
    arg,
    cfg_str,
    cli_version,
    disable_secret_source,
    flag,
    print_status_panel,
    print_table,
    register_subcommands,
    require_enabled,
    rotate_token,
    secret_cli_env,
    section_cfg,
    yn,
)
from hermes_cli.config import get_env_path, load_config, save_config, save_env_value

_DEFAULT_TOKEN_ENV = "OP_SERVICE_ACCOUNT_TOKEN"
_DOCS_URL = "https://developer.1password.com/docs/cli/get-started/"

# Old name kept bound: tests call ``onepassword_secrets_cli._op_version`` directly.
_op_version = cli_version


def _op_cfg() -> dict:
    return section_cfg(load_config(), "onepassword")


def _op_cfg_for_write(cfg: dict) -> dict:
    return cfg.setdefault("secrets", {}).setdefault("onepassword", {})


def _references(op_cfg: dict) -> dict:
    env = op_cfg.get("env")
    return env if isinstance(env, dict) else {}



def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``onepassword`` subcommand tree to a parent parser."""
    register_subcommands(parent_parser, "secrets_op_command", (
        ("setup", "Verify the op CLI, set account / token env var, and enable", cmd_setup, (
            arg("--account", "1Password account shorthand or sign-in address (op --account)"),
            arg("--token-env", f"Env var holding a service-account token (default {_DEFAULT_TOKEN_ENV})"),
            arg("--token", "Service-account token to store in .env non-interactively"),
            arg("--binary-path", "Absolute path to the op binary (skips PATH lookup)"),
        )),
        ("status", "Show config + op binary + references", cmd_status, ()),
        ("token", "Rotate the service-account token: validate and store it in .env", cmd_token, (
            arg("--token", "Provide the new token non-interactively (default: masked prompt)"),
            flag("--no-verify", "Store without probing 1Password first (not recommended)"),
        )),
        ("set", "Map an env var to an op:// reference", cmd_set, (
            arg("env_var", "Environment variable name, e.g. OPENAI_API_KEY"),
            arg("reference", "1Password reference, e.g. op://Private/OpenAI/api key"),
        )),
        ("remove", "Remove an env-var → reference mapping", cmd_remove, (
            arg("env_var", "Environment variable name to unmap"),
        )),
        ("sync", "Resolve references now and report what changed", cmd_sync, (
            flag("--apply", "Actually export resolved values into the current shell (default: dry-run)"),
        )),
        ("disable", "Turn off the 1Password integration", cmd_disable, ()),
    ))


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
    op_cfg = _op_cfg_for_write(cfg)

    console.print()
    console.print("[bold]Step 1[/bold]  Locate the op CLI")
    binary_path = (args.binary_path or op_cfg.get("binary_path", "") or "").strip()
    binary = op_src.find_op(binary_path)
    if binary is None:
        console.print(
            f"  [red]✗ {binary_path} is not an executable op binary.[/red]"
            if binary_path
            else "  [red]✗ op not found on PATH.[/red]"
        )
        console.print(f"  Install the 1Password CLI: {_DOCS_URL}")
        return 1
    console.print(f"  [green]✓[/green] {binary}  ({_op_version(binary)})")
    if binary_path:
        op_cfg["binary_path"] = binary_path

    if args.account and args.account.strip():
        op_cfg["account"] = args.account.strip()
        console.print(f"  Account: [cyan]{op_cfg['account']}[/cyan]")

    console.print()
    console.print("[bold]Step 2[/bold]  Authentication")
    token_env = (args.token_env or op_cfg.get("service_account_token_env") or _DEFAULT_TOKEN_ENV).strip()
    op_cfg["service_account_token_env"] = token_env

    token = (args.token or "").strip()
    if token:
        save_env_value(token_env, token)
        os.environ[token_env] = token
        console.print(f"  [green]✓[/green] service-account token stored in {get_env_path()} as {token_env}")
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
    op_cfg = _op_cfg()

    enabled = bool(op_cfg.get("enabled"))
    account = cfg_str(op_cfg, "account")
    token_env = op_cfg.get("service_account_token_env", _DEFAULT_TOKEN_ENV)
    binary_path = cfg_str(op_cfg, "binary_path")
    references = _references(op_cfg)
    token_set = bool(os.environ.get(token_env))

    binary = op_src.find_op(binary_path)

    print_status_panel(console, "1Password secret source", (
        ("Enabled", yn(enabled)),
        ("Account", account or "[dim]default[/dim]"),
        ("Token env var", token_env),
        ("Token in env", yn(token_set)),
        ("Override existing", yn(bool(op_cfg.get("override_existing", True)))),
        ("Cache TTL (s)", str(op_cfg.get("cache_ttl_seconds", 300))),
        ("op binary", f"{binary} ({_op_version(binary)})" if binary else "[yellow]not found[/yellow]"),
        ("References", str(len(references))),
    ))

    if references:
        print_table(console, (("Env var", {"style": "cyan"}), "Reference"),
                    ((name, str(references[name])) for name in sorted(references)))

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
    # Backend validator keeps CLI and startup in agreement; store the validated/stripped value.
    valid, warnings = op_src._validate_references({args.env_var: args.reference})
    if args.env_var not in valid:
        for w in warnings:
            console.print(f"[red]{w}[/red]")
        return 1

    cfg = load_config()
    op_cfg = _op_cfg_for_write(cfg)
    if not isinstance(op_cfg.get("env"), dict):
        op_cfg["env"] = {}
    op_cfg["env"][args.env_var] = valid[args.env_var]
    save_config(cfg)
    console.print(f"[green]✓[/green] mapped [cyan]{args.env_var}[/cyan] → {valid[args.env_var]}")
    if not op_cfg.get("enabled"):
        console.print(
            "  [yellow]Note: the integration is disabled — run "
            "[cyan]hermes secrets onepassword setup[/cyan] to turn it on.[/yellow]"
        )
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    op_cfg = _op_cfg_for_write(cfg)
    env_map = op_cfg.get("env")
    if not isinstance(env_map, dict) or args.env_var not in env_map:
        console.print(f"[yellow]{args.env_var} is not mapped.[/yellow]")
        return 1
    del env_map[args.env_var]
    save_config(cfg)
    console.print(f"[green]✓[/green] removed mapping for [cyan]{args.env_var}[/cyan]")
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    """Rotate the service-account token: verify with ``op whoami`` (unless ``--no-verify``) and
    only then persist to .env, so a bad paste never bricks the working token."""
    console = Console()
    op_cfg = _op_cfg()
    token_env = op_cfg.get("service_account_token_env", _DEFAULT_TOKEN_ENV)
    account = cfg_str(op_cfg, "account")
    binary_path = cfg_str(op_cfg, "binary_path")

    def verify(token: str) -> bool:
        binary = op_src.find_op(binary_path)
        if binary is None:
            console.print(
                f"[red]op CLI not found — install it ({_DOCS_URL}) or "
                "re-run with --no-verify to store anyway.[/red]"
            )
            return False
        console.print("Verifying with `op whoami`…")
        who = _op_whoami(binary, account, token_value=token)
        if who is None:
            console.print("[red]✗ New token was rejected by op — nothing was changed.[/red]")
            return False
        console.print(f"[green]✓ Token accepted[/green] ({who}).")
        return True

    return rotate_token(
        console, args.token, token_env,
        flag="--token",
        intro=(
            "Create a new service-account token at "
            "https://my.1password.com → Developer → Service Accounts.\n"
        ),
        prompt=f"Paste new token ({token_env}): ",
        verify=None if args.no_verify else verify,
        save=save_env_value, env_path=get_env_path, clear_caches=op_src.clear_caches,
        disabled_note=None if op_cfg.get("enabled") else (
            "[yellow]Note: the 1Password integration is currently disabled — "
            "run `hermes secrets onepassword setup` to turn it on.[/yellow]"
        ),
    )


def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    op_cfg = _op_cfg()
    if not require_enabled(console, op_cfg, "1Password", "onepassword"):
        return 1

    references = _references(op_cfg)
    if not references:
        console.print(
            "[yellow]No op:// references configured.  Add one with "
            "`hermes secrets onepassword set ENV_VAR \"op://…\"`.[/yellow]"
        )
        return 0

    account = cfg_str(op_cfg, "account")
    token_env = op_cfg.get("service_account_token_env", _DEFAULT_TOKEN_ENV)
    binary_path = cfg_str(op_cfg, "binary_path")

    # --apply uses the startup code path so the skip/override/token-guard policy lives in one place.
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
        print_table(
            console, (("Env var", {"style": "cyan"}), "Action"),
            [(name, "[green]exported[/green]") for name in sorted(result.applied)]
            + [(name, "[dim]skipped (already set / token var)[/dim]") for name in sorted(result.skipped)],
            result.warnings,
        )
        console.print(f"\n  [green]Exported {len(result.applied)} secret(s) into current process.[/green]")
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

    def action(name: str) -> str:
        if name == token_env:
            return "[dim]skip (token var)[/dim]"
        if name not in secrets:
            return "[red]unresolved (see warnings)[/red]"
        already = bool(os.environ.get(name))
        if already and not override:
            return "[dim]skip (already set)[/dim]"
        return "[green]would export[/green]" + (" (overrides)" if already else "")

    print_table(console, (("Env var", {"style": "cyan"}), "Action"),
                ((name, action(name)) for name in sorted(references)), warnings)
    console.print(
        "\n  This was a dry-run — references resolve automatically on the next "
        "[cyan]hermes[/cyan] invocation.  Re-run with [cyan]--apply[/cyan] to export "
        "into the current shell instead."
    )
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    return disable_secret_source(
        "onepassword",
        "[green]Disabled.[/green]  1Password references will NOT be resolved on the "
        "next Hermes invocation.\n"
        "  Your reference mappings are left in config.yaml — remove them with "
        "[cyan]hermes secrets onepassword remove ENV_VAR[/cyan] if you no longer "
        "need them.",
    )


def _op_whoami(binary: Path, account: str, *, token_value: str = "") -> Optional[str]:
    """Short identity string if op is authenticated, else None. ``token_value`` probes a candidate
    token via the child's ``OP_SERVICE_ACCOUNT_TOKEN`` without touching the caller's environment."""
    cmd = [str(binary), "whoami"]
    if account:
        cmd += ["--account", account]
    env = secret_cli_env()
    if token_value:
        env["OP_SERVICE_ACCOUNT_TOKEN"] = token_value
    try:
        res = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    return (res.stdout or "").strip().replace("\n", " ")[:120] or "authenticated"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from rich.table import Table  # noqa: F401,E402
import sys  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'masked_secret_prompt': ('hermes_cli.secret_prompt', 'masked_secret_prompt'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
