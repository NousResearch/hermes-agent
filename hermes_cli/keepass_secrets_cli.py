"""CLI handlers for ``hermes secrets keepass ...``.

KeePassXC is local-first: the ``keepassxc-cli`` binary is NOT installed by Hermes (the
vendor ships it with KeePassXC itself), and there is no account or token to authenticate.
Setup therefore verifies the pieces a user already has — the CLI, the ``.kdbx`` path, the
key file, and a database key — by opening the database once.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agent.secret_sources import keepass as kp_src
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
    section_cfg,
    yn,
)
from hermes_cli.config import load_config, save_config
from hermes_cli.secret_prompt import masked_secret_prompt

_DOCS_URL = "https://keepassxc.org/download/"


def _kp_cfg() -> dict:
    return section_cfg(load_config(), "keepass")


def _kp_cfg_for_write(cfg: dict) -> dict:
    return cfg.setdefault("secrets", {}).setdefault("keepass", {})


def _references(kp_cfg: dict) -> dict:
    env = kp_cfg.get("env")
    return env if isinstance(env, dict) else {}


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``keepass`` subcommand tree to a parent parser."""
    register_subcommands(parent_parser, "secrets_keepass_command", (
        ("setup", "Verify keepassxc-cli + the database, then enable the source", (
            arg("--db", "Path to the .kdbx database (e.g. ~/KeePass/Passwords.kdbx)"),
            arg("--keyfile", "Key file that opens the database"),
            arg("--password-file", "File whose first line is the database password"),
            arg("--password-env", f"Env var holding the database password "
                                  f"(default {kp_src.DEFAULT_PASSWORD_ENV})"),
            arg("--binary-path", "Absolute path to keepassxc-cli (skips PATH lookup)"),
        )),
        ("status", "Show config + keepassxc-cli + mapped entry paths", ()),
        ("list", "List entries in the database (paths and metadata; never passwords)", cmd_list, ()),
        ("set", "Map an env var to an entry path", (
            arg("env_var", "Environment variable name, e.g. OPENAI_API_KEY"),
            arg("entry", "Entry path, e.g. \"Work/OpenAI\" (see `hermes secrets keepass list`)"),
        )),
        ("remove", "Remove an env-var → entry mapping", (
            arg("env_var", "Environment variable name to unmap"),
        )),
        ("sync", "Resolve the mapped entries now and report what changed", (
            flag("--apply", "Actually export resolved values into the current process (default: dry-run)"),
        )),
        ("disable", "Turn off the KeePass integration", ()),
    ))


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(Panel.fit(
        "[bold]KeePassXC secret source setup[/bold]\n\n"
        "Hermes resolves entry passwords through your local\n"
        "[cyan]keepassxc-cli[/cyan] and the encrypted [cyan].kdbx[/cyan] file.  Nothing is\n"
        "uploaded anywhere: the database password is written to the CLI's stdin\n"
        "and the resolved values go straight into the process environment.\n\n"
        f"Don't have the CLI yet? Install KeePassXC: [cyan]{_DOCS_URL}[/cyan]",
        border_style="cyan"))

    cfg = load_config()
    kp_cfg = _kp_cfg_for_write(cfg)

    console.print()
    console.print("[bold]Step 1[/bold]  Locate keepassxc-cli")
    binary_path = (args.binary_path or cfg_str(kp_cfg, "binary_path"))
    binary = kp_src.find_keepassxc(binary_path)
    if binary is None:
        console.print(f"  [red]✗ {binary_path} is not an executable keepassxc-cli.[/red]"
                      if binary_path else "  [red]✗ keepassxc-cli not found on PATH.[/red]")
        console.print(f"  Install KeePassXC: {_DOCS_URL}")
        return 1
    console.print(f"  [green]✓[/green] {binary}  ({cli_version(binary)})")

    console.print()
    console.print("[bold]Step 2[/bold]  Locate the database")
    db = (args.db or cfg_str(kp_cfg, "db")).strip()
    if not db:
        if not sys.stdin.isatty():
            console.print("[red]✗ No --db and nothing configured.[/red]")
            return 1
        db = console.input("  Path to the .kdbx database: ").strip()
    db_path = Path(db).expanduser()
    if not db_path.is_file():
        console.print(f"  [red]✗ {db_path} is not a file.[/red]")
        return 1
    console.print(f"  [green]✓[/green] {db_path}")

    console.print()
    console.print("[bold]Step 3[/bold]  Open it once to prove the key works")
    keyfile = (args.keyfile or cfg_str(kp_cfg, "keyfile")).strip()
    password_env = (args.password_env or cfg_str(kp_cfg, "password_env")
                    or kp_src.DEFAULT_PASSWORD_ENV).strip()
    if args.password_file:
        kp_cfg["password_file"] = args.password_file.strip()
    try:
        password = kp_src.resolve_password(kp_cfg, password_env, env=os.environ) or ""
    except RuntimeError as exc:
        console.print(f"  [red]✗ {exc}[/red]")
        return 1
    if not password and not keyfile and sys.stdin.isatty():
        password = masked_secret_prompt(f"  Database password ({db_path.name}): ")
    try:
        kp_src.verify_database(db=str(db_path), key=password or None, keyfile=keyfile,
                               binary_path=str(binary))
    except RuntimeError as exc:
        console.print(f"  [red]✗ {exc}[/red]")
        return 1
    finally:
        del password
    console.print("  [green]✓[/green] database opened")

    kp_cfg["db"] = str(db_path)
    if keyfile:
        kp_cfg["keyfile"] = keyfile
    if args.password_env:
        kp_cfg["password_env"] = password_env
    if binary_path:
        kp_cfg["binary_path"] = binary_path
    kp_cfg["enabled"] = True
    kp_cfg.setdefault("env", {})
    kp_cfg.setdefault("override_existing", True)
    save_config(cfg)

    console.print()
    console.print("[green]✓ KeePass secret source is enabled.[/green]")
    console.print(
        "  List entries:     [cyan]hermes secrets keepass list[/cyan]\n"
        "  Map a secret:     [cyan]hermes secrets keepass set OPENAI_API_KEY \"Work/OpenAI\"[/cyan]\n"
        "  Preview:          [cyan]hermes secrets keepass sync[/cyan]"
    )
    if not kp_cfg.get("password_file"):
        console.print(
            f"  [dim]The startup source needs the database password without prompting: put it in\n"
            f"  ~/.hermes/.env as [cyan]{password_env}[/cyan], or set "
            f"[cyan]secrets.keepass.password_file[/cyan] to a 0600 file. "
            f"A key-file-only database needs neither.[/dim]"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    kp_cfg = _kp_cfg()
    enabled = bool(kp_cfg.get("enabled"))
    db = cfg_str(kp_cfg, "db")
    keyfile = cfg_str(kp_cfg, "keyfile")
    password_env = kp_cfg.get("password_env", kp_src.DEFAULT_PASSWORD_ENV)
    password_file = cfg_str(kp_cfg, "password_file")
    references = _references(kp_cfg)
    binary_path = cfg_str(kp_cfg, "binary_path")
    binary = kp_src.find_keepassxc(binary_path)

    if password_file:
        key_source = f"password file ({password_file})"
    elif os.environ.get(password_env):
        key_source = f"{password_env} (set)"
    elif keyfile:
        key_source = "key file only"
    else:
        key_source = "[yellow]not configured[/yellow]"

    print_status_panel(console, "KeePassXC secret source", (
        ("Enabled", yn(enabled)),
        ("Database", db if db and Path(db).expanduser().is_file() else
         (f"[yellow]{db or 'not set'}[/yellow]")),
        ("Key file", keyfile or "[dim]none[/dim]"),
        ("Database key", key_source),
        ("Override existing", yn(bool(kp_cfg.get("override_existing", True)))),
        ("keepassxc-cli", f"{binary} ({cli_version(binary)})" if binary else "[yellow]not found[/yellow]"),
        ("Mapped entries", str(len(references))),
    ))

    if references:
        print_table(console, (("Env var", {"style": "cyan"}), "Entry path"),
                    ((name, str(references[name])) for name in sorted(references)))

    if not enabled:
        console.print("\n  Run [cyan]hermes secrets keepass setup[/cyan] to enable.")
        return 0
    if not db:
        console.print("\n  [yellow]No database configured — "
                      "run `hermes secrets keepass setup`.[/yellow]")
    elif not references:
        console.print("\n  [yellow]No entries mapped yet.[/yellow]  Add one: "
                      "[cyan]hermes secrets keepass set ENV_VAR \"Group/Entry\"[/cyan]")
    return 0


def _open_database(kp_cfg: dict) -> tuple:
    """``(db, key, keyfile, binary)`` for a read: the configured key, else one masked prompt."""
    db = cfg_str(kp_cfg, "db")
    if not db:
        raise RuntimeError("No database configured — run `hermes secrets keepass setup`.")
    db_path = Path(db).expanduser()
    if not db_path.is_file():
        raise RuntimeError(f"{db_path} is not a file — check secrets.keepass.db.")
    keyfile = cfg_str(kp_cfg, "keyfile")
    key = kp_src.resolve_password(kp_cfg, kp_cfg.get("password_env", kp_src.DEFAULT_PASSWORD_ENV),
                                  env=os.environ)
    if key is None and not keyfile and sys.stdin.isatty():
        key = masked_secret_prompt(f"Database password ({db_path.name}): ").strip() or None
    return str(db_path), key, keyfile, kp_src.find_keepassxc(cfg_str(kp_cfg, "binary_path"))


def cmd_list(args: argparse.Namespace) -> int:
    """Entry paths a user can paste into ``set`` (metadata only — never a password)."""
    console = Console()
    kp_cfg = _kp_cfg()
    try:
        db, key, keyfile, binary = _open_database(kp_cfg)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    if binary is None:
        console.print(f"[red]keepassxc-cli not found — install KeePassXC ({_DOCS_URL}) or set "
                      "secrets.keepass.binary_path.[/red]")
        return 1

    try:
        entries = kp_src.export_entries(db=db, key=key, keyfile=keyfile, binary_path=str(binary))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    finally:
        del key

    if not entries:
        console.print("[yellow]The database has no entries.[/yellow]")
        return 0
    print_table(console, (("Entry path", {"style": "cyan"}), "Username", "URL"),
                ((e.path, e.username or "-", e.url.splitlines()[0] if e.url else "-") for e in entries))
    console.print(f"\n  {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}.  "
                  "Map one: [cyan]hermes secrets keepass set ENV_VAR \"<entry path>\"[/cyan]")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    console = Console()
    valid, warnings = kp_src._validate_references({args.env_var: args.entry})
    if args.env_var not in valid:
        for w in warnings:
            console.print(f"[red]{w}[/red]")
        return 1
    cfg = load_config()
    kp_cfg = _kp_cfg_for_write(cfg)
    if not isinstance(kp_cfg.get("env"), dict):
        kp_cfg["env"] = {}
    kp_cfg["env"][args.env_var] = valid[args.env_var]
    save_config(cfg)
    console.print(f"[green]✓[/green] mapped [cyan]{args.env_var}[/cyan] → {valid[args.env_var]}")
    if not kp_cfg.get("enabled"):
        console.print("  [yellow]Note: the integration is disabled — run "
                      "[cyan]hermes secrets keepass setup[/cyan] to turn it on.[/yellow]")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    kp_cfg = _kp_cfg_for_write(cfg)
    env_map = kp_cfg.get("env")
    if not isinstance(env_map, dict) or args.env_var not in env_map:
        console.print(f"[yellow]{args.env_var} is not mapped.[/yellow]")
        return 1
    del env_map[args.env_var]
    save_config(cfg)
    console.print(f"[green]✓[/green] removed mapping for [cyan]{args.env_var}[/cyan]")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    from hermes_constants import get_hermes_home

    from agent.secret_sources.keepass import KeePassSource

    console = Console()
    kp_cfg = _kp_cfg()
    if not require_enabled(console, kp_cfg, "KeePass", "keepass"):
        return 1
    references = _references(kp_cfg)
    if not references:
        console.print("[yellow]No entries mapped.  Add one with "
                      "`hermes secrets keepass set ENV_VAR \"Group/Entry\"`.[/yellow]")
        return 0

    section = {**kp_cfg, "enabled": True}
    home_path = get_hermes_home()

    if args.apply:
        # The startup path applies it (skip/override/protection policy lives in one place).
        from agent.secret_sources.registry import apply_all

        report = apply_all({"keepass": section}, home_path)
        source_report = report.sources[0] if report.sources else None
        if source_report is None or not source_report.result.ok:
            console.print(f"[red]{source_report.result.error if source_report else 'no result'}[/red]")
            return 1
        rows = [(name, "[green]exported[/green]") for name in sorted(source_report.applied)]
        rows += [(name, "[dim]skipped (already set / password var)[/dim]")
                 for name in sorted(source_report.skipped_existing + source_report.skipped_protected)]
        rows += [(name, "[red]unresolved (see warnings)[/red]") for name in sorted(references)
                 if name not in source_report.applied]
        print_table(console, (("Env var", {"style": "cyan"}), "Action"), rows,
                    source_report.result.warnings)
        console.print(f"\n  [green]Exported {len(source_report.applied)} secret(s) into this "
                      "process.[/green]")
        return 0

    result = KeePassSource().fetch(section, home_path)
    if not result.ok:
        console.print(f"[red]{result.error}[/red]")
        return 1
    print_table(console, (("Env var", {"style": "cyan"}), "Action"),
                ((name, "[green]would export[/green]" if name in result.secrets
                  else "[red]unresolved (see warnings)[/red]") for name in sorted(references)),
                result.warnings)
    console.print("\n  This was a dry-run — entries resolve automatically on the next "
                  "[cyan]hermes[/cyan] invocation.  Re-run with [cyan]--apply[/cyan] to export "
                  "into this process instead.")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    return disable_secret_source(
        "keepass",
        "[green]Disabled.[/green]  KeePass entries will NOT be resolved on the next "
        "Hermes invocation.")
