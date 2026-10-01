"""Helpers shared by the Bitwarden and 1Password ``hermes secrets`` CLIs.

Import-light on purpose: ``hermes_cli.secrets_cli`` must stay free of the Bitwarden backend
(``cryptography``) at import time, so nothing here touches a secret-source backend.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hermes_cli.config import load_config, save_config
from hermes_cli.secret_prompt import masked_secret_prompt


def yn(b: bool) -> str:
    return "[green]yes[/green]" if b else "[dim]no[/dim]"


def section_cfg(cfg: dict, section: str) -> dict:
    """``cfg["secrets"][section]`` or ``{}``."""
    return (cfg.get("secrets") or {}).get(section) or {}


def cfg_str(cfg: dict, key: str) -> str:
    """A stripped string config value; ``""`` for missing/None."""
    return str(cfg.get(key, "") or "").strip()


def arg(name: str, help_text: str, **kwargs) -> tuple:
    """One ``add_argument`` spec for :func:`register_subcommands`."""
    return name, dict(help=help_text, **kwargs)


def flag(name: str, help_text: str) -> tuple:
    """A boolean ``store_true`` spec for :func:`register_subcommands`."""
    return arg(name, help_text, action="store_true")


def register_subcommands(parent: argparse.ArgumentParser, dest: str, commands: Iterable) -> None:
    """Attach ``(name, help, handler, [arg(...), ...])`` subcommands to ``parent``."""
    sub = parent.add_subparsers(dest=dest)
    for name, help_text, func, arguments in commands:
        parser = sub.add_parser(name, help=help_text)
        for arg_name, kwargs in arguments:
            parser.add_argument(arg_name, **kwargs)
        parser.set_defaults(func=func)


def require_enabled(console: Console, cfg: dict, product: str, command: str) -> bool:
    """Print the "integration is disabled" hint and return False unless ``cfg["enabled"]``."""
    if cfg.get("enabled"):
        return True
    console.print(f"[yellow]{product} integration is disabled.  Run "
                  f"`hermes secrets {command} setup` first.[/yellow]")
    return False


def print_status_panel(console: Console, title: str, rows: Iterable) -> None:
    """Two-column key/value table inside a cyan panel (the ``status`` layout)."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    for label, value in rows:
        table.add_row(label, value)
    console.print(Panel(table, title=title, border_style="cyan"))


def print_table(console: Console, columns: Sequence, rows: Iterable,
                warnings: Iterable[str] = (), indent: str = "") -> None:
    """Headed table; ``columns`` entries are ``header`` or ``(header, add_column kwargs)``.

    ``warnings`` are echoed after the table, one ``warning:`` line each.
    """
    table = Table(show_header=True, header_style="bold")
    for col in columns:
        header, kwargs = (col, {}) if isinstance(col, str) else col
        table.add_column(header, **kwargs)
    for row in rows:
        table.add_row(*row)
    console.print(table)
    for w in warnings:
        console.print(f"{indent}[yellow]warning:[/yellow] {w}")


def cli_version(binary: Path) -> str:
    """Return the first line of ``<binary> --version`` or ``"version unknown"``."""
    try:
        res = subprocess.run([str(binary), "--version"], capture_output=True, text=True, encoding='utf-8',
                             errors='replace', timeout=5)
        if res.returncode == 0:
            return (res.stdout or res.stderr).strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "version unknown"


def secret_cli_env() -> dict:
    """Env for a secret-manager CLI child (``bws`` / ``op``).

    Intentionally receives tokens — no scrub, no HOME rewrite (both CLIs store state under the
    real user home).
    """
    from tools.environments.local import build_subprocess_env

    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    env.setdefault("NO_COLOR", "1")
    return env


def rotate_token(
    console: Console, given: Optional[str], token_env: str, *, flag: str, intro: str, prompt: str,
    verify: Optional[Callable[[str], bool]], save: Callable[[str, str], object],
    env_path: Callable[[], object], clear_caches: Callable[[], object],
    disabled_note: Optional[str],
) -> int:
    """Shared ``token`` subcommand: prompt, optionally verify, then persist. Returns the exit code.

    ``verify(token)`` prints its own diagnostics and returns False to abort without touching
    .env — so a bad paste never bricks the working token; None skips verification
    (``--no-verify``). ``save``/``env_path`` are passed in (not imported here) so each CLI
    module's own ``save_env_value``/``get_env_path`` bindings — which tests monkeypatch — stay
    in effect. Old cached pulls are keyed on the previous token's fingerprint; clearing them
    makes the next startup fetch fresh with the new credential.
    """
    token = (given or "").strip()
    if not token:
        if not sys.stdin.isatty():
            console.print(f"[red]No TTY — pass the token with {flag}.[/red]")
            return 1
        console.print(intro)
        token = masked_secret_prompt(prompt).strip()
    if not token:
        console.print("[red]Empty token, aborting.[/red]")
        return 1
    if verify is not None and not verify(token):
        return 1
    save(token_env, token)
    os.environ[token_env] = token
    clear_caches()
    console.print(f"[green]✓[/green] stored in {env_path()} as {token_env}.  "
                  "Takes effect on the next Hermes invocation.")
    if disabled_note:
        console.print(disabled_note)
    return 0


def prompt_index(console: Console, prompt: str, count: int, *,
                 allow_empty: bool = False, empty_message: Optional[str] = None) -> int:
    """Loop until the user enters an integer in ``1..count``; return it.

    Blank input returns 0 when ``allow_empty``; otherwise ``empty_message`` (if any)
    is printed and the prompt repeats.
    """
    while True:
        choice = console.input(prompt).strip()
        if not choice:
            if allow_empty:
                return 0
            if empty_message:
                console.print(empty_message)
            continue
        try:
            idx = int(choice)
        except ValueError:
            console.print("  [red]Enter a number.[/red]")
            continue
        if 1 <= idx <= count:
            return idx
        console.print(f"  [red]Out of range — pick 1-{count}.[/red]")


def disable_secret_source(section: str, message: str) -> int:
    """Set ``secrets.<section>.enabled = False`` in config.yaml and print ``message``."""
    cfg = load_config()
    cfg.setdefault("secrets", {}).setdefault(section, {})["enabled"] = False
    save_config(cfg)
    Console().print(message)
    return 0
