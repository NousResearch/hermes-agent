from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from plugins.memory import load_memory_provider
from hermes_constants import get_hermes_home

_console = Console()


def _load_provider():
    hermes_home = get_hermes_home()
    provider = load_memory_provider("layered")
    provider.initialize(session_id="skill-candidates-cli", hermes_home=str(hermes_home), platform="cli")
    return provider


def _print_candidates_table(candidates: list[dict], console: Console) -> None:
    if not candidates:
        console.print("[dim]No skill candidates found.[/]\n")
        return
    table = Table(title="Skill Candidates")
    table.add_column("Name", style="bold cyan")
    table.add_column("Status", style="dim")
    table.add_column("Reason", style="dim")
    table.add_column("Recurrence", style="dim")
    for item in candidates:
        table.add_row(
            item.get("skill_name", ""),
            item.get("review_status", ""),
            item.get("review_gate_reason", ""),
            str(item.get("effective_recurrence", "")),
        )
    console.print(table)
    for item in candidates:
        console.print(
            f"- {item.get('skill_name', '')} | {item.get('review_status', '')} | "
            f"{item.get('review_gate_reason', '')} | recurrence={item.get('effective_recurrence', '')}"
        )
    console.print()


def handle_skill_candidates_slash(cmd: str, console: Optional[Console] = None) -> None:
    c = console or _console
    parts = cmd.strip().split(maxsplit=3)
    if len(parts) < 2:
        c.print("[bold red]Usage:[/] /skill-candidates [list|inspect|approve|reject] ...\n")
        return

    action = parts[1].lower()
    provider = _load_provider()

    try:
        if action == "list":
            _print_candidates_table(provider.list_skill_candidates(), c)
            return

        if action == "inspect":
            if len(parts) < 3:
                c.print("[bold red]Usage:[/] /skill-candidates inspect <name>\n")
                return
            details = provider.inspect_skill_candidate(parts[2])
            if not details:
                c.print(f"[bold red]Error:[/] No candidate found for {parts[2]}\n")
                return
            c.print_json(json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True))
            c.print()
            return

        if action == "approve":
            if len(parts) < 3:
                c.print("[bold red]Usage:[/] /skill-candidates approve <name>\n")
                return
            installed_path = provider.approve_skill_candidate(parts[2])
            c.print(f"[bold green]Approved and installed:[/] {installed_path}\n")
            return

        if action == "reject":
            if len(parts) < 3:
                c.print("[bold red]Usage:[/] /skill-candidates reject <name> [reason]\n")
                return
            reason = parts[3] if len(parts) >= 4 else "manual_reject"
            provider.reject_skill_candidate(parts[2], reason=reason)
            c.print(f"[bold yellow]Rejected:[/] {parts[2]} ({reason})\n")
            return

        c.print("[bold red]Unknown subcommand:[/] Use list, inspect, approve, reject\n")
    finally:
        try:
            provider.shutdown()
        except Exception:
            pass
