"""Unified admission-control CLI helpers."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

_console = Console()


def do_audit(console: Console | None = None) -> None:
    from hermes_cli.mcp_config import audit_mcp_integrity
    from hermes_cli.skills_hub import do_audit as audit_skills

    c = console or _console
    c.print("\n[bold]Admission Audit[/]\n")
    drift_messages = audit_mcp_integrity()
    if drift_messages:
        for message in drift_messages:
            c.print(f"[yellow]{message}[/]")
        c.print()
    audit_skills(console=c)


def do_list(console: Console | None = None) -> None:
    from agent.security.admission import admission_store

    c = console or _console
    records = admission_store().list_records()
    if not records:
        c.print("[dim]No admission records.[/]\n")
        return
    table = Table(title="Admission Records")
    table.add_column("Kind", style="dim")
    table.add_column("Name", style="bold cyan")
    table.add_column("Status", style="dim")
    table.add_column("Revision", justify="right")
    table.add_column("Lineage", style="dim")
    for record in records:
        table.add_row(
            str(record.kind),
            record.source.display_name,
            str(record.status),
            str(record.revision),
            record.lineage_id or "",
        )
    c.print(table)
    c.print()


def admission_command(args) -> None:
    action = getattr(args, "admission_action", None)
    if action == "audit":
        do_audit()
    elif action == "list":
        do_list()
    else:
        _console.print("Usage: hermes admission [audit|list]\n")
