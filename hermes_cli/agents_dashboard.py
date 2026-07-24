"""Agents Dashboard TUI module.

Provides a navigable terminal dashboard for the Hermes AI team. Uses
``rich`` for rendering (table, panels, syntax highlighting) when
available, with a plain-text fallback for non-interactive environments.

The dashboard:
- Reuses ``hermes_cli.agents.load_agents()`` (no duplicated registry parsing).
- Shows all agents with profile, model, provider, role, status,
  can_edit_files, escalates_to, and skills.
- Marks the ``default`` profile as fallback/legacy (not a primary worker).
- Supports interactive numeric selection to inspect a single agent's details.
- Exits cleanly on ``q``, Esc, or Ctrl-C.
- Falls back to a non-interactive table dump when stdin is not a TTY.
"""

from __future__ import annotations

import sys
from typing import Any

from hermes_cli.agents import AgentSpec, load_agents

# Escape character as delivered by most terminals when the user presses Esc.
# In a cooked-mode terminal, a bare Esc often arrives as ``\x1b`` (sometimes
# followed by a sequence), so we check for a leading ``\x1b``.
ESC = "\x1b"

# Sentinel values that mean "quit the dashboard" in any prompt.
_QUIT_KEYS = ("q", "Q", "quit", "exit", ESC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_interactive() -> bool:
    """Return True if stdin appears to be a real TTY."""
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _skills_str(skills: list[str]) -> str:
    if not skills:
        return "-"
    return ", ".join(skills)


def _edit_label(spec: AgentSpec) -> str:
    val = spec.can_edit_files
    if val is True:
        return "yes"
    if val == "limited":
        return "limited"
    return "no"


def _status_label(spec: AgentSpec) -> str:
    if spec.profile == "default":
        return f"{spec.status} (fallback)"
    return spec.status


def _is_fallback(spec: AgentSpec) -> bool:
    return spec.profile == "default" or spec.role == "fallback"


# ---------------------------------------------------------------------------
# Rich-based rendering
# ---------------------------------------------------------------------------

def _render_rich_table(agents: list[AgentSpec]) -> str:
    """Render the agent team as a rich table and return the string output."""
    from rich.console import Console
    from rich.table import Table

    console = Console(file=sys.stdout, force_terminal=False, width=120)
    table = Table(title="Hermes AI Team", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Agent", style="bold")
    table.add_column("Profile")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Role")
    table.add_column("Status")
    table.add_column("Edit")
    table.add_column("Escalates")

    for idx, agent in enumerate(agents, 1):
        fallback = _is_fallback(agent)
        style = "dim" if fallback else ""
        table.add_row(
            str(idx),
            agent.name,
            agent.profile,
            agent.model or "-",
            agent.provider or "-",
            agent.role,
            _status_label(agent),
            _edit_label(agent),
            agent.escalates_to,
            style=style,
        )
    console.print(table)
    # Capture: rich Console writes to stdout; we need the string.
    return ""  # Actually printed to stdout already


def _render_rich_table_str(agents: list[AgentSpec]) -> str:
    """Render the agent team as a rich table and return it as a string."""
    from rich.console import Console
    from rich.table import Table
    import io

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    table = Table(title="Hermes AI Team", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Agent", style="bold")
    table.add_column("Profile")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Role")
    table.add_column("Status")
    table.add_column("Edit")
    table.add_column("Escalates")

    for idx, agent in enumerate(agents, 1):
        fallback = _is_fallback(agent)
        style = "dim" if fallback else ""
        table.add_row(
            str(idx),
            agent.name,
            agent.profile,
            agent.model or "-",
            agent.provider or "-",
            agent.role,
            _status_label(agent),
            _edit_label(agent),
            agent.escalates_to,
            style=style,
        )
    console.print(table)
    return buf.getvalue()


def _render_agent_detail_str(agent: AgentSpec) -> str:
    """Render a single agent's full details as a string."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    import io

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)

    # Build a detail table
    detail = Table(show_header=False, box=None, show_lines=False)
    detail.add_column("Field", style="bold", width=16)
    detail.add_column("Value")

    detail.add_row("Agent", agent.name)
    detail.add_row("Profile", agent.profile)
    detail.add_row("Model", agent.model or "-")
    detail.add_row("Provider", agent.provider or "-")
    detail.add_row("Base URL", agent.base_url or "-")
    detail.add_row("Role", agent.role)
    detail.add_row("Description", agent.description or "-")
    detail.add_row("Can edit files", _edit_label(agent))
    detail.add_row("Escalates to", agent.escalates_to)
    detail.add_row("Skills", _skills_str(agent.skills))
    detail.add_row("Status", _status_label(agent))

    title = f"Agent: {agent.name}"
    if _is_fallback(agent):
        title += "  [fallback/legacy]"

    console.print(Panel(detail, title=title, border_style="cyan"))
    return buf.getvalue()


def _render_actions_str(interactive: bool = True) -> str:
    """Render the actions/help panel as a string.

    When ``interactive`` is False (non-TTY / --no-detail), the hint about
    pressing keys is replaced with guidance to run in a TTY.
    """
    from rich.console import Console
    from rich.panel import Panel
    import io

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    if interactive:
        hint = "[dim]Press a number to inspect an agent, q/Esc to quit.[/dim]"
    else:
        hint = "[dim]Run 'hermes agents dashboard' in a TTY to inspect agent details interactively.[/dim]"
    actions = (
        "[bold]Actions:[/bold]\n"
        "  hermes agents orchestrate \"<task>\"   Send a task to orch for planning\n"
        "  hermes kanban list                     View team task board\n"
        "  hermes profile list                    Inspect profile configs\n"
        "\n"
        + hint
    )
    console.print(Panel(actions, title="Actions", border_style="blue"))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Plain-text fallback (no rich)
# ---------------------------------------------------------------------------

def _render_plain_table(agents: list[AgentSpec]) -> str:
    """Render the agent team as a plain text table (no rich dependency)."""
    cols = ["#", "Agent", "Profile", "Model", "Provider", "Role", "Status", "Edit", "Escalates"]
    rows = []
    for idx, a in enumerate(agents, 1):
        rows.append([
            str(idx),
            a.name,
            a.profile,
            a.model or "-",
            a.provider or "-",
            a.role,
            _status_label(a),
            _edit_label(a),
            a.escalates_to,
        ])
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  "
    lines = ["Hermes AI Team", ""]
    lines.append(sep.join(c.ljust(widths[i]) for i, c in enumerate(cols)))
    lines.append(sep.join("-" * widths[i] for i in range(len(cols))))
    for row in rows:
        lines.append(sep.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    lines.append("")
    lines.append("Actions:")
    lines.append('  hermes agents orchestrate "<task>"   Send a task to orch for planning')
    lines.append("  hermes kanban list                     View team task board")
    lines.append("  hermes profile list                    Inspect profile configs")
    return "\n".join(lines)


def _render_plain_detail(agent: AgentSpec) -> str:
    """Render a single agent's details as plain text (no rich)."""
    lines = [f"Agent: {agent.name}"]
    if _is_fallback(agent):
        lines.append("  [fallback/legacy]")
    lines.append(f"  Profile:       {agent.profile}")
    lines.append(f"  Model:         {agent.model or '-'}")
    lines.append(f"  Provider:      {agent.provider or '-'}")
    lines.append(f"  Base URL:      {agent.base_url or '-'}")
    lines.append(f"  Role:          {agent.role}")
    lines.append(f"  Description:   {agent.description or '-'}")
    lines.append(f"  Can edit:      {_edit_label(agent)}")
    lines.append(f"  Escalates to:  {agent.escalates_to}")
    lines.append(f"  Skills:        {_skills_str(agent.skills)}")
    lines.append(f"  Status:        {_status_label(agent)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_dashboard_str(agents: list[AgentSpec] | None = None) -> str:
    """Render the dashboard as a string (non-interactive mode).

    Uses rich when available, plain text otherwise. Returns the full
    dashboard output (table + actions panel).
    """
    resolved = agents if agents is not None else load_agents()
    try:
        table = _render_rich_table_str(resolved)
        actions = _render_actions_str(interactive=False)
        return table + actions
    except ImportError:
        return _render_plain_table(resolved)


def render_agent_detail_str(agent: AgentSpec) -> str:
    """Render a single agent detail as a string."""
    try:
        return _render_agent_detail_str(agent)
    except ImportError:
        return _render_plain_detail(agent)


def run_dashboard(agents: list[AgentSpec] | None = None) -> int:
    """Run the interactive or fallback dashboard.

    In an interactive TTY: shows the table, prompts for a number to
    inspect an agent detail, or q/Esc to quit. Loops until quit.

    In a non-TTY: prints the table + actions and exits 0.

    Returns 0 on clean exit, non-zero on error.
    """
    resolved = agents if agents is not None else load_agents()

    if not resolved:
        print("No agents found. Run 'hermes agents init' to create a starter registry.")
        return 0

    # Non-interactive: dump table and exit.
    if not _is_interactive():
        print(render_dashboard_str(resolved))
        return 0

    # Interactive: show table + actions, loop for selection.
    return _interactive_loop(resolved)


def _interactive_loop(agents: list[AgentSpec]) -> int:
    """Interactive selection loop using rich + input().

    Quit keys: ``q``, ``Q``, ``quit``, ``exit``, or Esc (``\\x1b``).
    In the detail prompt, Esc goes back to the dashboard (not quit), so
    the user can Esc out of a detail view without accidentally exiting.
    """
    try:
        from rich.console import Console
        console: Any = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    while True:
        # Render the dashboard
        if use_rich:
            console.print(_render_rich_table_str(agents))
            console.print(_render_actions_str(interactive=True))
        else:
            print(_render_plain_table(agents))

        # Prompt — q/Esc quit, number selects an agent.
        try:
            choice = input("\nSelect agent # (or q/Esc to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice in _QUIT_KEYS or choice.startswith(ESC):
            return 0

        # Try to parse as a number
        try:
            idx = int(choice)
        except ValueError:
            print("Invalid input. Enter a number, or q/Esc to quit.")
            continue

        if idx < 1 or idx > len(agents):
            print(f"Number out of range (1-{len(agents)}).")
            continue

        agent = agents[idx - 1]
        detail = render_agent_detail_str(agent)
        if use_rich:
            console.print(detail)
        else:
            print(detail)

        # Wait for user to go back or quit.
        # Enter = back to dashboard, q = quit, Esc = back to dashboard.
        try:
            back = input("\nPress Enter/Esc to go back, q to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if back in ("q", "Q"):
            return 0
        # Esc or Enter or anything else: loop and re-render the dashboard.