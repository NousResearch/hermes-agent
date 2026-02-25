"""Welcome banner, ASCII art, and skills summary for the CLI.

Pure display functions with no HermesCLI state dependency.
"""

from pathlib import Path
from typing import Any, Dict, List

from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# =========================================================================
# ANSI building blocks for conversation display
# =========================================================================

_GOLD = "\033[1;33m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's renderer."""
    _pt_print(_PT_ANSI(text))


# =========================================================================
# ASCII Art & Branding
# =========================================================================

from hermes_cli import __version__ as VERSION

HERMES_AGENT_LOGO = """[bold #FFD700]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD700]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#FFBF00]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#FFBF00]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#CD7F32]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#CD7F32]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""

HERMES_CADUCEUS = """[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀[/]
[#FFBF00]⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]"""

COMPACT_BANNER = """
[bold #FFD700]╔══════════════════════════════════════════════════════════════╗[/]
[bold #FFD700]║[/]  [#FFBF00]⚕ NOUS HERMES[/] [dim #B8860B]- AI Agent Framework[/]              [bold #FFD700]║[/]
[bold #FFD700]║[/]  [#CD7F32]Messenger of the Digital Gods[/]    [dim #B8860B]Nous Research[/]   [bold #FFD700]║[/]
[bold #FFD700]╚══════════════════════════════════════════════════════════════╝[/]
"""


# =========================================================================
# Skills scanning
# =========================================================================


def get_available_skills() -> Dict[str, List[str]]:
    """Scan ~/.hermes/skills/ and return skills grouped by category."""
    import os

    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    skills_dir = hermes_home / "skills"
    skills_by_category = {}

    if not skills_dir.exists():
        return skills_by_category

    for skill_file in skills_dir.rglob("SKILL.md"):
        rel_path = skill_file.relative_to(skills_dir)
        parts = rel_path.parts
        if len(parts) >= 2:
            category = parts[0]
            skill_name = parts[-2]
        else:
            category = "general"
            skill_name = skill_file.parent.name
        skills_by_category.setdefault(category, []).append(skill_name)

    return skills_by_category


# =========================================================================
# Welcome banner
# =========================================================================


def build_welcome_banner(
    console: Console,
    model: str,
    cwd: str,
    tools: List[dict] = None,
    enabled_toolsets: List[str] = None,
    session_id: str = None,
    get_toolset_for_tool=None,
):
    """Build and print a welcome banner with caduceus on left and info on right.

    Args:
        console: Rich Console instance.
        model: Current model name.
        cwd: Current working directory.
        tools: List of tool definitions.
        enabled_toolsets: List of enabled toolset names.
        session_id: Session identifier.
        get_toolset_for_tool: Callable to map tool name -> toolset name.
    """
    from model_tools import TOOLSET_REQUIREMENTS, check_tool_availability

    if get_toolset_for_tool is None:
        from model_tools import get_toolset_for_tool

    tools = tools or []
    enabled_toolsets = enabled_toolsets or []

    _, unavailable_toolsets = check_tool_availability(quiet=True)
    disabled_tools = set()
    for item in unavailable_toolsets:
        disabled_tools.update(item.get("tools", []))

    layout_table = Table.grid(padding=(0, 2))
    layout_table.add_column("left", justify="center")
    layout_table.add_column("right", justify="left")

    left_lines = ["", HERMES_CADUCEUS, ""]
    model_short = model.split("/")[-1] if "/" in model else model
    if len(model_short) > 28:
        model_short = model_short[:25] + "..."
    left_lines.append(f"[#FFBF00]{model_short}[/] [dim #B8860B]·[/] [dim #B8860B]Nous Research[/]")
    left_lines.append(f"[dim #B8860B]{cwd}[/]")
    if session_id:
        left_lines.append(f"[dim #8B8682]Session: {session_id}[/]")
    left_content = "\n".join(left_lines)

    right_lines = ["[bold #FFBF00]Available Tools[/]"]
    toolsets_dict: Dict[str, list] = {}

    for tool in tools:
        tool_name = tool["function"]["name"]
        toolset = get_toolset_for_tool(tool_name) or "other"
        toolsets_dict.setdefault(toolset, []).append(tool_name)

    for item in unavailable_toolsets:
        toolset_id = item.get("id", item.get("name", "unknown"))
        display_name = f"{toolset_id}_tools" if not toolset_id.endswith("_tools") else toolset_id
        if display_name not in toolsets_dict:
            toolsets_dict[display_name] = []
        for tool_name in item.get("tools", []):
            if tool_name not in toolsets_dict[display_name]:
                toolsets_dict[display_name].append(tool_name)

    sorted_toolsets = sorted(toolsets_dict.keys())
    display_toolsets = sorted_toolsets[:8]
    remaining_toolsets = len(sorted_toolsets) - 8

    for toolset in display_toolsets:
        tool_names = toolsets_dict[toolset]
        colored_names = []
        for name in sorted(tool_names):
            if name in disabled_tools:
                colored_names.append(f"[red]{name}[/]")
            else:
                colored_names.append(f"[#FFF8DC]{name}[/]")

        tools_str = ", ".join(colored_names)
        if len(", ".join(sorted(tool_names))) > 45:
            short_names = []
            length = 0
            for name in sorted(tool_names):
                if length + len(name) + 2 > 42:
                    short_names.append("...")
                    break
                short_names.append(name)
                length += len(name) + 2
            colored_names = []
            for name in short_names:
                if name == "...":
                    colored_names.append("[dim]...[/]")
                elif name in disabled_tools:
                    colored_names.append(f"[red]{name}[/]")
                else:
                    colored_names.append(f"[#FFF8DC]{name}[/]")
            tools_str = ", ".join(colored_names)

        right_lines.append(f"[dim #B8860B]{toolset}:[/] {tools_str}")

    if remaining_toolsets > 0:
        right_lines.append(f"[dim #B8860B](and {remaining_toolsets} more toolsets...)[/]")

    right_lines.append("")
    right_lines.append("[bold #FFBF00]Available Skills[/]")
    skills_by_category = get_available_skills()
    total_skills = sum(len(s) for s in skills_by_category.values())

    if skills_by_category:
        for category in sorted(skills_by_category.keys()):
            skill_names = sorted(skills_by_category[category])
            if len(skill_names) > 8:
                display_names = skill_names[:8]
                skills_str = ", ".join(display_names) + f" +{len(skill_names) - 8} more"
            else:
                skills_str = ", ".join(skill_names)
            if len(skills_str) > 50:
                skills_str = skills_str[:47] + "..."
            right_lines.append(f"[dim #B8860B]{category}:[/] [#FFF8DC]{skills_str}[/]")
    else:
        right_lines.append("[dim #B8860B]No skills installed[/]")

    right_lines.append("")
    right_lines.append(f"[dim #B8860B]{len(tools)} tools · {total_skills} skills · /help for commands[/]")

    right_content = "\n".join(right_lines)
    layout_table.add_row(left_content, right_content)

    outer_panel = Panel(
        layout_table,
        title=f"[bold #FFD700]Hermes Agent {VERSION}[/]",
        border_style="#CD7F32",
        padding=(0, 2),
    )

    console.print()
    console.print(HERMES_AGENT_LOGO)
    console.print()
    console.print(outer_panel)
