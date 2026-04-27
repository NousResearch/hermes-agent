"""Welcome banner, ASCII art, skills summary, and update check for the CLI.

Pure display functions with no HermesCLI state dependency.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI

logger = logging.getLogger(__name__)


# =========================================================================
# ANSI building blocks for conversation display
# =========================================================================

_GOLD = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's renderer."""
    _pt_print(_PT_ANSI(text))


# =========================================================================
# Skin-aware color helpers
# =========================================================================

def _skin_color(key: str, fallback: str) -> str:
    """Get a color from the active skin, or return fallback."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin().get_color(key, fallback)
    except Exception:
        return fallback


def _skin_branding(key: str, fallback: str) -> str:
    """Get a branding string from the active skin, or return fallback."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin().get_branding(key, fallback)
    except Exception:
        return fallback


# =========================================================================
# ASCII Art & Branding
# =========================================================================

from hermes_cli import __version__ as VERSION, __release_date__ as RELEASE_DATE

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

HERMES_CODE_LOGO = """[bold #00E5FF]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗     ██████╗ ██████╗ ██████╗ ███████╗[/]
[bold #56B6C2]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝    ██╔════╝██╔═══██╗██╔══██╗██╔════╝[/]
[#2DD4BF]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗    ██║     ██║   ██║██║  ██║█████╗  [/]
[#2DD4BF]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║    ██║     ██║   ██║██║  ██║██╔══╝  [/]
[#00838F]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║    ╚██████╗╚██████╔╝██████╔╝███████╗[/]
[#546E7A]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝[/]"""

HERMES_CODE_CADUCEUS = """[#00838F]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#00838F]⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀[/]
[#56B6C2]⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀[/]
[#56B6C2]⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀[/]
[#00E5FF]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#00E5FF]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#56B6C2]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#56B6C2]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#00838F]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#00838F]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#546E7A]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#546E7A]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#546E7A]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#546E7A]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#546E7A]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]"""



# =========================================================================
# Skills scanning
# =========================================================================

def get_available_skills() -> Dict[str, List[str]]:
    """Return skills grouped by category, filtered by platform and disabled state.

    Delegates to ``_find_all_skills()`` from ``tools/skills_tool`` which already
    handles platform gating (``platforms:`` frontmatter) and respects the
    user's ``skills.disabled`` config list.
    """
    try:
        from tools.skills_tool import _find_all_skills
        all_skills = _find_all_skills()  # already filtered
    except Exception:
        return {}

    skills_by_category: Dict[str, List[str]] = {}
    for skill in all_skills:
        category = skill.get("category") or "general"
        skills_by_category.setdefault(category, []).append(skill["name"])
    return skills_by_category


# =========================================================================
# Update check
# =========================================================================

# Cache update check results for 6 hours to avoid repeated git fetches
_UPDATE_CHECK_CACHE_SECONDS = 6 * 3600


def check_for_updates() -> Optional[int]:
    """Check how many commits behind origin/main the local repo is.

    Does a ``git fetch`` at most once every 6 hours (cached to
    ``~/.hermes/.update_check``).  Returns the number of commits behind,
    or ``None`` if the check fails or isn't applicable.
    """
    hermes_home = get_hermes_home()
    repo_dir = hermes_home / "hermes-agent"
    cache_file = hermes_home / ".update_check"

    # Must be a git repo — fall back to project root for dev installs
    if not (repo_dir / ".git").exists():
        repo_dir = Path(__file__).parent.parent.resolve()
    if not (repo_dir / ".git").exists():
        return None

    # Read cache
    now = time.time()
    try:
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if now - cached.get("ts", 0) < _UPDATE_CHECK_CACHE_SECONDS:
                return cached.get("behind")
    except Exception:
        pass

    # Fetch latest refs (fast — only downloads ref metadata, no files)
    try:
        subprocess.run(
            ["git", "fetch", "origin", "--quiet"],
            capture_output=True, timeout=10,
            cwd=str(repo_dir),
        )
    except Exception:
        pass  # Offline or timeout — use stale refs, that's fine

    # Count commits behind
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True, text=True, timeout=5,
            cwd=str(repo_dir),
        )
        if result.returncode == 0:
            behind = int(result.stdout.strip())
        else:
            behind = None
    except Exception:
        behind = None

    # Write cache
    try:
        cache_file.write_text(json.dumps({"ts": now, "behind": behind}))
    except Exception:
        pass

    return behind


def _resolve_repo_dir() -> Optional[Path]:
    """Return the active Hermes git checkout, or None if this isn't a git install."""
    hermes_home = get_hermes_home()
    repo_dir = hermes_home / "hermes-agent"
    if not (repo_dir / ".git").exists():
        repo_dir = Path(__file__).parent.parent.resolve()
    return repo_dir if (repo_dir / ".git").exists() else None


def _git_short_hash(repo_dir: Path, rev: str) -> Optional[str]:
    """Resolve a git revision to an 8-character short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", rev],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(repo_dir),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def get_git_banner_state(repo_dir: Optional[Path] = None) -> Optional[dict]:
    """Return upstream/local git hashes for the startup banner."""
    repo_dir = repo_dir or _resolve_repo_dir()
    if repo_dir is None:
        return None

    upstream = _git_short_hash(repo_dir, "origin/main")
    local = _git_short_hash(repo_dir, "HEAD")
    if not upstream or not local:
        return None

    ahead = 0
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "origin/main..HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(repo_dir),
        )
        if result.returncode == 0:
            ahead = int((result.stdout or "0").strip() or "0")
    except Exception:
        ahead = 0

    return {"upstream": upstream, "local": local, "ahead": max(ahead, 0)}


def format_banner_version_label() -> str:
    """Return the version label shown in the startup banner title."""
    base = f"Hermes Agent v{VERSION} ({RELEASE_DATE})"
    state = get_git_banner_state()
    if not state:
        return base

    upstream = state["upstream"]
    local = state["local"]
    ahead = int(state.get("ahead") or 0)

    if ahead <= 0 or upstream == local:
        return f"{base} · upstream {upstream}"

    carried_word = "commit" if ahead == 1 else "commits"
    return f"{base} · upstream {upstream} · local {local} (+{ahead} carried {carried_word})"


# =========================================================================
# Non-blocking update check
# =========================================================================

_update_result: Optional[int] = None
_update_check_done = threading.Event()


def prefetch_update_check():
    """Kick off update check in a background daemon thread."""
    def _run():
        global _update_result
        _update_result = check_for_updates()
        _update_check_done.set()
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def get_update_result(timeout: float = 0.5) -> Optional[int]:
    """Get result of prefetched check. Returns None if not ready."""
    _update_check_done.wait(timeout=timeout)
    return _update_result


# =========================================================================
# Welcome banner
# =========================================================================

def _format_context_length(tokens: int) -> str:
    """Format a token count for display (e.g. 128000 → '128K', 1048576 → '1M')."""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        rounded = round(val)
        if abs(val - rounded) < 0.05:
            return f"{rounded}M"
        return f"{val:.1f}M"
    elif tokens >= 1_000:
        val = tokens / 1_000
        rounded = round(val)
        if abs(val - rounded) < 0.05:
            return f"{rounded}K"
        return f"{val:.1f}K"
    return str(tokens)


def _display_toolset_name(toolset_name: str) -> str:
    """Normalize internal/legacy toolset identifiers for banner display."""
    if not toolset_name:
        return "unknown"
    return (
        toolset_name[:-6]
        if toolset_name.endswith("_tools")
        else toolset_name
    )


def build_welcome_banner(console: Console, model: str, cwd: str,
                         tools: List[dict] = None,
                         enabled_toolsets: List[str] = None,
                         session_id: str = None,
                         get_toolset_for_tool=None,
                         context_length: int = None):
    """Build and print a welcome banner with caduceus on left and info on right.

    Args:
        console: Rich Console instance.
        model: Current model name.
        cwd: Current working directory.
        tools: List of tool definitions.
        enabled_toolsets: List of enabled toolset names.
        session_id: Session identifier.
        get_toolset_for_tool: Callable to map tool name -> toolset name.
        context_length: Model's context window size in tokens.
    """
    from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
    if get_toolset_for_tool is None:
        from model_tools import get_toolset_for_tool

    tools = tools or []
    enabled_toolsets = enabled_toolsets or []

    _, unavailable_toolsets = check_tool_availability(quiet=True)
    disabled_tools = set()
    # Tools whose toolset has a check_fn are lazy-initialized (e.g. honcho,
    # homeassistant) — they show as unavailable at banner time because the
    # check hasn't run yet, but they aren't misconfigured.
    lazy_tools = set()
    for item in unavailable_toolsets:
        toolset_name = item.get("name", "")
        ts_req = TOOLSET_REQUIREMENTS.get(toolset_name, {})
        tools_in_ts = item.get("tools", [])
        if ts_req.get("check_fn"):
            lazy_tools.update(tools_in_ts)
        else:
            disabled_tools.update(tools_in_ts)

    layout_table = Table.grid(padding=(0, 2))
    layout_table.add_column("left", justify="center")
    layout_table.add_column("right", justify="left")

    # Resolve skin colors once for the entire banner
    accent = _skin_color("banner_accent", "#FFBF00")
    dim = _skin_color("banner_dim", "#B8860B")
    text = _skin_color("banner_text", "#FFF8DC")
    session_color = _skin_color("session_border", "#8B8682")

    # Use skin's custom caduceus art if provided
    try:
        from hermes_cli.skin_engine import get_active_skin
        _bskin = get_active_skin()
        _hero = _bskin.banner_hero if hasattr(_bskin, 'banner_hero') and _bskin.banner_hero else HERMES_CADUCEUS
    except Exception:
        _bskin = None
        _hero = HERMES_CADUCEUS
    left_lines = ["", _hero, ""]
    model_short = model.split("/")[-1] if "/" in model else model
    if model_short.endswith(".gguf"):
        model_short = model_short[:-5]
    if len(model_short) > 28:
        model_short = model_short[:25] + "..."
    ctx_str = f" [dim {dim}]·[/] [dim {dim}]{_format_context_length(context_length)} context[/]" if context_length else ""
    left_lines.append(f"[{accent}]{model_short}[/]{ctx_str} [dim {dim}]·[/] [dim {dim}]Nous Research[/]")
    left_lines.append(f"[dim {dim}]{cwd}[/]")
    if session_id:
        left_lines.append(f"[dim {session_color}]Session: {session_id}[/]")
    left_content = "\n".join(left_lines)

    right_lines = [f"[bold {accent}]Available Tools[/]"]
    toolsets_dict: Dict[str, list] = {}

    for tool in tools:
        tool_name = tool["function"]["name"]
        toolset = _display_toolset_name(get_toolset_for_tool(tool_name) or "other")
        toolsets_dict.setdefault(toolset, []).append(tool_name)

    for item in unavailable_toolsets:
        toolset_id = item.get("id", item.get("name", "unknown"))
        display_name = _display_toolset_name(toolset_id)
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
            elif name in lazy_tools:
                colored_names.append(f"[yellow]{name}[/]")
            else:
                colored_names.append(f"[{text}]{name}[/]")

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
                elif name in lazy_tools:
                    colored_names.append(f"[yellow]{name}[/]")
                else:
                    colored_names.append(f"[{text}]{name}[/]")
            tools_str = ", ".join(colored_names)

        right_lines.append(f"[dim {dim}]{toolset}:[/] {tools_str}")

    if remaining_toolsets > 0:
        right_lines.append(f"[dim {dim}](and {remaining_toolsets} more toolsets...)[/]")

    # MCP Servers section (only if configured)
    try:
        from tools.mcp_tool import get_mcp_status
        mcp_status = get_mcp_status()
    except Exception:
        mcp_status = []

    if mcp_status:
        right_lines.append("")
        right_lines.append(f"[bold {accent}]MCP Servers[/]")
        for srv in mcp_status:
            if srv["connected"]:
                right_lines.append(
                    f"[dim {dim}]{srv['name']}[/] [{text}]({srv['transport']})[/] "
                    f"[dim {dim}]—[/] [{text}]{srv['tools']} tool(s)[/]"
                )
            else:
                right_lines.append(
                    f"[red]{srv['name']}[/] [dim]({srv['transport']})[/] "
                    f"[red]— failed[/]"
                )

    right_lines.append("")
    right_lines.append(f"[bold {accent}]Available Skills[/]")
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
            right_lines.append(f"[dim {dim}]{category}:[/] [{text}]{skills_str}[/]")
    else:
        right_lines.append(f"[dim {dim}]No skills installed[/]")

    right_lines.append("")
    mcp_connected = sum(1 for s in mcp_status if s["connected"]) if mcp_status else 0
    summary_parts = [f"{len(tools)} tools", f"{total_skills} skills"]
    if mcp_connected:
        summary_parts.append(f"{mcp_connected} MCP servers")
    summary_parts.append("/help for commands")
    # Show active profile name when not 'default'
    try:
        from hermes_cli.profiles import get_active_profile_name
        _profile_name = get_active_profile_name()
        if _profile_name and _profile_name != "default":
            right_lines.append(f"[bold {accent}]Profile:[/] [{text}]{_profile_name}[/]")
    except Exception:
        pass  # Never break the banner over a profiles.py bug

    right_lines.append(f"[dim {dim}]{' · '.join(summary_parts)}[/]")

    # Update check — use prefetched result if available
    try:
        behind = get_update_result(timeout=0.5)
        if behind and behind > 0:
            from hermes_cli.config import recommended_update_command
            commits_word = "commit" if behind == 1 else "commits"
            right_lines.append(
                f"[bold yellow]⚠ {behind} {commits_word} behind[/]"
                f"[dim yellow] — run [bold]{recommended_update_command()}[/bold] to update[/]"
            )
    except Exception:
        pass  # Never break the banner over an update check

    right_content = "\n".join(right_lines)
    layout_table.add_row(left_content, right_content)

    agent_name = _skin_branding("agent_name", "Hermes Agent")
    title_color = _skin_color("banner_title", "#FFD700")
    border_color = _skin_color("banner_border", "#CD7F32")
    outer_panel = Panel(
        layout_table,
        title=f"[bold {title_color}]{format_banner_version_label()}[/]",
        border_style=border_color,
        padding=(0, 2),
    )

    console.print()
    term_width = getattr(console, "width", None) or shutil.get_terminal_size().columns
    if term_width >= 95:
        _logo = _bskin.banner_logo if _bskin and hasattr(_bskin, 'banner_logo') and _bskin.banner_logo else HERMES_AGENT_LOGO
        console.print(_logo)
        console.print()
    console.print(outer_panel)


# =========================================================================
# Hermes Code Console — new banner for Code Mode
# =========================================================================

CODE_MODE_SKILLS = [
    "fix_build",
    "review_diff",
    "stabilize_hanging_task",
    "fix_runtime_error",
    "implement_feature",
    "refactor_react_page",
    "benchmark_provider",
]
CODE_MODE_SUMMARY_SKILLS = ["fix_build", "review_diff", "implement_feature"]


def _first_list(payload, key: str) -> list:
    """Return a list from either a bare list response or a JSON envelope."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        value = payload.get("data")
        if isinstance(value, list):
            return value
    return []


def _format_code_version_label() -> str:
    """Return the real Hermes version label rebranded for Code Mode."""
    label = format_banner_version_label()
    if label.startswith("Hermes Agent "):
        return label.replace("Hermes Agent ", "Hermes Agent Code ", 1)
    if label.startswith("Hermes "):
        return label.replace("Hermes ", "Hermes Code ", 1)
    return f"Hermes Agent Code {label}"


def _shorten_middle(value: str, max_len: int) -> str:
    value = str(value or "")
    if len(value) <= max_len:
        return value
    if max_len <= 4:
        return value[:max_len]
    return "..." + value[-(max_len - 3):]


def _infer_provider_from_model(model: str) -> str:
    if not model:
        return "unknown"
    if "/" in model:
        return model.split("/", 1)[0]
    if ":" in model:
        return model.split(":", 1)[0]
    return "local"


def _schema_label(value) -> Optional[str]:
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    return text_value if text_value.startswith("v") else f"v{text_value}"


def _schema_number(label: str) -> Optional[int]:
    try:
        return int(str(label).strip().lstrip("vV"))
    except Exception:
        return None


def _local_state_db_schema_label() -> str:
    try:
        import sqlite3
        db_path = get_hermes_home() / "state.db"
        if db_path.exists():
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.2)
            try:
                row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            finally:
                conn.close()
            label = _schema_label(row[0] if row else None)
            if label:
                return label
    except Exception:
        pass

    try:
        from hermes_state import SCHEMA_VERSION
        return f"v{SCHEMA_VERSION}"
    except Exception:
        return "unknown"


def _get_code_mode_data():
    """Collect real-time data for Code Mode console with fast safe fallbacks."""
    schema = _local_state_db_schema_label()

    workspace = os.getenv("TERMINAL_CWD", os.getcwd())
    data = {
        "provider": "unknown",
        "model": "unknown",
        "profile": "default",
        "workspace": workspace,
        "branch": "not a git repo",
        "session_id": "",
        "backend_status": "offline",
        "backend_port": "9119",
        "web_url": "http://localhost:3001/code",
        "db_schema": schema,
        "tools_available": 0,
        "skills_available": 0,
        "code_skills": CODE_MODE_SUMMARY_SKILLS,
        "active_sessions": 0,
        "pending_approvals": 0,
    }

    try:
        from hermes_cli.profiles import get_active_profile_name
        data["profile"] = get_active_profile_name() or "default"
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=workspace,
        )
        if result.returncode == 0 and (result.stdout or "").strip():
            data["branch"] = result.stdout.strip()
    except Exception:
        pass

    try:
        import requests
        status_response = requests.get("http://localhost:9119/api/status", timeout=0.8)
        if status_response.status_code == 200:
            data["backend_status"] = "online"
            payload = status_response.json() or {}
            if isinstance(payload, dict):
                port = payload.get("port") or payload.get("backend_port")
                if port:
                    data["backend_port"] = str(port)
                schema_value = (
                    payload.get("schema_version")
                    or payload.get("db_schema")
                    or payload.get("database_schema")
                    or payload.get("config_version")
                    or payload.get("latest_config_version")
                )
                schema_label = _schema_label(schema_value)
                if schema_label:
                    current = _schema_number(data["db_schema"])
                    candidate = _schema_number(schema_label)
                    if current is None or candidate is None or candidate >= current:
                        data["db_schema"] = schema_label
    except Exception:
        pass

    try:
        import requests
        sessions_response = requests.get("http://localhost:9119/api/code/sessions", timeout=0.8)
        if sessions_response.status_code == 200:
            sessions = _first_list(sessions_response.json() or [], "sessions")
            data["active_sessions"] = len([
                item for item in sessions
                if isinstance(item, dict) and item.get("status") == "active"
            ])
    except Exception:
        pass

    try:
        import requests
        approvals_response = requests.get("http://localhost:9119/api/approvals", timeout=0.8)
        if approvals_response.status_code == 200:
            approvals = _first_list(approvals_response.json() or [], "approvals")
            data["pending_approvals"] = len([
                item for item in approvals
                if isinstance(item, dict) and item.get("status") == "pending"
            ])
    except Exception:
        pass

    return data


def build_hermes_code_console(
    console: Console,
    model: str = None,
    provider: str = None,
    profile: str = None,
    session_id: str = None,
    tools_count: int = None,
    skills_count: int = None,
    context_length: int = None,
):
    """Build and print the Hermes Code home screen in the classic banner style."""
    data = _get_code_mode_data()

    if model:
        data["model"] = model
    if provider:
        data["provider"] = provider
    elif data.get("provider") in ("", "unknown"):
        data["provider"] = _infer_provider_from_model(data.get("model", ""))
    if profile:
        data["profile"] = profile
    if session_id:
        data["session_id"] = session_id
    if tools_count is not None:
        data["tools_available"] = tools_count
    if skills_count is not None:
        data["skills_available"] = skills_count

    title = _skin_color("code_title", "#00E5FF")
    title_dim = _skin_color("code_title_dim", "#56B6C2")
    accent = _skin_color("code_accent", "#2DD4BF")
    accent_dim = _skin_color("code_accent_dim", "#00838F")
    border = _skin_color("code_border", "#30363D")
    text = _skin_color("code_text", "#E6EDF3")
    muted = _skin_color("code_muted", "#8B949E")
    success = _skin_color("code_success", "#7EE787")
    warning = _skin_color("code_warning", "#D29922")
    error = _skin_color("code_error", "#F85149")

    term_width = getattr(console, "width", None) or shutil.get_terminal_size().columns
    version_label = _format_code_version_label()
    model_display = data["model"].split("/", 1)[-1] if "/" in data["model"] else data["model"]
    backend_color = success if data["backend_status"] == "online" else warning
    backend_display = f"{data['backend_status']} :{data['backend_port']}"
    code_skills = data.get("code_skills") or CODE_MODE_SUMMARY_SKILLS
    code_skills_line = " · ".join(code_skills[:3])

    if term_width < 100:
        workspace = _shorten_middle(data["workspace"], max(28, term_width - 18))
        console.print()
        console.print(f"[bold {title}]HERMES CODE[/] [dim {muted}]· AI Development Console[/]")
        console.print(f"[{text}]{version_label}[/]")
        console.print(f"[dim {muted}]Hermes Code Mode Status[/]")
        console.print(
            f"[{text}]Provider: {data['provider']} · {model_display} | "
            f"Profile: {data['profile']}[/]"
        )
        console.print(f"[{text}]Workspace: {workspace}[/]")
        console.print(f"[{text}]Branch: {data['branch']}[/]")
        console.print(
            f"[{text}]Backend: {backend_display} | Web: {data['web_url']} | "
            f"DB: {data['db_schema']}[/]"
        )
        console.print(f"[{text}]Web Cockpit: {data['web_url']}[/]")
        console.print(f"[bold {accent}]Quick Actions[/]")
        console.print(f"[{accent}]Quick: /code /web /workspace /session /approvals /skills-code /help[/]")
        console.print(
            f"[{text}]Tools: {data['tools_available']} — /tools | "
            f"Skills: {data['skills_available']} — /skills[/]"
        )
        console.print()
        return

    status_lines = [
        f"[bold {accent}]Code Mode Status[/]",
        f"[dim {muted}]Provider[/]   [{text}]{data['provider']}[/]",
        f"[dim {muted}]Model[/]      [{text}]{model_display}[/]",
        f"[dim {muted}]Profile[/]    [{text}]{data['profile']}[/]",
        f"[dim {muted}]Workspace[/]  [{text}]{_shorten_middle(data['workspace'], 52)}[/]",
        f"[dim {muted}]Branch[/]     [{text}]{data['branch']}[/]",
        f"[dim {muted}]Session[/]    [{text}]{data.get('session_id') or 'new'}[/]",
        f"[dim {muted}]Backend[/]    [{backend_color}]{backend_display}[/]",
        f"[dim {muted}]Web Cockpit[/] [{text}]{data['web_url']}[/]",
        f"[dim {muted}]DB[/]         [{text}]{data['db_schema']}[/]",
    ]
    if context_length:
        status_lines.append(f"[dim {muted}]Context[/]    [{text}]{_format_context_length(context_length)}[/]")
    if data["active_sessions"] or data["pending_approvals"]:
        approvals = (
            f" · [{warning}]{data['pending_approvals']} approvals[/]"
            if data["pending_approvals"]
            else ""
        )
        status_lines.append(
            f"[dim {muted}]Activity[/]   [{text}]{data['active_sessions']} sessions[/]{approvals}"
        )

    quick_lines = [
        f"[bold {accent}]Quick Actions[/]",
        f"[{title_dim}]/code[/]         [{muted}]Code Mode help[/]",
        f"[{title_dim}]/web[/]          [{muted}]URLs and logs[/]",
        f"[{title_dim}]/workspace[/]    [{muted}]Workspace info[/]",
        f"[{title_dim}]/session[/]      [{muted}]Code sessions[/]",
        f"[{title_dim}]/approvals[/]    [{muted}]Pending approvals[/]",
        f"[{title_dim}]/skills-code[/]  [{muted}]Coding skills[/]",
        f"[{title_dim}]/help[/]         [{muted}]All commands[/]",
    ]

    summary_lines = [
        f"[{text}]Tools: {data['tools_available']} available[/] [dim {muted}]— /tools[/]",
        f"[{text}]Skills: {data['skills_available']} available[/] [dim {muted}]— /skills[/]",
        f"[{text}]Code Skills: {code_skills_line}[/]",
    ]

    layout = Table.grid(padding=(0, 3))
    layout.add_column("symbol", justify="center", no_wrap=True)
    layout.add_column("status", justify="left")
    layout.add_row(HERMES_CODE_CADUCEUS, "\n".join(status_lines))

    panel_content = Group(
        layout,
        f"[{accent_dim}]{'─' * 78}[/]",
        "\n".join(quick_lines),
        f"[{accent_dim}]{'─' * 78}[/]",
        "\n".join(summary_lines),
    )
    outer_panel = Panel(
        panel_content,
        title=f"[bold {title}]Code Mode Status[/]",
        border_style=border,
        padding=(0, 2),
    )

    console.print()
    console.print(HERMES_CODE_LOGO)
    console.print(f"[bold {title}]HERMES CODE[/] [dim {muted}]· AI Development Console[/]")
    console.print(f"[dim {muted}]Hermes Code Mode[/]")
    console.print(f"[{text}]{version_label}[/] [dim {muted}]· AI Development Console[/]")
    console.print()
    console.print(outer_panel)
    console.print(f"[{text}]Welcome to Hermes Agent Code. Type your message or /help for commands.[/]")
