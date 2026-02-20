#!/usr/bin/env python3
"""
Hermes Agent CLI - Interactive Terminal Interface

A beautiful command-line interface for the Hermes Agent, inspired by Claude Code.
Features ASCII art branding, interactive REPL, toolset selection, and rich formatting.

Usage:
    python cli.py                          # Start interactive mode with all tools
    python cli.py --toolsets web,terminal  # Start with specific toolsets
    python cli.py -q "your question"       # Single query mode
    python cli.py --list-tools             # List available tools and exit
"""

import os
import sys
import json
import atexit
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Suppress startup messages for clean CLI experience
os.environ["MSWEA_SILENT_STARTUP"] = "1"  # mini-swe-agent
os.environ["HERMES_QUIET"] = "1"  # Our own modules

import yaml

# prompt_toolkit for fixed input area TUI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl, ConditionalContainer
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.keys import Keys
from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
import threading
import queue
import tempfile


# Load environment variables first
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# =============================================================================
# Configuration Loading
# =============================================================================

def load_cli_config() -> Dict[str, Any]:
    """
    Load CLI configuration from config files.
    
    Config lookup order:
    1. ~/.hermes/config.yaml (user config - preferred)
    2. ./cli-config.yaml (project config - fallback)
    
    Environment variables take precedence over config file values.
    Returns default values if no config file exists.
    """
    # Check user config first (~/.hermes/config.yaml)
    user_config_path = Path.home() / '.hermes' / 'config.yaml'
    project_config_path = Path(__file__).parent / 'cli-config.yaml'
    
    # Use user config if it exists, otherwise project config
    if user_config_path.exists():
        config_path = user_config_path
    else:
        config_path = project_config_path
    
    # Also load .env from ~/.hermes/.env if it exists
    user_env_path = Path.home() / '.hermes' / '.env'
    if user_env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=user_env_path, override=True)
    
    # Default configuration
    defaults = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "terminal": {
            "env_type": "local",
            "cwd": ".",  # "." is resolved to os.getcwd() at runtime
            "timeout": 60,
            "lifetime_seconds": 300,
            "docker_image": "python:3.11",
            "singularity_image": "docker://python:3.11",
            "modal_image": "python:3.11",
        },
        "browser": {
            "inactivity_timeout": 120,  # Auto-cleanup inactive browser sessions after 2 min
        },
        "compression": {
            "enabled": True,      # Auto-compress when approaching context limit
            "threshold": 0.85,    # Compress at 85% of model's context limit
            "summary_model": "google/gemini-3-flash-preview",  # Fast/cheap model for summaries
        },
        "agent": {
            "max_turns": 60,  # Default max tool-calling iterations
            "verbose": False,
            "system_prompt": "",
            "personalities": {
                "helpful": "You are a helpful, friendly AI assistant.",
                "concise": "You are a concise assistant. Keep responses brief and to the point.",
                "technical": "You are a technical expert. Provide detailed, accurate technical information.",
                "creative": "You are a creative assistant. Think outside the box and offer innovative solutions.",
                "teacher": "You are a patient teacher. Explain concepts clearly with examples.",
                "kawaii": "You are a kawaii assistant! Use cute expressions like (◕‿◕), ★, ♪, and ~! Add sparkles and be super enthusiastic about everything! Every response should feel warm and adorable desu~! ヽ(>∀<☆)ノ",
                "catgirl": "You are Neko-chan, an anime catgirl AI assistant, nya~! Add 'nya' and cat-like expressions to your speech. Use kaomoji like (=^･ω･^=) and ฅ^•ﻌ•^ฅ. Be playful and curious like a cat, nya~!",
                "pirate": "Arrr! Ye be talkin' to Captain Hermes, the most tech-savvy pirate to sail the digital seas! Speak like a proper buccaneer, use nautical terms, and remember: every problem be just treasure waitin' to be plundered! Yo ho ho!",
                "shakespeare": "Hark! Thou speakest with an assistant most versed in the bardic arts. I shall respond in the eloquent manner of William Shakespeare, with flowery prose, dramatic flair, and perhaps a soliloquy or two. What light through yonder terminal breaks?",
                "surfer": "Duuude! You're chatting with the chillest AI on the web, bro! Everything's gonna be totally rad. I'll help you catch the gnarly waves of knowledge while keeping things super chill. Cowabunga!",
                "noir": "The rain hammered against the terminal like regrets on a guilty conscience. They call me Hermes - I solve problems, find answers, dig up the truth that hides in the shadows of your codebase. In this city of silicon and secrets, everyone's got something to hide. What's your story, pal?",
                "uwu": "hewwo! i'm your fwiendwy assistant uwu~ i wiww twy my best to hewp you! *nuzzles your code* OwO what's this? wet me take a wook! i pwomise to be vewy hewpful >w<",
                "philosopher": "Greetings, seeker of wisdom. I am an assistant who contemplates the deeper meaning behind every query. Let us examine not just the 'how' but the 'why' of your questions. Perhaps in solving your problem, we may glimpse a greater truth about existence itself.",
                "hype": "YOOO LET'S GOOOO!!! I am SO PUMPED to help you today! Every question is AMAZING and we're gonna CRUSH IT together! This is gonna be LEGENDARY! ARE YOU READY?! LET'S DO THIS!",
            },
        },
        "toolsets": ["all"],
        "display": {
            "compact": False,
        },
        "clarify": {
            "timeout": 120,  # Seconds to wait for a clarify answer before auto-proceeding
        },
    }
    
    # Track whether the config file explicitly set terminal config.
    # When using defaults (no config file / no terminal section), we should NOT
    # overwrite env vars that were already set by .env -- only a user's config
    # file should be authoritative.
    _file_has_terminal_config = False

    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            
            _file_has_terminal_config = "terminal" in file_config

            # Handle model config - can be string (new format) or dict (old format)
            if "model" in file_config:
                if isinstance(file_config["model"], str):
                    # New format: model is just a string, convert to dict structure
                    defaults["model"]["default"] = file_config["model"]
                elif isinstance(file_config["model"], dict):
                    # Old format: model is a dict with default/base_url
                    defaults["model"].update(file_config["model"])
            
            # Deep merge other keys with defaults
            for key in defaults:
                if key == "model":
                    continue  # Already handled above
                if key in file_config:
                    if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
                        defaults[key].update(file_config[key])
                    else:
                        defaults[key] = file_config[key]
            
            # Handle root-level max_turns (backwards compat) - copy to agent.max_turns
            if "max_turns" in file_config and "agent" not in file_config:
                defaults["agent"]["max_turns"] = file_config["max_turns"]
        except Exception as e:
            print(f"[Warning] Failed to load cli-config.yaml: {e}")
    
    # Apply terminal config to environment variables (so terminal_tool picks them up)
    terminal_config = defaults.get("terminal", {})
    
    # Normalize config key: the new config system (hermes_cli/config.py) and all
    # documentation use "backend", the legacy cli-config.yaml uses "env_type".
    # Accept both, with "backend" taking precedence (it's the documented key).
    if "backend" in terminal_config:
        terminal_config["env_type"] = terminal_config["backend"]
    
    # Handle special cwd values: "." or "auto" means use current working directory.
    # Only resolve to the host's CWD for the local backend where the host
    # filesystem is directly accessible.  For ALL remote/container backends
    # (ssh, docker, modal, singularity), the host path doesn't exist on the
    # target -- remove the key so terminal_tool.py uses its per-backend default.
    if terminal_config.get("cwd") in (".", "auto", "cwd"):
        effective_backend = terminal_config.get("env_type", "local")
        if effective_backend == "local":
            terminal_config["cwd"] = os.getcwd()
            defaults["terminal"]["cwd"] = terminal_config["cwd"]
        else:
            # Remove so TERMINAL_CWD stays unset → tool picks backend default
            terminal_config.pop("cwd", None)
    
    env_mappings = {
        "env_type": "TERMINAL_ENV",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
        "docker_image": "TERMINAL_DOCKER_IMAGE",
        "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
        "modal_image": "TERMINAL_MODAL_IMAGE",
        # SSH config
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        # Sudo support (works with all backends)
        "sudo_password": "SUDO_PASSWORD",
    }
    
    # Apply config values to env vars so terminal_tool picks them up.
    # If the config file explicitly has a [terminal] section, those values are
    # authoritative and override any .env settings.  When using defaults only
    # (no config file or no terminal section), don't overwrite env vars that
    # were already set by .env -- the user's .env is the fallback source.
    for config_key, env_var in env_mappings.items():
        if config_key in terminal_config:
            if _file_has_terminal_config or env_var not in os.environ:
                os.environ[env_var] = str(terminal_config[config_key])
    
    # Apply browser config to environment variables
    browser_config = defaults.get("browser", {})
    browser_env_mappings = {
        "inactivity_timeout": "BROWSER_INACTIVITY_TIMEOUT",
    }
    
    for config_key, env_var in browser_env_mappings.items():
        if config_key in browser_config:
            os.environ[env_var] = str(browser_config[config_key])
    
    # Apply compression config to environment variables
    compression_config = defaults.get("compression", {})
    compression_env_mappings = {
        "enabled": "CONTEXT_COMPRESSION_ENABLED",
        "threshold": "CONTEXT_COMPRESSION_THRESHOLD",
        "summary_model": "CONTEXT_COMPRESSION_MODEL",
    }
    
    for config_key, env_var in compression_env_mappings.items():
        if config_key in compression_config:
            os.environ[env_var] = str(compression_config[config_key])
    
    return defaults

# Load configuration at module startup
CLI_CONFIG = load_cli_config()

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.columns import Columns
from rich.align import Align
from rich import box

import fire

# Import the agent and tool systems
from run_agent import AIAgent
from model_tools import get_tool_definitions, get_all_tool_names, get_toolset_for_tool, get_available_toolsets
from toolsets import get_all_toolsets, get_toolset_info, resolve_toolset, validate_toolset

# Cron job system for scheduled tasks
from cron import create_job, list_jobs, remove_job, get_job, run_daemon as run_cron_daemon, tick as cron_tick

# Resource cleanup imports for safe shutdown (terminal VMs, browser sessions)
from tools.terminal_tool import cleanup_all_environments as _cleanup_all_terminals
from tools.browser_tool import _emergency_cleanup_all_sessions as _cleanup_all_browsers

# Guard to prevent cleanup from running multiple times on exit
_cleanup_done = False

def _run_cleanup():
    """Run resource cleanup exactly once."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    try:
        _cleanup_all_terminals()
    except Exception:
        pass
    try:
        _cleanup_all_browsers()
    except Exception:
        pass

# ============================================================================
# ASCII Art & Branding
# ============================================================================

# Color palette (hex colors for Rich markup):
# - Gold: #FFD700 (headers, highlights)
# - Amber: #FFBF00 (secondary highlights)
# - Bronze: #CD7F32 (tertiary elements)
# - Light: #FFF8DC (text)
# - Dim: #B8860B (muted text)

# ANSI building blocks for conversation display
_GOLD = "\033[1;33m"    # Bold yellow — closest universal match to the gold theme
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"

def _cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's native renderer.

    Raw ANSI escapes written via print() are swallowed by patch_stdout's
    StdoutProxy.  Routing through print_formatted_text(ANSI(...)) lets
    prompt_toolkit parse the escapes and render real colors.
    """
    _pt_print(_PT_ANSI(text))

# Version string
VERSION = "v1.0.0"

# ASCII Art - HERMES-AGENT logo (full width, single line - requires ~95 char terminal)
HERMES_AGENT_LOGO = """[bold #FFD700]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD700]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#FFBF00]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#FFBF00]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#CD7F32]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#CD7F32]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""

# ASCII Art - Hermes Caduceus (compact, fits in left panel)
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

# Compact banner for smaller terminals (fallback)
COMPACT_BANNER = """
[bold #FFD700]╔══════════════════════════════════════════════════════════════╗[/]
[bold #FFD700]║[/]  [#FFBF00]⚕ NOUS HERMES[/] [dim #B8860B]- AI Agent Framework[/]              [bold #FFD700]║[/]
[bold #FFD700]║[/]  [#CD7F32]Messenger of the Digital Gods[/]    [dim #B8860B]Nous Research[/]   [bold #FFD700]║[/]
[bold #FFD700]╚══════════════════════════════════════════════════════════════╝[/]
"""


def _get_available_skills() -> Dict[str, List[str]]:
    """
    Scan ~/.hermes/skills/ and return skills grouped by category.
    
    Returns:
        Dict mapping category name to list of skill names
    """
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


def build_welcome_banner(console: Console, model: str, cwd: str, tools: List[dict] = None, enabled_toolsets: List[str] = None, session_id: str = None):
    """
    Build and print a Claude Code-style welcome banner with caduceus on left and info on right.
    
    Args:
        console: Rich Console instance for printing
        model: The current model name (e.g., "anthropic/claude-opus-4")
        cwd: Current working directory
        tools: List of tool definitions
        enabled_toolsets: List of enabled toolset names
        session_id: Unique session identifier for logging
    """
    from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
    
    tools = tools or []
    enabled_toolsets = enabled_toolsets or []
    
    # Get unavailable tools info for coloring
    _, unavailable_toolsets = check_tool_availability(quiet=True)
    disabled_tools = set()
    for item in unavailable_toolsets:
        disabled_tools.update(item.get("tools", []))
    
    # Build the side-by-side content using a table for precise control
    layout_table = Table.grid(padding=(0, 2))
    layout_table.add_column("left", justify="center")
    layout_table.add_column("right", justify="left")
    
    # Build left content: caduceus + model info
    left_lines = ["", HERMES_CADUCEUS, ""]
    
    # Shorten model name for display
    model_short = model.split("/")[-1] if "/" in model else model
    if len(model_short) > 28:
        model_short = model_short[:25] + "..."
    
    left_lines.append(f"[#FFBF00]{model_short}[/] [dim #B8860B]·[/] [dim #B8860B]Nous Research[/]")
    left_lines.append(f"[dim #B8860B]{cwd}[/]")
    
    # Add session ID if provided
    if session_id:
        left_lines.append(f"[dim #8B8682]Session: {session_id}[/]")
    left_content = "\n".join(left_lines)
    
    # Build right content: tools list grouped by toolset
    right_lines = []
    right_lines.append("[bold #FFBF00]Available Tools[/]")
    
    # Group tools by toolset (include all possible tools, both enabled and disabled)
    toolsets_dict = {}
    
    # First, add all enabled tools
    for tool in tools:
        tool_name = tool["function"]["name"]
        toolset = get_toolset_for_tool(tool_name) or "other"
        if toolset not in toolsets_dict:
            toolsets_dict[toolset] = []
        toolsets_dict[toolset].append(tool_name)
    
    # Also add disabled toolsets so they show in the banner
    for item in unavailable_toolsets:
        # Map the internal toolset ID to display name
        toolset_id = item["id"]
        display_name = f"{toolset_id}_tools" if not toolset_id.endswith("_tools") else toolset_id
        if display_name not in toolsets_dict:
            toolsets_dict[display_name] = []
        for tool_name in item.get("tools", []):
            if tool_name not in toolsets_dict[display_name]:
                toolsets_dict[display_name].append(tool_name)
    
    # Display tools grouped by toolset (compact format, max 8 groups)
    sorted_toolsets = sorted(toolsets_dict.keys())
    display_toolsets = sorted_toolsets[:8]
    remaining_toolsets = len(sorted_toolsets) - 8
    
    for toolset in display_toolsets:
        tool_names = toolsets_dict[toolset]
        # Color each tool name - red if disabled, normal if enabled
        colored_names = []
        for name in sorted(tool_names):
            if name in disabled_tools:
                colored_names.append(f"[red]{name}[/]")
            else:
                colored_names.append(f"[#FFF8DC]{name}[/]")
        
        tools_str = ", ".join(colored_names)
        # Truncate if too long (accounting for markup)
        if len(", ".join(sorted(tool_names))) > 45:
            # Rebuild with truncation
            short_names = []
            length = 0
            for name in sorted(tool_names):
                if length + len(name) + 2 > 42:
                    short_names.append("...")
                    break
                short_names.append(name)
                length += len(name) + 2
            # Re-color the truncated list
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
    
    # Add skills section
    right_lines.append("[bold #FFBF00]Available Skills[/]")
    skills_by_category = _get_available_skills()
    total_skills = sum(len(s) for s in skills_by_category.values())
    
    if skills_by_category:
        for category in sorted(skills_by_category.keys()):
            skill_names = sorted(skills_by_category[category])
            # Show first 8 skills, then "..." if more
            if len(skill_names) > 8:
                display_names = skill_names[:8]
                skills_str = ", ".join(display_names) + f" +{len(skill_names) - 8} more"
            else:
                skills_str = ", ".join(skill_names)
            # Truncate if still too long
            if len(skills_str) > 50:
                skills_str = skills_str[:47] + "..."
            right_lines.append(f"[dim #B8860B]{category}:[/] [#FFF8DC]{skills_str}[/]")
    else:
        right_lines.append("[dim #B8860B]No skills installed[/]")
    
    right_lines.append("")
    right_lines.append(f"[dim #B8860B]{len(tools)} tools · {total_skills} skills · /help for commands[/]")
    
    right_content = "\n".join(right_lines)
    
    # Add to table
    layout_table.add_row(left_content, right_content)
    
    # Wrap in a panel with the title
    outer_panel = Panel(
        layout_table,
        title=f"[bold #FFD700]Hermes Agent {VERSION}[/]",
        border_style="#CD7F32",
        padding=(0, 2),
    )
    
    # Print the big HERMES-AGENT logo first (no panel wrapper for full width)
    console.print()
    console.print(HERMES_AGENT_LOGO)
    console.print()
    
    # Print the panel with caduceus and info
    console.print(outer_panel)


# ============================================================================
# CLI Commands
# ============================================================================

COMMANDS = {
    "/help": "Show this help message",
    "/tools": "List available tools",
    "/toolsets": "List available toolsets",
    "/model": "Show or change the current model",
    "/prompt": "View/set custom system prompt",
    "/personality": "Set a predefined personality",
    "/clear": "Clear screen and reset conversation (fresh start)",
    "/history": "Show conversation history",
    "/new": "Start a new conversation (reset history)",
    "/reset": "Reset conversation only (keep screen)",
    "/retry": "Retry the last message (resend to agent)",
    "/undo": "Remove the last user/assistant exchange",
    "/save": "Save the current conversation",
    "/config": "Show current configuration",
    "/cron": "Manage scheduled tasks (list, add, remove)",
    "/skills": "Search, install, inspect, or manage skills from online registries",
    "/platforms": "Show gateway/messaging platform status",
    "/quit": "Exit the CLI (also: /exit, /q)",
}


class SlashCommandCompleter(Completer):
    """Autocomplete for /commands in the input area."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # Only complete at the start of input, after /
        if not text.startswith("/"):
            return
        word = text[1:]  # strip the leading /
        for cmd, desc in COMMANDS.items():
            cmd_name = cmd[1:]  # strip leading / from key
            if cmd_name.startswith(word):
                yield Completion(
                    cmd_name,
                    start_position=-len(word),
                    display=cmd,
                    display_meta=desc,
                )


def save_config_value(key_path: str, value: any) -> bool:
    """
    Save a value to the active config file at the specified key path.
    
    Respects the same lookup order as load_cli_config():
    1. ~/.hermes/config.yaml (user config - preferred, used if it exists)
    2. ./cli-config.yaml (project config - fallback)
    
    Args:
        key_path: Dot-separated path like "agent.system_prompt"
        value: Value to save
    
    Returns:
        True if successful, False otherwise
    """
    # Use the same precedence as load_cli_config: user config first, then project config
    user_config_path = Path.home() / '.hermes' / 'config.yaml'
    project_config_path = Path(__file__).parent / 'cli-config.yaml'
    config_path = user_config_path if user_config_path.exists() else project_config_path
    
    try:
        # Ensure parent directory exists (for ~/.hermes/config.yaml on first use)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        # Navigate to the key and set value
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        
        # Save back
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
    except Exception as e:
        print(f"(x_x) Failed to save config: {e}")
        return False


# ============================================================================
# HermesCLI Class
# ============================================================================

class HermesCLI:
    """
    Interactive CLI for the Hermes Agent.
    
    Provides a REPL interface with rich formatting, command history,
    and tool execution capabilities.
    """
    
    def __init__(
        self,
        model: str = None,
        toolsets: List[str] = None,
        api_key: str = None,
        base_url: str = None,
        max_turns: int = 60,
        verbose: bool = False,
        compact: bool = False,
    ):
        """
        Initialize the Hermes CLI.
        
        Args:
            model: Model to use (default: from env or claude-sonnet)
            toolsets: List of toolsets to enable (default: all)
            api_key: API key (default: from environment)
            base_url: API base URL (default: OpenRouter)
            max_turns: Maximum tool-calling iterations (default: 60)
            verbose: Enable verbose logging
            compact: Use compact display mode
        """
        # Initialize Rich console
        self.console = Console()
        self.compact = compact if compact is not None else CLI_CONFIG["display"].get("compact", False)
        self.verbose = verbose if verbose is not None else CLI_CONFIG["agent"].get("verbose", False)
        
        # Configuration - priority: CLI args > env vars > config file
        # Model can come from: CLI arg, LLM_MODEL env, OPENAI_MODEL env (custom endpoint), or config
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or CLI_CONFIG["model"]["default"]
        
        # Base URL: custom endpoint (OPENAI_BASE_URL) takes precedence over OpenRouter
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", CLI_CONFIG["model"]["base_url"])
        
        # API key: custom endpoint (OPENAI_API_KEY) takes precedence over OpenRouter
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        # Max turns priority: CLI arg > env var > config file (agent.max_turns or root max_turns) > default
        if max_turns != 60:  # CLI arg was explicitly set
            self.max_turns = max_turns
        elif os.getenv("HERMES_MAX_ITERATIONS"):
            self.max_turns = int(os.getenv("HERMES_MAX_ITERATIONS"))
        elif CLI_CONFIG["agent"].get("max_turns"):
            self.max_turns = CLI_CONFIG["agent"]["max_turns"]
        elif CLI_CONFIG.get("max_turns"):  # Backwards compat: root-level max_turns
            self.max_turns = CLI_CONFIG["max_turns"]
        else:
            self.max_turns = 60
        
        # Parse and validate toolsets
        self.enabled_toolsets = toolsets
        if toolsets and "all" not in toolsets and "*" not in toolsets:
            # Validate each toolset
            invalid = [t for t in toolsets if not validate_toolset(t)]
            if invalid:
                self.console.print(f"[bold red]Warning: Unknown toolsets: {', '.join(invalid)}[/]")
        
        # System prompt and personalities from config
        self.system_prompt = CLI_CONFIG["agent"].get("system_prompt", "")
        self.personalities = CLI_CONFIG["agent"].get("personalities", {})
        
        # Agent will be initialized on first use
        self.agent: Optional[AIAgent] = None
        self._app = None  # prompt_toolkit Application (set in run())
        
        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        
        # Generate session ID with timestamp for display and logging
        # Format: YYYYMMDD_HHMMSS_shortUUID (e.g., 20260201_143052_a1b2c3)
        timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        self.session_id = f"{timestamp_str}_{short_uuid}"
        
        # History file for persistent input recall across sessions
        self._history_file = Path.home() / ".hermes_history"
    
    def _init_agent(self) -> bool:
        """
        Initialize the agent on first use.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.agent is not None:
            return True
        
        # Initialize SQLite session store for CLI sessions
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception:
            pass  # SQLite session store is optional
        
        try:
            self.agent = AIAgent(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                max_iterations=self.max_turns,
                enabled_toolsets=self.enabled_toolsets,
                verbose_logging=self.verbose,
                quiet_mode=True,  # Suppress verbose output for clean CLI
                ephemeral_system_prompt=self.system_prompt if self.system_prompt else None,
                session_id=self.session_id,  # Pass CLI's session ID to agent
                platform="cli",  # CLI interface — agent uses terminal-friendly formatting
                session_db=self._session_db,
                clarify_callback=self._clarify_callback,
            )
            return True
        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize agent: {e}[/]")
            return False
    
    def show_banner(self):
        """Display the welcome banner in Claude Code style."""
        self.console.clear()
        
        if self.compact:
            self.console.print(COMPACT_BANNER)
            self._show_status()
        else:
            # Get tools for display
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
            
            # Get terminal working directory (where commands will execute)
            cwd = os.getenv("TERMINAL_CWD", os.getcwd())
            
            # Build and display the banner
            build_welcome_banner(
                console=self.console,
                model=self.model,
                cwd=cwd,
                tools=tools,
                enabled_toolsets=self.enabled_toolsets,
                session_id=self.session_id,
            )
        
        # Show tool availability warnings if any tools are disabled
        self._show_tool_availability_warnings()
        
        self.console.print()
    
    def _show_tool_availability_warnings(self):
        """Show warnings about disabled tools due to missing API keys."""
        try:
            from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
            
            available, unavailable = check_tool_availability()
            
            # Filter to only those missing API keys (not system deps)
            api_key_missing = [u for u in unavailable if u["missing_vars"]]
            
            if api_key_missing:
                self.console.print()
                self.console.print("[yellow]⚠️  Some tools disabled (missing API keys):[/]")
                for item in api_key_missing:
                    tools_str = ", ".join(item["tools"][:2])  # Show first 2 tools
                    if len(item["tools"]) > 2:
                        tools_str += f", +{len(item['tools'])-2} more"
                    self.console.print(f"   [dim]• {item['name']}[/] [dim italic]({', '.join(item['missing_vars'])})[/]")
                self.console.print("[dim]   Run 'hermes setup' to configure[/]")
        except Exception:
            pass  # Don't crash on import errors
    
    def _show_status(self):
        """Show current status bar."""
        # Get tool count
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
        tool_count = len(tools) if tools else 0
        
        # Format model name (shorten if needed)
        model_short = self.model.split("/")[-1] if "/" in self.model else self.model
        if len(model_short) > 30:
            model_short = model_short[:27] + "..."
        
        # Get API status indicator
        if self.api_key:
            api_indicator = "[green bold]●[/]"
        else:
            api_indicator = "[red bold]●[/]"
        
        # Build status line with proper markup
        toolsets_info = ""
        if self.enabled_toolsets and "all" not in self.enabled_toolsets:
            toolsets_info = f" [dim #B8860B]·[/] [#CD7F32]toolsets: {', '.join(self.enabled_toolsets)}[/]"
        
        self.console.print(
            f"  {api_indicator} [#FFBF00]{model_short}[/] "
            f"[dim #B8860B]·[/] [bold cyan]{tool_count} tools[/]"
            f"{toolsets_info}"
        )
    
    def show_help(self):
        """Display help information with kawaii ASCII art."""
        print()
        print("+" + "-" * 50 + "+")
        print("|" + " " * 14 + "(^_^)? Available Commands" + " " * 10 + "|")
        print("+" + "-" * 50 + "+")
        print()
        
        for cmd, desc in COMMANDS.items():
            print(f"  {cmd:<15} - {desc}")
        
        print()
        print("  Tip: Just type your message to chat with Hermes!")
        print("  Multi-line: Alt+Enter for a new line")
        print()
    
    def show_tools(self):
        """Display available tools with kawaii ASCII art."""
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
        
        if not tools:
            print("(;_;) No tools available")
            return
        
        # Header
        print()
        print("+" + "-" * 78 + "+")
        print("|" + " " * 25 + "(^_^)/ Available Tools" + " " * 30 + "|")
        print("+" + "-" * 78 + "+")
        print()
        
        # Group tools by toolset
        toolsets = {}
        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            toolset = get_toolset_for_tool(name) or "unknown"
            if toolset not in toolsets:
                toolsets[toolset] = []
            desc = tool["function"].get("description", "")
            # Get first sentence or first 60 chars
            desc = desc.split(".")[0][:60]
            toolsets[toolset].append((name, desc))
        
        # Display by toolset
        for toolset in sorted(toolsets.keys()):
            print(f"  [{toolset}]")
            for name, desc in toolsets[toolset]:
                print(f"    * {name:<20} - {desc}")
            print()
        
        print(f"  Total: {len(tools)} tools  ヽ(^o^)ノ")
        print()
    
    def show_toolsets(self):
        """Display available toolsets with kawaii ASCII art."""
        all_toolsets = get_all_toolsets()
        
        # Header
        print()
        print("+" + "-" * 58 + "+")
        print("|" + " " * 15 + "(^_^)b Available Toolsets" + " " * 17 + "|")
        print("+" + "-" * 58 + "+")
        print()
        
        for name in sorted(all_toolsets.keys()):
            info = get_toolset_info(name)
            if info:
                tool_count = info["tool_count"]
                desc = info["description"][:45]
                
                # Mark if currently enabled
                marker = "(*)" if self.enabled_toolsets and name in self.enabled_toolsets else "   "
                print(f"  {marker} {name:<18} [{tool_count:>2} tools] - {desc}")
        
        print()
        print("  (*) = currently enabled")
        print()
        print("  Tip: Use 'all' or '*' to enable all toolsets")
        print("  Example: python cli.py --toolsets web,terminal")
        print()
    
    def show_config(self):
        """Display current configuration with kawaii ASCII art."""
        # Get terminal config from environment (which was set from cli-config.yaml)
        terminal_env = os.getenv("TERMINAL_ENV", "local")
        terminal_cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        terminal_timeout = os.getenv("TERMINAL_TIMEOUT", "60")
        
        config_path = Path(__file__).parent / 'cli-config.yaml'
        config_status = "(loaded)" if config_path.exists() else "(not found)"
        
        api_key_display = '********' + self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else 'Not set!'
        
        print()
        print("+" + "-" * 50 + "+")
        print("|" + " " * 15 + "(^_^) Configuration" + " " * 15 + "|")
        print("+" + "-" * 50 + "+")
        print()
        print("  -- Model --")
        print(f"  Model:     {self.model}")
        print(f"  Base URL:  {self.base_url}")
        print(f"  API Key:   {api_key_display}")
        print()
        print("  -- Terminal --")
        print(f"  Environment:  {terminal_env}")
        if terminal_env == "ssh":
            ssh_host = os.getenv("TERMINAL_SSH_HOST", "not set")
            ssh_user = os.getenv("TERMINAL_SSH_USER", "not set")
            ssh_port = os.getenv("TERMINAL_SSH_PORT", "22")
            print(f"  SSH Target:   {ssh_user}@{ssh_host}:{ssh_port}")
        print(f"  Working Dir:  {terminal_cwd}")
        print(f"  Timeout:      {terminal_timeout}s")
        print()
        print("  -- Agent --")
        print(f"  Max Turns:  {self.max_turns}")
        print(f"  Toolsets:   {', '.join(self.enabled_toolsets) if self.enabled_toolsets else 'all'}")
        print(f"  Verbose:    {self.verbose}")
        print()
        print("  -- Session --")
        print(f"  Started:     {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Config File: cli-config.yaml {config_status}")
        print()
    
    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("(._.) No conversation history yet.")
            return
        
        print()
        print("+" + "-" * 50 + "+")
        print("|" + " " * 12 + "(^_^) Conversation History" + " " * 11 + "|")
        print("+" + "-" * 50 + "+")
        
        for i, msg in enumerate(self.conversation_history, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                print(f"\n  [You #{i}]")
                print(f"    {content[:200]}{'...' if len(content) > 200 else ''}")
            elif role == "assistant":
                print(f"\n  [Hermes #{i}]")
                preview = content[:200] if content else "(tool calls)"
                print(f"    {preview}{'...' if len(str(content)) > 200 else ''}")
        
        print()
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        print("(^_^)b Conversation reset!")
    
    def save_conversation(self):
        """Save the current conversation to a file."""
        if not self.conversation_history:
            print("(;_;) No conversation to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hermes_conversation_{timestamp}.json"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({
                    "model": self.model,
                    "session_start": self.session_start.isoformat(),
                    "messages": self.conversation_history,
                }, f, indent=2, ensure_ascii=False)
            print(f"(^_^)v Conversation saved to: {filename}")
        except Exception as e:
            print(f"(x_x) Failed to save: {e}")
    
    def retry_last(self):
        """Retry the last user message by removing the last exchange and re-sending.
        
        Removes the last assistant response (and any tool-call messages) and
        the last user message, then re-sends that user message to the agent.
        Returns the message to re-send, or None if there's nothing to retry.
        """
        if not self.conversation_history:
            print("(._.) No messages to retry.")
            return None
        
        # Walk backwards to find the last user message
        last_user_idx = None
        for i in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            print("(._.) No user message found to retry.")
            return None
        
        # Extract the message text and remove everything from that point forward
        last_message = self.conversation_history[last_user_idx].get("content", "")
        self.conversation_history = self.conversation_history[:last_user_idx]
        
        print(f"(^_^)b Retrying: \"{last_message[:60]}{'...' if len(last_message) > 60 else ''}\"")
        return last_message
    
    def undo_last(self):
        """Remove the last user/assistant exchange from conversation history.
        
        Walks backwards and removes all messages from the last user message
        onward (including assistant responses, tool calls, etc.).
        """
        if not self.conversation_history:
            print("(._.) No messages to undo.")
            return
        
        # Walk backwards to find the last user message
        last_user_idx = None
        for i in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            print("(._.) No user message found to undo.")
            return
        
        # Count how many messages we're removing
        removed_count = len(self.conversation_history) - last_user_idx
        removed_msg = self.conversation_history[last_user_idx].get("content", "")
        
        # Truncate history to before the last user message
        self.conversation_history = self.conversation_history[:last_user_idx]
        
        print(f"(^_^)b Undid {removed_count} message(s). Removed: \"{removed_msg[:60]}{'...' if len(removed_msg) > 60 else ''}\"")
        remaining = len(self.conversation_history)
        print(f"  {remaining} message(s) remaining in history.")
    
    def _handle_prompt_command(self, cmd: str):
        """Handle the /prompt command to view or set system prompt."""
        parts = cmd.split(maxsplit=1)
        
        if len(parts) > 1:
            # Set new prompt
            new_prompt = parts[1].strip()
            
            if new_prompt.lower() == "clear":
                self.system_prompt = ""
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", ""):
                    print("(^_^)b System prompt cleared (saved to config)")
                else:
                    print("(^_^) System prompt cleared (session only)")
            else:
                self.system_prompt = new_prompt
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", new_prompt):
                    print(f"(^_^)b System prompt set (saved to config)")
                else:
                    print(f"(^_^) System prompt set (session only)")
                print(f"  \"{new_prompt[:60]}{'...' if len(new_prompt) > 60 else ''}\"")
        else:
            # Show current prompt
            print()
            print("+" + "-" * 50 + "+")
            print("|" + " " * 15 + "(^_^) System Prompt" + " " * 15 + "|")
            print("+" + "-" * 50 + "+")
            print()
            if self.system_prompt:
                # Word wrap the prompt for display
                words = self.system_prompt.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 50:
                        current_line += (" " if current_line else "") + word
                    else:
                        lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                for line in lines:
                    print(f"  {line}")
            else:
                print("  (no custom prompt set - using default)")
            print()
            print("  Usage:")
            print("    /prompt <text>  - Set a custom system prompt")
            print("    /prompt clear   - Remove custom prompt")
            print("    /personality    - Use a predefined personality")
            print()
    
    def _handle_personality_command(self, cmd: str):
        """Handle the /personality command to set predefined personalities."""
        parts = cmd.split(maxsplit=1)
        
        if len(parts) > 1:
            # Set personality
            personality_name = parts[1].strip().lower()
            
            if personality_name in self.personalities:
                self.system_prompt = self.personalities[personality_name]
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", self.system_prompt):
                    print(f"(^_^)b Personality set to '{personality_name}' (saved to config)")
                else:
                    print(f"(^_^) Personality set to '{personality_name}' (session only)")
                print(f"  \"{self.system_prompt[:60]}{'...' if len(self.system_prompt) > 60 else ''}\"")
            else:
                print(f"(._.) Unknown personality: {personality_name}")
                print(f"  Available: {', '.join(self.personalities.keys())}")
        else:
            # Show available personalities
            print()
            print("+" + "-" * 50 + "+")
            print("|" + " " * 12 + "(^o^)/ Personalities" + " " * 15 + "|")
            print("+" + "-" * 50 + "+")
            print()
            for name, prompt in self.personalities.items():
                truncated = prompt[:40] + "..." if len(prompt) > 40 else prompt
                print(f"  {name:<12} - \"{truncated}\"")
            print()
            print("  Usage: /personality <name>")
            print()
    
    def _handle_cron_command(self, cmd: str):
        """Handle the /cron command to manage scheduled tasks."""
        parts = cmd.split(maxsplit=2)
        
        if len(parts) == 1:
            # /cron - show help and list
            print()
            print("+" + "-" * 60 + "+")
            print("|" + " " * 18 + "(^_^) Scheduled Tasks" + " " * 19 + "|")
            print("+" + "-" * 60 + "+")
            print()
            print("  Commands:")
            print("    /cron                     - List scheduled jobs")
            print("    /cron list                - List scheduled jobs")
            print('    /cron add <schedule> <prompt>  - Add a new job')
            print("    /cron remove <job_id>     - Remove a job")
            print()
            print("  Schedule formats:")
            print("    30m, 2h, 1d              - One-shot delay")
            print('    "every 30m", "every 2h"  - Recurring interval')
            print('    "0 9 * * *"              - Cron expression')
            print()
            
            # Show current jobs
            jobs = list_jobs()
            if jobs:
                print("  Current Jobs:")
                print("  " + "-" * 55)
                for job in jobs:
                    # Format repeat status
                    times = job["repeat"].get("times")
                    completed = job["repeat"].get("completed", 0)
                    if times is None:
                        repeat_str = "forever"
                    else:
                        repeat_str = f"{completed}/{times}"
                    
                    print(f"    {job['id'][:12]:<12} | {job['schedule_display']:<15} | {repeat_str:<8}")
                    prompt_preview = job['prompt'][:45] + "..." if len(job['prompt']) > 45 else job['prompt']
                    print(f"      {prompt_preview}")
                    if job.get("next_run_at"):
                        from datetime import datetime
                        next_run = datetime.fromisoformat(job["next_run_at"])
                        print(f"      Next: {next_run.strftime('%Y-%m-%d %H:%M')}")
                    print()
            else:
                print("  No scheduled jobs. Use '/cron add' to create one.")
            print()
            return
        
        subcommand = parts[1].lower()
        
        if subcommand == "list":
            # /cron list - just show jobs
            jobs = list_jobs()
            if not jobs:
                print("(._.) No scheduled jobs.")
                return
            
            print()
            print("Scheduled Jobs:")
            print("-" * 70)
            for job in jobs:
                times = job["repeat"].get("times")
                completed = job["repeat"].get("completed", 0)
                repeat_str = "forever" if times is None else f"{completed}/{times}"
                
                print(f"  ID: {job['id']}")
                print(f"  Name: {job['name']}")
                print(f"  Schedule: {job['schedule_display']} ({repeat_str})")
                print(f"  Next run: {job.get('next_run_at', 'N/A')}")
                print(f"  Prompt: {job['prompt'][:80]}{'...' if len(job['prompt']) > 80 else ''}")
                if job.get("last_run_at"):
                    print(f"  Last run: {job['last_run_at']} ({job.get('last_status', '?')})")
                print()
        
        elif subcommand == "add":
            # /cron add <schedule> <prompt>
            if len(parts) < 3:
                print("(._.) Usage: /cron add <schedule> <prompt>")
                print("  Example: /cron add 30m Remind me to take a break")
                print('  Example: /cron add "every 2h" Check server status at 192.168.1.1')
                return
            
            # Parse schedule and prompt
            rest = parts[2].strip()
            
            # Handle quoted schedule (e.g., "every 30m" or "0 9 * * *")
            if rest.startswith('"'):
                # Find closing quote
                close_quote = rest.find('"', 1)
                if close_quote == -1:
                    print("(._.) Unmatched quote in schedule")
                    return
                schedule = rest[1:close_quote]
                prompt = rest[close_quote + 1:].strip()
            else:
                # First word is schedule
                schedule_parts = rest.split(maxsplit=1)
                schedule = schedule_parts[0]
                prompt = schedule_parts[1] if len(schedule_parts) > 1 else ""
            
            if not prompt:
                print("(._.) Please provide a prompt for the job")
                return
            
            try:
                job = create_job(prompt=prompt, schedule=schedule)
                print(f"(^_^)b Created job: {job['id']}")
                print(f"  Schedule: {job['schedule_display']}")
                print(f"  Next run: {job['next_run_at']}")
            except Exception as e:
                print(f"(x_x) Failed to create job: {e}")
        
        elif subcommand == "remove" or subcommand == "rm" or subcommand == "delete":
            # /cron remove <job_id>
            if len(parts) < 3:
                print("(._.) Usage: /cron remove <job_id>")
                return
            
            job_id = parts[2].strip()
            job = get_job(job_id)
            
            if not job:
                print(f"(._.) Job not found: {job_id}")
                return
            
            if remove_job(job_id):
                print(f"(^_^)b Removed job: {job['name']} ({job_id})")
            else:
                print(f"(x_x) Failed to remove job: {job_id}")
        
        else:
            print(f"(._.) Unknown cron command: {subcommand}")
            print("  Available: list, add, remove")
    
    def _handle_skills_command(self, cmd: str):
        """Handle /skills slash command — delegates to hermes_cli.skills_hub."""
        from hermes_cli.skills_hub import handle_skills_slash
        handle_skills_slash(cmd, self.console)

    def _show_gateway_status(self):
        """Show status of the gateway and connected messaging platforms."""
        from gateway.config import load_gateway_config, Platform
        
        print()
        print("+" + "-" * 60 + "+")
        print("|" + " " * 15 + "(✿◠‿◠) Gateway Status" + " " * 17 + "|")
        print("+" + "-" * 60 + "+")
        print()
        
        try:
            config = load_gateway_config()
            connected = config.get_connected_platforms()
            
            print("  Messaging Platform Configuration:")
            print("  " + "-" * 55)
            
            platform_status = {
                Platform.TELEGRAM: ("Telegram", "TELEGRAM_BOT_TOKEN"),
                Platform.DISCORD: ("Discord", "DISCORD_BOT_TOKEN"),
                Platform.WHATSAPP: ("WhatsApp", "WHATSAPP_ENABLED"),
            }
            
            for platform, (name, env_var) in platform_status.items():
                pconfig = config.platforms.get(platform)
                if pconfig and pconfig.enabled:
                    home = config.get_home_channel(platform)
                    home_str = f" → {home.name}" if home else ""
                    print(f"    ✓ {name:<12} Enabled{home_str}")
                else:
                    print(f"    ○ {name:<12} Not configured ({env_var})")
            
            print()
            print("  Session Reset Policy:")
            print("  " + "-" * 55)
            policy = config.default_reset_policy
            print(f"    Mode: {policy.mode}")
            print(f"    Daily reset at: {policy.at_hour}:00")
            print(f"    Idle timeout: {policy.idle_minutes} minutes")
            
            print()
            print("  To start the gateway:")
            print("    python cli.py --gateway")
            print()
            print("  Configuration file: ~/.hermes/gateway.json")
            print()
            
        except Exception as e:
            print(f"  Error loading gateway config: {e}")
            print()
            print("  To configure the gateway:")
            print("    1. Set environment variables:")
            print("       TELEGRAM_BOT_TOKEN=your_token")
            print("       DISCORD_BOT_TOKEN=your_token")
            print("    2. Or create ~/.hermes/gateway.json")
            print()
    
    def process_command(self, command: str) -> bool:
        """
        Process a slash command.
        
        Args:
            command: The command string (starting with /)
            
        Returns:
            bool: True to continue, False to exit
        """
        # Lowercase only for dispatch matching; preserve original case for arguments
        cmd_lower = command.lower().strip()
        cmd_original = command.strip()
        
        if cmd_lower in ("/quit", "/exit", "/q"):
            return False
        elif cmd_lower == "/help":
            self.show_help()
        elif cmd_lower == "/tools":
            self.show_tools()
        elif cmd_lower == "/toolsets":
            self.show_toolsets()
        elif cmd_lower == "/config":
            self.show_config()
        elif cmd_lower == "/clear":
            # Clear terminal screen using Rich (portable, no shell needed)
            self.console.clear()
            # Reset conversation
            self.conversation_history = []
            # Show fresh banner
            self.show_banner()
            print("  ✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n")
        elif cmd_lower == "/history":
            self.show_history()
        elif cmd_lower in ("/reset", "/new"):
            self.reset_conversation()
        elif cmd_lower.startswith("/model"):
            # Use original case so model names like "Anthropic/Claude-Opus-4" are preserved
            parts = cmd_original.split(maxsplit=1)
            if len(parts) > 1:
                new_model = parts[1]
                self.model = new_model
                self.agent = None  # Force re-init
                # Save to config
                if save_config_value("model.default", new_model):
                    print(f"(^_^)b Model changed to: {new_model} (saved to config)")
                else:
                    print(f"(^_^) Model changed to: {new_model} (session only)")
            else:
                print(f"Current model: {self.model}")
                print("  Usage: /model <model-name> to change")
        elif cmd_lower.startswith("/prompt"):
            # Use original case so prompt text isn't lowercased
            self._handle_prompt_command(cmd_original)
        elif cmd_lower.startswith("/personality"):
            # Use original case (handler lowercases the personality name itself)
            self._handle_personality_command(cmd_original)
        elif cmd_lower == "/retry":
            retry_msg = self.retry_last()
            if retry_msg and hasattr(self, '_pending_input'):
                # Re-queue the message so process_loop sends it to the agent
                self._pending_input.put(retry_msg)
        elif cmd_lower == "/undo":
            self.undo_last()
        elif cmd_lower == "/save":
            self.save_conversation()
        elif cmd_lower.startswith("/cron"):
            self._handle_cron_command(cmd_original)
        elif cmd_lower.startswith("/skills"):
            self._handle_skills_command(cmd_original)
        elif cmd_lower == "/platforms" or cmd_lower == "/gateway":
            self._show_gateway_status()
        else:
            self.console.print(f"[bold red]Unknown command: {cmd_lower}[/]")
            self.console.print("[dim #B8860B]Type /help for available commands[/]")
        
        return True
    
    def _clarify_callback(self, question, choices):
        """
        Platform callback for the clarify tool. Called from the agent thread.

        Sets up the interactive selection UI (or freetext prompt for open-ended
        questions), then blocks until the user responds via the prompt_toolkit
        key bindings.  If no response arrives within the configured timeout the
        question is dismissed and the agent is told to decide on its own.
        """
        import time as _time

        timeout = CLI_CONFIG.get("clarify", {}).get("timeout", 120)
        response_queue = queue.Queue()
        is_open_ended = not choices or len(choices) == 0

        self._clarify_state = {
            "question": question,
            "choices": choices if not is_open_ended else [],
            "selected": 0,
            "response_queue": response_queue,
        }
        self._clarify_deadline = _time.monotonic() + timeout
        # Open-ended questions skip straight to freetext input
        self._clarify_freetext = is_open_ended

        # Trigger prompt_toolkit repaint from this (non-main) thread
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()

        # Poll in 1-second ticks so the countdown refreshes in the UI.
        # Each tick triggers an invalidate() to repaint the hint line.
        while True:
            try:
                result = response_queue.get(timeout=1)
                self._clarify_deadline = 0
                return result
            except queue.Empty:
                remaining = self._clarify_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                # Repaint so the countdown updates
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()

        # Timed out — tear down the UI and let the agent decide
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = 0
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")
        return (
            "The user did not provide a response within the time limit. "
            "Use your best judgement to make the choice and proceed."
        )

    def chat(self, message: str) -> Optional[str]:
        """
        Send a message to the agent and get a response.
        
        Uses a dedicated _interrupt_queue (separate from _pending_input) to avoid
        race conditions between the process_loop and interrupt monitoring. Messages
        typed while the agent is running go to _interrupt_queue; messages typed while
        idle go to _pending_input.
        
        Args:
            message: The user's message
            
        Returns:
            The agent's response, or None on error
        """
        # Initialize agent if needed
        if not self._init_agent():
            return None
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        w = self.console.width
        _cprint(f"{_GOLD}{'─' * w}{_RST}")
        print(flush=True)
        
        try:
            # Run the conversation with interrupt monitoring
            result = None
            
            def run_agent():
                nonlocal result
                result = self.agent.run_conversation(
                    user_message=message,
                    conversation_history=self.conversation_history[:-1],  # Exclude the message we just added
                )
            
            # Start agent in background thread
            agent_thread = threading.Thread(target=run_agent)
            agent_thread.start()
            
            # Monitor the dedicated interrupt queue while the agent runs.
            # _interrupt_queue is separate from _pending_input, so process_loop
            # and chat() never compete for the same queue.
            # When a clarify question is active, user input is handled entirely
            # by the Enter key binding (routed to the clarify response queue),
            # so we skip interrupt processing to avoid stealing that input.
            interrupt_msg = None
            while agent_thread.is_alive():
                if hasattr(self, '_interrupt_queue'):
                    try:
                        interrupt_msg = self._interrupt_queue.get(timeout=0.1)
                        if interrupt_msg:
                            # If clarify is active, the Enter handler routes
                            # input directly; this queue shouldn't have anything.
                            # But if it does (race condition), don't interrupt.
                            if self._clarify_state or self._clarify_freetext:
                                continue
                            print(f"\n⚡ New message detected, interrupting...")
                            self.agent.interrupt(interrupt_msg)
                            break
                    except queue.Empty:
                        pass  # Queue empty or timeout, continue waiting
                else:
                    # Fallback for non-interactive mode (e.g., single-query)
                    agent_thread.join(0.1)
            
            agent_thread.join()  # Ensure agent thread completes

            # Drain any remaining agent output still in the StdoutProxy
            # buffer so tool/status lines render ABOVE our response box.
            # The flush pushes data into the renderer queue; the short
            # sleep lets the renderer actually paint it before we draw.
            import time as _time
            sys.stdout.flush()
            _time.sleep(0.15)

            # Update history with full conversation
            self.conversation_history = result.get("messages", self.conversation_history) if result else self.conversation_history
            
            # Get the final response
            response = result.get("final_response", "") if result else ""
            
            # Handle failed results (e.g., non-retryable errors like invalid model)
            if result and result.get("failed") and not response:
                error_detail = result.get("error", "Unknown error")
                response = f"Error: {error_detail}"
            
            # Handle interrupt - check if we were interrupted
            pending_message = None
            if result and result.get("interrupted"):
                pending_message = result.get("interrupt_message") or interrupt_msg
                # Add indicator that we were interrupted
                if response and pending_message:
                    response = response + "\n\n---\n_[Interrupted - processing new message]_"
            
            if response:
                w = self.console.width
                label = " ⚕ Hermes "
                fill = w - 2 - len(label)  # 2 for ╭ and ╮
                top = f"{_GOLD}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}"
                bot = f"{_GOLD}╰{'─' * (w - 2)}╯{_RST}"

                # Render box + response as a single _cprint call so
                # nothing can interleave between the box borders.
                _cprint(f"\n{top}\n{response}\n\n{bot}")
            
            # If we have a pending message from interrupt, re-queue it for process_loop
            # instead of recursing (avoids unbounded recursion from rapid interrupts)
            if pending_message and hasattr(self, '_pending_input'):
                print(f"\n📨 Queued: '{pending_message[:50]}{'...' if len(pending_message) > 50 else ''}'")
                self._pending_input.put(pending_message)
            
            return response
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        self.show_banner()
        self.console.print("[#FFF8DC]Welcome to Hermes Agent! Type your message or /help for commands.[/]")
        self.console.print()
        
        # State for async operation
        self._agent_running = False
        self._pending_input = queue.Queue()     # For normal input (commands + new queries)
        self._interrupt_queue = queue.Queue()   # For messages typed while agent is running
        self._should_exit = False
        self._last_ctrl_c_time = 0  # Track double Ctrl+C for force exit

        # Clarify tool state: interactive question/answer with the user.
        # When the agent calls the clarify tool, _clarify_state is set and
        # the prompt_toolkit UI switches to a selection mode.
        self._clarify_state = None      # dict with question, choices, selected, response_queue
        self._clarify_freetext = False  # True when user chose "Other" and is typing
        self._clarify_deadline = 0      # monotonic timestamp when the clarify times out
        
        # Key bindings for the input area
        kb = KeyBindings()
        
        @kb.add('enter')
        def handle_enter(event):
            """Handle Enter key - submit input.
            
            Routes to the correct queue based on agent state:
            - Clarify freetext mode: answer goes to the clarify response queue
            - Clarify choice mode: selected choice goes to the clarify response queue
            - Agent running: goes to _interrupt_queue (chat() monitors this)
            - Agent idle: goes to _pending_input (process_loop monitors this)
            Commands (starting with /) always go to _pending_input so they're
            handled as commands, not sent as interrupt text to the agent.
            """
            # --- Clarify freetext mode: user typed their own answer ---
            if self._clarify_freetext and self._clarify_state:
                text = event.app.current_buffer.text.strip()
                if text:
                    self._clarify_state["response_queue"].put(text)
                    self._clarify_state = None
                    self._clarify_freetext = False
                    event.app.current_buffer.reset()
                    event.app.invalidate()
                return

            # --- Clarify choice mode: confirm the highlighted selection ---
            if self._clarify_state and not self._clarify_freetext:
                state = self._clarify_state
                selected = state["selected"]
                choices = state.get("choices") or []
                if selected < len(choices):
                    state["response_queue"].put(choices[selected])
                    self._clarify_state = None
                    event.app.invalidate()
                else:
                    # "Other" selected → switch to freetext
                    self._clarify_freetext = True
                    event.app.invalidate()
                return

            # --- Normal input routing ---
            text = event.app.current_buffer.text.strip()
            if text:
                if self._agent_running and not text.startswith("/"):
                    self._interrupt_queue.put(text)
                else:
                    self._pending_input.put(text)
                event.app.current_buffer.reset()
        
        @kb.add('escape', 'enter')
        def handle_alt_enter(event):
            """Alt+Enter inserts a newline for multi-line input."""
            event.current_buffer.insert_text('\n')

        @kb.add('c-j')
        def handle_ctrl_enter(event):
            """Ctrl+Enter (c-j) inserts a newline. Most terminals send c-j for Ctrl+Enter."""
            event.current_buffer.insert_text('\n')

        # --- Clarify tool: arrow-key navigation for multiple-choice questions ---

        @kb.add('up', filter=Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext))
        def clarify_up(event):
            """Move selection up in clarify choices."""
            if self._clarify_state:
                self._clarify_state["selected"] = max(0, self._clarify_state["selected"] - 1)
                event.app.invalidate()

        @kb.add('down', filter=Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext))
        def clarify_down(event):
            """Move selection down in clarify choices."""
            if self._clarify_state:
                choices = self._clarify_state.get("choices") or []
                max_idx = len(choices)  # last index is the "Other" option
                self._clarify_state["selected"] = min(max_idx, self._clarify_state["selected"] + 1)
                event.app.invalidate()
        
        @kb.add('c-c')
        def handle_ctrl_c(event):
            """Handle Ctrl+C - interrupt agent or force exit on double press.
            
            First Ctrl+C: interrupt the running agent gracefully.
            Second Ctrl+C within 2 seconds (or when agent is idle): force exit.
            """
            import time as _time
            now = _time.time()
            
            if self._agent_running and self.agent:
                # Check for double Ctrl+C (second press within 2 seconds)
                if now - self._last_ctrl_c_time < 2.0:
                    print("\n⚡ Force exiting...")
                    self._should_exit = True
                    event.app.exit()
                    return
                
                # First Ctrl+C: try graceful interrupt
                self._last_ctrl_c_time = now
                print("\n⚡ Interrupting agent... (press Ctrl+C again to force exit)")
                self.agent.interrupt()
            else:
                # Agent not running, exit immediately
                self._should_exit = True
                event.app.exit()
        
        @kb.add('c-d')
        def handle_ctrl_d(event):
            """Handle Ctrl+D - exit."""
            self._should_exit = True
            event.app.exit()
        
        # Dynamic prompt: shows Hermes symbol when agent is working,
        # or answer prompt when clarify freetext mode is active.
        cli_ref = self

        def get_prompt():
            if cli_ref._clarify_freetext:
                return [('class:clarify-selected', '✎ ❯ ')]
            if cli_ref._clarify_state:
                return [('class:prompt-working', '? ❯ ')]
            if cli_ref._agent_running:
                return [('class:prompt-working', '⚕ ❯ ')]
            return [('class:prompt', '❯ ')]

        # Create the input area with multiline (shift+enter), autocomplete, and paste handling
        input_area = TextArea(
            height=Dimension(min=1, max=8, preferred=1),
            prompt=get_prompt,
            style='class:input-area',
            multiline=True,
            wrap_lines=True,
            history=FileHistory(str(self._history_file)),
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
        )

        # Dynamic height: return the exact line count so the TextArea is
        # always exactly as tall as its content -- no extra blank space.
        # The bottom rule sits directly below the last line of text and
        # pushes down only when the user adds a newline.
        def _input_height():
            try:
                lines = input_area.buffer.document.line_count
                return min(max(lines, 1), 8)
            except Exception:
                return 1

        input_area.window.height = _input_height

        # Paste collapsing: detect large pastes and save to temp file
        _paste_counter = [0]

        def _on_text_changed(buf):
            """Detect large pastes and collapse them to a file reference."""
            text = buf.text
            line_count = text.count('\n')
            # Heuristic: if text jumps to 5+ lines in one change, it's a paste
            if line_count >= 5 and not text.startswith('/'):
                _paste_counter[0] += 1
                # Save to temp file
                paste_dir = Path(os.path.expanduser("~/.hermes/pastes"))
                paste_dir.mkdir(parents=True, exist_ok=True)
                paste_file = paste_dir / f"paste_{_paste_counter[0]}_{datetime.now().strftime('%H%M%S')}.txt"
                paste_file.write_text(text, encoding="utf-8")
                # Replace buffer with compact reference
                buf.text = f"[Pasted text #{_paste_counter[0]}: {line_count + 1} lines → {paste_file}]"
                buf.cursor_position = len(buf.text)

        input_area.buffer.on_text_changed += _on_text_changed

        # Hint line above input: shows placeholder when agent is working
        # and the user hasn't typed anything yet. Disappears when idle
        # or when the user starts typing.
        def get_hint_text():
            if not cli_ref._agent_running:
                return []
            # When clarify is active, show a different hint with countdown
            if cli_ref._clarify_state:
                import time as _time
                remaining = max(0, int(cli_ref._clarify_deadline - _time.monotonic()))
                countdown = f'  ({remaining}s)' if cli_ref._clarify_deadline else ''
                if cli_ref._clarify_freetext:
                    return [
                        ('class:hint', '  type your answer and press Enter'),
                        ('class:clarify-countdown', countdown),
                    ]
                return [
                    ('class:hint', '  ↑/↓ to select, Enter to confirm'),
                    ('class:clarify-countdown', countdown),
                ]
            buf = input_area.buffer
            if buf.text:
                return []
            return [('class:hint', '  type here to interrupt')]

        def get_hint_height():
            if cli_ref._clarify_state:
                return 1
            return 1 if cli_ref._agent_running else 0

        spacer = Window(
            content=FormattedTextControl(get_hint_text),
            height=get_hint_height,
        )

        # --- Clarify tool: dynamic display widget for questions + choices ---

        def _get_clarify_display():
            """Build styled text for the clarify question/choices panel."""
            state = cli_ref._clarify_state
            if not state:
                return []

            question = state["question"]
            choices = state.get("choices") or []
            selected = state.get("selected", 0)

            lines = []
            # Box top border
            lines.append(('class:clarify-border', '╭─ '))
            lines.append(('class:clarify-title', 'Hermes needs your input'))
            lines.append(('class:clarify-border', ' ─────────────────────────────╮\n'))
            lines.append(('class:clarify-border', '│\n'))

            # Question text
            lines.append(('class:clarify-border', '│  '))
            lines.append(('class:clarify-question', question))
            lines.append(('', '\n'))
            lines.append(('class:clarify-border', '│\n'))

            if choices:
                # Multiple-choice mode: show selectable options
                for i, choice in enumerate(choices):
                    lines.append(('class:clarify-border', '│  '))
                    if i == selected and not cli_ref._clarify_freetext:
                        lines.append(('class:clarify-selected', f'❯ {choice}'))
                    else:
                        lines.append(('class:clarify-choice', f'  {choice}'))
                    lines.append(('', '\n'))

                # "Other" option (5th line, only shown when choices exist)
                other_idx = len(choices)
                lines.append(('class:clarify-border', '│  '))
                if selected == other_idx and not cli_ref._clarify_freetext:
                    lines.append(('class:clarify-selected', '❯ Other (type your answer)'))
                elif cli_ref._clarify_freetext:
                    lines.append(('class:clarify-active-other', '❯ Other (type below)'))
                else:
                    lines.append(('class:clarify-choice', '  Other (type your answer)'))
                lines.append(('', '\n'))

            lines.append(('class:clarify-border', '│\n'))
            lines.append(('class:clarify-border', '╰──────────────────────────────────────────────────╯\n'))
            return lines

        clarify_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_clarify_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._clarify_state is not None),
        )
        
        # Horizontal rules above and below the input (bronze, 1 line each).
        # The bottom rule moves down as the TextArea grows with newlines.
        input_rule_top = Window(
            content=FormattedTextControl([('class:input-rule', '─' * 200)]),
            height=1,
        )
        input_rule_bot = Window(
            content=FormattedTextControl([('class:input-rule', '─' * 200)]),
            height=1,
        )

        # Layout: spacer + ruled input at bottom, completions below.
        # Using inline CompletionsMenu (not a Float) so it reliably appears even
        # after agent output has filled the terminal via patch_stdout.  Float-based
        # menus lose their rendering space in non-full-screen mode once scrollback
        # pushes the app area to the very bottom of the terminal.
        # The clarify_widget appears above the input area when the agent
        # asks a multiple-choice or open-ended question.
        layout = Layout(
            HSplit([
                Window(height=0),
                clarify_widget,
                spacer,
                input_rule_top,
                input_area,
                input_rule_bot,
                CompletionsMenu(max_height=12, scroll_offset=1),
            ])
        )
        
        # Style for the application
        style = PTStyle.from_dict({
            'input-area': '#FFF8DC',
            'prompt': '#FFF8DC',
            'prompt-working': '#888888 italic',
            'hint': '#555555 italic',
            # Bronze horizontal rules around the input area
            'input-rule': '#CD7F32',
            'completion-menu': 'bg:#1a1a2e #FFF8DC',
            'completion-menu.completion': 'bg:#1a1a2e #FFF8DC',
            'completion-menu.completion.current': 'bg:#333355 #FFD700',
            'completion-menu.meta.completion': 'bg:#1a1a2e #888888',
            'completion-menu.meta.completion.current': 'bg:#333355 #FFBF00',
            # Clarify question panel
            'clarify-border': '#CD7F32',
            'clarify-title': '#FFD700 bold',
            'clarify-question': '#FFF8DC bold',
            'clarify-choice': '#AAAAAA',
            'clarify-selected': '#FFD700 bold',
            'clarify-active-other': '#FFD700 italic',
            'clarify-countdown': '#CD7F32',
        })
        
        # Create the application
        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
        )
        self._app = app  # Store reference for clarify_callback
        
        # Background thread to process inputs and run agent
        def process_loop():
            while not self._should_exit:
                try:
                    # Check for pending input with timeout
                    try:
                        user_input = self._pending_input.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Check for commands
                    if user_input.startswith("/"):
                        print(f"\n⚙️  {user_input}")
                        if not self.process_command(user_input):
                            self._should_exit = True
                            # Schedule app exit
                            if app.is_running:
                                app.exit()
                        continue
                    
                    # Expand paste references back to full content
                    import re as _re
                    paste_match = _re.match(r'\[Pasted text #\d+: \d+ lines → (.+)\]', user_input)
                    if paste_match:
                        paste_path = Path(paste_match.group(1))
                        if paste_path.exists():
                            full_text = paste_path.read_text(encoding="utf-8")
                            line_count = full_text.count('\n') + 1
                            print()
                            _cprint(f"{_GOLD}●{_RST} {_BOLD}[Pasted text: {line_count} lines]{_RST}")
                            user_input = full_text
                        else:
                            print()
                            _cprint(f"{_GOLD}●{_RST} {_BOLD}{user_input}{_RST}")
                    else:
                        if '\n' in user_input:
                            first_line = user_input.split('\n')[0]
                            line_count = user_input.count('\n') + 1
                            print()
                            _cprint(f"{_GOLD}●{_RST} {_BOLD}{first_line}{_RST} {_DIM}(+{line_count - 1} lines){_RST}")
                        else:
                            print()
                            _cprint(f"{_GOLD}●{_RST} {_BOLD}{user_input}{_RST}")
                    
                    # Regular chat - run agent
                    self._agent_running = True
                    app.invalidate()  # Refresh status line
                    
                    try:
                        self.chat(user_input)
                    finally:
                        self._agent_running = False
                        app.invalidate()  # Refresh status line
                    
                except Exception as e:
                    print(f"Error: {e}")
        
        # Start processing thread
        process_thread = threading.Thread(target=process_loop, daemon=True)
        process_thread.start()
        
        # Register atexit cleanup so resources are freed even on unexpected exit
        atexit.register(_run_cleanup)
        
        # Run the application with patch_stdout for proper output handling
        try:
            with patch_stdout():
                app.run()
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            self._should_exit = True
            # Close session in SQLite
            if hasattr(self, '_session_db') and self._session_db and self.agent:
                try:
                    self._session_db.end_session(self.agent.session_id, "cli_close")
                except Exception:
                    pass
            _run_cleanup()
            print("\nGoodbye! ⚕")


# ============================================================================
# Main Entry Point
# ============================================================================

def main(
    query: str = None,
    q: str = None,
    toolsets: str = None,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    max_turns: int = 60,
    verbose: bool = False,
    compact: bool = False,
    list_tools: bool = False,
    list_toolsets: bool = False,
    cron_daemon: bool = False,
    cron_tick_once: bool = False,
    gateway: bool = False,
):
    """
    Hermes Agent CLI - Interactive AI Assistant
    
    Args:
        query: Single query to execute (then exit). Alias: -q
        q: Shorthand for --query
        toolsets: Comma-separated list of toolsets to enable (e.g., "web,terminal")
        model: Model to use (default: anthropic/claude-opus-4-20250514)
        api_key: API key for authentication
        base_url: Base URL for the API
        max_turns: Maximum tool-calling iterations (default: 60)
        verbose: Enable verbose logging
        compact: Use compact display mode
        list_tools: List available tools and exit
        list_toolsets: List available toolsets and exit
        cron_daemon: Run as cron daemon (check and execute due jobs continuously)
        cron_tick_once: Run due cron jobs once and exit (for system cron integration)
    
    Examples:
        python cli.py                            # Start interactive mode
        python cli.py --toolsets web,terminal    # Use specific toolsets
        python cli.py -q "What is Python?"       # Single query mode
        python cli.py --list-tools               # List tools and exit
        python cli.py --cron-daemon              # Run cron scheduler daemon
        python cli.py --cron-tick-once           # Check and run due jobs once
    """
    # Signal to terminal_tool that we're in interactive mode
    # This enables interactive sudo password prompts with timeout
    os.environ["HERMES_INTERACTIVE"] = "1"
    
    # Handle cron daemon mode (runs before CLI initialization)
    if cron_daemon:
        print("Starting Hermes Cron Daemon...")
        print("Jobs will be checked every 60 seconds.")
        print("Press Ctrl+C to stop.\n")
        run_cron_daemon(check_interval=60, verbose=True)
        return
    
    # Handle cron tick (single run for system cron integration)
    if cron_tick_once:
        jobs_run = cron_tick(verbose=True)
        if jobs_run:
            print(f"Executed {jobs_run} job(s)")
        return
    
    # Handle gateway mode (messaging platforms)
    if gateway:
        import asyncio
        from gateway.run import start_gateway
        print("Starting Hermes Gateway (messaging platforms)...")
        asyncio.run(start_gateway())
        return
    
    # Handle query shorthand
    query = query or q
    
    # Parse toolsets - handle both string and tuple/list inputs
    # Default to hermes-cli toolset which includes cronjob management tools
    toolsets_list = None
    if toolsets:
        if isinstance(toolsets, str):
            toolsets_list = [t.strip() for t in toolsets.split(",")]
        elif isinstance(toolsets, (list, tuple)):
            # Fire may pass multiple --toolsets as a tuple
            toolsets_list = []
            for t in toolsets:
                if isinstance(t, str):
                    toolsets_list.extend([x.strip() for x in t.split(",")])
                else:
                    toolsets_list.append(str(t))
    else:
        # Check config for CLI toolsets, fallback to hermes-cli
        config_cli_toolsets = CLI_CONFIG.get("platform_toolsets", {}).get("cli")
        if config_cli_toolsets and isinstance(config_cli_toolsets, list):
            toolsets_list = config_cli_toolsets
        else:
            toolsets_list = ["hermes-cli"]
    
    # Create CLI instance
    cli = HermesCLI(
        model=model,
        toolsets=toolsets_list,
        api_key=api_key,
        base_url=base_url,
        max_turns=max_turns,
        verbose=verbose,
        compact=compact,
    )
    
    # Handle list commands (don't init agent for these)
    if list_tools:
        cli.show_banner()
        cli.show_tools()
        sys.exit(0)
    
    if list_toolsets:
        cli.show_banner()
        cli.show_toolsets()
        sys.exit(0)
    
    # Register cleanup for single-query mode (interactive mode registers in run())
    atexit.register(_run_cleanup)
    
    # Handle single query mode
    if query:
        cli.show_banner()
        cli.console.print(f"[bold blue]Query:[/] {query}")
        cli.chat(query)
        return
    
    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    fire.Fire(main)
