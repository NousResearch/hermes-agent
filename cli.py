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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Suppress startup messages for clean CLI experience
os.environ["MSWEA_SILENT_STARTUP"] = "1"  # mini-swe-agent
os.environ["HERMES_QUIET"] = "1"  # Our own modules

import yaml

# prompt_toolkit for fixed input area TUI
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout

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
    Load CLI configuration from cli-config.yaml.
    
    Environment variables take precedence over config file values.
    Returns default values if config file doesn't exist.
    """
    config_path = Path(__file__).parent / 'cli-config.yaml'
    
    # Default configuration
    defaults = {
        "model": {
            "default": "anthropic/claude-opus-4-20250514",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "terminal": {
            "env_type": "local",
            "cwd": "/tmp",
            "timeout": 60,
            "lifetime_seconds": 300,
            "docker_image": "python:3.11",
            "singularity_image": "docker://python:3.11",
            "modal_image": "python:3.11",
        },
        "browser": {
            "inactivity_timeout": 120,  # Auto-cleanup inactive browser sessions after 2 min
        },
        "agent": {
            "max_turns": 20,
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
    }
    
    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            # Deep merge with defaults
            for key in defaults:
                if key in file_config:
                    if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
                        defaults[key].update(file_config[key])
                    else:
                        defaults[key] = file_config[key]
        except Exception as e:
            print(f"[Warning] Failed to load cli-config.yaml: {e}")
    
    # Apply terminal config to environment variables (so terminal_tool picks them up)
    # Only set if not already set in environment (env vars take precedence)
    terminal_config = defaults.get("terminal", {})
    
    # Handle special cwd values: "." or "auto" means use current working directory
    if terminal_config.get("cwd") in (".", "auto", "cwd"):
        terminal_config["cwd"] = os.getcwd()
        defaults["terminal"]["cwd"] = terminal_config["cwd"]
    
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
    
    # CLI config overrides .env for terminal settings
    for config_key, env_var in env_mappings.items():
        if config_key in terminal_config:
            os.environ[env_var] = str(terminal_config[config_key])
    
    # Apply browser config to environment variables
    browser_config = defaults.get("browser", {})
    browser_env_mappings = {
        "inactivity_timeout": "BROWSER_INACTIVITY_TIMEOUT",
    }
    
    for config_key, env_var in browser_env_mappings.items():
        if config_key in browser_config:
            os.environ[env_var] = str(browser_config[config_key])
    
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

# ============================================================================
# ASCII Art & Branding
# ============================================================================

# Color palette (hex colors for Rich markup):
# - Gold: #FFD700 (headers, highlights)
# - Amber: #FFBF00 (secondary highlights)
# - Bronze: #CD7F32 (tertiary elements)
# - Light: #FFF8DC (text)
# - Dim: #B8860B (muted text)

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
    Scan the skills directory and return skills grouped by category.
    
    Returns:
        Dict mapping category name to list of skill names
    """
    skills_dir = Path(__file__).parent / "skills"
    skills_by_category = {}
    
    if not skills_dir.exists():
        return skills_by_category
    
    # Scan for SKILL.md files
    for skill_file in skills_dir.rglob("SKILL.md"):
        # Get category (parent of parent if nested, else parent)
        rel_path = skill_file.relative_to(skills_dir)
        parts = rel_path.parts
        
        if len(parts) >= 2:
            category = parts[0]
            skill_name = parts[-2]  # Folder containing SKILL.md
        else:
            category = "general"
            skill_name = skill_file.parent.name
        
        if category not in skills_by_category:
            skills_by_category[category] = []
        skills_by_category[category].append(skill_name)
    
    return skills_by_category


def build_welcome_banner(console: Console, model: str, cwd: str, tools: List[dict] = None, enabled_toolsets: List[str] = None):
    """
    Build and print a Claude Code-style welcome banner with caduceus on left and info on right.
    
    Args:
        console: Rich Console instance for printing
        model: The current model name (e.g., "anthropic/claude-opus-4")
        cwd: Current working directory
        tools: List of tool definitions
        enabled_toolsets: List of enabled toolset names
    """
    tools = tools or []
    enabled_toolsets = enabled_toolsets or []
    
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
    left_content = "\n".join(left_lines)
    
    # Build right content: tools list grouped by toolset
    right_lines = []
    right_lines.append("[bold #FFBF00]Available Tools[/]")
    
    # Group tools by toolset
    toolsets_dict = {}
    for tool in tools:
        tool_name = tool["function"]["name"]
        toolset = get_toolset_for_tool(tool_name) or "other"
        if toolset not in toolsets_dict:
            toolsets_dict[toolset] = []
        toolsets_dict[toolset].append(tool_name)
    
    # Display tools grouped by toolset (compact format, max 8 groups)
    sorted_toolsets = sorted(toolsets_dict.keys())
    display_toolsets = sorted_toolsets[:8]
    remaining_toolsets = len(sorted_toolsets) - 8
    
    for toolset in display_toolsets:
        tool_names = toolsets_dict[toolset]
        # Join tool names with commas, wrap if too long
        tools_str = ", ".join(sorted(tool_names))
        if len(tools_str) > 45:
            tools_str = tools_str[:42] + "..."
        right_lines.append(f"[dim #B8860B]{toolset}:[/] [#FFF8DC]{tools_str}[/]")
    
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
    "/reset": "Reset conversation only (keep screen)",
    "/save": "Save the current conversation",
    "/config": "Show current configuration",
    "/quit": "Exit the CLI (also: /exit, /q)",
}


def save_config_value(key_path: str, value: any) -> bool:
    """
    Save a value to cli-config.yaml at the specified key path.
    
    Args:
        key_path: Dot-separated path like "agent.system_prompt"
        value: Value to save
    
    Returns:
        True if successful, False otherwise
    """
    config_path = Path(__file__).parent / 'cli-config.yaml'
    
    try:
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
        max_turns: int = 20,
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
            max_turns: Maximum conversation turns
            verbose: Enable verbose logging
            compact: Use compact display mode
        """
        # Initialize Rich console
        self.console = Console()
        self.compact = compact if compact is not None else CLI_CONFIG["display"].get("compact", False)
        self.verbose = verbose if verbose is not None else CLI_CONFIG["agent"].get("verbose", False)
        
        # Configuration - priority: CLI args > env vars > config file
        self.model = model or os.getenv("LLM_MODEL", CLI_CONFIG["model"]["default"])
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", CLI_CONFIG["model"]["base_url"])
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.max_turns = max_turns if max_turns != 20 else CLI_CONFIG["agent"].get("max_turns", 20)
        
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
        
        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        
        # Setup prompt_toolkit session with history
        self._setup_prompt_session()
    
    def _setup_prompt_session(self):
        """Setup prompt_toolkit session with history and styling."""
        history_file = Path.home() / ".hermes_history"
        
        # Custom style for the prompt
        self.prompt_style = PTStyle.from_dict({
            'prompt': '#FFD700 bold',
            'input': '#FFF8DC',
        })
        
        # Create prompt session with file history
        # Note: multiline disabled - Enter submits, use \ at end of line for continuation
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            style=self.prompt_style,
            enable_history_search=True,
        )
    
    def _init_agent(self) -> bool:
        """
        Initialize the agent on first use.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.agent is not None:
            return True
        
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
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets)
            
            # Get terminal working directory (where commands will execute)
            cwd = os.getenv("TERMINAL_CWD", os.getcwd())
            
            # Build and display the banner
            build_welcome_banner(
                console=self.console,
                model=self.model,
                cwd=cwd,
                tools=tools,
                enabled_toolsets=self.enabled_toolsets,
            )
        
        self.console.print()
    
    def _show_status(self):
        """Show current status bar."""
        # Get tool count
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets)
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
        print("  Multi-line: End a line with \\ to continue on next line")
        print()
    
    def show_tools(self):
        """Display available tools with kawaii ASCII art."""
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets)
        
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
        terminal_cwd = os.getenv("TERMINAL_CWD", "/tmp")
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
    
    def process_command(self, command: str) -> bool:
        """
        Process a slash command.
        
        Args:
            command: The command string (starting with /)
            
        Returns:
            bool: True to continue, False to exit
        """
        cmd = command.lower().strip()
        
        if cmd in ("/quit", "/exit", "/q"):
            return False
        elif cmd == "/help":
            self.show_help()
        elif cmd == "/tools":
            self.show_tools()
        elif cmd == "/toolsets":
            self.show_toolsets()
        elif cmd == "/config":
            self.show_config()
        elif cmd == "/clear":
            # Clear terminal screen
            import os as _os
            _os.system('clear' if _os.name != 'nt' else 'cls')
            # Reset conversation
            self.conversation_history = []
            # Show fresh banner
            self.show_banner()
            print("  ✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n")
        elif cmd == "/history":
            self.show_history()
        elif cmd == "/reset":
            self.reset_conversation()
        elif cmd.startswith("/model"):
            parts = cmd.split(maxsplit=1)
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
        elif cmd.startswith("/prompt"):
            self._handle_prompt_command(cmd)
        elif cmd.startswith("/personality"):
            self._handle_personality_command(cmd)
        elif cmd == "/save":
            self.save_conversation()
        else:
            self.console.print(f"[bold red]Unknown command: {cmd}[/]")
            self.console.print("[dim #B8860B]Type /help for available commands[/]")
        
        return True
    
    def chat(self, message: str) -> Optional[str]:
        """
        Send a message to the agent and get a response.
        
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
        
        # Visual separator after user input
        print("─" * 60, flush=True)
        
        try:
            # Run the conversation
            result = self.agent.run_conversation(
                user_message=message,
                conversation_history=self.conversation_history[:-1],  # Exclude the message we just added
            )
            
            # Update history with full conversation
            self.conversation_history = result.get("messages", self.conversation_history)
            
            # Get the final response
            response = result.get("final_response", "")
            
            if response:
                # Use simple print for compatibility with prompt_toolkit's patch_stdout
                print()
                print("╭" + "─" * 58 + "╮")
                print("│ ⚕ Hermes" + " " * 49 + "│")
                print("╰" + "─" * 58 + "╯")
                print()
                print(response)
                print()
                print("─" * 60)
            
            return response
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def get_input(self) -> Optional[str]:
        """
        Get user input using prompt_toolkit.
        
        Enter submits. For multiline, end line with \\ to continue.
        
        Returns:
            The user's input, or None if EOF/interrupt
        """
        try:
            # Get first line
            line = self.prompt_session.prompt(
                HTML('<prompt>❯ </prompt>'),
                style=self.prompt_style,
            )
            
            # Handle multi-line input (lines ending with \)
            lines = [line]
            while line.endswith("\\"):
                lines[-1] = line[:-1]  # Remove trailing backslash
                line = self.prompt_session.prompt(
                    HTML('<prompt>  </prompt>'),  # Continuation prompt
                    style=self.prompt_style,
                )
                lines.append(line)
            
            return "\n".join(lines).strip()
            
        except (EOFError, KeyboardInterrupt):
            return None
    
    def run(self):
        """Run the interactive CLI loop with fixed input at bottom."""
        self.show_banner()
        
        # These Rich prints work fine BEFORE patch_stdout
        self.console.print("[#FFF8DC]Welcome to Hermes Agent! Type your message or /help for commands.[/]")
        self.console.print()
        
        # Use patch_stdout to ensure all output appears above the input prompt
        with patch_stdout():
            while True:
                try:
                    user_input = self.get_input()
                    
                    if user_input is None:
                        print("\nGoodbye! ⚕")
                        break
                    
                    if not user_input:
                        continue
                    
                    # Check for commands
                    if user_input.startswith("/"):
                        if not self.process_command(user_input):
                            print("\nGoodbye! ⚕")
                            break
                        continue
                    
                    # Regular chat message
                    self.chat(user_input)
                    
                except KeyboardInterrupt:
                    print("\nInterrupted. Type /quit to exit.")
                    continue


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
    max_turns: int = 20,
    verbose: bool = False,
    compact: bool = False,
    list_tools: bool = False,
    list_toolsets: bool = False,
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
        max_turns: Maximum conversation turns (default: 20)
        verbose: Enable verbose logging
        compact: Use compact display mode
        list_tools: List available tools and exit
        list_toolsets: List available toolsets and exit
    
    Examples:
        python cli.py                            # Start interactive mode
        python cli.py --toolsets web,terminal    # Use specific toolsets
        python cli.py -q "What is Python?"       # Single query mode
        python cli.py --list-tools               # List tools and exit
    """
    # Handle query shorthand
    query = query or q
    
    # Parse toolsets - handle both string and tuple/list inputs
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
