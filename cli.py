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

import logging
import os
import shutil
import sys
import json
import atexit
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

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
from prompt_toolkit.layout.processors import Processor, Transformation, PasswordProcessor, ConditionalProcessor
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
import threading
import queue


# Load .env from ~/.hermes/.env first, then project root as dev fallback
from dotenv import load_dotenv
from hermes_constants import OPENROUTER_BASE_URL

_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_user_env = _hermes_home / ".env"
_project_env = Path(__file__).parent / '.env'
if _user_env.exists():
    try:
        load_dotenv(dotenv_path=_user_env, encoding="utf-8", override=True)
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_user_env, encoding="latin-1", override=True)
if _project_env.exists():
    try:
        load_dotenv(dotenv_path=_project_env, encoding="utf-8", override=True)
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_project_env, encoding="latin-1", override=True)

# Point mini-swe-agent at ~/.hermes/ so it shares our config
os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(_hermes_home))

# =============================================================================
# Configuration Loading
# =============================================================================

def _load_prefill_messages(file_path: str) -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from a JSON file.
    
    The file should contain a JSON array of {role, content} dicts, e.g.:
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    
    Relative paths are resolved from ~/.hermes/.
    Returns an empty list if the path is empty or the file doesn't exist.
    """
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path.home() / ".hermes" / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Prefill messages file must contain a JSON array: %s", path)
            return []
        return data
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []


def _parse_reasoning_config(effort: str) -> dict | None:
    """Parse a reasoning effort level into an OpenRouter reasoning config dict.
    
    Valid levels: "xhigh", "high", "medium", "low", "minimal", "none".
    Returns None to use the default (xhigh), or a config dict to override.
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    valid = ("xhigh", "high", "medium", "low", "minimal")
    if effort in valid:
        return {"enabled": True, "effort": effort}
    logger.warning("Unknown reasoning_effort '%s', using default (xhigh)", effort)
    return None


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
    
    # Default configuration
    defaults = {
        "model": {
            "default": "claude-haiku-4-5",
            "base_url": OPENROUTER_BASE_URL,
            "provider": "auto",
        },
        "terminal": {
            "env_type": "local",
            "cwd": ".",  # "." is resolved to os.getcwd() at runtime
            "timeout": 60,
            "lifetime_seconds": 300,
            "docker_image": "python:3.11",
            "singularity_image": "docker://python:3.11",
            "modal_image": "python:3.11",
            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
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
            "prefill_messages_file": "",
            "reasoning_effort": "",
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
        "code_execution": {
            "timeout": 300,    # Max seconds a sandbox script can run before being killed (5 min)
            "max_tool_calls": 50,  # Max RPC tool calls per execution
        },
        "delegation": {
            "max_iterations": 45,  # Max tool-calling turns per child agent
            "default_toolsets": ["terminal", "file", "web"],  # Default toolsets for subagents
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
            
            # Deep merge file_config into defaults.
            # First: merge keys that exist in both (deep-merge dicts, overwrite scalars)
            for key in defaults:
                if key == "model":
                    continue  # Already handled above
                if key in file_config:
                    if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
                        defaults[key].update(file_config[key])
                    else:
                        defaults[key] = file_config[key]
            
            # Second: carry over keys from file_config that aren't in defaults
            # (e.g. platform_toolsets, provider_routing, memory, honcho, etc.)
            for key in file_config:
                if key not in defaults and key != "model":
                    defaults[key] = file_config[key]
            
            # Handle root-level max_turns (backwards compat) - copy to agent.max_turns
            if "max_turns" in file_config and "agent" not in file_config:
                defaults["agent"]["max_turns"] = file_config["max_turns"]
        except Exception as e:
            logger.warning("Failed to load cli-config.yaml: %s", e)
    
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
        "daytona_image": "TERMINAL_DAYTONA_IMAGE",
        # SSH config
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        # Container resource config (docker, singularity, modal, daytona -- ignored for local/ssh)
        "container_cpu": "TERMINAL_CONTAINER_CPU",
        "container_memory": "TERMINAL_CONTAINER_MEMORY",
        "container_disk": "TERMINAL_CONTAINER_DISK",
        "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
        "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
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
                val = terminal_config[config_key]
                if isinstance(val, list):
                    import json
                    os.environ[env_var] = json.dumps(val)
                else:
                    os.environ[env_var] = str(val)
    
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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import fire

# Import the agent and tool systems
from run_agent import AIAgent
from model_tools import get_tool_definitions, get_toolset_for_tool

# Extracted CLI modules (Phase 3)
from hermes_cli.banner import (
    cprint as _cprint, _GOLD, _BOLD, _DIM, _RST,
    VERSION, HERMES_AGENT_LOGO, HERMES_CADUCEUS, COMPACT_BANNER,
    get_available_skills as _get_available_skills,
    build_welcome_banner,
)
from hermes_cli.commands import COMMANDS, SlashCommandCompleter
from hermes_cli import callbacks as _callbacks
from toolsets import get_all_toolsets, get_toolset_info, resolve_toolset, validate_toolset

# Cron job system for scheduled tasks (CRUD only — execution is handled by the gateway)
from cron import create_job, list_jobs, remove_job, get_job

# Resource cleanup imports for safe shutdown (terminal VMs, browser sessions)
from tools.terminal_tool import cleanup_all_environments as _cleanup_all_terminals
from tools.terminal_tool import set_sudo_password_callback, set_approval_callback
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
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
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
def _term_width() -> int:
    """Usable terminal width — accounts for scrollbar gutter in embedded terminals."""
    import shutil
    w = shutil.get_terminal_size((80, 24)).columns
    # Embedded terminals (VS Code, JetBrains) reserve ~1 col for scrollbar
    return max(w - 1, 40)

_GOLD = "\033[1;33m"    # Bold yellow — closest universal match to the gold theme
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_THINK = "\033[2;3;37m"  # dim + italic + gray — wispy thought text
_RST = "\033[0m"


def _format_think_blocks(text: str) -> str:
    """Render <think>...</think> blocks as dim italic gray text.

    Replaces raw think tags with styled ANSI output so the model's
    internal reasoning appears as wispy, faded thoughts.
    """
    import re

    def _style_think(m):
        thought = m.group(1).strip()
        if not thought:
            return ""
        # Wrap each line so long thoughts stay dim
        lines = thought.splitlines()
        styled = "\n".join(f"  {_THINK}{line}{_RST}" for line in lines)
        return f"{_THINK}  ~ thinking ~{_RST}\n{styled}\n"

    return re.sub(r'<think>(.*?)</think>', _style_think, text, flags=re.DOTALL)


def _cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's native renderer.

    Raw ANSI escapes written via print() are swallowed by patch_stdout's
    StdoutProxy.  Routing through print_formatted_text(ANSI(...)) lets
    prompt_toolkit parse the escapes and render real colors.
    """
    _pt_print(_PT_ANSI(text))


class ChatConsole:
    """Rich Console adapter for prompt_toolkit's patch_stdout context.

    Captures Rich's rendered ANSI output and routes it through _cprint
    so colors and markup render correctly inside the interactive chat loop.
    Drop-in replacement for Rich Console — just pass this to any function
    that expects a console.print() interface.
    """

    def __init__(self):
        from io import StringIO
        self._buffer = StringIO()
        self._inner = Console(file=self._buffer, force_terminal=True, highlight=False)

    def print(self, *args, **kwargs):
        self._buffer.seek(0)
        self._buffer.truncate()
        self._inner.print(*args, **kwargs)
        output = self._buffer.getvalue()
        for line in output.rstrip("\n").split("\n"):
            _cprint(line)

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


def _format_context_length(tokens: int) -> str:
    """Format a token count for display (e.g. 128000 → '128K', 1048576 → '1M')."""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        return f"{val:g}M"
    elif tokens >= 1_000:
        val = tokens / 1_000
        return f"{val:g}K"
    return str(tokens)


def build_welcome_banner(console: Console, model: str, cwd: str, tools: List[dict] = None, enabled_toolsets: List[str] = None, session_id: str = None, context_length: int = None):
    """
    Build and print a Claude Code-style welcome banner with caduceus on left and info on right.
    
    Args:
        console: Rich Console instance for printing
        model: The current model name (e.g., "anthropic/claude-opus-4")
        cwd: Current working directory
        tools: List of tool definitions
        enabled_toolsets: List of enabled toolset names
        session_id: Unique session identifier for logging
        context_length: Model's context window size in tokens
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
    
    ctx_str = f" [dim #B8860B]·[/] [dim #B8860B]{_format_context_length(context_length)} context[/]" if context_length else ""
    left_lines.append(f"[#FFBF00]{model_short}[/]{ctx_str} [dim #B8860B]·[/] [dim #B8860B]Nous Research[/]")
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
        toolset_id = item.get("id", item.get("name", "unknown"))
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
    "/model": "Switch model (opus/sonnet/haiku/qwen or full name)",
    "/provider": "Switch provider (openrouter, local, custom) with API key",
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
    "/paste": "Check clipboard for an image and attach it",
    "/reload-mcp": "Reload MCP servers from config.yaml",
    "/copycode": "Import Claude Code commands as Hermes skills",
    "/swarm": "Run parallel agent swarm on a complex task",
    "/context": "Show remaining context window (ASCII bar)",
    "/jobs": "List background agent tasks",
    "/fg": "Bring a background task to foreground (/fg <id>)",
    "/quit": "Exit the CLI (also: /exit, /q)",
}


# ============================================================================
# Skill Slash Commands — dynamic commands generated from installed skills
# ============================================================================

from agent.skill_commands import scan_skill_commands, get_skill_commands, build_skill_invocation_message

_skill_commands = scan_skill_commands()


class SlashCommandCompleter(Completer):
    """Autocomplete for /commands and /skill-name in the input area."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        word = text[1:]  # strip the leading /

        # Built-in commands
        for cmd, desc in COMMANDS.items():
            cmd_name = cmd[1:]
            if cmd_name.startswith(word):
                yield Completion(
                    cmd_name,
                    start_position=-len(word),
                    display=cmd,
                    display_meta=desc,
                )

        # Skill commands
        for cmd, info in _skill_commands.items():
            cmd_name = cmd[1:]
            if cmd_name.startswith(word):
                yield Completion(
                    cmd_name,
                    start_position=-len(word),
                    display=cmd,
                    display_meta=f"⚡ {info['description'][:50]}{'...' if len(info['description']) > 50 else ''}",
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
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        
        # Save back
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
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
        provider: str = None,
        api_key: str = None,
        base_url: str = None,
        max_turns: int = None,
        verbose: bool = False,
        compact: bool = False,
        resume: str = None,
    ):
        """
        Initialize the Hermes CLI.

        Args:
            model: Model to use (default: from env or claude-sonnet)
            toolsets: List of toolsets to enable (default: all)
            provider: Inference provider ("auto", "openrouter", "nous", "openai-codex")
            api_key: API key (default: from environment)
            base_url: API base URL (default: OpenRouter)
            max_turns: Maximum tool-calling iterations (default: 60)
            verbose: Enable verbose logging
            compact: Use compact display mode
            resume: Session ID to resume (restores conversation history from SQLite)
        """
        # Initialize Rich console
        self.console = Console()
        self.compact = compact if compact is not None else CLI_CONFIG["display"].get("compact", False)
        # tool_progress: "off", "new", "all", "verbose" (from config.yaml display section)
        self.tool_progress_mode = CLI_CONFIG["display"].get("tool_progress", "all")
        self.verbose = verbose if verbose is not None else (self.tool_progress_mode == "verbose")
        # Whether to show <think> blocks in model responses (default: off)
        self.show_thinking = CLI_CONFIG["display"].get("show_thinking", False)
        
        # Configuration - priority: CLI args > env vars > config file
        # Model can come from: CLI arg, LLM_MODEL env, OPENAI_MODEL env (custom endpoint), or config
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or CLI_CONFIG["model"]["default"]
        # Normalize model names: dots to dashes (e.g. claude-opus-4.6 -> claude-opus-4-6)
        # Anthropic API requires dashes, but configs/env may use dots
        if "claude" in self.model:
            import re as _re_model
            self.model = _re_model.sub(r'(\d+)\.(\d+)', r'\1-\2', self.model)

        self._explicit_api_key = api_key
        self._explicit_base_url = base_url

        # Provider selection is resolved lazily at use-time via _ensure_runtime_credentials().
        # Infer provider from model prefix if not explicitly set.
        _inferred_provider = None
        if self.model.startswith("local/"):
            _inferred_provider = "local"
        elif self.model.startswith("claude") or self.model.startswith("anthropic/"):
            _inferred_provider = "anthropic"
        elif self.model.startswith("openai/"):
            _inferred_provider = "auto"
        self.requested_provider = (
            provider
            or os.getenv("HERMES_INFERENCE_PROVIDER")
            or _inferred_provider
            or CLI_CONFIG["model"].get("provider")
            or "auto"
        )
        self._provider_source: Optional[str] = None
        self.provider = self.requested_provider
        self.api_mode = "chat_completions"
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL", CLI_CONFIG["model"]["base_url"])
        )
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._nous_key_expires_at: Optional[str] = None
        self._nous_key_source: Optional[str] = None
        # Max turns priority: CLI arg > config file > env var > default
        if max_turns is not None:  # CLI arg was explicitly set
            self.max_turns = max_turns
        elif CLI_CONFIG["agent"].get("max_turns"):
            self.max_turns = CLI_CONFIG["agent"]["max_turns"]
        elif CLI_CONFIG.get("max_turns"):  # Backwards compat: root-level max_turns
            self.max_turns = CLI_CONFIG["max_turns"]
        elif os.getenv("HERMES_MAX_ITERATIONS"):
            self.max_turns = int(os.getenv("HERMES_MAX_ITERATIONS"))
        else:
            self.max_turns = 60
        
        # Parse and validate toolsets
        self.enabled_toolsets = toolsets
        if toolsets and "all" not in toolsets and "*" not in toolsets:
            # Validate each toolset
            invalid = [t for t in toolsets if not validate_toolset(t)]
            if invalid:
                self.console.print(f"[bold red]Warning: Unknown toolsets: {', '.join(invalid)}[/]")
        
        # Ephemeral system prompt: env var takes precedence, then config
        self.system_prompt = (
            os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
            or CLI_CONFIG["agent"].get("system_prompt", "")
        )
        self.personalities = CLI_CONFIG["agent"].get("personalities", {})
        
        # Ephemeral prefill messages (few-shot priming, never persisted)
        self.prefill_messages = _load_prefill_messages(
            CLI_CONFIG["agent"].get("prefill_messages_file", "")
        )
        
        # Reasoning config (OpenRouter reasoning effort level)
        self.reasoning_config = _parse_reasoning_config(
            CLI_CONFIG["agent"].get("reasoning_effort", "")
        )
        
        # OpenRouter provider routing preferences
        pr = CLI_CONFIG.get("provider_routing", {}) or {}
        self._provider_sort = pr.get("sort")
        self._providers_only = pr.get("only")
        self._providers_ignore = pr.get("ignore")
        self._providers_order = pr.get("order")
        self._provider_require_params = pr.get("require_parameters", False)
        self._provider_data_collection = pr.get("data_collection")
        
        # Agent will be initialized on first use
        self.agent: Optional[AIAgent] = None
        self._app = None  # prompt_toolkit Application (set in run())
        
        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        self._resumed = False
        
        # Session ID: reuse existing one when resuming, otherwise generate fresh
        if resume:
            self.session_id = resume
            self._resumed = True
        else:
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        
        # History file for persistent input recall across sessions
        self._history_file = Path.home() / ".hermes_history"
        self._last_invalidate: float = 0.0  # throttle UI repaints

    def _invalidate(self, min_interval: float = 0.25) -> None:
        """Throttled UI repaint — prevents terminal blinking on slow/SSH connections."""
        import time as _time
        now = _time.monotonic()
        if hasattr(self, "_app") and self._app and (now - self._last_invalidate) >= min_interval:
            self._last_invalidate = now
            self._app.invalidate()

    def _ensure_runtime_credentials(self) -> bool:
        """
        Ensure runtime credentials are resolved before agent use.
        Re-resolves provider credentials so key rotation and token refresh
        are picked up without restarting the CLI.
        Returns True if credentials are ready, False on auth failure.
        """
        from hermes_cli.runtime_provider import (
            resolve_runtime_provider,
            format_runtime_provider_error,
        )

        try:
            runtime = resolve_runtime_provider(
                requested=self.requested_provider,
                explicit_api_key=self._explicit_api_key,
                explicit_base_url=self._explicit_base_url,
            )
        except Exception as exc:
            message = format_runtime_provider_error(exc)
            self.console.print(f"[bold red]{message}[/]")
            return False

        api_key = runtime.get("api_key")
        base_url = runtime.get("base_url")
        resolved_provider = runtime.get("provider", "openrouter")
        resolved_api_mode = runtime.get("api_mode", self.api_mode)
        is_local = resolved_provider == "local"
        is_litellm = resolved_provider == "anthropic"  # litellm handles routing
        if not is_local and not is_litellm and (not isinstance(api_key, str) or not api_key):
            self.console.print("[bold red]Provider resolver returned an empty API key.[/]")
            return False
        if not is_litellm and (not isinstance(base_url, str) or not base_url):
            self.console.print("[bold red]Provider resolver returned an empty base URL.[/]")
            return False

        credentials_changed = api_key != self.api_key or base_url != self.base_url
        routing_changed = (
            resolved_provider != self.provider
            or resolved_api_mode != self.api_mode
        )
        self.provider = resolved_provider
        self.api_mode = resolved_api_mode
        self._provider_source = runtime.get("source")
        self.api_key = api_key
        self.base_url = base_url

        # AIAgent/OpenAI client holds auth at init time, so rebuild if key rotated
        if (credentials_changed or routing_changed) and self.agent is not None:
            self.agent = None

        return True

    def _init_agent(self) -> bool:
        """
        Initialize the agent on first use.
        When resuming a session, restores conversation history from SQLite.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.agent is not None:
            return True

        if not self._ensure_runtime_credentials():
            return False

        # Initialize SQLite session store for CLI sessions
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            logger.debug("SQLite session store not available: %s", e)
        
        # If resuming, validate the session exists and load its history
        if self._resumed and self._session_db:
            session_meta = self._session_db.get_session(self.session_id)
            if not session_meta:
                _cprint(f"\033[1;31mSession not found: {self.session_id}{_RST}")
                _cprint(f"{_DIM}Use a session ID from a previous CLI run (hermes sessions list).{_RST}")
                return False
            restored = self._session_db.get_messages_as_conversation(self.session_id)
            if restored:
                self.conversation_history = restored
                msg_count = len([m for m in restored if m.get("role") == "user"])
                _cprint(
                    f"{_GOLD}↻ Resumed session {_BOLD}{self.session_id}{_RST}{_GOLD} "
                    f"({msg_count} user message{'s' if msg_count != 1 else ''}, "
                    f"{len(restored)} total messages){_RST}"
                )
            else:
                _cprint(f"{_GOLD}Session {self.session_id} found but has no messages. Starting fresh.{_RST}")
            # Re-open the session (clear ended_at so it's active again)
            try:
                self._session_db._conn.execute(
                    "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                    (self.session_id,),
                )
                self._session_db._conn.commit()
            except Exception:
                pass
        
        try:
            self.agent = AIAgent(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                provider=self.provider,
                api_mode=self.api_mode,
                max_iterations=self.max_turns,
                enabled_toolsets=self.enabled_toolsets,
                verbose_logging=self.verbose,
                quiet_mode=True,
                ephemeral_system_prompt=self.system_prompt if self.system_prompt else None,
                prefill_messages=self.prefill_messages or None,
                reasoning_config=self.reasoning_config,
                providers_allowed=self._providers_only,
                providers_ignored=self._providers_ignore,
                providers_order=self._providers_order,
                provider_sort=self._provider_sort,
                provider_require_parameters=self._provider_require_params,
                provider_data_collection=self._provider_data_collection,
                session_id=self.session_id,
                platform="cli",
                session_db=self._session_db,
                clarify_callback=self._clarify_callback,
                honcho_session_key=self.session_id,
            )
            return True
        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize agent: {e}[/]")
            return False
    
    def show_banner(self):
        """Display the welcome banner in Claude Code style."""
        # First launch: let user pick color scheme with a single keypress
        from hermes_cli.color_scheme import needs_scheme_pick, pick_color_scheme
        if needs_scheme_pick():
            pick_color_scheme()

        self.console.clear()

        if self.compact:
            self.console.print(COMPACT_BANNER)
            self._show_status()
        else:
            # Get tools for display
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
            
            # Get terminal working directory (where commands will execute)
            cwd = os.getenv("TERMINAL_CWD", os.getcwd())
            
            # Get context length for display
            ctx_len = None
            if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'context_compressor'):
                ctx_len = self.agent.context_compressor.context_length
            
            # Build and display the banner
            build_welcome_banner(
                console=self.console,
                model=self.model,
                cwd=cwd,
                tools=tools,
                enabled_toolsets=self.enabled_toolsets,
                session_id=self.session_id,
                context_length=ctx_len,
            )
        
        # Show tool availability warnings if any tools are disabled
        self._show_tool_availability_warnings()
        
        self.console.print()
    
    def _try_attach_clipboard_image(self) -> bool:
        """Check clipboard for an image and attach it if found.

        Saves the image to ~/.hermes/images/ and appends the path to
        ``_attached_images``.  Returns True if an image was attached.
        """
        from hermes_cli.clipboard import save_clipboard_image

        img_dir = Path.home() / ".hermes" / "images"
        self._image_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = img_dir / f"clip_{ts}_{self._image_counter}.png"

        if save_clipboard_image(img_path):
            self._attached_images.append(img_path)
            return True
        self._image_counter -= 1
        return False

    def _handle_paste_command(self):
        """Handle /paste — explicitly check clipboard for an image.

        This is the reliable fallback for terminals where BracketedPaste
        doesn't fire for image-only clipboard content (e.g., VSCode terminal,
        Windows Terminal with WSL2).
        """
        from hermes_cli.clipboard import has_clipboard_image
        if has_clipboard_image():
            if self._try_attach_clipboard_image():
                n = len(self._attached_images)
                _cprint(f"  📎 Image #{n} attached from clipboard")
            else:
                _cprint(f"  {_DIM}(>_<) Clipboard has an image but extraction failed{_RST}")
        else:
            _cprint(f"  {_DIM}(._.) No image found in clipboard{_RST}")

    def _build_multimodal_content(self, text: str, images: list) -> list:
        """Convert text + image paths into OpenAI vision multimodal content.

        Returns a list of content parts suitable for the ``content`` field
        of a ``user`` message.
        """
        import base64 as _b64

        content_parts = []
        text_part = text if isinstance(text, str) and text else "What do you see in this image?"
        content_parts.append({"type": "text", "text": text_part})

        _MIME = {
            "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "webp": "image/webp",
        }
        for img_path in images:
            if img_path.exists():
                data = _b64.b64encode(img_path.read_bytes()).decode()
                ext = img_path.suffix.lower().lstrip(".")
                mime = _MIME.get(ext, "image/png")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"}
                })
        return content_parts

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

        provider_info = f" [dim #B8860B]·[/] [dim]provider: {self.provider}[/]"
        if self._provider_source:
            provider_info += f" [dim #B8860B]·[/] [dim]auth: {self._provider_source}[/]"

        self.console.print(
            f"  {api_indicator} [#FFBF00]{model_short}[/] "
            f"[dim #B8860B]·[/] [bold cyan]{tool_count} tools[/]"
            f"{toolsets_info}{provider_info}"
        )
    
    def show_help(self):
        """Display help information."""
        _cprint(f"\n{_BOLD}+{'-' * 50}+{_RST}")
        _cprint(f"{_BOLD}|{' ' * 14}(^_^)? Available Commands{' ' * 10}|{_RST}")
        _cprint(f"{_BOLD}+{'-' * 50}+{_RST}\n")
        
        for cmd, desc in COMMANDS.items():
            _cprint(f"  {_GOLD}{cmd:<15}{_RST} {_DIM}-{_RST} {desc}")
        
        if _skill_commands:
            _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} ({len(_skill_commands)} installed):")
            for cmd, info in sorted(_skill_commands.items()):
                _cprint(f"  {_GOLD}{cmd:<22}{_RST} {_DIM}-{_RST} {info['description']}")

        _cprint(f"\n  {_DIM}Tip: Just type your message to chat with Hermes!{_RST}")
        _cprint(f"  {_DIM}Multi-line: Alt+Enter for a new line{_RST}")
        _cprint(f"  {_DIM}Paste image: Alt+V (or /paste){_RST}\n")
    
    def show_tools(self):
        """Display available tools with kawaii ASCII art."""
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
        
        if not tools:
            print("(;_;) No tools available")
            return
        
        # Header
        print()
        title = "(^_^)/ Available Tools"
        width = 78
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        print()
        
        # Group tools by toolset
        toolsets = {}
        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            toolset = get_toolset_for_tool(name) or "unknown"
            if toolset not in toolsets:
                toolsets[toolset] = []
            desc = tool["function"].get("description", "")
            # First sentence: split on ". " (period+space) to avoid breaking on "e.g." or "v2.0"
            desc = desc.split("\n")[0]
            if ". " in desc:
                desc = desc[:desc.index(". ") + 1]
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
        title = "(^_^)b Available Toolsets"
        width = 58
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        print()
        
        for name in sorted(all_toolsets.keys()):
            info = get_toolset_info(name)
            if info:
                tool_count = info["tool_count"]
                desc = info["description"]
                
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
        
        user_config_path = Path.home() / '.hermes' / 'config.yaml'
        project_config_path = Path(__file__).parent / 'cli-config.yaml'
        if user_config_path.exists():
            config_path = user_config_path
        else:
            config_path = project_config_path
        config_status = "(loaded)" if config_path.exists() else "(not found)"
        
        api_key_display = '********' + self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else 'Not set!'
        
        print()
        title = "(^_^) Configuration"
        width = 50
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
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
        print(f"  Config File: {config_path} {config_status}")
        print()
    
    def _handle_provider_command(self, cmd: str):
        """Switch provider, model, and API key mid-session."""
        parts = cmd.split(maxsplit=1)
        arg = parts[1].strip().lower() if len(parts) > 1 else ""

        if not arg:
            print()
            print(f"  Current provider config:")
            print(f"    Model:    {self.model}")
            print(f"    Base URL: {self.base_url}")
            api = '********' + self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else 'Not set'
            print(f"    API Key:  {api}")
            print()
            print("  Usage:")
            print("    /provider openrouter    Switch to OpenRouter (prompts for API key)")
            print("    /provider local         Switch to local Qwen3.5-9B")
            print("    /provider custom        Set custom endpoint + key")
            print()
            return

        try:
            if arg == "openrouter":
                key = input("  OpenRouter API key: ").strip()
                if not key:
                    print("  Cancelled.")
                    return
                model = input(f"  Model [{self.model}]: ").strip() or self.model
                self.api_key = key
                self.base_url = "https://openrouter.ai/api/v1"
                self.model = model
                self.agent = None
                save_config_value("model.default", model)
                from hermes_cli.config import save_env_value
                save_env_value("OPENROUTER_API_KEY", key)
                save_env_value("OPENAI_BASE_URL", self.base_url)
                print(f"  Switched to OpenRouter: {model}")

            elif arg == "local":
                self.api_key = "local"
                self.base_url = "http://127.0.0.1:8800/v1"
                self.model = "local/qwen3.5-9b"
                self.agent = None
                save_config_value("model.default", self.model)
                from hermes_cli.config import save_env_value
                save_env_value("OPENAI_BASE_URL", self.base_url)
                save_env_value("OPENAI_API_KEY", "local")
                print(f"  Switched to local Qwen3.5-9B (port 8800)")

            elif arg == "custom":
                url = input("  Base URL: ").strip()
                if not url:
                    print("  Cancelled.")
                    return
                key = input("  API key (optional): ").strip() or "none"
                model = input(f"  Model name [{self.model}]: ").strip() or self.model
                self.api_key = key
                self.base_url = url.rstrip("/")
                self.model = model
                self.agent = None
                save_config_value("model.default", model)
                from hermes_cli.config import save_env_value
                save_env_value("OPENAI_BASE_URL", self.base_url)
                save_env_value("OPENAI_API_KEY", key)
                print(f"  Switched to {model} at {self.base_url}")

            else:
                print(f"  Unknown provider: {arg}")
                print("  Options: openrouter, local, custom")
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")

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
            content = msg.get("content") or ""
            
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
        if self.agent and self.conversation_history:
            try:
                self.agent.flush_memories(self.conversation_history)
            except Exception:
                pass
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
                print(f"  {name:<12} - \"{prompt}\"")
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
        handle_skills_slash(cmd, ChatConsole())

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
            # Flush memories before clearing
            if self.agent and self.conversation_history:
                try:
                    self.agent.flush_memories(self.conversation_history)
                except Exception:
                    pass
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
            # Shortcut aliases for quick model switching
            # Maps alias -> (anthropic_direct_id, openrouter_id)
            _CLAUDE_MODELS = {
                "opus": ("claude-opus-4-6", "anthropic/claude-opus-4"),
                "sonnet": ("claude-sonnet-4-6", "anthropic/claude-sonnet-4"),
                "haiku": ("claude-haiku-4-5", "anthropic/claude-haiku-4"),
            }
            _MODEL_ALIASES = {
                "qwen": "local/qwen3.5-9b",
            }
            parts = cmd_original.split(maxsplit=1)
            if len(parts) > 1:
                arg = parts[1].strip()
                arg_lower = arg.lower()

                # Resolve Claude shortcuts based on available API keys
                claude_alias = _CLAUDE_MODELS.get(arg_lower)
                if claude_alias:
                    ant_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
                    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
                    direct_id, openrouter_id = claude_alias
                    if ant_key:
                        new_model = direct_id
                        self.requested_provider = "anthropic"
                        self.api_key = ant_key
                        self.base_url = ""
                        self._explicit_base_url = ""
                    elif or_key:
                        new_model = openrouter_id
                        self.requested_provider = "openrouter"
                        self.api_key = or_key
                        self.base_url = "https://openrouter.ai/api/v1"
                        self._explicit_base_url = self.base_url
                    else:
                        print("  Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY first")
                        return True
                elif arg_lower in _MODEL_ALIASES:
                    new_model = _MODEL_ALIASES[arg_lower]
                else:
                    new_model = arg

                self.model = new_model
                self.agent = None  # Force re-init
                if new_model.startswith("local/"):
                    self.requested_provider = "local"
                    self.api_key = "local"
                    self.base_url = "http://127.0.0.1:8800/v1"
                    self._explicit_base_url = self.base_url
                elif not claude_alias:
                    # Non-alias, non-local model — try OpenRouter
                    if new_model.startswith("anthropic/") or new_model.startswith("claude"):
                        ant_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
                        or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
                        if ant_key:
                            self.requested_provider = "anthropic"
                            self.api_key = ant_key
                            self.base_url = ""
                            self._explicit_base_url = ""
                        elif or_key:
                            self.requested_provider = "openrouter"
                            self.api_key = or_key
                            self.base_url = "https://openrouter.ai/api/v1"
                            self._explicit_base_url = self.base_url
                    else:
                        or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
                        if or_key:
                            self.requested_provider = "openrouter"
                            self.api_key = or_key
                            self.base_url = "https://openrouter.ai/api/v1"
                            self._explicit_base_url = self.base_url
                save_config_value("model.default", new_model)
                provider_label = self.requested_provider
                if provider_label == "anthropic":
                    provider_label = "anthropic (direct)"
                print(f"  Switched to: {new_model} ({provider_label})")
            else:
                print(f"Current model: {self.model}")
                print("  Shortcuts: /model opus | sonnet | haiku | qwen")
                print("  Or: /model <full-model-name>")
        elif cmd_lower.startswith("/provider"):
            self._handle_provider_command(cmd_original)
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
        elif cmd_lower == "/verbose":
            self._toggle_verbose()
        elif cmd_lower == "/thinkoff":
            self.show_thinking = False
            print("Thinking blocks hidden. Use /thinkon to show them again.")
        elif cmd_lower == "/thinkon":
            self.show_thinking = True
            print("Thinking blocks visible.")
        elif cmd_lower == "/compress":
            self._manual_compress()
        elif cmd_lower == "/usage":
            self._show_usage()
        elif cmd_lower == "/paste":
            self._handle_paste_command()
        elif cmd_lower == "/reload-mcp":
            self._reload_mcp()
        elif cmd_lower == "/copycode":
            self._copycode()
        elif cmd_lower == "/context":
            self._show_context_gauge()
        elif cmd_lower.startswith("/swarm"):
            self._handle_swarm_command(cmd_original)
        elif cmd_lower == "/jobs":
            self._show_jobs()
        elif cmd_lower.startswith("/fg"):
            self._handle_fg_command(cmd_original)
        else:
            # Check for skill slash commands (/gif-search, /axolotl, etc.)
            base_cmd = cmd_lower.split()[0]
            if base_cmd in _skill_commands:
                user_instruction = cmd_original[len(base_cmd):].strip()
                msg = build_skill_invocation_message(base_cmd, user_instruction)
                if msg:
                    skill_name = _skill_commands[base_cmd]["name"]
                    print(f"\n⚡ Loading skill: {skill_name}")
                    if hasattr(self, '_pending_input'):
                        self._pending_input.put(msg)
                else:
                    self.console.print(f"[bold red]Failed to load skill for {base_cmd}[/]")
            else:
                self.console.print(f"[bold red]Unknown command: {cmd_lower}[/]")
                self.console.print("[dim #B8860B]Type /help for available commands[/]")
        
        return True
    
    def _toggle_verbose(self):
        """Cycle tool progress mode: off → new → all → verbose → off."""
        cycle = ["off", "new", "all", "verbose"]
        try:
            idx = cycle.index(self.tool_progress_mode)
        except ValueError:
            idx = 2  # default to "all"
        self.tool_progress_mode = cycle[(idx + 1) % len(cycle)]
        self.verbose = self.tool_progress_mode == "verbose"

        if self.agent:
            self.agent.verbose_logging = self.verbose
            self.agent.quiet_mode = not self.verbose

        labels = {
            "off": "[dim]Tool progress: OFF[/] — silent mode, just the final response.",
            "new": "[yellow]Tool progress: NEW[/] — show each new tool (skip repeats).",
            "all": "[green]Tool progress: ALL[/] — show every tool call.",
            "verbose": "[bold green]Tool progress: VERBOSE[/] — full args, results, and debug logs.",
        }
        self.console.print(labels.get(self.tool_progress_mode, ""))

    def _manual_compress(self):
        """Manually trigger context compression on the current conversation."""
        if not self.conversation_history or len(self.conversation_history) < 4:
            print("(._.) Not enough conversation to compress (need at least 4 messages).")
            return

        if not self.agent:
            print("(._.) No active agent -- send a message first.")
            return

        if not self.agent.compression_enabled:
            print("(._.) Compression is disabled in config.")
            return

        original_count = len(self.conversation_history)
        try:
            from agent.model_metadata import estimate_messages_tokens_rough
            approx_tokens = estimate_messages_tokens_rough(self.conversation_history)
            print(f"🗜️  Compressing {original_count} messages (~{approx_tokens:,} tokens)...")

            compressed, new_system = self.agent._compress_context(
                self.conversation_history,
                self.agent._cached_system_prompt or "",
                approx_tokens=approx_tokens,
            )
            self.conversation_history = compressed
            new_count = len(self.conversation_history)
            new_tokens = estimate_messages_tokens_rough(self.conversation_history)
            print(
                f"  ✅ Compressed: {original_count} → {new_count} messages "
                f"(~{approx_tokens:,} → ~{new_tokens:,} tokens)"
            )
        except Exception as e:
            print(f"  ❌ Compression failed: {e}")

    def _show_usage(self):
        """Show cumulative token usage for the current session."""
        if not self.agent:
            print("(._.) No active agent -- send a message first.")
            return

        agent = self.agent
        prompt = agent.session_prompt_tokens
        completion = agent.session_completion_tokens
        total = agent.session_total_tokens
        calls = agent.session_api_calls

        if calls == 0:
            print("(._.) No API calls made yet in this session.")
            return

        # Current context window state
        compressor = agent.context_compressor
        last_prompt = compressor.last_prompt_tokens
        ctx_len = compressor.context_length
        pct = (last_prompt / ctx_len * 100) if ctx_len else 0
        compressions = compressor.compression_count

        msg_count = len(self.conversation_history)

        print(f"  📊 Session Token Usage")
        print(f"  {'─' * 40}")
        print(f"  Prompt tokens (input):     {prompt:>10,}")
        print(f"  Completion tokens (output): {completion:>9,}")
        print(f"  Total tokens:              {total:>10,}")
        print(f"  API calls:                 {calls:>10,}")
        print(f"  {'─' * 40}")
        print(f"  Current context:  {last_prompt:,} / {ctx_len:,} ({pct:.0f}%)")
        print(f"  Messages:         {msg_count}")
        print(f"  Compressions:     {compressions}")

        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            for noisy in ('openai', 'openai._base_client', 'httpx', 'httpcore', 'asyncio', 'hpack', 'grpc', 'modal'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.INFO)
            for quiet_logger in ('tools', 'minisweagent', 'run_agent', 'trajectory_compressor', 'cron', 'hermes_cli'):
                logging.getLogger(quiet_logger).setLevel(logging.ERROR)

    def _show_jobs(self):
        """Show background agent tasks."""
        import time as _time
        if not self._background_jobs:
            print("  No background jobs.")
            return
        for job in self._background_jobs:
            done = job["done"].is_set()
            elapsed = _time.time() - job["started"]
            status = "done" if done else "running"
            icon = "\u2705" if done else "\u23f3"
            print(f"  {icon} [{job['id']}] {status} ({elapsed:.0f}s) \u2014 {job['prompt']}")

    def _handle_fg_command(self, command: str):
        """Bring a background job result to foreground."""
        parts = command.split()
        if len(parts) < 2:
            if not self._background_jobs:
                print("  No background jobs. Usage: /fg <id>")
                return
            job = self._background_jobs[-1]
        else:
            try:
                job_id = int(parts[1])
            except ValueError:
                print(f"  Invalid job ID: {parts[1]}")
                return
            job = next((j for j in self._background_jobs if j["id"] == job_id), None)
            if not job:
                print(f"  Job {job_id} not found.")
                return

        if not job["done"].is_set():
            print(f"  Job [{job['id']}] is still running. Use /jobs to check status.")
            return

        # Job is done — remove from list
        self._background_jobs = [j for j in self._background_jobs if j["id"] != job["id"]]
        if job["result"][0] and isinstance(job["result"][0], Exception):
            print(f"  Job [{job['id']}] failed: {job['result'][0]}")
        else:
            print(f"  Job [{job['id']}] completed: {job['prompt']}")

    def _show_context_gauge(self):
        """Show ASCII context window gauge for the current model."""
        # Model context limits (only Qwen3.5-9B for now)
        MODEL_CONTEXTS = {
            "local/qwen3.5-9b": 32768,
            "qwen/qwen3.5-9b": 262144,
        }

        # Try to get context length from compressor first, then hardcoded
        ctx_len = 0
        used_tokens = 0

        if self.agent and hasattr(self.agent, 'context_compressor'):
            compressor = self.agent.context_compressor
            ctx_len = compressor.context_length
            used_tokens = compressor.last_prompt_tokens

        if not ctx_len:
            # Fuzzy match model name
            for key, length in MODEL_CONTEXTS.items():
                if key in self.model or self.model in key:
                    ctx_len = length
                    break

        if not ctx_len:
            print(f"  (._.) Context tracking not available for model: {self.model}")
            print(f"  Only local/qwen3.5-9b is supported right now.")
            return

        # Estimate used tokens from conversation history if compressor has no data
        if not used_tokens and self.conversation_history:
            from agent.model_metadata import estimate_messages_tokens_rough
            used_tokens = estimate_messages_tokens_rough(self.conversation_history)

        remaining = max(0, ctx_len - used_tokens)
        pct_used = (used_tokens / ctx_len * 100) if ctx_len else 0
        pct_free = 100.0 - pct_used

        # ASCII bar — 50 chars wide
        bar_width = 50
        filled = int(bar_width * pct_used / 100)
        empty = bar_width - filled

        # Color thresholds
        if pct_used < 50:
            color = "green"
            status = "plenty of room"
        elif pct_used < 75:
            color = "yellow"
            status = "getting warm"
        elif pct_used < 90:
            color = "#FF8C00"
            status = "running low"
        else:
            color = "red"
            status = "CRITICAL"

        # Build the gauge
        bar_filled = "\u2588" * filled
        bar_empty = "\u2591" * empty

        print()
        print(f"  Context Window \u2014 {self.model}")
        print(f"  {'\u2500' * 56}")
        self.console.print(f"  [{color}]{bar_filled}[/][dim]{bar_empty}[/] {pct_used:.1f}%")
        print()
        print(f"  Used:      {used_tokens:>8,} tokens")
        print(f"  Remaining: {remaining:>8,} tokens")
        print(f"  Capacity:  {ctx_len:>8,} tokens")
        print(f"  Status:    {status}")
        print()

    def _reload_mcp(self):
        """Reload MCP servers: disconnect all, re-read config.yaml, reconnect.

        After reconnecting, refreshes the agent's tool list so the model
        sees the updated tools on the next turn.
        """
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _load_mcp_config, _servers, _lock

            # Capture old server names
            with _lock:
                old_servers = set(_servers.keys())

            print("🔄 Reloading MCP servers...")

            # Shutdown existing connections
            shutdown_mcp_servers()

            # Reconnect (reads config.yaml fresh)
            new_tools = discover_mcp_tools()

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            if reconnected:
                print(f"  ♻️  Reconnected: {', '.join(sorted(reconnected))}")
            if added:
                print(f"  ➕ Added: {', '.join(sorted(added))}")
            if removed:
                print(f"  ➖ Removed: {', '.join(sorted(removed))}")
            if not connected_servers:
                print("  No MCP servers connected.")
            else:
                print(f"  🔧 {len(new_tools)} tool(s) available from {len(connected_servers)} server(s)")

            # Refresh the agent's tool list so the model can call new tools
            if self.agent is not None:
                from model_tools import get_tool_definitions
                self.agent.tools = get_tool_definitions(
                    enabled_toolsets=self.agent.enabled_toolsets
                    if hasattr(self.agent, "enabled_toolsets") else None,
                    quiet_mode=True,
                )
                self.agent.valid_tool_names = {
                    tool["function"]["name"] for tool in self.agent.tools
                } if self.agent.tools else set()

            # Inject a message at the END of conversation history so the
            # model knows tools changed.  Appended after all existing
            # messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            self.conversation_history.append({
                "role": "user",
                "content": f"[SYSTEM: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            })

            # Persist session immediately so the session log reflects the
            # updated tools list (self.agent.tools was refreshed above).
            if self.agent is not None:
                try:
                    self.agent._persist_session(
                        self.conversation_history,
                        self.conversation_history,
                    )
                except Exception:
                    pass  # Best-effort

            print(f"  ✅ Agent updated — {len(self.agent.tools if self.agent else [])} tool(s) available")

        except Exception as e:
            print(f"  ❌ MCP reload failed: {e}")

    def _copycode(self):
        """Import Claude Code skills and commands into Hermes skills directory."""
        import shutil
        import re

        claude_base = Path.home() / ".claude"
        skills_dir = Path.home() / ".hermes" / "skills" / "imported-from-claude"
        skills_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        skipped = 0

        # --- 1. Copy SKILL.md directories (skills already in hermes format) ---
        for skill_md in claude_base.rglob("SKILL.md"):
            skill_src_dir = skill_md.parent
            name = skill_src_dir.name  # e.g. "frontend-design", "skill-creator"
            dest_dir = skills_dir / name

            if dest_dir.exists():
                skipped += 1
                continue

            # Copy the whole skill directory (SKILL.md + references/)
            shutil.copytree(skill_src_dir, dest_dir)
            imported += 1
            print(f"  + {name} (skill)")

        # --- 2. Convert command .md files to SKILL.md format ---
        for cmd_md in claude_base.rglob("*.md"):
            # Only grab files inside a "commands" directory
            if "commands" not in cmd_md.parts:
                continue
            if cmd_md.name == "SKILL.md":
                continue

            name = cmd_md.stem
            dest_dir = skills_dir / name

            if dest_dir.exists():
                skipped += 1
                continue

            content = cmd_md.read_text(encoding="utf-8")

            # Parse frontmatter
            fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
            if fm_match:
                fm_text = fm_match.group(1)
                body = content[fm_match.end():]
                desc_match = re.search(r'^description:\s*(.+)$', fm_text, re.MULTILINE)
                desc = desc_match.group(1).strip() if desc_match else f"Imported command: {name}"
            else:
                body = content
                desc = f"Imported command: {name}"

            # Figure out plugin name for attribution
            parts = cmd_md.parts
            plugin_name = name
            if "plugins" in parts:
                idx = list(parts).index("plugins")
                if idx + 2 < len(parts):
                    plugin_name = parts[idx + 2]

            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {desc}
version: 1.0.0
author: Imported from Claude Code ({plugin_name})
license: MIT
metadata:
  hermes:
    tags: [imported, claude-code, command]
    source: "{cmd_md}"
---

{body}""", encoding="utf-8")
            imported += 1
            print(f"  + {name} (command)")

        # --- 3. Pull from Anthropic official skills repo (github.com/anthropics/skills) ---
        print("\nChecking Anthropic skills repo...")
        try:
            import subprocess as _sp
            result = _sp.run(
                ["gh", "api", "repos/anthropics/skills/contents/skills", "--jq", ".[].name"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                remote_skills = result.stdout.strip().splitlines()
                for skill_name in remote_skills:
                    dest_dir = skills_dir / skill_name
                    if dest_dir.exists():
                        skipped += 1
                        continue
                    # Get file listing for this skill
                    files_result = _sp.run(
                        ["gh", "api", f"repos/anthropics/skills/contents/skills/{skill_name}",
                         "--jq", r'.[] | "\(.name)\t\(.download_url)"'],
                        capture_output=True, text=True, timeout=15,
                    )
                    if files_result.returncode != 0:
                        continue
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for line in files_result.stdout.strip().splitlines():
                        parts = line.split("\t", 1)
                        if len(parts) != 2:
                            continue
                        fname, url = parts
                        if url == "null" or not url:
                            continue
                        dl = _sp.run(["curl", "-sL", url], capture_output=True, timeout=30)
                        if dl.returncode == 0 and dl.stdout:
                            (dest_dir / fname).write_bytes(dl.stdout)
                    if (dest_dir / "SKILL.md").exists():
                        imported += 1
                        print(f"  + {skill_name} (anthropic)")
                    else:
                        shutil.rmtree(dest_dir, ignore_errors=True)
            else:
                print("  (gh CLI not available or API error — skipping remote skills)")
        except Exception as e:
            print(f"  (skipping remote: {e})")

        print(f"\nImported {imported} to {skills_dir}")
        if skipped:
            print(f"  ({skipped} already existed, skipped)")

    def _handle_swarm_command(self, command: str):
        """Handle /swarm <goal> — run a parallel agent swarm."""
        parts = command.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print("  Usage: /swarm <goal>")
            print("  Example: /swarm refactor the auth module and add tests")
            return

        goal = parts[1].strip()

        try:
            from swarm.orchestrator import SwarmOrchestrator
            from swarm.planner import TaskPlanner
            from swarm.types import SwarmConfig
        except ImportError:
            self.console.print("[bold red]Swarm module not available.[/]")
            return

        print("  Decomposing goal...")

        # Mark agent as running so typed messages queue for later
        self._agent_running = True

        # Snapshot files before swarm for change detection
        import time as _time
        _swarm_start = _time.time()

        try:
            planner = TaskPlanner()
            tasks = planner.decompose(goal)
            swarm_model = self.model
            if "opus" in swarm_model or "sonnet" in swarm_model:
                swarm_model = "claude-haiku-4-5"
            for i, t in enumerate(tasks):
                t.model = swarm_model
                t.name = f"Agent {i + 1}"

            if not tasks:
                print("  Planner returned no tasks.")
                return

            id_to_name = {t.id: t.name for t in tasks}
            plan = [
                {"name": t.name, "prompt": t.prompt, "role": t.role,
                 "deps": [id_to_name.get(d, d) for d in t.deps],
                 "model": t.model or None}
                for t in tasks
            ]

            config = SwarmConfig()
            orch = SwarmOrchestrator(config)
            orch.add_plan(plan)

            print(f"  Swarm ({len(plan)} agents, model={swarm_model})")

            def _update_display(data):
                self._swarm_display = data

            summary = orch.run_with_display(display_callback=_update_display)

            # Final results
            sched_status = summary.get("tasks", {})
            completed = sched_status.get("completed", 0)
            failed = sched_status.get("failed", 0)
            total = summary.get("total_tasks", len(plan))

            print(f"\n  {'─' * 50}")
            if failed == 0:
                print(f"  {completed}/{total} agents completed")
            else:
                print(f"  {completed}/{total} completed, {failed} failed")

            # Show each agent's output
            import re as _re_files
            _ORANGE = "\033[38;5;208m"
            _YELLOW = "\033[33m"
            _GREEN = "\033[32m"
            _RED = "\033[31m"
            _DIM2 = "\033[2m"
            _RST2 = "\033[0m"
            _BOLD2 = "\033[1m"

            def _highlight_files(line):
                """Highlight filenames in yellow."""
                return _re_files.sub(
                    r'{0}\g<0>{1}'.format(_YELLOW, _RST2),
                    line,
                    # Match common file patterns
                ) if False else _re_files.sub(
                    r'(\b\w+\.(?:py|txt|md|json|yaml|yml|js|ts|html|css|toml|cfg|ini|sh|rs)\b)',
                    f'{_YELLOW}\\1{_RST2}',
                    line,
                )

            for tid, task in orch._scheduler._tasks.items():
                r = task.result
                if not r:
                    continue
                out = getattr(r, 'output', r) if not isinstance(r, dict) else r
                if isinstance(out, dict):
                    text = out.get('output', '') or out.get('final_response', '') or ''
                elif isinstance(out, str):
                    text = out
                else:
                    text = str(out) if out else ''
                text = (text or '').strip()
                duration = getattr(r, 'duration_seconds', 0)
                status = "done" if task.state.value == "completed" else task.state.value
                if text:
                    icon = f"{_GREEN}●{_RST2}" if status == 'done' else f"{_RED}✗{_RST2}"
                    _cprint(f"\n  {icon} {_ORANGE}{_BOLD2}{task.name}{_RST2} {_DIM2}[{task.role}, {duration:.0f}s]{_RST2}")
                    out_lines = text.split('\n')
                    for line in out_lines[:8]:
                        colored = _highlight_files(line[:120])
                        _cprint(f"    {colored}")
                    if len(out_lines) > 8:
                        _cprint(f"    {_DIM2}... +{len(out_lines) - 8} lines{_RST2}")

            # Show files created/modified during swarm
            import os as _os
            _cwd = _os.getcwd()
            _new_files = []
            _mod_files = []
            for _root, _dirs, _fnames in _os.walk(_cwd):
                # Skip hidden dirs, __pycache__, .git, .venv, node_modules
                _dirs[:] = [d for d in _dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.venv')]
                for _fn in _fnames:
                    _fp = _os.path.join(_root, _fn)
                    try:
                        st = _os.stat(_fp)
                        if st.st_mtime >= _swarm_start:
                            rel = _os.path.relpath(_fp, _cwd)
                            if st.st_ctime >= _swarm_start:
                                _new_files.append(rel)
                            else:
                                _mod_files.append(rel)
                    except OSError:
                        pass

            if _new_files or _mod_files:
                _cprint(f"\n  {_DIM2}{'─' * 50}{_RST2}")
                _cprint(f"  {_BOLD2}Files changed:{_RST2}")
                for f in sorted(_new_files):
                    _cprint(f"    {_GREEN}+{_RST2} {_YELLOW}{f}{_RST2}")
                for f in sorted(_mod_files):
                    _cprint(f"    {_ORANGE}~{_RST2} {_YELLOW}{f}{_RST2}")

            # Final completion message
            _cprint(f"\n  {_GREEN}✓{_RST2} {_BOLD2}Swarm complete{_RST2} {_DIM2}— {completed} agents finished in {sum(getattr(t.result, 'duration_seconds', 0) for t in orch._scheduler._tasks.values() if t.result):.0f}s{_RST2}\n")

        except Exception as e:
            print(f"Swarm error: {e}")
        finally:
            self._agent_running = False
            self._swarm_display = None
            self._queued_messages.clear()
            # Re-queue messages typed during swarm so they process next
            while hasattr(self, '_interrupt_queue') and not self._interrupt_queue.empty():
                try:
                    msg = self._interrupt_queue.get_nowait()
                    if msg:
                        self._pending_input.put(msg)
                except queue.Empty:
                    break

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
        self._invalidate()

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
                self._invalidate()

        # Timed out — tear down the UI and let the agent decide
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = 0
        self._invalidate()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")
        return (
            "The user did not provide a response within the time limit. "
            "Use your best judgement to make the choice and proceed."
        )

    def _sudo_password_callback(self) -> str:
        """
        Prompt for sudo password through the prompt_toolkit UI.
        
        Called from the agent thread when a sudo command is encountered.
        Uses the same clarify-style mechanism: sets UI state, waits on a
        queue for the user's response via the Enter key binding.
        """
        import time as _time

        timeout = 45
        response_queue = queue.Queue()

        self._sudo_state = {
            "response_queue": response_queue,
        }
        self._sudo_deadline = _time.monotonic() + timeout

        self._invalidate()

        while True:
            try:
                result = response_queue.get(timeout=1)
                self._sudo_state = None
                self._sudo_deadline = 0
                self._invalidate()
                if result:
                    _cprint(f"\n{_DIM}  ✓ Password received (cached for session){_RST}")
                else:
                    _cprint(f"\n{_DIM}  ⏭ Skipped{_RST}")
                return result
            except queue.Empty:
                remaining = self._sudo_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                self._invalidate()

        self._sudo_state = None
        self._sudo_deadline = 0
        self._invalidate()
        _cprint(f"\n{_DIM}  ⏱ Timeout — continuing without sudo{_RST}")
        return ""

    def _approval_callback(self, command: str, description: str) -> str:
        """
        Prompt for dangerous command approval through the prompt_toolkit UI.
        
        Called from the agent thread. Shows a selection UI similar to clarify
        with choices: once / session / always / deny.
        """
        import time as _time

        timeout = 60
        response_queue = queue.Queue()
        choices = ["once", "session", "always", "deny"]

        self._approval_state = {
            "command": command,
            "description": description,
            "choices": choices,
            "selected": 0,
            "response_queue": response_queue,
        }
        self._approval_deadline = _time.monotonic() + timeout

        self._invalidate()

        while True:
            try:
                result = response_queue.get(timeout=1)
                self._approval_state = None
                self._approval_deadline = 0
                self._invalidate()
                return result
            except queue.Empty:
                remaining = self._approval_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                self._invalidate()

        self._approval_state = None
        self._approval_deadline = 0
        self._invalidate()
        _cprint(f"\n{_DIM}  Timeout — denying command{_RST}")
        return "deny"

    def chat(self, message, images: list = None) -> Optional[str]:
        """
        Send a message to the agent and get a response.
        
        Handles streaming output, interrupt detection (user typing while agent
        is working), and re-queueing of interrupted messages.
        
        Uses a dedicated _interrupt_queue (separate from _pending_input) to avoid
        race conditions between the process_loop and interrupt monitoring. Messages
        typed while the agent is running go to _interrupt_queue; messages typed while
        idle go to _pending_input.
        
        Args:
            message: The user's message (str or multimodal content list)
            images: Optional list of Path objects for attached images
            
        Returns:
            The agent's response, or None on error
        """
        # Refresh provider credentials if needed (handles key rotation transparently)
        if not self._ensure_runtime_credentials():
            return None

        # Initialize agent if needed
        if not self._init_agent():
            return None
        
        # Convert attached images to OpenAI vision multimodal content
        if images:
            message = self._build_multimodal_content(
                message if isinstance(message, str) else "", images
            )
            for img_path in images:
                if img_path.exists():
                    _cprint(f"  {_DIM}📎 attached {img_path.name} ({img_path.stat().st_size // 1024}KB){_RST}")

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        w = _term_width()
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
                    task_id=self.session_id,
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
                w = _term_width()
                label = " ⚕ Hermes "
                fill = w - 2 - len(label)
                top = f"{_GOLD}──{label}{'─' * max(fill, 0)}{_RST}"
                bot = f"{_GOLD}{'─' * w}{_RST}"

                # Style or strip <think> blocks based on user preference
                import re as _re
                if self.show_thinking:
                    styled_response = _format_think_blocks(response)
                else:
                    # Strip closed think blocks
                    styled_response = _re.sub(r'<think>.*?</think>\s*', '', response, flags=_re.DOTALL)
                    # Strip unclosed think blocks (model ran out of tokens mid-think)
                    styled_response = _re.sub(r'<think>.*', '', styled_response, flags=_re.DOTALL)
                    # Strip orphaned </think> (opening tag in separate field)
                    styled_response = _re.sub(r'^.*?</think>\s*', '', styled_response, flags=_re.DOTALL)
                    styled_response = styled_response.strip()

                # Always strip <tool_call> blocks from display — these are
                # machine-readable tags that the adapter processes internally.
                styled_response = _re.sub(
                    r'<tool_call>.*?</tool_call>\s*', '', styled_response, flags=_re.DOTALL
                )
                # Strip truncated/unclosed tool_call tags too
                styled_response = _re.sub(
                    r'<tool_call>.*', '', styled_response, flags=_re.DOTALL
                )
                styled_response = styled_response.strip()

                # Render box + response as a single _cprint call so
                # nothing can interleave between the box borders.
                _cprint(f"\n{top}\n{styled_response}\n\n{bot}")
            
            # Re-queue interrupt message for process_loop
            if pending_message and hasattr(self, '_pending_input'):
                # Drain any extra interrupt messages
                extras = []
                while not self._interrupt_queue.empty():
                    try:
                        extra = self._interrupt_queue.get_nowait()
                        if extra:
                            extras.append(extra)
                    except queue.Empty:
                        break
                # If pending_message is a tuple (text, images), preserve it as-is
                if isinstance(pending_message, tuple) or not extras:
                    display = pending_message[0] if isinstance(pending_message, tuple) else str(pending_message)
                    print(f"\n📨 Queued: '{str(display)[:50]}'")
                    self._pending_input.put(pending_message)
                else:
                    all_texts = [pending_message] + extras
                    combined = "\n".join(str(p) for p in all_texts)
                    print(f"\n📨 Queued: '{combined[:50]}{'...' if len(combined) > 50 else ''}'")
                    self._pending_input.put(combined)
                # Re-queue any remaining extras
                for ex in extras if isinstance(pending_message, tuple) else []:
                    self._pending_input.put(ex)
            
            return response
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def _print_exit_summary(self):
        """Print session resume info on exit, similar to Claude Code."""
        print()
        msg_count = len(self.conversation_history)
        if msg_count > 0:
            user_msgs = len([m for m in self.conversation_history if m.get("role") == "user"])
            tool_calls = len([m for m in self.conversation_history if m.get("role") == "tool" or m.get("tool_calls")])
            elapsed = datetime.now() - self.session_start
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"
            
            print(f"Resume this session with:")
            print(f"  hermes --resume {self.session_id}")
            print()
            print(f"Session:        {self.session_id}")
            print(f"Duration:       {duration_str}")
            print(f"Messages:       {msg_count} ({user_msgs} user, {tool_calls} tool calls)")
        else:
            print("Goodbye! ⚕")

    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        self.show_banner()
        self.console.print("[#FFF8DC]Welcome to Hermes Agent! Type your message or /help for commands.[/]")
        self.console.print()
        
        # State for async operation
        self._agent_running = False
        self._pending_input = queue.Queue()     # For normal input (commands + new queries)
        self._interrupt_queue = queue.Queue()   # For messages typed while agent is running
        self._queued_messages: list[str] = []   # Display list for queued-while-busy messages
        self._swarm_display: list[dict] | None = None  # Live swarm task states for widget
        self._should_exit = False
        self._last_ctrl_c_time = 0  # Track double Ctrl+C for force exit
        self._background_jobs = []  # list of {"id": int, "thread": Thread, "prompt": str, "started": float, "result": None}
        self._next_job_id = 1
        self._current_chat_thread = None  # reference to current chat thread for backgrounding
        self._background_requested = False  # flag set by Ctrl+B handler

        # Clarify tool state: interactive question/answer with the user.
        # When the agent calls the clarify tool, _clarify_state is set and
        # the prompt_toolkit UI switches to a selection mode.
        self._clarify_state = None      # dict with question, choices, selected, response_queue
        self._clarify_freetext = False  # True when user chose "Other" and is typing
        self._clarify_deadline = 0      # monotonic timestamp when the clarify times out

        # Sudo password prompt state (similar mechanism to clarify)
        self._sudo_state = None         # dict with response_queue when active
        self._sudo_deadline = 0

        # Dangerous command approval state (similar mechanism to clarify)
        self._approval_state = None     # dict with command, description, choices, selected, response_queue
        self._approval_deadline = 0

        # Clipboard image attachments (paste images into the CLI)
        self._attached_images: list[Path] = []
        self._image_counter = 0
        self._image_select_idx: int = -1  # -1 = not in image select mode

        # Register callbacks so terminal_tool prompts route through our UI
        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)
        
        # Key bindings for the input area
        kb = KeyBindings()
        
        @kb.add('enter')
        def handle_enter(event):
            """Handle Enter key - submit input.
            
            Routes to the correct queue based on active UI state:
            - Sudo password prompt: password goes to sudo response queue
            - Approval selection: selected choice goes to approval response queue
            - Clarify freetext mode: answer goes to the clarify response queue
            - Clarify choice mode: selected choice goes to the clarify response queue
            - Agent running: goes to _interrupt_queue (chat() monitors this)
            - Agent idle: goes to _pending_input (process_loop monitors this)
            Commands (starting with /) always go to _pending_input so they're
            handled as commands, not sent as interrupt text to the agent.
            """
            # --- Sudo password prompt: submit the typed password ---
            if self._sudo_state:
                text = event.app.current_buffer.text
                self._sudo_state["response_queue"].put(text)
                self._sudo_state = None
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # --- Approval selection: confirm the highlighted choice ---
            if self._approval_state:
                state = self._approval_state
                selected = state["selected"]
                choices = state["choices"]
                if 0 <= selected < len(choices):
                    state["response_queue"].put(choices[selected])
                self._approval_state = None
                event.app.invalidate()
                return

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
            has_images = bool(self._attached_images)
            if text or has_images:
                # Snapshot and clear attached images
                images = list(self._attached_images)
                self._attached_images.clear()
                self._image_select_idx = -1
                event.app.invalidate()
                # Bundle text + images as a tuple when images are present
                payload = (text, images) if images else text
                if self._agent_running and not (text and text.startswith("/")):
                    self._interrupt_queue.put(payload)
                    display_text = text or ""
                    if images:
                        img_labels = " ".join(f"[Image]" for _ in images)
                        display_text = f"{display_text}  {img_labels}".strip() if display_text else img_labels
                    self._queued_messages.append(display_text or "(empty)")
                    event.app.invalidate()
                else:
                    self._pending_input.put(payload)
                event.app.current_buffer.reset(append_to_history=True)
        
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

        # --- Dangerous command approval: arrow-key navigation ---

        @kb.add('up', filter=Condition(lambda: bool(self._approval_state)))
        def approval_up(event):
            if self._approval_state:
                self._approval_state["selected"] = max(0, self._approval_state["selected"] - 1)
                event.app.invalidate()

        @kb.add('down', filter=Condition(lambda: bool(self._approval_state)))
        def approval_down(event):
            if self._approval_state:
                max_idx = len(self._approval_state["choices"]) - 1
                self._approval_state["selected"] = min(max_idx, self._approval_state["selected"] + 1)
                event.app.invalidate()

        # --- History navigation: up/down browse history in normal input mode ---
        _history_navigating = [False]  # flag to suppress clipboard check on history nav
        # The TextArea is multiline, so by default up/down only move the cursor.
        # Buffer.auto_up/auto_down handle both: cursor movement when multi-line,
        # history browsing when on the first/last line (or single-line input).
        _normal_input = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._sudo_state
        )

        @kb.add('up', filter=_normal_input)
        def history_up(event):
            """Up arrow: image select if images attached, else history."""
            if self._attached_images and self._image_select_idx == -1:
                # Enter image select mode — select last image
                self._image_select_idx = len(self._attached_images) - 1
                event.app.invalidate()
                return
            if self._image_select_idx >= 0:
                # Navigate within image selection
                self._image_select_idx = max(0, self._image_select_idx - 1)
                event.app.invalidate()
                return
            _history_navigating[0] = True
            event.app.current_buffer.auto_up(count=event.arg)

        @kb.add('down', filter=_normal_input)
        def history_down(event):
            """Down arrow: navigate image select or history."""
            if self._image_select_idx >= 0:
                if self._image_select_idx < len(self._attached_images) - 1:
                    self._image_select_idx += 1
                else:
                    # Exit image select mode
                    self._image_select_idx = -1
                event.app.invalidate()
                return
            _history_navigating[0] = True
            event.app.current_buffer.auto_down(count=event.arg)

        @kb.add('delete', filter=Condition(lambda: self._image_select_idx >= 0))
        @kb.add('backspace', filter=Condition(lambda: self._image_select_idx >= 0))
        def delete_selected_image(event):
            """Delete/Backspace removes the selected image."""
            if 0 <= self._image_select_idx < len(self._attached_images):
                self._attached_images.pop(self._image_select_idx)
                if not self._attached_images:
                    self._image_select_idx = -1
                elif self._image_select_idx >= len(self._attached_images):
                    self._image_select_idx = len(self._attached_images) - 1
                event.app.invalidate()

        @kb.add('escape', filter=Condition(lambda: self._image_select_idx >= 0))
        def exit_image_select(event):
            """Escape exits image select mode."""
            self._image_select_idx = -1
            event.app.invalidate()

        @kb.add('c-c')
        def handle_ctrl_c(event):
            """Handle Ctrl+C - cancel interactive prompts, interrupt agent, or exit.
            
            Priority:
            1. Cancel active sudo/approval/clarify prompt
            2. Interrupt the running agent (first press)
            3. Force exit (second press within 2s, or when idle)
            """
            import time as _time
            now = _time.time()

            # Cancel sudo prompt
            if self._sudo_state:
                self._sudo_state["response_queue"].put("")
                self._sudo_state = None
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # Cancel approval prompt (deny)
            if self._approval_state:
                self._approval_state["response_queue"].put("deny")
                self._approval_state = None
                event.app.invalidate()
                return

            # Cancel clarify prompt
            if self._clarify_state:
                self._clarify_state["response_queue"].put(
                    "The user cancelled. Use your best judgement to proceed."
                )
                self._clarify_state = None
                self._clarify_freetext = False
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            if self._agent_running and self.agent:
                if now - self._last_ctrl_c_time < 2.0:
                    print("\n⚡ Force exiting...")
                    self._should_exit = True
                    event.app.exit()
                    return
                
                self._last_ctrl_c_time = now
                print("\n⚡ Interrupting agent... (press Ctrl+C again to force exit)")
                self.agent.interrupt()
            else:
                # If there's text or images, clear them (like bash).
                # If everything is already empty, exit.
                if event.app.current_buffer.text or self._attached_images:
                    event.app.current_buffer.reset()
                    self._attached_images.clear()
                    self._image_select_idx = -1
                    event.app.invalidate()
                else:
                    self._should_exit = True
                    event.app.exit()
        
        @kb.add('escape', eager=True)
        def handle_escape(event):
            """Escape key interrupts the running agent."""
            if self._agent_running and self.agent:
                print("\n⚡ Interrupting agent...")
                self.agent.interrupt()

        @kb.add('c-d')
        def handle_ctrl_d(event):
            """Handle Ctrl+D - exit."""
            self._should_exit = True
            event.app.exit()

        @kb.add('c-b')
        def handle_ctrl_b(event):
            """Background the current agent task."""
            if not self._agent_running:
                return
            self._background_requested = True
            event.app.invalidate()

        from prompt_toolkit.keys import Keys

        @kb.add(Keys.BracketedPaste, eager=True)
        def handle_paste(event):
            """Handle terminal paste — detect clipboard images.

            When the terminal supports bracketed paste, Ctrl+V / Cmd+V
            triggers this with the pasted text.  We also check the
            clipboard for an image on every paste event.
            """
            pasted_text = event.data or ""
            if self._try_attach_clipboard_image():
                event.app.invalidate()
            if pasted_text:
                event.current_buffer.insert_text(pasted_text)

        @kb.add('c-v')
        def handle_ctrl_v(event):
            """Fallback image paste for terminals without bracketed paste.

            On Linux terminals (GNOME Terminal, Konsole, etc.), Ctrl+V
            sends raw byte 0x16 instead of triggering a paste.  This
            binding catches that and checks the clipboard for images.
            On terminals that DO intercept Ctrl+V for paste (macOS
            Terminal, iTerm2, VSCode, Windows Terminal), the bracketed
            paste handler fires instead and this binding never triggers.
            """
            if self._try_attach_clipboard_image():
                event.app.invalidate()

        @kb.add('escape', 'v')
        def handle_alt_v(event):
            """Alt+V — paste image from clipboard.

            Alt key combos pass through all terminal emulators (sent as
            ESC + key), unlike Ctrl+V which terminals intercept for text
            paste.  This is the reliable way to attach clipboard images
            on WSL2, VSCode, and any terminal over SSH where Ctrl+V
            can't reach the application for image-only clipboard.
            """
            if self._try_attach_clipboard_image():
                event.app.invalidate()
            else:
                # No image found — show a hint
                pass  # silent when no image (avoid noise on accidental press)

        # Dynamic prompt: shows Hermes symbol when agent is working,
        # or answer prompt when clarify freetext mode is active.
        cli_ref = self

        _spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        _spin_frame = [0]

        def get_prompt():
            if cli_ref._sudo_state:
                return [('class:sudo-prompt', '🔐 ❯ ')]
            if cli_ref._approval_state:
                return [('class:prompt-working', '⚠ ❯ ')]
            if cli_ref._clarify_freetext:
                return [('class:clarify-selected', '✎ ❯ ')]
            if cli_ref._clarify_state:
                return [('class:prompt-working', '? ❯ ')]
            if cli_ref._agent_running:
                spin = _spinners[_spin_frame[0] % len(_spinners)]
                return [('class:prompt-working', f'{spin} ❯ ')]
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

        # Dynamic height: accounts for both explicit newlines AND visual
        # wrapping of long lines so the input area always fits its content.
        # The prompt characters ("❯ " etc.) consume ~4 columns.
        def _input_height():
            try:
                doc = input_area.buffer.document
                available_width = _term_width() - 4  # subtract prompt width
                if available_width < 10:
                    available_width = 40
                visual_lines = 0
                for line in doc.lines:
                    # Each logical line takes at least 1 visual row; long lines wrap
                    if len(line) == 0:
                        visual_lines += 1
                    else:
                        visual_lines += max(1, -(-len(line) // available_width))  # ceil division
                return min(max(visual_lines, 1), 8)
            except Exception:
                return 1

        input_area.window.height = _input_height

        # Paste collapsing: detect large pastes and save to temp file
        _paste_counter = [0]
        _prev_text_len = [0]

        def _on_text_changed(buf):
            """Detect large pastes and collapse them to a file reference."""
            text = buf.text
            line_count = text.count('\n')
            chars_added = len(text) - _prev_text_len[0]
            _prev_text_len[0] = len(text)
            # Skip clipboard check during history navigation
            if _history_navigating[0]:
                _history_navigating[0] = False
                return
            # Heuristic: a real paste adds many characters at once
            if chars_added > 1 and not text.startswith('/'):
                if line_count >= 5:
                    _paste_counter[0] += 1
                    paste_dir = Path(os.path.expanduser("~/.hermes/pastes"))
                    paste_dir.mkdir(parents=True, exist_ok=True)
                    paste_file = paste_dir / f"paste_{_paste_counter[0]}_{datetime.now().strftime('%H%M%S')}.txt"
                    paste_file.write_text(text, encoding="utf-8")
                    buf.text = f"[Pasted text #{_paste_counter[0]}: {line_count + 1} lines \u2192 {paste_file}]"
                    buf.cursor_position = len(buf.text)

        input_area.buffer.on_text_changed += _on_text_changed

        # --- Input processors for password masking and inline placeholder ---

        # Mask input with '*' when the sudo password prompt is active
        input_area.control.input_processors.append(
            ConditionalProcessor(
                PasswordProcessor(),
                filter=Condition(lambda: bool(cli_ref._sudo_state)),
            )
        )

        class _PlaceholderProcessor(Processor):
            """Render grayed-out placeholder text inside the input when empty."""
            def __init__(self, get_text):
                self._get_text = get_text

            def apply_transformation(self, ti):
                if not ti.document.text and ti.lineno == 0:
                    text = self._get_text()
                    if text:
                        # Append after existing fragments (preserves the ❯ prompt)
                        return Transformation(fragments=ti.fragments + [('class:placeholder', text)])
                return Transformation(fragments=ti.fragments)

        def _get_placeholder():
            if cli_ref._sudo_state:
                return "type password (hidden), Enter to skip"
            if cli_ref._approval_state:
                return ""
            if cli_ref._clarify_state:
                return ""
            if cli_ref._agent_running:
                return "type a message + Enter to interrupt, Ctrl+B to background, Ctrl+C to cancel"
            return ""

        input_area.control.input_processors.append(_PlaceholderProcessor(_get_placeholder))

        # Hint line above input: shown only for interactive prompts that need
        # extra instructions (sudo countdown, approval navigation, clarify).
        # The agent-running interrupt hint is now an inline placeholder above.
        def get_hint_text():
            import time as _time

            if cli_ref._sudo_state:
                remaining = max(0, int(cli_ref._sudo_deadline - _time.monotonic()))
                return [
                    ('class:hint', '  password hidden · Enter to skip'),
                    ('class:clarify-countdown', f'  ({remaining}s)'),
                ]

            if cli_ref._approval_state:
                remaining = max(0, int(cli_ref._approval_deadline - _time.monotonic()))
                return [
                    ('class:hint', '  ↑/↓ to select, Enter to confirm'),
                    ('class:clarify-countdown', f'  ({remaining}s)'),
                ]

            if cli_ref._clarify_state:
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

            return []

        def get_hint_height():
            if cli_ref._sudo_state or cli_ref._approval_state or cli_ref._clarify_state:
                return 1
            # Keep a 1-line spacer while agent runs so output doesn't push
            # right up against the top rule of the input area
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
            _cw = _term_width()
            _fill = max(_cw - 3 - len('Hermes needs your input') - 2, 1)
            lines.append(('class:clarify-border', ' ' + '─' * _fill + '╮\n'))
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
            lines.append(('class:clarify-border', '╰' + '─' * max(_cw - 2, 1) + '╯\n'))
            return lines

        clarify_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_clarify_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._clarify_state is not None),
        )

        # --- Sudo password: display widget ---

        def _get_sudo_display():
            state = cli_ref._sudo_state
            if not state:
                return []
            lines = []
            lines.append(('class:sudo-border', '╭─ '))
            lines.append(('class:sudo-title', '🔐 Sudo Password Required'))
            _sw = _term_width()
            _sfill = max(_sw - 3 - len('🔐 Sudo Password Required') - 2, 1)
            lines.append(('class:sudo-border', ' ' + '─' * _sfill + '╮\n'))
            lines.append(('class:sudo-border', '│\n'))
            lines.append(('class:sudo-border', '│  '))
            lines.append(('class:sudo-text', 'Enter password below (hidden), or press Enter to skip'))
            lines.append(('', '\n'))
            lines.append(('class:sudo-border', '│\n'))
            lines.append(('class:sudo-border', '╰' + '─' * max(_sw - 2, 1) + '╯\n'))
            return lines

        sudo_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_sudo_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._sudo_state is not None),
        )

        # --- Dangerous command approval: display widget ---

        def _get_approval_display():
            state = cli_ref._approval_state
            if not state:
                return []
            command = state["command"]
            description = state["description"]
            choices = state["choices"]
            selected = state.get("selected", 0)

            cmd_display = command[:70] + '...' if len(command) > 70 else command
            choice_labels = {
                "once": "Allow once",
                "session": "Allow for this session",
                "always": "Add to permanent allowlist",
                "deny": "Deny",
            }

            lines = []
            lines.append(('class:approval-border', '╭─ '))
            lines.append(('class:approval-title', '⚠️  Dangerous Command'))
            _aw = _term_width()
            _afill = max(_aw - 3 - len('⚠️  Dangerous Command') - 2, 1)
            lines.append(('class:approval-border', ' ' + '─' * _afill + '╮\n'))
            lines.append(('class:approval-border', '│\n'))
            lines.append(('class:approval-border', '│  '))
            lines.append(('class:approval-desc', description))
            lines.append(('', '\n'))
            lines.append(('class:approval-border', '│  '))
            lines.append(('class:approval-cmd', cmd_display))
            lines.append(('', '\n'))
            lines.append(('class:approval-border', '│\n'))
            for i, choice in enumerate(choices):
                lines.append(('class:approval-border', '│  '))
                label = choice_labels.get(choice, choice)
                if i == selected:
                    lines.append(('class:approval-selected', f'❯ {label}'))
                else:
                    lines.append(('class:approval-choice', f'  {label}'))
                lines.append(('', '\n'))
            lines.append(('class:approval-border', '│\n'))
            lines.append(('class:approval-border', '╰' + '─' * max(_aw - 2, 1) + '╯\n'))
            return lines

        approval_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_approval_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._approval_state is not None),
        )

        # --- Attached images display (like Claude Code's [Image #N]) ---
        def _get_images_display():
            if not cli_ref._attached_images:
                return []
            fragments = []
            sel = cli_ref._image_select_idx
            base_num = cli_ref._image_counter - len(cli_ref._attached_images) + 1
            for i, img_path in enumerate(cli_ref._attached_images):
                num = base_num + i
                size_kb = img_path.stat().st_size // 1024 if img_path.exists() else 0
                if i > 0:
                    fragments.append(('', ' '))
                if i == sel:
                    fragments.append(('class:image-selected', f' [Image #{num}] '))
                    fragments.append(('class:image-selected-hint', f' {size_kb}KB ⌫'))
                else:
                    fragments.append(('class:image-tag', f' [Image #{num}] '))
                    fragments.append(('class:image-hint', f' {size_kb}KB'))
            if sel >= 0:
                fragments.append(('class:image-hint', '  ↑↓ select · backspace delete · esc cancel'))
            else:
                fragments.append(('class:image-hint', '  (↑ to select)'))
            return fragments

        images_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_images_display),
                height=1,
            ),
            filter=Condition(lambda: len(cli_ref._attached_images) > 0),
        )

        def _get_queued_display():
            if not cli_ref._queued_messages:
                return []
            fragments = []
            for i, msg in enumerate(cli_ref._queued_messages):
                if i > 0:
                    fragments.append(('', '\n'))
                fragments.append(('class:queued-label', '  queued '))
                fragments.append(('class:queued-text', f' {msg[:80]}'))
            return fragments

        def _queued_height():
            return len(cli_ref._queued_messages) if cli_ref._queued_messages else 0

        queued_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_queued_display),
                height=_queued_height,
            ),
            filter=Condition(lambda: len(cli_ref._queued_messages) > 0),
        )

        # Swarm live display widget — shows animated task status
        _swarm_spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        def _get_swarm_display():
            data = cli_ref._swarm_display
            if not data:
                return []
            frm = _spin_frame[0]
            fragments = []
            for i, t in enumerate(data):
                if i > 0:
                    fragments.append(('', '\n'))
                state = t.get('state', 'pending')
                name = t.get('name', '?')
                elapsed = t.get('elapsed', 0)
                ts = f"{elapsed:.0f}s" if elapsed > 0 else ""
                preview = t.get('preview', '')

                if state == 'completed':
                    fragments.append(('class:swarm-done', '  ● '))
                    fragments.append(('class:swarm-name', name))
                    fragments.append(('class:swarm-done', '  done '))
                    fragments.append(('class:swarm-dim', ts))
                    if preview:
                        fragments.append(('class:swarm-dim', f' - {preview}'))
                elif state == 'failed':
                    fragments.append(('class:swarm-fail', '  ✗ '))
                    fragments.append(('class:swarm-name', name))
                    fragments.append(('class:swarm-fail', '  failed '))
                    fragments.append(('class:swarm-dim', ts))
                    if preview:
                        fragments.append(('class:swarm-dim', f' - {preview}'))
                elif state == 'running':
                    spin = _swarm_spinners[frm % len(_swarm_spinners)]
                    fragments.append(('class:swarm-run', f'  {spin} '))
                    fragments.append(('class:swarm-name', name))
                    fragments.append(('class:swarm-run', '  running '))
                    fragments.append(('class:swarm-time', ts))
                else:
                    fragments.append(('class:swarm-dim', f'  ○ {name}  pending'))

            # Summary line
            done = sum(1 for t in data if t.get('state') == 'completed')
            failed = sum(1 for t in data if t.get('state') == 'failed')
            running = sum(1 for t in data if t.get('state') == 'running')
            total = len(data)
            fragments.append(('', '\n'))
            fragments.append(('class:swarm-rule', '  ' + '─' * 40))
            fragments.append(('', '\n'))
            fragments.append(('class:swarm-done', f'  {done}'))
            fragments.append(('class:swarm-summary', f'/{total} done'))
            if running:
                fragments.append(('class:swarm-run', f'  {running}'))
                fragments.append(('class:swarm-summary', ' running'))
            if failed:
                fragments.append(('class:swarm-fail', f'  {failed}'))
                fragments.append(('class:swarm-summary', ' failed'))
            return fragments

        def _swarm_display_height():
            data = cli_ref._swarm_display
            if not data:
                return 0
            return len(data) + 2  # tasks + separator + summary

        swarm_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_swarm_display),
                height=_swarm_display_height,
            ),
            filter=Condition(lambda: cli_ref._swarm_display is not None),
        )

        # Horizontal rules above and below the input (bronze, 1 line each).
        input_rule_top = Window(height=1, char='─', style='class:input-rule')
        input_rule_bot = Window(height=1, char='─', style='class:input-rule')

        # Image attachment indicator — shows badges like [Image #1] above input
        cli_ref = self

        def _get_image_bar():
            if not cli_ref._attached_images:
                return []
            base = cli_ref._image_counter - len(cli_ref._attached_images) + 1
            badges = " ".join(
                f"[Image #{base + i}]"
                for i in range(len(cli_ref._attached_images))
            )
            return [("class:image-badge", f" {badges} ")]

        image_bar = Window(
            content=FormattedTextControl(_get_image_bar),
            height=Condition(lambda: bool(cli_ref._attached_images)),
        )

        # Layout: interactive prompt widgets + ruled input at bottom.
        # The sudo, approval, and clarify widgets appear above the input when
        # the corresponding interactive prompt is active.
        layout = Layout(
            HSplit([
                Window(height=0),
                sudo_widget,
                approval_widget,
                clarify_widget,
                spacer,
                swarm_widget,
                queued_widget,
                images_widget,
                input_rule_top,
                image_bar,
                input_area,
                input_rule_bot,
                CompletionsMenu(max_height=12, scroll_offset=1),
            ])
        )
        
        # Style for the application
        style = PTStyle.from_dict({
            'input-area': '#FFF8DC',
            'placeholder': '#555555 italic',
            'prompt': '#FFF8DC',
            'prompt-working': '#888888 italic',
            'hint': '#555555 italic',
            # Bronze horizontal rules around the input area
            'input-rule': '#CD7F32',
            # Clipboard image attachment badges
            'image-badge': '#87CEEB bold',
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
            # Sudo password panel
            'sudo-prompt': '#FF6B6B bold',
            'sudo-border': '#CD7F32',
            'sudo-title': '#FF6B6B bold',
            'sudo-text': '#FFF8DC',
            # Dangerous command approval panel
            'approval-border': '#CD7F32',
            'approval-title': '#FF8C00 bold',
            'approval-desc': '#FFF8DC bold',
            'approval-cmd': '#AAAAAA italic',
            'approval-choice': '#AAAAAA',
            'approval-selected': '#FFD700 bold',
            # Attached images
            'image-tag': 'bg:#333355 #FFD700 bold underline',
            'image-hint': '#555555 italic',
            'image-selected': 'bg:#552222 #FF6B6B bold',
            'image-selected-hint': '#FF6B6B italic',
            # Queued messages (typed while agent/swarm is running)
            'queued-label': '#555555 italic',
            'queued-text': '#777777',
            # Swarm live display
            'swarm-done': '#32CD32',
            'swarm-fail': '#FF4444',
            'swarm-run': '#FFD700',
            'swarm-time': '#00CED1',
            'swarm-dim': '#666666',
            'swarm-name': '#FF8C00 bold',
            'swarm-rule': '#CD7F32',
            'swarm-summary': '#AAAAAA',
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

        # Spinner animation thread — ticks the prompt spinner when agent is running
        def _spinner_loop():
            while not self._should_exit:
                if self._agent_running:
                    _spin_frame[0] += 1
                    try:
                        app.invalidate()
                    except Exception:
                        pass
                import time as _st
                _st.sleep(0.12)

        _spinner_thread = threading.Thread(target=_spinner_loop, daemon=True, name="spinner")
        _spinner_thread.start()

        # Background thread to process inputs and run agent
        def process_loop():
            while not self._should_exit:
                try:
                    # Check for pending input with timeout
                    try:
                        user_input = self._pending_input.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    # Unpack (text, images) tuple if present
                    submit_images = []
                    if isinstance(user_input, tuple):
                        user_input, submit_images = user_input

                    if not user_input and not submit_images:
                        continue

                    # Check for commands
                    if isinstance(user_input, str) and user_input.startswith("/"):
                        print(f"\n\u2699\uFE0F  {user_input}")
                        if not self.process_command(user_input):
                            self._should_exit = True
                            if app.is_running:
                                app.exit()
                        continue

                    # Expand paste references back to full content
                    import re as _re
                    paste_match = _re.match(r'\[Pasted text #\d+: \d+ lines \u2192 (.+)\]', user_input) if isinstance(user_input, str) else None
                    if paste_match:
                        paste_path = Path(paste_match.group(1))
                        if paste_path.exists():
                            full_text = paste_path.read_text(encoding="utf-8")
                            line_count = full_text.count('\n') + 1
                            print()
                            _cprint(f"{_GOLD}\u25CF{_RST} {_BOLD}[Pasted text: {line_count} lines]{_RST}")
                            user_input = full_text
                        else:
                            print()
                            _cprint(f"{_GOLD}\u25CF{_RST} {_BOLD}{user_input}{_RST}")
                    elif isinstance(user_input, str):
                        if user_input and '\n' in user_input:
                            first_line = user_input.split('\n')[0]
                            line_count = user_input.count('\n') + 1
                            print()
                            _cprint(f"{_GOLD}\u25CF{_RST} {_BOLD}{first_line}{_RST} {_DIM}(+{line_count - 1} lines){_RST}")
                        elif user_input:
                            print()
                            _cprint(f"{_GOLD}\u25CF{_RST} {_BOLD}{user_input}{_RST}")

                    # Show image attachment count
                    if submit_images:
                        n = len(submit_images)
                        _cprint(f"  {_DIM}attached {n} image{'s' if n > 1 else ''}{_RST}")

                    # Auto-detect swarm intent in natural language
                    if isinstance(user_input, str) and _re.search(
                        r'\bswarm\b', user_input, _re.IGNORECASE
                    ):
                        self._handle_swarm_command(f"/swarm {user_input}")
                        continue

                    # Regular chat - run agent
                    self._agent_running = True
                    self._background_requested = False
                    app.invalidate()  # Refresh status line

                    import time as _bg_time

                    # Run chat in a detachable thread
                    chat_done = threading.Event()
                    chat_result = [None]

                    def _run_chat():
                        try:
                            self.chat(user_input, images=submit_images or None)
                        except Exception as e:
                            chat_result[0] = e
                        finally:
                            chat_done.set()

                    chat_thread = threading.Thread(target=_run_chat, daemon=True, name="chat-fg")
                    chat_thread.start()

                    # Wait for completion OR background request
                    while not chat_done.is_set():
                        if self._background_requested:
                            # Detach to background
                            job_id = self._next_job_id
                            self._next_job_id += 1
                            job = {
                                "id": job_id,
                                "thread": chat_thread,
                                "prompt": user_input[:80],
                                "started": _bg_time.time(),
                                "done": chat_done,
                                "result": chat_result,
                            }
                            self._background_jobs.append(job)
                            self._background_requested = False
                            self._agent_running = False
                            app.invalidate()
                            _cprint(f"\n{_DIM}[{job_id}] Backgrounded: {user_input[:60]}...{_RST}")
                            _cprint(f"{_DIM}Use /jobs to check status, /fg {job_id} to see result{_RST}")
                            break
                        chat_done.wait(timeout=0.1)
                    else:
                        # Chat completed normally (not backgrounded)
                        self._agent_running = False
                        self._queued_messages.clear()
                        app.invalidate()  # Refresh status line
                        if chat_result[0] and isinstance(chat_result[0], Exception):
                            print(f"Error: {chat_result[0]}")
                    
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
            # Flush memories before exit (only for substantial conversations)
            if self.agent and self.conversation_history:
                try:
                    self.agent.flush_memories(self.conversation_history)
                except Exception:
                    pass
            # Unregister terminal_tool callbacks to avoid dangling references
            set_sudo_password_callback(None)
            set_approval_callback(None)
            # Close session in SQLite
            if hasattr(self, '_session_db') and self._session_db and self.agent:
                try:
                    self._session_db.end_session(self.agent.session_id, "cli_close")
                except Exception as e:
                    logger.debug("Could not close session in DB: %s", e)
            _run_cleanup()
            self._print_exit_summary()


# ============================================================================
# Main Entry Point
# ============================================================================

def main(
    query: str = None,
    q: str = None,
    toolsets: str = None,
    model: str = None,
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
    max_turns: int = None,
    verbose: bool = False,
    compact: bool = False,
    list_tools: bool = False,
    list_toolsets: bool = False,
    gateway: bool = False,
    resume: str = None,
):
    """
    Hermes Agent CLI - Interactive AI Assistant
    
    Args:
        query: Single query to execute (then exit). Alias: -q
        q: Shorthand for --query
        toolsets: Comma-separated list of toolsets to enable (e.g., "web,terminal")
        model: Model to use (default: anthropic/claude-opus-4-20250514)
        provider: Inference provider ("auto", "openrouter", "nous")
        api_key: API key for authentication
        base_url: Base URL for the API
        max_turns: Maximum tool-calling iterations (default: 60)
        verbose: Enable verbose logging
        compact: Use compact display mode
        list_tools: List available tools and exit
        list_toolsets: List available toolsets and exit
        resume: Resume a previous session by its ID (e.g., 20260225_143052_a1b2c3)
    
    Examples:
        python cli.py                            # Start interactive mode
        python cli.py --toolsets web,terminal    # Use specific toolsets
        python cli.py -q "What is Python?"       # Single query mode
        python cli.py --list-tools               # List tools and exit
        python cli.py --resume 20260225_143052_a1b2c3  # Resume session
    """
    # Signal to terminal_tool that we're in interactive mode
    # This enables interactive sudo password prompts with timeout
    os.environ["HERMES_INTERACTIVE"] = "1"
    
    # Handle gateway mode (messaging + cron)
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
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        max_turns=max_turns,
        verbose=verbose,
        compact=compact,
        resume=resume,
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
        cli._print_exit_summary()
        return
    
    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    fire.Fire(main)
