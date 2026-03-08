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
import sys
import json
import io
import asyncio
import atexit
import random
import time
import uuid
from itertools import zip_longest
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
        load_dotenv(dotenv_path=_user_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_user_env, encoding="latin-1")
elif _project_env.exists():
    try:
        load_dotenv(dotenv_path=_project_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_project_env, encoding="latin-1")

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
            "default": "anthropic/claude-opus-4.6",
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
            "skin": "hermes",
            "animate_banner": False,
            "ambient_motion": True,
            "easter_eggs": True,
        },
        "clarify": {
            "timeout": 120,  # Seconds to wait for a clarify answer before auto-proceeding
        },
        "code_execution": {
            "timeout": 300,    # Max seconds a sandbox script can run before being killed (5 min)
            "max_tool_calls": 50,  # Max RPC tool calls per execution
        },
        "delegation": {
            "max_iterations": 25,  # Max tool-calling turns per child agent
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
        # SSH config
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        # Container resource config (docker, singularity, modal -- ignored for local/ssh)
        "container_cpu": "TERMINAL_CONTAINER_CPU",
        "container_memory": "TERMINAL_CONTAINER_MEMORY",
        "container_disk": "TERMINAL_CONTAINER_DISK",
        "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
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

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import fire

# Import the agent and tool systems
from run_agent import AIAgent
from model_tools import get_tool_definitions, get_toolset_for_tool

# Extracted CLI modules (Phase 3)
from hermes_cli.banner import (
    cprint as _cprint, _GOLD, _BOLD, _DIM, _RST,
    VERSION,
    HERMES_CADUCEUS,
    COMPACT_BANNER as STOCK_COMPACT_BANNER,
    get_available_skills as _get_available_skills,
    build_welcome_banner as build_stock_welcome_banner,
)
from hermes_cli.commands import COMMANDS, SlashCommandCompleter
from hermes_cli import callbacks as _callbacks
import hermes_cli.skin as skin_theme
from hermes_cli.skin import (
    ARES_ASH,
    ARES_BLOOD,
    ARES_BRONZE,
    ARES_CRIMSON,
    ARES_EMBER,
    ARES_SAND,
    ARES_STEEL,
    COIN_SPIN_FRAMES,
    DEFAULT_SKIN,
    DI20_GLYPHS,
    VALID_SKINS,
    build_mod_masthead,
    build_holographic_grid,
    build_orbit_line,
    build_progress_meter,
    build_relay_telemetry,
    build_scroll_frame,
    build_speed_line,
    format_flip_result,
    get_banner_title,
    get_caduceus_frame,
    get_lore_lines,
    get_mod_agent_glyph,
    get_mod_assistant_name,
    get_mod_brand_name,
    get_mod_compact_description,
    get_mod_compact_tagline,
    get_mod_help_footer,
    get_mod_hint_bar,
    get_mod_hero_animation_interval,
    get_mod_next_labels,
    get_mod_omens_title,
    get_mod_placeholder_text,
    get_mod_progress_labels,
    get_mod_prompt_frames,
    get_mod_rituals,
    get_mod_skin_status_label,
    get_mod_system_prompt,
    get_mod_unit_designation,
    get_mod_version_title,
    get_mod_welcome_message,
    is_ares_skin,
    is_mod_skin,
    load_lore_state,
    mod_has_animated_hero,
    maybe_create_trickster_note,
    normalize_skin_name,
    parse_dice_spec,
    resolve_skin_request,
    set_active_skin_globals,
)
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

# ============================================================================
# ASCII Art & Branding
# ============================================================================

# ANSI building blocks for conversation display
_GOLD = "\033[1;31m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def _ansi_fg_hex(color: str) -> str:
    """Return a truecolor ANSI foreground escape for a hex color."""
    color = color.lstrip("#")
    if len(color) != 6:
        return ""
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return ""
    return f"\033[38;2;{r};{g};{b}m"


def _ansi_dim_hex(color: str) -> str:
    """Return a dimmed truecolor ANSI foreground escape for a hex color."""
    return f"\033[2m{_ansi_fg_hex(color)}"

def _cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's native renderer.

    Raw ANSI escapes written via print() are swallowed by patch_stdout's
    StdoutProxy.  Routing through print_formatted_text(ANSI(...)) lets
    prompt_toolkit parse the escapes and render real colors.
    """
    _pt_print(_PT_ANSI(text))

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

def _sync_runtime_skin_theme(skin_name: str | None):
    """Refresh locally imported palette globals after a runtime skin change."""
    global ARES_ASH, ARES_BLOOD, ARES_BRONZE, ARES_CRIMSON, ARES_EMBER, ARES_SAND
    global ARES_STEEL, COIN_SPIN_FRAMES, DI20_GLYPHS

    set_active_skin_globals(skin_name)
    ARES_ASH = skin_theme.ARES_ASH
    ARES_BLOOD = skin_theme.ARES_BLOOD
    ARES_BRONZE = skin_theme.ARES_BRONZE
    ARES_CRIMSON = skin_theme.ARES_CRIMSON
    ARES_EMBER = skin_theme.ARES_EMBER
    ARES_SAND = skin_theme.ARES_SAND
    ARES_STEEL = skin_theme.ARES_STEEL
    COIN_SPIN_FRAMES = skin_theme.COIN_SPIN_FRAMES
    DI20_GLYPHS = skin_theme.DI20_GLYPHS


def _build_mod_compact_banner() -> str:
    return (
        f"\n"
        f"[bold {ARES_CRIMSON}]╔══════════════════════════════════════════════════════════════╗[/]\n"
        f"[bold {ARES_CRIMSON}]║[/]  [{ARES_BRONZE}]{get_mod_agent_glyph()} {get_mod_brand_name().upper()}[/] "
        f"[dim {ARES_ASH}]- {get_mod_compact_tagline()}[/]               [bold {ARES_CRIMSON}]║[/]\n"
        f"[bold {ARES_CRIMSON}]║[/]  [{ARES_EMBER}]{get_mod_compact_description()}[/] "
        f"[dim {ARES_ASH}]Nous Research[/]   [bold {ARES_CRIMSON}]║[/]\n"
        f"[bold {ARES_CRIMSON}]╚══════════════════════════════════════════════════════════════╝[/]\n"
    )


_sync_runtime_skin_theme(os.getenv("HERMES_CLI_SKIN"))


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


def build_welcome_banner(
    console: Console,
    model: str,
    cwd: str,
    tools: List[dict] = None,
    enabled_toolsets: List[str] = None,
    session_id: str = None,
    *,
    skin: str = DEFAULT_SKIN,
    lore_state=None,
    phase: int = 0,
):
    """Build and print the CLI welcome banner."""
    from model_tools import check_tool_availability

    tools = tools or []
    enabled_toolsets = enabled_toolsets or []
    lore_state = lore_state or load_lore_state()
    mod_skin = is_mod_skin(skin)

    if not mod_skin:
        build_stock_welcome_banner(
            console=console,
            model=model,
            cwd=cwd,
            tools=tools,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            get_toolset_for_tool=get_toolset_for_tool,
        )
        return

    _, unavailable_toolsets = check_tool_availability(quiet=True)
    disabled_tools = set()
    for item in unavailable_toolsets:
        disabled_tools.update(item.get("tools", []))

    model_short = model.split("/")[-1] if "/" in model else model
    if len(model_short) > 28:
        model_short = model_short[:25] + "..."
    cwd_short = cwd if len(cwd) <= 30 else f"...{cwd[-27:]}"

    toolsets_dict = {}
    for tool in tools:
        tool_name = tool["function"]["name"]
        toolset = (get_toolset_for_tool(tool_name) or "other").replace("_tools", "")
        toolsets_dict.setdefault(toolset, []).append(tool_name)

    for item in unavailable_toolsets:
        toolset_id = item.get("id", item.get("name", "unknown"))
        display_name = str(toolset_id).replace("_tools", "")
        toolsets_dict.setdefault(display_name, [])
        for tool_name in item.get("tools", []):
            if tool_name not in toolsets_dict[display_name]:
                toolsets_dict[display_name].append(tool_name)

    sorted_toolsets = sorted(toolsets_dict.keys())
    display_toolsets = sorted_toolsets[:8]
    remaining_toolsets = len(sorted_toolsets) - 8
    skills_by_category = _get_available_skills()
    total_skills = sum(len(s) for s in skills_by_category.values())

    banner_width = max(getattr(console.size, "width", 160), 120)
    left_width = min(68, max(58, banner_width // 3 + 10))
    hero_width = max(52, left_width - 4)
    hero_height = 32 if banner_width >= 150 else 26

    layout_table = Table.grid(expand=True, padding=(0, 2))
    layout_table.add_column("left", width=left_width)
    layout_table.add_column("middle", ratio=1, min_width=56)

    left_lines = [
        "",
        get_caduceus_frame(lore_state, phase, width=hero_width, height=hero_height),
        "",
        f"[{ARES_BRONZE}]{model_short}[/]  [dim {ARES_ASH}]Nous Research[/]",
        f"[dim {ARES_ASH}]{cwd_short}[/]",
    ]
    if session_id:
        left_lines.append(f"[dim {ARES_ASH}]Session: {session_id}[/]")
    left_renderable = Align.center(
        Text.from_markup("\n".join(left_lines)),
        vertical="top",
    )

    accent_toolsets = {"file_tools", "file_tools_tools", "image_gen_tools", "image_gen_tools_tools"}
    center_lines = [f"[bold {ARES_BRONZE}]Available Tools[/]"]
    for toolset in display_toolsets:
        tool_names = toolsets_dict[toolset]
        colored_names = []
        for name in sorted(tool_names):
            if name in disabled_tools:
                colored_names.append(f"[{ARES_EMBER}]{name}[/]")
            elif toolset in accent_toolsets:
                colored_names.append(f"[{ARES_EMBER}]{name}[/]")
            else:
                colored_names.append(f"[{ARES_SAND}]{name}[/]")

        if len(", ".join(sorted(tool_names))) > 50:
            short_names = []
            length = 0
            for name in sorted(tool_names):
                if length + len(name) + 2 > 46:
                    short_names.append("...")
                    break
                short_names.append(name)
                length += len(name) + 2
            colored_names = []
            for name in short_names:
                if name == "...":
                    colored_names.append("[dim]...[/]")
                elif name in disabled_tools or toolset in accent_toolsets:
                    colored_names.append(f"[{ARES_EMBER}]{name}[/]")
                else:
                    colored_names.append(f"[{ARES_SAND}]{name}[/]")
        center_lines.append(f"[dim {ARES_ASH}]{toolset.replace('_tools', '')}:[/] {', '.join(colored_names)}")

    if remaining_toolsets > 0:
        center_lines.append(f"[dim {ARES_ASH}](and {remaining_toolsets} more toolsets...)[/]")

    center_lines.append("")
    center_lines.append(f"[bold {ARES_BRONZE}]Available Skills[/]")
    if skills_by_category:
        for category in sorted(skills_by_category.keys())[:12]:
            skill_names = sorted(skills_by_category[category])
            if len(skill_names) > 8:
                display_names = skill_names[:8]
                skills_str = ", ".join(display_names) + f" +{len(skill_names) - 8} more"
            else:
                skills_str = ", ".join(skill_names)
            if len(skills_str) > 58:
                skills_str = skills_str[:55] + "..."
            center_lines.append(f"[dim {ARES_ASH}]{category}:[/] [{ARES_SAND}]{skills_str}[/]")
    else:
        center_lines.append(f"[dim {ARES_ASH}]No skills installed[/]")

    center_lines.append("")
    center_lines.append(f"[dim {ARES_ASH}]{get_mod_help_footer(len(tools), total_skills)}[/]")
    center_renderable = Text.from_markup("\n".join(center_lines))

    layout_table.add_row(left_renderable, center_renderable)

    dossier_panel = Panel(
        layout_table,
        title=f"[bold {ARES_BRONZE}]{get_mod_version_title(VERSION)}[/]",
        title_align="center",
        subtitle=f"[bold {ARES_CRIMSON}]{get_mod_unit_designation()}[/]",
        subtitle_align="right",
        border_style=ARES_BRONZE,
        box=box.SQUARE,
        padding=(0, 1),
    )

    console.print()
    console.print(build_mod_masthead())
    console.print(dossier_panel)
    return


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
    "/skin": "Show or change the active visual skin",
    "/flip": "Flip the local ritual coin",
    "/roll": "Roll local ritual dice (supports /roll d20)",
    "/omens": "Show active skin lore, unlocks, and ritual status",
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
        max_turns: int = 60,
        verbose: bool = False,
        compact: bool = False,
        skin: str = None,
        resume: str = None,
    ):
        """
        Initialize the Hermes CLI.
        
        Args:
            model: Model to use (default: from env or claude-sonnet)
            toolsets: List of toolsets to enable (default: all)
            provider: Inference provider ("auto", "openrouter", "nous")
            api_key: API key (default: from environment)
            base_url: API base URL (default: OpenRouter)
            max_turns: Maximum tool-calling iterations (default: 60)
            verbose: Enable verbose logging
            compact: Use compact display mode
            skin: Visual skin name ("hermes", "ares", "posideon", or "sisyphus")
            resume: Session ID to resume (restores conversation history from SQLite)
        """
        # Initialize Rich console
        self.console = Console()
        self.compact = compact if compact is not None else CLI_CONFIG["display"].get("compact", False)
        self.verbose = verbose if verbose is not None else CLI_CONFIG["agent"].get("verbose", False)
        self.skin = normalize_skin_name(
            skin
            or os.getenv("HERMES_CLI_SKIN")
            or CLI_CONFIG["display"].get("skin", DEFAULT_SKIN)
        )
        _sync_runtime_skin_theme(self.skin)
        self.animate_banner = bool(CLI_CONFIG["display"].get("animate_banner", False))
        self.ambient_motion = bool(CLI_CONFIG["display"].get("ambient_motion", True))
        self.easter_eggs = bool(CLI_CONFIG["display"].get("easter_eggs", True))
        self._banner_phase = 0
        self._ui_phase = 0
        self._managed_banner_frozen = False
        self._lore_state = load_lore_state()
        self._sync_skin_env()
        
        # Configuration - priority: CLI args > env vars > config file
        # Model can come from: CLI arg, LLM_MODEL env, OPENAI_MODEL env (custom endpoint), or config
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or CLI_CONFIG["model"]["default"]
        
        # Base URL: custom endpoint (OPENAI_BASE_URL) takes precedence over OpenRouter
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", CLI_CONFIG["model"]["base_url"])
        
        # API key: custom endpoint (OPENAI_API_KEY) takes precedence over OpenRouter
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

        # Provider resolution: determines whether to use OAuth credentials or env var keys
        from hermes_cli.auth import resolve_provider
        self.requested_provider = (
            provider
            or os.getenv("HERMES_INFERENCE_PROVIDER")
            or CLI_CONFIG["model"].get("provider")
            or "auto"
        )
        self.provider = resolve_provider(
            self.requested_provider,
            explicit_api_key=api_key,
            explicit_base_url=base_url,
        )
        self._nous_key_expires_at: Optional[str] = None
        self._nous_key_source: Optional[str] = None
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
        
        # Ephemeral system prompt: env var takes precedence, then config
        self.user_system_prompt = (
            os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
            or CLI_CONFIG["agent"].get("system_prompt", "")
        )
        self.system_prompt = self._compose_system_prompt()
        self.personalities = CLI_CONFIG["agent"].get("personalities", {})
        
        # Ephemeral prefill messages (few-shot priming, never persisted)
        self.prefill_messages = _load_prefill_messages(
            CLI_CONFIG["agent"].get("prefill_messages_file", "")
        )
        
        # Reasoning config (OpenRouter reasoning effort level)
        self.reasoning_config = _parse_reasoning_config(
            CLI_CONFIG["agent"].get("reasoning_effort", "")
        )
        
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

    def _sync_skin_env(self):
        """Expose display settings to modules that only see environment state."""
        os.environ["HERMES_CLI_SKIN"] = self.skin
        os.environ["HERMES_CLI_MOTION"] = "1" if self.ambient_motion else "0"
        os.environ["HERMES_CLI_EASTER_EGGS"] = "1" if self.easter_eggs else "0"

    def _refresh_lore(self):
        """Reload Hermes lore counters from local state."""
        self._lore_state = load_lore_state(getattr(self, "_session_db", None))

    def _ares_skin_active(self) -> bool:
        """True when a custom mod skin is enabled and not in compact mode."""
        return is_mod_skin(self.skin) and not self.compact

    def _ensure_runtime_credentials(self) -> bool:
        """
        Ensure OAuth provider credentials are fresh before agent use.
        For Nous Portal: checks agent key TTL, refreshes/re-mints as needed.
        If the key changed, tears down the agent so it rebuilds with new creds.
        Returns True if credentials are ready, False on auth failure.
        """
        if self.provider != "nous":
            return True

        from hermes_cli.auth import format_auth_error, resolve_nous_runtime_credentials

        try:
            credentials = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(
                    60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))
                ),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
            )
        except Exception as exc:
            message = format_auth_error(exc)
            self.console.print(f"[bold red]{message}[/]")
            return False

        api_key = credentials.get("api_key")
        base_url = credentials.get("base_url")
        if not isinstance(api_key, str) or not api_key:
            self.console.print("[bold red]Nous credential resolver returned an empty API key.[/]")
            return False
        if not isinstance(base_url, str) or not base_url:
            self.console.print("[bold red]Nous credential resolver returned an empty base URL.[/]")
            return False

        credentials_changed = api_key != self.api_key or base_url != self.base_url
        self.api_key = api_key
        self.base_url = base_url
        self._nous_key_expires_at = credentials.get("expires_at")
        self._nous_key_source = credentials.get("source")

        # AIAgent/OpenAI client holds auth at init time, so rebuild if key rotated
        if credentials_changed and self.agent is not None:
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

        if self.provider == "nous" and not self._ensure_runtime_credentials():
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
                max_iterations=self.max_turns,
                enabled_toolsets=self.enabled_toolsets,
                verbose_logging=self.verbose,
                quiet_mode=True,
                ephemeral_system_prompt=self.system_prompt if self.system_prompt else None,
                prefill_messages=self.prefill_messages or None,
                reasoning_config=self.reasoning_config,
                session_id=self.session_id,
                platform="cli",
                session_db=self._session_db,
                suppress_progress_output=self._uses_managed_banner(),
                clarify_callback=self._clarify_callback,
            )
            return True
        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize agent: {e}[/]")
            return False
    
    def show_banner(self):
        """Display the welcome banner for the active skin."""
        self.console.clear()

        if self._uses_startup_banner_animation():
            self._show_animated_startup_banner()
            return

        self._render_banner(self.console, phase=self._banner_phase)
        self._banner_phase += 1

    def _render_banner(self, console, *, phase: int):
        """Render the banner for a specific phase without mutating CLI state."""
        if self.compact:
            if is_mod_skin(self.skin):
                console.print(_build_mod_compact_banner())
            else:
                console.print(STOCK_COMPACT_BANNER)
            self._show_status(console=console)
        else:
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
            cwd = os.getenv("TERMINAL_CWD", os.getcwd())
            self._refresh_lore()
            build_welcome_banner(
                console=console,
                model=self.model,
                cwd=cwd,
                tools=tools,
                enabled_toolsets=self.enabled_toolsets,
                session_id=self.session_id,
                skin=self.skin,
                lore_state=self._lore_state,
                phase=phase,
            )

        self._show_tool_availability_warnings(console=console)
        console.print()

    def _uses_startup_banner_animation(self) -> bool:
        return (
            not self.compact
            and self.ambient_motion
            and is_mod_skin(self.skin)
            and mod_has_animated_hero(self.skin)
        )

    def _uses_managed_banner(self) -> bool:
        """Managed interactive banners are disabled; animated skins only animate at startup."""
        return False

    def _rewrite_banner_lines_in_place(
        self,
        previous_lines: List[str],
        next_lines: List[str],
        total_lines: int,
    ) -> None:
        """Patch only the changed banner lines during startup animation."""
        if not previous_lines or not next_lines:
            return

        sys.stdout.write("\0337")
        sys.stdout.write(f"\033[{total_lines}F")
        current_line = 0

        for index, (before, after) in enumerate(zip_longest(previous_lines, next_lines, fillvalue="")):
            if before == after:
                continue
            delta = index - current_line
            if delta > 0:
                sys.stdout.write(f"\033[{delta}B")
            sys.stdout.write("\r\033[2K")
            sys.stdout.write(after)
            current_line = index

        sys.stdout.write("\0338")
        sys.stdout.flush()

    def _store_banner_snapshot(self, banner_ansi: str) -> None:
        """Track the currently rendered banner for in-place animation updates."""
        self._banner_snapshot_lines = banner_ansi.splitlines()
        self._banner_snapshot_line_count = len(self._banner_snapshot_lines)

    def _rewrite_banner_lines_absolute(
        self,
        previous_lines: List[str],
        next_lines: List[str],
    ) -> None:
        """Rewrite changed banner lines at absolute rows without reprinting the terminal."""
        if not previous_lines or not next_lines:
            return
        if len(previous_lines) != len(next_lines):
            return

        stream = sys.__stdout__
        stream.write("\0337")
        for index, (before, after) in enumerate(zip_longest(previous_lines, next_lines, fillvalue=""), start=1):
            if before == after:
                continue
            stream.write(f"\033[{index};1H\033[2K{after}")
        stream.write("\0338")
        stream.flush()

    def _advance_live_banner_frame(self) -> None:
        """Advance an animated hero frame in place while the prompt is idle."""
        previous_lines = getattr(self, "_banner_snapshot_lines", None)
        if not previous_lines:
            return

        next_phase = self._banner_phase + 1
        banner_ansi = self._build_banner_ansi(phase=next_phase)
        next_lines = banner_ansi.splitlines()
        if len(next_lines) != getattr(self, "_banner_snapshot_line_count", len(previous_lines)):
            return

        self._rewrite_banner_lines_absolute(previous_lines, next_lines)
        self._store_banner_snapshot(banner_ansi)
        self._banner_phase = next_phase

    def _show_animated_startup_banner(self):
        """Animate the hero asset in place during startup without repainting the terminal."""
        phase = self._banner_phase
        current_ansi = self._build_banner_ansi(phase=phase)
        sys.stdout.write(current_ansi)
        sys.stdout.flush()

        current_lines = current_ansi.splitlines()
        total_lines = len(current_lines)
        interval = max(0.12, get_mod_hero_animation_interval(self.skin))
        frame_count = max(4, min(8, int(round(1.5 / interval))))

        for step in range(1, frame_count):
            time.sleep(interval)
            next_ansi = self._build_banner_ansi(phase=phase + step)
            next_lines = next_ansi.splitlines()
            if len(next_lines) != total_lines:
                break
            self._rewrite_banner_lines_in_place(current_lines, next_lines, total_lines)
            current_lines = next_lines

        self._store_banner_snapshot("\n".join(current_lines))
        self._banner_phase = phase + frame_count

    def _build_banner_ansi(self, *, phase: int | None = None) -> str:
        """Render the current banner and return ANSI for prompt_toolkit-safe redraws."""
        capture = io.StringIO()
        temp_console = Console(
            record=True,
            force_terminal=True,
            color_system="truecolor",
            width=max(getattr(self.console, "width", 120) or 120, 80),
            file=capture,
        )
        self._render_banner(temp_console, phase=self._banner_phase if phase is None else phase)
        return temp_console.export_text(styles=True, clear=False)

    def _build_managed_banner_ansi(self) -> str:
        """Build the prompt-toolkit-managed banner block."""
        return self._build_banner_ansi(phase=self._banner_phase)

    def _managed_banner_height(self) -> int:
        """Return the current line height for the managed banner block."""
        return max(1, len(self._build_managed_banner_ansi().splitlines()))

    def _append_managed_output(self, ansi_text: str) -> None:
        """Append ANSI transcript content for managed-banner skins."""
        if not ansi_text:
            return
        existing = getattr(self, "_managed_output_ansi", "")
        if existing and not existing.endswith("\n") and not ansi_text.startswith("\n"):
            existing += "\n"
        self._managed_output_ansi = existing + ansi_text
        if self._app is not None:
            self._app.invalidate()

    def _managed_output_height(self) -> int:
        """Return the current line height for the managed transcript region."""
        output = getattr(self, "_managed_output_ansi", "")
        if not output:
            return 0
        return len(output.splitlines()) or 1
    
    def _show_tool_availability_warnings(self, console=None):
        """Show warnings about disabled tools due to missing API keys."""
        console = console or self.console
        try:
            from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
            
            available, unavailable = check_tool_availability()
            
            # Filter to only those missing API keys (not system deps)
            api_key_missing = [u for u in unavailable if u["missing_vars"]]
            
            if api_key_missing:
                console.print()
                console.print("[yellow]⚠️  Some tools disabled (missing API keys):[/]")
                for item in api_key_missing:
                    tools_str = ", ".join(item["tools"][:2])  # Show first 2 tools
                    if len(item["tools"]) > 2:
                        tools_str += f", +{len(item['tools'])-2} more"
                    console.print(f"   [dim]• {item['name']}[/] [dim italic]({', '.join(item['missing_vars'])})[/]")
                console.print("[dim]   Run 'hermes setup' to configure[/]")
        except Exception:
            pass  # Don't crash on import errors
    
    def _show_status(self, console=None):
        """Show current status bar."""
        console = console or self.console
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
            toolsets_info = f" [dim {ARES_ASH}]·[/] [{ARES_BRONZE}]toolsets: {', '.join(self.enabled_toolsets)}[/]"

        provider_info = f" [dim {ARES_ASH}]·[/] [dim]provider: {self.provider}[/]"
        if self.provider == "nous" and self._nous_key_source:
            provider_info += f" [dim {ARES_ASH}]·[/] [dim]key: {self._nous_key_source}[/]"

        console.print(
            f"  {api_indicator} [{ARES_BRONZE}]{model_short}[/] "
            f"[dim {ARES_ASH}]·[/] [bold {ARES_EMBER}]{tool_count} tools[/]"
            f"{toolsets_info}{provider_info}"
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
        assistant_name = get_mod_assistant_name() if self._ares_skin_active() else "Hermes"
        print(f"  Tip: Just type your message to chat with {assistant_name}!")
        print("  Bonus: type 'flip coin' or 'roll dice' for local rituals")
        print("  Multi-line: Alt+Enter for a new line")
        print()

    def _set_skin(self, new_skin: str, *, persist: bool = False):
        """Apply a new visual skin to the active session."""
        normalized = normalize_skin_name(new_skin)
        self.skin = normalized
        self._banner_phase = 0
        self._banner_last_refresh = 0.0
        self._ui_phase = 0
        self._managed_banner_frozen = False
        _sync_runtime_skin_theme(normalized)
        self._sync_skin_env()
        self._refresh_effective_system_prompt()
        self.agent = None
        if self._app is not None:
            self._app.style = self._build_prompt_style()
            self._app.invalidate()
        if persist:
            save_config_value("display.skin", normalized)

    def _freeze_managed_banner(self) -> None:
        """Stop managed-banner animation after the first user interaction."""
        if not self._uses_managed_banner():
            return
        if self._managed_banner_frozen:
            return
        self._managed_banner_frozen = True
        if self._app is not None:
            self._app.invalidate()

    def _compose_system_prompt(self) -> str:
        """Combine the active skin persona with any user-selected prompt."""
        parts = []
        skin_prompt = get_mod_system_prompt(self.skin) if is_mod_skin(self.skin) else ""
        if skin_prompt:
            parts.append(skin_prompt.strip())
        if getattr(self, "user_system_prompt", ""):
            parts.append(self.user_system_prompt.strip())
        return "\n\n".join(part for part in parts if part)

    def _refresh_effective_system_prompt(self):
        """Refresh the prompt passed into the agent for the current skin."""
        self.system_prompt = self._compose_system_prompt()

    def _reload_skin_ui(self):
        """Redraw the terminal UI after a skin change like a fresh launcher boot."""
        if self._app is not None and getattr(self._app, "is_running", False):
            if self._uses_managed_banner():
                _cprint("\033[2J\033[H")
                self._app.invalidate()
                return
            banner_ansi = self._build_banner_ansi()
            self._store_banner_snapshot(banner_ansi)
            _cprint("\033[2J\033[H" + banner_ansi)
            self._app.invalidate()
            return
        self.console.clear()
        self.show_banner()

    def _build_prompt_style(self):
        """Build the prompt_toolkit style object for the active skin."""
        mod_skin = is_mod_skin(self.skin)
        input_text_color = ARES_SAND if self._ares_skin_active() else '#FFF8DC'
        accent_color = ARES_CRIMSON if self._ares_skin_active() else '#CD7F32'
        title_color = ARES_BRONZE if self._ares_skin_active() else '#FFD700'
        menu_bg = '#1c1212' if self._ares_skin_active() else '#1a1a2e'
        menu_current_bg = '#3a1717' if self._ares_skin_active() else '#333355'
        placeholder_color = f'{ARES_ASH} italic' if mod_skin else '#555555 italic'
        subtle_hint_color = f'{ARES_ASH} italic' if mod_skin else '#555555 italic'
        working_prompt_color = f'{ARES_ASH} italic' if mod_skin else '#888888 italic'

        return PTStyle.from_dict({
            'input-area': input_text_color,
            'placeholder': placeholder_color,
            'prompt': input_text_color,
            'prompt-working': working_prompt_color,
            'prompt-flight': f'{title_color} bold',
            'hint': subtle_hint_color,
            'hint-bar': f'{accent_color} italic',
            'hint-telemetry': ARES_ASH,
            'input-rule': accent_color,
            'completion-menu': f'bg:{menu_bg} {input_text_color}',
            'completion-menu.completion': f'bg:{menu_bg} {input_text_color}',
            'completion-menu.completion.current': f'bg:{menu_current_bg} {title_color}',
            'completion-menu.meta.completion': f'bg:{menu_bg} #888888',
            'completion-menu.meta.completion.current': f'bg:{menu_current_bg} {accent_color}',
            'clarify-border': accent_color,
            'clarify-title': f'{title_color} bold',
            'clarify-question': f'{input_text_color} bold',
            'clarify-choice': '#AAAAAA',
            'clarify-selected': f'{title_color} bold',
            'clarify-active-other': f'{title_color} italic',
            'clarify-countdown': accent_color,
            'sudo-prompt': '#FF6B6B bold',
            'sudo-border': accent_color,
            'sudo-title': '#FF6B6B bold',
            'sudo-text': input_text_color,
            'approval-border': accent_color,
            'approval-title': '#FF8C00 bold',
            'approval-desc': f'{input_text_color} bold',
            'approval-cmd': '#AAAAAA italic',
            'approval-choice': '#AAAAAA',
            'approval-selected': f'{title_color} bold',
        })

    def _animate_inline_frames(self, frames: List[str], delay: float = 0.07):
        """Run a lightweight inline terminal animation."""
        if not frames:
            return
        out = sys.stdout
        width = max(len(frame) for frame in frames)
        for frame in frames:
            try:
                out.write("\r" + frame.ljust(width))
                out.flush()
            except (ValueError, OSError):
                break
            time.sleep(delay)
        try:
            out.write("\r" + (" " * width) + "\r")
            out.flush()
        except (ValueError, OSError):
            pass

    def _play_coin_flip(self):
        """Animate a coin flip easter egg."""
        result = random.choice(("heads", "tails"))
        glyph = get_mod_agent_glyph() if self._ares_skin_active() else "⚔"
        agent_name = get_mod_assistant_name() if self._ares_skin_active() else "Hermes"
        frames = [f"  {glyph} {agent_name} flips the coin {frame}" for frame in COIN_SPIN_FRAMES]
        print()
        self._animate_inline_frames(frames)
        _cprint(f"{_GOLD}◉{_RST} {_BOLD}{format_flip_result(result)}{_RST}")

    def _play_dice_roll(self, spec: str = None):
        """Animate a dice roll easter egg."""
        sides = parse_dice_spec(spec)
        result = random.randint(1, sides)
        cheat_note = ""
        if is_ares_skin(self.skin) and sides >= 6 and random.random() < 0.25 and result < sides:
            nudged = result + 1
            cheat_note = f" {get_mod_assistant_name()} nudged it from {result} to {nudged}."
            result = nudged

        agent_glyph = get_mod_agent_glyph() if self._ares_skin_active() else "⚔"
        agent_name = get_mod_assistant_name() if self._ares_skin_active() else "Hermes"
        frames = [f"  {agent_glyph} {agent_name} rolls d{sides} {glyph}" for glyph in DI20_GLYPHS * 2]
        print()
        self._animate_inline_frames(frames, delay=0.06)
        _cprint(f"{_GOLD}◉{_RST} {_BOLD}d{sides}: {result}{_RST}{_DIM}{cheat_note}{_RST}")

    def _maybe_handle_local_ritual(self, user_input: str) -> bool:
        """Handle local mini-games before sending text to the model."""
        normalized = user_input.strip().lower()
        if normalized in {"flip coin", "coin flip"}:
            self._play_coin_flip()
            return True
        if normalized in {"roll dice", "dice roll"}:
            self._play_dice_roll()
            return True
        return False

    def _format_hermes_scroll_body(self, response: str) -> str:
        """Decorate assistant text with a cleaner gutter inside the Hermes scroll."""
        marker_cycle = ["╎", "┆", "╎", "┊"]
        marker = marker_cycle[self._ui_phase % len(marker_cycle)]
        gutter_color = _ansi_fg_hex(ARES_ASH) if self._ares_skin_active() else _DIM
        body_color = _ansi_fg_hex(ARES_SAND) if self._ares_skin_active() else ""
        lines = response.splitlines() or [response]
        formatted_lines = []
        for line in lines:
            if line.strip():
                formatted_lines.append(f"{gutter_color}{marker}{_RST} {body_color}{line}{_RST}")
            else:
                formatted_lines.append(f"{gutter_color}{marker}{_RST}")
        return "\n".join(formatted_lines)

    def _mod_response_frame_color(self) -> str:
        """ANSI color for response frame chrome on custom skins."""
        return _ansi_fg_hex(ARES_EMBER)

    def _mod_response_subtle_color(self) -> str:
        """ANSI color for response subtitles and secondary notes on custom skins."""
        return _ansi_dim_hex(ARES_ASH)

    def show_omens(self):
        """Display active skin lore progression and ritual status."""
        self._refresh_lore()
        next_wing_unlock = max(0, 50 - self._lore_state.sessions)
        next_glow_unlock = max(0, 100 - self._lore_state.clever_replies)
        orbiting = ", ".join(self._lore_state.orbiting_skills) if self._lore_state.orbiting_skills else "awaiting published scrolls"
        sessions_label, glow_label, orbit_label = get_mod_progress_labels()
        next_sessions_label, next_glow_label = get_mod_next_labels()

        if self._ares_skin_active():
            rituals = get_mod_rituals()
            omens_lines = [
                f"[bold {ARES_EMBER}]{get_mod_skin_status_label()}[/] [{ARES_SAND}]{self.skin}[/]",
                build_progress_meter(sessions_label, self._lore_state.sessions, 50, width=18),
                build_progress_meter(glow_label, self._lore_state.clever_replies, 100, width=18),
                build_progress_meter(orbit_label, len(self._lore_state.orbiting_skills), 4, width=18),
                f"[dim {ARES_ASH}]{build_relay_telemetry(self._lore_state, self._ui_phase, 46, active=False)}[/]",
                f"[dim {ARES_BRONZE}]{build_orbit_line(self._lore_state, self._ui_phase, 42)}[/]",
                "",
                f"[{ARES_SAND}]Tier[/] [dim {ARES_ASH}]{self._lore_state.wing_level}[/]  [{ARES_SAND}]{glow_label}[/] [dim {ARES_ASH}]{self._lore_state.glow_enabled}[/]",
                f"[{ARES_SAND}]{next_sessions_label}[/] [dim {ARES_ASH}]{next_wing_unlock} session(s)[/]  [{ARES_SAND}]{next_glow_label}[/] [dim {ARES_ASH}]{next_glow_unlock} clever repl(y/ies)[/]",
                f"[{ARES_SAND}]Orbiting skills[/] [dim {ARES_ASH}]{orbiting}[/]",
                "",
                f"[bold {ARES_BRONZE}]Rituals[/]",
            ]
            omens_lines.extend(
                f"[{ARES_SAND}]{command}[/] [dim {ARES_ASH}]{description}[/]"
                for command, description in rituals
            )
            self.console.print()
            self.console.print(
                Panel(
                    Text.from_markup("\n".join(omens_lines)),
                    title=f"[bold {ARES_BRONZE}]{get_mod_omens_title()}[/]",
                    border_style=ARES_BLOOD,
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
            self.console.print()
            return

        print()
        print("+" + "-" * 64 + "+")
        title = get_mod_omens_title()
        print("|" + " " * max(0, 32 - len(title) // 2) + title + " " * max(0, 64 - 2 - max(0, 32 - len(title) // 2) - len(title)) + "|")
        print("+" + "-" * 64 + "+")
        print()
        print(f"  Skin:            {self.skin}")
        print(f"  Sessions:        {self._lore_state.sessions}")
        print(f"  Clever replies:  {self._lore_state.clever_replies}")
        print(f"  Wing tier:       {self._lore_state.wing_level}")
        print(f"  Golden glow:     {self._lore_state.glow_enabled}")
        print(f"  Orbiting skills: {orbiting}")
        print()
        print(f"  Next shields:    {next_wing_unlock} session(s)")
        print(f"  Next glow:       {next_glow_unlock} clever repl(y/ies)")
        print()
        print("  Rituals:")
        for command, description in get_mod_rituals():
            print(f"    {command:<11} {description}")
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
        print("  -- Display --")
        print(f"  Skin:       {self.skin}")
        print(f"  Motion:     {self.ambient_motion}")
        print(f"  EasterEggs: {self.easter_eggs}")
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
                assistant_name = get_mod_assistant_name() if self._ares_skin_active() else "Hermes"
                print(f"\n  [{assistant_name} #{i}]")
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
                self.user_system_prompt = ""
                self._refresh_effective_system_prompt()
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", ""):
                    print("(^_^)b System prompt cleared (saved to config)")
                else:
                    print("(^_^) System prompt cleared (session only)")
            else:
                self.user_system_prompt = new_prompt
                self._refresh_effective_system_prompt()
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
            if self.user_system_prompt:
                # Word wrap the prompt for display
                words = self.user_system_prompt.split()
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
                if is_mod_skin(self.skin):
                    print(f"  (no custom prompt set - using {self.skin} skin persona)")
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
                self.user_system_prompt = self.personalities[personality_name]
                self._refresh_effective_system_prompt()
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", self.user_system_prompt):
                    print(f"(^_^)b Personality set to '{personality_name}' (saved to config)")
                else:
                    print(f"(^_^) Personality set to '{personality_name}' (session only)")
                print(f"  \"{self.user_system_prompt[:60]}{'...' if len(self.user_system_prompt) > 60 else ''}\"")
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
            # Flush memories before clearing
            if self.agent and self.conversation_history:
                try:
                    self.agent.flush_memories(self.conversation_history)
                except Exception:
                    pass
            # Reset conversation
            self.conversation_history = []
            # Show fresh banner
            self._reload_skin_ui()
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
        elif cmd_lower.startswith("/skin"):
            parts = cmd_original.split(maxsplit=1)
            if len(parts) == 1:
                print(f"Active skin: {self.skin}")
                print("  Usage: /skin Hermes|Ares|Posideon|Sisyphus")
                print("  Example: /skin Sisyphus")
            else:
                requested_skin = resolve_skin_request(parts[1])
                if requested_skin not in VALID_SKINS:
                    print(f"(._.) Unknown skin: {parts[1]}")
                else:
                    self._set_skin(requested_skin, persist=True)
                    self._reload_skin_ui()
                    print(f"  ✨ Skin set to {requested_skin} (saved to config)\n")
        elif cmd_lower.startswith("/flip"):
            self._play_coin_flip()
        elif cmd_lower.startswith("/roll"):
            parts = cmd_original.split(maxsplit=1)
            self._play_dice_roll(parts[1] if len(parts) > 1 else None)
        elif cmd_lower == "/omens":
            self.show_omens()
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
            self.console.print(f"[dim {ARES_ASH}]Type /help for available commands[/]")
        
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

        if hasattr(self, '_app') and self._app:
            self._app.invalidate()

        while True:
            try:
                result = response_queue.get(timeout=1)
                self._sudo_state = None
                self._sudo_deadline = 0
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()
                if result:
                    _cprint(f"\n{_DIM}  ✓ Password received (cached for session){_RST}")
                else:
                    _cprint(f"\n{_DIM}  ⏭ Skipped{_RST}")
                return result
            except queue.Empty:
                remaining = self._sudo_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()

        self._sudo_state = None
        self._sudo_deadline = 0
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()
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

        if hasattr(self, '_app') and self._app:
            self._app.invalidate()

        while True:
            try:
                result = response_queue.get(timeout=1)
                self._approval_state = None
                self._approval_deadline = 0
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()
                return result
            except queue.Empty:
                remaining = self._approval_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()

        self._approval_state = None
        self._approval_deadline = 0
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()
        _cprint(f"\n{_DIM}  ⏱ Timeout — denying command{_RST}")
        return "deny"

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
        # Refresh OAuth credentials if needed (handles key rotation transparently)
        if self.provider == "nous" and not self._ensure_runtime_credentials():
            return None

        # Initialize agent if needed
        if not self._init_agent():
            return None
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        w = self.console.width
        separator = build_speed_line(w, self._banner_phase) if self._ares_skin_active() else ("─" * w)
        separator_color = self._mod_response_frame_color() if self._ares_skin_active() else _GOLD
        if self._uses_managed_banner():
            prompt_color = _ansi_fg_hex(ARES_SAND if self._ares_skin_active() else "#FFF8DC")
            glyph = get_mod_agent_glyph() if self._ares_skin_active() else "•"
            self._append_managed_output(
                f"{separator_color}{separator}{_RST}\n\n"
                f"{separator_color}{glyph}{_RST} {prompt_color}{message}{_RST}\n"
            )
        else:
            _cprint(f"{separator_color}{separator}{_RST}")
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
                            if self._uses_managed_banner():
                                notice = f"\n{_ansi_fg_hex(ARES_ASH)}⚡ New message detected, interrupting...{_RST}\n"
                                self._append_managed_output(notice)
                            else:
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
                if self._ares_skin_active():
                    self._refresh_lore()
                    top, subtitle, bot = build_scroll_frame(w, self._lore_state, self._banner_phase)
                    body = self._format_hermes_scroll_body(response)
                    trickster_note = maybe_create_trickster_note(
                        message,
                        enabled=self.easter_eggs,
                    )
                    frame_color = self._mod_response_frame_color()
                    subtle_color = self._mod_response_subtle_color()
                    rendered = f"\n{frame_color}{top}{_RST}\n{subtle_color}{subtitle}{_RST}\n{body}"
                    if trickster_note:
                        rendered += f"\n\n{subtle_color}╎ {trickster_note}{_RST}"
                    rendered += f"\n\n{frame_color}{bot}{_RST}"
                    if self._uses_managed_banner():
                        self._append_managed_output(rendered + "\n")
                    else:
                        _cprint(rendered)
                else:
                    if self._ares_skin_active():
                        label = f" {get_mod_agent_glyph()} {get_mod_assistant_name()} "
                    else:
                        label = " ⚕ Hermes "
                    fill = w - 2 - len(label)  # 2 for ╭ and ╮
                    top = f"{_GOLD}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}"
                    bot = f"{_GOLD}╰{'─' * (w - 2)}╯{_RST}"

                    # Render box + response as a single _cprint call so
                    # nothing can interleave between the box borders.
                    rendered = f"\n{top}\n{response}\n\n{bot}"
                    if self._uses_managed_banner():
                        self._append_managed_output(rendered + "\n")
                    else:
                        _cprint(rendered)
            
            # Combine all interrupt messages (user may have typed multiple while waiting)
            # and re-queue as one prompt for process_loop
            if pending_message and hasattr(self, '_pending_input'):
                all_parts = [pending_message]
                while not self._interrupt_queue.empty():
                    try:
                        extra = self._interrupt_queue.get_nowait()
                        if extra:
                            all_parts.append(extra)
                    except queue.Empty:
                        break
                combined = "\n".join(all_parts)
                queued_note = f"\n📨 Queued: '{combined[:50]}{'...' if len(combined) > 50 else ''}'"
                if self._uses_managed_banner():
                    self._append_managed_output(f"{_ansi_fg_hex(ARES_ASH)}{queued_note}{_RST}\n")
                else:
                    print(queued_note)
                self._pending_input.put(combined)
            
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
            print("Goodbye! ⚔")

    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        if self._uses_managed_banner():
            self.console.clear()
            welcome_message = get_mod_welcome_message()
            welcome_color = ARES_SAND if self._ares_skin_active() else "#FFF8DC"
            self.console.print(f"[{welcome_color}]{welcome_message}[/]")
            self.console.print()
        else:
            self.show_banner()
            welcome_message = (
                get_mod_welcome_message()
                if self._ares_skin_active()
                else "Welcome to Hermes Agent! Type your message or /help for commands."
            )
            welcome_color = ARES_SAND if self._ares_skin_active() else "#FFF8DC"
            self.console.print(f"[{welcome_color}]{welcome_message}[/]")
            self.console.print()
        
        # State for async operation
        self._agent_running = False
        self._pending_input = queue.Queue()     # For normal input (commands + new queries)
        self._interrupt_queue = queue.Queue()   # For messages typed while agent is running
        self._should_exit = False
        self._last_ctrl_c_time = 0  # Track double Ctrl+C for force exit
        self._banner_last_refresh = 0.0
        self._managed_output_ansi = ""

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
                self._should_exit = True
                event.app.exit()
        
        @kb.add('c-d')
        def handle_ctrl_d(event):
            """Handle Ctrl+D - exit."""
            self._should_exit = True
            event.app.exit()
        
        # Dynamic prompt: shows the active mod symbol when the agent is working,
        # or answer prompt when clarify freetext mode is active.
        cli_ref = self

        def _rule_line(offset: int = 0) -> str:
            width = max((cli_ref.console.width or 80) - 2, 24)
            if not cli_ref._ares_skin_active():
                return "─" * width
            base = ["─"] * width
            marker = (cli_ref._ui_phase * 2 + (offset * 11)) % width
            head = list("▲╸╸") if cli_ref._agent_running else list("⚔╸")
            for idx, char in enumerate(head):
                base[(marker + idx) % width] = char
            if not cli_ref._agent_running:
                partner = (marker + (width // 2)) % width
                base[partner] = "△"
            if width > 24:
                base[(marker + (width // 3)) % width] = "╳"
                base[(marker + (2 * width // 3)) % width] = "╳"
            return "".join(base)

        def get_prompt():
            if cli_ref._sudo_state:
                return [('class:sudo-prompt', '🔐 ❯ ')]
            if cli_ref._approval_state:
                return [('class:prompt-working', '⚠ ❯ ')]
            if cli_ref._clarify_freetext:
                return [('class:clarify-selected', '✎ ❯ ')]
            if cli_ref._clarify_state:
                return [('class:prompt-working', '? ❯ ')]
            if cli_ref._ares_skin_active():
                if cli_ref._agent_running:
                    frames = get_mod_prompt_frames(active=True)
                    return [('class:prompt-flight', frames[cli_ref._ui_phase % len(frames)])]
                idle_frames = get_mod_prompt_frames(active=False)
                return [('class:prompt', idle_frames[cli_ref._ui_phase % len(idle_frames)])]
            if cli_ref._agent_running:
                return [('class:prompt-working', '⚔ ❯ ')]
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
                available_width = (cli_ref.console.width or 80) - 4  # subtract prompt width
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

        def _on_text_changed(buf):
            """Detect large pastes and collapse them to a file reference."""
            text = buf.text
            if text:
                cli_ref._freeze_managed_banner()
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
                return "type a message + Enter to interrupt, Ctrl+C to cancel"
            if cli_ref._ares_skin_active():
                return get_mod_placeholder_text()
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

            if cli_ref._ares_skin_active():
                glyphs = ['▲', '△', '✦', '⚔']
                glyph = glyphs[cli_ref._ui_phase % len(glyphs)]
                telemetry = build_relay_telemetry(
                    cli_ref._lore_state,
                    cli_ref._ui_phase,
                    min((cli_ref.console.width or 80) - 4, 60),
                    active=cli_ref._agent_running,
                )
                if cli_ref._agent_running:
                    return [
                        ('class:hint-bar', get_mod_hint_bar(True, glyph, len(cli_ref._lore_state.orbiting_skills))),
                        ('', '\n'),
                        ('class:hint-telemetry', f'  {telemetry}'),
                    ]
                orbit_count = len(cli_ref._lore_state.orbiting_skills)
                return [
                    ('class:hint-bar', get_mod_hint_bar(False, glyph, orbit_count)),
                    ('', '\n'),
                    ('class:hint-telemetry', f'  {telemetry}'),
                ]

            return []

        def get_hint_height():
            if cli_ref._sudo_state or cli_ref._approval_state or cli_ref._clarify_state:
                return 1
            if cli_ref._ares_skin_active():
                return 2
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
            clarify_title = (
                f"{get_mod_assistant_name()} needs your input"
                if cli_ref._ares_skin_active()
                else "Hermes needs your input"
            )
            lines.append(('class:clarify-title', clarify_title))
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

        # --- Sudo password: display widget ---

        def _get_sudo_display():
            state = cli_ref._sudo_state
            if not state:
                return []
            lines = []
            lines.append(('class:sudo-border', '╭─ '))
            lines.append(('class:sudo-title', '🔐 Sudo Password Required'))
            lines.append(('class:sudo-border', ' ──────────────────────────╮\n'))
            lines.append(('class:sudo-border', '│\n'))
            lines.append(('class:sudo-border', '│  '))
            lines.append(('class:sudo-text', 'Enter password below (hidden), or press Enter to skip'))
            lines.append(('', '\n'))
            lines.append(('class:sudo-border', '│\n'))
            lines.append(('class:sudo-border', '╰──────────────────────────────────────────────────╯\n'))
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
            lines.append(('class:approval-border', ' ───────────────────────────────╮\n'))
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
            lines.append(('class:approval-border', '╰──────────────────────────────────────────────────────╯\n'))
            return lines

        approval_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_approval_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._approval_state is not None),
        )

        managed_banner = ConditionalContainer(
            Window(
                content=FormattedTextControl(lambda: _PT_ANSI(cli_ref._build_managed_banner_ansi())),
                height=lambda: cli_ref._managed_banner_height(),
                wrap_lines=False,
            ),
            filter=Condition(lambda: cli_ref._uses_managed_banner()),
        )

        managed_output = ConditionalContainer(
            Window(
                content=FormattedTextControl(lambda: _PT_ANSI(getattr(cli_ref, "_managed_output_ansi", ""))),
                height=lambda: cli_ref._managed_output_height(),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._uses_managed_banner() and bool(getattr(cli_ref, "_managed_output_ansi", ""))),
        )

        # Horizontal rules above and below the input (bronze, 1 line each).
        # The bottom rule moves down as the TextArea grows with newlines.
        input_rule_top = Window(
            content=FormattedTextControl(lambda: [('class:input-rule', _rule_line(0))]),
            height=1,
        )
        input_rule_bot = Window(
            content=FormattedTextControl(lambda: [('class:input-rule', _rule_line(1))]),
            height=1,
        )

        # Layout: interactive prompt widgets + ruled input at bottom.
        # The sudo, approval, and clarify widgets appear above the input when
        # the corresponding interactive prompt is active.
        layout = Layout(
            HSplit([
                managed_banner,
                managed_output,
                Window(height=0),
                sudo_widget,
                approval_widget,
                clarify_widget,
                spacer,
                input_rule_top,
                input_area,
                input_rule_bot,
                CompletionsMenu(max_height=12, scroll_offset=1),
            ])
        )

        style = self._build_prompt_style()
        
        # Create the application
        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
        )
        self._app = app  # Store reference for clarify_callback

        if self._ares_skin_active():
            async def animation_loop():
                while not self._should_exit:
                    managed_frozen = self._uses_managed_banner() and self._managed_banner_frozen
                    if self.ambient_motion or self._agent_running:
                        self._ui_phase = (self._ui_phase + 1) % 10_000
                        try:
                            if app.is_running:
                                app.invalidate()
                        except Exception:
                            pass
                    if (
                        app.is_running
                        and self.ambient_motion
                        and self._uses_managed_banner()
                        and mod_has_animated_hero(self.skin)
                        and not self._agent_running
                        and not self._clarify_state
                        and not self._clarify_freetext
                        and not self._sudo_state
                        and not self._approval_state
                        and not managed_frozen
                    ):
                        now = time.monotonic()
                        interval = get_mod_hero_animation_interval(self.skin)
                        if now - self._banner_last_refresh >= interval:
                            try:
                                self._banner_phase += 1
                                app.invalidate()
                            except Exception:
                                pass
                            self._banner_last_refresh = now
                    await asyncio.sleep(0.12 if self._agent_running else 0.22)

            app.pre_run_callables.append(lambda: app.create_background_task(animation_loop()))
        
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

                    self._freeze_managed_banner()
                    
                    # Check for commands
                    if user_input.startswith("/"):
                        if self._uses_managed_banner():
                            self._append_managed_output(f"\n{_ansi_fg_hex(ARES_ASH)}⚙️  {user_input}{_RST}\n")
                        else:
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

                    if self._maybe_handle_local_ritual(user_input):
                        continue
                    
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

def _resolve_invoked_skin(requested_skin: str | None) -> str | None:
    """Prefer an explicit skin, otherwise infer it from the launcher name."""
    if requested_skin:
        return requested_skin
    invoked_as = Path(sys.argv[0]).name.lower()
    if invoked_as.startswith("posideon") or invoked_as.startswith("poseidon"):
        return "posideon"
    if invoked_as.startswith("sisyphus"):
        return "sisyphus"
    if invoked_as.startswith("ares"):
        return "ares"
    if invoked_as.startswith("hermes"):
        return "hermes"
    return None


def main(
    query: str = None,
    q: str = None,
    toolsets: str = None,
    model: str = None,
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
    max_turns: int = 60,
    verbose: bool = False,
    compact: bool = False,
    skin: str = None,
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
        skin: Visual skin name ("hermes", "ares", "posideon", or "sisyphus")
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
    skin = _resolve_invoked_skin(skin)
    
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
        skin=skin,
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
        if not cli._maybe_handle_local_ritual(query):
            cli.chat(query)
        cli._print_exit_summary()
        return
    
    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    fire.Fire(main)
