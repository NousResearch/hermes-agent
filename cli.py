#!/usr/bin/env python3
"""
Hermes Agent CLI - Interactive Terminal Interface

A beautiful command-line interface for the Hermes Agent, inspired by Claude Code.
Features ASCII art branding, interactive REPL, toolset selection, and rich formatting.

Usage:
    python cli.py                          # Start interactive mode with all tools
    python cli.py --toolsets web,terminal  # Start with specific toolsets
    python cli.py --skills hermes-agent-dev,github-auth
    python cli.py -q "your question"       # Single query mode
    python cli.py --list-tools             # List available tools and exit
"""

import logging
import os
import shutil
import sys
import json
import atexit
import tempfile
import time
import uuid
import textwrap
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Suppress startup messages for clean CLI experience
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
from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
try:
    from prompt_toolkit.cursor_shapes import CursorShape
    _STEADY_CURSOR = CursorShape.BLOCK  # Non-blinking block cursor
except (ImportError, AttributeError):
    _STEADY_CURSOR = None
import threading
import queue

from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    format_duration_compact,
    format_token_count_compact,
)
from hermes_cli.banner import _format_context_length

_COMMAND_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_constants import get_hermes_home, display_hermes_home, OPENROUTER_BASE_URL
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)


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
        path = _hermes_home / path
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
    """Parse a reasoning effort level into an OpenRouter reasoning config dict."""
    from hermes_constants import parse_reasoning_effort
    result = parse_reasoning_effort(effort)
    if effort and effort.strip() and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def load_cli_config() -> Dict[str, Any]:
    """
    Load CLI configuration from config files.
    
    Config lookup order:
    1. ~/.hermes/config.yaml (user config - preferred)
    2. ./cli-config.yaml (project config - fallback)
    
    Environment variables take precedence over config file values.
    Returns default values if no config file exists.
    """
    # Check user config first ({HERMES_HOME}/config.yaml)
    user_config_path = _hermes_home / 'config.yaml'
    project_config_path = Path(__file__).parent / 'cli-config.yaml'

    # Use user config if it exists, otherwise project config
    if user_config_path.exists():
        config_path = user_config_path
    else:
        config_path = project_config_path

    # Default configuration
    defaults = {
        "model": {
            "default": "",
            "base_url": "",
            "provider": "auto",
        },
        "terminal": {
            "env_type": "local",
            "cwd": ".",  # "." is resolved to os.getcwd() at runtime
            "timeout": 60,
            "lifetime_seconds": 300,
            "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_forward_env": [],
            "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
            "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_volumes": [],  # host:container volume mounts for Docker backend
            "docker_mount_cwd_to_workspace": False,  # explicit opt-in only; default off for sandbox isolation
        },
        "browser": {
            "inactivity_timeout": 120,  # Auto-cleanup inactive browser sessions after 2 min
            "record_sessions": False,  # Auto-record browser sessions as WebM videos
        },
        "compression": {
            "enabled": True,      # Auto-compress when approaching context limit
            "threshold": 0.50,    # Compress at 50% of model's context limit
            "summary_model": "",  # Model for summaries (empty = use main model)
        },
        "smart_model_routing": {
            "enabled": False,
            "max_simple_chars": 160,
            "max_simple_words": 28,
            "cheap_model": {},
        },
        "agent": {
            "max_turns": 90,  # Default max tool-calling iterations (shared with subagents)
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

        "display": {
            "compact": False,
            "resume_display": "full",
            "show_reasoning": False,
            "streaming": True,
            "busy_input_mode": "interrupt",

            "skin": "default",
        },
        "clarify": {
            "timeout": 120,  # Seconds to wait for a clarify answer before auto-proceeding
        },
        "code_execution": {
            "timeout": 300,    # Max seconds a sandbox script can run before being killed (5 min)
            "max_tool_calls": 50,  # Max RPC tool calls per execution
        },
        "auxiliary": {
            "vision": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
            },
            "web_extract": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
            },
        },
        "delegation": {
            "max_iterations": 45,  # Max tool-calling turns per child agent
            "default_toolsets": ["terminal", "file", "web"],  # Default toolsets for subagents
            "model": "",       # Subagent model override (empty = inherit parent model)
            "provider": "",    # Subagent provider override (empty = inherit parent provider)
            "base_url": "",    # Direct OpenAI-compatible endpoint for subagents
            "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
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
                    # If the user config sets model.model but not model.default,
                    # promote model.model to model.default so the user's explicit
                    # choice isn't shadowed by the hardcoded default.  Without this,
                    # profile configs that only set "model:" (not "default:") silently
                    # fall back to claude-opus because the merge preserves the
                    # hardcoded default and HermesCLI.__init__ checks "default" first.
                    if "model" in file_config["model"] and "default" not in file_config["model"]:
                        defaults["model"]["default"] = file_config["model"]["model"]

            # Legacy root-level provider/base_url fallback.
            # Some users (or old code) put provider: / base_url: at the
            # config root instead of inside the model: section.  These are
            # only used as a FALLBACK when model.provider / model.base_url
            # is not already set — never as an override.  The canonical
            # location is model.provider (written by `hermes model`).
            if not defaults["model"].get("provider"):
                root_provider = file_config.get("provider")
                if root_provider:
                    defaults["model"]["provider"] = root_provider
            if not defaults["model"].get("base_url"):
                root_base_url = file_config.get("base_url")
                if root_base_url:
                    defaults["model"]["base_url"] = root_base_url
            
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
            
            # Handle legacy root-level max_turns (backwards compat) - copy to
            # agent.max_turns whenever the nested key is missing.
            agent_file_config = file_config.get("agent")
            if "max_turns" in file_config and not (
                isinstance(agent_file_config, dict)
                and agent_file_config.get("max_turns") is not None
            ):
                defaults["agent"]["max_turns"] = file_config["max_turns"]
        except Exception as e:
            logger.warning("Failed to load cli-config.yaml: %s", e)

    # Expand ${ENV_VAR} references in config values before bridging to env vars.
    from hermes_cli.config import _expand_env_vars
    defaults = _expand_env_vars(defaults)

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
        "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
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
        "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
        "sandbox_dir": "TERMINAL_SANDBOX_DIR",
        # Persistent shell (non-local backends)
        "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
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
    
    # Apply auxiliary model/direct-endpoint overrides to environment variables.
    # Vision and web_extract each have their own provider/model/base_url/api_key tuple.
    # Compression config is read directly from config.yaml by run_agent.py and
    # auxiliary_client.py — no env var bridging needed.
    # Only set env vars for non-empty / non-default values so auto-detection
    # still works.
    auxiliary_config = defaults.get("auxiliary", {})
    auxiliary_task_env = {
        # config key → env var mapping
        "vision": {
            "provider": "AUXILIARY_VISION_PROVIDER",
            "model": "AUXILIARY_VISION_MODEL",
            "base_url": "AUXILIARY_VISION_BASE_URL",
            "api_key": "AUXILIARY_VISION_API_KEY",
        },
        "web_extract": {
            "provider": "AUXILIARY_WEB_EXTRACT_PROVIDER",
            "model": "AUXILIARY_WEB_EXTRACT_MODEL",
            "base_url": "AUXILIARY_WEB_EXTRACT_BASE_URL",
            "api_key": "AUXILIARY_WEB_EXTRACT_API_KEY",
        },
        "approval": {
            "provider": "AUXILIARY_APPROVAL_PROVIDER",
            "model": "AUXILIARY_APPROVAL_MODEL",
            "base_url": "AUXILIARY_APPROVAL_BASE_URL",
            "api_key": "AUXILIARY_APPROVAL_API_KEY",
        },
    }
    
    for task_key, env_map in auxiliary_task_env.items():
        task_cfg = auxiliary_config.get(task_key, {})
        if not isinstance(task_cfg, dict):
            continue
        prov = str(task_cfg.get("provider", "")).strip()
        model = str(task_cfg.get("model", "")).strip()
        base_url = str(task_cfg.get("base_url", "")).strip()
        api_key = str(task_cfg.get("api_key", "")).strip()
        if prov and prov != "auto":
            os.environ[env_map["provider"]] = prov
        if model:
            os.environ[env_map["model"]] = model
        if base_url:
            os.environ[env_map["base_url"]] = base_url
        if api_key:
            os.environ[env_map["api_key"]] = api_key
    
    # Security settings
    security_config = defaults.get("security", {})
    if isinstance(security_config, dict):
        redact = security_config.get("redact_secrets")
        if redact is not None:
            os.environ["HERMES_REDACT_SECRETS"] = str(redact).lower()

    return defaults

# Load configuration at module startup
CLI_CONFIG = load_cli_config()

# Initialize the skin engine from config
try:
    from hermes_cli.skin_engine import init_skin_from_config
    init_skin_from_config(CLI_CONFIG)
except Exception:
    pass  # Skin engine is optional — default skin used if unavailable

# Initialize tool preview length from config
try:
    from agent.display import set_tool_preview_max_len
    _tpl = CLI_CONFIG.get("display", {}).get("tool_preview_length", 0)
    set_tool_preview_max_len(int(_tpl) if _tpl else 0)
except Exception:
    pass

# Neuter AsyncHttpxClientWrapper.__del__ before any AsyncOpenAI clients are
# created.  The SDK's __del__ schedules aclose() on asyncio.get_running_loop()
# which, during CLI idle time, finds prompt_toolkit's event loop and tries to
# close TCP transports bound to dead worker loops — producing
# "Event loop is closed" / "Press ENTER to continue..." errors.
try:
    from agent.auxiliary_client import neuter_async_httpx_del
    neuter_async_httpx_del()
except Exception:
    pass

from rich import box as rich_box
from rich.console import Console
from rich.markup import escape as _escape
from rich.panel import Panel
from rich.text import Text as _RichText

import fire

# Import the agent and tool systems
from run_agent import AIAgent
from model_tools import get_tool_definitions, get_toolset_for_tool

# Extracted CLI modules (Phase 3)
from hermes_cli.banner import build_welcome_banner
from hermes_cli.commands import SlashCommandCompleter, SlashCommandAutoSuggest
from toolsets import get_all_toolsets, get_toolset_info, validate_toolset

# Cron job system for scheduled tasks (execution is handled by the gateway)
from cron import get_job

# Resource cleanup imports for safe shutdown (terminal VMs, browser sessions)
from tools.terminal_tool import cleanup_all_environments as _cleanup_all_terminals
from tools.terminal_tool import set_sudo_password_callback, set_approval_callback
from tools.skills_tool import set_secret_capture_callback
from hermes_cli.callbacks import prompt_for_secret
from tools.browser_tool import _emergency_cleanup_all_sessions as _cleanup_all_browsers

# Guard to prevent cleanup from running multiple times on exit
_cleanup_done = False
# Weak reference to the active AIAgent for memory provider shutdown at exit
_active_agent_ref = None

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
    # Close cached auxiliary LLM clients (sync + async) so that
    # AsyncHttpxClientWrapper.__del__ doesn't fire on a closed event loop
    # and trigger prompt_toolkit's "Press ENTER to continue..." handler.
    try:
        from agent.auxiliary_client import shutdown_cached_clients
        shutdown_cached_clients()
    except Exception:
        pass
    # Shut down memory provider (on_session_end + shutdown_all) at actual
    # session boundary — NOT per-turn inside run_conversation().
    try:
        if _active_agent_ref and hasattr(_active_agent_ref, 'shutdown_memory_provider'):
            _active_agent_ref.shutdown_memory_provider(
                getattr(_active_agent_ref, 'conversation_history', None) or []
            )
    except Exception:
        pass


# =============================================================================
# Git Worktree Isolation (#652)
# =============================================================================

# Tracks the active worktree for cleanup on exit
_active_worktree: Optional[Dict[str, str]] = None


def _git_repo_root() -> Optional[str]:
    """Return the git repo root for CWD, or None if not in a repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _path_is_within_root(path: Path, root: Path) -> bool:
    """Return True when a resolved path stays within the expected root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _setup_worktree(repo_root: str = None) -> Optional[Dict[str, str]]:
    """Create an isolated git worktree for this CLI session.

    Returns a dict with worktree metadata on success, None on failure.
    The dict contains: path, branch, repo_root.
    """
    import subprocess

    repo_root = repo_root or _git_repo_root()
    if not repo_root:
        print("\033[31m✗ --worktree requires being inside a git repository.\033[0m")
        print("  cd into your project repo first, then run hermes -w")
        return None

    short_id = uuid.uuid4().hex[:8]
    wt_name = f"hermes-{short_id}"
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    wt_path = worktrees_dir / wt_name

    # Ensure .worktrees/ is in .gitignore
    gitignore = Path(repo_root) / ".gitignore"
    _ignore_entry = ".worktrees/"
    try:
        existing = gitignore.read_text() if gitignore.exists() else ""
        if _ignore_entry not in existing.splitlines():
            with open(gitignore, "a") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{_ignore_entry}\n")
    except Exception as e:
        logger.debug("Could not update .gitignore: %s", e)

    # Create the worktree
    try:
        result = subprocess.run(
            ["git", "worktree", "add", str(wt_path), "-b", branch_name, "HEAD"],
            capture_output=True, text=True, timeout=30, cwd=repo_root,
        )
        if result.returncode != 0:
            print(f"\033[31m✗ Failed to create worktree: {result.stderr.strip()}\033[0m")
            return None
    except Exception as e:
        print(f"\033[31m✗ Failed to create worktree: {e}\033[0m")
        return None

    # Copy files listed in .worktreeinclude (gitignored files the agent needs)
    include_file = Path(repo_root) / ".worktreeinclude"
    if include_file.exists():
        try:
            repo_root_resolved = Path(repo_root).resolve()
            wt_path_resolved = wt_path.resolve()
            for line in include_file.read_text().splitlines():
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                src = Path(repo_root) / entry
                dst = wt_path / entry
                # Prevent path traversal and symlink escapes: both the resolved
                # source and the resolved destination must stay inside their
                # expected roots before any file or symlink operation happens.
                try:
                    src_resolved = src.resolve(strict=False)
                    dst_resolved = dst.resolve(strict=False)
                except (OSError, ValueError):
                    logger.debug("Skipping invalid .worktreeinclude entry: %s", entry)
                    continue
                if not _path_is_within_root(src_resolved, repo_root_resolved):
                    logger.warning("Skipping .worktreeinclude entry outside repo root: %s", entry)
                    continue
                if not _path_is_within_root(dst_resolved, wt_path_resolved):
                    logger.warning("Skipping .worktreeinclude entry that escapes worktree: %s", entry)
                    continue
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                elif src.is_dir():
                    # Symlink directories (faster, saves disk)
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(str(src_resolved), str(dst))
        except Exception as e:
            logger.debug("Error copying .worktreeinclude entries: %s", e)

    info = {
        "path": str(wt_path),
        "branch": branch_name,
        "repo_root": repo_root,
    }

    print(f"\033[32m✓ Worktree created:\033[0m {wt_path}")
    print(f"  Branch: {branch_name}")

    return info


def _cleanup_worktree(info: Dict[str, str] = None) -> None:
    """Remove a worktree and its branch on exit.

    If the worktree has uncommitted changes, warn and keep it.
    """
    global _active_worktree
    info = info or _active_worktree
    if not info:
        return

    import subprocess

    wt_path = info["path"]
    branch = info["branch"]
    repo_root = info["repo_root"]

    if not Path(wt_path).exists():
        return

    # Check for uncommitted changes
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, cwd=wt_path,
        )
        has_changes = bool(status.stdout.strip())
    except Exception:
        has_changes = True  # Assume dirty on error — don't delete

    if has_changes:
        print(f"\n\033[33m⚠ Worktree has uncommitted changes, keeping: {wt_path}\033[0m")
        print(f"  To clean up manually: git worktree remove {wt_path}")
        _active_worktree = None
        return

    # Remove worktree
    try:
        subprocess.run(
            ["git", "worktree", "remove", wt_path, "--force"],
            capture_output=True, text=True, timeout=15, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("Failed to remove worktree: %s", e)

    # Delete the branch (only if it was never pushed / has no upstream)
    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            capture_output=True, text=True, timeout=10, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("Failed to delete branch %s: %s", branch, e)

    _active_worktree = None
    print(f"\033[32m✓ Worktree cleaned up: {wt_path}\033[0m")


def _prune_stale_worktrees(repo_root: str, max_age_hours: int = 24) -> None:
    """Remove worktrees older than max_age_hours that have no uncommitted changes.

    Runs silently on startup to clean up after crashed/killed sessions.
    """
    import subprocess
    import time

    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        return

    now = time.time()
    cutoff = now - (max_age_hours * 3600)

    for entry in worktrees_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("hermes-"):
            continue

        # Check age
        try:
            mtime = entry.stat().st_mtime
            if mtime > cutoff:
                continue  # Too recent — skip
        except Exception:
            continue

        # Check for uncommitted changes
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            if status.stdout.strip():
                continue  # Has changes — skip
        except Exception:
            continue  # Can't check — skip

        # Safe to remove
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()

            subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, timeout=15, cwd=repo_root,
            )
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, timeout=10, cwd=repo_root,
                )
            logger.debug("Pruned stale worktree: %s", entry.name)
        except Exception as e:
            logger.debug("Failed to prune worktree %s: %s", entry.name, e)

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
_GOLD = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold — matches Rich Panel gold
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"

def _accent_hex() -> str:
    """Return the active skin accent color for legacy CLI output lines."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin().get_color("ui_accent", "#FFBF00")
    except Exception:
        return "#FFBF00"


def _rich_text_from_ansi(text: str) -> _RichText:
    """Safely render assistant/tool output that may contain ANSI escapes.

    Using Rich Text.from_ansi preserves literal bracketed text like
    ``[not markup]`` while still interpreting real ANSI color codes.
    """
    return _RichText.from_ansi(text or "")


def _cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's native renderer.

    Raw ANSI escapes written via print() are swallowed by patch_stdout's
    StdoutProxy.  Routing through print_formatted_text(ANSI(...)) lets
    prompt_toolkit parse the escapes and render real colors.
    """
    _pt_print(_PT_ANSI(text))


# ---------------------------------------------------------------------------
# File-drop detection — extracted as a pure function for testability.
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.gif', '.webp',
    '.bmp', '.tiff', '.tif', '.svg', '.ico',
})


def _detect_file_drop(user_input: str) -> "dict | None":
    """Detect if *user_input* is a dragged/pasted file path, not a slash command.

    When a user drags a file into the terminal, macOS pastes the absolute path
    (e.g. ``/Users/roland/Desktop/file.png``) which starts with ``/`` and would
    otherwise be mistaken for a slash command.

    Returns a dict on match::

        {
            "path": Path,          # resolved file path
            "is_image": bool,      # True when suffix is a known image type
            "remainder": str,      # any text after the path
        }

    Returns ``None`` when the input is not a real file path.
    """
    if not isinstance(user_input, str) or not user_input.startswith("/"):
        return None

    # Walk the string absorbing backslash-escaped spaces ("\ ").
    raw = user_input
    pos = 0
    while pos < len(raw):
        ch = raw[pos]
        if ch == '\\' and pos + 1 < len(raw) and raw[pos + 1] == ' ':
            pos += 2  # skip escaped space
        elif ch == ' ':
            break
        else:
            pos += 1

    first_token_raw = raw[:pos]
    first_token = first_token_raw.replace('\\ ', ' ')
    drop_path = Path(first_token)

    if not drop_path.exists() or not drop_path.is_file():
        return None

    remainder = raw[pos:].strip()
    return {
        "path": drop_path,
        "is_image": drop_path.suffix.lower() in _IMAGE_EXTENSIONS,
        "remainder": remainder,
    }


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
        self._inner = Console(
            file=self._buffer,
            force_terminal=True,
            color_system="truecolor",
            highlight=False,
        )

    def print(self, *args, **kwargs):
        self._buffer.seek(0)
        self._buffer.truncate()
        # Read terminal width at render time so panels adapt to current size
        self._inner.width = shutil.get_terminal_size((80, 24)).columns
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
# Note: built dynamically by _build_compact_banner() to fit terminal width
COMPACT_BANNER = """
[bold #FFD700]╔══════════════════════════════════════════════════════════════╗[/]
[bold #FFD700]║[/]  [#FFBF00]⚕ NOUS HERMES[/] [dim #B8860B]- AI Agent Framework[/]              [bold #FFD700]║[/]
[bold #FFD700]║[/]  [#CD7F32]Messenger of the Digital Gods[/]    [dim #B8860B]Nous Research[/]   [bold #FFD700]║[/]
[bold #FFD700]╚══════════════════════════════════════════════════════════════╝[/]
"""


def _build_compact_banner() -> str:
    """Build a compact banner that fits the current terminal width."""
    w = min(shutil.get_terminal_size().columns - 2, 64)
    if w < 30:
        return "\n[#FFBF00]⚕ NOUS HERMES[/] [dim #B8860B]- Nous Research[/]\n"
    inner = w - 2  # inside the box border
    bar = "═" * w
    line1 = "⚕ NOUS HERMES - AI Agent Framework"
    line2 = "Messenger of the Digital Gods  ·  Nous Research"
    # Truncate and pad to fit
    line1 = line1[:inner - 2].ljust(inner - 2)
    line2 = line2[:inner - 2].ljust(inner - 2)
    return (
        f"\n[bold #FFD700]╔{bar}╗[/]\n"
        f"[bold #FFD700]║[/] [#FFBF00]{line1}[/] [bold #FFD700]║[/]\n"
        f"[bold #FFD700]║[/] [dim #B8860B]{line2}[/] [bold #FFD700]║[/]\n"
        f"[bold #FFD700]╚{bar}╝[/]\n"
    )



# ============================================================================
# Slash-command detection helper
# ============================================================================

def _looks_like_slash_command(text: str) -> bool:
    """Return True if *text* looks like a slash command, not a file path.

    Slash commands are ``/help``, ``/model gpt-4``, ``/q``, etc.
    File paths like ``/Users/ironin/file.md:45-46 can you fix this?``
    also start with ``/`` but contain additional ``/`` characters in
    the first whitespace-delimited word.  This helper distinguishes
    the two so that pasted paths are sent to the agent instead of
    triggering "Unknown command".
    """
    if not text or not text.startswith("/"):
        return False
    first_word = text.split()[0]
    # After stripping the leading /, a command name has no slashes.
    # A path like /Users/foo/bar.md always does.
    return "/" not in first_word[1:]


# ============================================================================
# Skill Slash Commands — dynamic commands generated from installed skills
# ============================================================================

from agent.skill_commands import (
    scan_skill_commands,
    build_skill_invocation_message,
    build_plan_path,
    build_preloaded_skills_prompt,
)

_skill_commands = scan_skill_commands()


def _get_plugin_cmd_handler_names() -> set:
    """Return plugin command names (without slash prefix) for dispatch matching."""
    try:
        from hermes_cli.plugins import get_plugin_manager
        return set(get_plugin_manager()._plugin_commands.keys())
    except Exception:
        return set()


def _parse_skills_argument(skills: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a CLI skills flag into a deduplicated list of skill identifiers."""
    if not skills:
        return []

    if isinstance(skills, str):
        raw_values = [skills]
    elif isinstance(skills, (list, tuple)):
        raw_values = [str(item) for item in skills if item is not None]
    else:
        raw_values = [str(skills)]

    parsed: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            normalized = part.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            parsed.append(normalized)
    return parsed


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
    user_config_path = _hermes_home / 'config.yaml'
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
        
        # Save back atomically — write to temp file + fsync + os.replace
        # so an interrupt never leaves config.yaml truncated or empty.
        from utils import atomic_yaml_write
        atomic_yaml_write(config_path, config)
        
        # Enforce owner-only permissions on config files (contain API keys)
        try:
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass
        
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
        checkpoints: bool = False,
        pass_session_id: bool = False,
    ):
        """
        Initialize the Hermes CLI.

        Args:
            model: Model to use (default: from env or claude-sonnet)
            toolsets: List of toolsets to enable (default: all)
            provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
            api_key: API key (default: from environment)
            base_url: API base URL (default: OpenRouter)
            max_turns: Maximum tool-calling iterations shared with subagents (default: 90)
            verbose: Enable verbose logging
            compact: Use compact display mode
            resume: Session ID to resume (restores conversation history from SQLite)
            pass_session_id: Include the session ID in the agent's system prompt
        """
        # Initialize Rich console
        self.console = Console()
        self.config = CLI_CONFIG
        self.compact = compact if compact is not None else CLI_CONFIG["display"].get("compact", False)
        # tool_progress: "off", "new", "all", "verbose" (from config.yaml display section)
        # YAML 1.1 parses bare `off` as boolean False — normalise to string.
        _raw_tp = CLI_CONFIG["display"].get("tool_progress", "all")
        self.tool_progress_mode = "off" if _raw_tp is False else str(_raw_tp)
        # resume_display: "full" (show history) | "minimal" (one-liner only)
        self.resume_display = CLI_CONFIG["display"].get("resume_display", "full")
        # bell_on_complete: play terminal bell (\a) when agent finishes a response
        self.bell_on_complete = CLI_CONFIG["display"].get("bell_on_complete", False)
        # show_reasoning: display model thinking/reasoning before the response
        self.show_reasoning = CLI_CONFIG["display"].get("show_reasoning", False)
        # busy_input_mode: "interrupt" (Enter interrupts current run) or "queue" (Enter queues for next turn)
        _bim = CLI_CONFIG["display"].get("busy_input_mode", "interrupt")
        self.busy_input_mode = "queue" if str(_bim).strip().lower() == "queue" else "interrupt"

        self.verbose = verbose if verbose is not None else (self.tool_progress_mode == "verbose")
        
        # streaming: stream tokens to the terminal as they arrive (display.streaming in config.yaml)
        self.streaming_enabled = CLI_CONFIG["display"].get("streaming", False)

        # Inline diff previews for write actions (display.inline_diffs in config.yaml)
        self._inline_diffs_enabled = CLI_CONFIG["display"].get("inline_diffs", True)

        # Streaming display state
        self._stream_buf = ""        # Partial line buffer for line-buffered rendering
        self._stream_started = False  # True once first delta arrives
        self._stream_box_opened = False  # True once the response box header is printed
        self._reasoning_stream_started = False  # True once live reasoning starts streaming
        self._reasoning_preview_buf = ""  # Coalesce tiny reasoning chunks for [thinking] output
        self._pending_edit_snapshots = {}
        
        # Configuration - priority: CLI args > env vars > config file
        # Model comes from: CLI arg or config.yaml (single source of truth).
        # LLM_MODEL/OPENAI_MODEL env vars are NOT checked — config.yaml is
        # authoritative.  This avoids conflicts in multi-agent setups where
        # env vars would stomp each other.
        _model_config = CLI_CONFIG.get("model", {})
        _config_model = (_model_config.get("default") or _model_config.get("model") or "") if isinstance(_model_config, dict) else (_model_config or "")
        _DEFAULT_CONFIG_MODEL = ""
        self.model = model or _config_model or _DEFAULT_CONFIG_MODEL
        # Auto-detect model from local server if still on default
        if self.model == _DEFAULT_CONFIG_MODEL:
            _base_url = (_model_config.get("base_url") or "") if isinstance(_model_config, dict) else ""
            if "localhost" in _base_url or "127.0.0.1" in _base_url:
                from hermes_cli.runtime_provider import _auto_detect_local_model
                _detected = _auto_detect_local_model(_base_url)
                if _detected:
                    self.model = _detected
        # Track whether model was explicitly chosen by the user or fell back
        # to the global default.  Provider-specific normalisation may override
        # the default silently but should warn when overriding an explicit choice.
        # A config model that matches the global fallback is NOT considered an
        # explicit choice — the user just never changed it.  But a config model
        # like "gpt-5.3-codex" IS explicit and must be preserved.
        self._model_is_default = not model and (
            not _config_model or _config_model == _DEFAULT_CONFIG_MODEL
        )

        self._explicit_api_key = api_key
        self._explicit_base_url = base_url

        # Provider selection is resolved lazily at use-time via _ensure_runtime_credentials().
        self.requested_provider = (
            provider
            or CLI_CONFIG["model"].get("provider")
            or os.getenv("HERMES_INFERENCE_PROVIDER")
            or "auto"
        )
        self._provider_source: Optional[str] = None
        self.provider = self.requested_provider
        self.api_mode = "chat_completions"
        self.acp_command: Optional[str] = None
        self.acp_args: list[str] = []
        self.base_url = (
            base_url
            or CLI_CONFIG["model"].get("base_url", "")
            or os.getenv("OPENROUTER_BASE_URL", "")
        ) or None
        # Match key to resolved base_url: OpenRouter URL → prefer OPENROUTER_API_KEY,
        # custom endpoint → prefer OPENAI_API_KEY (issue #560).
        # Note: _ensure_runtime_credentials() re-resolves this before first use.
        if self.base_url and "openrouter.ai" in self.base_url:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
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
            self.max_turns = 90
        
        # Parse and validate toolsets
        self.enabled_toolsets = toolsets
        if toolsets and "all" not in toolsets and "*" not in toolsets:
            # Validate each toolset — MCP server names are added by
            # _get_platform_tools() but aren't registered in TOOLSETS yet
            # (that happens later in _sync_mcp_toolsets), so exclude them.
            mcp_names = set((CLI_CONFIG.get("mcp_servers") or {}).keys())
            invalid = [t for t in toolsets if not validate_toolset(t) and t not in mcp_names]
            if invalid:
                self.console.print(f"[bold red]Warning: Unknown toolsets: {', '.join(invalid)}[/]")
        
        # Filesystem checkpoints: CLI flag > config
        cp_cfg = CLI_CONFIG.get("checkpoints", {})
        if isinstance(cp_cfg, bool):
            cp_cfg = {"enabled": cp_cfg}
        self.checkpoints_enabled = checkpoints or cp_cfg.get("enabled", False)
        self.checkpoint_max_snapshots = cp_cfg.get("max_snapshots", 50)
        self.pass_session_id = pass_session_id
        
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
        
        # Fallback provider chain — tried in order when primary fails after retries.
        # Supports new list format (fallback_providers) and legacy single-dict (fallback_model).
        fb = CLI_CONFIG.get("fallback_providers") or CLI_CONFIG.get("fallback_model") or []
        # Normalize legacy single-dict to a one-element list
        if isinstance(fb, dict):
            fb = [fb] if fb.get("provider") and fb.get("model") else []
        self._fallback_model = fb

        # Optional cheap-vs-strong routing for simple turns
        self._smart_model_routing = CLI_CONFIG.get("smart_model_routing", {}) or {}
        self._active_agent_route_signature = None

        # Agent will be initialized on first use
        self.agent: Optional[AIAgent] = None
        self._app = None  # prompt_toolkit Application (set in run())
        
        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        self._resumed = False
        # Initialize SQLite session store early so /title works before first message
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            logger.warning("Failed to initialize SessionDB — session will NOT be indexed for search: %s", e)
        
        # Deferred title: stored in memory until the session is created in the DB
        self._pending_title: Optional[str] = None
        
        # Session ID: reuse existing one when resuming, otherwise generate fresh
        if resume:
            self.session_id = resume
            self._resumed = True
        else:
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        
        # History file for persistent input recall across sessions
        self._history_file = _hermes_home / ".hermes_history"
        self._last_invalidate: float = 0.0  # throttle UI repaints
        self._app = None

        # State shared by interactive run() and single-query chat mode.
        # These must exist before any direct chat() call because single-query
        # mode does not go through run().
        self._agent_running = False
        self._pending_input = queue.Queue()
        self._interrupt_queue = queue.Queue()
        self._should_exit = False
        self._last_ctrl_c_time = 0
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = 0
        self._sudo_state = None
        self._sudo_deadline = 0
        self._approval_state = None
        self._approval_deadline = 0
        self._approval_lock = threading.Lock()
        self._secret_state = None
        self._secret_deadline = 0
        self._spinner_text: str = ""  # thinking spinner text for TUI
        self._command_running = False
        self._command_status = ""
        self._attached_images: list[Path] = []
        self._image_counter = 0
        self.preloaded_skills: list[str] = []
        self._startup_skills_line_shown = False

        # Voice mode state (also reinitialized inside run() for interactive TUI).
        self._voice_lock = threading.Lock()
        self._voice_mode = False
        self._voice_tts = False
        self._voice_recorder = None
        self._voice_recording = False
        self._voice_processing = False
        self._voice_continuous = False
        self._voice_tts_done = threading.Event()
        self._voice_tts_done.set()

        # Status bar visibility (toggled via /statusbar)
        self._status_bar_visible = True

        # Background task tracking: {task_id: threading.Thread}
        self._background_tasks: Dict[str, threading.Thread] = {}
        self._background_task_counter = 0

    def _invalidate(self, min_interval: float = 0.25) -> None:
        """Throttled UI repaint — prevents terminal blinking on slow/SSH connections."""
        import time as _time
        now = _time.monotonic()
        if hasattr(self, "_app") and self._app and (now - self._last_invalidate) >= min_interval:
            self._last_invalidate = now
            self._app.invalidate()

    def _status_bar_context_style(self, percent_used: Optional[int]) -> str:
        if percent_used is None:
            return "class:status-bar-dim"
        if percent_used >= 95:
            return "class:status-bar-critical"
        if percent_used > 80:
            return "class:status-bar-bad"
        if percent_used >= 50:
            return "class:status-bar-warn"
        return "class:status-bar-good"

    def _build_context_bar(self, percent_used: Optional[int], width: int = 10) -> str:
        safe_percent = max(0, min(100, percent_used or 0))
        filled = round((safe_percent / 100) * width)
        return f"[{('█' * filled) + ('░' * max(0, width - filled))}]"

    def _get_status_bar_snapshot(self) -> Dict[str, Any]:
        model_name = self.model or "unknown"
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        if model_short.endswith(".gguf"):
            model_short = model_short[:-5]
        if len(model_short) > 26:
            model_short = f"{model_short[:23]}..."

        elapsed_seconds = max(0.0, (datetime.now() - self.session_start).total_seconds())
        snapshot = {
            "model_name": model_name,
            "model_short": model_short,
            "duration": format_duration_compact(elapsed_seconds),
            "context_tokens": 0,
            "context_length": None,
            "context_percent": None,
            "session_input_tokens": 0,
            "session_output_tokens": 0,
            "session_cache_read_tokens": 0,
            "session_cache_write_tokens": 0,
            "session_prompt_tokens": 0,
            "session_completion_tokens": 0,
            "session_total_tokens": 0,
            "session_api_calls": 0,
            "compressions": 0,
        }

        agent = getattr(self, "agent", None)
        if not agent:
            return snapshot

        snapshot["session_input_tokens"] = getattr(agent, "session_input_tokens", 0) or 0
        snapshot["session_output_tokens"] = getattr(agent, "session_output_tokens", 0) or 0
        snapshot["session_cache_read_tokens"] = getattr(agent, "session_cache_read_tokens", 0) or 0
        snapshot["session_cache_write_tokens"] = getattr(agent, "session_cache_write_tokens", 0) or 0
        snapshot["session_prompt_tokens"] = getattr(agent, "session_prompt_tokens", 0) or 0
        snapshot["session_completion_tokens"] = getattr(agent, "session_completion_tokens", 0) or 0
        snapshot["session_total_tokens"] = getattr(agent, "session_total_tokens", 0) or 0
        snapshot["session_api_calls"] = getattr(agent, "session_api_calls", 0) or 0

        compressor = getattr(agent, "context_compressor", None)
        if compressor:
            context_tokens = getattr(compressor, "last_prompt_tokens", 0) or 0
            context_length = getattr(compressor, "context_length", 0) or 0
            snapshot["context_tokens"] = context_tokens
            snapshot["context_length"] = context_length or None
            snapshot["compressions"] = getattr(compressor, "compression_count", 0) or 0
            if context_length:
                snapshot["context_percent"] = max(0, min(100, round((context_tokens / context_length) * 100)))

        return snapshot

    @staticmethod
    def _status_bar_display_width(text: str) -> int:
        """Return terminal cell width for status-bar text.

        len() is not enough for prompt_toolkit layout decisions because some
        glyphs can render wider than one Python codepoint. Keeping the status
        bar within the real display width prevents it from wrapping onto a
        second line and leaving behind duplicate rows.
        """
        try:
            from prompt_toolkit.utils import get_cwidth
            return get_cwidth(text or "")
        except Exception:
            return len(text or "")

    @classmethod
    def _trim_status_bar_text(cls, text: str, max_width: int) -> str:
        """Trim status-bar text to a single terminal row."""
        if max_width <= 0:
            return ""
        try:
            from prompt_toolkit.utils import get_cwidth
        except Exception:
            get_cwidth = None

        if cls._status_bar_display_width(text) <= max_width:
            return text

        ellipsis = "..."
        ellipsis_width = cls._status_bar_display_width(ellipsis)
        if max_width <= ellipsis_width:
            return ellipsis[:max_width]

        out = []
        width = 0
        for ch in text:
            ch_width = get_cwidth(ch) if get_cwidth else len(ch)
            if width + ch_width + ellipsis_width > max_width:
                break
            out.append(ch)
            width += ch_width
        return "".join(out).rstrip() + ellipsis

    def _build_status_bar_text(self, width: Optional[int] = None) -> str:
        try:
            snapshot = self._get_status_bar_snapshot()
            if width is None:
                try:
                    from prompt_toolkit.application import get_app
                    width = get_app().output.get_size().columns
                except Exception:
                    width = shutil.get_terminal_size((80, 24)).columns
            percent = snapshot["context_percent"]
            percent_label = f"{percent}%" if percent is not None else "--"
            duration_label = snapshot["duration"]

            if width < 52:
                text = f"⚕ {snapshot['model_short']} · {duration_label}"
                return self._trim_status_bar_text(text, width)
            if width < 76:
                parts = [f"⚕ {snapshot['model_short']}", percent_label]
                parts.append(duration_label)
                return self._trim_status_bar_text(" · ".join(parts), width)

            if snapshot["context_length"]:
                ctx_total = _format_context_length(snapshot["context_length"])
                ctx_used = format_token_count_compact(snapshot["context_tokens"])
                context_label = f"{ctx_used}/{ctx_total}"
            else:
                context_label = "ctx --"

            parts = [f"⚕ {snapshot['model_short']}", context_label, percent_label]
            parts.append(duration_label)
            return self._trim_status_bar_text(" │ ".join(parts), width)
        except Exception:
            return f"⚕ {self.model if getattr(self, 'model', None) else 'Hermes'}"

    def _get_status_bar_fragments(self):
        if not self._status_bar_visible:
            return []
        try:
            snapshot = self._get_status_bar_snapshot()
            # Use prompt_toolkit's own terminal width when running inside the
            # TUI — shutil.get_terminal_size() can return stale or fallback
            # values (especially on SSH) that differ from what prompt_toolkit
            # actually renders, causing the fragments to overflow to a second
            # line and produce duplicated status bar rows over long sessions.
            try:
                from prompt_toolkit.application import get_app
                width = get_app().output.get_size().columns
            except Exception:
                width = shutil.get_terminal_size((80, 24)).columns
            duration_label = snapshot["duration"]

            if width < 52:
                frags = [
                    ("class:status-bar", " ⚕ "),
                    ("class:status-bar-strong", snapshot["model_short"]),
                    ("class:status-bar-dim", " · "),
                    ("class:status-bar-dim", duration_label),
                    ("class:status-bar", " "),
                ]
            else:
                percent = snapshot["context_percent"]
                percent_label = f"{percent}%" if percent is not None else "--"
                if width < 76:
                    frags = [
                        ("class:status-bar", " ⚕ "),
                        ("class:status-bar-strong", snapshot["model_short"]),
                        ("class:status-bar-dim", " · "),
                        (self._status_bar_context_style(percent), percent_label),
                        ("class:status-bar-dim", " · "),
                        ("class:status-bar-dim", duration_label),
                        ("class:status-bar", " "),
                    ]
                else:
                    if snapshot["context_length"]:
                        ctx_total = _format_context_length(snapshot["context_length"])
                        ctx_used = format_token_count_compact(snapshot["context_tokens"])
                        context_label = f"{ctx_used}/{ctx_total}"
                    else:
                        context_label = "ctx --"

                    bar_style = self._status_bar_context_style(percent)
                    frags = [
                        ("class:status-bar", " ⚕ "),
                        ("class:status-bar-strong", snapshot["model_short"]),
                        ("class:status-bar-dim", " │ "),
                        ("class:status-bar-dim", context_label),
                        ("class:status-bar-dim", " │ "),
                        (bar_style, self._build_context_bar(percent)),
                        ("class:status-bar-dim", " "),
                        (bar_style, percent_label),
                        ("class:status-bar-dim", " │ "),
                        ("class:status-bar-dim", duration_label),
                        ("class:status-bar", " "),
                    ]

            total_width = sum(self._status_bar_display_width(text) for _, text in frags)
            if total_width > width:
                plain_text = "".join(text for _, text in frags)
                trimmed = self._trim_status_bar_text(plain_text, width)
                return [("class:status-bar", trimmed)]
            return frags
        except Exception:
            return [("class:status-bar", f" {self._build_status_bar_text()} ")]

    def _normalize_model_for_provider(self, resolved_provider: str) -> bool:
        """Normalize provider-specific model IDs and routing."""
        current_model = (self.model or "").strip()
        changed = False

        if resolved_provider == "copilot":
            try:
                from hermes_cli.models import copilot_model_api_mode, normalize_copilot_model_id

                canonical = normalize_copilot_model_id(current_model, api_key=self.api_key)
                if canonical and canonical != current_model:
                    if not self._model_is_default:
                        self.console.print(
                            f"[yellow]⚠️  Normalized Copilot model '{current_model}' to '{canonical}'.[/]"
                        )
                    self.model = canonical
                    current_model = canonical
                    changed = True

                resolved_mode = copilot_model_api_mode(current_model, api_key=self.api_key)
                if resolved_mode != self.api_mode:
                    self.api_mode = resolved_mode
                    changed = True
            except Exception:
                pass
            return changed

        if resolved_provider in {"opencode-zen", "opencode-go"}:
            try:
                from hermes_cli.models import normalize_opencode_model_id, opencode_model_api_mode

                canonical = normalize_opencode_model_id(resolved_provider, current_model)
                if canonical and canonical != current_model:
                    if not self._model_is_default:
                        self.console.print(
                            f"[yellow]⚠️  Stripped provider prefix from '{current_model}'; using '{canonical}' for {resolved_provider}.[/]"
                        )
                    self.model = canonical
                    current_model = canonical
                    changed = True

                resolved_mode = opencode_model_api_mode(resolved_provider, current_model)
                if resolved_mode != self.api_mode:
                    self.api_mode = resolved_mode
                    changed = True
            except Exception:
                pass
            return changed

        if resolved_provider != "openai-codex":
            return False

        # 1. Strip provider prefix ("openai/gpt-5.4" → "gpt-5.4")
        if "/" in current_model:
            slug = current_model.split("/", 1)[1]
            if not self._model_is_default:
                self.console.print(
                    f"[yellow]⚠️  Stripped provider prefix from '{current_model}'; "
                    f"using '{slug}' for OpenAI Codex.[/]"
                )
            self.model = slug
            current_model = slug
            changed = True

        # 2. Replace untouched default with a Codex model
        if self._model_is_default:
            fallback_model = "gpt-5.3-codex"
            try:
                from hermes_cli.codex_models import get_codex_model_ids

                available = get_codex_model_ids(
                    access_token=self.api_key if self.api_key else None,
                )
                if available:
                    fallback_model = available[0]
            except Exception:
                pass

            if current_model != fallback_model:
                self.model = fallback_model
                changed = True

        return changed

    def _on_thinking(self, text: str) -> None:
        """Called by agent when thinking starts/stops. Updates TUI spinner."""
        if not text:
            self._flush_reasoning_preview(force=True)
        self._spinner_text = text or ""
        self._invalidate()

    # ── Streaming display ────────────────────────────────────────────────

    def _current_reasoning_callback(self):
        """Return the active reasoning display callback for the current mode."""
        if self.show_reasoning and self.streaming_enabled:
            return self._stream_reasoning_delta
        if self.verbose and not self.show_reasoning:
            return self._on_reasoning
        return None

    def _emit_reasoning_preview(self, reasoning_text: str) -> None:
        """Render a buffered reasoning preview as a single [thinking] block."""
        import re
        import textwrap

        preview_text = reasoning_text.strip()
        if not preview_text:
            return

        try:
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 80
        prefix = "  [thinking] "
        wrap_width = max(30, term_width - len(prefix) - 2)

        paragraphs = []
        raw_paragraphs = re.split(r"\n\s*\n+", preview_text.replace("\r\n", "\n"))
        for paragraph in raw_paragraphs:
            compact = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
            if compact:
                paragraphs.append(textwrap.fill(compact, width=wrap_width))
        preview_text = "\n".join(paragraphs)
        if not preview_text:
            return

        if self.verbose:
            _cprint(f"  {_DIM}[thinking] {preview_text}{_RST}")
            return

        lines = preview_text.splitlines()
        if len(lines) > 5:
            preview = "\n".join(lines[:5])
            preview += f"\n  ... ({len(lines) - 5} more lines)"
        else:
            preview = preview_text
        _cprint(f"  {_DIM}[thinking] {preview}{_RST}")

    def _flush_reasoning_preview(self, *, force: bool = False) -> None:
        """Flush buffered reasoning text at natural boundaries.

        Some providers stream reasoning in tiny word or punctuation chunks.
        Buffer them here so the preview path does not print one `[thinking]`
        line per token.
        """
        buf = getattr(self, "_reasoning_preview_buf", "")
        if not buf:
            return

        try:
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 80
        target_width = max(40, term_width - len("  [thinking] ") - 4)

        flush_text = ""

        if force:
            flush_text = buf
            buf = ""
        else:
            line_break = buf.rfind("\n")
            min_newline_flush = max(16, target_width // 3)
            if line_break != -1 and (
                line_break >= min_newline_flush
                or buf.endswith("\n\n")
                or buf.endswith(".\n")
                or buf.endswith("!\n")
                or buf.endswith("?\n")
                or buf.endswith(":\n")
            ):
                flush_text = buf[: line_break + 1]
                buf = buf[line_break + 1 :]
            elif len(buf) >= target_width:
                search_start = max(20, target_width // 2)
                search_end = min(len(buf), max(target_width + (target_width // 3), target_width + 8))
                cut = -1
                for boundary in (" ", "\t", ".", "!", "?", ",", ";", ":"):
                    cut = max(cut, buf.rfind(boundary, search_start, search_end))
                if cut != -1:
                    flush_text = buf[: cut + 1]
                    buf = buf[cut + 1 :]

        self._reasoning_preview_buf = buf.lstrip() if flush_text else buf
        if flush_text:
            self._emit_reasoning_preview(flush_text)

    def _stream_reasoning_delta(self, text: str) -> None:
        """Stream reasoning/thinking tokens into a dim box above the response.

        Opens a dim reasoning box on first token, streams line-by-line.
        The box is closed automatically when content tokens start arriving
        (via _stream_delta → _emit_stream_text).

        Once the response box is open, suppress any further reasoning
        rendering — a late thinking block (e.g. after an interrupt) would
        otherwise draw a reasoning box inside the response box.
        """
        if not text:
            return
        self._reasoning_stream_started = True
        self._reasoning_shown_this_turn = True
        if getattr(self, "_stream_box_opened", False):
            return

        # Open reasoning box on first reasoning token
        if not getattr(self, "_reasoning_box_opened", False):
            self._reasoning_box_opened = True
            w = shutil.get_terminal_size().columns
            r_label = " Reasoning "
            r_fill = w - 2 - len(r_label)
            _cprint(f"\n{_DIM}┌─{r_label}{'─' * max(r_fill - 1, 0)}┐{_RST}")

        self._reasoning_buf = getattr(self, "_reasoning_buf", "") + text

        # Emit complete lines, and force-flush long partial lines so
        # reasoning is visible in real-time even without newlines.
        while "\n" in self._reasoning_buf:
            line, self._reasoning_buf = self._reasoning_buf.split("\n", 1)
            _cprint(f"{_DIM}{line}{_RST}")
        if len(self._reasoning_buf) > 80:
            _cprint(f"{_DIM}{self._reasoning_buf}{_RST}")
            self._reasoning_buf = ""

    def _close_reasoning_box(self) -> None:
        """Close the live reasoning box if it's open."""
        if getattr(self, "_reasoning_box_opened", False):
            # Flush remaining reasoning buffer
            buf = getattr(self, "_reasoning_buf", "")
            if buf:
                _cprint(f"{_DIM}{buf}{_RST}")
                self._reasoning_buf = ""
            w = shutil.get_terminal_size().columns
            _cprint(f"{_DIM}└{'─' * (w - 2)}┘{_RST}")
            self._reasoning_box_opened = False

    def _stream_delta(self, text) -> None:
        """Line-buffered streaming callback for real-time token rendering.

        Receives text deltas from the agent as tokens arrive. Buffers
        partial lines and emits complete lines via _cprint to work
        reliably with prompt_toolkit's patch_stdout.

        Reasoning/thinking blocks (<REASONING_SCRATCHPAD>, <think>, etc.)
        are suppressed during streaming since they'd display raw XML tags.
        The agent strips them from the final response anyway.

        A ``None`` value signals an intermediate turn boundary (tools are
        about to execute).  Flushes any open boxes and resets state so
        tool feed lines render cleanly between turns.
        """
        if text is None:
            self._flush_stream()
            self._reset_stream_state()
            return
        if not text:
            return

        self._stream_started = True

        # ── Tag-based reasoning suppression ──
        # Track whether we're inside a reasoning/thinking block.
        # These tags are model-generated (system prompt tells the model
        # to use them) and get stripped from final_response. We must
        # suppress them during streaming too — unless show_reasoning is
        # enabled, in which case we route the inner content to the
        # reasoning display box instead of discarding it.
        _OPEN_TAGS = ("<REASONING_SCRATCHPAD>", "<think>", "<reasoning>", "<THINKING>", "<thinking>")
        _CLOSE_TAGS = ("</REASONING_SCRATCHPAD>", "</think>", "</reasoning>", "</THINKING>", "</thinking>")

        # Append to a pre-filter buffer first
        self._stream_prefilt = getattr(self, "_stream_prefilt", "") + text

        # Check if we're entering a reasoning block
        if not getattr(self, "_in_reasoning_block", False):
            for tag in _OPEN_TAGS:
                idx = self._stream_prefilt.find(tag)
                if idx != -1:
                    # Emit everything before the tag
                    before = self._stream_prefilt[:idx]
                    if before:
                        self._emit_stream_text(before)
                    self._in_reasoning_block = True
                    self._stream_prefilt = self._stream_prefilt[idx + len(tag):]
                    break

            # Could also be a partial open tag at the end — hold it back
            if not getattr(self, "_in_reasoning_block", False):
                # Check for partial tag match at the end
                safe = self._stream_prefilt
                for tag in _OPEN_TAGS:
                    for i in range(1, len(tag)):
                        if self._stream_prefilt.endswith(tag[:i]):
                            safe = self._stream_prefilt[:-i]
                            break
                if safe:
                    self._emit_stream_text(safe)
                    self._stream_prefilt = self._stream_prefilt[len(safe):]
                return

        # Inside a reasoning block — look for close tag.
        # Keep accumulating _stream_prefilt because close tags can arrive
        # split across multiple tokens (e.g. "</REASONING_SCRATCH" + "PAD>...").
        if getattr(self, "_in_reasoning_block", False):
            for tag in _CLOSE_TAGS:
                idx = self._stream_prefilt.find(tag)
                if idx != -1:
                    self._in_reasoning_block = False
                    # When show_reasoning is on, route inner content to
                    # the reasoning display box instead of discarding.
                    if self.show_reasoning:
                        inner = self._stream_prefilt[:idx]
                        if inner:
                            self._stream_reasoning_delta(inner)
                    after = self._stream_prefilt[idx + len(tag):]
                    self._stream_prefilt = ""
                    # Process remaining text after close tag through full
                    # filtering (it could contain another open tag)
                    if after:
                        self._stream_delta(after)
                    return
            # When show_reasoning is on, stream reasoning content live
            # instead of silently accumulating. Keep only the tail that
            # could be a partial close tag prefix.
            max_tag_len = max(len(t) for t in _CLOSE_TAGS)
            if len(self._stream_prefilt) > max_tag_len:
                if self.show_reasoning:
                    # Route the safe prefix to reasoning display
                    safe_reasoning = self._stream_prefilt[:-max_tag_len]
                    self._stream_reasoning_delta(safe_reasoning)
                self._stream_prefilt = self._stream_prefilt[-max_tag_len:]
            return

    def _emit_stream_text(self, text: str) -> None:
        """Emit filtered text to the streaming display."""
        if not text:
            return

        # Close the live reasoning box before opening the response box
        self._close_reasoning_box()

        # Open the response box header on the very first visible text
        if not self._stream_box_opened:
            # Strip leading whitespace/newlines before first visible content
            text = text.lstrip("\n")
            if not text:
                return
            self._stream_box_opened = True
            try:
                from hermes_cli.skin_engine import get_active_skin
                _skin = get_active_skin()
                label = _skin.get_branding("response_label", "⚕ Hermes")
                _text_hex = _skin.get_color("banner_text", "#FFF8DC")
            except Exception:
                label = "⚕ Hermes"
                _text_hex = "#FFF8DC"
            # Build a true-color ANSI escape for the response text color
            # so streamed content matches the Rich Panel appearance.
            try:
                _r = int(_text_hex[1:3], 16)
                _g = int(_text_hex[3:5], 16)
                _b = int(_text_hex[5:7], 16)
                self._stream_text_ansi = f"\033[38;2;{_r};{_g};{_b}m"
            except (ValueError, IndexError):
                self._stream_text_ansi = ""
            w = shutil.get_terminal_size().columns
            fill = w - 2 - len(label)
            _cprint(f"\n{_GOLD}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}")

        self._stream_buf += text

        # Emit complete lines, keep partial remainder in buffer
        _tc = getattr(self, "_stream_text_ansi", "")
        while "\n" in self._stream_buf:
            line, self._stream_buf = self._stream_buf.split("\n", 1)
            _cprint(f"{_tc}{line}{_RST}" if _tc else line)

    def _flush_stream(self) -> None:
        """Emit any remaining partial line from the stream buffer and close the box."""
        # Close reasoning box if still open (in case no content tokens arrived)
        self._close_reasoning_box()

        if self._stream_buf:
            _tc = getattr(self, "_stream_text_ansi", "")
            _cprint(f"{_tc}{self._stream_buf}{_RST}" if _tc else self._stream_buf)
            self._stream_buf = ""

        # Close the response box
        if self._stream_box_opened:
            w = shutil.get_terminal_size().columns
            _cprint(f"{_GOLD}╰{'─' * (w - 2)}╯{_RST}")

    def _reset_stream_state(self) -> None:
        """Reset streaming state before each agent invocation."""
        self._stream_buf = ""
        self._stream_started = False
        self._stream_box_opened = False
        self._reasoning_stream_started = False
        self._stream_text_ansi = ""
        self._stream_prefilt = ""
        self._in_reasoning_block = False
        self._reasoning_box_opened = False
        self._reasoning_buf = ""
        self._reasoning_preview_buf = ""

    def _slow_command_status(self, command: str) -> str:
        """Return a user-facing status message for slower slash commands."""
        cmd_lower = command.lower().strip()
        if cmd_lower.startswith("/skills search"):
            return "Searching skills..."
        if cmd_lower.startswith("/skills browse"):
            return "Loading skills..."
        if cmd_lower.startswith("/skills inspect"):
            return "Inspecting skill..."
        if cmd_lower.startswith("/skills install"):
            return "Installing skill..."
        if cmd_lower.startswith("/skills"):
            return "Processing skills command..."
        if cmd_lower == "/reload-mcp":
            return "Reloading MCP servers..."
        if cmd_lower.startswith("/browser"):
            return "Configuring browser..."
        return "Processing command..."

    def _command_spinner_frame(self) -> str:
        """Return the current spinner frame for slow slash commands."""
        import time as _time

        frame_idx = int(_time.monotonic() * 10) % len(_COMMAND_SPINNER_FRAMES)
        return _COMMAND_SPINNER_FRAMES[frame_idx]

    @contextmanager
    def _busy_command(self, status: str):
        """Expose a temporary busy state in the TUI while a slash command runs."""
        self._command_running = True
        self._command_status = status
        self._invalidate(min_interval=0.0)
        try:
            print(f"⏳ {status}")
            yield
        finally:
            self._command_running = False
            self._command_status = ""
            self._invalidate(min_interval=0.0)

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
        resolved_acp_command = runtime.get("command")
        resolved_acp_args = list(runtime.get("args") or [])
        resolved_credential_pool = runtime.get("credential_pool")
        if not isinstance(api_key, str) or not api_key:
            # Custom / local endpoints (llama.cpp, ollama, vLLM, etc.) often
            # don't require authentication.  When a base_url IS configured but
            # no API key was found, use a placeholder so the OpenAI SDK
            # doesn't reject the request and local servers just ignore it.
            _source = runtime.get("source", "")
            _has_custom_base = isinstance(base_url, str) and base_url and "openrouter.ai" not in base_url
            if _has_custom_base:
                api_key = "no-key-required"
                logger.debug(
                    "No API key for custom endpoint %s (source=%s), "
                    "using placeholder — local servers typically ignore auth",
                    base_url, _source,
                )
            else:
                print("\n⚠️  Provider resolver returned an empty API key. "
                      "Set OPENROUTER_API_KEY or run: hermes setup")
                return False
        if not isinstance(base_url, str) or not base_url:
            print("\n⚠️  Provider resolver returned an empty base URL. "
                  "Check your provider config or run: hermes setup")
            return False

        credentials_changed = api_key != self.api_key or base_url != self.base_url
        routing_changed = (
            resolved_provider != self.provider
            or resolved_api_mode != self.api_mode
            or resolved_acp_command != self.acp_command
            or resolved_acp_args != self.acp_args
        )
        self.provider = resolved_provider
        self.api_mode = resolved_api_mode
        self.acp_command = resolved_acp_command
        self.acp_args = resolved_acp_args
        self._credential_pool = resolved_credential_pool
        self._provider_source = runtime.get("source")
        self.api_key = api_key
        self.base_url = base_url

        # Normalize model for the resolved provider (e.g. swap non-Codex
        # models when provider is openai-codex).  Fixes #651.
        model_changed = self._normalize_model_for_provider(resolved_provider)

        # AIAgent/OpenAI client holds auth at init time, so rebuild if key,
        # routing, or the effective model changed.
        if (credentials_changed or routing_changed or model_changed) and self.agent is not None:
            self.agent = None
            self._active_agent_route_signature = None

        return True

    def _resolve_turn_agent_config(self, user_message: str) -> dict:
        """Resolve model/runtime overrides for a single user turn."""
        from agent.smart_model_routing import resolve_turn_route

        return resolve_turn_route(
            user_message,
            self._smart_model_routing,
            {
                "model": self.model,
                "api_key": self.api_key,
                "base_url": self.base_url,
                "provider": self.provider,
                "api_mode": self.api_mode,
                "command": self.acp_command,
                "args": list(self.acp_args or []),
                "credential_pool": getattr(self, "_credential_pool", None),
            },
        )

    def _init_agent(self, *, model_override: str = None, runtime_override: dict = None, route_label: str = None) -> bool:
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

        # Initialize SQLite session store for CLI sessions (if not already done in __init__)
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.warning("SQLite session store not available — session will NOT be indexed: %s", e)
        
        # If resuming, validate the session exists and load its history.
        # _preload_resumed_session() may have already loaded it (called from
        # run() for immediate display).  In that case, conversation_history
        # is non-empty and we skip the DB round-trip.
        if self._resumed and self._session_db and not self.conversation_history:
            session_meta = self._session_db.get_session(self.session_id)
            if not session_meta:
                _cprint(f"\033[1;31mSession not found: {self.session_id}{_RST}")
                _cprint(f"{_DIM}Use a session ID from a previous CLI run (hermes sessions list).{_RST}")
                return False
            restored = self._session_db.get_messages_as_conversation(self.session_id)
            if restored:
                restored = [m for m in restored if m.get("role") != "session_meta"]
                self.conversation_history = restored
                msg_count = len([m for m in restored if m.get("role") == "user"])
                title_part = ""
                if session_meta.get("title"):
                    title_part = f" \"{session_meta['title']}\""
                ChatConsole().print(
                    f"[bold {_accent_hex()}]↻ Resumed session[/] "
                    f"[bold]{_escape(self.session_id)}[/]"
                    f"[bold {_accent_hex()}]{_escape(title_part)}[/] "
                    f"({msg_count} user message{'s' if msg_count != 1 else ''}, {len(restored)} total messages)"
                )
            else:
                ChatConsole().print(
                    f"[bold {_accent_hex()}]Session {_escape(self.session_id)} found but has no messages. Starting fresh.[/]"
                )
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
            runtime = runtime_override or {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "provider": self.provider,
                "api_mode": self.api_mode,
                "command": self.acp_command,
                "args": list(self.acp_args or []),
                "credential_pool": getattr(self, "_credential_pool", None),
            }
            effective_model = model_override or self.model
            self.agent = AIAgent(
                model=effective_model,
                api_key=runtime.get("api_key"),
                base_url=runtime.get("base_url"),
                provider=runtime.get("provider"),
                api_mode=runtime.get("api_mode"),
                acp_command=runtime.get("command"),
                acp_args=runtime.get("args"),
                credential_pool=runtime.get("credential_pool"),
                max_iterations=self.max_turns,
                enabled_toolsets=self.enabled_toolsets,
                verbose_logging=self.verbose,
                quiet_mode=not self.verbose,
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
                reasoning_callback=self._current_reasoning_callback(),

                fallback_model=self._fallback_model,
                thinking_callback=self._on_thinking,
                checkpoints_enabled=self.checkpoints_enabled,
                checkpoint_max_snapshots=self.checkpoint_max_snapshots,
                pass_session_id=self.pass_session_id,
                tool_progress_callback=self._on_tool_progress,
                tool_start_callback=self._on_tool_start if self._inline_diffs_enabled else None,
                tool_complete_callback=self._on_tool_complete if self._inline_diffs_enabled else None,
                stream_delta_callback=self._stream_delta if self.streaming_enabled else None,
                tool_gen_callback=self._on_tool_gen_start if self.streaming_enabled else None,
            )
            # Store reference for atexit memory provider shutdown
            global _active_agent_ref
            _active_agent_ref = self.agent
            # Route agent status output through prompt_toolkit so ANSI escape
            # sequences aren't garbled by patch_stdout's StdoutProxy (#2262).
            self.agent._print_fn = _cprint
            self._active_agent_route_signature = (
                effective_model,
                runtime.get("provider"),
                runtime.get("base_url"),
                runtime.get("api_mode"),
                runtime.get("command"),
                tuple(runtime.get("args") or ()),
            )

            if self._pending_title and self._session_db:
                try:
                    self._session_db.set_session_title(self.session_id, self._pending_title)
                    _cprint(f"  Session title applied: {self._pending_title}")
                    self._pending_title = None
                except (ValueError, Exception) as e:
                    _cprint(f"  Could not apply pending title: {e}")
                    self._pending_title = None
            return True
        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize agent: {e}[/]")
            return False
    
    def show_banner(self):
        """Display the welcome banner in Claude Code style."""
        self.console.clear()

        # Get context length for display before branching so it remains
        # available to the low-context warning logic in compact mode too.
        ctx_len = None
        if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'context_compressor'):
            ctx_len = self.agent.context_compressor.context_length
        
        # Auto-compact for narrow terminals — the full banner with caduceus
        # + tool list needs ~80 columns minimum to render without wrapping.
        term_width = shutil.get_terminal_size().columns
        use_compact = self.compact or term_width < 80
        
        if use_compact:
            self.console.print(_build_compact_banner())
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
                context_length=ctx_len,
            )
        
        # Show tool availability warnings if any tools are disabled
        self._show_tool_availability_warnings()

        # Warn about very low context lengths (common with local servers)
        if ctx_len and ctx_len <= 8192:
            self.console.print()
            self.console.print(
                f"[yellow]⚠️  Context length is only {ctx_len:,} tokens — "
                f"this is likely too low for agent use with tools.[/]"
            )
            self.console.print(
                "[dim]   Hermes needs 16k–32k minimum. Tool schemas + system prompt alone use ~4k–8k.[/]"
            )
            base_url = getattr(self, "base_url", "") or ""
            if "11434" in base_url or "ollama" in base_url.lower():
                self.console.print(
                    "[dim]   Ollama fix: OLLAMA_CONTEXT_LENGTH=32768 ollama serve[/]"
                )
            elif "1234" in base_url:
                self.console.print(
                    "[dim]   LM Studio fix: Set context length in model settings → reload model[/]"
                )
            else:
                self.console.print(
                    "[dim]   Fix: Set model.context_length in config.yaml, or increase your server's context setting[/]"
                )

        self.console.print()

    def _preload_resumed_session(self) -> bool:
        """Load a resumed session's history from the DB early (before first chat).

        Called from run() so the conversation history is available for display
        before the user sends their first message.  Sets
        ``self.conversation_history`` and prints the one-liner status.  Returns
        True if history was loaded, False otherwise.

        The corresponding block in ``_init_agent()`` checks whether history is
        already populated and skips the DB round-trip.
        """
        if not self._resumed or not self._session_db:
            return False

        session_meta = self._session_db.get_session(self.session_id)
        if not session_meta:
            self.console.print(
                f"[bold red]Session not found: {self.session_id}[/]"
            )
            self.console.print(
                "[dim]Use a session ID from a previous CLI run "
                "(hermes sessions list).[/]"
            )
            return False

        restored = self._session_db.get_messages_as_conversation(self.session_id)
        if restored:
            restored = [m for m in restored if m.get("role") != "session_meta"]
            self.conversation_history = restored
            msg_count = len([m for m in restored if m.get("role") == "user"])
            title_part = ""
            if session_meta.get("title"):
                title_part = f' "{session_meta["title"]}"'
            self.console.print(
                f"[#DAA520]↻ Resumed session [bold]{self.session_id}[/bold]"
                f"{title_part} "
                f"({msg_count} user message{'s' if msg_count != 1 else ''}, "
                f"{len(restored)} total messages)[/]"
            )
        else:
            self.console.print(
                f"[#DAA520]Session {self.session_id} found but has no "
                f"messages. Starting fresh.[/]"
            )
            return False

        # Re-open the session (clear ended_at so it's active again)
        try:
            self._session_db._conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                "WHERE id = ?",
                (self.session_id,),
            )
            self._session_db._conn.commit()
        except Exception:
            pass

        return True

    def _display_resumed_history(self):
        """Render a compact recap of previous conversation messages.

        Uses Rich markup with dim/muted styling so the recap is visually
        distinct from the active conversation.  Caps the display at the
        last ``MAX_DISPLAY_EXCHANGES`` user/assistant exchanges and shows
        an indicator for earlier hidden messages.
        """
        if not self.conversation_history:
            return

        # Check config: resume_display setting
        if self.resume_display == "minimal":
            return

        MAX_DISPLAY_EXCHANGES = 10   # max user+assistant pairs to show
        MAX_USER_LEN = 300           # truncate user messages
        MAX_ASST_LEN = 200           # truncate assistant text
        MAX_ASST_LINES = 3           # max lines of assistant text

        def _strip_reasoning(text: str) -> str:
            """Remove <REASONING_SCRATCHPAD>...</REASONING_SCRATCHPAD> blocks
            from displayed text (reasoning model internal thoughts)."""
            import re
            cleaned = re.sub(
                r"<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>\s*",
                "", text, flags=re.DOTALL,
            )
            # Also strip unclosed reasoning tags at the end
            cleaned = re.sub(
                r"<REASONING_SCRATCHPAD>.*$",
                "", cleaned, flags=re.DOTALL,
            )
            return cleaned.strip()

        # Collect displayable entries (skip system, tool-result messages)
        entries = []  # list of (role, display_text)
        for msg in self.conversation_history:
            role = msg.get("role", "")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []

            if role == "system":
                continue
            if role == "tool":
                continue

            if role == "user":
                text = "" if content is None else str(content)
                # Handle multimodal content (list of dicts)
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and part.get("type") == "image_url":
                            parts.append("[image]")
                    text = " ".join(parts)
                if len(text) > MAX_USER_LEN:
                    text = text[:MAX_USER_LEN] + "..."
                entries.append(("user", text))

            elif role == "assistant":
                text = "" if content is None else str(content)
                text = _strip_reasoning(text)
                parts = []
                if text:
                    lines = text.splitlines()
                    if len(lines) > MAX_ASST_LINES:
                        text = "\n".join(lines[:MAX_ASST_LINES]) + " ..."
                    if len(text) > MAX_ASST_LEN:
                        text = text[:MAX_ASST_LEN] + "..."
                    parts.append(text)
                if tool_calls:
                    tc_count = len(tool_calls)
                    # Extract tool names
                    names = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "unknown") if isinstance(fn, dict) else "unknown"
                        if name not in names:
                            names.append(name)
                    names_str = ", ".join(names[:4])
                    if len(names) > 4:
                        names_str += ", ..."
                    noun = "call" if tc_count == 1 else "calls"
                    parts.append(f"[{tc_count} tool {noun}: {names_str}]")
                if not parts:
                    # Skip pure-reasoning messages that have no visible output
                    continue
                entries.append(("assistant", " ".join(parts)))

        if not entries:
            return

        # Determine if we need to truncate
        skipped = 0
        if len(entries) > MAX_DISPLAY_EXCHANGES * 2:
            skipped = len(entries) - MAX_DISPLAY_EXCHANGES * 2
            entries = entries[skipped:]

        # Build the display using Rich
        from rich.panel import Panel
        from rich.text import Text

        try:
            from hermes_cli.skin_engine import get_active_skin
            _skin = get_active_skin()
            _history_text_c = _skin.get_color("banner_text", "#FFF8DC")
            _session_label_c = _skin.get_color("session_label", "#DAA520")
            _session_border_c = _skin.get_color("session_border", "#8B8682")
            _assistant_label_c = _skin.get_color("ui_ok", "#8FBC8F")
        except Exception:
            _history_text_c = "#FFF8DC"
            _session_label_c = "#DAA520"
            _session_border_c = "#8B8682"
            _assistant_label_c = "#8FBC8F"

        lines = Text()
        if skipped:
            lines.append(
                f"  ... {skipped} earlier messages ...\n\n",
                style="dim italic",
            )

        for i, (role, text) in enumerate(entries):
            if role == "user":
                lines.append("  ● You: ", style=f"dim bold {_session_label_c}")
                # Show first line inline, indent rest
                msg_lines = text.splitlines()
                lines.append(msg_lines[0] + "\n", style="dim")
                for ml in msg_lines[1:]:
                    lines.append(f"         {ml}\n", style="dim")
            else:
                lines.append("  ◆ Hermes: ", style=f"dim bold {_assistant_label_c}")
                msg_lines = text.splitlines()
                lines.append(msg_lines[0] + "\n", style="dim")
                for ml in msg_lines[1:]:
                    lines.append(f"            {ml}\n", style="dim")
            if i < len(entries) - 1:
                lines.append("")  # small gap

        panel = Panel(
            lines,
            title=f"[dim {_session_label_c}]Previous Conversation[/]",
            border_style=f"dim {_session_border_c}",
            padding=(0, 1),
            style=_history_text_c,
        )
        self.console.print(panel)

    def _try_attach_clipboard_image(self) -> bool:
        """Check clipboard for an image and attach it if found.

        Saves the image to ~/.hermes/images/ and appends the path to
        ``_attached_images``.  Returns True if an image was attached.
        """
        from hermes_cli.clipboard import save_clipboard_image

        img_dir = get_hermes_home() / "images"
        self._image_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = img_dir / f"clip_{ts}_{self._image_counter}.png"

        if save_clipboard_image(img_path):
            self._attached_images.append(img_path)
            return True
        self._image_counter -= 1
        return False

    def _handle_rollback_command(self, command: str):
        """Handle /rollback — list, diff, or restore filesystem checkpoints.

        Syntax:
            /rollback                 — list checkpoints
            /rollback <N>             — restore checkpoint N (also undoes last chat turn)
            /rollback diff <N>        — preview changes since checkpoint N
            /rollback <N> <file>      — restore a single file from checkpoint N
        """
        from tools.checkpoint_manager import format_checkpoint_list

        if not hasattr(self, 'agent') or not self.agent:
            print("  No active agent session.")
            return

        mgr = self.agent._checkpoint_mgr
        if not mgr.enabled:
            print("  Checkpoints are not enabled.")
            print("  Enable with: hermes --checkpoints")
            print("  Or in config.yaml: checkpoints: { enabled: true }")
            return

        cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        parts = command.split()
        args = parts[1:] if len(parts) > 1 else []

        if not args:
            # List checkpoints
            checkpoints = mgr.list_checkpoints(cwd)
            print(format_checkpoint_list(checkpoints, cwd))
            return

        # Handle /rollback diff <N>
        if args[0].lower() == "diff":
            if len(args) < 2:
                print("  Usage: /rollback diff <N>")
                return
            checkpoints = mgr.list_checkpoints(cwd)
            if not checkpoints:
                print(f"  No checkpoints found for {cwd}")
                return
            target_hash = self._resolve_checkpoint_ref(args[1], checkpoints)
            if not target_hash:
                return
            result = mgr.diff(cwd, target_hash)
            if result["success"]:
                stat = result.get("stat", "")
                diff = result.get("diff", "")
                if not stat and not diff:
                    print("  No changes since this checkpoint.")
                else:
                    if stat:
                        print(f"\n{stat}")
                    if diff:
                        # Limit diff output to avoid terminal flood
                        diff_lines = diff.splitlines()
                        if len(diff_lines) > 80:
                            print("\n".join(diff_lines[:80]))
                            print(f"\n  ... ({len(diff_lines) - 80} more lines, showing first 80)")
                        else:
                            print(f"\n{diff}")
            else:
                print(f"  ❌ {result['error']}")
            return

        # Resolve checkpoint reference (number or hash)
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            print(f"  No checkpoints found for {cwd}")
            return

        target_hash = self._resolve_checkpoint_ref(args[0], checkpoints)
        if not target_hash:
            return

        # Check for file-level restore: /rollback <N> <file>
        file_path = args[1] if len(args) > 1 else None

        result = mgr.restore(cwd, target_hash, file_path=file_path)
        if result["success"]:
            if file_path:
                print(f"  ✅ Restored {file_path} from checkpoint {result['restored_to']}: {result['reason']}")
            else:
                print(f"  ✅ Restored to checkpoint {result['restored_to']}: {result['reason']}")
            print("  A pre-rollback snapshot was saved automatically.")

            # Also undo the last conversation turn so the agent's context
            # matches the restored filesystem state
            if self.conversation_history:
                self.undo_last()
                print("  Chat turn undone to match restored file state.")
        else:
            print(f"  ❌ {result['error']}")

    def _resolve_checkpoint_ref(self, ref: str, checkpoints: list) -> str | None:
        """Resolve a checkpoint number or hash to a full commit hash."""
        try:
            idx = int(ref) - 1  # 1-indexed for user
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]["hash"]
            else:
                print(f"  Invalid checkpoint number. Use 1-{len(checkpoints)}.")
                return None
        except ValueError:
            # Treat as a git hash
            return ref

    def _handle_stop_command(self):
        """Handle /stop — kill all running background processes.

        Inspired by OpenAI Codex's separation of interrupt (stop current turn)
        from /stop (clean up background processes). See openai/codex#14602.
        """
        from tools.process_registry import process_registry

        processes = process_registry.list_sessions()
        running = [p for p in processes if p.get("status") == "running"]

        if not running:
            print("  No running background processes.")
            return

        print(f"  Stopping {len(running)} background process(es)...")
        killed = process_registry.kill_all()
        print(f"  ✅ Stopped {killed} process(es).")

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

    def _preprocess_images_with_vision(self, text: str, images: list) -> str:
        """Analyze attached images via the vision tool and return enriched text.

        Instead of embedding raw base64 ``image_url`` content parts in the
        conversation (which only works with vision-capable models), this
        pre-processes each image through the auxiliary vision model (Gemini
        Flash) and prepends the descriptions to the user's message — the
        same approach the messaging gateway uses.

        The local file path is included so the agent can re-examine the
        image later with ``vision_analyze`` if needed.
        """
        import asyncio as _asyncio
        import json as _json
        from tools.vision_tools import vision_analyze_tool

        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        enriched_parts = []
        for img_path in images:
            if not img_path.exists():
                continue
            size_kb = img_path.stat().st_size // 1024
            _cprint(f"  {_DIM}👁️  analyzing {img_path.name} ({size_kb}KB)...{_RST}")
            try:
                result_json = _asyncio.run(
                    vision_analyze_tool(image_url=str(img_path), user_prompt=analysis_prompt)
                )
                result = _json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    enriched_parts.append(
                        f"[The user attached an image. Here's what it contains:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {img_path}]"
                    )
                    _cprint(f"  {_DIM}✓ image analyzed{_RST}")
                else:
                    enriched_parts.append(
                        f"[The user attached an image but it couldn't be analyzed. "
                        f"You can try examining it with vision_analyze using "
                        f"image_url: {img_path}]"
                    )
                    _cprint(f"  {_DIM}⚠ vision analysis failed — path included for retry{_RST}")
            except Exception as e:
                enriched_parts.append(
                    f"[The user attached an image but analysis failed ({e}). "
                    f"You can try examining it with vision_analyze using "
                    f"image_url: {img_path}]"
                )
                _cprint(f"  {_DIM}⚠ vision analysis error — path included for retry{_RST}")

        # Combine: vision descriptions first, then the user's original text
        user_text = text if isinstance(text, str) and text else ""
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            return f"{prefix}\n\n{user_text}" if user_text else prefix
        return user_text or "What do you see in this image?"

    def _show_tool_availability_warnings(self):
        """Show warnings about disabled tools due to missing API keys."""
        try:
            from model_tools import check_tool_availability
            
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
        """Display help information with categorized commands."""
        from hermes_cli.commands import COMMANDS_BY_CATEGORY

        try:
            from hermes_cli.skin_engine import get_active_help_header
            header = get_active_help_header("(^_^)? Available Commands")
        except Exception:
            header = "(^_^)? Available Commands"
        header = (header or "").strip() or "(^_^)? Available Commands"
        inner_width = 55
        if len(header) > inner_width:
            header = header[:inner_width]
        _cprint(f"\n{_BOLD}+{'-' * inner_width}+{_RST}")
        _cprint(f"{_BOLD}|{header:^{inner_width}}|{_RST}")
        _cprint(f"{_BOLD}+{'-' * inner_width}+{_RST}")

        for category, commands in COMMANDS_BY_CATEGORY.items():
            _cprint(f"\n  {_BOLD}── {category} ──{_RST}")
            for cmd, desc in commands.items():
                ChatConsole().print(f"    [bold {_accent_hex()}]{cmd:<15}[/] [dim]-[/] {_escape(desc)}")

        if _skill_commands:
            _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} ({len(_skill_commands)} installed):")
            for cmd, info in sorted(_skill_commands.items()):
                ChatConsole().print(
                    f"    [bold {_accent_hex()}]{cmd:<22}[/] [dim]-[/] {_escape(info['description'])}"
                )

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

    def _handle_tools_command(self, cmd: str):
        """Handle /tools [list|disable|enable] slash commands.

        /tools (no args) shows the tool list.
        /tools list shows enabled/disabled status per toolset.
        /tools disable/enable saves the change to config and resets
        the session so the new tool set takes effect cleanly (no
        prompt-cache breakage mid-conversation).
        """
        import shlex
        from argparse import Namespace
        from hermes_cli.tools_config import tools_disable_enable_command

        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()

        subcommand = parts[1] if len(parts) > 1 else ""
        if subcommand not in ("list", "disable", "enable"):
            self.show_tools()
            return

        if subcommand == "list":
            tools_disable_enable_command(
                Namespace(tools_action="list", platform="cli"))
            return

        names = parts[2:]
        if not names:
            print(f"(._.) Usage: /tools {subcommand} <name> [name ...]")
            print(f"  Built-in toolset:  /tools {subcommand} web")
            print(f"  MCP tool:          /tools {subcommand} github:create_issue")
            return

        # Apply the change directly — the user typing the command is implicit
        # consent.  Do NOT use input() here; it hangs inside prompt_toolkit's
        # TUI event loop (known pitfall).
        verb = "Disabling" if subcommand == "disable" else "Enabling"
        label = ", ".join(names)
        _cprint(f"{_GOLD}{verb} {label}...{_RST}")

        tools_disable_enable_command(
            Namespace(tools_action=subcommand, names=names, platform="cli"))

        # Reset session so the new tool config is picked up from a clean state
        from hermes_cli.tools_config import _get_platform_tools
        from hermes_cli.config import load_config
        self.enabled_toolsets = _get_platform_tools(load_config(), "cli")
        self.new_session()
        _cprint(f"{_DIM}Session reset. New tool configuration is active.{_RST}")

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
    
    def _handle_profile_command(self):
        """Display active profile name and home directory."""
        from hermes_constants import get_hermes_home, display_hermes_home

        home = get_hermes_home()
        display = display_hermes_home()

        profiles_parent = Path.home() / ".hermes" / "profiles"
        try:
            rel = home.relative_to(profiles_parent)
            profile_name = str(rel).split("/")[0]
        except ValueError:
            profile_name = None

        print()
        if profile_name:
            print(f"  Profile: {profile_name}")
        else:
            print("  Profile: default")
        print(f"  Home:    {display}")
        print()

    def show_config(self):
        """Display current configuration with kawaii ASCII art."""
        # Get terminal config from environment (which was set from cli-config.yaml)
        terminal_env = os.getenv("TERMINAL_ENV", "local")
        terminal_cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        terminal_timeout = os.getenv("TERMINAL_TIMEOUT", "60")
        
        user_config_path = _hermes_home / 'config.yaml'
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
    
    def _list_recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent CLI sessions for in-chat browsing/resume affordances."""
        if not self._session_db:
            return []
        try:
            sessions = self._session_db.list_sessions_rich(
                source="cli",
                exclude_sources=["tool"],
                limit=limit,
            )
        except Exception:
            return []
        return [s for s in sessions if s.get("id") != self.session_id]

    def _show_recent_sessions(self, *, reason: str = "history", limit: int = 10) -> bool:
        """Render recent sessions inline from the active chat TUI.

        Returns True when something was shown, False if no session list was available.
        """
        sessions = self._list_recent_sessions(limit=limit)
        if not sessions:
            return False

        from hermes_cli.main import _relative_time

        print()
        if reason == "history":
            print("(._.) No messages in the current chat yet — here are recent sessions you can resume:")
        else:
            print("  Recent sessions:")
        print()
        print(f"  {'Title':<32} {'Preview':<40} {'Last Active':<13} {'ID'}")
        print(f"  {'─' * 32} {'─' * 40} {'─' * 13} {'─' * 24}")
        for session in sessions:
            title = (session.get("title") or "—")[:30]
            preview = (session.get("preview") or "")[:38]
            last_active = _relative_time(session.get("last_active"))
            print(f"  {title:<32} {preview:<40} {last_active:<13} {session['id']}")
        print()
        print("  Use /resume <session id or title> to continue where you left off.")
        print()
        return True

    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            if not self._show_recent_sessions(reason="history"):
                print("(._.) No conversation history yet.")
            return

        preview_limit = 400
        visible_index = 0
        hidden_tool_messages = 0

        def flush_tool_summary():
            nonlocal hidden_tool_messages
            if not hidden_tool_messages:
                return

            noun = "message" if hidden_tool_messages == 1 else "messages"
            print("\n  [Tools]")
            print(f"    ({hidden_tool_messages} tool {noun} hidden)")
            hidden_tool_messages = 0

        print()
        print("+" + "-" * 50 + "+")
        print("|" + " " * 12 + "(^_^) Conversation History" + " " * 11 + "|")
        print("+" + "-" * 50 + "+")

        for msg in self.conversation_history:
            role = msg.get("role", "unknown")

            if role == "tool":
                hidden_tool_messages += 1
                continue

            if role not in {"user", "assistant"}:
                continue

            flush_tool_summary()
            visible_index += 1

            content = msg.get("content")
            content_text = "" if content is None else str(content)

            if role == "user":
                print(f"\n  [You #{visible_index}]")
                print(
                    f"    {content_text[:preview_limit]}{'...' if len(content_text) > preview_limit else ''}"
                )
                continue

            print(f"\n  [Hermes #{visible_index}]")
            tool_calls = msg.get("tool_calls") or []
            if content_text:
                preview = content_text[:preview_limit]
                suffix = "..." if len(content_text) > preview_limit else ""
            elif tool_calls:
                tool_count = len(tool_calls)
                noun = "call" if tool_count == 1 else "calls"
                preview = f"(requested {tool_count} tool {noun})"
                suffix = ""
            else:
                preview = "(no text response)"
                suffix = ""
            print(f"    {preview}{suffix}")

        flush_tool_summary()
        print()
    
    def new_session(self, silent=False):
        """Start a fresh session with a new session ID and cleared agent state."""
        if self.agent and self.conversation_history:
            try:
                self.agent.flush_memories(self.conversation_history)
            except (Exception, KeyboardInterrupt):
                pass

        old_session_id = self.session_id
        if self._session_db and old_session_id:
            try:
                self._session_db.end_session(old_session_id, "new_session")
            except Exception:
                pass

        self.session_start = datetime.now()
        timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        self.session_id = f"{timestamp_str}_{short_uuid}"
        self.conversation_history = []
        self._pending_title = None
        self._resumed = False

        if self.agent:
            self.agent.session_id = self.session_id
            self.agent.session_start = self.session_start
            self.agent.reset_session_state()
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = 0
            if hasattr(self.agent, "_todo_store"):
                try:
                    from tools.todo_tool import TodoStore
                    self.agent._todo_store = TodoStore()
                except Exception:
                    pass
            if hasattr(self.agent, "_invalidate_system_prompt"):
                self.agent._invalidate_system_prompt()

            if self._session_db:
                try:
                    self._session_db.create_session(
                        session_id=self.session_id,
                        source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                        model=self.model,
                        model_config={
                            "max_iterations": self.max_turns,
                            "reasoning_config": self.reasoning_config,
                        },
                    )
                except Exception:
                    pass

        if not silent:
            print("(^_^)v New session started!")

    def _handle_resume_command(self, cmd_original: str) -> None:
        """Handle /resume <session_id_or_title> — switch to a previous session mid-conversation."""
        parts = cmd_original.split(None, 1)
        target = parts[1].strip() if len(parts) > 1 else ""

        if not target:
            _cprint("  Usage: /resume <session_id_or_title>")
            if self._show_recent_sessions(reason="resume"):
                return
            _cprint("  Tip:   Use /history or `hermes sessions list` to find sessions.")
            return

        if not self._session_db:
            _cprint("  Session database not available.")
            return

        # Resolve title or ID
        from hermes_cli.main import _resolve_session_by_name_or_id
        resolved = _resolve_session_by_name_or_id(target)
        target_id = resolved or target

        session_meta = self._session_db.get_session(target_id)
        if not session_meta:
            _cprint(f"  Session not found: {target}")
            _cprint("  Use /history or `hermes sessions list` to see available sessions.")
            return

        if target_id == self.session_id:
            _cprint("  Already on that session.")
            return

        # End current session
        try:
            self._session_db.end_session(self.session_id, "resumed_other")
        except Exception:
            pass

        # Switch to the target session
        self.session_id = target_id
        self._resumed = True
        self._pending_title = None

        # Load conversation history (strip transcript-only metadata entries)
        restored = self._session_db.get_messages_as_conversation(target_id)
        restored = [m for m in (restored or []) if m.get("role") != "session_meta"]
        self.conversation_history = restored

        # Re-open the target session so it's not marked as ended
        try:
            self._session_db.reopen_session(target_id)
        except Exception:
            pass

        # Sync the agent if already initialised
        if self.agent:
            self.agent.session_id = target_id
            self.agent.reset_session_state()
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = len(self.conversation_history)
            if hasattr(self.agent, "_todo_store"):
                try:
                    from tools.todo_tool import TodoStore
                    self.agent._todo_store = TodoStore()
                except Exception:
                    pass
            if hasattr(self.agent, "_invalidate_system_prompt"):
                self.agent._invalidate_system_prompt()

        title_part = f" \"{session_meta['title']}\"" if session_meta.get("title") else ""
        msg_count = len([m for m in self.conversation_history if m.get("role") == "user"])
        if self.conversation_history:
            _cprint(
                f"  ↻ Resumed session {target_id}{title_part}"
                f" ({msg_count} user message{'s' if msg_count != 1 else ''},"
                f" {len(self.conversation_history)} total)"
            )
        else:
            _cprint(f"  ↻ Resumed session {target_id}{title_part} — no messages, starting fresh.")

    def _handle_branch_command(self, cmd_original: str) -> None:
        """Handle /branch [name] — fork the current session into a new independent copy.

        Copies the full conversation history to a new session so the user can
        explore a different approach without losing the original session state.
        Inspired by Claude Code's /branch command.
        """
        if not self.conversation_history:
            _cprint("  No conversation to branch — send a message first.")
            return

        if not self._session_db:
            _cprint("  Session database not available.")
            return

        parts = cmd_original.split(None, 1)
        branch_name = parts[1].strip() if len(parts) > 1 else ""

        # Generate the new session ID
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        # Determine branch title
        if branch_name:
            branch_title = branch_name
        else:
            # Auto-generate from the current session title
            current_title = None
            if self._session_db:
                current_title = self._session_db.get_session_title(self.session_id)
            base = current_title or "branch"
            branch_title = self._session_db.get_next_title_in_lineage(base)

        # Save the current session's state before branching
        parent_session_id = self.session_id

        # End the old session
        try:
            self._session_db.end_session(self.session_id, "branched")
        except Exception:
            pass

        # Create the new session with parent link
        try:
            self._session_db.create_session(
                session_id=new_session_id,
                source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                model=self.model,
                model_config={
                    "max_iterations": self.max_turns,
                    "reasoning_config": self.reasoning_config,
                },
                parent_session_id=parent_session_id,
            )
        except Exception as e:
            _cprint(f"  Failed to create branch session: {e}")
            return

        # Copy conversation history to the new session
        for msg in self.conversation_history:
            try:
                self._session_db.append_message(
                    session_id=new_session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name") or msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    reasoning=msg.get("reasoning"),
                )
            except Exception:
                pass  # Best-effort copy

        # Set title on the branch
        try:
            self._session_db.set_session_title(new_session_id, branch_title)
        except Exception:
            pass

        # Switch to the new session
        self.session_id = new_session_id
        self.session_start = now
        self._pending_title = None
        self._resumed = True  # Prevents auto-title generation

        # Sync the agent
        if self.agent:
            self.agent.session_id = new_session_id
            self.agent.session_start = now
            self.agent.reset_session_state()
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = len(self.conversation_history)
            if hasattr(self.agent, "_todo_store"):
                try:
                    from tools.todo_tool import TodoStore
                    self.agent._todo_store = TodoStore()
                except Exception:
                    pass
            if hasattr(self.agent, "_invalidate_system_prompt"):
                self.agent._invalidate_system_prompt()

        msg_count = len([m for m in self.conversation_history if m.get("role") == "user"])
        _cprint(
            f"  ⑂ Branched session \"{branch_title}\""
            f" ({msg_count} user message{'s' if msg_count != 1 else ''})"
        )
        _cprint(f"  Original session: {parent_session_id}")
        _cprint(f"  Branch session:   {new_session_id}")

    def reset_conversation(self):
        """Reset the conversation by starting a new session."""
        # Shut down memory provider before resetting — actual session boundary
        if hasattr(self, 'agent') and self.agent:
            self.agent.shutdown_memory_provider(self.conversation_history)
        self.new_session()
    
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
    
    def _handle_model_switch(self, cmd_original: str):
        """Handle /model command — switch model for this session.

        Supports:
          /model                              — show current model + usage hints
          /model <name>                       — switch for this session only
          /model <name> --global              — switch and persist to config.yaml
          /model <name> --provider <provider> — switch provider + model
          /model --provider <provider>        — switch to provider, auto-detect model
        """
        from hermes_cli.model_switch import switch_model, parse_model_flags, list_authenticated_providers
        from hermes_cli.providers import get_label

        # Parse args from the original command
        parts = cmd_original.split(None, 1)  # split off '/model'
        raw_args = parts[1].strip() if len(parts) > 1 else ""

        # Parse --provider and --global flags
        model_input, explicit_provider, persist_global = parse_model_flags(raw_args)

        # No args at all: show available providers + models
        if not model_input and not explicit_provider:
            model_display = self.model or "unknown"
            provider_display = get_label(self.provider) if self.provider else "unknown"
            _cprint(f"  Current: {model_display} on {provider_display}")
            _cprint("")

            # Show authenticated providers with top models
            try:
                # Load user providers from config
                user_provs = None
                try:
                    from hermes_cli.config import load_config
                    cfg = load_config()
                    user_provs = cfg.get("providers")
                except Exception:
                    pass

                providers = list_authenticated_providers(
                    current_provider=self.provider or "",
                    user_providers=user_provs,
                    max_models=6,
                )
                if providers:
                    for p in providers:
                        tag = " (current)" if p["is_current"] else ""
                        _cprint(f"  {p['name']} [--provider {p['slug']}]{tag}:")
                        if p["models"]:
                            model_strs = ", ".join(p["models"])
                            extra = f"  (+{p['total_models'] - len(p['models'])} more)" if p["total_models"] > len(p["models"]) else ""
                            _cprint(f"    {model_strs}{extra}")
                        elif p.get("api_url"):
                            _cprint(f"    {p['api_url']} (use /model <name> --provider {p['slug']})")
                        else:
                            _cprint(f"    (no models listed)")
                        _cprint("")
                else:
                    _cprint("  No authenticated providers found.")
                    _cprint("")
            except Exception:
                pass

            # Aliases
            from hermes_cli.model_switch import MODEL_ALIASES
            alias_list = ", ".join(sorted(MODEL_ALIASES.keys()))
            _cprint(f"  Aliases: {alias_list}")
            _cprint("")
            _cprint("  /model <name>                        switch model")
            _cprint("  /model <name> --provider <slug>      switch provider")
            _cprint("  /model <name> --global               persist to config")
            return

        # Perform the switch
        result = switch_model(
            raw_input=model_input,
            current_provider=self.provider or "",
            current_model=self.model or "",
            current_base_url=self.base_url or "",
            current_api_key=self.api_key or "",
            is_global=persist_global,
            explicit_provider=explicit_provider,
        )

        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return

        # Apply to CLI state.
        # Update requested_provider so _ensure_runtime_credentials() doesn't
        # overwrite the switch on the next turn (it re-resolves from this).
        old_model = self.model
        self.model = result.new_model
        self.provider = result.target_provider
        self.requested_provider = result.target_provider
        if result.api_key:
            self.api_key = result.api_key
            self._explicit_api_key = result.api_key
        if result.base_url:
            self.base_url = result.base_url
            self._explicit_base_url = result.base_url
        if result.api_mode:
            self.api_mode = result.api_mode

        # Apply to running agent (in-place swap)
        if self.agent is not None:
            try:
                self.agent.switch_model(
                    new_model=result.new_model,
                    new_provider=result.target_provider,
                    api_key=result.api_key,
                    base_url=result.base_url,
                    api_mode=result.api_mode,
                )
            except Exception as exc:
                _cprint(f"  ⚠ Agent swap failed ({exc}); change applied to next session.")

        # Store a note to prepend to the next user message so the model
        # knows a switch occurred (avoids injecting system messages mid-history
        # which breaks providers and prompt caching).
        self._pending_model_switch_note = (
            f"[Note: model was just switched from {old_model} to {result.new_model} "
            f"via {result.provider_label or result.target_provider}. "
            f"Adjust your self-identification accordingly.]"
        )

        # Display confirmation with full metadata
        provider_label = result.provider_label or result.target_provider
        _cprint(f"  ✓ Model switched: {result.new_model}")
        _cprint(f"    Provider: {provider_label}")

        # Rich metadata from models.dev
        mi = result.model_info
        if mi:
            if mi.context_window:
                _cprint(f"    Context: {mi.context_window:,} tokens")
            if mi.max_output:
                _cprint(f"    Max output: {mi.max_output:,} tokens")
            if mi.has_cost_data():
                _cprint(f"    Cost: {mi.format_cost()}")
            _cprint(f"    Capabilities: {mi.format_capabilities()}")
        else:
            # Fallback to old context length lookup
            try:
                from agent.model_metadata import get_model_context_length
                ctx = get_model_context_length(
                    result.new_model,
                    base_url=result.base_url or self.base_url,
                    api_key=result.api_key or self.api_key,
                    provider=result.target_provider,
                )
                _cprint(f"    Context: {ctx:,} tokens")
            except Exception:
                pass

        # Cache notice
        cache_enabled = (
            ("openrouter" in (result.base_url or "").lower() and "claude" in result.new_model.lower())
            or result.api_mode == "anthropic_messages"
        )
        if cache_enabled:
            _cprint("    Prompt caching: enabled")

        # Warning from validation
        if result.warning_message:
            _cprint(f"    ⚠ {result.warning_message}")

        # Persistence
        if persist_global:
            save_config_value("model.name", result.new_model)
            if result.provider_changed:
                save_config_value("model.provider", result.target_provider)
            _cprint("    Saved to config.yaml (--global)")
        else:
            _cprint("    (session only — add --global to persist)")

    def _show_model_and_providers(self):
        """Show current model + provider and list all authenticated providers.

        Shows current model + provider, then lists all authenticated
        providers with their available models.
        """
        from hermes_cli.models import (
            curated_models_for_provider, list_available_providers,
            normalize_provider, _PROVIDER_LABELS,
        )
        from hermes_cli.auth import resolve_provider as _resolve_provider

        # Resolve current provider
        raw_provider = normalize_provider(self.provider)
        if raw_provider == "auto":
            try:
                current = _resolve_provider(
                    self.requested_provider,
                    explicit_api_key=self._explicit_api_key,
                    explicit_base_url=self._explicit_base_url,
                )
            except Exception:
                current = "openrouter"
        else:
            current = raw_provider
        current_label = _PROVIDER_LABELS.get(current, current)

        print(f"\n  Current: {self.model} via {current_label}")
        print()

        # Show all authenticated providers with their models
        providers = list_available_providers()
        authed = [p for p in providers if p["authenticated"]]
        unauthed = [p for p in providers if not p["authenticated"]]

        if authed:
            print("  Authenticated providers & models:")
            for p in authed:
                is_active = p["id"] == current
                marker = " ← active" if is_active else ""
                print(f"    [{p['id']}]{marker}")
                curated = curated_models_for_provider(p["id"])
                if curated:
                    for mid, desc in curated:
                        current_marker = " ← current" if (is_active and mid == self.model) else ""
                        print(f"      {mid}{current_marker}")
                elif p["id"] == "custom":
                    from hermes_cli.models import _get_custom_base_url
                    custom_url = _get_custom_base_url()
                    if custom_url:
                        print(f"      endpoint: {custom_url}")
                    if is_active:
                        print(f"      model: {self.model} ← current")
                    print("      (use hermes model to change)")
                else:
                    print("      (use hermes model to change)")
                print()

        if unauthed:
            names = ", ".join(p["label"] for p in unauthed)
            print(f"  Not configured: {names}")
            print("  Run: hermes setup")
            print()

        print("  To change model or provider, use: hermes model")

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
                    print("(^_^)b System prompt set (saved to config)")
                else:
                    print("(^_^) System prompt set (session only)")
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
    

    @staticmethod
    def _resolve_personality_prompt(value) -> str:
        """Accept string or dict personality value; return system prompt string."""
        if isinstance(value, dict):
            parts = [value.get("system_prompt", "")]
            if value.get("tone"):
                parts.append(f'Tone: {value["tone"]}' )
            if value.get("style"):
                parts.append(f'Style: {value["style"]}' )
            return "\n".join(p for p in parts if p)
        return str(value)

    def _handle_personality_command(self, cmd: str):
        """Handle the /personality command to set predefined personalities."""
        parts = cmd.split(maxsplit=1)
        
        if len(parts) > 1:
            # Set personality
            personality_name = parts[1].strip().lower()
            
            if personality_name in ("none", "default", "neutral"):
                self.system_prompt = ""
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", ""):
                    print("(^_^)b Personality cleared (saved to config)")
                else:
                    print("(^_^) Personality cleared (session only)")
                print("  No personality overlay — using base agent behavior.")
            elif personality_name in self.personalities:
                self.system_prompt = self._resolve_personality_prompt(self.personalities[personality_name])
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", self.system_prompt):
                    print(f"(^_^)b Personality set to '{personality_name}' (saved to config)")
                else:
                    print(f"(^_^) Personality set to '{personality_name}' (session only)")
                print(f"  \"{self.system_prompt[:60]}{'...' if len(self.system_prompt) > 60 else ''}\"")
            else:
                print(f"(._.) Unknown personality: {personality_name}")
                print(f"  Available: none, {', '.join(self.personalities.keys())}")
        else:
            # Show available personalities
            print()
            print("+" + "-" * 50 + "+")
            print("|" + " " * 12 + "(^o^)/ Personalities" + " " * 15 + "|")
            print("+" + "-" * 50 + "+")
            print()
            print(f"  {'none':<12} - (no personality overlay)")
            for name, prompt in self.personalities.items():
                if isinstance(prompt, dict):
                    preview = prompt.get("description") or prompt.get("system_prompt", "")[:50]
                else:
                    preview = str(prompt)[:50]
                print(f"  {name:<12} - {preview}")
            print()
            print("  Usage: /personality <name>")
            print()
    
    def _handle_cron_command(self, cmd: str):
        """Handle the /cron command to manage scheduled tasks."""
        import shlex
        from tools.cronjob_tools import cronjob as cronjob_tool

        def _cron_api(**kwargs):
            return json.loads(cronjob_tool(**kwargs))

        def _normalize_skills(values):
            normalized = []
            for value in values:
                text = str(value or "").strip()
                if text and text not in normalized:
                    normalized.append(text)
            return normalized

        def _parse_flags(tokens):
            opts = {
                "name": None,
                "deliver": None,
                "repeat": None,
                "skills": [],
                "add_skills": [],
                "remove_skills": [],
                "clear_skills": False,
                "all": False,
                "prompt": None,
                "schedule": None,
                "positionals": [],
            }
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token == "--name" and i + 1 < len(tokens):
                    opts["name"] = tokens[i + 1]
                    i += 2
                elif token == "--deliver" and i + 1 < len(tokens):
                    opts["deliver"] = tokens[i + 1]
                    i += 2
                elif token == "--repeat" and i + 1 < len(tokens):
                    try:
                        opts["repeat"] = int(tokens[i + 1])
                    except ValueError:
                        print("(._.) --repeat must be an integer")
                        return None
                    i += 2
                elif token == "--skill" and i + 1 < len(tokens):
                    opts["skills"].append(tokens[i + 1])
                    i += 2
                elif token == "--add-skill" and i + 1 < len(tokens):
                    opts["add_skills"].append(tokens[i + 1])
                    i += 2
                elif token == "--remove-skill" and i + 1 < len(tokens):
                    opts["remove_skills"].append(tokens[i + 1])
                    i += 2
                elif token == "--clear-skills":
                    opts["clear_skills"] = True
                    i += 1
                elif token == "--all":
                    opts["all"] = True
                    i += 1
                elif token == "--prompt" and i + 1 < len(tokens):
                    opts["prompt"] = tokens[i + 1]
                    i += 2
                elif token == "--schedule" and i + 1 < len(tokens):
                    opts["schedule"] = tokens[i + 1]
                    i += 2
                else:
                    opts["positionals"].append(token)
                    i += 1
            return opts

        tokens = shlex.split(cmd)

        if len(tokens) == 1:
            print()
            print("+" + "-" * 68 + "+")
            print("|" + " " * 22 + "(^_^) Scheduled Tasks" + " " * 23 + "|")
            print("+" + "-" * 68 + "+")
            print()
            print("  Commands:")
            print("    /cron list")
            print('    /cron add "every 2h" "Check server status" [--skill blogwatcher]')
            print('    /cron edit <job_id> --schedule "every 4h" --prompt "New task"')
            print("    /cron edit <job_id> --skill blogwatcher --skill find-nearby")
            print("    /cron edit <job_id> --remove-skill blogwatcher")
            print("    /cron edit <job_id> --clear-skills")
            print("    /cron pause <job_id>")
            print("    /cron resume <job_id>")
            print("    /cron run <job_id>")
            print("    /cron remove <job_id>")
            print()
            result = _cron_api(action="list")
            jobs = result.get("jobs", []) if result.get("success") else []
            if jobs:
                print("  Current Jobs:")
                print("  " + "-" * 63)
                for job in jobs:
                    repeat_str = job.get("repeat", "?")
                    print(f"    {job['job_id'][:12]:<12} | {job['schedule']:<15} | {repeat_str:<8}")
                    if job.get("skills"):
                        print(f"      Skills: {', '.join(job['skills'])}")
                    print(f"      {job.get('prompt_preview', '')}")
                    if job.get("next_run_at"):
                        print(f"      Next: {job['next_run_at']}")
                    print()
            else:
                print("  No scheduled jobs. Use '/cron add' to create one.")
            print()
            return

        subcommand = tokens[1].lower()
        opts = _parse_flags(tokens[2:])
        if opts is None:
            return

        if subcommand == "list":
            result = _cron_api(action="list", include_disabled=opts["all"])
            jobs = result.get("jobs", []) if result.get("success") else []
            if not jobs:
                print("(._.) No scheduled jobs.")
                return

            print()
            print("Scheduled Jobs:")
            print("-" * 80)
            for job in jobs:
                print(f"  ID: {job['job_id']}")
                print(f"  Name: {job['name']}")
                print(f"  State: {job.get('state', '?')}")
                print(f"  Schedule: {job['schedule']} ({job.get('repeat', '?')})")
                print(f"  Next run: {job.get('next_run_at', 'N/A')}")
                if job.get("skills"):
                    print(f"  Skills: {', '.join(job['skills'])}")
                print(f"  Prompt: {job.get('prompt_preview', '')}")
                if job.get("last_run_at"):
                    print(f"  Last run: {job['last_run_at']} ({job.get('last_status', '?')})")
                print()
            return

        if subcommand in {"add", "create"}:
            positionals = opts["positionals"]
            if not positionals:
                print("(._.) Usage: /cron add <schedule> <prompt>")
                return
            schedule = opts["schedule"] or positionals[0]
            prompt = opts["prompt"] or " ".join(positionals[1:])
            skills = _normalize_skills(opts["skills"])
            if not prompt and not skills:
                print("(._.) Please provide a prompt or at least one skill")
                return
            result = _cron_api(
                action="create",
                schedule=schedule,
                prompt=prompt or None,
                name=opts["name"],
                deliver=opts["deliver"],
                repeat=opts["repeat"],
                skills=skills or None,
            )
            if result.get("success"):
                print(f"(^_^)b Created job: {result['job_id']}")
                print(f"  Schedule: {result['schedule']}")
                if result.get("skills"):
                    print(f"  Skills: {', '.join(result['skills'])}")
                print(f"  Next run: {result['next_run_at']}")
            else:
                print(f"(x_x) Failed to create job: {result.get('error')}")
            return

        if subcommand == "edit":
            positionals = opts["positionals"]
            if not positionals:
                print("(._.) Usage: /cron edit <job_id> [--schedule ...] [--prompt ...] [--skill ...]")
                return
            job_id = positionals[0]
            existing = get_job(job_id)
            if not existing:
                print(f"(._.) Job not found: {job_id}")
                return

            final_skills = None
            replacement_skills = _normalize_skills(opts["skills"])
            add_skills = _normalize_skills(opts["add_skills"])
            remove_skills = set(_normalize_skills(opts["remove_skills"]))
            existing_skills = list(existing.get("skills") or ([] if not existing.get("skill") else [existing.get("skill")]))
            if opts["clear_skills"]:
                final_skills = []
            elif replacement_skills:
                final_skills = replacement_skills
            elif add_skills or remove_skills:
                final_skills = [skill for skill in existing_skills if skill not in remove_skills]
                for skill in add_skills:
                    if skill not in final_skills:
                        final_skills.append(skill)

            result = _cron_api(
                action="update",
                job_id=job_id,
                schedule=opts["schedule"],
                prompt=opts["prompt"],
                name=opts["name"],
                deliver=opts["deliver"],
                repeat=opts["repeat"],
                skills=final_skills,
            )
            if result.get("success"):
                job = result["job"]
                print(f"(^_^)b Updated job: {job['job_id']}")
                print(f"  Schedule: {job['schedule']}")
                if job.get("skills"):
                    print(f"  Skills: {', '.join(job['skills'])}")
                else:
                    print("  Skills: none")
            else:
                print(f"(x_x) Failed to update job: {result.get('error')}")
            return

        if subcommand in {"pause", "resume", "run", "remove", "rm", "delete"}:
            positionals = opts["positionals"]
            if not positionals:
                print(f"(._.) Usage: /cron {subcommand} <job_id>")
                return
            job_id = positionals[0]
            action = "remove" if subcommand in {"remove", "rm", "delete"} else subcommand
            result = _cron_api(action=action, job_id=job_id, reason="paused from /cron" if action == "pause" else None)
            if not result.get("success"):
                print(f"(x_x) Failed to {action} job: {result.get('error')}")
                return
            if action == "pause":
                print(f"(^_^)b Paused job: {result['job']['name']} ({job_id})")
            elif action == "resume":
                print(f"(^_^)b Resumed job: {result['job']['name']} ({job_id})")
                print(f"  Next run: {result['job'].get('next_run_at')}")
            elif action == "run":
                print(f"(^_^)b Triggered job: {result['job']['name']} ({job_id})")
                print("  It will run on the next scheduler tick.")
            else:
                removed = result.get("removed_job", {})
                print(f"(^_^)b Removed job: {removed.get('name', job_id)} ({job_id})")
            return

        print(f"(._.) Unknown cron command: {subcommand}")
        print("  Available: list, add, edit, pause, resume, run, remove")
    
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
            print(f"  Configuration file: {display_hermes_home()}/config.yaml")
            print()
            
        except Exception as e:
            print(f"  Error loading gateway config: {e}")
            print()
            print("  To configure the gateway:")
            print("    1. Set environment variables:")
            print("       TELEGRAM_BOT_TOKEN=your_token")
            print("       DISCORD_BOT_TOKEN=your_token")
            print(f"    2. Or configure settings in {display_hermes_home()}/config.yaml")
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

        # Resolve aliases via central registry so adding an alias is a one-line
        # change in hermes_cli/commands.py instead of touching every dispatch site.
        from hermes_cli.commands import resolve_command as _resolve_cmd
        _base_word = cmd_lower.split()[0].lstrip("/")
        _cmd_def = _resolve_cmd(_base_word)
        canonical = _cmd_def.name if _cmd_def else _base_word
        
        if canonical in ("quit", "exit", "q"):
            return False
        elif canonical == "help":
            self.show_help()
        elif canonical == "profile":
            self._handle_profile_command()
        elif canonical == "tools":
            self._handle_tools_command(cmd_original)
        elif canonical == "toolsets":
            self.show_toolsets()
        elif canonical == "config":
            self.show_config()
        elif canonical == "clear":
            self.new_session(silent=True)
            # Clear terminal screen.  Inside the TUI, Rich's console.clear()
            # goes through patch_stdout's StdoutProxy which swallows the
            # screen-clear escape sequences.  Use prompt_toolkit's output
            # object directly to actually clear the terminal.
            if self._app:
                out = self._app.output
                out.erase_screen()
                out.cursor_goto(0, 0)
                out.flush()
            else:
                self.console.clear()
            # Show fresh banner.  Inside the TUI we must route Rich output
            # through ChatConsole (which uses prompt_toolkit's native ANSI
            # renderer) instead of self.console (which writes raw to stdout
            # and gets mangled by patch_stdout).
            if self._app:
                cc = ChatConsole()
                term_w = shutil.get_terminal_size().columns
                if self.compact or term_w < 80:
                    cc.print(_build_compact_banner())
                else:
                    tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
                    cwd = os.getenv("TERMINAL_CWD", os.getcwd())
                    ctx_len = None
                    if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'context_compressor'):
                        ctx_len = self.agent.context_compressor.context_length
                    build_welcome_banner(
                        console=cc,
                        model=self.model,
                        cwd=cwd,
                        tools=tools,
                        enabled_toolsets=self.enabled_toolsets,
                        session_id=self.session_id,
                        context_length=ctx_len,
                    )
                _cprint("  ✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n")
            else:
                self.show_banner()
                print("  ✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n")
        elif canonical == "history":
            self.show_history()
        elif canonical == "title":
            parts = cmd_original.split(maxsplit=1)
            if len(parts) > 1:
                raw_title = parts[1].strip()
                if raw_title:
                    if self._session_db:
                        # Sanitize the title early so feedback matches what gets stored
                        try:
                            from hermes_state import SessionDB
                            new_title = SessionDB.sanitize_title(raw_title)
                        except ValueError as e:
                            _cprint(f"  {e}")
                            new_title = None
                        if not new_title:
                            _cprint("  Title is empty after cleanup. Please use printable characters.")
                        elif self._session_db.get_session(self.session_id):
                            # Session exists in DB — set title directly
                            try:
                                if self._session_db.set_session_title(self.session_id, new_title):
                                    _cprint(f"  Session title set: {new_title}")
                                else:
                                    _cprint("  Session not found in database.")
                            except ValueError as e:
                                _cprint(f"  {e}")
                        else:
                            # Session not created yet — defer the title
                            # Check uniqueness proactively with the sanitized title
                            existing = self._session_db.get_session_by_title(new_title)
                            if existing:
                                _cprint(f"  Title '{new_title}' is already in use by session {existing['id']}")
                            else:
                                self._pending_title = new_title
                                _cprint(f"  Session title queued: {new_title} (will be saved on first message)")
                    else:
                        _cprint("  Session database not available.")
                else:
                    _cprint("  Usage: /title <your session title>")
            else:
                # Show current title and session ID if no argument given
                if self._session_db:
                    _cprint(f"  Session ID: {self.session_id}")
                    session = self._session_db.get_session(self.session_id)
                    if session and session.get("title"):
                        _cprint(f"  Title: {session['title']}")
                    elif self._pending_title:
                        _cprint(f"  Title (pending): {self._pending_title}")
                    else:
                        _cprint("  No title set. Usage: /title <your session title>")
                else:
                    _cprint("  Session database not available.")
        elif canonical == "new":
            self.new_session()
        elif canonical == "resume":
            self._handle_resume_command(cmd_original)
        elif canonical == "model":
            self._handle_model_switch(cmd_original)
        elif canonical == "provider":
            self._show_model_and_providers()
        elif canonical == "prompt":
            # Use original case so prompt text isn't lowercased
            self._handle_prompt_command(cmd_original)
        elif canonical == "personality":
            # Use original case (handler lowercases the personality name itself)
            self._handle_personality_command(cmd_original)
        elif canonical == "plan":
            self._handle_plan_command(cmd_original)
        elif canonical == "retry":
            retry_msg = self.retry_last()
            if retry_msg and hasattr(self, '_pending_input'):
                # Re-queue the message so process_loop sends it to the agent
                self._pending_input.put(retry_msg)
        elif canonical == "undo":
            self.undo_last()
        elif canonical == "branch":
            self._handle_branch_command(cmd_original)
        elif canonical == "save":
            self.save_conversation()
        elif canonical == "cron":
            self._handle_cron_command(cmd_original)
        elif canonical == "skills":
            with self._busy_command(self._slow_command_status(cmd_original)):
                self._handle_skills_command(cmd_original)
        elif canonical == "platforms":
            self._show_gateway_status()
        elif canonical == "statusbar":
            self._status_bar_visible = not self._status_bar_visible
            state = "visible" if self._status_bar_visible else "hidden"
            self.console.print(f"  Status bar {state}")
        elif canonical == "verbose":
            self._toggle_verbose()
        elif canonical == "yolo":
            self._toggle_yolo()
        elif canonical == "reasoning":
            self._handle_reasoning_command(cmd_original)
        elif canonical == "compress":
            self._manual_compress()
        elif canonical == "usage":
            self._show_usage()
        elif canonical == "insights":
            self._show_insights(cmd_original)
        elif canonical == "paste":
            self._handle_paste_command()
        elif canonical == "reload-mcp":
            with self._busy_command(self._slow_command_status(cmd_original)):
                self._reload_mcp()
        elif canonical == "browser":
            self._handle_browser_command(cmd_original)
        elif canonical == "plugins":
            try:
                from hermes_cli.plugins import get_plugin_manager
                mgr = get_plugin_manager()
                plugins = mgr.list_plugins()
                if not plugins:
                    print("No plugins installed.")
                    print(f"Drop plugin directories into {display_hermes_home()}/plugins/ to get started.")
                else:
                    print(f"Plugins ({len(plugins)}):")
                    for p in plugins:
                        status = "✓" if p["enabled"] else "✗"
                        version = f" v{p['version']}" if p["version"] else ""
                        tools = f"{p['tools']} tools" if p["tools"] else ""
                        hooks = f"{p['hooks']} hooks" if p["hooks"] else ""
                        parts = [x for x in [tools, hooks] if x]
                        detail = f" ({', '.join(parts)})" if parts else ""
                        error = f" — {p['error']}" if p["error"] else ""
                        print(f"  {status} {p['name']}{version}{detail}{error}")
            except Exception as e:
                print(f"Plugin system error: {e}")
        elif canonical == "rollback":
            self._handle_rollback_command(cmd_original)
        elif canonical == "stop":
            self._handle_stop_command()
        elif canonical == "background":
            self._handle_background_command(cmd_original)
        elif canonical == "btw":
            self._handle_btw_command(cmd_original)
        elif canonical == "queue":
            # Extract prompt after "/queue " or "/q "
            parts = cmd_original.split(None, 1)
            payload = parts[1].strip() if len(parts) > 1 else ""
            if not payload:
                _cprint("  Usage: /queue <prompt>")
            else:
                self._pending_input.put(payload)
                if self._agent_running:
                    _cprint(f"  Queued for the next turn: {payload[:80]}{'...' if len(payload) > 80 else ''}")
                else:
                    _cprint(f"  Queued: {payload[:80]}{'...' if len(payload) > 80 else ''}")
        elif canonical == "skin":
            self._handle_skin_command(cmd_original)
        elif canonical == "voice":
            self._handle_voice_command(cmd_original)
        else:
            # Check for user-defined quick commands (bypass agent loop, no LLM call)
            base_cmd = cmd_lower.split()[0]
            quick_commands = self.config.get("quick_commands", {})
            if base_cmd.lstrip("/") in quick_commands:
                qcmd = quick_commands[base_cmd.lstrip("/")]
                if qcmd.get("type") == "exec":
                    import subprocess
                    exec_cmd = qcmd.get("command", "")
                    if exec_cmd:
                        try:
                            result = subprocess.run(
                                exec_cmd, shell=True, capture_output=True,
                                text=True, timeout=30
                            )
                            output = result.stdout.strip() or result.stderr.strip()
                            if output:
                                self.console.print(_rich_text_from_ansi(output))
                            else:
                                self.console.print("[dim]Command returned no output[/]")
                        except subprocess.TimeoutExpired:
                            self.console.print("[bold red]Quick command timed out (30s)[/]")
                        except Exception as e:
                            self.console.print(f"[bold red]Quick command error: {e}[/]")
                    else:
                        self.console.print(f"[bold red]Quick command '{base_cmd}' has no command defined[/]")
                elif qcmd.get("type") == "alias":
                    target = qcmd.get("target", "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        user_args = cmd_original[len(base_cmd):].strip()
                        aliased_command = f"{target} {user_args}".strip()
                        return self.process_command(aliased_command)
                    else:
                        self.console.print(f"[bold red]Quick command '{base_cmd}' has no target defined[/]")
                else:
                    self.console.print(f"[bold red]Quick command '{base_cmd}' has unsupported type (supported: 'exec', 'alias')[/]")
            # Check for plugin-registered slash commands
            elif base_cmd.lstrip("/") in _get_plugin_cmd_handler_names():
                from hermes_cli.plugins import get_plugin_command_handler
                plugin_handler = get_plugin_command_handler(base_cmd.lstrip("/"))
                if plugin_handler:
                    user_args = cmd_original[len(base_cmd):].strip()
                    try:
                        result = plugin_handler(user_args)
                        if result:
                            _cprint(str(result))
                    except Exception as e:
                        _cprint(f"\033[1;31mPlugin command error: {e}{_RST}")
            # Check for skill slash commands (/gif-search, /axolotl, etc.)
            elif base_cmd in _skill_commands:
                user_instruction = cmd_original[len(base_cmd):].strip()
                msg = build_skill_invocation_message(
                    base_cmd, user_instruction, task_id=self.session_id
                )
                if msg:
                    skill_name = _skill_commands[base_cmd]["name"]
                    print(f"\n⚡ Loading skill: {skill_name}")
                    if hasattr(self, '_pending_input'):
                        self._pending_input.put(msg)
                else:
                    self.console.print(f"[bold red]Failed to load skill for {base_cmd}[/]")
            else:
                # Prefix matching: if input uniquely identifies one command, execute it.
                # Matches against both built-in COMMANDS and installed skill commands so
                # that execution-time resolution agrees with tab-completion.
                from hermes_cli.commands import COMMANDS
                typed_base = cmd_lower.split()[0]
                all_known = set(COMMANDS) | set(_skill_commands)
                matches = [c for c in all_known if c.startswith(typed_base)]
                if len(matches) > 1:
                    # Prefer an exact match (typed the full command name)
                    exact = [c for c in matches if c == typed_base]
                    if len(exact) == 1:
                        matches = exact
                    else:
                        # Prefer the unique shortest match:
                        # /qui → /quit (5) wins over /quint-pipeline (15)
                        min_len = min(len(c) for c in matches)
                        shortest = [c for c in matches if len(c) == min_len]
                        if len(shortest) == 1:
                            matches = shortest
                if len(matches) == 1:
                    # Expand the prefix to the full command name, preserving arguments.
                    # Guard against redispatching the same token to avoid infinite
                    # recursion when the expanded name still doesn't hit an exact branch
                    # (e.g. /config with extra args that are not yet handled above).
                    full_name = matches[0]
                    if full_name == typed_base:
                        # Already an exact token — no expansion possible; fall through
                        _cprint(f"\033[1;31mUnknown command: {cmd_lower}{_RST}")
                        _cprint(f"{_DIM}{_GOLD}Type /help for available commands{_RST}")
                    else:
                        remainder = cmd_original.strip()[len(typed_base):]
                        full_cmd = full_name + remainder
                        return self.process_command(full_cmd)
                elif len(matches) > 1:
                    _cprint(f"{_GOLD}Ambiguous command: {cmd_lower}{_RST}")
                    _cprint(f"{_DIM}Did you mean: {', '.join(sorted(matches))}?{_RST}")
                else:
                    _cprint(f"\033[1;31mUnknown command: {cmd_lower}{_RST}")
                    _cprint(f"{_DIM}{_GOLD}Type /help for available commands{_RST}")
        
        return True
    
    def _handle_plan_command(self, cmd: str):
        """Handle /plan [request] — load the bundled plan skill."""
        parts = cmd.strip().split(maxsplit=1)
        user_instruction = parts[1].strip() if len(parts) > 1 else ""

        plan_path = build_plan_path(user_instruction)
        msg = build_skill_invocation_message(
            "/plan",
            user_instruction,
            task_id=self.session_id,
            runtime_note=(
                "Save the markdown plan with write_file to this exact relative path "
                f"inside the active workspace/backend cwd: {plan_path}"
            ),
        )

        if not msg:
            self.console.print("[bold red]Failed to load the bundled /plan skill[/]")
            return

        _cprint(f"  📝 Plan mode queued via skill. Markdown plan target: {plan_path}")
        if hasattr(self, '_pending_input'):
            self._pending_input.put(msg)
        else:
            self.console.print("[bold red]Plan mode unavailable: input queue not initialized[/]")
    
    def _handle_background_command(self, cmd: str):
        """Handle /background <prompt> — run a prompt in a separate background session.

        Spawns a new AIAgent in a background thread with its own session.
        When it completes, prints the result to the CLI without modifying
        the active session's conversation history.
        """
        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            _cprint("  Usage: /background <prompt>")
            _cprint("  Example: /background Summarize the top HN stories today")
            _cprint("  The task runs in a separate session and results display here when done.")
            return

        prompt = parts[1].strip()
        self._background_task_counter += 1
        task_num = self._background_task_counter
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Make sure we have valid credentials
        if not self._ensure_runtime_credentials():
            _cprint("  (>_<) Cannot start background task: no valid credentials.")
            return

        _cprint(f"  🔄 Background task #{task_num} started: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
        _cprint(f"  Task ID: {task_id}")
        _cprint("  You can continue chatting — results will appear when done.\n")

        turn_route = self._resolve_turn_agent_config(prompt)

        def run_background():
            try:
                bg_agent = AIAgent(
                    model=turn_route["model"],
                    api_key=turn_route["runtime"].get("api_key"),
                    base_url=turn_route["runtime"].get("base_url"),
                    provider=turn_route["runtime"].get("provider"),
                    api_mode=turn_route["runtime"].get("api_mode"),
                    acp_command=turn_route["runtime"].get("command"),
                    acp_args=turn_route["runtime"].get("args"),
                    max_iterations=self.max_turns,
                    enabled_toolsets=self.enabled_toolsets,
                    quiet_mode=True,
                    verbose_logging=False,
                    session_id=task_id,
                    platform="cli",
                    session_db=self._session_db,
                    reasoning_config=self.reasoning_config,
                    providers_allowed=self._providers_only,
                    providers_ignored=self._providers_ignore,
                    providers_order=self._providers_order,
                    provider_sort=self._provider_sort,
                    provider_require_parameters=self._provider_require_params,
                    provider_data_collection=self._provider_data_collection,
                    fallback_model=self._fallback_model,
                )
                # Silence raw spinner; route thinking through TUI widget when no foreground agent is active.
                bg_agent._print_fn = lambda *_a, **_kw: None

                def _bg_thinking(text: str) -> None:
                    # Concurrent bg tasks may race on _spinner_text; acceptable for best-effort UI.
                    if not self._agent_running:
                        self._spinner_text = text
                        if self._app:
                            self._app.invalidate()

                bg_agent.thinking_callback = _bg_thinking

                result = bg_agent.run_conversation(
                    user_message=prompt,
                    task_id=task_id,
                )

                response = result.get("final_response", "") if result else ""
                if not response and result and result.get("error"):
                    response = f"Error: {result['error']}"

                # Display result in the CLI (thread-safe via patch_stdout).
                # Force a TUI refresh first so spinner/status bar don't overlap
                # with the output (fixes #2718).
                if self._app:
                    self._app.invalidate()
                    import time as _tmod
                    _tmod.sleep(0.05)  # brief pause for refresh
                print()
                ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
                _cprint(f"  ✅ Background task #{task_num} complete")
                _cprint(f"  Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
                ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
                if response:
                    try:
                        from hermes_cli.skin_engine import get_active_skin
                        _skin = get_active_skin()
                        label = _skin.get_branding("response_label", "⚕ Hermes")
                        _resp_color = _skin.get_color("response_border", "#CD7F32")
                        _resp_text = _skin.get_color("banner_text", "#FFF8DC")
                    except Exception:
                        label = "⚕ Hermes"
                        _resp_color = "#CD7F32"
                        _resp_text = "#FFF8DC"

                    _chat_console = ChatConsole()
                    _chat_console.print(Panel(
                        _rich_text_from_ansi(response),
                        title=f"[{_resp_color} bold]{label} (background #{task_num})[/]",
                        title_align="left",
                        border_style=_resp_color,
                        style=_resp_text,
                        box=rich_box.HORIZONTALS,
                        padding=(1, 2),
                    ))
                else:
                    _cprint("  (No response generated)")

                # Play bell if enabled
                if self.bell_on_complete:
                    sys.stdout.write("\a")
                    sys.stdout.flush()

            except Exception as e:
                # Same TUI refresh pattern as success path (#2718)
                if self._app:
                    self._app.invalidate()
                    import time as _tmod
                    _tmod.sleep(0.05)
                print()
                _cprint(f"  ❌ Background task #{task_num} failed: {e}")
            finally:
                self._background_tasks.pop(task_id, None)
                # Clear spinner only if no foreground agent owns it
                if not self._agent_running:
                    self._spinner_text = ""
                if self._app:
                    self._invalidate(min_interval=0)

        thread = threading.Thread(target=run_background, daemon=True, name=f"bg-task-{task_id}")
        self._background_tasks[task_id] = thread
        thread.start()

    def _handle_btw_command(self, cmd: str):
        """Handle /btw <question> — ephemeral side question using session context.

        Snapshots the current conversation history, spawns a no-tools agent in
        a background thread, and prints the answer without persisting anything
        to the main session.
        """
        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            _cprint("  Usage: /btw <question>")
            _cprint("  Example: /btw what module owns session title sanitization?")
            _cprint("  Answers using session context. No tools, not persisted.")
            return

        question = parts[1].strip()
        task_id = f"btw_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"

        if not self._ensure_runtime_credentials():
            _cprint("  (>_<) Cannot start /btw: no valid credentials.")
            return

        turn_route = self._resolve_turn_agent_config(question)
        history_snapshot = list(self.conversation_history)

        preview = question[:60] + ("..." if len(question) > 60 else "")
        _cprint(f'  💬 /btw: "{preview}"')

        def run_btw():
            try:
                btw_agent = AIAgent(
                    model=turn_route["model"],
                    api_key=turn_route["runtime"].get("api_key"),
                    base_url=turn_route["runtime"].get("base_url"),
                    provider=turn_route["runtime"].get("provider"),
                    api_mode=turn_route["runtime"].get("api_mode"),
                    acp_command=turn_route["runtime"].get("command"),
                    acp_args=turn_route["runtime"].get("args"),
                    max_iterations=8,
                    enabled_toolsets=[],
                    quiet_mode=True,
                    verbose_logging=False,
                    session_id=task_id,
                    platform="cli",
                    reasoning_config=self.reasoning_config,
                    providers_allowed=self._providers_only,
                    providers_ignored=self._providers_ignore,
                    providers_order=self._providers_order,
                    provider_sort=self._provider_sort,
                    provider_require_parameters=self._provider_require_params,
                    provider_data_collection=self._provider_data_collection,
                    fallback_model=self._fallback_model,
                    session_db=None,
                    skip_memory=True,
                    skip_context_files=True,
                    persist_session=False,
                )

                btw_prompt = (
                    "[Ephemeral /btw side question. Answer using the conversation "
                    "context. No tools available. Be direct and concise.]\n\n"
                    + question
                )
                result = btw_agent.run_conversation(
                    user_message=btw_prompt,
                    conversation_history=history_snapshot,
                    task_id=task_id,
                )

                response = (result.get("final_response") or "") if result else ""
                if not response and result and result.get("error"):
                    response = f"Error: {result['error']}"

                # TUI refresh before printing
                if self._app:
                    self._app.invalidate()
                    time.sleep(0.05)
                print()

                if response:
                    try:
                        from hermes_cli.skin_engine import get_active_skin
                        _skin = get_active_skin()
                        _resp_color = _skin.get_color("response_border", "#4F6D4A")
                    except Exception:
                        _resp_color = "#4F6D4A"

                    ChatConsole().print(Panel(
                        _rich_text_from_ansi(response),
                        title=f"[{_resp_color} bold]⚕ /btw[/]",
                        title_align="left",
                        border_style=_resp_color,
                        box=rich_box.HORIZONTALS,
                        padding=(1, 2),
                    ))
                else:
                    _cprint("  💬 /btw: (no response)")

                if self.bell_on_complete:
                    sys.stdout.write("\a")
                    sys.stdout.flush()

            except Exception as e:
                if self._app:
                    self._app.invalidate()
                    time.sleep(0.05)
                print()
                _cprint(f"  ❌ /btw failed: {e}")
            finally:
                if self._app:
                    self._invalidate(min_interval=0)

        thread = threading.Thread(target=run_btw, daemon=True, name=f"btw-{task_id}")
        thread.start()

    @staticmethod
    def _try_launch_chrome_debug(port: int, system: str) -> bool:
        """Try to launch Chrome/Chromium with remote debugging enabled.

        Returns True if a launch command was executed (doesn't guarantee success).
        """
        import shutil
        import subprocess as _sp

        candidates = []
        if system == "Darwin":
            # macOS: try common app bundle locations
            for app in (
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            ):
                if os.path.isfile(app):
                    candidates.append(app)
        else:
            # Linux: try common binary names
            for name in ("google-chrome", "google-chrome-stable", "chromium-browser",
                         "chromium", "brave-browser", "microsoft-edge"):
                path = shutil.which(name)
                if path:
                    candidates.append(path)

        if not candidates:
            return False

        chrome = candidates[0]
        try:
            _sp.Popen(
                [chrome, f"--remote-debugging-port={port}"],
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
                start_new_session=True,  # detach from terminal
            )
            return True
        except Exception:
            return False

    def _handle_browser_command(self, cmd: str):
        """Handle /browser connect|disconnect|status — manage live Chrome CDP connection."""
        import platform as _plat

        parts = cmd.strip().split(None, 1)
        sub = parts[1].lower().strip() if len(parts) > 1 else "status"

        _DEFAULT_CDP = "http://localhost:9222"
        current = os.environ.get("BROWSER_CDP_URL", "").strip()

        if sub.startswith("connect"):
            # Optionally accept a custom CDP URL: /browser connect ws://host:port
            connect_parts = cmd.strip().split(None, 2)  # ["/browser", "connect", "ws://..."]
            cdp_url = connect_parts[2].strip() if len(connect_parts) > 2 else _DEFAULT_CDP

            # Clear any existing browser sessions so the next tool call uses the new backend
            try:
                from tools.browser_tool import cleanup_all_browsers
                cleanup_all_browsers()
            except Exception:
                pass

            print()

            # Extract port for connectivity checks
            _port = 9222
            try:
                _port = int(cdp_url.rsplit(":", 1)[-1].split("/")[0])
            except (ValueError, IndexError):
                pass

            # Check if Chrome is already listening on the debug port
            import socket
            _already_open = False
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect(("127.0.0.1", _port))
                s.close()
                _already_open = True
            except (OSError, socket.timeout):
                pass

            if _already_open:
                print(f"   ✓ Chrome is already listening on port {_port}")
            elif cdp_url == _DEFAULT_CDP:
                # Try to auto-launch Chrome with remote debugging
                print("   Chrome isn't running with remote debugging — attempting to launch...")
                _launched = self._try_launch_chrome_debug(_port, _plat.system())
                if _launched:
                    # Wait for the port to come up
                    import time as _time
                    for _wait in range(10):
                        try:
                            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s.settimeout(1)
                            s.connect(("127.0.0.1", _port))
                            s.close()
                            _already_open = True
                            break
                        except (OSError, socket.timeout):
                            _time.sleep(0.5)
                    if _already_open:
                        print(f"   ✓ Chrome launched and listening on port {_port}")
                    else:
                        print(f"   ⚠ Chrome launched but port {_port} isn't responding yet")
                        print("     You may need to close existing Chrome windows first and retry")
                else:
                    print("   ⚠ Could not auto-launch Chrome")
                    # Show manual instructions as fallback
                    sys_name = _plat.system()
                    if sys_name == "Darwin":
                        chrome_cmd = 'open -a "Google Chrome" --args --remote-debugging-port=9222'
                    elif sys_name == "Windows":
                        chrome_cmd = 'chrome.exe --remote-debugging-port=9222'
                    else:
                        chrome_cmd = "google-chrome --remote-debugging-port=9222"
                    print(f"     Launch Chrome manually: {chrome_cmd}")
            else:
                print(f"   ⚠ Port {_port} is not reachable at {cdp_url}")

            os.environ["BROWSER_CDP_URL"] = cdp_url
            print()
            print("🌐 Browser connected to live Chrome via CDP")
            print(f"   Endpoint: {cdp_url}")
            print()

            # Inject context message so the model knows
            if hasattr(self, '_pending_input'):
                self._pending_input.put(
                    "[System note: The user has connected your browser tools to their live Chrome browser "
                    "via Chrome DevTools Protocol. Your browser_navigate, browser_snapshot, browser_click, "
                    "and other browser tools now control their real browser — including any pages they have "
                    "open, logged-in sessions, and cookies. They likely opened specific sites or logged into "
                    "services before connecting. Please await their instruction before attempting to operate "
                    "the browser. When you do act, be mindful that your actions affect their real browser — "
                    "don't close tabs or navigate away from pages without asking.]"
                )

        elif sub == "disconnect":
            if current:
                os.environ.pop("BROWSER_CDP_URL", None)
                try:
                    from tools.browser_tool import cleanup_all_browsers
                    cleanup_all_browsers()
                except Exception:
                    pass
                print()
                print("🌐 Browser disconnected from live Chrome")
                print("   Browser tools reverted to default mode (local headless or Browserbase)")
                print()

                if hasattr(self, '_pending_input'):
                    self._pending_input.put(
                        "[System note: The user has disconnected the browser tools from their live Chrome. "
                        "Browser tools are back to default mode (headless local browser or Browserbase cloud).]"
                    )
            else:
                print()
                print("Browser is not connected to live Chrome (already using default mode)")
                print()

        elif sub == "status":
            print()
            if current:
                print("🌐 Browser: connected to live Chrome via CDP")
                print(f"   Endpoint: {current}")

                _port = 9222
                try:
                    _port = int(current.rsplit(":", 1)[-1].split("/")[0])
                except (ValueError, IndexError):
                    pass
                try:
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect(("127.0.0.1", _port))
                    s.close()
                    print("   Status: ✓ reachable")
                except (OSError, Exception):
                    print("   Status: ⚠ not reachable (Chrome may not be running)")
            elif os.environ.get("BROWSERBASE_API_KEY"):
                print("🌐 Browser: Browserbase (cloud)")
            else:
                print("🌐 Browser: local headless Chromium (agent-browser)")
            print()
            print("   /browser connect      — connect to your live Chrome")
            print("   /browser disconnect   — revert to default")
            print()

        else:
            print()
            print("Usage: /browser connect|disconnect|status")
            print()
            print("   connect      Connect browser tools to your live Chrome session")
            print("   disconnect   Revert to default browser backend")
            print("   status       Show current browser mode")
            print()

    def _handle_skin_command(self, cmd: str):
        """Handle /skin [name] — show or change the display skin."""
        try:
            from hermes_cli.skin_engine import list_skins, set_active_skin, get_active_skin_name
        except ImportError:
            print("Skin engine not available.")
            return

        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            # Show current skin and list available
            current = get_active_skin_name()
            skins = list_skins()
            print(f"\n  Current skin: {current}")
            print("  Available skins:")
            for s in skins:
                marker = " ●" if s["name"] == current else "  "
                source = f" ({s['source']})" if s["source"] == "user" else ""
                print(f"   {marker} {s['name']}{source} — {s['description']}")
            print("\n  Usage: /skin <name>")
            print(f"  Custom skins: drop a YAML file in {display_hermes_home()}/skins/\n")
            return

        new_skin = parts[1].strip().lower()
        available = {s["name"] for s in list_skins()}
        if new_skin not in available:
            print(f"  Unknown skin: {new_skin}")
            print(f"  Available: {', '.join(sorted(available))}")
            return

        set_active_skin(new_skin)
        if save_config_value("display.skin", new_skin):
            print(f"  Skin set to: {new_skin} (saved)")
        else:
            print(f"  Skin set to: {new_skin}")
        print("  Note: banner colors will update on next session start.")
        if self._apply_tui_skin_style():
            print("  Prompt + TUI colors updated.")

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
            self.agent.reasoning_callback = self._current_reasoning_callback()

        # Use raw ANSI codes via _cprint so the output is routed through
        # prompt_toolkit's renderer.  self.console.print() with Rich markup
        # writes directly to stdout which patch_stdout's StdoutProxy mangles
        # into garbled sequences like '?[33mTool progress: NEW?[0m' (#2262).
        from hermes_cli.colors import Colors as _Colors
        labels = {
            "off": f"{_Colors.DIM}Tool progress: OFF{_Colors.RESET} — silent mode, just the final response.",
            "new": f"{_Colors.YELLOW}Tool progress: NEW{_Colors.RESET} — show each new tool (skip repeats).",
            "all": f"{_Colors.GREEN}Tool progress: ALL{_Colors.RESET} — show every tool call.",
            "verbose": f"{_Colors.BOLD}{_Colors.GREEN}Tool progress: VERBOSE{_Colors.RESET} — full args, results, think blocks, and debug logs.",
        }
        _cprint(labels.get(self.tool_progress_mode, ""))

    def _toggle_yolo(self):
        """Toggle YOLO mode — skip all dangerous command approval prompts."""
        import os
        current = bool(os.environ.get("HERMES_YOLO_MODE"))
        if current:
            os.environ.pop("HERMES_YOLO_MODE", None)
            self.console.print("  ⚠ YOLO mode [bold red]OFF[/] — dangerous commands will require approval.")
        else:
            os.environ["HERMES_YOLO_MODE"] = "1"
            self.console.print("  ⚡ YOLO mode [bold green]ON[/] — all commands auto-approved. Use with caution.")

    def _handle_reasoning_command(self, cmd: str):
        """Handle /reasoning — manage effort level and display toggle.

        Usage:
            /reasoning              Show current effort level and display state
            /reasoning <level>      Set reasoning effort (none, low, medium, high, xhigh)
            /reasoning show|on      Show model thinking/reasoning in output
            /reasoning hide|off     Hide model thinking/reasoning from output
        """
        parts = cmd.strip().split(maxsplit=1)

        if len(parts) < 2:
            # Show current state
            rc = self.reasoning_config
            if rc is None:
                level = "medium (default)"
            elif rc.get("enabled") is False:
                level = "none (disabled)"
            else:
                level = rc.get("effort", "medium")
            display_state = "on ✓" if self.show_reasoning else "off"
            _cprint(f"  {_GOLD}Reasoning effort:  {level}{_RST}")
            _cprint(f"  {_GOLD}Reasoning display: {display_state}{_RST}")
            _cprint(f"  {_DIM}Usage: /reasoning <none|low|medium|high|xhigh|show|hide>{_RST}")
            return

        arg = parts[1].strip().lower()

        # Display toggle
        if arg in ("show", "on"):
            self.show_reasoning = True
            if self.agent:
                self.agent.reasoning_callback = self._current_reasoning_callback()
            save_config_value("display.show_reasoning", True)
            _cprint(f"  {_GOLD}✓ Reasoning display: ON (saved){_RST}")
            _cprint(f"  {_DIM}  Model thinking will be shown during and after each response.{_RST}")
            return
        if arg in ("hide", "off"):
            self.show_reasoning = False
            if self.agent:
                self.agent.reasoning_callback = self._current_reasoning_callback()
            save_config_value("display.show_reasoning", False)
            _cprint(f"  {_GOLD}✓ Reasoning display: OFF (saved){_RST}")
            return

        # Effort level change
        parsed = _parse_reasoning_config(arg)
        if parsed is None:
            _cprint(f"  {_DIM}(._.) Unknown argument: {arg}{_RST}")
            _cprint(f"  {_DIM}Valid levels: none, low, minimal, medium, high, xhigh{_RST}")
            _cprint(f"  {_DIM}Display:      show, hide{_RST}")
            return

        self.reasoning_config = parsed
        self.agent = None  # Force agent re-init with new reasoning config

        if save_config_value("agent.reasoning_effort", arg):
            _cprint(f"  {_GOLD}✓ Reasoning effort set to '{arg}' (saved to config){_RST}")
        else:
            _cprint(f"  {_GOLD}✓ Reasoning effort set to '{arg}' (session only){_RST}")

    def _on_reasoning(self, reasoning_text: str):
        """Callback for intermediate reasoning display during tool-call loops."""
        if not reasoning_text:
            return
        self._reasoning_preview_buf = getattr(self, "_reasoning_preview_buf", "") + reasoning_text
        self._flush_reasoning_preview(force=False)

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
        input_tokens = getattr(agent, "session_input_tokens", 0) or 0
        output_tokens = getattr(agent, "session_output_tokens", 0) or 0
        cache_read_tokens = getattr(agent, "session_cache_read_tokens", 0) or 0
        cache_write_tokens = getattr(agent, "session_cache_write_tokens", 0) or 0
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
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        compressions = compressor.compression_count

        msg_count = len(self.conversation_history)
        cost_result = estimate_usage_cost(
            agent.model,
            CanonicalUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
            provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None),
        )
        elapsed = format_duration_compact((datetime.now() - self.session_start).total_seconds())

        print("  📊 Session Token Usage")
        print(f"  {'─' * 40}")
        print(f"  Model:                     {agent.model}")
        print(f"  Input tokens:              {input_tokens:>10,}")
        print(f"  Cache read tokens:         {cache_read_tokens:>10,}")
        print(f"  Cache write tokens:        {cache_write_tokens:>10,}")
        print(f"  Output tokens:             {output_tokens:>10,}")
        print(f"  Prompt tokens (total):     {prompt:>10,}")
        print(f"  Completion tokens:         {completion:>10,}")
        print(f"  Total tokens:              {total:>10,}")
        print(f"  API calls:                 {calls:>10,}")
        print(f"  Session duration:          {elapsed:>10}")
        print(f"  Cost status:              {cost_result.status:>10}")
        print(f"  Cost source:              {cost_result.source:>10}")
        if cost_result.amount_usd is not None:
            prefix = "~" if cost_result.status == "estimated" else ""
            print(f"  Total cost:              {prefix}${float(cost_result.amount_usd):>10.4f}")
        elif cost_result.status == "included":
            print(f"  Total cost:              {'included':>10}")
        else:
            print(f"  Total cost:              {'n/a':>10}")
        print(f"  {'─' * 40}")
        print(f"  Current context:  {last_prompt:,} / {ctx_len:,} ({pct:.0f}%)")
        print(f"  Messages:         {msg_count}")
        print(f"  Compressions:     {compressions}")
        if cost_result.status == "unknown":
            print(f"  Note:             Pricing unknown for {agent.model}")

        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            for noisy in ('openai', 'openai._base_client', 'httpx', 'httpcore', 'asyncio', 'hpack', 'grpc', 'modal'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.INFO)
            for quiet_logger in ('tools', 'run_agent', 'trajectory_compressor', 'cron', 'hermes_cli'):
                logging.getLogger(quiet_logger).setLevel(logging.ERROR)

    def _show_insights(self, command: str = "/insights"):
        """Show usage insights and analytics from session history."""
        # Parse optional --days flag
        parts = command.split()
        days = 30
        source = None
        i = 1
        while i < len(parts):
            if parts[i] == "--days" and i + 1 < len(parts):
                try:
                    days = int(parts[i + 1])
                except ValueError:
                    print(f"  Invalid --days value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--source" and i + 1 < len(parts):
                source = parts[i + 1]
                i += 2
            else:
                i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            db = SessionDB()
            engine = InsightsEngine(db)
            report = engine.generate(days=days, source=source)
            print(engine.format_terminal(report))
            db.close()
        except Exception as e:
            print(f"  Error generating insights: {e}")

    def _check_config_mcp_changes(self) -> None:
        """Detect mcp_servers changes in config.yaml and auto-reload MCP connections.

        Called from process_loop every CONFIG_WATCH_INTERVAL seconds.
        Compares config.yaml mtime + mcp_servers section against the last
        known state.  When a change is detected, triggers _reload_mcp() and
        informs the user so they know the tool list has been refreshed.
        """
        import time
        import yaml as _yaml

        CONFIG_WATCH_INTERVAL = 5.0  # seconds between config.yaml stat() calls

        now = time.monotonic()
        if now - self._last_config_check < CONFIG_WATCH_INTERVAL:
            return
        self._last_config_check = now

        from hermes_cli.config import get_config_path as _get_config_path
        cfg_path = _get_config_path()
        if not cfg_path.exists():
            return

        try:
            mtime = cfg_path.stat().st_mtime
        except OSError:
            return

        if mtime == self._config_mtime:
            return  # File unchanged — fast path

        # File changed — check whether mcp_servers section changed
        self._config_mtime = mtime
        try:
            with open(cfg_path, encoding="utf-8") as f:
                new_cfg = _yaml.safe_load(f) or {}
        except Exception:
            return

        new_mcp = new_cfg.get("mcp_servers") or {}
        if new_mcp == self._config_mcp_servers:
            return  # mcp_servers unchanged (some other section was edited)

        self._config_mcp_servers = new_mcp
        # Notify user and reload.  Run in a separate thread with a hard
        # timeout so a hung MCP server cannot block the process_loop
        # indefinitely (which would freeze the entire TUI).
        print()
        print("🔄 MCP server config changed — reloading connections...")
        _reload_thread = threading.Thread(
            target=self._reload_mcp, daemon=True
        )
        _reload_thread.start()
        _reload_thread.join(timeout=30)
        if _reload_thread.is_alive():
            print("  ⚠️  MCP reload timed out (30s). Some servers may not have reconnected.")

    def _reload_mcp(self):
        """Reload MCP servers: disconnect all, re-read config.yaml, reconnect.

        After reconnecting, refreshes the agent's tool list so the model
        sees the updated tools on the next turn.
        """
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _servers, _lock

            # Capture old server names
            with _lock:
                old_servers = set(_servers.keys())

            if not self._command_running:
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

    # ====================================================================
    # Tool-call generation indicator (shown during streaming)
    # ====================================================================

    def _on_tool_gen_start(self, tool_name: str) -> None:
        """Called when the model begins generating tool-call arguments.

        Closes any open streaming boxes (reasoning / response) exactly once,
        then prints a short status line so the user sees activity instead of
        a frozen screen while a large payload (e.g. 45 KB write_file) streams.
        """
        if getattr(self, "_stream_box_opened", False):
            self._flush_stream()
            self._stream_box_opened = False
        self._close_reasoning_box()

        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(tool_name, default="⚡")
        _cprint(f"  ┊ {emoji} preparing {tool_name}…")

    # ====================================================================
    # Tool progress callback (audio cues for voice mode)
    # ====================================================================

    def _on_tool_progress(self, event_type: str, function_name: str = None, preview: str = None, function_args: dict = None, **kwargs):
        """Called on tool lifecycle events (tool.started, tool.completed, reasoning.available, etc.).

        Updates the TUI spinner widget so the user can see what the agent
        is doing during tool execution (fills the gap between thinking
        spinner and next response).  Also plays audio cue in voice mode.
        """
        # Only act on tool.started; ignore tool.completed, reasoning.available, etc.
        if event_type != "tool.started":
            return
        if function_name and not function_name.startswith("_"):
            from agent.display import get_tool_emoji
            emoji = get_tool_emoji(function_name)
            label = preview or function_name
            from agent.display import get_tool_preview_max_len
            _pl = get_tool_preview_max_len()
            if _pl > 0 and len(label) > _pl:
                label = label[:_pl - 3] + "..."
            self._spinner_text = f"{emoji} {label}"
            self._invalidate()

        if not self._voice_mode:
            return
        if not function_name or function_name.startswith("_"):
            return
        try:
            from tools.voice_mode import play_beep
            threading.Thread(
                target=play_beep,
                kwargs={"frequency": 1200, "duration": 0.06, "count": 1},
                daemon=True,
            ).start()
        except Exception:
            pass

    def _on_tool_start(self, tool_call_id: str, function_name: str, function_args: dict):
        """Capture local before-state for write-capable tools."""
        try:
            from agent.display import capture_local_edit_snapshot

            snapshot = capture_local_edit_snapshot(function_name, function_args)
            if snapshot is not None:
                self._pending_edit_snapshots[tool_call_id] = snapshot
        except Exception:
            logger.debug("Edit snapshot capture failed for %s", function_name, exc_info=True)

    def _on_tool_complete(self, tool_call_id: str, function_name: str, function_args: dict, function_result: str):
        """Render file edits with inline diff after write-capable tools complete."""
        snapshot = self._pending_edit_snapshots.pop(tool_call_id, None)
        try:
            from agent.display import render_edit_diff_with_delta

            render_edit_diff_with_delta(
                function_name,
                function_result,
                function_args=function_args,
                snapshot=snapshot,
                print_fn=_cprint,
            )
        except Exception:
            logger.debug("Edit diff preview failed for %s", function_name, exc_info=True)

    # ====================================================================
    # Voice mode methods
    # ====================================================================

    def _voice_start_recording(self):
        """Start capturing audio from the microphone."""
        if getattr(self, '_should_exit', False):
            return
        from tools.voice_mode import AudioRecorder, check_voice_requirements

        reqs = check_voice_requirements()
        if not reqs["audio_available"]:
            raise RuntimeError(
                "Voice mode requires sounddevice and numpy.\n"
                "Install with: pip install sounddevice numpy\n"
                "Or: pip install hermes-agent[voice]"
            )
        if not reqs.get("stt_available", reqs.get("stt_key_set")):
            raise RuntimeError(
                "Voice mode requires an STT provider for transcription.\n"
                "Option 1: pip install faster-whisper  (free, local)\n"
                "Option 2: Set GROQ_API_KEY (free tier)\n"
                "Option 3: Set VOICE_TOOLS_OPENAI_KEY (paid)"
            )

        # Prevent double-start from concurrent threads (atomic check-and-set)
        with self._voice_lock:
            if self._voice_recording:
                return
            self._voice_recording = True

        # Load silence detection params from config
        voice_cfg = {}
        try:
            from hermes_cli.config import load_config
            voice_cfg = load_config().get("voice", {})
        except Exception:
            pass

        if self._voice_recorder is None:
            self._voice_recorder = AudioRecorder()

        # Apply config-driven silence params
        self._voice_recorder._silence_threshold = voice_cfg.get("silence_threshold", 200)
        self._voice_recorder._silence_duration = voice_cfg.get("silence_duration", 3.0)

        def _on_silence():
            """Called by AudioRecorder when silence is detected after speech."""
            with self._voice_lock:
                if not self._voice_recording:
                    return
            _cprint(f"\n{_DIM}Silence detected, auto-stopping...{_RST}")
            if hasattr(self, '_app') and self._app:
                self._app.invalidate()
            self._voice_stop_and_transcribe()

        # Audio cue: single beep BEFORE starting stream (avoid CoreAudio conflict)
        try:
            from tools.voice_mode import play_beep
            play_beep(frequency=880, count=1)
        except Exception:
            pass

        try:
            self._voice_recorder.start(on_silence_stop=_on_silence)
        except Exception:
            with self._voice_lock:
                self._voice_recording = False
            raise
        _cprint(f"\n{_GOLD}● Recording...{_RST} {_DIM}(auto-stops on silence | Ctrl+B to stop & exit continuous){_RST}")

        # Periodically refresh prompt to update audio level indicator
        def _refresh_level():
            while True:
                with self._voice_lock:
                    still_recording = self._voice_recording
                if not still_recording:
                    break
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()
                time.sleep(0.15)
        threading.Thread(target=_refresh_level, daemon=True).start()

    def _voice_stop_and_transcribe(self):
        """Stop recording, transcribe via STT, and queue the transcript as input."""
        # Atomic guard: only one thread can enter stop-and-transcribe.
        # Set _voice_processing immediately so concurrent Ctrl+B presses
        # don't race into the START path while recorder.stop() holds its lock.
        with self._voice_lock:
            if not self._voice_recording:
                return
            self._voice_recording = False
            self._voice_processing = True

        submitted = False
        wav_path = None
        try:
            if self._voice_recorder is None:
                return

            wav_path = self._voice_recorder.stop()

            # Audio cue: double beep after stream stopped (no CoreAudio conflict)
            try:
                from tools.voice_mode import play_beep
                play_beep(frequency=660, count=2)
            except Exception:
                pass

            if wav_path is None:
                _cprint(f"{_DIM}No speech detected.{_RST}")
                return

            # _voice_processing is already True (set atomically above)
            if hasattr(self, '_app') and self._app:
                self._app.invalidate()
            _cprint(f"{_DIM}Transcribing...{_RST}")

            # Get STT model from config
            stt_model = None
            try:
                from hermes_cli.config import load_config
                stt_config = load_config().get("stt", {})
                stt_model = stt_config.get("model")
            except Exception:
                pass

            from tools.voice_mode import transcribe_recording
            result = transcribe_recording(wav_path, model=stt_model)

            if result.get("success") and result.get("transcript", "").strip():
                transcript = result["transcript"].strip()
                self._pending_input.put(transcript)
                submitted = True
            elif result.get("success"):
                _cprint(f"{_DIM}No speech detected.{_RST}")
            else:
                error = result.get("error", "Unknown error")
                _cprint(f"\n{_DIM}Transcription failed: {error}{_RST}")

        except Exception as e:
            _cprint(f"\n{_DIM}Voice processing error: {e}{_RST}")
        finally:
            with self._voice_lock:
                self._voice_processing = False
            if hasattr(self, '_app') and self._app:
                self._app.invalidate()
            # Clean up temp file
            try:
                if wav_path and os.path.isfile(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass

            # Track consecutive no-speech cycles to avoid infinite restart loops.
            if not submitted:
                self._no_speech_count = getattr(self, '_no_speech_count', 0) + 1
                if self._no_speech_count >= 3:
                    self._voice_continuous = False
                    self._no_speech_count = 0
                    _cprint(f"{_DIM}No speech detected 3 times, continuous mode stopped.{_RST}")
                    return
            else:
                self._no_speech_count = 0

            # If no transcript was submitted but continuous mode is active,
            # restart recording so the user can keep talking.
            # (When transcript IS submitted, process_loop handles restart
            # after chat() completes.)
            if self._voice_continuous and not submitted and not self._voice_recording:
                def _restart_recording():
                    try:
                        self._voice_start_recording()
                        if hasattr(self, '_app') and self._app:
                            self._app.invalidate()
                    except Exception as e:
                        _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
                threading.Thread(target=_restart_recording, daemon=True).start()

    def _voice_speak_response(self, text: str):
        """Speak the agent's response aloud using TTS (runs in background thread)."""
        if not self._voice_tts:
            return
        self._voice_tts_done.clear()
        try:
            from tools.tts_tool import text_to_speech_tool
            from tools.voice_mode import play_audio_file
            import re

            # Strip markdown and non-speech content for cleaner TTS
            tts_text = text[:4000] if len(text) > 4000 else text
            tts_text = re.sub(r'```[\s\S]*?```', ' ', tts_text)   # fenced code blocks
            tts_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', tts_text)  # [text](url) -> text
            tts_text = re.sub(r'https?://\S+', '', tts_text)      # URLs
            tts_text = re.sub(r'\*\*(.+?)\*\*', r'\1', tts_text)  # bold
            tts_text = re.sub(r'\*(.+?)\*', r'\1', tts_text)      # italic
            tts_text = re.sub(r'`(.+?)`', r'\1', tts_text)        # inline code
            tts_text = re.sub(r'^#+\s*', '', tts_text, flags=re.MULTILINE)  # headers
            tts_text = re.sub(r'^\s*[-*]\s+', '', tts_text, flags=re.MULTILINE)  # list items
            tts_text = re.sub(r'---+', '', tts_text)              # horizontal rules
            tts_text = re.sub(r'\n{3,}', '\n\n', tts_text)        # excessive newlines
            tts_text = tts_text.strip()
            if not tts_text:
                return

            # Use MP3 output for CLI playback (afplay doesn't handle OGG well).
            # The TTS tool may auto-convert MP3->OGG, but the original MP3 remains.
            os.makedirs(os.path.join(tempfile.gettempdir(), "hermes_voice"), exist_ok=True)
            mp3_path = os.path.join(
                tempfile.gettempdir(), "hermes_voice",
                f"tts_{time.strftime('%Y%m%d_%H%M%S')}.mp3",
            )

            text_to_speech_tool(text=tts_text, output_path=mp3_path)

            # Play the MP3 directly (the TTS tool returns OGG path but MP3 still exists)
            if os.path.isfile(mp3_path) and os.path.getsize(mp3_path) > 0:
                play_audio_file(mp3_path)
                # Clean up
                try:
                    os.unlink(mp3_path)
                    ogg_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
                    if os.path.isfile(ogg_path):
                        os.unlink(ogg_path)
                except OSError:
                    pass
        except Exception as e:
            logger.warning("Voice TTS playback failed: %s", e)
            _cprint(f"{_DIM}TTS playback failed: {e}{_RST}")
        finally:
            self._voice_tts_done.set()

    def _handle_voice_command(self, command: str):
        """Handle /voice [on|off|tts|status] command."""
        parts = command.strip().split(maxsplit=1)
        subcommand = parts[1].lower().strip() if len(parts) > 1 else ""

        if subcommand == "on":
            self._enable_voice_mode()
        elif subcommand == "off":
            self._disable_voice_mode()
        elif subcommand == "tts":
            self._toggle_voice_tts()
        elif subcommand == "status":
            self._show_voice_status()
        elif subcommand == "":
            # Toggle
            if self._voice_mode:
                self._disable_voice_mode()
            else:
                self._enable_voice_mode()
        else:
            _cprint(f"Unknown voice subcommand: {subcommand}")
            _cprint("Usage: /voice [on|off|tts|status]")

    def _enable_voice_mode(self):
        """Enable voice mode after checking requirements."""
        if self._voice_mode:
            _cprint(f"{_DIM}Voice mode is already enabled.{_RST}")
            return

        from tools.voice_mode import check_voice_requirements, detect_audio_environment

        # Environment detection -- warn and block in incompatible environments
        env_check = detect_audio_environment()
        if not env_check["available"]:
            _cprint(f"\n{_GOLD}Voice mode unavailable in this environment:{_RST}")
            for warning in env_check["warnings"]:
                _cprint(f"  {_DIM}{warning}{_RST}")
            return

        reqs = check_voice_requirements()
        if not reqs["available"]:
            _cprint(f"\n{_GOLD}Voice mode requirements not met:{_RST}")
            for line in reqs["details"].split("\n"):
                _cprint(f"  {_DIM}{line}{_RST}")
            if reqs["missing_packages"]:
                _cprint(f"\n  {_BOLD}Install: pip install {' '.join(reqs['missing_packages'])}{_RST}")
                _cprint(f"  {_DIM}Or: pip install hermes-agent[voice]{_RST}")
            return

        with self._voice_lock:
            self._voice_mode = True

        # Check config for auto_tts
        try:
            from hermes_cli.config import load_config
            voice_config = load_config().get("voice", {})
            if voice_config.get("auto_tts", False):
                with self._voice_lock:
                    self._voice_tts = True
        except Exception:
            pass

        # Voice mode instruction is injected as a user message prefix (not a
        # system prompt change) to avoid invalidating the prompt cache.  See
        # _voice_message_prefix property and its usage in _process_message().

        tts_status = " (TTS enabled)" if self._voice_tts else ""
        try:
            from hermes_cli.config import load_config
            _raw_ptt = load_config().get("voice", {}).get("record_key", "ctrl+b")
            _ptt_key = _raw_ptt.lower().replace("ctrl+", "c-").replace("alt+", "a-")
        except Exception:
            _ptt_key = "c-b"
        _ptt_display = _ptt_key.replace("c-", "Ctrl+").upper()
        _cprint(f"\n{_GOLD}Voice mode enabled{tts_status}{_RST}")
        _cprint(f"  {_DIM}{_ptt_display} to start/stop recording{_RST}")
        _cprint(f"  {_DIM}/voice tts  to toggle speech output{_RST}")
        _cprint(f"  {_DIM}/voice off  to disable voice mode{_RST}")

    def _disable_voice_mode(self):
        """Disable voice mode, cancel any active recording, and stop TTS."""
        recorder = None
        with self._voice_lock:
            if self._voice_recording and self._voice_recorder:
                self._voice_recorder.cancel()
                self._voice_recording = False
            recorder = self._voice_recorder
            self._voice_mode = False
            self._voice_tts = False
            self._voice_continuous = False

        # Shut down the persistent audio stream in background
        if recorder is not None:
            def _bg_shutdown(rec=recorder):
                try:
                    rec.shutdown()
                except Exception:
                    pass
            threading.Thread(target=_bg_shutdown, daemon=True).start()
            self._voice_recorder = None

        # Stop any active TTS playback
        try:
            from tools.voice_mode import stop_playback
            stop_playback()
        except Exception:
            pass
        self._voice_tts_done.set()

        _cprint(f"\n{_DIM}Voice mode disabled.{_RST}")

    def _toggle_voice_tts(self):
        """Toggle TTS output for voice mode."""
        if not self._voice_mode:
            _cprint(f"{_DIM}Enable voice mode first: /voice on{_RST}")
            return

        with self._voice_lock:
            self._voice_tts = not self._voice_tts
        status = "enabled" if self._voice_tts else "disabled"

        if self._voice_tts:
            from tools.tts_tool import check_tts_requirements
            if not check_tts_requirements():
                _cprint(f"{_DIM}Warning: No TTS provider available. Install edge-tts or set API keys.{_RST}")

        _cprint(f"{_GOLD}Voice TTS {status}.{_RST}")

    def _show_voice_status(self):
        """Show current voice mode status."""
        from hermes_cli.config import load_config
        from tools.voice_mode import check_voice_requirements

        reqs = check_voice_requirements()

        _cprint(f"\n{_BOLD}Voice Mode Status{_RST}")
        _cprint(f"  Mode:      {'ON' if self._voice_mode else 'OFF'}")
        _cprint(f"  TTS:       {'ON' if self._voice_tts else 'OFF'}")
        _cprint(f"  Recording: {'YES' if self._voice_recording else 'no'}")
        _raw_key = load_config().get("voice", {}).get("record_key", "ctrl+b")
        _display_key = _raw_key.replace("ctrl+", "Ctrl+").upper() if "ctrl+" in _raw_key.lower() else _raw_key
        _cprint(f"  Record key: {_display_key}")
        _cprint(f"\n  {_BOLD}Requirements:{_RST}")
        for line in reqs["details"].split("\n"):
            _cprint(f"    {line}")

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

        # Poll for the user's response.  The countdown in the hint line
        # updates on each invalidate — but frequent repaints cause visible
        # flicker in some terminals (Kitty, ghostty).  We only refresh the
        # countdown every 5 s; selection changes (↑/↓) trigger instant
        # Poll for the user's response.  The countdown in the hint line
        # updates on each invalidate — but frequent repaints cause visible
        # flicker in some terminals (Kitty, ghostty).  We only refresh the
        # countdown every 5 s; selection changes (↑/↓) trigger instant
        # repaints via the key bindings.
        _last_countdown_refresh = _time.monotonic()
        while True:
            try:
                result = response_queue.get(timeout=1)
                self._clarify_deadline = 0
                return result
            except queue.Empty:
                remaining = self._clarify_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                # Only repaint every 5 s for the countdown — avoids flicker
                now = _time.monotonic()
                if now - _last_countdown_refresh >= 5.0:
                    _last_countdown_refresh = now
                    self._invalidate()
                if now - _last_countdown_refresh >= 5.0:
                    _last_countdown_refresh = now
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

    def _approval_callback(self, command: str, description: str,
                           *, allow_permanent: bool = True) -> str:
        """
        Prompt for dangerous command approval through the prompt_toolkit UI.

        Called from the agent thread. Shows a selection UI similar to clarify
        with choices: once / session / always / deny. When allow_permanent
        is False (tirith warnings present), the 'always' option is hidden.
        Long commands also get a 'view' option so the full command can be
        expanded before deciding.

        Uses _approval_lock to serialize concurrent requests (e.g. from
        parallel delegation subtasks) so each prompt gets its own turn
        and the shared _approval_state / _approval_deadline aren't clobbered.
        """
        import time as _time

        with self._approval_lock:
            timeout = 60
            response_queue = queue.Queue()

            self._approval_state = {
                "command": command,
                "description": description,
                "choices": self._approval_choices(command, allow_permanent=allow_permanent),
                "selected": 0,
                "response_queue": response_queue,
            }
            self._approval_deadline = _time.monotonic() + timeout

            self._invalidate()

            _last_countdown_refresh = _time.monotonic()
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
                    now = _time.monotonic()
                    if now - _last_countdown_refresh >= 5.0:
                        _last_countdown_refresh = now
                        self._invalidate()

            self._approval_state = None
            self._approval_deadline = 0
            self._invalidate()
            _cprint(f"\n{_DIM}  ⏱ Timeout — denying command{_RST}")
            return "deny"

    def _approval_choices(self, command: str, *, allow_permanent: bool = True) -> list[str]:
        """Return approval choices for a dangerous command prompt."""
        choices = ["once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]
        if len(command) > 70:
            choices.append("view")
        return choices

    def _handle_approval_selection(self) -> None:
        """Process the currently selected dangerous-command approval choice."""
        state = self._approval_state
        if not state:
            return

        selected = state.get("selected", 0)
        choices = state.get("choices") or []
        if not (0 <= selected < len(choices)):
            return

        chosen = choices[selected]
        if chosen == "view":
            state["show_full"] = True
            state["choices"] = [choice for choice in choices if choice != "view"]
            if state["selected"] >= len(state["choices"]):
                state["selected"] = max(0, len(state["choices"]) - 1)
            self._invalidate()
            return

        state["response_queue"].put(chosen)
        self._approval_state = None
        self._invalidate()

    def _get_approval_display_fragments(self):
        """Render the dangerous-command approval panel for the prompt_toolkit UI."""
        state = self._approval_state
        if not state:
            return []

        def _panel_box_width(title_text: str, content_lines: list[str], min_width: int = 46, max_width: int = 76) -> int:
            term_cols = shutil.get_terminal_size((100, 20)).columns
            longest = max([len(title_text)] + [len(line) for line in content_lines] + [min_width - 4])
            inner = min(max(longest + 4, min_width - 2), max_width - 2, max(24, term_cols - 6))
            return inner + 2

        def _wrap_panel_text(text: str, width: int, subsequent_indent: str = "") -> list[str]:
            wrapped = textwrap.wrap(
                text,
                width=max(8, width),
                replace_whitespace=False,
                drop_whitespace=False,
                subsequent_indent=subsequent_indent,
            )
            return wrapped or [""]

        def _append_panel_line(lines, border_style: str, content_style: str, text: str, box_width: int) -> None:
            inner_width = max(0, box_width - 2)
            lines.append((border_style, "│ "))
            lines.append((content_style, text.ljust(inner_width)))
            lines.append((border_style, " │\n"))

        def _append_blank_panel_line(lines, border_style: str, box_width: int) -> None:
            lines.append((border_style, "│" + (" " * box_width) + "│\n"))

        command = state["command"]
        description = state["description"]
        choices = state["choices"]
        selected = state.get("selected", 0)
        show_full = state.get("show_full", False)

        title = "⚠️  Dangerous Command"
        cmd_display = command if show_full or len(command) <= 70 else command[:70] + '...'
        choice_labels = {
            "once": "Allow once",
            "session": "Allow for this session",
            "always": "Add to permanent allowlist",
            "deny": "Deny",
            "view": "Show full command",
        }

        preview_lines = _wrap_panel_text(description, 60)
        preview_lines.extend(_wrap_panel_text(cmd_display, 60))
        for i, choice in enumerate(choices):
            prefix = '❯ ' if i == selected else '  '
            preview_lines.extend(_wrap_panel_text(
                f"{prefix}{choice_labels.get(choice, choice)}",
                60,
                subsequent_indent="  ",
            ))

        box_width = _panel_box_width(title, preview_lines)
        inner_text_width = max(8, box_width - 2)

        lines = []
        lines.append(('class:approval-border', '╭' + ('─' * box_width) + '╮\n'))
        _append_panel_line(lines, 'class:approval-border', 'class:approval-title', title, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        for wrapped in _wrap_panel_text(description, inner_text_width):
            _append_panel_line(lines, 'class:approval-border', 'class:approval-desc', wrapped, box_width)
        for wrapped in _wrap_panel_text(cmd_display, inner_text_width):
            _append_panel_line(lines, 'class:approval-border', 'class:approval-cmd', wrapped, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        for i, choice in enumerate(choices):
            label = choice_labels.get(choice, choice)
            style = 'class:approval-selected' if i == selected else 'class:approval-choice'
            prefix = '❯ ' if i == selected else '  '
            for wrapped in _wrap_panel_text(f"{prefix}{label}", inner_text_width, subsequent_indent="  "):
                _append_panel_line(lines, 'class:approval-border', style, wrapped, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        lines.append(('class:approval-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _secret_capture_callback(self, var_name: str, prompt: str, metadata=None) -> dict:
        return prompt_for_secret(self, var_name, prompt, metadata)

    def _submit_secret_response(self, value: str) -> None:
        if not self._secret_state:
            return
        self._secret_state["response_queue"].put(value)
        self._secret_state = None
        self._secret_deadline = 0
        self._invalidate()

    def _cancel_secret_capture(self) -> None:
        self._submit_secret_response("")

    def _clear_secret_input_buffer(self) -> None:
        if getattr(self, "_app", None):
            try:
                self._app.current_buffer.reset()
            except Exception:
                pass

    def _clear_current_input(self) -> None:
        if getattr(self, "_app", None):
            try:
                self._app.current_buffer.text = ""
            except Exception:
                pass


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
        # Single-query and direct chat callers do not go through run(), so
        # register secure secret capture here as well.
        set_secret_capture_callback(self._secret_capture_callback)

        # Refresh provider credentials if needed (handles key rotation transparently)
        if not self._ensure_runtime_credentials():
            return None

        turn_route = self._resolve_turn_agent_config(message)
        if turn_route["signature"] != self._active_agent_route_signature:
            self.agent = None

        # Initialize agent if needed
        if self.agent is None:
            _cprint(f"{_DIM}Initializing agent...{_RST}")
        if not self._init_agent(
            model_override=turn_route["model"],
            runtime_override=turn_route["runtime"],
            route_label=turn_route["label"],
        ):
            return None
        
        # Pre-process images through the vision tool (Gemini Flash) so the
        # main model receives text descriptions instead of raw base64 image
        # content — works with any model, not just vision-capable ones.
        if images:
            message = self._preprocess_images_with_vision(
                message if isinstance(message, str) else "", images
            )

        # Expand @ context references (e.g. @file:main.py, @diff, @folder:src/)
        if isinstance(message, str) and "@" in message:
            try:
                from agent.context_references import preprocess_context_references
                from agent.model_metadata import get_model_context_length
                _ctx_len = get_model_context_length(
                    self.model, base_url=self.base_url or "", api_key=self.api_key or "")
                _ctx_result = preprocess_context_references(
                    message, cwd=os.getcwd(), context_length=_ctx_len)
                if _ctx_result.expanded or _ctx_result.blocked:
                    if _ctx_result.references:
                        _cprint(
                            f"  {_DIM}[@ context: {len(_ctx_result.references)} ref(s), "
                            f"{_ctx_result.injected_tokens} tokens]{_RST}")
                    for w in _ctx_result.warnings:
                        _cprint(f"  {_DIM}⚠ {w}{_RST}")
                    if _ctx_result.blocked:
                        return "\n".join(_ctx_result.warnings) or "Context injection refused."
                    message = _ctx_result.message
            except Exception as e:
                logging.debug("@ context reference expansion failed: %s", e)

        # Sanitize surrogate characters that can arrive via clipboard paste from
        # rich-text editors (Google Docs, Word, etc.).  Lone surrogates are invalid
        # UTF-8 and crash JSON serialization in the OpenAI SDK.
        if isinstance(message, str):
            from run_agent import _sanitize_surrogates
            message = _sanitize_surrogates(message)

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})

        ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
        print(flush=True)
        
        try:
            # Run the conversation with interrupt monitoring
            result = None

            # Reset streaming display state for this turn
            self._reset_stream_state()
            # Separate from _reset_stream_state because this must persist
            # across intermediate turn boundaries (tool-calling loops) — only
            # reset at the start of each user turn.
            self._reasoning_shown_this_turn = False

            # --- Streaming TTS setup ---
            # When ElevenLabs is the TTS provider and sounddevice is available,
            # we stream audio sentence-by-sentence as the agent generates tokens
            # instead of waiting for the full response.
            use_streaming_tts = False
            _streaming_box_opened = False
            text_queue = None
            tts_thread = None
            stream_callback = None
            stop_event = None

            if self._voice_tts:
                try:
                    from tools.tts_tool import (
                        _load_tts_config as _load_tts_cfg,
                        _get_provider as _get_prov,
                        _import_elevenlabs,
                        _import_sounddevice,
                        stream_tts_to_speaker,
                    )
                    _tts_cfg = _load_tts_cfg()
                    if _get_prov(_tts_cfg) == "elevenlabs":
                        # Verify both ElevenLabs SDK and audio output are available
                        _import_elevenlabs()
                        _import_sounddevice()
                        use_streaming_tts = True
                except (ImportError, OSError):
                    pass
                except Exception:
                    pass

            if use_streaming_tts:
                text_queue = queue.Queue()
                stop_event = threading.Event()

                def display_callback(sentence: str):
                    """Called by TTS consumer when a sentence is ready to display + speak."""
                    nonlocal _streaming_box_opened
                    if not _streaming_box_opened:
                        _streaming_box_opened = True
                        w = self.console.width
                        label = " ⚕ Hermes "
                        fill = w - 2 - len(label)
                        _cprint(f"\n{_GOLD}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}")
                    _cprint(sentence.rstrip())

                tts_thread = threading.Thread(
                    target=stream_tts_to_speaker,
                    args=(text_queue, stop_event, self._voice_tts_done),
                    kwargs={"display_callback": display_callback},
                    daemon=True,
                )
                tts_thread.start()

                def stream_callback(delta: str):
                    if text_queue is not None:
                        text_queue.put(delta)

            # When voice mode is active, prepend a brief instruction so the
            # model responds concisely. The prefix is API-call-local only —
            # run_conversation persists the original clean user message.
            _voice_prefix = ""
            if self._voice_mode and isinstance(message, str):
                _voice_prefix = (
                    "[Voice input — respond concisely and conversationally, "
                    "2-3 sentences max. No code blocks or markdown.] "
                )

            def run_agent():
                nonlocal result
                agent_message = _voice_prefix + message if _voice_prefix else message
                # Prepend pending model switch note so the model knows about the switch
                _msn = getattr(self, '_pending_model_switch_note', None)
                if _msn:
                    agent_message = _msn + "\n\n" + agent_message
                    self._pending_model_switch_note = None
                try:
                    result = self.agent.run_conversation(
                        user_message=agent_message,
                        conversation_history=self.conversation_history[:-1],  # Exclude the message we just added
                        stream_callback=stream_callback,
                        task_id=self.session_id,
                        persist_user_message=message if _voice_prefix else None,
                    )
                except Exception as exc:
                    logging.error("run_conversation raised: %s", exc, exc_info=True)
                    _summary = getattr(self.agent, '_summarize_api_error', lambda e: str(e)[:300])(exc)
                    result = {
                        "final_response": f"Error: {_summary}",
                        "messages": [],
                        "api_calls": 0,
                        "completed": False,
                        "failed": True,
                        "error": _summary,
                    }

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
                            print("\n⚡ New message detected, interrupting...")
                            # Signal TTS to stop on interrupt
                            if stop_event is not None:
                                stop_event.set()
                            self.agent.interrupt(interrupt_msg)
                            # Debug: log to file (stdout may be devnull from redirect_stdout)
                            try:
                                _dbg = _hermes_home / "interrupt_debug.log"
                                with open(_dbg, "a") as _f:
                                    import time as _t
                                    _f.write(f"{_t.strftime('%H:%M:%S')} interrupt fired: msg={str(interrupt_msg)[:60]!r}, "
                                             f"children={len(self.agent._active_children)}, "
                                             f"parent._interrupt={self.agent._interrupt_requested}\n")
                                    for _ci, _ch in enumerate(self.agent._active_children):
                                        _f.write(f"  child[{_ci}]._interrupt={_ch._interrupt_requested}\n")
                            except Exception:
                                pass
                            break
                    except queue.Empty:
                        # Force prompt_toolkit to flush any pending stdout
                        # output from the agent thread.  Without this, the
                        # StdoutProxy buffer only flushes on renderer passes
                        # triggered by input events — on macOS this causes
                        # the CLI to appear frozen until the user types. (#1624)
                        self._invalidate(min_interval=0.15)
                else:
                    # Fallback for non-interactive mode (e.g., single-query)
                    agent_thread.join(0.1)

            agent_thread.join()  # Ensure agent thread completes

            # Proactively clean up async clients whose event loop is dead.
            # The agent thread may have created AsyncOpenAI clients bound
            # to a per-thread event loop; if that loop is now closed, those
            # clients' __del__ would crash prompt_toolkit's loop on GC.
            try:
                from agent.auxiliary_client import cleanup_stale_async_clients
                cleanup_stale_async_clients()
            except Exception:
                pass

            # Flush any remaining streamed text and close the box
            self._flush_stream()

            # Signal end-of-text to TTS consumer and wait for it to finish
            if use_streaming_tts and text_queue is not None:
                text_queue.put(None)  # sentinel
                if tts_thread is not None:
                    tts_thread.join(timeout=120)

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

            # Auto-generate session title after first exchange (non-blocking)
            if response and result and not result.get("failed") and not result.get("partial"):
                try:
                    from agent.title_generator import maybe_auto_title
                    maybe_auto_title(
                        self._session_db,
                        self.session_id,
                        message,
                        response,
                        self.conversation_history,
                    )
                except Exception:
                    pass

            # Handle failed or partial results (e.g., non-retryable errors, rate limits,
            # truncated output, invalid tool calls). Both "failed" and "partial" with
            # an empty final_response mean the agent couldn't produce a usable answer.
            if result and (result.get("failed") or result.get("partial")) and not response:
                error_detail = result.get("error", "Unknown error")
                response = f"Error: {error_detail}"
                # Stop continuous voice mode on persistent errors (e.g. 429 rate limit)
                # to avoid an infinite error → record → error loop
                if self._voice_continuous:
                    self._voice_continuous = False
                    _cprint(f"\n{_DIM}Continuous voice mode stopped due to error.{_RST}")

            # Handle interrupt - check if we were interrupted
            pending_message = None
            if result and result.get("interrupted"):
                pending_message = result.get("interrupt_message") or interrupt_msg
                # Add indicator that we were interrupted
                if response and pending_message:
                    response = response + "\n\n---\n_[Interrupted - processing new message]_"

            response_previewed = result.get("response_previewed", False) if result else False

            # Display reasoning (thinking) box if enabled and available.
            # Skip when streaming already showed reasoning live.  Use the
            # turn-persistent flag (_reasoning_shown_this_turn) instead of
            # _reasoning_stream_started — the latter gets reset during
            # intermediate turn boundaries (tool-calling loops), which caused
            # the reasoning box to re-render after the final response.
            _reasoning_already_shown = getattr(self, '_reasoning_shown_this_turn', False)
            if self.show_reasoning and result and not _reasoning_already_shown:
                reasoning = result.get("last_reasoning")
                if reasoning:
                    w = shutil.get_terminal_size().columns
                    r_label = " Reasoning "
                    r_fill = w - 2 - len(r_label)
                    r_top = f"{_DIM}┌─{r_label}{'─' * max(r_fill - 1, 0)}┐{_RST}"
                    r_bot = f"{_DIM}└{'─' * (w - 2)}┘{_RST}"
                    # Collapse long reasoning: show first 10 lines
                    lines = reasoning.strip().splitlines()
                    if len(lines) > 10:
                        display_reasoning = "\n".join(lines[:10])
                        display_reasoning += f"\n{_DIM}  ... ({len(lines) - 10} more lines){_RST}"
                    else:
                        display_reasoning = reasoning.strip()
                    _cprint(f"\n{r_top}\n{_DIM}{display_reasoning}{_RST}\n{r_bot}")

            if response and not response_previewed:
                # Use skin engine for label/color with fallback
                try:
                    from hermes_cli.skin_engine import get_active_skin
                    _skin = get_active_skin()
                    label = _skin.get_branding("response_label", "⚕ Hermes")
                    _resp_color = _skin.get_color("response_border", "#CD7F32")
                    _resp_text = _skin.get_color("banner_text", "#FFF8DC")
                except Exception:
                    label = "⚕ Hermes"
                    _resp_color = "#CD7F32"
                    _resp_text = "#FFF8DC"

                is_error_response = result and (result.get("failed") or result.get("partial"))
                already_streamed = self._stream_started and self._stream_box_opened and not is_error_response
                if use_streaming_tts and _streaming_box_opened and not is_error_response:
                    # Text was already printed sentence-by-sentence; just close the box
                    w = shutil.get_terminal_size().columns
                    _cprint(f"\n{_GOLD}╰{'─' * (w - 2)}╯{_RST}")
                elif already_streamed:
                    # Response was already streamed token-by-token with box framing;
                    # _flush_stream() already closed the box. Skip Rich Panel.
                    pass
                else:
                    _chat_console = ChatConsole()
                    _chat_console.print(Panel(
                        _rich_text_from_ansi(response),
                        title=f"[{_resp_color} bold]{label}[/]",
                        title_align="left",
                        border_style=_resp_color,
                        style=_resp_text,
                        box=rich_box.HORIZONTALS,
                        padding=(1, 2),
                    ))


            # Play terminal bell when agent finishes (if enabled).
            # Works over SSH — the bell propagates to the user's terminal.
            if self.bell_on_complete:
                sys.stdout.write("\a")
                sys.stdout.flush()

            # Speak response aloud if voice TTS is enabled
            # Skip batch TTS when streaming TTS already handled it
            if self._voice_tts and response and not use_streaming_tts:
                threading.Thread(
                    target=self._voice_speak_response,
                    args=(response,),
                    daemon=True,
                ).start()


            # Re-queue the interrupt message (and any that arrived while we were
            # processing the first) as the next prompt for process_loop.
            # Only reached when busy_input_mode == "interrupt" (the default).
            # In "queue" mode Enter routes directly to _pending_input so this
            # block is never hit.
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
                n = len(all_parts)
                preview = combined[:50] + ("..." if len(combined) > 50 else "")
                if n > 1:
                    print(f"\n⚡ Sending {n} messages after interrupt: '{preview}'")
                else:
                    print(f"\n⚡ Sending after interrupt: '{preview}'")
                self._pending_input.put(combined)
            
            return response
            
        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            # Ensure streaming TTS resources are cleaned up even on error.
            # Normal path sends the sentinel at line ~3568; this is a safety
            # net for exception paths that skip it.  Duplicate sentinels are
            # harmless — stream_tts_to_speaker exits on the first None.
            if text_queue is not None:
                try:
                    text_queue.put_nowait(None)
                except Exception:
                    pass
            if stop_event is not None:
                stop_event.set()
            if tts_thread is not None and tts_thread.is_alive():
                tts_thread.join(timeout=5)
    
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
            
            # Look up session title for resume-by-name hint
            session_title = None
            if self._session_db:
                try:
                    session_title = self._session_db.get_session_title(self.session_id)
                except Exception:
                    pass

            print("Resume this session with:")
            print(f"  hermes --resume {self.session_id}")
            if session_title:
                print(f"  hermes -c \"{session_title}\"")
            print()
            print(f"Session:        {self.session_id}")
            if session_title:
                print(f"Title:          {session_title}")
            print(f"Duration:       {duration_str}")
            print(f"Messages:       {msg_count} ({user_msgs} user, {tool_calls} tool calls)")
        else:
            try:
                from hermes_cli.skin_engine import get_active_goodbye
                goodbye = get_active_goodbye("Goodbye! ⚕")
            except Exception:
                goodbye = "Goodbye! ⚕"
            print(goodbye)

    def _get_tui_prompt_symbols(self) -> tuple[str, str]:
        """Return ``(normal_prompt, state_suffix)`` for the active skin.

        ``normal_prompt`` is the full ``branding.prompt_symbol``.
        ``state_suffix`` is what special states (sudo/secret/approval/agent)
        should render after their leading icon.

        When a profile is active (not "default"), the profile name is
        prepended to the prompt symbol: ``coder ❯`` instead of ``❯``.
        """
        try:
            from hermes_cli.skin_engine import get_active_prompt_symbol
            symbol = get_active_prompt_symbol("❯ ")
        except Exception:
            symbol = "❯ "

        symbol = (symbol or "❯ ").rstrip() + " "

        # Prepend profile name when not default
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile not in ("default", "custom"):
                symbol = f"{profile} {symbol}"
        except Exception:
            pass
        stripped = symbol.rstrip()
        if not stripped:
            return "❯ ", "❯ "

        parts = stripped.split()
        candidate = parts[-1] if parts else ""
        arrow_chars = ("❯", ">", "$", "#", "›", "»", "→")
        if any(ch in candidate for ch in arrow_chars):
            return symbol, candidate.rstrip() + " "

        # Icon-only custom prompts should still remain visible in special states.
        return symbol, symbol

    def _audio_level_bar(self) -> str:
        """Return a visual audio level indicator based on current RMS."""
        _LEVEL_BARS = " ▁▂▃▄▅▆▇"
        rec = getattr(self, "_voice_recorder", None)
        if rec is None:
            return ""
        rms = rec.current_rms
        # Normalize RMS (0-32767) to 0-7 index, with log-ish scaling
        # Typical speech RMS is 500-5000, we cap display at ~8000
        level = min(rms, 8000) * 7 // 8000
        return _LEVEL_BARS[level]

    def _get_tui_prompt_fragments(self):
        """Return the prompt_toolkit fragments for the current interactive state."""
        symbol, state_suffix = self._get_tui_prompt_symbols()
        if self._voice_recording:
            bar = self._audio_level_bar()
            return [("class:voice-recording", f"● {bar} {state_suffix}")]
        if self._voice_processing:
            return [("class:voice-processing", f"◉ {state_suffix}")]
        if self._sudo_state:
            return [("class:sudo-prompt", f"🔐 {state_suffix}")]
        if self._secret_state:
            return [("class:sudo-prompt", f"🔑 {state_suffix}")]
        if self._approval_state:
            return [("class:prompt-working", f"⚠ {state_suffix}")]
        if self._clarify_freetext:
            return [("class:clarify-selected", f"✎ {state_suffix}")]
        if self._clarify_state:
            return [("class:prompt-working", f"? {state_suffix}")]
        if self._command_running:
            return [("class:prompt-working", f"{self._command_spinner_frame()} {state_suffix}")]
        if self._agent_running:
            return [("class:prompt-working", f"⚕ {state_suffix}")]
        if self._voice_mode:
            return [("class:voice-prompt", f"🎤 {state_suffix}")]
        return [("class:prompt", symbol)]

    def _get_tui_prompt_text(self) -> str:
        """Return the visible prompt text for width calculations."""
        return "".join(text for _, text in self._get_tui_prompt_fragments())

    def _build_tui_style_dict(self) -> dict[str, str]:
        """Layer the active skin's prompt_toolkit colors over the base TUI style."""
        style_dict = dict(getattr(self, "_tui_style_base", {}) or {})
        try:
            from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides
            style_dict.update(get_prompt_toolkit_style_overrides())
        except Exception:
            pass
        return style_dict

    def _apply_tui_skin_style(self) -> bool:
        """Refresh prompt_toolkit styling for a running interactive TUI."""
        if not getattr(self, "_app", None) or not getattr(self, "_tui_style_base", None):
            return False
        self._app.style = PTStyle.from_dict(self._build_tui_style_dict())
        self._invalidate(min_interval=0.0)
        return True

    # --- Protected TUI extension hooks for wrapper CLIs ---

    def _get_extra_tui_widgets(self) -> list:
        """Return extra prompt_toolkit widgets to insert into the TUI layout.

        Wrapper CLIs can override this to inject widgets (e.g. a mini-player,
        overlay menu) into the layout without overriding ``run()``.  Widgets
        are inserted between the spacer and the status bar.
        """
        return []

    def _register_extra_tui_keybindings(self, kb, *, input_area) -> None:
        """Register extra keybindings on the TUI ``KeyBindings`` object.

        Wrapper CLIs can override this to add keybindings (e.g. transport
        controls, modal shortcuts) without overriding ``run()``.

        Parameters
        ----------
        kb : KeyBindings
            The active keybinding registry for the prompt_toolkit application.
        input_area : TextArea
            The main input widget, for wrappers that need to inspect or
            manipulate user input from a keybinding handler.
        """

    def _build_tui_layout_children(
        self,
        *,
        sudo_widget,
        secret_widget,
        approval_widget,
        clarify_widget,
        spinner_widget,
        spacer,
        status_bar,
        input_rule_top,
        image_bar,
        input_area,
        input_rule_bot,
        voice_status_bar,
        completions_menu,
    ) -> list:
        """Assemble the ordered list of children for the root ``HSplit``.

        Wrapper CLIs typically override ``_get_extra_tui_widgets`` instead of
        this method.  Override this only when you need full control over widget
        ordering.
        """
        return [
            Window(height=0),
            sudo_widget,
            secret_widget,
            approval_widget,
            clarify_widget,
            spinner_widget,
            spacer,
            *self._get_extra_tui_widgets(),
            status_bar,
            input_rule_top,
            image_bar,
            input_area,
            input_rule_bot,
            voice_status_bar,
            completions_menu,
        ]

    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        # Push the entire TUI to the bottom of the terminal so the banner,
        # responses, and prompt all appear pinned to the bottom — empty
        # space stays above, not below.  This prints enough blank lines to
        # scroll the cursor to the last row before any content is rendered.
        try:
            _term_lines = shutil.get_terminal_size().lines
            if _term_lines > 2:
                print("\n" * (_term_lines - 1), end="", flush=True)
        except Exception:
            pass

        self.show_banner()

        # One-line Honcho session indicator (TTY-only, not captured by agent).
        # Only show when the user explicitly configured Honcho for Hermes
        # (not auto-enabled from a stray HONCHO_API_KEY env var).
        # If resuming a session, load history and display it immediately
        # so the user has context before typing their first message.
        if self._resumed:
            if self._preload_resumed_session():
                self._display_resumed_history()

        try:
            from hermes_cli.skin_engine import get_active_skin
            _welcome_skin = get_active_skin()
            _welcome_text = _welcome_skin.get_branding("welcome", "Welcome to Hermes Agent! Type your message or /help for commands.")
            _welcome_color = _welcome_skin.get_color("banner_text", "#FFF8DC")
        except Exception:
            _welcome_text = "Welcome to Hermes Agent! Type your message or /help for commands."
            _welcome_color = "#FFF8DC"
        self.console.print(f"[{_welcome_color}]{_welcome_text}[/]")
        if self.preloaded_skills and not self._startup_skills_line_shown:
            skills_label = ", ".join(self.preloaded_skills)
            self.console.print(
                f"[bold {_accent_hex()}]Activated skills:[/] {skills_label}"
            )
            self._startup_skills_line_shown = True
        self.console.print()
        
        # State for async operation
        self._agent_running = False
        self._pending_input = queue.Queue()     # For normal input (commands + new queries)
        self._interrupt_queue = queue.Queue()   # For messages typed while agent is running
        self._should_exit = False
        self._last_ctrl_c_time = 0  # Track double Ctrl+C for force exit

        # Give plugin manager a CLI reference so plugins can inject messages
        from hermes_cli.plugins import get_plugin_manager
        get_plugin_manager()._cli_ref = self

        # Config file watcher — detect mcp_servers changes and auto-reload
        from hermes_cli.config import get_config_path as _get_config_path
        _cfg_path = _get_config_path()
        self._config_mtime: float = _cfg_path.stat().st_mtime if _cfg_path.exists() else 0.0
        self._config_mcp_servers: dict = self.config.get("mcp_servers") or {}
        self._last_config_check: float = 0.0  # monotonic time of last check

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
        self._approval_lock = threading.Lock()  # serialize concurrent approval prompts (delegation race fix)

        # Slash command loading state
        self._command_running = False
        self._command_status = ""

        # Secure secret capture state for skill setup
        self._secret_state = None       # dict with var_name, prompt, metadata, response_queue
        self._secret_deadline = 0

        # Clipboard image attachments (paste images into the CLI)
        self._attached_images: list[Path] = []
        self._image_counter = 0

        # Voice mode state (protected by _voice_lock for cross-thread access)
        self._voice_lock = threading.Lock()
        self._voice_mode = False        # Whether voice mode is enabled
        self._voice_tts = False         # Whether TTS output is enabled
        self._voice_recorder = None     # AudioRecorder instance (lazy init)
        self._voice_recording = False   # Whether currently recording
        self._voice_processing = False  # Whether STT is in progress
        self._voice_continuous = False  # Whether to auto-restart after agent responds
        self._voice_tts_done = threading.Event()  # Signals TTS playback finished
        self._voice_tts_done.set()  # Initially "done" (no TTS pending)

        # Register callbacks so terminal_tool prompts route through our UI
        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)
        set_secret_capture_callback(self._secret_capture_callback)

        # Ensure tirith security scanner is available (downloads if needed).
        # Warn the user if tirith is enabled in config but not available,
        # so they know command security scanning is degraded.
        try:
            from tools.tirith_security import ensure_installed
            tirith_path = ensure_installed(log_failures=False)
            if tirith_path is None:
                security_cfg = self.config.get("security", {}) or {}
                tirith_enabled = security_cfg.get("tirith_enabled", True)
                if tirith_enabled:
                    _cprint(f"  {_DIM}⚠ tirith security scanner enabled but not available "
                            f"— command scanning will use pattern matching only{_RST}")
        except Exception:
            pass  # Non-fatal — fail-open at scan time if unavailable
        
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

            # --- Secret prompt: submit the typed secret ---
            if self._secret_state:
                text = event.app.current_buffer.text
                self._submit_secret_response(text)
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # --- Approval selection: confirm the highlighted choice ---
            if self._approval_state:
                self._handle_approval_selection()
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
                event.app.invalidate()
                # Bundle text + images as a tuple when images are present
                payload = (text, images) if images else text
                if self._agent_running and not (text and _looks_like_slash_command(text)):
                    if self.busy_input_mode == "queue":
                        # Queue for the next turn instead of interrupting
                        self._pending_input.put(payload)
                        preview = text if text else f"[{len(images)} image{'s' if len(images) != 1 else ''} attached]"
                        _cprint(f"  Queued for the next turn: {preview[:80]}{'...' if len(preview) > 80 else ''}")
                    else:
                        self._interrupt_queue.put(payload)
                        # Debug: log to file when message enters interrupt queue
                        try:
                            _dbg = _hermes_home / "interrupt_debug.log"
                            with open(_dbg, "a") as _f:
                                import time as _t
                                _f.write(f"{_t.strftime('%H:%M:%S')} ENTER: queued interrupt msg={str(payload)[:60]!r}, "
                                         f"agent_running={self._agent_running}\n")
                        except Exception:
                            pass
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

        @kb.add('tab', eager=True)
        def handle_tab(event):
            """Tab: accept completion, auto-suggestion, or start completions.

            Priority:
            1. Completion menu open → accept selected completion
            2. Ghost text suggestion available → accept auto-suggestion
            3. Otherwise → start completion menu

            After accepting a provider like 'anthropic:', the completion menu
            closes and complete_while_typing doesn't fire (no keystroke).
            This binding re-triggers completions so stage-2 models appear
            immediately.
            """
            buf = event.current_buffer
            if buf.complete_state:
                # Completion menu is open — accept the selection
                completion = buf.complete_state.current_completion
                if completion is None:
                    # Menu open but nothing selected — select first then grab it
                    buf.go_to_completion(0)
                    completion = buf.complete_state and buf.complete_state.current_completion
                if completion is None:
                    return
                # Accept the selected completion
                buf.apply_completion(completion)
            elif buf.suggestion and buf.suggestion.text:
                # No completion menu, but there's a ghost text auto-suggestion — accept it
                buf.insert_text(buf.suggestion.text)
            else:
                # No menu and no suggestion — start completions from scratch
                buf.start_completion()

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
        # The TextArea is multiline, so by default up/down only move the cursor.
        # Buffer.auto_up/auto_down handle both: cursor movement when multi-line,
        # history browsing when on the first/last line (or single-line input).
        _normal_input = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._sudo_state and not self._secret_state
        )

        @kb.add('up', filter=_normal_input)
        def history_up(event):
            """Up arrow: browse history when on first line, else move cursor up."""
            event.app.current_buffer.auto_up(count=event.arg)

        @kb.add('down', filter=_normal_input)
        def history_down(event):
            """Down arrow: browse history when on last line, else move cursor down."""
            event.app.current_buffer.auto_down(count=event.arg)

        @kb.add('c-c')
        def handle_ctrl_c(event):
            """Handle Ctrl+C - cancel interactive prompts, interrupt agent, or exit.
            
            Priority:
            0. Cancel active voice recording
            1. Cancel active sudo/approval/clarify prompt
            2. Interrupt the running agent (first press)
            3. Force exit (second press within 2s, or when idle)
            """
            import time as _time
            now = _time.time()

            # Cancel active voice recording.
            # Run cancel() in a background thread to prevent blocking the
            # event loop if AudioRecorder._lock or CoreAudio takes time.
            _should_cancel_voice = False
            _recorder_ref = None
            with cli_ref._voice_lock:
                if cli_ref._voice_recording and cli_ref._voice_recorder:
                    _recorder_ref = cli_ref._voice_recorder
                    cli_ref._voice_recording = False
                    cli_ref._voice_continuous = False
                    _should_cancel_voice = True
            if _should_cancel_voice:
                _cprint(f"\n{_DIM}Recording cancelled.{_RST}")
                threading.Thread(
                    target=_recorder_ref.cancel, daemon=True
                ).start()
                event.app.invalidate()
                return

            # Cancel sudo prompt
            if self._sudo_state:
                self._sudo_state["response_queue"].put("")
                self._sudo_state = None
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # Cancel secret prompt
            if self._secret_state:
                self._cancel_secret_capture()
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
                    event.app.invalidate()
                else:
                    self._should_exit = True
                    event.app.exit()
        
        @kb.add('c-d')
        def handle_ctrl_d(event):
            """Handle Ctrl+D - exit."""
            self._should_exit = True
            event.app.exit()

        @kb.add('c-z')
        def handle_ctrl_z(event):
            """Handle Ctrl+Z - suspend process to background (Unix only)."""
            import sys
            if sys.platform == 'win32':
                _cprint(f"\n{_DIM}Suspend (Ctrl+Z) is not supported on Windows.{_RST}")
                event.app.invalidate()
                return
            import os, signal as _sig
            from prompt_toolkit.application import run_in_terminal
            from hermes_cli.skin_engine import get_active_skin
            agent_name = get_active_skin().get_branding("agent_name", "Hermes Agent")
            msg = f"\n{agent_name} has been suspended. Run `fg` to bring {agent_name} back."
            def _suspend():
                os.write(1, msg.encode())
                os.kill(0, _sig.SIGTSTP)
            run_in_terminal(_suspend)

        # Voice push-to-talk key: configurable via config.yaml (voice.record_key)
        # Default: Ctrl+B (avoids conflict with Ctrl+R readline reverse-search)
        # Config uses "ctrl+b" format; prompt_toolkit expects "c-b" format.
        try:
            from hermes_cli.config import load_config
            _raw_key = load_config().get("voice", {}).get("record_key", "ctrl+b")
            _voice_key = _raw_key.lower().replace("ctrl+", "c-").replace("alt+", "a-")
        except Exception:
            _voice_key = "c-b"

        @kb.add(_voice_key)
        def handle_voice_record(event):
            """Toggle voice recording when voice mode is active.

            IMPORTANT: This handler runs in prompt_toolkit's event-loop thread.
            Any blocking call here (locks, sd.wait, disk I/O) freezes the
            entire UI.  All heavy work is dispatched to daemon threads.
            """
            if not cli_ref._voice_mode:
                return
            # Always allow STOPPING a recording (even when agent is running)
            if cli_ref._voice_recording:
                # Manual stop via push-to-talk key: stop continuous mode
                with cli_ref._voice_lock:
                    cli_ref._voice_continuous = False
                # Flag clearing is handled atomically inside _voice_stop_and_transcribe
                event.app.invalidate()
                threading.Thread(
                    target=cli_ref._voice_stop_and_transcribe,
                    daemon=True,
                ).start()
            else:
                # Guard: don't START recording during agent run or interactive prompts
                if cli_ref._agent_running:
                    return
                if cli_ref._clarify_state or cli_ref._sudo_state or cli_ref._approval_state:
                    return
                # Guard: don't start while a previous stop/transcribe cycle is
                # still running — recorder.stop() holds AudioRecorder._lock and
                # start() would block the event-loop thread waiting for it.
                if cli_ref._voice_processing:
                    return

                # Interrupt TTS if playing, so user can start talking.
                # stop_playback() is fast (just terminates a subprocess).
                if not cli_ref._voice_tts_done.is_set():
                    try:
                        from tools.voice_mode import stop_playback
                        stop_playback()
                        cli_ref._voice_tts_done.set()
                    except Exception:
                        pass

                with cli_ref._voice_lock:
                    cli_ref._voice_continuous = True

                # Dispatch to a daemon thread so play_beep(sd.wait),
                # AudioRecorder.start(lock acquire), and config I/O
                # never block the prompt_toolkit event loop.
                def _start_recording():
                    try:
                        cli_ref._voice_start_recording()
                        if hasattr(cli_ref, '_app') and cli_ref._app:
                            cli_ref._app.invalidate()
                    except Exception as e:
                        _cprint(f"\n{_DIM}Voice recording failed: {e}{_RST}")

                threading.Thread(target=_start_recording, daemon=True).start()
                event.app.invalidate()
        from prompt_toolkit.keys import Keys

        @kb.add(Keys.BracketedPaste, eager=True)
        def handle_paste(event):
            """Handle terminal paste — detect clipboard images.

            When the terminal supports bracketed paste, Ctrl+V / Cmd+V
            triggers this with the pasted text.  We also check the
            clipboard for an image on every paste event.

            Large pastes (5+ lines) are collapsed to a file reference
            placeholder while preserving any existing user text in the
            buffer.
            """
            pasted_text = event.data or ""
            # Normalise line endings — Windows \r\n and old Mac \r both become \n
            # so the 5-line collapse threshold and display are consistent.
            pasted_text = pasted_text.replace('\r\n', '\n').replace('\r', '\n')
            if self._try_attach_clipboard_image():
                event.app.invalidate()
            if pasted_text:
                line_count = pasted_text.count('\n')
                buf = event.current_buffer
                if line_count >= 5 and not buf.text.strip().startswith('/'):
                    _paste_counter[0] += 1
                    paste_dir = _hermes_home / "pastes"
                    paste_dir.mkdir(parents=True, exist_ok=True)
                    paste_file = paste_dir / f"paste_{_paste_counter[0]}_{datetime.now().strftime('%H%M%S')}.txt"
                    paste_file.write_text(pasted_text, encoding="utf-8")
                    placeholder = f"[Pasted text #{_paste_counter[0]}: {line_count + 1} lines \u2192 {paste_file}]"
                    prefix = ""
                    if buf.cursor_position > 0 and buf.text[buf.cursor_position - 1] != '\n':
                        prefix = "\n"
                    _paste_just_collapsed[0] = True
                    buf.insert_text(prefix + placeholder)
                else:
                    buf.insert_text(pasted_text)

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

        def get_prompt():
            return cli_ref._get_tui_prompt_fragments()

        # Create the input area with multiline (shift+enter), autocomplete, and paste handling
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory


        _completer = SlashCommandCompleter(
            skill_commands_provider=lambda: _skill_commands,
        )
        input_area = TextArea(
            height=Dimension(min=1, max=8, preferred=1),
            prompt=get_prompt,
            style='class:input-area',
            multiline=True,
            wrap_lines=True,
            read_only=Condition(lambda: bool(cli_ref._command_running)),
            history=FileHistory(str(self._history_file)),
            completer=_completer,
            complete_while_typing=True,
            auto_suggest=SlashCommandAutoSuggest(
                history_suggest=AutoSuggestFromHistory(),
                completer=_completer,
            ),
        )

        # Dynamic height: accounts for both explicit newlines AND visual
        # wrapping of long lines so the input area always fits its content.
        def _input_height():
            try:
                doc = input_area.buffer.document
                prompt_width = max(2, len(self._get_tui_prompt_text()))
                available_width = shutil.get_terminal_size().columns - prompt_width
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
        _prev_newline_count = [0]
        _paste_just_collapsed = [False]

        def _on_text_changed(buf):
            """Detect large pastes and collapse them to a file reference.

            When bracketed paste is available, handle_paste collapses
            large pastes directly.  This handler is a fallback for
            terminals without bracketed paste support.

            Two heuristics (either triggers collapse):
            1. Many characters added at once (chars_added > 1) — works
               when the terminal delivers the paste in one event-loop tick.
            2. Newline count jumped by 4+ in a single text-change event —
               catches terminals that feed characters individually but
               still batch newlines.  Alt+Enter only adds 1 newline per
               event so it never triggers this.
            """
            text = buf.text
            chars_added = len(text) - _prev_text_len[0]
            _prev_text_len[0] = len(text)
            if _paste_just_collapsed[0]:
                _paste_just_collapsed[0] = False
                _prev_newline_count[0] = text.count('\n')
                return
            line_count = text.count('\n')
            newlines_added = line_count - _prev_newline_count[0]
            _prev_newline_count[0] = line_count
            is_paste = chars_added > 1 or newlines_added >= 4
            if line_count >= 5 and is_paste and not text.startswith('/'):
                _paste_counter[0] += 1
                # Save to temp file
                paste_dir = _hermes_home / "pastes"
                paste_dir.mkdir(parents=True, exist_ok=True)
                paste_file = paste_dir / f"paste_{_paste_counter[0]}_{datetime.now().strftime('%H%M%S')}.txt"
                paste_file.write_text(text, encoding="utf-8")
                # Replace buffer with compact reference
                _paste_just_collapsed[0] = True
                buf.text = f"[Pasted text #{_paste_counter[0]}: {line_count + 1} lines \u2192 {paste_file}]"
                buf.cursor_position = len(buf.text)

        input_area.buffer.on_text_changed += _on_text_changed

        # --- Input processors for password masking and inline placeholder ---

        # Mask input with '*' when the sudo password prompt is active
        input_area.control.input_processors.append(
            ConditionalProcessor(
                PasswordProcessor(),
                filter=Condition(
                    lambda: bool(cli_ref._sudo_state) or bool(cli_ref._secret_state)
                ),
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
            if cli_ref._voice_recording:
                return "recording... Ctrl+B to stop, Ctrl+C to cancel"
            if cli_ref._voice_processing:
                return "transcribing..."
            if cli_ref._sudo_state:
                return "type password (hidden), Enter to skip"
            if cli_ref._secret_state:
                return "type secret (hidden), Enter to skip"
            if cli_ref._approval_state:
                return ""
            if cli_ref._clarify_freetext:
                return "type your answer here and press Enter"
            if cli_ref._clarify_state:
                return ""
            if cli_ref._command_running:
                frame = cli_ref._command_spinner_frame()
                status = cli_ref._command_status or "Processing command..."
                return f"{frame} {status}"
            if cli_ref._agent_running:
                return "type a message + Enter to interrupt, Ctrl+C to cancel"
            if cli_ref._voice_mode:
                return "type or Ctrl+B to record"
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

            if cli_ref._secret_state:
                remaining = max(0, int(cli_ref._secret_deadline - _time.monotonic()))
                return [
                    ('class:hint', '  secret hidden · Enter to skip'),
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

            if cli_ref._command_running:
                frame = cli_ref._command_spinner_frame()
                return [
                    ('class:hint', f'  {frame} command in progress · input temporarily disabled'),
                ]

            return []

        def get_hint_height():
            if cli_ref._sudo_state or cli_ref._secret_state or cli_ref._approval_state or cli_ref._clarify_state or cli_ref._command_running:
                return 1
            # Keep a 1-line spacer while agent runs so output doesn't push
            # right up against the top rule of the input area
            return 1 if cli_ref._agent_running else 0

        def get_spinner_text():
            txt = cli_ref._spinner_text
            if not txt:
                return []
            return [('class:hint', f'  {txt}')]

        def get_spinner_height():
            return 1 if cli_ref._spinner_text else 0

        spinner_widget = Window(
            content=FormattedTextControl(get_spinner_text),
            height=get_spinner_height,
        )

        spacer = Window(
            content=FormattedTextControl(get_hint_text),
            height=get_hint_height,
        )

        # --- Clarify tool: dynamic display widget for questions + choices ---

        def _panel_box_width(title: str, content_lines: list[str], min_width: int = 46, max_width: int = 76) -> int:
            """Choose a stable panel width wide enough for the title and content."""
            term_cols = shutil.get_terminal_size((100, 20)).columns
            longest = max([len(title)] + [len(line) for line in content_lines] + [min_width - 4])
            inner = min(max(longest + 4, min_width - 2), max_width - 2, max(24, term_cols - 6))
            return inner + 2  # account for the single leading/trailing spaces inside borders

        def _wrap_panel_text(text: str, width: int, subsequent_indent: str = "") -> list[str]:
            wrapped = textwrap.wrap(
                text,
                width=max(8, width),
                break_long_words=False,
                break_on_hyphens=False,
                subsequent_indent=subsequent_indent,
            )
            return wrapped or [""]

        def _append_panel_line(lines, border_style: str, content_style: str, text: str, box_width: int) -> None:
            inner_width = max(0, box_width - 2)
            lines.append((border_style, "│ "))
            lines.append((content_style, text.ljust(inner_width)))
            lines.append((border_style, " │\n"))

        def _append_blank_panel_line(lines, border_style: str, box_width: int) -> None:
            lines.append((border_style, "│" + (" " * box_width) + "│\n"))

        def _get_clarify_display():
            """Build styled text for the clarify question/choices panel."""
            state = cli_ref._clarify_state
            if not state:
                return []

            question = state["question"]
            choices = state.get("choices") or []
            selected = state.get("selected", 0)
            preview_lines = _wrap_panel_text(question, 60)
            for i, choice in enumerate(choices):
                prefix = "❯ " if i == selected and not cli_ref._clarify_freetext else "  "
                preview_lines.extend(_wrap_panel_text(f"{prefix}{choice}", 60, subsequent_indent="  "))
            other_label = (
                "❯ Other (type below)" if cli_ref._clarify_freetext
                else "❯ Other (type your answer)" if selected == len(choices)
                else "  Other (type your answer)"
            )
            preview_lines.extend(_wrap_panel_text(other_label, 60, subsequent_indent="  "))
            box_width = _panel_box_width("Hermes needs your input", preview_lines)
            inner_text_width = max(8, box_width - 2)

            lines = []
            # Box top border
            lines.append(('class:clarify-border', '╭─ '))
            lines.append(('class:clarify-title', 'Hermes needs your input'))
            lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len("Hermes needs your input") - 3)) + '╮\n'))
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)

            # Question text
            for wrapped in _wrap_panel_text(question, inner_text_width):
                _append_panel_line(lines, 'class:clarify-border', 'class:clarify-question', wrapped, box_width)
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)

            if cli_ref._clarify_freetext and not choices:
                guidance = "Type your answer in the prompt below, then press Enter."
                for wrapped in _wrap_panel_text(guidance, inner_text_width):
                    _append_panel_line(lines, 'class:clarify-border', 'class:clarify-choice', wrapped, box_width)
                _append_blank_panel_line(lines, 'class:clarify-border', box_width)

            if choices:
                # Multiple-choice mode: show selectable options
                for i, choice in enumerate(choices):
                    style = 'class:clarify-selected' if i == selected and not cli_ref._clarify_freetext else 'class:clarify-choice'
                    prefix = '❯ ' if i == selected and not cli_ref._clarify_freetext else '  '
                    wrapped_lines = _wrap_panel_text(f"{prefix}{choice}", inner_text_width, subsequent_indent="  ")
                    for wrapped in wrapped_lines:
                        _append_panel_line(lines, 'class:clarify-border', style, wrapped, box_width)

                # "Other" option (5th line, only shown when choices exist)
                other_idx = len(choices)
                if selected == other_idx and not cli_ref._clarify_freetext:
                    other_style = 'class:clarify-selected'
                    other_label = '❯ Other (type your answer)'
                elif cli_ref._clarify_freetext:
                    other_style = 'class:clarify-active-other'
                    other_label = '❯ Other (type below)'
                else:
                    other_style = 'class:clarify-choice'
                    other_label = '  Other (type your answer)'
                for wrapped in _wrap_panel_text(other_label, inner_text_width, subsequent_indent="  "):
                    _append_panel_line(lines, 'class:clarify-border', other_style, wrapped, box_width)

            _append_blank_panel_line(lines, 'class:clarify-border', box_width)
            lines.append(('class:clarify-border', '╰' + ('─' * box_width) + '╯\n'))
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
            title = '🔐 Sudo Password Required'
            body = 'Enter password below (hidden), or press Enter to skip'
            box_width = _panel_box_width(title, [body])
            inner = max(0, box_width - 2)
            lines = []
            lines.append(('class:sudo-border', '╭─ '))
            lines.append(('class:sudo-title', title))
            lines.append(('class:sudo-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
            _append_blank_panel_line(lines, 'class:sudo-border', box_width)
            _append_panel_line(lines, 'class:sudo-border', 'class:sudo-text', body, box_width)
            _append_blank_panel_line(lines, 'class:sudo-border', box_width)
            lines.append(('class:sudo-border', '╰' + ('─' * box_width) + '╯\n'))
            return lines

        sudo_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_sudo_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._sudo_state is not None),
        )

        def _get_secret_display():
            state = cli_ref._secret_state
            if not state:
                return []

            title = '🔑 Skill Setup Required'
            prompt = state.get("prompt") or f"Enter value for {state.get('var_name', 'secret')}"
            metadata = state.get("metadata") or {}
            help_text = metadata.get("help")
            body = 'Enter secret below (hidden), or press Enter to skip'
            content_lines = [prompt, body]
            if help_text:
                content_lines.insert(1, str(help_text))
            box_width = _panel_box_width(title, content_lines)
            lines = []
            lines.append(('class:sudo-border', '╭─ '))
            lines.append(('class:sudo-title', title))
            lines.append(('class:sudo-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
            _append_blank_panel_line(lines, 'class:sudo-border', box_width)
            _append_panel_line(lines, 'class:sudo-border', 'class:sudo-text', prompt, box_width)
            if help_text:
                _append_panel_line(lines, 'class:sudo-border', 'class:sudo-text', str(help_text), box_width)
            _append_blank_panel_line(lines, 'class:sudo-border', box_width)
            _append_panel_line(lines, 'class:sudo-border', 'class:sudo-text', body, box_width)
            _append_blank_panel_line(lines, 'class:sudo-border', box_width)
            lines.append(('class:sudo-border', '╰' + ('─' * box_width) + '╯\n'))
            return lines

        secret_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_secret_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._secret_state is not None),
        )

        # --- Dangerous command approval: display widget ---

        def _get_approval_display():
            return cli_ref._get_approval_display_fragments()

        approval_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_approval_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._approval_state is not None),
        )

        # Horizontal rules above and below the input (bronze, 1 line each).
        # The bottom rule moves down as the TextArea grows with newlines.
        # Using char='─' instead of hardcoded repetition so the rule
        # always spans the full terminal width on any screen size.
        input_rule_top = Window(
            char='─',
            height=1,
            style='class:input-rule',
        )
        input_rule_bot = Window(
            char='─',
            height=1,
            style='class:input-rule',
        )

        # Image attachment indicator — shows badges like [📎 Image #1] above input
        cli_ref = self

        def _get_image_bar():
            if not cli_ref._attached_images:
                return []
            base = cli_ref._image_counter - len(cli_ref._attached_images) + 1
            badges = " ".join(
                f"[📎 Image #{base + i}]"
                for i in range(len(cli_ref._attached_images))
            )
            return [("class:image-badge", f" {badges} ")]

        image_bar = Window(
            content=FormattedTextControl(_get_image_bar),
            height=Condition(lambda: bool(cli_ref._attached_images)),
        )

        # Persistent voice mode status bar (visible only when voice mode is on)
        def _get_voice_status():
            if cli_ref._voice_recording:
                return [('class:voice-status-recording', ' ● REC  Ctrl+B to stop ')]
            if cli_ref._voice_processing:
                return [('class:voice-status', ' ◉ Transcribing... ')]
            tts = " | TTS on" if cli_ref._voice_tts else ""
            cont = " | Continuous" if cli_ref._voice_continuous else ""
            return [('class:voice-status', f' 🎤 Voice mode{tts}{cont}  —  Ctrl+B to record ')]

        voice_status_bar = ConditionalContainer(
            Window(
                FormattedTextControl(_get_voice_status),
                height=1,
            ),
            filter=Condition(lambda: cli_ref._voice_mode),
        )

        status_bar = ConditionalContainer(
            Window(
                content=FormattedTextControl(lambda: cli_ref._get_status_bar_fragments()),
                height=1,
                # Prevent fragments that overflow the terminal width from
                # wrapping onto a second line, which causes the status bar to
                # appear duplicated (one full + one partial row) during long
                # sessions, especially on SSH where shutil.get_terminal_size
                # may return stale values.  _get_status_bar_fragments now reads
                # width from prompt_toolkit's own output object, so fragments
                # will always fit; wrap_lines=False is the belt-and-suspenders
                # guard against any future width mismatch.
                wrap_lines=False,
            ),
            filter=Condition(lambda: cli_ref._status_bar_visible),
        )

        # Allow wrapper CLIs to register extra keybindings.
        self._register_extra_tui_keybindings(kb, input_area=input_area)

        # Layout: interactive prompt widgets + ruled input at bottom.
        # The sudo, approval, and clarify widgets appear above the input when
        # the corresponding interactive prompt is active.
        completions_menu = CompletionsMenu(max_height=12, scroll_offset=1)

        layout = Layout(
            HSplit(
                self._build_tui_layout_children(
                    sudo_widget=sudo_widget,
                    secret_widget=secret_widget,
                    approval_widget=approval_widget,
                    clarify_widget=clarify_widget,
                    spinner_widget=spinner_widget,
                    spacer=spacer,
                    status_bar=status_bar,
                    input_rule_top=input_rule_top,
                    image_bar=image_bar,
                    input_area=input_area,
                    input_rule_bot=input_rule_bot,
                    voice_status_bar=voice_status_bar,
                    completions_menu=completions_menu,
                )
            )
        )
        
        # Style for the application
        self._tui_style_base = {
            'input-area': '#FFF8DC',
            'placeholder': '#555555 italic',
            'prompt': '#FFF8DC',
            'prompt-working': '#888888 italic',
            'hint': '#555555 italic',
            'status-bar': 'bg:#1a1a2e #C0C0C0',
            'status-bar-strong': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-dim': 'bg:#1a1a2e #8B8682',
            'status-bar-good': 'bg:#1a1a2e #8FBC8F bold',
            'status-bar-warn': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-bad': 'bg:#1a1a2e #FF8C00 bold',
            'status-bar-critical': 'bg:#1a1a2e #FF6B6B bold',
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
            # Voice mode
            'voice-prompt': '#87CEEB',
            'voice-recording': '#FF4444 bold',
            'voice-processing': '#FFA500 italic',
            'voice-status': 'bg:#1a1a2e #87CEEB',
            'voice-status-recording': 'bg:#1a1a2e #FF4444 bold',
        }
        style = PTStyle.from_dict(self._build_tui_style_dict())
        
        # Create the application
        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
            **({'cursor': _STEADY_CURSOR} if _STEADY_CURSOR is not None else {}),
        )
        self._app = app  # Store reference for clarify_callback

        # ── Fix ghost status-bar lines on terminal resize ──────────────
        # When the terminal shrinks (e.g. un-maximize), the emulator reflows
        # the previously-rendered full-width rows (status bar, input rules)
        # into multiple narrower rows.  prompt_toolkit's _on_resize handler
        # only cursor_up()s by the stored layout height, missing the extra
        # rows created by reflow — leaving ghost duplicates visible.
        #
        # Fix: before the standard erase, inflate _cursor_pos.y so the
        # cursor moves up far enough to cover the reflowed ghost content.
        _original_on_resize = app._on_resize

        def _resize_clear_ghosts():
            from prompt_toolkit.data_structures import Point as _Pt
            renderer = app.renderer
            try:
                old_size = renderer._last_size
                new_size = renderer.output.get_size()
                if (
                    old_size
                    and new_size.columns < old_size.columns
                    and new_size.columns > 0
                ):
                    reflow_factor = (
                        (old_size.columns + new_size.columns - 1)
                        // new_size.columns
                    )
                    last_h = (
                        renderer._last_screen.height
                        if renderer._last_screen
                        else 0
                    )
                    extra = last_h * (reflow_factor - 1)
                    if extra > 0:
                        renderer._cursor_pos = _Pt(
                            x=renderer._cursor_pos.x,
                            y=renderer._cursor_pos.y + extra,
                        )
            except Exception:
                pass  # never break resize handling
            _original_on_resize()

        app._on_resize = _resize_clear_ghosts

        def spinner_loop():
            import time as _time

            last_idle_refresh = 0.0
            while not self._should_exit:
                if not self._app:
                    _time.sleep(0.1)
                    continue
                if self._command_running:
                    self._invalidate(min_interval=0.1)
                    _time.sleep(0.1)
                else:
                    now = _time.monotonic()
                    if now - last_idle_refresh >= 1.0:
                        last_idle_refresh = now
                        self._invalidate(min_interval=1.0)
                    _time.sleep(0.2)

        spinner_thread = threading.Thread(target=spinner_loop, daemon=True)
        spinner_thread.start()
        
        # Background thread to process inputs and run agent
        def process_loop():
            while not self._should_exit:
                try:
                    # Check for pending input with timeout
                    try:
                        user_input = self._pending_input.get(timeout=0.1)
                    except queue.Empty:
                        # Periodic config watcher — auto-reload MCP on mcp_servers change
                        if not self._agent_running:
                            self._check_config_mcp_changes()
                        continue
                    
                    if not user_input:
                        continue

                    # Unpack image payload: (text, [Path, ...]) or plain str
                    submit_images = []
                    if isinstance(user_input, tuple):
                        user_input, submit_images = user_input
                    
                    # Check for commands — but detect dragged/pasted file paths first.
                    # See _detect_file_drop() for details.
                    _file_drop = _detect_file_drop(user_input) if isinstance(user_input, str) else None
                    if _file_drop:
                        _drop_path = _file_drop["path"]
                        _remainder = _file_drop["remainder"]
                        if _file_drop["is_image"]:
                            submit_images.append(_drop_path)
                            user_input = _remainder or f"[User attached image: {_drop_path.name}]"
                            _cprint(f"  📎 Auto-attached image: {_drop_path.name}")
                        else:
                            _cprint(f"  📄 Detected file: {_drop_path.name}")
                            user_input = (
                                f"[User attached file: {_drop_path}]"
                                + (f"\n{_remainder}" if _remainder else "")
                            )

                    if not _file_drop and isinstance(user_input, str) and _looks_like_slash_command(user_input):
                        _cprint(f"\n⚙️  {user_input}")
                        if not self.process_command(user_input):
                            self._should_exit = True
                            # Schedule app exit
                            if app.is_running:
                                app.exit()
                        continue
                    
                    # Expand paste references back to full content
                    import re as _re
                    _paste_ref_re = _re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')
                    paste_refs = list(_paste_ref_re.finditer(user_input)) if isinstance(user_input, str) else []
                    if paste_refs:
                        def _expand_ref(m):
                            p = Path(m.group(1))
                            return p.read_text(encoding="utf-8") if p.exists() else m.group(0)
                        expanded = _paste_ref_re.sub(_expand_ref, user_input)
                        total_lines = expanded.count('\n') + 1
                        n_pastes = len(paste_refs)
                        _user_bar = f"[{_accent_hex()}]{'─' * 40}[/]"
                        print()
                        ChatConsole().print(_user_bar)
                        # Show any surrounding user text alongside the paste summary
                        split_parts = _paste_ref_re.split(user_input)
                        visible_user_text = " ".join(
                            split_parts[i].strip() for i in range(0, len(split_parts), 2) if split_parts[i].strip()
                        )
                        if visible_user_text:
                            ChatConsole().print(
                                f"[bold {_accent_hex()}]\u25cf[/] [bold]{_escape(visible_user_text)}[/] "
                                f"[dim]({n_pastes} pasted block{'s' if n_pastes > 1 else ''}, {total_lines} lines total)[/]"
                            )
                        else:
                            ChatConsole().print(
                                f"[bold {_accent_hex()}]\u25cf[/] [bold]{_escape(f'[Pasted text: {total_lines} lines]')}[/]"
                            )
                        user_input = expanded
                    else:
                        _user_bar = f"[{_accent_hex()}]{'─' * 40}[/]"
                        if '\n' in user_input:
                            first_line = user_input.split('\n')[0]
                            line_count = user_input.count('\n') + 1
                            print()
                            ChatConsole().print(_user_bar)
                            ChatConsole().print(
                                f"[bold {_accent_hex()}]●[/] [bold]{_escape(first_line)}[/] "
                                f"[dim](+{line_count - 1} lines)[/]"
                            )
                        else:
                            print()
                            ChatConsole().print(_user_bar)
                            ChatConsole().print(f"[bold {_accent_hex()}]●[/] [bold]{_escape(user_input)}[/]")
                    
                    # Show image attachment count
                    if submit_images:
                        n = len(submit_images)
                        _cprint(f"  {_DIM}📎 {n} image{'s' if n > 1 else ''} attached{_RST}")

                    # Regular chat - run agent
                    self._agent_running = True
                    app.invalidate()  # Refresh status line

                    try:
                        self.chat(user_input, images=submit_images or None)
                    finally:
                        self._agent_running = False
                        self._spinner_text = ""

                        app.invalidate()  # Refresh status line

                        # Continuous voice: auto-restart recording after agent responds.
                        # Dispatch to a daemon thread so play_beep (sd.wait) and
                        # AudioRecorder.start (lock acquire) never block process_loop —
                        # otherwise queued user input would stall silently.
                        if self._voice_mode and self._voice_continuous and not self._voice_recording:
                            def _restart_recording():
                                try:
                                    if self._voice_tts:
                                        self._voice_tts_done.wait(timeout=60)
                                        time.sleep(0.3)
                                    self._voice_start_recording()
                                    app.invalidate()
                                except Exception as e:
                                    _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
                            threading.Thread(target=_restart_recording, daemon=True).start()
                    
                except Exception as e:
                    print(f"Error: {e}")
        
        # Start processing thread
        process_thread = threading.Thread(target=process_loop, daemon=True)
        process_thread.start()
        
        # Register atexit cleanup so resources are freed even on unexpected exit
        atexit.register(_run_cleanup)
        
        # Register signal handlers for graceful shutdown on SSH disconnect / SIGTERM
        def _signal_handler(signum, frame):
            """Handle SIGHUP/SIGTERM by triggering graceful cleanup."""
            logger.debug("Received signal %s, triggering graceful shutdown", signum)
            raise KeyboardInterrupt()
        
        try:
            import signal as _signal
            _signal.signal(_signal.SIGTERM, _signal_handler)
            if hasattr(_signal, 'SIGHUP'):
                _signal.signal(_signal.SIGHUP, _signal_handler)
        except Exception:
            pass  # Signal handlers may fail in restricted environments
        
        # Install a custom asyncio exception handler that suppresses the
        # "Event loop is closed" RuntimeError from httpx transport cleanup.
        # This is defense-in-depth — the primary fix is neuter_async_httpx_del
        # which disables __del__ entirely, but older clients or SDK upgrades
        # could bypass it.
        def _suppress_closed_loop_errors(loop, context):
            exc = context.get("exception")
            if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
                return  # silently suppress
            # Fall back to default handler for everything else
            loop.default_exception_handler(context)

        # Run the application with patch_stdout for proper output handling
        try:
            with patch_stdout():
                # Set the custom handler on prompt_toolkit's event loop
                try:
                    import asyncio as _aio
                    _loop = _aio.get_event_loop()
                    _loop.set_exception_handler(_suppress_closed_loop_errors)
                except Exception:
                    pass
                app.run()
        except (EOFError, KeyboardInterrupt, BrokenPipeError):
            pass
        finally:
            self._should_exit = True
            # Flush memories before exit (only for substantial conversations)
            if self.agent and self.conversation_history:
                try:
                    self.agent.flush_memories(self.conversation_history)
                except (Exception, KeyboardInterrupt):
                    pass
            # Shut down voice recorder (release persistent audio stream)
            if hasattr(self, '_voice_recorder') and self._voice_recorder:
                try:
                    self._voice_recorder.shutdown()
                except Exception:
                    pass
                self._voice_recorder = None
            # Clean up old temp voice recordings
            try:
                from tools.voice_mode import cleanup_temp_recordings
                cleanup_temp_recordings()
            except Exception:
                pass
            # Unregister callbacks to avoid dangling references
            set_sudo_password_callback(None)
            set_approval_callback(None)
            set_secret_capture_callback(None)
            # Close session in SQLite
            if hasattr(self, '_session_db') and self._session_db and self.agent:
                try:
                    self._session_db.end_session(self.agent.session_id, "cli_close")
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not close session in DB: %s", e)
            # Plugin hook: on_session_end — safety net for interrupted exits.
            # run_conversation() already fires this per-turn on normal completion,
            # so only fire here if the agent was mid-turn (_agent_running) when
            # the exit occurred, meaning run_conversation's hook didn't fire.
            if self.agent and getattr(self, '_agent_running', False):
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_end",
                        session_id=self.agent.session_id,
                        completed=False,
                        interrupted=True,
                        model=getattr(self.agent, 'model', None),
                        platform=getattr(self.agent, 'platform', None) or "cli",
                    )
                except Exception:
                    pass
            _run_cleanup()
            self._print_exit_summary()


# ============================================================================
# Main Entry Point
# ============================================================================

def main(
    query: str = None,
    q: str = None,
    toolsets: str = None,
    skills: str | list[str] | tuple[str, ...] = None,
    model: str = None,
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
    max_turns: int = None,
    verbose: bool = False,
    quiet: bool = False,
    compact: bool = False,
    list_tools: bool = False,
    list_toolsets: bool = False,
    gateway: bool = False,
    resume: str = None,
    worktree: bool = False,
    w: bool = False,
    checkpoints: bool = False,
    pass_session_id: bool = False,
):
    """
    Hermes Agent CLI - Interactive AI Assistant
    
    Args:
        query: Single query to execute (then exit). Alias: -q
        q: Shorthand for --query
        toolsets: Comma-separated list of toolsets to enable (e.g., "web,terminal")
        skills: Comma-separated or repeated list of skills to preload for the session
        model: Model to use (default: anthropic/claude-opus-4-20250514)
        provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
        api_key: API key for authentication
        base_url: Base URL for the API
        max_turns: Maximum tool-calling iterations (default: 60)
        verbose: Enable verbose logging
        compact: Use compact display mode
        list_tools: List available tools and exit
        list_toolsets: List available toolsets and exit
        resume: Resume a previous session by its ID (e.g., 20260225_143052_a1b2c3)
        worktree: Run in an isolated git worktree (for parallel agents). Alias: -w
        w: Shorthand for --worktree
    
    Examples:
        python cli.py                            # Start interactive mode
        python cli.py --toolsets web,terminal    # Use specific toolsets
        python cli.py --skills hermes-agent-dev,github-auth
        python cli.py -q "What is Python?"       # Single query mode
        python cli.py --list-tools               # List tools and exit
        python cli.py --resume 20260225_143052_a1b2c3  # Resume session
        python cli.py -w                         # Start in isolated git worktree
        python cli.py -w -q "Fix issue #123"     # Single query in worktree
    """
    global _active_worktree

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

    # Skip worktree for list commands (they exit immediately)
    if not list_tools and not list_toolsets:
        # ── Git worktree isolation (#652) ──
        # Create an isolated worktree so this agent instance doesn't collide
        # with other agents working on the same repo.
        use_worktree = worktree or w or CLI_CONFIG.get("worktree", False)
        wt_info = None
        if use_worktree:
            # Prune stale worktrees from crashed/killed sessions
            _repo = _git_repo_root()
            if _repo:
                _prune_stale_worktrees(_repo)
            wt_info = _setup_worktree()
            if wt_info:
                _active_worktree = wt_info
                os.environ["TERMINAL_CWD"] = wt_info["path"]
                atexit.register(_cleanup_worktree, wt_info)
            else:
                # Worktree was explicitly requested but setup failed —
                # don't silently run without isolation.
                return
    else:
        wt_info = None
    
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
        # Use the shared resolver so MCP servers are included at runtime
        from hermes_cli.tools_config import _get_platform_tools
        toolsets_list = sorted(_get_platform_tools(CLI_CONFIG, "cli"))
    
    parsed_skills = _parse_skills_argument(skills)

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
        checkpoints=checkpoints,
        pass_session_id=pass_session_id,
    )

    if parsed_skills:
        skills_prompt, loaded_skills, missing_skills = build_preloaded_skills_prompt(
            parsed_skills,
            task_id=cli.session_id,
        )
        if missing_skills:
            missing_display = ", ".join(missing_skills)
            raise ValueError(f"Unknown skill(s): {missing_display}")
        if skills_prompt:
            cli.system_prompt = "\n\n".join(
                part for part in (cli.system_prompt, skills_prompt) if part
            ).strip()
            cli.preloaded_skills = loaded_skills

    # Inject worktree context into agent's system prompt
    if wt_info:
        wt_note = (
            f"\n\n[System note: You are working in an isolated git worktree at "
            f"{wt_info['path']}. Your branch is `{wt_info['branch']}`. "
            f"Changes here do not affect the main working tree or other agents. "
            f"Remember to commit and push your changes, and create a PR if appropriate. "
            f"The original repo is at {wt_info['repo_root']}.]"
        )
        cli.system_prompt = (cli.system_prompt or "") + wt_note
    
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
        if quiet:
            # Quiet mode: suppress banner, spinner, tool previews.
            # Only print the final response and parseable session info.
            cli.tool_progress_mode = "off"
            if cli._ensure_runtime_credentials():
                turn_route = cli._resolve_turn_agent_config(query)
                if turn_route["signature"] != cli._active_agent_route_signature:
                    cli.agent = None
                if cli._init_agent(
                    model_override=turn_route["model"],
                    runtime_override=turn_route["runtime"],
                    route_label=turn_route["label"],
                ):
                    cli.agent.quiet_mode = True
                    result = cli.agent.run_conversation(
                        user_message=query,
                        conversation_history=cli.conversation_history,
                    )
                    response = result.get("final_response", "") if isinstance(result, dict) else str(result)
                    if response:
                        print(response)
                    print(f"\nsession_id: {cli.session_id}")
                    
                    # Ensure proper exit code for automation wrappers
                    sys.exit(1 if isinstance(result, dict) and result.get("failed") else 0)
            
            # Exit with error code if credentials or agent init fails
            sys.exit(1)
        else:
            cli.show_banner()
            cli.console.print(f"[bold blue]Query:[/] {query}")
            cli.chat(query)
            cli._print_exit_summary()
        return
    
    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    fire.Fire(main)
