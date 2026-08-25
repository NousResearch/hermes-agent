#!/usr/bin/env python3
"""Hermes Agent CLI — interactive terminal interface (``python cli.py --help`` for usage)."""

# Must be the very first import (UTF-8 stdio on Windows). Missing only mid-``hermes update``.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import logging
import os
import functools
import shutil
import sys
import json
import re
import atexit
import errno
import time
import textwrap
from collections import deque
from dataclasses import dataclass
from urllib.parse import unquote, urlparse
from contextlib import contextmanager, suppress
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Mapping

logger = logging.getLogger(__name__)

os.environ["HERMES_QUIET"] = "1"  # suppress our modules' startup chatter

from hermes_cli.fallback_config import get_fallback_chain
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.cli_billing_mixin import CLIBillingMixin
from hermes_cli.cli_loops_mixin import CLILoopsMixin
from hermes_cli.cli_info_mixin import CLIInfoMixin
from hermes_cli.cli_terminal_mixin import CLITerminalMixin
from hermes_cli.cli_modal_mixin import CLIModalMixin
from hermes_cli.cli_stream_mixin import CLIStreamMixin
from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
from hermes_cli.cli_voice_mixin import CLIVoiceMixin
from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin
from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin
from agent.interrupt_compat import request_hard_interrupt
from agent.pet import render as pet_render

from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import Application
try:
    from prompt_toolkit.enums import EditingMode
except ImportError:  # partial prompt_toolkit stubs in tests
    EditingMode = None
from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
try:
    from prompt_toolkit.cursor_shapes import CursorShape
    _STEADY_CURSOR = CursorShape.BLOCK
except (ImportError, AttributeError):
    _STEADY_CURSOR = None

try:
    from hermes_cli import pt_input_extras as _pt_extras

    _pt_extras.install_shift_enter_alias()
    _pt_extras.install_ctrl_enter_alias()
    _pt_extras.install_cmd_backspace_alias()
    _pt_extras.install_modify_other_keys_aliases()
    _pt_extras.install_keypress_data_normalization()
    _pt_extras.install_ignored_terminal_sequences()
    del _pt_extras
except Exception:
    pass
import threading
import queue


def _lazy_shim(module: str, name: str, alias: str | None = None):
    """Import ``module.name`` on first call; keeps heavy imports off startup while ``cli.<name>`` stays patchable."""
    import importlib

    def shim(*args, **kwargs):
        return getattr(importlib.import_module(module), name)(*args, **kwargs)

    shim.__name__ = shim.__qualname__ = alias or name
    return shim


def format_duration_compact(*args, **kwargs):
    seconds = float(args[0] if args else kwargs.get("seconds", 0.0))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        remaining_min = int(minutes % 60)
        return f"{int(hours)}h {remaining_min}m" if remaining_min else f"{int(hours)}h"
    days = hours / 24
    return f"{days:.1f}d"


# model id -> shortest configured alias (process-lifetime cache; config is read once).
_REVERSE_ALIAS_CACHE: dict[str, str] | None = None


def _reverse_alias_for_display(model_name: str) -> str:
    """Shortest alias for ``model_name`` from ``model_aliases:`` or ``model.aliases:``, else ``model_name``."""
    global _REVERSE_ALIAS_CACHE
    if not model_name:
        return model_name
    if _REVERSE_ALIAS_CACHE is None:
        rmap: dict[str, str] = {}

        def _put(m: str, alias: str) -> None:
            if m and (m not in rmap or len(alias) < len(rmap[m])):
                rmap[m] = alias

        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            ma = cfg.get("model_aliases")
            if isinstance(ma, dict):
                for alias, entry in ma.items():
                    if isinstance(entry, dict):
                        _put(str(entry.get("model", "") or "").strip(), alias)
            mdl = cfg.get("model", {}) or {}
            if isinstance(mdl, dict):
                simple = mdl.get("aliases")
                if isinstance(simple, dict):
                    for alias, val in simple.items():
                        if isinstance(val, str) and val.strip():
                            v = val.strip()
                            _put(v.split("/", 1)[1] if "/" in v else v, alias)
        except Exception:
            pass
        _REVERSE_ALIAS_CACHE = rmap
    return _REVERSE_ALIAS_CACHE.get(model_name, model_name)


def format_token_count_compact(*args, **kwargs):
    value = int(args[0] if args else kwargs.get("value", 0))
    abs_value = abs(value)
    if abs_value < 1_000:
        return str(value)

    sign = "-" if value < 0 else ""
    units = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for threshold, suffix in units:
        if abs_value >= threshold:
            scaled = abs_value / threshold
            text = f"{scaled:.{2 if scaled < 10 else 1 if scaled < 100 else 0}f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"

    return f"{value:,}"


realign_markdown_tables = _lazy_shim("agent.markdown_tables", "realign_markdown_tables")
from hermes_cli.banner import format_banner_version_label

_COMMAND_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


# ~/.hermes/.env first, project .env as dev fallback; user env files override stale shell exports.
from hermes_constants import get_hermes_home
from hermes_state_ids import new_session_id
from hermes_cli.env_loader import load_hermes_dotenv
from utils import base_url_host_matches, base_url_hostname, fast_safe_load, is_truthy_value

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)


_REASONING_TAGS = ("REASONING_SCRATCHPAD", "think", "thinking", "reasoning", "thought")
_TOOL_CALL_TAGS = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")


def _strip_reasoning_tags(text: str) -> str:
    """Strip reasoning blocks (closed, unterminated, orphan-close) and leaked tool-call XML from display text.

    Keep in sync with ``run_agent._strip_think_blocks`` and the stream consumer's think-tag sets.

    Also strips tool-call XML blocks some open models leak into visible content (``<tool_call>``,
    ``<function_calls>``, Gemma-style ``<function name="…">…</function>``). Ported from
    openclaw/openclaw#67318.
    """
    cleaned = text
    for tag in _REASONING_TAGS:
        cleaned = re.sub(rf"<{tag}>.*?</{tag}>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(rf"<{tag}>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(rf"</{tag}>\s*", "", cleaned, flags=re.IGNORECASE)
    for tc_tag in _TOOL_CALL_TAGS:
        cleaned = re.sub(rf"<{tc_tag}\b[^>]*>.*?</{tc_tag}>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # <function name="..."> — boundary + attribute gated to avoid prose false positives.
    cleaned = re.sub(
        r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*<function\b[^>]*\bname\s*=[^>]*>(?:(?:(?!</function>).)*)</function>\s*',
        '', cleaned, flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*', '', cleaned,
        flags=re.IGNORECASE,
    )
    # Unterminated opener / stray <arg_key>/<arg_value> markup = stream cut
    # mid tool-call serialization (#101899); strip to end of text.
    cleaned = re.sub(
        r'(?:^|\n)[ \t]*<(?:tool_call|tool_calls|tool_result|function_call|function_calls)\b[^>]*>.*$'
        r'|(?:^|\n)[^\n<]*</?arg_(?:key|value)\b.*$',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def _assistant_content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return str(content)


def _assistant_copy_text(content: Any) -> str:
    return _strip_reasoning_tags(_assistant_content_as_text(content))


def _load_prefill_messages(file_path: str) -> List[Dict[str, Any]]:
    """Load prefill messages (JSON array) from *file_path*; relative to ~/.hermes/; missing/empty -> []."""
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


def _resolve_prefill_messages_file(config: Dict[str, Any]) -> str:
    """Prefill file path: env, then top-level ``prefill_messages_file``, then legacy ``agent.*``."""
    agent_cfg = config.get("agent", {})
    return (
        os.getenv("HERMES_PREFILL_MESSAGES_FILE", "").strip()
        or str(config.get("prefill_messages_file", "") or "").strip()
        or (str(agent_cfg.get("prefill_messages_file", "") or "").strip() if isinstance(agent_cfg, dict) else "")
    )


def _parse_reasoning_config(effort) -> dict | None:
    """Parse a reasoning effort level (string or YAML bool; ``false``/``off`` = disabled)."""
    from hermes_constants import parse_reasoning_effort
    result = parse_reasoning_effort(effort)
    if effort and str(effort).strip() and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def _parse_service_tier_config(raw: str) -> str | None:
    """Parse a persisted fast-mode preference: None, "priority", "auto", or "cold"."""
    value = str(raw or "").strip().lower()
    if not value or value in {"normal", "default", "standard", "off", "none"}:
        return None
    if value in {"fast", "priority", "on"}:
        return "priority"
    if value in {"auto", "cold"}:
        return value
    logger.warning("Unknown service_tier '%s', ignoring", raw)
    return None


# terminal.<key> -> TERMINAL_<KEY> env var. Container-resource keys apply to docker,
# singularity, modal, daytona and vercel_sandbox only (ignored for local/ssh).
_TERMINAL_ENV_MAPPINGS = {
    key: f"TERMINAL_{key.upper()}"
    for key in (
        "degraded_mode", "cwd", "timeout", "home_mode", "lifetime_seconds", "docker_image",
        "docker_forward_env", "singularity_image", "modal_image", "daytona_image", "vercel_runtime",
        "ssh_host", "ssh_user", "ssh_port", "ssh_key", "container_cpu", "container_memory",
        "container_disk", "container_persistent", "docker_volumes", "docker_env", "docker_extra_args",
        "docker_shm_size", "docker_mount_cwd_to_workspace", "docker_network", "docker_run_as_host_user",
        "docker_snap_compat",
        "docker_persist_across_processes", "docker_shared_container_key", "docker_orphan_reaper",
        "sandbox_dir", "persistent_shell",
    )
}
_TERMINAL_ENV_MAPPINGS = {"env_type": "TERMINAL_ENV", **_TERMINAL_ENV_MAPPINGS, "sudo_password": "SUDO_PASSWORD"}
# Per-task auxiliary endpoint tuples (config key -> env var).
_AUXILIARY_TASK_ENV = {
    "vision": {
        "provider": "AUXILIARY_VISION_PROVIDER",
        "model": "AUXILIARY_VISION_MODEL",
        "base_url": "AUXILIARY_VISION_BASE_URL",
        "api_key": "AUXILIARY_VISION_API_KEY",
    },
    "approval": {
        "provider": "AUXILIARY_APPROVAL_PROVIDER",
        "model": "AUXILIARY_APPROVAL_MODEL",
        "base_url": "AUXILIARY_APPROVAL_BASE_URL",
        "api_key": "AUXILIARY_APPROVAL_API_KEY",
    },
}
_CWD_PLACEHOLDERS = (".", "auto", "cwd")


    # Use user config if it exists, otherwise project config
    if user_config_path.exists() and not ignore_user_config:
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
            "home_mode": "auto",
            "lifetime_seconds": 300,
            "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_forward_env": [],
            "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
            "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_volumes": [],  # host:container volume mounts for Docker backend
            "docker_mount_cwd_to_workspace": False,  # explicit opt-in only; default off for sandbox isolation
            "docker_shared_container_key": "",
        },
        "browser": {
            "inactivity_timeout": 120,  # Auto-cleanup inactive browser sessions after 2 min
            "record_sessions": False,  # Auto-record browser sessions as WebM videos
            "engine": "auto",  # Browser engine: auto (Chrome), lightpanda, chrome
            "camofox": {
                "rewrite_loopback_urls": False,
                "loopback_host_alias": "host.docker.internal",
            },
        },
        "compression": {
            "enabled": True,      # Auto-compress when approaching context limit
            "threshold": 0.50,    # Compress at 50% of model's context limit
            "min_tail_user_messages": 1,  # Real user messages guaranteed in the tail (1 = existing single anchor)
        },
        "agent": {
            "max_turns": 500,  # Default max tool-calling iterations (shared with subagents)
            "verbose": False,
            "system_prompt": "",
            "prefill_messages_file": "",
            "reasoning_effort": "",
            "service_tier": "",
            # Built-in personalities live in hermes_cli.personality
            # (BUILTIN_PERSONALITIES) — the single owner. Entries here are
            # user-defined additions/overrides merged on top by name.
            "personalities": {},
        },

        "display": {
            "compact": False,
            "resume_display": "full",
            # Recap tuning for /resume — see hermes_cli/config.py DEFAULT_CONFIG.
            "resume_exchanges": 10,
            "resume_max_user_chars": 300,
            "resume_max_assistant_chars": 200,
            "resume_max_assistant_lines": 3,
            "resume_skip_tool_only": True,
            # Live reasoning display default ON — keep in sync with
            # hermes_cli/config.py DEFAULT_CONFIG (display.show_reasoning).
            "show_reasoning": True,
            "reasoning_full": False,
            "streaming": True,
            "busy_input_mode": "interrupt",
            "persistent_output": True,
            "persistent_output_max_lines": 200,
            # Clear terminal scrollback as well as the visible viewport when the
            # classic CLI performs a full redraw/resize recovery. Disabled by
            # default because some users prefer preserving terminal history;
            # enable when a terminal/tmux stack stamps stale prompt chrome into
            # scrollback during fullscreen/restore resizes.
            "cli_rebuild_scrollback_on_redraw": False,
            # Print a one-line summary of resolved modal prompts (approval /
            # clarify) into scrollback so the decision survives the repaint.
            "persist_prompts": True,

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
        },
        "delegation": {
            "max_iterations": 45,  # Max tool-calling turns per child agent
            "model": "",       # Subagent model override (empty = inherit parent model)
            "provider": "",    # Subagent provider override (empty = inherit parent provider)
            "base_url": "",    # Direct OpenAI-compatible endpoint for subagents
            "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        },
        "onboarding": {
            # First-touch hint flags (see agent/onboarding.py).  Each hint is
            # shown once per install then latched here.
            "seen": {},
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
            with open(config_path, "r", encoding="utf-8") as f:
                from hermes_cli.config import _normalize_root_model_keys

                file_config = _normalize_root_model_keys(fast_safe_load(f) or {})
            
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

            # Deep merge file_config into defaults.
            # First: merge keys that exist in both (deep-merge dicts, overwrite scalars)
            for key in defaults:
                if key == "model":
                    continue  # Already handled above
                if key in file_config:
                    if isinstance(defaults[key], dict) and file_config[key] is None:
                        continue
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

    # Managed scope: overlay administrator-pinned values LAST so they win over
    # the user's config here too. cli.py builds its config independently of
    # hermes_cli.config._load_config_impl (which has its own managed merge), so
    # without this the entire interactive CLI/TUI surface — skin, display prefs,
    # etc. read from CLI_CONFIG — would silently ignore managed scope while
    # `hermes config`/`doctor`/guards (which use load_config) honor it. The
    # shared helper mirrors _load_config_impl (env-only expansion, root-model
    # normalization, leaf-merge) and is fail-open.
    from hermes_cli import managed_scope

    defaults = managed_scope.apply_managed_overlay(defaults)

    # Apply terminal config to environment variables (so terminal_tool picks them up)
    terminal_config = defaults.get("terminal", {})

    # "backend" (documented) and legacy "env_type" are both accepted; "backend" wins.
    if "backend" in terminal_config:
        terminal_config["env_type"] = terminal_config["backend"]

    # Local backend: cwd is always os.getcwd(). Non-local: a placeholder is popped so
    # terminal_tool uses its per-backend default; an explicit path is kept.
    effective_backend = terminal_config.get("env_type", "local")
    if effective_backend == "local":
        terminal_config["cwd"] = os.getcwd()
        defaults["terminal"]["cwd"] = terminal_config["cwd"]
    elif terminal_config.get("cwd") in _CWD_PLACEHOLDERS:
        terminal_config.pop("cwd", None)
    
    env_mappings = {
        "env_type": "TERMINAL_ENV",
        "degraded_mode": "TERMINAL_DEGRADED_MODE",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "home_mode": "TERMINAL_HOME_MODE",
        "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
        "docker_image": "TERMINAL_DOCKER_IMAGE",
        "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
        "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
        "modal_image": "TERMINAL_MODAL_IMAGE",
        "daytona_image": "TERMINAL_DAYTONA_IMAGE",
        "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
        # SSH config
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        # Container resource config (docker, singularity, modal, daytona, vercel_sandbox -- ignored for local/ssh)
        "container_cpu": "TERMINAL_CONTAINER_CPU",
        "container_memory": "TERMINAL_CONTAINER_MEMORY",
        "container_disk": "TERMINAL_CONTAINER_DISK",
        "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
        "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
        "docker_env": "TERMINAL_DOCKER_ENV",
        "docker_extra_args": "TERMINAL_DOCKER_EXTRA_ARGS",
        "docker_shm_size": "TERMINAL_DOCKER_SHM_SIZE",
        "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
        "docker_network": "TERMINAL_DOCKER_NETWORK",
        "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
        "docker_persist_across_processes": "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
        "docker_shared_container_key": "TERMINAL_DOCKER_SHARED_CONTAINER_KEY",
        "docker_orphan_reaper": "TERMINAL_DOCKER_ORPHAN_REAPER",
        "sandbox_dir": "TERMINAL_SANDBOX_DIR",
        # Persistent shell (non-local backends)
        "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
        # Sudo support (works with all backends)
        "sudo_password": "SUDO_PASSWORD",
    }
    
    # Bridge config → env vars for terminal_tool. TERMINAL_CWD is force-exported
    # UNLESS we're inside a gateway process (detected by _HERMES_GATEWAY marker)
    # where it was already set correctly by gateway/run.py's config bridge.
    _is_gateway = os.environ.get("_HERMES_GATEWAY") == "1"
    for config_key, env_var in _TERMINAL_ENV_MAPPINGS.items():
        if config_key not in terminal_config:
            continue
        val = terminal_config[config_key]
        if env_var == "TERMINAL_CWD":
            if not _is_gateway:
                os.environ[env_var] = str(val)
        elif _file_has_terminal_config or env_var not in os.environ:
            os.environ[env_var] = json.dumps(val) if isinstance(val, (list, dict)) else str(val)

    browser_config = defaults.get("browser", {})
    if "inactivity_timeout" in browser_config:
        os.environ["BROWSER_INACTIVITY_TIMEOUT"] = str(browser_config["inactivity_timeout"])

    # Only non-empty / non-"auto" auxiliary values are bridged so auto-detection still works.
    auxiliary_config = defaults.get("auxiliary", {})
    auxiliary_task_env = {
        # config key → env var mapping
        "vision": {
            "provider": "AUXILIARY_VISION_PROVIDER",
            "model": "AUXILIARY_VISION_MODEL",
            "base_url": "AUXILIARY_VISION_BASE_URL",
            "api_key": "AUXILIARY_VISION_API_KEY",
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
        for field, env_var in env_map.items():
            val = str(task_cfg.get(field, "")).strip()
            if val and not (field == "provider" and val == "auto"):
                os.environ[env_var] = val

    security_config = defaults.get("security", {})
    if isinstance(security_config, dict):
        redact = security_config.get("redact_secrets")
        if redact is not None:
            os.environ["HERMES_REDACT_SECRETS"] = str(redact).lower()

    # Session-search index knobs (hermes_state reads the env carriers).
    sessions_config = defaults.get("sessions", {})
    if isinstance(sessions_config, dict):
        if "cjk_fts" in sessions_config:
            os.environ["HERMES_CJK_FTS"] = str(sessions_config["cjk_fts"])
        if "search_slow_ms" in sessions_config:
            os.environ["HERMES_SEARCH_SLOW_MS"] = str(sessions_config["search_slow_ms"])


def _cli_config_defaults():
    """Built-in defaults for every config key the CLI reads (the file overlays these)."""
    img = "nikolaik/python-nodejs:python3.11-nodejs20"
    return {
        "model": {"default": "", "base_url": "", "provider": "auto"},
        "terminal": {
            "env_type": "local", "cwd": ".", "home_mode": "auto", "lifetime_seconds": 300,  # cwd "." -> os.getcwd()
            "docker_image": img, "docker_forward_env": [], "singularity_image": f"docker://{img}",
            "modal_image": img, "daytona_image": img, "docker_volumes": [],
            "docker_mount_cwd_to_workspace": False,  # opt-in only: sandbox isolation
            "docker_shared_container_key": "",
        },
        "browser": {
            "inactivity_timeout": 120, "record_sessions": False, "engine": "auto",  # auto (Chrome) | lightpanda | chrome
            "camofox": {"rewrite_loopback_urls": False, "loopback_host_alias": "host.docker.internal"},
        },
        # threshold: fraction of the model's context limit; min_tail: real user messages kept in the tail
        "compression": {"enabled": True, "threshold": 0.50, "min_tail_user_messages": 1},
        "agent": {
            "max_turns": 500, "verbose": False, "system_prompt": "", "prefill_messages_file": "",  # max_turns shared with subagents
            "reasoning_effort": "", "service_tier": "",
            "personalities": {},  # user overrides merged by name over hermes_cli.personality builtins
        },
        "display": {
            "compact": False,
            # /resume recap tuning and show_reasoning: keep in sync with hermes_cli/config.py DEFAULT_CONFIG
            "resume_display": "full", "resume_exchanges": 10, "resume_max_user_chars": 300,
            "resume_max_assistant_chars": 200, "resume_max_assistant_lines": 3, "resume_skip_tool_only": True,
            "show_reasoning": True, "reasoning_full": False, "streaming": True, "busy_input_mode": "interrupt",
            "persistent_output": True, "persistent_output_max_lines": 200,
            # Also clear scrollback on redraw/resize recovery; off because users prefer history.
            "cli_rebuild_scrollback_on_redraw": False,
            "persist_prompts": True,  # one-line summary of resolved modal prompts into scrollback
            "skin": "default",
        },
        "clarify": {"timeout": 120},  # seconds before a clarify prompt auto-proceeds
        "code_execution": {"timeout": 300, "max_tool_calls": 50},
        "auxiliary": {"vision": {"provider": "auto", "model": "", "base_url": "", "api_key": ""}},
        # delegation: empty model/provider = inherit parent; api_key falls back to OPENAI_API_KEY
        "delegation": {"max_iterations": 45, "model": "", "provider": "", "base_url": "", "api_key": ""},
        "onboarding": {"seen": {}},  # first-touch hint flags (agent/onboarding.py), latched once shown
    }


def _merge_file_config(defaults: Dict[str, Any], file_config: Dict[str, Any]) -> None:
    """Overlay a parsed config file onto *defaults* in place (model normalization, deep merge, legacy keys)."""
    # model: string (new format) or dict (old format with default/base_url)
    if "model" in file_config:
        if isinstance(file_config["model"], str):
            defaults["model"]["default"] = file_config["model"]
        elif isinstance(file_config["model"], dict):
            defaults["model"].update(file_config["model"])
            # Promote model.model -> model.default (HermesCLI checks "default" first).
            if "model" in file_config["model"] and "default" not in file_config["model"]:
                defaults["model"]["default"] = file_config["model"]["model"]

    # Deep-merge dict sections, overwrite scalars; a None section keeps the defaults;
    # unknown keys (platform_toolsets, memory, ...) are carried over.
    for key, value in file_config.items():
        if key == "model":
            continue
        if isinstance(defaults.get(key), dict):
            if isinstance(value, dict):
                defaults[key].update(value)
            elif value is not None:
                defaults[key] = value
        else:
            defaults[key] = value

    # Legacy root-level max_turns -> agent.max_turns whenever the nested key is missing.
    agent_file_config = file_config.get("agent")
    if "max_turns" in file_config and not (
        isinstance(agent_file_config, dict) and agent_file_config.get("max_turns") is not None
    ):
        defaults["agent"]["max_turns"] = file_config["max_turns"]


def load_cli_config() -> Dict[str, Any]:
    """~/.hermes/config.yaml (else ./cli-config.yaml) over built-in defaults; env vars win.

    ``HERMES_IGNORE_USER_CONFIG=1`` skips the user config entirely (``.env`` still loads).
    """
    config_path = _hermes_home / 'config.yaml'
    if not config_path.exists() or os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1":
        config_path = Path(__file__).parent / 'cli-config.yaml'

    defaults = _cli_config_defaults()

    # Only a file's terminal section may overwrite terminal env vars already set by .env.
    _file_has_terminal_config = False

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                from hermes_cli.config import _normalize_root_model_keys

                file_config = _normalize_root_model_keys(fast_safe_load(f) or {})

            _file_has_terminal_config = "terminal" in file_config
            _merge_file_config(defaults, file_config)
        except Exception as e:
            logger.warning("Failed to load cli-config.yaml: %s", e)

    # Expand ${ENV_VAR} references before bridging to env vars.
    from hermes_cli.config import _expand_env_vars
    defaults = _expand_env_vars(defaults)

    # Administrator-pinned (managed scope) values overlay LAST; cli.py builds its config
    # independently of hermes_cli.config, so this keeps parity with `hermes config`. Fail-open.
    from hermes_cli import managed_scope

    defaults = managed_scope.apply_managed_overlay(defaults)

    _mirror_config_to_env(defaults, _file_has_terminal_config)

    return defaults

CLI_CONFIG = load_cli_config()


def _init_logging_and_display_from_config() -> None:
    """Best-effort startup side effects: logging, config warnings, skin, display knobs."""
    from importlib import import_module as _im

    def _display(key, default):
        return CLI_CONFIG.get("display", {}).get(key, default)

    for step in (
        lambda: _im("hermes_logging").setup_logging(mode="cli"),
        lambda: _im("hermes_cli.config").print_config_warnings(),
        lambda: _im("hermes_cli.skin_engine").init_skin_from_config(CLI_CONFIG),
        lambda: _im("agent.display").set_tool_preview_max_len(int(_display("tool_preview_length", 0) or 0)),
        lambda: _im("agent.display").set_friendly_tool_labels(bool(_display("friendly_tool_labels", True))),
    ):
        try:
            step()
        except Exception:
            pass


_init_logging_and_display_from_config()

# Neuter AsyncHttpxClientWrapper.__del__ before any AsyncOpenAI client exists: it
# schedules aclose() on the running loop (prompt_toolkit's, during idle), closing
# transports bound to dead worker loops ("Event loop is closed" / "Press ENTER to
# continue..."). A meta_path finder patches ``openai._base_client`` at first import —
# eager import costs ~166ms/30MB cold, and the patch is guaranteed to land before
# instantiation. See ``agent.auxiliary_client.neuter_async_httpx_del``.
try:
    import sys as _httpx_neuter_sys
    import importlib.util as _httpx_neuter_imp_util

    class _AsyncHttpxDelNeuter:
        """Patch ``AsyncHttpxClientWrapper.__del__`` to a no-op when ``openai._base_client`` loads."""

        _armed = True

        def find_spec(self, fullname, path=None, target=None):
            if not self._armed or fullname != "openai._base_client":
                return None
            # Disarm before delegating so the recursive find_spec doesn't loop through us.
            self._armed = False
            try:
                _httpx_neuter_sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = _httpx_neuter_imp_util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            _orig_exec = spec.loader.exec_module

            def _patched_exec(module):
                _orig_exec(module)
                try:
                    cls = getattr(module, "AsyncHttpxClientWrapper", None)
                    if cls is not None:
                        cls.__del__ = lambda self: None  # type: ignore[assignment]
                except Exception:
                    pass

            spec.loader.exec_module = _patched_exec  # type: ignore[method-assign]
            return spec

    _httpx_neuter_sys.meta_path.insert(0, _AsyncHttpxDelNeuter())
except Exception:
    pass

from rich.console import Console
from rich.markup import escape as _escape
from rich.text import Text as _RichText

# Agent/tool systems load lazily: bare startup only needs the prompt.
def get_tool_definitions(*args, **kwargs):
    from hermes_cli.mcp_startup import wait_for_mcp_discovery
    from model_tools import get_tool_definitions as _get_tool_definitions

    wait_for_mcp_discovery()
    return _get_tool_definitions(*args, **kwargs)


validate_toolset = _lazy_shim("toolsets", "validate_toolset")


def _sync_process_session_id(session_id: str) -> None:
    """Keep process-local session-id consumers aligned after CLI switches."""
    from gateway.session_context import set_current_session_id

    set_current_session_id(session_id)


_cleanup_all_terminals = _lazy_shim("tools.terminal_tool", "cleanup_all_environments", "_cleanup_all_terminals")
set_sudo_password_callback = _lazy_shim("tools.terminal_tool", "set_sudo_password_callback")
set_approval_callback = _lazy_shim("tools.terminal_tool", "set_approval_callback")
set_secret_capture_callback = _lazy_shim("tools.skills_tool", "set_secret_capture_callback")
_cleanup_all_browsers = _lazy_shim("tools.browser_tool_lifecycle", "_emergency_cleanup_all_sessions", "_cleanup_all_browsers")

_cleanup_done = False  # _run_cleanup runs exactly once
_cleanup_in_progress = False
_cli_wake_owner = None
# One-shot finalization runs before process cleanup (plugins see the boundary while the
# agent is attached); atexit cleanup must not finalize those sessions again.
_single_query_finalize_attempted_session_ids: set[str | None] = set()
# /handoff sessions belong to the gateway: finalizing them here would stamp end_reason on
# a row the gateway just reopened, making the handoff leg vanish from history.
# Session IDs that were handed off to the gateway via /handoff. The CLI process exits after a successful
# handoff, but the gateway now owns the session lifecycle — _run_cleanup must NOT call finalize_session on
# these, because doing so sets end_reason on a row the gateway just reopened and is actively writing to
# (#88234). The race made the handoff leg vanish from session history and broke session_search recall for
# the handed-off session.
_handed_off_session_ids: set[str | None] = set()
_active_agent_ref = None  # active AIAgent, for memory-provider shutdown at exit
_deferred_agent_startup_done = False
# Set once the TUI app starts (focus reporting + mouse tracking on); gates the on-exit
# terminal reset so non-TUI one-shot runs never emit codes for modes they never enabled.
_tui_input_modes_active = False


# Set True once the TUI's prompt_toolkit app starts (which enables focus reporting + mouse tracking). Gates
# the on-exit terminal reset so non-TUI one-shot CLI runs — which also register _run_cleanup via atexit —
# don't emit escape codes for modes they never enabled (#36823).
def _mark_tui_input_modes_active() -> None:
    """Record that the TUI app started, so _run_cleanup resets input modes."""
    global _tui_input_modes_active
    _tui_input_modes_active = True


def _prepare_deferred_agent_startup() -> None:
    """Run Termux-deferred agent discovery before the first real agent turn."""
    global _deferred_agent_startup_done
    if _deferred_agent_startup_done:
        return
    if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
        return
    _deferred_agent_startup_done = True
    _accept_hooks = os.environ.get("HERMES_ACCEPT_HOOKS", "").lower() in {"1", "true", "yes", "on"}
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception:
        logger.warning("plugin discovery failed at deferred CLI startup", exc_info=True)
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(logger=logger, thread_name="termux-cli-mcp-discovery")
    except Exception:
        logger.debug("MCP tool discovery failed at deferred CLI startup", exc_info=True)
    try:
        from agent.shell_hooks import register_from_config
        from agent.outbound_webhooks import register_from_config as register_outbound_webhooks
        from hermes_cli.config import load_config

        _hooks_cfg = load_config()
        register_from_config(_hooks_cfg, accept_hooks=_accept_hooks)
        register_outbound_webhooks(_hooks_cfg)
    except Exception:
        logger.debug("shell-hook registration failed at deferred CLI startup", exc_info=True)


def _flush_logging_and_stdio() -> None:
    """Best-effort ``logging.shutdown()`` + stdout/stderr flush before ``os._exit``."""
    with suppress(Exception):
        logging.shutdown()
    for _stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            _stream.flush()


def _float_env(name: str, default: float) -> float:
    """``float(os.getenv(name))``, or ``default`` when unset/unparseable."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _exit_watchdog_timeout() -> float:
    """``HERMES_EXIT_WATCHDOG_S`` as a float (default 30; ``0`` disables)."""
    return _float_env("HERMES_EXIT_WATCHDOG_S", 30.0)


def _arm_exit_watchdog(timeout_s: float | None = None, *, from_signal: bool = False) -> None:
    """Daemon timer that ``os._exit(0)``s after ``timeout_s`` once shutdown has begun.

    Backstop for a cleanup step wedged on network I/O and for interpreter teardown
    blocked joining non-daemon threads (ThreadPoolExecutor's atexit join). The daemon
    timer survives ``Py_FinalizeEx``'s joins. ``HERMES_EXIT_WATCHDOG_S=0`` disables.

    1. 2. Interpreter teardown blocked joining non-daemon threads — stdlib ``ThreadPoolExecutor`` workers
    are joined unconditionally by ``concurrent.futures``' atexit hook even after ``shutdown(wait=False)``,
    so one tool thread wedged on a socket held the process open forever (#27563 class).
    """
    if timeout_s is None:
        timeout_s = _exit_watchdog_timeout()
    if timeout_s <= 0:
        return
    # Never under pytest: a delayed os._exit(0) would silently kill the test worker.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    def _watchdog():
        time.sleep(timeout_s)
        # The signal-armed watchdog yields to cleanup's own timer once cleanup is running.
        if from_signal and _cleanup_in_progress:
            return

        try:
            logger.warning(
                "Exit watchdog fired after %.0fs — forcing process exit "
                "(a cleanup step or non-daemon thread is wedged).",
                timeout_s,
            )
        except Exception:
            pass
        _flush_logging_and_stdio()
        os._exit(0)

    with suppress(Exception):  # never block shutdown on watchdog setup
        threading.Thread(target=_watchdog, daemon=True, name="exit-watchdog").start()


_signal_watchdog_armed = False


def _arm_exit_watchdog_on_shutdown_signal() -> None:
    """Arm the exit backstop the moment a termination signal arrives (idempotent; never raises).

    The graceful unwind has wedge points BEFORE ``_run_cleanup`` arms its own watchdog
    (main thread in a syscall, prompt_toolkit teardown never returning). Leash is 2x
    the cleanup timeout so a progressing cleanup is never cut short. Never arm at
    startup: the timer exits unconditionally.

    SIGTERM/SIGHUP establish unambiguous shutdown intent, but the graceful path from signal →
    ``agent.interrupt()`` → ``app.exit()`` / ``KeyboardInterrupt`` → ``finally`` → ``_run_cleanup`` has
    several wedge points BEFORE ``_run_cleanup`` arms the normal watchdog: a main thread parked in a syscall
    that never observes the unwind, a prompt_toolkit teardown that never returns, or an agent worker
    blocking the ``finally``. When that happens the process has NO backstop and a "dead" CLI lingers
    (observed: ``hermes --tui`` alive ~47 min at 4% CPU after terminal close — the #65998 class).
    """
    global _signal_watchdog_armed
    if _signal_watchdog_armed:
        return
    _signal_watchdog_armed = True
    base = _exit_watchdog_timeout()
    if base <= 0:
        return  # explicitly disabled
    with suppress(Exception):  # never let the backstop break signal handling
        _arm_exit_watchdog(timeout_s=base * 2, from_signal=True)


def _shutdown_agent_memory_provider(agent) -> None:
    """Memory-provider shutdown (on_session_end + shutdown_all) at the real session boundary."""
    if not (agent and hasattr(agent, 'shutdown_memory_provider')):
        return
    # A /new shortly before exit leaves an LLM-bound boundary task queued; shutdown_all()'s
    # ~5s drain would cancel it, so give it a bounded head start (watchdog is the backstop).
    _mm = getattr(agent, '_memory_manager', None)
    if _mm is not None and hasattr(_mm, 'flush_pending'):
        with suppress(Exception):
            _mm.flush_pending(timeout=10)
    # Forward the agent's transcript so on_session_end hooks see the real conversation;
    # no-arg fallback for stubs / partially-initialised agents.
    _session_msgs = getattr(agent, '_session_messages', None)
    _sid = getattr(agent, "session_id", None) or "<unknown>"
    # ``_session_messages`` is set on ``AIAgent.__init__`` and refreshed every turn via
    # ``_persist_session``. Fall back to no-arg on test stubs / partially-initialised agents where the
    # attribute is missing. See #15165.
    if isinstance(_session_msgs, list):
        logger.info("CLI cleanup calling memory shutdown for session %s with %d message(s)", _sid, len(_session_msgs))
        agent.shutdown_memory_provider(_session_msgs)
    else:
        logger.info("CLI cleanup calling memory shutdown for session %s without session message list", _sid)
        agent.shutdown_memory_provider()


def _stop_cli_wake_word() -> None:
    from tools.wake_word import stop_listening
    if _cli_wake_owner is not None:
        stop_listening(owner=_cli_wake_owner)


def _interrupt_async_delegations() -> None:
    from tools.async_delegation import interrupt_all
    interrupt_all(reason="CLI shutdown")


def _shutdown_mcp_servers() -> None:
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    shutdown_mcp_servers()


def _shutdown_cached_aux_clients() -> None:
    # Otherwise AsyncHttpxClientWrapper.__del__ fires on a closed loop ("Press ENTER to continue...").
    from agent.auxiliary_client import shutdown_cached_clients
    shutdown_cached_clients()


# Ordered teardown steps (attribute names, resolved at call time so tests can patch them)
# and the exception class each swallows.
_CLEANUP_STEPS = (
    ("_stop_cli_wake_word", Exception), ("_cleanup_all_terminals", Exception),
    ("_interrupt_async_delegations", Exception), ("_cleanup_all_browsers", Exception),
    ("_shutdown_mcp_servers", BaseException), ("_shutdown_cached_aux_clients", Exception),
)


def _run_cleanup(*, notify_session_finalize: bool = True):
    """Run resource cleanup exactly once."""
    global _cleanup_done, _cleanup_in_progress
    if _cleanup_done:
        return
    _cleanup_done = True
    _cleanup_in_progress = True

    try:
        _arm_exit_watchdog()
        # Reset terminal input modes FIRST: teardown below can take seconds and a later
        # step raising must not skip the reset. No-op unless the TUI ran.
        # See #36823.
        _reset_terminal_input_modes_on_exit()

        for step, swallow in _CLEANUP_STEPS:
            with suppress(swallow):
                globals()[step]()
        if notify_session_finalize:
            cleanup_session_id = _active_agent_ref.session_id if _active_agent_ref else None
            if _should_emit_cleanup_session_finalize(cleanup_session_id):
                _notify_session_finalize(session_id=cleanup_session_id, platform="cli", reason="shutdown")
        try:
            _shutdown_agent_memory_provider(_active_agent_ref)
        except Exception as e:
            logger.warning("CLI cleanup memory shutdown failed: %s", e, exc_info=True)
    finally:
        _cleanup_in_progress = False


def _should_emit_cleanup_session_finalize(session_id: str | None) -> bool:
    # A handed-off session is owned by the gateway process — never finalize it here.
    # The CLI must not finalize it on exit — that sets end_reason on a row the gateway reopened and is
    # actively writing to, causing the handoff leg to vanish from session history (#88234).
    if session_id is not None and session_id in _handed_off_session_ids:
        return False
    if not _single_query_finalize_attempted_session_ids:
        return True
    if session_id is None:
        return False
    return session_id not in _single_query_finalize_attempted_session_ids


def _notify_session_finalize(*, session_id: str | None, platform: str = "cli", reason: str = "shutdown") -> None:
    with suppress(Exception):
        from hermes_cli.lifecycle import finalize_session
        finalize_session(session_id=session_id, platform=platform, reason=reason)


def _oneshot_agent_and_session(cli):
    """``(agent, session_id)`` for a one-shot run; the agent's id wins over the CLI's."""
    agent = getattr(cli, "agent", None)
    return agent, getattr(agent, "session_id", None) or getattr(cli, "session_id", None)


def _invoke_interrupted_session_end(agent, session_id, reason: str, **extra) -> None:
    """Best-effort ``on_session_end`` hook for a turn cut short (never raises)."""
    with suppress(Exception):
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end", session_id=session_id, completed=False, interrupted=True,
            model=getattr(agent, "model", None), platform=getattr(agent, "platform", None) or "cli",
            reason=reason, **extra,
        )


def _emit_interrupted_session_end(cli, *, reason: str = "keyboard_interrupt") -> None:
    """Best-effort on_session_end hook for interrupted non-interactive runs."""
    agent, session_id = _oneshot_agent_and_session(cli)
    if agent is None:
        return

    with suppress(Exception):
        agent.interrupt(reason.replace("_", " "))

    if session_id in _handed_off_session_ids:  # gateway owns the lifecycle now
        return
    if session_id:
        with suppress(Exception):
            cli.session_id = session_id

    _invoke_interrupted_session_end(
        agent, session_id, reason,
        task_id=getattr(agent, "_current_task_id", "") or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
    )


def _notify_single_query_session_finalize(cli, *, reason: str = "shutdown") -> None:
    agent, session_id = _oneshot_agent_and_session(cli)
    if session_id in _single_query_finalize_attempted_session_ids:
        return
    if session_id in _handed_off_session_ids:  # gateway owns the lifecycle now
        return

    try:
        _notify_session_finalize(session_id=session_id, platform=getattr(agent, "platform", None) or "cli", reason=reason)
    finally:
        _single_query_finalize_attempted_session_ids.add(session_id)


def _flush_one_shot_session_store(cli) -> None:
    """Durably flush + finalize the one-shot session row before exit (idempotent, best-effort).

    One-shot runs get a single turn, so nothing retries a transiently-failed transcript
    flush, closes the session row, or drains token deltas the kanban ``os._exit(0)``
    path skips. Handed-off sessions are left alone.

    - a turn whose in-loop ``_flush_messages_to_session_db`` failed under write-lock contention (e.g. a busy
    multiplex gateway sharing state.db) was silently lost — the reply reached stdout and agent.log but the
    resumed session's stored history never changed (#88583); - the resumed/created titled session row was
    left dangling open (``ended_at``/``end_reason`` NULL) on every one-shot exit; - queued async
    token-accounting deltas relied on interpreter-exit hooks, which the kanban SIGTERM path's
    ``os._exit(0)`` skips entirely.
    Idempotent and best-effort: ``_persist_session`` dedupes via the per-message ``_DB_PERSISTED_MARKER``
    stamps (already-written turns are not re-written) and ``end_session`` no-ops on an already-ended row.
    See #88234.
    """
    agent, session_id = _oneshot_agent_and_session(cli)
    if agent is None or not session_id or session_id in _handed_off_session_ids:
        return
    if getattr(agent, "_persist_disabled", False):
        return
    # Passing cli.conversation_history keeps resumed messages identity-skipped even when
    # the failed flush never stamped them.
    try:
        msgs = getattr(agent, "_session_messages", None)
        if isinstance(msgs, list) and msgs and hasattr(agent, "_persist_session"):
            agent._persist_session(msgs, getattr(cli, "conversation_history", None))
    except Exception:
        logger.debug("one-shot final session persist retry failed", exc_info=True)
    db = getattr(agent, "_session_db", None) or getattr(cli, "_session_db", None)
    if db is None:
        return
    try:
        db.flush_token_counts()
    except Exception:
        logger.debug("one-shot token-count drain failed", exc_info=True)
    try:
        db.end_session(session_id, "cli_close")
    except Exception:
        logger.debug("one-shot end_session failed", exc_info=True)


def _wait_for_oneshot_background_completions(cli) -> None:
    """Bounded linger for notify_on_complete background processes (#90879).

    A one-shot run (``-q`` / ``-Q``) that spawned bounded background work —
    most importantly a Bot Mode handoff reply via ``message_agent`` /
    ``bot_relay``, spawned as ``terminal(background=true,
    notify_on_complete=true)`` — must not exit while that work is still
    running: the children write to pipes owned by this process and are
    destroyed shortly after it dies. Delegates the actual wait (and its
    ``terminal.oneshot_completion_wait_seconds`` bound) to the process
    registry. Cheap no-op when nothing is pending.
    """
    from tools.process_registry import process_registry

    agent = getattr(cli, "agent", None)
    task_id = getattr(agent, "session_id", None) or getattr(cli, "session_id", None)
    # Wait on the whole registry, not just this task's processes: a one-shot
    # CLI process hosts exactly one agent, so every tracked process in this
    # interpreter was spawned by this run (task_id filtering would silently
    # skip processes registered before the session id settled).
    result = process_registry.wait_for_pending_completions(None)
    if result.get("waited"):
        logger.info(
            "One-shot exit linger for session %s: completed=%s timed_out=%s",
            task_id or "<unknown>",
            result.get("completed"),
            result.get("timed_out"),
        )


def _finalize_single_query(cli) -> None:
    """Close one-shot CLI resources before releasing the active session lease."""
    try:
        # Linger (bounded) for background processes the turn spawned with
        # notify_on_complete=true BEFORE any teardown. The one-shot parent
        # owns those children's stdout pipes; exiting now kills the delivery
        # a few seconds later. Bot Mode handoff replies dispatched from a
        # short-lived `hermes -p <bot> chat -Q` recipient (message_agent /
        # bot_relay spawns) are exactly this shape and were silently
        # destroyed on parent exit (#90879).
        try:
            _wait_for_oneshot_background_completions(cli)
        except Exception:
            logger.debug("one-shot background completion wait failed", exc_info=True)
        # Durable flush FIRST: memory-provider shutdown inside _run_cleanup
        # can issue aux-LLM calls, and nothing after it may fail in a way
        # that loses the turn (#88583).
        try:
            _flush_one_shot_session_store(cli)
        except Exception:
            logger.debug("one-shot session store flush failed", exc_info=True)
        _notify_single_query_session_finalize(cli)
        _run_cleanup(notify_session_finalize=False)
    finally:
        cli._release_active_session()


def _reset_terminal_input_modes_on_exit() -> None:
    """Disable focus reporting + mouse tracking on TUI exit (best-effort).

    Ctrl+C / SIGTERM / crashes bypass prompt_toolkit's unwind, leaving focus events and
    mouse reports as visible text in the next shell. Writes to stdout when it is the
    terminal, else /dev/tty (the TUI may have run with stdout redirected).

    Called from ``_run_cleanup`` (atexit-registered + invoked on the normal / EOF / interrupt exit paths)
    this covers normal quit, Ctrl+C and SIGTERM/SIGHUP. ``kill -9`` is uncatchable, and the kanban worker's
    ``os._exit(0)`` path bypasses ``atexit``; neither runs this — but both are non-TTY / non-TUI, so there
    is nothing to reset there. See #36823.
    """
    global _tui_input_modes_active
    if not _tui_input_modes_active:
        return
    # Clear first so a re-armed _run_cleanup doesn't re-emit.
    _tui_input_modes_active = False
    try:
        stream = sys.stdout
        if stream is not None and stream.isatty():
            stream.write(_TERMINAL_INPUT_MODE_RESET_SEQ)
            stream.flush()
            return
    except Exception:
        pass
    with suppress(Exception), open("/dev/tty", "w", encoding="ascii") as tty:
        tty.write(_TERMINAL_INPUT_MODE_RESET_SEQ)
        tty.flush()


from hermes_cli.worktree_ops import (
    _git_quiet,
    _git_repo_root,
    _maintain_pack_health,
    _prune_stale_worktrees,
    _repo_is_shallow,
    _setup_worktree,
    _worktree_has_unpushed_commits,
)

# ============================================================================= Git Worktree Isolation
# (#652) =============================================================================
_active_worktree: Optional[Dict[str, str]] = None


def _normalize_git_bash_path(p: Optional[str]) -> Optional[str]:
    """Translate a Git Bash-style path (``/c/Users/...``) to the native
    Windows form (``C:\\Users\\...``) that Python's ``subprocess.Popen``
    and ``pathlib.Path`` accept.

    No-op on non-Windows and for paths that already look native.  Git on
    native Windows normally emits forward-slash Windows paths
    (``C:/Users/...``) which both bash and Python handle, but certain
    configurations (Git Bash shells, MSYS2, WSL-mounted repos) surface
    ``/c/...`` or ``/cygdrive/c/...`` variants.
    """
    if not p:
        return p
    if sys.platform != "win32":
        return p
    import re as _re
    # /c/Users/... or /C/Users/...
    m = _re.match(r"^/([a-zA-Z])/(.*)$", p)
    if m:
        drive, rest = m.group(1), m.group(2)
        return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"
    # /cygdrive/c/... or /mnt/c/...
    m = _re.match(r"^/(?:cygdrive|mnt)/([a-zA-Z])/(.*)$", p)
    if m:
        drive, rest = m.group(1), m.group(2)
        return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"
    return p


def _git_repo_root() -> Optional[str]:
    """Return the git repo root for CWD, or None if not in a repo.

    Runs through :func:`_normalize_git_bash_path` so callers can pass
    the result directly to ``Path``/``subprocess.Popen(cwd=...)`` on
    Windows without hitting ``C:\\c\\Users\\...`` style resolution
    mistakes.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        if result.returncode == 0:
            return _normalize_git_bash_path(result.stdout.strip())
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


def _cleanup_failed_worktree_add(repo_root: str, wt_path: Path, branch_name: str) -> None:
    """Make a failed/timed-out ``git worktree add`` atomic after the fact.

    ``git worktree add`` is not transactional: killed mid-checkout (the 30s
    timeout) it leaves (a) the partially-materialized worktree directory,
    (b) an admin entry under ``.git/worktrees/<name>`` that is LOCKED with a
    reason naming the *current, live* pid — so the startup pruner's
    dead-pid unlock will never touch it — and (c) sometimes the new branch.
    Any retry of the same name then fails on the leftovers. Sweep all three,
    quietly; every step is fail-soft because this runs on an error path.
    """
    import shutil
    import subprocess

    def _git(*args: str) -> None:
        try:
            subprocess.run(
                ["git", *args],
                capture_output=True, text=True, timeout=15, cwd=repo_root, check=False,
            )
        except Exception:
            pass

    try:
        # Unlock first: `worktree remove --force` refuses a locked tree.
        _git("worktree", "unlock", str(wt_path))
        _git("worktree", "remove", "--force", str(wt_path))
        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)
        # Drop the orphaned admin entry when the dir is already gone
        # (`remove` needs the dir; `prune` handles the dirless case).
        _git("worktree", "prune")
        _git("branch", "-D", branch_name)
    except Exception as e:
        logger.debug("cleanup after failed worktree add: %s", e)


_PACK_SPRAWL_THRESHOLD = 15


def _maintain_pack_health(repo_root: str) -> None:
    """Repack the object store when pack files sprawl (background thread).

    On a multi-agent box every fetch/salvage session adds packs; git never
    consolidates them on its own aggressively enough (``gc --auto``'s
    threshold is 50 *and* it counts only non-kept packs). Past a few dozen
    packs every object lookup scans every pack index, and worktree creation
    can blow its 30s timeout under concurrent load (Aug 2026 incident: 39
    packs, 638MB → ``hermes -w`` timing out; a full repack halved the store
    and restored 0.5s creates). Threshold 15 keeps lookups fast without
    repacking on every startup; ``nice`` + background thread keeps it off
    the startup path. Fail-soft everywhere.
    """
    import subprocess

    try:
        pack_dir = Path(repo_root) / ".git" / "objects" / "pack"
        if not pack_dir.is_dir():
            return
        packs = len(list(pack_dir.glob("*.pack")))
        if packs < _PACK_SPRAWL_THRESHOLD:
            return
        logger.info("git pack sprawl (%d packs) — repacking in background", packs)
        cmd = ["git", "repack", "-a", "-d", "--quiet"]
        if os.name == "posix":
            cmd = ["nice", "-n", "19", *cmd]
        subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=1800, cwd=repo_root, check=False,
        )
        # Repacking can strand now-duplicated admin files; a prune here keeps
        # the worktree bookkeeping tight on the same maintenance pass.
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True, text=True, timeout=60, cwd=repo_root, check=False,
        )
    except Exception as e:
        logger.debug("pack maintenance skipped: %s", e)


def _resolve_worktree_base(
    repo_root: str,
    fetch_timeout: float = 5,
    freshness_window: float = 300,
) -> tuple:
    """Resolve the freshest base ref to branch a new worktree from.

    The standalone clone's ``HEAD`` can lag the remote by hundreds of commits
    (the ``~/.hermes/hermes-agent`` clone is updated only by ``hermes update``,
    not on every session). Branching a worktree from that stale ``HEAD`` roots
    every new branch on an old base — so the PR diff GitHub computes against
    current ``main`` balloons with unrelated changes, and the agent has to
    discover the staleness via the pre-push gate and rebase. Branching from the
    freshly-fetched remote tip instead means the worktree starts current.

    Strategy (each step falls back to the next on failure):
      1. If the current branch tracks an upstream, refresh and use that
         upstream ref — so a deliberate feature-branch worktree tracks its own
         remote, not the default branch.
      2. Else refresh the remote's default branch (``origin/HEAD`` → e.g.
         ``origin/main``) and use it.
      3. Else fall back to ``HEAD`` (offline, no remote, or detached) — the
         old behavior, never worse than before.

    "Refresh" is deliberately cheap on the startup path (the fetch here used
    to stall ``hermes -w`` launches for 30-60s on flaky smart-HTTP
    connections):

    - The fetch is SKIPPED entirely when the repo's ``FETCH_HEAD`` is younger
      than *freshness_window* seconds — a base fetched moments ago cannot have
      meaningfully moved, so repeated launches don't re-pay a network round
      trip.
    - The fetch is capped at *fetch_timeout* seconds. On timeout or failure we
      fall back to the locally-known remote-tracking ref (labelled "cached")
      instead of cascading into a second fetch attempt. Genuine staleness is
      backstopped by the pre-push stale-base gate.

    Returns ``(base_ref, label)`` where *base_ref* is a git revision suitable
    for ``git worktree add ... <base_ref>`` and *label* is a short
    human-readable description for the session banner.
    """
    import subprocess

    from hermes_cli._subprocess_compat import noninteractive_git_env

    def _git(args, timeout: float = 20):
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
            stdin=subprocess.DEVNULL,
            env=noninteractive_git_env(),
        )

    def _ref_exists(ref: str) -> bool:
        try:
            return _git(["rev-parse", "--verify", "--quiet", ref + "^{commit}"]).returncode == 0
        except Exception:
            return False

    def _fetch_head_age() -> Optional[float]:
        """Seconds since the last fetch in this repo, or None if unknown."""
        try:
            gd = _git(["rev-parse", "--git-dir"])
            if gd.returncode != 0:
                return None
            git_dir = Path(gd.stdout.strip())
            if not git_dir.is_absolute():
                git_dir = Path(repo_root) / git_dir
            fetch_head = git_dir / "FETCH_HEAD"
            if not fetch_head.exists():
                return None
            return max(0.0, time.time() - fetch_head.stat().st_mtime)
        except Exception:
            return None

    def _refresh(remote: str, branch: str, ref: str) -> tuple:
        """Return (ref, label) after a cheap best-effort refresh of *ref*.

        Never raises, never fetches twice, never blocks longer than
        *fetch_timeout*.
        """
        age = _fetch_head_age()
        if age is not None and age < freshness_window and _ref_exists(ref):
            return ref, f"{ref} (fetched {int(age)}s ago)"
        try:
            fetched = _git(["fetch", remote, branch], timeout=fetch_timeout)
            if fetched.returncode == 0:
                return ref, f"{ref} (fetched)"
            reason = "fetch failed"
        except subprocess.TimeoutExpired:
            reason = f"fetch timed out after {fetch_timeout:g}s"
        except Exception as e:
            reason = f"fetch error: {e}"
        if _ref_exists(ref):
            logger.debug("worktree base: %s — using cached %s", reason, ref)
            return ref, f"{ref} (cached — {reason})"
        return "HEAD", f"HEAD (local — {reason}, no cached {ref})"

    # 1. Current branch's upstream, if it tracks one.
    try:
        up = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
        if up.returncode == 0:
            upstream = up.stdout.strip()  # e.g. "origin/main"
            if upstream and "/" in upstream:
                remote, branch = upstream.split("/", 1)
                return _refresh(remote, branch, upstream)
    except Exception as e:
        logger.debug("worktree base: upstream resolution failed: %s", e)

    # 2. Remote default branch (origin/HEAD).
    try:
        # Resolve the remote's default branch symref.
        head_ref = _git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
        default_ref = ""
        if head_ref.returncode == 0:
            default_ref = head_ref.stdout.strip().replace("refs/remotes/", "", 1)
        if not default_ref:
            # origin/HEAD not set locally; ask the remote (network — capped
            # like the fetch so a stalled connection can't hang startup).
            show = _git(["remote", "show", "origin"], timeout=max(fetch_timeout, 5))
            for line in show.stdout.splitlines():
                line = line.strip()
                if line.startswith("HEAD branch:"):
                    _branch = line.split(":", 1)[1].strip()
                    # A remote with no default branch reports "(unknown)";
                    # don't construct a bogus "origin/(unknown)" ref from it.
                    if _branch and _branch != "(unknown)":
                        default_ref = "origin/" + _branch
                    break
        if default_ref and "/" in default_ref:
            remote, branch = default_ref.split("/", 1)
            return _refresh(remote, branch, default_ref)
    except Exception as e:
        logger.debug("worktree base: default-branch resolution failed: %s", e)

    # 3. Fall back to local HEAD (offline / no remote / detached).
    return "HEAD", "HEAD (local — could not reach remote)"


def _setup_worktree(repo_root: str = None, sync_base: bool = True,
                    name: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Create an isolated git worktree for this CLI session.

    Returns a dict with worktree metadata on success, None on failure.
    The dict contains: path, branch, repo_root.

    When *sync_base* is True (default), the worktree branches from the
    freshly-fetched remote tip rather than the (possibly stale) local ``HEAD``
    — see ``_resolve_worktree_base``. Set ``worktree_sync: false`` in config to
    branch from local ``HEAD`` (the pre-#10760-followup behavior).

    When *name* is given (``/worktree new <name>``), the worktree directory
    and branch use the sanitized name instead of a random ``hermes-<id>``.
    Named trees intentionally skip the ``hermes-`` prefix so the startup
    pruner ages them on its slower named-tree schedule.
    """
    import subprocess

    repo_root = repo_root or _git_repo_root()
    if not repo_root:
        _cprint("\033[31m✗ --worktree requires being inside a git repository.\033[0m")
        print("  cd into your project repo first, then run hermes -w")
        return None

    if name:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")[:40]
        if safe:
            wt_name = safe
        else:
            wt_name = f"hermes-{uuid.uuid4().hex[:8]}"
    else:
        wt_name = f"hermes-{uuid.uuid4().hex[:8]}"
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    wt_path = worktrees_dir / wt_name
    if name and wt_path.exists():
        _cprint(f"\033[31m✗ Worktree already exists: {wt_path}\033[0m")
        print("  Pick a different name, or remove it with: "
              f"git worktree remove {wt_path}")
        return None

    # Ensure .worktrees/ is in .gitignore
    gitignore = Path(repo_root) / ".gitignore"
    _ignore_entry = ".worktrees/"
    try:
        # utf-8-sig: git files are UTF-8 and Notepad prepends a BOM, which
        # would glue to the first line and defeat the membership check below
        # (duplicating the entry); the locale default also breaks non-ASCII
        # patterns on Windows. The append below already writes UTF-8.
        existing = (
            gitignore.read_text(encoding="utf-8-sig", errors="replace")
            if gitignore.exists()
            else ""
        )
        if _ignore_entry not in existing.splitlines():
            with open(gitignore, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{_ignore_entry}\n")
    except Exception as e:
        logger.debug("Could not update .gitignore: %s", e)

    # Resolve the base ref. By default branch from the freshly-fetched remote
    # tip so the worktree starts current with the project, not from the
    # (possibly stale) local HEAD of the standalone clone (#10760 follow-up).
    if sync_base:
        base_ref, base_label = _resolve_worktree_base(repo_root)
    else:
        base_ref, base_label = "HEAD", "HEAD (local — worktree_sync disabled)"

    # Create the worktree. checkout.workers parallelizes the file
    # materialization (~6k files on this repo): 0.6s serial → ~0.2s with 8
    # workers. Harmless on git builds without parallel-checkout support —
    # unknown -c keys are ignored for checkout, and the fallback retry
    # below drops the flags entirely.
    _wt_add_cfg = [
        "-c", "checkout.workers=8",
        "-c", "checkout.thresholdForParallelism=100",
    ]
    try:
        # 120s, not 30: on a multi-agent box the ~10k-file materialization
        # contends with sibling sessions' checkouts/fetches/Electron dev
        # builds for the same disk — measured 113s wall at near-zero CPU
        # under load vs 1.2s idle (Aug 2026). A too-tight timeout kills a
        # legitimately slow create and wastes the work already done.
        result = subprocess.run(
            ["git", *_wt_add_cfg, "worktree", "add", str(wt_path), "-b", branch_name, base_ref],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=repo_root,
        )
        if result.returncode != 0:
            # If branching from the resolved remote ref failed for any reason
            # (e.g. a partial fetch left the ref unusable), retry from local
            # HEAD so worktree creation never hard-fails on a sync hiccup.
            if base_ref != "HEAD":
                logger.warning(
                    "worktree add from %s failed (%s); retrying from local HEAD",
                    base_ref, result.stderr.strip(),
                )
                _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
                base_ref, base_label = "HEAD", "HEAD (fallback — remote base failed)"
                result = subprocess.run(
                    ["git", "worktree", "add", str(wt_path), "-b", branch_name, base_ref],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=repo_root,
                )
            if result.returncode != 0:
                _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
                _cprint(f"\033[31m✗ Failed to create worktree: {result.stderr.strip()}\033[0m")
                return None
    except Exception as e:
        # A timed-out/failed `worktree add` is NOT atomic: git leaves the
        # partially-materialized directory plus a LOCKED admin entry under
        # .git/worktrees/<name> whose lock pid is THIS live process — so the
        # startup pruner's dead-pid unlock never reaps it and every retry of
        # the same name fails. Clean up our own wreckage before surfacing
        # the error (Aug 2026 incident: 30s timeout during pack-sprawl left
        # exactly this poison).
        _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
        _cprint(f"\033[31m✗ Failed to create worktree: {e}\033[0m")
        return None

    # Copy files listed in .worktreeinclude (gitignored files the agent needs)
    include_file = Path(repo_root) / ".worktreeinclude"
    if include_file.exists():
        try:
            repo_root_resolved = Path(repo_root).resolve()
            wt_path_resolved = wt_path.resolve()
            # utf-8-sig, not the locale default: on a cp1251/GBK Windows
            # machine a UTF-8 include list either decodes to mojibake paths
            # (entries silently not copied) or raises UnicodeDecodeError,
            # which the enclosing handler swallows at DEBUG — no include is
            # copied at all. A Notepad BOM likewise glued to the first entry.
            for line in include_file.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines():
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
                    # Symlink directories (faster, saves disk).  On Windows,
                    # symlink creation requires Developer Mode or elevation,
                    # and fails with OSError otherwise — fall back to a
                    # recursive copy so the worktree is still usable.  The
                    # copy is slower and uses disk, but it doesn't require
                    # admin and matches the Linux/macOS symlink outcome
                    # functionally.
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.symlink(str(src_resolved), str(dst))
                        except (OSError, NotImplementedError) as _sym_err:
                            if sys.platform == "win32":
                                logger.info(
                                    ".worktreeinclude: symlink failed (%s) — "
                                    "falling back to copytree on Windows.",
                                    _sym_err,
                                )
                                try:
                                    shutil.copytree(
                                        str(src_resolved),
                                        str(dst),
                                        symlinks=True,
                                        dirs_exist_ok=False,
                                    )
                                except Exception as _copy_err:
                                    logger.warning(
                                        ".worktreeinclude: copy fallback "
                                        "also failed for %s -> %s: %s",
                                        src, dst, _copy_err,
                                    )
                            else:
                                raise
        except Exception as e:
            logger.debug("Error copying .worktreeinclude entries: %s", e)

    # Lock the worktree so other processes (and `git worktree remove`) can see
    # it is actively in use.  Fail-soft: a lock failure never blocks the session.
    try:
        subprocess.run(
            ["git", "worktree", "lock", "--reason", f"hermes pid={os.getpid()}", str(wt_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        logger.debug("Worktree locked: %s (pid=%s)", wt_path, os.getpid())
    except Exception as e:
        logger.debug("git worktree lock failed (non-fatal): %s", e)

    info = {
        "path": str(wt_path),
        "branch": branch_name,
        "repo_root": repo_root,
        "base": base_ref,
    }

    _cprint(f"\033[32m✓ Worktree created:\033[0m {wt_path}")
    print(f"  Branch: {branch_name}")
    print(f"  Base:   {base_label}")

    return info


def _worktree_has_unpushed_commits(worktree_path: str, timeout: int = 10) -> bool:
    """Return whether a worktree has commits not reachable from any remote branch.

    ``git log HEAD --not --remotes`` compares against remote-tracking refs under
    ``refs/remotes/*``. If a repo has no remote-tracking refs yet, there is no
    usable remote baseline to compare against, so treat it as having no
    "unpushed" commits.

    SHALLOW-CLONE CAVEAT: in a shallow clone (the installer default) the
    shallow boundary can disconnect an older worktree HEAD from origin/*,
    making already-public commits look unpushed. The verdict here stays
    conservative (True) on purpose — deleting on unverifiable history would
    risk real work. Callers that can afford it should deepen first via
    ``_deepen_shallow_repo`` (the startup pruner does) or check
    ``_repo_is_shallow`` before presenting this verdict as fact.
    """
    import subprocess

    try:
        remote_refs = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)", "refs/remotes"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if remote_refs.returncode != 0:
            return True
        if not remote_refs.stdout.strip():
            return False

        result = subprocess.run(
            ["git", "log", "--oneline", "HEAD", "--not", "--remotes"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _worktree_is_dirty(worktree_path: str, timeout: int = 10) -> bool:
    """Return whether a worktree has uncommitted changes (staged, unstaged, or
    untracked).

    Fails SAFE: on any error returns True so callers do not delete a worktree
    whose state they cannot determine.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _repo_is_shallow(repo_path: str, timeout: int = 5) -> bool:
    """Return whether *repo_path* belongs to a shallow clone.

    Shallowness poisons every history-connectivity verdict the worktree
    machinery relies on: an older worktree's HEAD (a past snapshot of main)
    is disconnected from current ``origin/main`` by the shallow boundary, so
    ``git log HEAD --not --remotes`` misreports thousands of already-public
    commits as "unpushed" and the worktree is preserved forever. The default
    installer clones with ``--depth 1``, so this is the normal state of a
    user install, not an edge case.

    Fails toward False: if git can't be queried we don't want callers to
    take shallow-specific branches on top of an unknown state.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_path,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def _deepen_shallow_repo(repo_root: str, timeout: int = 600) -> bool:
    """One-time blobless unshallow so history-based verdicts become correct.

    Fetches the full commit/tree graph (``--unshallow --filter=blob:none``)
    without downloading historical file contents, which keeps the transfer a
    small fraction of a full clone. Runs only from background paths (the
    startup pruner thread), never on the interactive session-close path.

    Falls back to a plain ``--unshallow`` if the server rejects partial-clone
    filters. Fail-soft: returns whether the repo is actually non-shallow
    afterwards; on failure (offline, no remote) callers keep today's
    preserve-everything behavior.
    """
    import subprocess

    if not _repo_is_shallow(repo_root):
        return True

    try:
        remotes = subprocess.run(
            ["git", "remote"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        names = [r.strip() for r in remotes.stdout.splitlines() if r.strip()]
        if remotes.returncode != 0 or not names:
            return False
        remote = "origin" if "origin" in names else names[0]

        for extra in (["--filter=blob:none"], []):
            try:
                result = subprocess.run(
                    ["git", "fetch", remote, "--unshallow", *extra],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
                )
            except subprocess.TimeoutExpired:
                return False
            if result.returncode == 0:
                break
            logger.debug(
                "git fetch --unshallow%s failed: %s",
                " " + " ".join(extra) if extra else "",
                result.stderr.strip()[-500:],
            )
    except Exception as e:
        logger.debug("Deepening shallow repo failed (non-fatal): %s", e)
        return False

    deepened = not _repo_is_shallow(repo_root)
    if deepened:
        logger.info(
            "Deepened shallow clone at %s so worktree cleanup can verify "
            "push state", repo_root,
        )
    return deepened


# Upper bound on retained `git cherry` verdict entries (see
# _save_worktree_merge_cache). Each entry is ~90 bytes, so this caps the cache
# near 90 KB even on a repo that churns thousands of worktree branches.
_WORKTREE_MERGE_CACHE_MAX = 1000


def _worktree_merge_cache_path() -> Path:
    """Path of the patch-equivalence verdict cache (profile-aware)."""
    return get_hermes_home() / "cache" / "worktree_merge_verdicts.json"


def _load_worktree_merge_cache() -> Dict[str, bool]:
    """Load the ``git cherry`` verdict cache. Missing/corrupt cache = empty."""
    try:
        raw = json.loads(
            _worktree_merge_cache_path().read_text(encoding="utf-8")
        )
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("verdicts")
    if not isinstance(entries, dict):
        return {}
    # Only keep well-formed bool verdicts — a hand-edited or partially written
    # cache must never inject a non-bool into the prune decision.
    return {k: v for k, v in entries.items() if isinstance(v, bool)}


def _save_worktree_merge_cache(verdicts: Dict[str, bool]) -> None:
    """Persist the verdict cache atomically. Best-effort — never raises.

    Bounded to the most recent ``_WORKTREE_MERGE_CACHE_MAX`` entries so the
    file can't grow without limit across thousands of sessions.
    """
    path = _worktree_merge_cache_path()
    tmp = None
    try:
        items = list(verdicts.items())
        if len(items) > _WORKTREE_MERGE_CACHE_MAX:
            items = items[-_WORKTREE_MERGE_CACHE_MAX:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"version": 1, "verdicts": dict(items)}),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    except Exception as e:
        logger.debug("Could not persist worktree merge cache: %s", e)
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass


def _worktree_commits_all_merged_upstream(
    worktree_path: str,
    timeout: int = 30,
    max_ahead: int = 20,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Return whether every local-only commit is patch-equivalent to a commit
    already on the default upstream branch.

    The dominant ``.worktrees/`` leak: a branch is pushed, its PR is
    squash-merged (or cherry-picked), and the remote branch is deleted. The
    local commits are then unreachable from ``refs/remotes/*`` forever, so the
    unpushed-commits guard preserves the worktree indefinitely even though its
    content is fully merged. ``git cherry`` detects patch-equivalence, letting
    the pruner reap these.

    Bounded: skips (returns False) when the branch is more than ``max_ahead``
    commits ahead — a stale-base tree, too expensive to diff-hash and unlikely
    to be a merged scratch branch. Fails SAFE toward False (preserve).

    ``git cherry`` diff-hashes every commit in the range, which on a large repo
    costs ~0.2-1.0s per worktree — and a tree preserved for unpushed work is
    re-tested on *every* startup, forever, always reaching the same answer. When
    *cache* is provided, the verdict is memoized against
    ``(base_sha, head_sha, max_ahead)``: the exact inputs ``git cherry``
    consumes. A cache hit is therefore identical to recomputation by
    construction — if either ref moves the key changes and the real git call
    runs again.
    """
    import subprocess

    base = None
    for candidate in ("origin/HEAD", "origin/main", "origin/master"):
        try:
            probe = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", candidate],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                base = candidate
                break
        except Exception:
            return False
    if base is None:
        return False

    try:
        # Resolve both endpoints to shas up front. These are the complete
        # inputs to the range below, so they form an exact cache key. Cheap
        # (~1ms) relative to the diff-hashing `git cherry` they guard.
        cache_key = None
        if cache is not None:
            revs = subprocess.run(
                ["git", "rev-parse", f"{base}^{{commit}}", "HEAD^{commit}"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            )
            if revs.returncode == 0:
                shas = revs.stdout.split()
                if len(shas) == 2:
                    cache_key = f"{shas[0]}..{shas[1]}:{max_ahead}"
                    if cache_key in cache:
                        return cache[cache_key]

        def _memo(verdict: bool) -> bool:
            if cache is not None and cache_key is not None:
                cache[cache_key] = verdict
            return verdict

        ahead = subprocess.run(
            ["git", "rev-list", "--count", f"{base}..HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if ahead.returncode != 0:
            return False
        count = int(ahead.stdout.strip() or "0")
        if count == 0:
            return _memo(True)
        if count > max_ahead:
            return _memo(False)

        cherry = subprocess.run(
            ["git", "cherry", base, "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if cherry.returncode != 0:
            return False
        lines = [ln for ln in cherry.stdout.splitlines() if ln.strip()]
        # "-" = patch-equivalent commit exists upstream; "+" = unique local work
        return _memo(bool(lines) and all(ln.startswith("-") for ln in lines))
    except Exception:
        return False


def _worktree_branch_pr_merged(
    worktree_path: str,
    timeout: int = 15,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Return whether the worktree branch's PR is MERGED on GitHub.

    Escape hatch for the case ``git cherry`` cannot catch: a rebase-merge that
    altered the diff (conflict resolution against a moved base, follow-up
    commits added during salvage/CI-fix) changes the patch-id, so the local
    commits are no longer patch-equivalent to anything upstream even though
    the PR merged. Those trees survive the cherry check forever (Aug 2026:
    12 of 22 "unpushed" trees on a loaded box had MERGED PRs).

    GitHub's PR state is the authoritative merge signal, so a clean tree
    whose branch has a MERGED PR is reaped. Verdicts are memoized keyed on
    ``(branch, head_sha)`` — MERGED is monotonic, so a True verdict is cached
    permanently; False is never cached (the PR may merge later without new
    local commits, which would leave the key unchanged).

    Fails SAFE toward False (preserve): no gh binary, offline, rate-limited,
    detached HEAD, or any parse failure keeps the tree.
    """
    import subprocess

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if head.returncode != 0:
            return False
        branch = head.stdout.strip()
        if not branch or branch == "HEAD":  # detached — no PR to look up
            return False

        cache_key = None
        if cache is not None:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            )
            if sha.returncode == 0 and sha.stdout.strip():
                cache_key = f"pr-merged:{branch}:{sha.stdout.strip()}"
                if cache.get(cache_key) is True:
                    return True

        result = subprocess.run(
            ["gh", "pr", "list", "--head", branch, "--state", "merged",
             "--json", "number", "--limit", "1"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return False
        prs = json.loads(result.stdout or "[]")
        merged = isinstance(prs, list) and len(prs) > 0
        if merged and cache is not None and cache_key is not None:
            cache[cache_key] = True
        return merged
    except Exception:
        return False


def _worktree_lock_is_live(repo_root: str, worktree_path: str, timeout: int = 10):
    """Classify a worktree's git lock as live, dead, or absent.

    ``hermes -w`` locks each worktree with reason ``hermes pid=<pid>`` so a
    concurrent hermes process' startup prune leaves an in-use worktree alone.
    But a *crashed* session leaves the lock behind forever, and
    ``git worktree remove --force`` (single ``-f``) refuses to remove a locked
    worktree — so dead-locked worktrees accumulate indefinitely. This lets the
    pruner tell the two apart:

    - ``"live"``  — locked and the owning pid is still running (skip it).
    - ``"dead"``  — locked but the owning pid is gone, or the reason isn't a
                    parseable hermes lock (safe to unlock + reap).
    - ``None``    — not locked at all.

    Fails SAFE toward ``"live"``: if git can't be queried at all we cannot
    prove the worktree is safe to touch, so we report it as live.
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
        )
        if result.returncode != 0:
            return "live"
    except Exception:
        return "live"

    target = Path(worktree_path).resolve()
    current: Optional[Path] = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            try:
                current = Path(line[len("worktree "):].strip()).resolve()
            except Exception:
                current = None
        elif line == "locked" or line.startswith("locked "):
            if current != target:
                continue
            reason = line[len("locked"):].strip()
            m = re.search(r"hermes pid=(\d+)", reason)
            if not m:
                # Locked by something we don't recognize as a hermes session
                # (or lock reason unavailable). Treat as dead — a foreign lock
                # on a hermes -w worktree is almost certainly a leftover, and
                # the age/dirty/unpushed gates already ran before we got here.
                return "dead"
            pid = int(m.group(1))
            if pid == os.getpid():
                return "live"
            try:
                from gateway.status import _pid_exists
                return "live" if _pid_exists(pid) else "dead"
            except Exception:
                # Can't determine liveness — fail safe toward keeping it.
                return "live"
    return None


def _cleanup_worktree(info: Dict[str, str] = None) -> None:
    """Remove a worktree and its branch on exit; kept only when it has unpushed commits."""
    global _active_worktree
    info = info or _active_worktree
    if not info:
        return

    wt_path, branch, repo_root = info["path"], info["branch"], info["repo_root"]
    if not Path(wt_path).exists():
        return

    if _worktree_has_unpushed_commits(wt_path, timeout=10):
        if _repo_is_shallow(repo_root):
            # In a shallow clone the unpushed verdict is unreliable: the
            # shallow boundary disconnects this worktree's history from
            # origin/*, so already-public commits look "unpushed". Be honest
            # about why we're keeping it — the startup pruner deepens the
            # clone in the background and will reap it on a later startup.
            _cprint(f"\n\033[33m⚠ Shallow clone — cannot verify push state, keeping: {wt_path}\033[0m")
            print("  The next `hermes -w` session deepens the clone and prunes merged worktrees automatically.")
        else:
            _cprint(f"\n\033[33m⚠ Worktree has unpushed commits, keeping: {wt_path}\033[0m")
            print(f"  To clean up manually: git worktree remove --force {wt_path}")
        _active_worktree = None
        return

    # Unlock first so `remove` isn't blocked by the lock placed at creation. Fail-soft.
    _git_quiet(["worktree", "unlock", wt_path], repo_root, log="git worktree unlock failed (non-fatal)")
    _git_quiet(["worktree", "remove", wt_path, "--force"], repo_root, timeout=15, log="Failed to remove worktree")
    _git_quiet(["branch", "-D", branch], repo_root, log=f"Failed to delete branch {branch}")

    _active_worktree = None
    _cprint(f"\033[32m✓ Worktree cleaned up: {wt_path}\033[0m")


def _run_state_db_auto_maintenance(session_db) -> None:
    """One-time repairs + auto-archive/prune/vacuum per the ``sessions:`` config. Never raises."""
    if session_db is None:
        return
    try:
        from hermes_cli.config import load_config as _load_full_config
        from hermes_constants import get_hermes_home as _get_hermes_home  # lazy: tests patch it
        _hermes_home_maint = _get_hermes_home()

        # One-time repairs, each latched in state_meta once it has run.
        for meta_key, repair, done_msg, skip_msg in (
            (
                "ghost_session_prune_v1",
                lambda: session_db.prune_empty_ghost_sessions(sessions_dir=_hermes_home_maint / "sessions"),
                "Pruned %d empty TUI ghost sessions", "Ghost session prune skipped: %s",
            ),
            (
                "orphaned_compression_finalize_v1",
                session_db.finalize_orphaned_compression_sessions,
                "Finalized %d orphaned compression sessions", "Orphan compression finalize skipped: %s",
            ),
        ):
            try:
                if not session_db.get_meta(meta_key):
                    count = repair()
                    session_db.set_meta(meta_key, "1")
                    if count:
                        logger.info(done_msg, count)
            except Exception as _exc:
                logger.debug(skip_msg, _exc)

        cfg = (_load_full_config().get("sessions") or {})

        # Auto-archive is independent of auto_prune: run it before prune's early return.
        if cfg.get("auto_archive", False):
            session_db.maybe_auto_archive(
                idle_days=float(cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            )

        if not cfg.get("auto_prune", False):
            return
        session_db.maybe_auto_prune_and_vacuum(
            retention_days=int(cfg.get("retention_days", 90)),
            min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            min_vacuum_interval_days=int(cfg.get("min_vacuum_interval_days", 30)),
            vacuum=bool(cfg.get("vacuum_after_prune", True)),
            sessions_dir=_hermes_home_maint / "sessions",
        )
    except Exception as exc:
        logger.debug("state.db auto-maintenance skipped: %s", exc)


def _run_checkpoint_auto_maintenance() -> None:
    """Call ``maybe_auto_prune_checkpoints`` per the ``checkpoints:`` config. Never raises."""
    try:
        from hermes_cli.config import load_config as _load_full_config
        cfg = (_load_full_config().get("checkpoints") or {})
        if not cfg.get("auto_prune", False):
            return
        from tools.checkpoint_manager import maybe_auto_prune_checkpoints
        # delete_orphans stays False: a missing workdir at startup is ambiguous (unmounted
        # volume / VPN down); orphans are only reclaimed by `hermes checkpoints prune`.
        maybe_auto_prune_checkpoints(
            retention_days=int(cfg.get("retention_days", 7)),
            min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            delete_orphans=False,
            max_total_size_mb=int(cfg.get("max_total_size_mb", 500)),
        )
    except Exception as exc:
        logger.debug("checkpoint auto-maintenance skipped: %s", exc)


def _prune_stale_worktrees(repo_root: str, max_age_hours: int = 24) -> None:
    """Remove stale worktrees and orphaned branches on startup.

    Covers EVERY directory under ``.worktrees/`` except kanban task trees
    (``t_<hex>`` — owned by the kanban dispatcher's own gc). Scratch trees
    created by ``hermes -w`` (``hermes-*``) age out fast; named trees created
    manually for salvage/review lanes age out on a slower schedule:

    - ``hermes-*``: skip under 24h; reap 24h+ when clean and merged/pushed;
      72h+ is the aggressive tier (still never deletes real work).
    - named trees: same logic at 3x the timeline (72h soft / 9d hard).

    Work-preservation guards (all tiers, any age):
    - uncommitted changes (dirty) — never removed;
    - unpushed commits — never removed, UNLESS every local-only commit is
      patch-equivalent to a commit already on upstream (``git cherry``): the
      squash-merged-PR case, which is the dominant ``.worktrees/`` leak since
      those commits stay unreachable from ``refs/remotes/*`` forever.

    Lock handling (orthogonal to age): ``hermes -w`` locks each worktree with
    reason ``hermes pid=<pid>`` so a concurrent hermes process leaves an in-use
    worktree alone. A *live*-locked worktree is skipped at any age; a
    *dead*-locked one (owning pid gone — a crashed session) is unlocked first
    so ``git worktree remove --force`` can actually reap it, otherwise those
    leftovers accumulate forever (``remove --force`` refuses a locked tree).

    Branch deletion is gated on ``git worktree remove`` succeeding, so a failed
    removal never orphans the branch (which would drop easy reachability of any
    commits still in the worktree).

    Preserved-work visibility: trees skipped for unpushed/dirty reasons that
    are older than 7 days are listed in a single WARNING so real in-flight
    work can't rot silently.

    Also prunes orphaned ``hermes/*`` and ``pr-*`` local branches that
    have no corresponding worktree.

    Performance: this runs on the startup path of every ``hermes -w`` session,
    and each candidate tree costs several git subprocesses (the ``git cherry``
    patch-equivalence probe dominates at ~0.2-1.0s on a large repo). With
    dozens of accumulated worktrees the serial version added ~11-18s of latency
    before the banner. Two changes keep the decisions byte-identical while
    removing nearly all of that:

    1. The read-only classification of each tree (dirty / unpushed / merged /
       lock state) is independent per tree, so it runs on a thread pool. Only
       the mutating phase (unlock, remove, branch -D) stays serial and ordered.
    2. ``git cherry`` verdicts are memoized on disk keyed by the exact
       ``(base_sha, head_sha)`` range they were computed from, so a tree
       preserved for unpushed work is not re-diff-hashed on every subsequent
       startup.
    """
    import re
    import subprocess
    import time

    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        _prune_orphaned_branches(repo_root)
        return

    # A shallow clone (the installer's default `--depth 1`) disconnects old
    # worktree HEADs from current origin/main, so the unpushed-commits guard
    # misclassifies every aged worktree as unpushed work and preserves it
    # forever. Deepen once — bloblessly, in this background thread — so all
    # history verdicts below (and the session-exit cleanup) become correct.
    # Fail-soft: offline, we just keep today's preserve-everything behavior.
    if _repo_is_shallow(repo_root):
        _deepen_shallow_repo(repo_root)

    now = time.time()
    stale_work_cutoff = now - (7 * 24 * 3600)
    preserved_stale: list = []
    # Kanban task worktrees (<repo>/.worktrees/t_<hex>) have their own
    # dispatcher-driven lifecycle (hermes kanban gc) — never touch them here.
    kanban_re = re.compile(r"^t_[0-9a-f]+$")

    # ── Phase 1: age filter (no subprocesses) ───────────────────────────────
    # Cheap stat-only pass so the thread pool below is sized to the trees that
    # actually need git work, not to everything on disk.
    candidates: list = []
    for entry in sorted(worktrees_dir.iterdir()):
        if not entry.is_dir() or kanban_re.match(entry.name):
            continue

        # Scratch trees (hermes-*) age out on the default schedule; named
        # trees (salvage/review lanes someone created deliberately) get 3x.
        scratch = entry.name.startswith("hermes-")
        tier_hours = max_age_hours if scratch else max_age_hours * 3
        soft_cutoff = now - (tier_hours * 3600)
        hard_cutoff = now - (tier_hours * 3 * 3600)

        try:
            mtime = entry.stat().st_mtime
            if mtime > soft_cutoff:
                continue  # Too recent — skip
        except Exception:
            continue

        candidates.append((entry, mtime, mtime <= hard_cutoff))

    if not candidates:
        _prune_orphaned_branches(repo_root)
        return

    # ── Phase 2: classify in parallel (read-only git queries) ───────────────
    # Every check here is a read-only git query against a distinct worktree, so
    # they are safe to run concurrently (git takes no repo-wide lock for these,
    # and each has its own index). Verdicts are collected and applied serially
    # below so removal order and log output stay deterministic.
    merge_cache = _load_worktree_merge_cache()
    cache_size_before = len(merge_cache)
    cache_lock = threading.Lock()

    def _classify(item):
        entry, mtime, force = item
        # Never delete real work, regardless of age or tier. Uncommitted
        # changes and unpushed commits may be a crashed session's in-flight
        # work; only clean, fully-merged/pushed trees (the scratch trees that
        # actually cause .worktrees/ bloat) are ever reaped.
        if _worktree_is_dirty(str(entry), timeout=5):
            return (entry, mtime, force, "dirty", None)
        if _worktree_has_unpushed_commits(str(entry), timeout=5):
            # Squash-merge escape hatch: commits unreachable from any remote
            # ref but patch-equivalent to upstream commits are merged work,
            # not unpushed work.
            with cache_lock:
                snapshot = dict(merge_cache)
            merged = _worktree_commits_all_merged_upstream(
                str(entry), timeout=30, cache=snapshot
            )
            if not merged:
                # Rebase-merge escape hatch: conflict resolution or follow-up
                # commits change the patch-id, so cherry misses them — but
                # GitHub knows the PR merged. Authoritative and cheap (~0.3s,
                # memoized on (branch, head_sha) so it's paid once per tree).
                merged = _worktree_branch_pr_merged(
                    str(entry), timeout=15, cache=snapshot
                )
            with cache_lock:
                merge_cache.update(snapshot)
            if not merged:
                return (entry, mtime, force, "unpushed", None)

        # Respect git-native session locks. A lock owned by a still-running
        # hermes process means the worktree is actively in use — never touch
        # it. A lock whose owning pid is gone is a crashed session's leftover:
        # unlock it so `git worktree remove --force` (single -f) can reap it,
        # otherwise dead-locked worktrees pile up indefinitely.
        lock_state = _worktree_lock_is_live(repo_root, str(entry), timeout=5)
        if lock_state == "live":
            return (entry, mtime, force, "locked-live", None)
        return (entry, mtime, force, "reap", lock_state)

    # Bounded pool: enough to hide git's per-process startup latency without
    # spawning dozens of concurrent git processes on a small machine.
    workers = max(1, min(8, (os.cpu_count() or 4), len(candidates)))
    try:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="hermes-wt-prune"
            ) as pool:
                verdicts = list(pool.map(_classify, candidates))
        else:
            verdicts = [_classify(c) for c in candidates]
    except Exception as e:
        # Never let a pool failure block startup — fall back to serial.
        logger.debug("Parallel worktree classification failed (%s); serial", e)
        verdicts = [_classify(c) for c in candidates]

    if len(merge_cache) != cache_size_before:
        _save_worktree_merge_cache(merge_cache)

    # ── Phase 3: mutate serially (unlock / remove / branch -D) ──────────────
    for entry, mtime, force, verdict, lock_state in verdicts:
        if verdict == "dirty":
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} (uncommitted changes)")
            continue
        if verdict == "unpushed":
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} (unpushed commits)")
            continue
        if verdict == "locked-live":
            logger.debug("Skipping live-locked worktree: %s", entry.name)
            continue

        if lock_state == "dead":
            try:
                subprocess.run(
                    ["git", "worktree", "unlock", str(entry)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
                )
            except Exception as e:
                logger.debug("Failed to unlock dead worktree %s: %s", entry.name, e)

        # Safe to remove
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()

            remove_result = subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, cwd=repo_root,
            )
            if remove_result.returncode != 0:
                # Removal failed — keep the branch so any commits stay
                # reachable rather than orphaning it.
                logger.debug(
                    "Failed to remove worktree %s: %s",
                    entry.name, remove_result.stderr.strip(),
                )
                continue
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
                )
            logger.debug("Pruned stale worktree: %s (force=%s)", entry.name, force)
        except Exception as e:
            logger.debug("Failed to prune worktree %s: %s", entry.name, e)

    if preserved_stale:
        logger.warning(
            "Preserving %d worktree(s) older than 7 days with unmerged work "
            "(run `hermes worktree prune` to review and reclaim): %s",
            len(preserved_stale), ", ".join(sorted(preserved_stale)),
        )

    _prune_orphaned_branches(repo_root)

    # Escalation notice: the startup pass is deliberately conservative, so
    # installs accumulate preserved trees it can never reclaim. Once the
    # footprint is clearly a problem (many trees or multi-GB), say so once
    # per launch and name the attended reclaim command — silence here is how
    # boxes reach 15GB+ of .worktrees/ without anyone noticing.
    try:
        from hermes_cli.worktree_gc import worktrees_summary

        count, size_mb = worktrees_summary(repo_root)
        if count >= 10 or (size_mb or 0) >= 5120:
            size_txt = f"{size_mb / 1024:.1f}GB" if size_mb else "unknown size"
            logger.warning(
                ".worktrees/ holds %d tree(s) (%s) — run `hermes worktree list` "
                "to audit and `hermes worktree prune` to reclaim safely.",
                count, size_txt,
            )
    except Exception:
        pass


def _prune_orphaned_branches(repo_root: str) -> None:
    """Delete local ``hermes/hermes-*`` and ``pr-*`` branches with no worktree.

    These are auto-generated by ``hermes -w`` sessions and PR review
    workflows respectively.  Once their worktree is gone they serve no
    purpose and just accumulate.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        if result.returncode != 0:
            return
        all_branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
    except Exception:
        return

    # Collect branches that are actively checked out in a worktree
    active_branches: set = set()
    try:
        wt_result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        for line in wt_result.stdout.split("\n"):
            if line.startswith("branch refs/heads/"):
                active_branches.add(line.split("branch refs/heads/", 1)[-1].strip())
    except Exception:
        return  # Can't determine active branches — bail

    # Also protect the currently checked-out branch and main
    try:
        head_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, cwd=repo_root,
        )
        current = head_result.stdout.strip()
        if current:
            active_branches.add(current)
    except Exception:
        pass
    active_branches.add("main")

    orphaned = [
        b for b in all_branches
        if b not in active_branches
        and (b.startswith("hermes/hermes-") or b.startswith("pr-"))
    ]

    if not orphaned:
        return

    # Delete in batches
    for i in range(0, len(orphaned), 50):
        batch = orphaned[i:i + 50]
        try:
            subprocess.run(
                ["git", "branch", "-D"] + batch,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=repo_root,
            )
        except Exception as e:
            logger.debug("Failed to prune orphaned branches: %s", e)

    logger.debug("Pruned %d orphaned branches", len(orphaned))

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
_ACCENT_ANSI_DEFAULT = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold — fallback
_BOLD = "\033[1m"
_RST = "\033[0m"
_STREAM_PAD = ""  # no indent: leading whitespace pollutes copy/paste
_STREAM_PARTIAL_PREVIEW_LEN = 60  # tail of an unfinished line mirrored into the spinner


def _hex_to_ansi(hex_color: str, *, bold: bool = False) -> str:
    """Convert '#RRGGBB' to a true-color ANSI escape, remapping dark-tuned colors in light mode."""
    hex_color = _maybe_remap_for_light_mode(hex_color)
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        return f"\033[{'1;' if bold else ''}38;2;{r};{g};{b}m"
    except (ValueError, IndexError):
        return _ACCENT_ANSI_DEFAULT if bold else "\033[38;2;184;134;11m"


# Light/dark terminal detection (mirrors ui-tui/src/theme.ts detectLightMode()). Priority:
# HERMES_LIGHT/HERMES_TUI_LIGHT env, HERMES_TUI_THEME, HERMES_TUI_BACKGROUND, COLORFGBG
# (bg slot 7/15 = light), OSC 11 query, default dark. Cached so the terminal is queried once.
_LIGHT_MODE_CACHE: bool | None = None
_TRUE_RE = re.compile(r"^(1|true|on|yes|y)$")
_FALSE_RE = re.compile(r"^(0|false|off|no|n)$")
_LIGHT_DEFAULT_TERM_PROGRAMS = frozenset()  # Apple_Terminal isn't reliable; require explicit config


def _luminance_from_hex(hex_str: str) -> float | None:
    """Rec.709 luma in [0, 1] for '#RGB'/'#RRGGBB', or None when malformed."""
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or not all(c in "0123456789abcdefABCDEF" for c in s):
        return None
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


_DA1_REPLY_RE = re.compile(rb"\x1b\[\?[0-9;]*c")


def _query_osc11_background() -> str | None:
    """Terminal background via OSC 11 as "#RRGGBB", or None.

    Fenced with a DA1 sentinel (``ESC[c``): terminals answer in order and virtually all
    answer DA1, so its reply proves our OSC 11 was processed — otherwise a late reply
    leaks into prompt_toolkit's stdin as typed text. Skipped over SSH (round-trip too
    slow; a late BEL reads as Ctrl+G). A 50 ms drain after TCSAFLUSH catches stragglers.

    After the main read + TCSAFLUSH, a short drain window (50 ms) catches late-arriving bytes that slipped
    past the flush — a race observed on VPS and container terminals under load (#40250).
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    if any(os.environ.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return None
    try:
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return None
    try:
        try:
            tty.setcbreak(fd)
        except Exception:
            return None
        try:
            # One write so the OSC 11 query and DA1 fence cannot reorder.
            sys.stdout.write("\x1b]11;?\x1b\\\x1b[c")
            sys.stdout.flush()
        except Exception:
            return None
        # Read until the DA1 fence closes; the 1s deadline only covers terminals ignoring DA1.
        deadline = time.monotonic() + 1.0
        buf = b""
        while time.monotonic() < deadline:
            r, _, _ = select.select([fd], [], [], deadline - time.monotonic())
            if not r:
                continue
            try:
                chunk = os.read(fd, 64)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            if _DA1_REPLY_RE.search(buf):
                break
        # Reply: \x1b]11;rgb:RRRR/GGGG/BBBB\x1b\\ — components are 1-4 hex digits.
        m = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
        if not m:
            return None

        def norm(h: bytes) -> int:
            v = int(h, 16)
            bits = len(h) * 4
            return (v * 255) // ((1 << bits) - 1) if bits else 0
        r, g, b = norm(m.group(1)), norm(m.group(2)), norm(m.group(3))
        return f"#{r:02X}{g:02X}{b:02X}"
    finally:
        # TCSAFLUSH discards unread input, scrubbing a partial reply before prompt_toolkit reads it.
        with suppress(Exception):
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        try:
            drain_deadline = time.monotonic() + 0.05
            while time.monotonic() < drain_deadline:
                r, _, _ = select.select([fd], [], [], drain_deadline - time.monotonic())
                if not r or not os.read(fd, 64):
                    break
        except Exception:
            pass


def _heal_cooked_mode_drift(fd: int) -> bool:
    """Re-apply raw mode on *fd* when termios drifted back to cooked (POSIX only).

    A lost ``run_in_terminal`` cooked_mode() restore makes the kernel line-buffer every
    keystroke and the CLI looks dead. Mirrors prompt_toolkit's raw_mode flag surgery in
    place. Returns True when healed; False when already raw or not inspectable.
    """
    try:
        import termios
        attrs = termios.tcgetattr(fd)
    except Exception:
        return False
    lflag = attrs[3]
    if not (lflag & (termios.ICANON | termios.ECHO)):
        return False  # still raw — nothing to do
    attrs[3] = lflag & ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
    attrs[0] = attrs[0] & ~(termios.IXON | termios.IXOFF | termios.ICRNL | termios.INLCR | termios.IGNCR)
    attrs[6][termios.VMIN] = 1
    try:
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        return False
    return True


def _detect_light_mode_uncached() -> bool:
    """The detection ladder documented above; may raise (caller maps errors to dark)."""
    for var in ("HERMES_LIGHT", "HERMES_TUI_LIGHT"):
        v = (os.environ.get(var) or "").strip().lower()
        if _TRUE_RE.match(v):
            return True
        if _FALSE_RE.match(v):
            return False
    theme = (os.environ.get("HERMES_TUI_THEME") or "").strip().lower()
    if theme == "light":
        return True
    if theme == "dark":
        return False
    bg_lum = _luminance_from_hex(os.environ.get("HERMES_TUI_BACKGROUND") or "")
    if bg_lum is not None:
        return bg_lum >= 0.5
    last = (os.environ.get("COLORFGBG") or "").strip().split(";")[-1]
    if last.isdigit() and 0 <= int(last) < 16:
        return int(last) in {7, 15}
    bg_color = _query_osc11_background()
    if bg_color:
        lum = _luminance_from_hex(bg_color)
        if lum is not None:
            return lum >= 0.5
    return (os.environ.get("TERM_PROGRAM") or "").strip() in _LIGHT_DEFAULT_TERM_PROGRAMS


def _detect_light_mode() -> bool:
    global _LIGHT_MODE_CACHE
    if _LIGHT_MODE_CACHE is not None:
        return _LIGHT_MODE_CACHE
    try:
        result = _detect_light_mode_uncached()
    except Exception:
        result = False
    _LIGHT_MODE_CACHE = result
    return result


# Light-mode equivalents of skin colors unreadable on cream backgrounds. Only colors used
# as STANDALONE foregrounds: ones paired with a dark bg (status bar text on #1a1a2e) would
# become invisible the other direction, hence #C0C0C0/#888888/#555555/#8B8682 are skipped.
_LIGHT_MODE_REMAP: dict[str, str] = {
    "#FFF8DC": "#1A1A1A", "#FFD700": "#9A6B00", "#FFBF00": "#8A5A00", "#B8860B": "#5C4500",
    "#DAA520": "#6B4F00", "#F1E6CF": "#1A1A1A", "#c9d1d9": "#24292F", "#EAF7FF": "#0F1B26",
    "#F5F5F5": "#1A1A1A", "#FFF0D4": "#1A1A1A", "#CD7F32": "#8A4F1A", "#FFEFB5": "#3A2A00",
}
_LIGHT_MODE_REMAP_UPPER = {k.upper(): v for k, v in _LIGHT_MODE_REMAP.items()}


def _maybe_remap_for_light_mode(hex_color: str) -> str:
    """In light mode, remap a dark-tuned color to its higher-contrast equivalent."""
    if not _detect_light_mode():
        return hex_color
    if not hex_color or not hex_color.startswith("#"):
        return hex_color
    return _LIGHT_MODE_REMAP_UPPER.get(hex_color.upper(), hex_color)


def _install_skin_light_mode_hook() -> None:
    """Wrap SkinConfig.get_color so EVERY skin color read goes through the light-mode remap. Idempotent."""
    try:
        from hermes_cli.skin_engine import SkinConfig  # type: ignore[import]
    except Exception:
        return
    if getattr(SkinConfig, "_hermes_light_mode_hook_installed", False):
        return
    _orig_get_color = SkinConfig.get_color

    def _wrapped_get_color(self, key, fallback=""):
        value = _orig_get_color(self, key, fallback)
        try:
            return _maybe_remap_for_light_mode(value)
        except Exception:
            return value

    SkinConfig.get_color = _wrapped_get_color  # type: ignore[method-assign]
    SkinConfig._hermes_light_mode_hook_installed = True  # type: ignore[attr-defined]


_install_skin_light_mode_hook()


# Prime the light-mode cache when interactive so OSC 11 happens before prompt_toolkit owns the tty.
with suppress(Exception):
    if sys.stdin.isatty() and sys.stdout.isatty():
        _detect_light_mode()


class _SkinAwareAnsi:
    """Lazy ANSI escape resolved from the skin on first use; ``.reset()`` after a ``/skin`` switch."""

    def __init__(self, skin_key: str, fallback_hex: str = "#FFD700", *, bold: bool = False):
        self._skin_key = skin_key
        self._fallback_hex = fallback_hex
        self._bold = bold
        self._cached: str | None = None

    def __str__(self) -> str:
        if self._cached is None:
            try:
                from hermes_cli.skin_engine import get_active_skin
                self._cached = _hex_to_ansi(
                    get_active_skin().get_color(self._skin_key, self._fallback_hex),
                    bold=self._bold,
                )
            except Exception:
                self._cached = _hex_to_ansi(self._fallback_hex, bold=self._bold)
        return self._cached

    def __add__(self, other: str) -> str:
        return str(self) + other

    def __radd__(self, other: str) -> str:
        return other + str(self)

    def reset(self) -> None:
        """Clear cache so the next access re-reads the skin."""
        self._cached = None


_ACCENT = _SkinAwareAnsi("response_border", "#FFD700", bold=True)
# dim+italic attributes (not a hex) so dim text inherits the terminal foreground in both modes.
_DIM = "\x1b[2;3m"


def _tty_wrap(s: str, sgr: str) -> str:
    """Wrap *s* in an SGR attribute when stdout is a real TTY; plain text otherwise."""
    try:
        return f"{sgr}{s}\x1b[0m" if sys.stdout.isatty() else str(s)
    except Exception:
        return str(s)


_b = functools.partial(_tty_wrap, sgr="\x1b[1m")  # bold when stdout is a real TTY
_d = functools.partial(_tty_wrap, sgr="\x1b[2;3m")  # dim-italic when stdout is a real TTY


def _accent_hex() -> str:
    """Return the active skin accent color for legacy CLI output lines."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin().get_color("ui_accent", "#FFBF00")
    except Exception:
        return "#FFBF00"


def _rich_text_from_ansi(text: str) -> _RichText:
    """Rich Text from ANSI output; literal ``[brackets]`` are not treated as markup."""
    return _RichText.from_ansi(text or "")


def _strip_markdown_syntax(text: str) -> str:
    """Best-effort markdown marker removal for plain-text display."""
    plain = _rich_text_from_ansi(text or "").plain
    # HR markers: "-"/"_" runs of 3+, but "*" only when exactly 3 (cron schedules "* * * * *").
    plain = re.sub(r"^\s{0,3}(?:[-_]\s*){3,}$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}(?:\*\s*){3}\s*$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}#{1,6}\s+", "", plain, flags=re.MULTILINE)
    # Blockquotes, lists, and checkboxes are preserved because they carry structure.
    plain = re.sub(r"(```+|~~~+)", "", plain)
    plain = re.sub(r"`([^`]*)`", r"\1", plain)
    plain = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)___([^_]+)___(?!\w)", r"\1", plain)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)__([^_]+)__(?!\w)", r"\1", plain)
    # `*emphasis*` only when the inner text is non-whitespace (cron expressions again).
    plain = re.sub(r"\*([^\s*][^*]*?[^\s*])\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", plain)
    plain = re.sub(r"~~([^~]+)~~", r"\1", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip("\n")


_WINDOWS_PATH_WITH_DOT_SEGMENT_RE = re.compile(r"(?i)(?:\b[a-z]:\\|\\\\)[^\s`]*\\\.[^\s`]*")


def _preserve_windows_dot_segments_for_markdown(text: str) -> str:
    r"""Double the ``\`` before hidden dirs in Windows paths: CommonMark reads ``\.`` as an escaped dot."""
    if "\\." not in text:
        return text

    def _protect(match: re.Match[str]) -> str:
        return re.sub(r"(?<!\\)\\(?=\.)", r"\\\\", match.group(0))

    return _WINDOWS_PATH_WITH_DOT_SEGMENT_RE.sub(_protect, text)


def _terminal_columns() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def _terminal_width_for_streaming() -> int:
    """Display cells inside the streamed response box (small margin for resize races)."""
    return max(20, _terminal_columns() - len(_STREAM_PAD) - 2)


def _render_final_assistant_content(text: str, mode: str = "render"):
    """Render final assistant content as markdown, stripped text, or raw text."""
    from rich.markdown import Markdown

    # 1 border cell each side + margin so resize races don't push a borderline table into soft-wrap.
    panel_width = max(20, _terminal_columns() - 4)

    normalized_mode = str(mode or "render").strip().lower()
    if normalized_mode == "strip":
        # Strip first (inline markdown changes cell width), then re-align padding.
        return _RichText(realign_markdown_tables(_strip_markdown_syntax(text), panel_width))
    if normalized_mode == "raw":
        return _rich_text_from_ansi(text or "")

    # Normalising under-padded tables up front gives narrow-panel fallbacks consistent input.
    plain = _rich_text_from_ansi(text or "").plain
    plain = _preserve_windows_dot_segments_for_markdown(plain)
    plain = realign_markdown_tables(plain, panel_width)
    return Markdown(plain)


def _post_stream_transform_output(response: str, result: dict | None) -> str:
    """Text still to display after a streamed response transform: the suffix, or the whole response when replaced."""
    if not result or not result.get("response_transformed"):
        return ""

    original = result.get("pre_transform_response") or ""
    if original and response.startswith(original):
        return response[len(original):]

    return f"\n[Response transformed after streaming]\n{response}"


_OUTPUT_HISTORY_ENABLED = True
_OUTPUT_HISTORY_REPLAYING = False
_OUTPUT_HISTORY_SUPPRESSED = False
_OUTPUT_HISTORY_MAX_LINES = 200
_OUTPUT_HISTORY = deque(maxlen=_OUTPUT_HISTORY_MAX_LINES)


def _coerce_output_history_limit(value) -> int:
    try:
        return max(10, int(value))
    except (TypeError, ValueError):
        return 200


def _configure_output_history(enabled: bool, max_lines=200) -> None:
    """Configure recent CLI output replayed after terminal redraws."""
    global _OUTPUT_HISTORY_ENABLED, _OUTPUT_HISTORY_MAX_LINES, _OUTPUT_HISTORY
    _OUTPUT_HISTORY_ENABLED = bool(enabled)
    _OUTPUT_HISTORY_MAX_LINES = _coerce_output_history_limit(max_lines)
    _OUTPUT_HISTORY = deque(maxlen=_OUTPUT_HISTORY_MAX_LINES)


def _clear_output_history() -> None:
    _OUTPUT_HISTORY.clear()


@contextmanager
def _suspend_output_history():
    global _OUTPUT_HISTORY_SUPPRESSED
    old_value = _OUTPUT_HISTORY_SUPPRESSED
    _OUTPUT_HISTORY_SUPPRESSED = True
    try:
        yield
    finally:
        _OUTPUT_HISTORY_SUPPRESSED = old_value


def _output_history_recording() -> bool:
    return _OUTPUT_HISTORY_ENABLED and not _OUTPUT_HISTORY_REPLAYING and not _OUTPUT_HISTORY_SUPPRESSED


def _record_output_history_entry(entry) -> None:
    if _output_history_recording():
        _OUTPUT_HISTORY.append(entry)


def _record_output_history(text: str) -> None:
    if _output_history_recording():
        _OUTPUT_HISTORY.extend(str(text).replace("\r", "").rstrip("\n").splitlines())


def _replay_output_history() -> None:
    """Repaint recent output above the prompt after a full screen clear."""
    global _OUTPUT_HISTORY_REPLAYING
    if not _OUTPUT_HISTORY_ENABLED or not _OUTPUT_HISTORY:
        return
    _OUTPUT_HISTORY_REPLAYING = True
    try:
        rendered_lines = []
        for entry in tuple(_OUTPUT_HISTORY):
            lines = [entry]
            if callable(entry):
                try:
                    lines = entry()
                except Exception:
                    continue
                if isinstance(lines, str):
                    lines = lines.splitlines()
            rendered_lines.extend(str(line) for line in lines)
        if rendered_lines:
            # One payload: per-line pt prints each force a sync redraw (a waterfall of old output).
            _pt_print(_PT_ANSI("\n".join(rendered_lines)))
    except Exception:
        pass
    finally:
        _OUTPUT_HISTORY_REPLAYING = False


def _pt_print_ansi(text: str) -> None:
    """``_pt_print(ANSI(text))``, falling back to ``print`` when stdout is not a real console."""
    try:
        _pt_print(_PT_ANSI(text))
    except Exception:
        # NoConsoleScreenBufferError (Windows) / OSError when stdout is e.g. a worker log file.
        with suppress(Exception):
            print(text)


def _cprint(text: str):
    """Print ANSI text through prompt_toolkit's renderer (patch_stdout swallows raw ANSI).

    From a background thread while an Application runs, a direct print races the input
    redraw and gets buried, so those go through ``run_in_terminal`` via ``call_soon_threadsafe``.
    """
    _record_output_history(text)

    try:
        from prompt_toolkit.application import get_app_or_none, run_in_terminal
    except Exception:
        _pt_print(_PT_ANSI(text))
        return

    try:
        app = get_app_or_none()
    except Exception:
        app = None

    if app is None or not getattr(app, "_is_running", False):
        _pt_print_ansi(text)
        return

    import asyncio as _asyncio

    try:
        loop = app.loop  # type: ignore[attr-defined]
    except Exception:
        loop = None
    try:
        # get_running_loop(): get_event_loop() warns from threads with no current loop.
        # Use get_running_loop() instead of get_event_loop() to avoid the DeprecationWarning /
        # RuntimeWarning emitted by Python 3.10+ when get_event_loop() is called from a thread that has no
        # current event loop set (e.g. the process_loop background thread). Fixes #19285.
        current_loop = _asyncio.get_running_loop()
    except Exception:
        current_loop = None
    if loop is None or (current_loop is loop and loop.is_running()):
        _pt_print(_PT_ANSI(text))
        return

    def _schedule():
        # run_in_terminal() returns an awaitable (pt >= 3.0) that must be scheduled or the
        # output is dropped, or None (mocks / older pt) when it already ran synchronously.
        # Never fall back to a bare print on error: the sync path already printed.
        with suppress(Exception):
            import inspect as _inspect
            coro = run_in_terminal(lambda: _pt_print(_PT_ANSI(text)))
            if coro is not None and (_inspect.isawaitable(coro) or _inspect.iscoroutine(coro)):
                _asyncio.ensure_future(coro)

    try:
        loop.call_soon_threadsafe(_schedule)
    except Exception:
        _pt_print_ansi(text)


def _prepend_note_to_message(message, note: str):
    """Prepend a one-shot note to a user message (str, or content-part list when an image is attached).

    For lists the note is folded into the first text part or inserted as a leading one.
    Unknown shapes are returned unchanged.
    """
    note = str(note or "").strip()
    if not note:
        return message
    if isinstance(message, str):
        return f"{note}\n\n{message}" if message else note
    if isinstance(message, list):
        parts = list(message)
        for i, part in enumerate(parts):
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                parts[i] = {**part, "text": f"{note}\n\n{text}" if text else note}
                return parts
        return [{"type": "text", "text": note}, *parts]
    return message


def _pt_app_is_running() -> bool:
    """Whether a prompt_toolkit Application currently owns the live terminal."""
    try:
        from prompt_toolkit.application import get_app_or_none
        app = get_app_or_none()
    except Exception:
        return False
    return app is not None and bool(getattr(app, "_is_running", False))


def _cli_visible_print(text: str = "") -> None:
    """``print`` unless a prompt_toolkit Application owns the terminal (patch_stdout swallows bare prints)."""
    if _pt_app_is_running():
        _cprint(text)
    else:
        print(text)


_IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.gif', '.webp',
    '.bmp', '.tiff', '.tif', '.svg', '.ico',
})


def _termux_example_image_path(filename: str = "cat.png") -> str:
    """Return a realistic example media path for the current Termux setup."""
    candidates = [
        os.path.expanduser("~/storage/shared"),
        "/sdcard",
        "/storage/emulated/0",
        "/storage/self/primary",
    ]
    # Literal "/" so the Android hint is right even on Windows.
    for root in candidates:
        if os.path.isdir(root):
            return f"{root}/Pictures/{filename}"
    return f"~/storage/shared/Pictures/{filename}"


def _split_path_input(raw: str) -> tuple[str, str]:
    r"""Split a leading path token (quoted or with ``\ `` escapes) from trailing free-form text."""
    raw = str(raw or "").strip()
    if not raw:
        return "", ""

    if raw[0] in {'"', "'"}:
        quote = raw[0]
        pos = 1
        while pos < len(raw):
            ch = raw[pos]
            if ch == '\\' and pos + 1 < len(raw):
                pos += 2
                continue
            if ch == quote:
                return raw[1:pos], raw[pos + 1 :].strip()
            pos += 1
        return raw[1:], ""

    pos = 0
    while pos < len(raw):
        ch = raw[pos]
        if ch == '\\' and pos + 1 < len(raw) and raw[pos + 1] == ' ':
            pos += 2
        elif ch == ' ':
            break
        else:
            pos += 1

    return raw[:pos].replace('\\ ', ' '), raw[pos:].strip()


def _resolve_attachment_path(raw_path: str) -> Path | None:
    """Resolve a user-supplied attachment path (quotes, ``~``, env vars, ``file://``; relative to TERMINAL_CWD).

    Returns ``None`` unless it resolves to an existing file.
    """
    token = str(raw_path or "").strip()
    if not token:
        return None

    if token[0] == token[-1] and token[0] in {'"', "'"}:
        token = token[1:-1].strip()
    token = token.replace('\\ ', ' ')
    if not token:
        return None

    expanded = token
    if token.startswith("file://"):
        try:
            parsed = urlparse(token)
            if parsed.scheme == "file":
                expanded = unquote(parsed.path or "")
                if parsed.netloc and os.name == "nt":
                    expanded = f"//{parsed.netloc}{expanded}"
                elif os.name == "nt" and len(expanded) >= 3 and expanded[0] == "/" and expanded[1].isalpha() and expanded[2] == ":":
                    # file:///C:/... parses to path "/C:/..." — drop the leading slash
                    # so it resolves as a drive-letter path.
                    expanded = expanded[1:]
        except Exception:
            expanded = token
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/" and normalized[0].isalpha():
            expanded = f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    path = Path(expanded)
    if not path.is_absolute():
        base_dir = Path(os.getenv("TERMINAL_CWD", os.getcwd()))
        path = base_dir / path

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    # ENAMETOOLONG for a pasted `/goal <long prose>` that passed the `/` prefilter
    # would otherwise reach process_loop and silently lose the input.
    try:
        if not resolved.exists() or not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved


def _file_drop_result(path: Path, remainder: str) -> dict:
    return {"path": path, "is_image": path.suffix.lower() in _IMAGE_EXTENSIONS, "remainder": remainder}


def _detect_file_drop(user_input: str) -> "dict | None":
    """Detect a dragged/pasted file path at the start of *user_input* -> ``{path, is_image, remainder}`` or None."""
    if not isinstance(user_input, str):
        return None

    stripped = user_input.strip()
    if not stripped:
        return None

    # Optionally quoted; then /, ~, ./, ../, a Windows drive prefix, or (unquoted) file://.
    quoted = stripped[:1] in {"'", '"'}
    unquoted = stripped[1:] if quoted else stripped
    starts_like_path = (
        unquoted.startswith(("/", "~", "./", "../"))
        or (not quoted and unquoted.startswith("file://"))
        or (len(unquoted) >= 3 and unquoted[1] == ":" and unquoted[2] in {"\\", "/"} and unquoted[0].isalpha())
    )
    if not starts_like_path:
        return None

    direct_path = _resolve_attachment_path(stripped)
    if direct_path is not None:
        return _file_drop_result(direct_path, "")

    first_token, remainder = _split_path_input(stripped)
    drop_path = _resolve_attachment_path(first_token)
    if drop_path is None and " " in stripped and not quoted:
        for pos in reversed([idx for idx, ch in enumerate(stripped) if ch == " "]):
            drop_path = _resolve_attachment_path(stripped[:pos].rstrip())
            if drop_path is not None:
                remainder = stripped[pos + 1 :].strip()
                break
    if drop_path is None:
        return None
    return _file_drop_result(drop_path, remainder)


def _format_image_attachment_badges(attached_images: list[Path], image_counter: int, width: int | None = None) -> str:
    """Attached-image badge row: compact summary on narrow terminals, per-image badges otherwise."""
    if not attached_images:
        return ""

    width = width or shutil.get_terminal_size((80, 24)).columns

    def _trunc(name: str, limit: int) -> str:
        return name if len(name) <= limit else name[: max(1, limit - 3)] + "..."

    if width < 52:
        if len(attached_images) == 1:
            return f"[📎 {_trunc(attached_images[0].name, 20)}]"
        return f"[📎 {len(attached_images)} images attached]"

    if width < 80:
        if len(attached_images) == 1:
            return f"[📎 {_trunc(attached_images[0].name, 32)}]"
        return f"[📎 {_trunc(attached_images[0].name, 20)}] [+{len(attached_images) - 1}]"

    base = image_counter - len(attached_images) + 1
    return " ".join(f"[📎 Image #{base + i}]" for i in range(len(attached_images)))


def _should_auto_attach_clipboard_image_on_paste(pasted_text: str) -> bool:
    """Auto-attach clipboard images only for image-only paste gestures."""
    return not pasted_text.strip()


_strip_leaked_bracketed_paste_wrappers = _lazy_shim(
    "hermes_cli.input_sanitize", "strip_leaked_bracketed_paste_wrappers", "_strip_leaked_bracketed_paste_wrappers"
)


def _hermes_call_output_screen_diff(
    orig_osd, app, output, screen, current_pos, color_depth, previous_screen, last_style, is_done, full_screen,
    attrs_for_style_string, style_string_has_style, size, previous_width,
):
    """prompt_toolkit ``_output_screen_diff`` with resize guards.

    Inflates ``previous_screen.height`` when the new screen is taller so pt skips the
    cursor move that stamps chrome into scrollback; on a corrupt previous paint buffer
    (tmux re-attach) retries once as a first paint instead of crashing the loop.

    1. 2. On AttributeError/TypeError from a corrupt previous paint buffer (classic after tmux attach with
    same width), retry once with ``previous_screen=None`` so pt first-paints cleanly instead of crashing the
    event loop with ``'cell' object has no attribute 'char'``. See #26137.
    """
    try:
        if previous_screen is not None and hasattr(previous_screen, "height") and previous_screen.height < screen.height:
            previous_screen.height = screen.height
    except Exception:
        pass

    common = (app, output, screen, current_pos, color_depth)
    tail = (is_done, full_screen, attrs_for_style_string, style_string_has_style, size)
    try:
        return orig_osd(*common, previous_screen, last_style, *tail, previous_width)
    except (AttributeError, TypeError):
        # Corrupt previous_screen / row cells after client reattach: previous_screen=None
        # takes the first-paint erase path, previous_width=0 treats the width as changed.
        return orig_osd(*common, None, None, *tail, 0)


def _apply_bracketed_paste_timeout_patch() -> None:
    """Patch ``Vt100Parser.feed`` to flush a bracketed paste whose ESC[201~ end mark never arrives.

    Without it a dropped end mark (SSH glitch, sleep/wake) freezes input forever. Idempotent.
    """
    try:
        import prompt_toolkit.input.vt100_parser as _vt100_mod
        from prompt_toolkit.keys import Keys as _PtKeys
        from prompt_toolkit.key_binding.key_processor import KeyPress as _PtKeyPress

        if getattr(_vt100_mod, "_hermes_bp_timeout_patched", False):
            return

        _BP_TIMEOUT_S = 2.0

        def _patched_vt100_feed(self_parser, data: str) -> None:
            if self_parser._in_bracketed_paste:
                self_parser._paste_buffer += data
                end_mark = "\x1b[201~"

                if end_mark in self_parser._paste_buffer:
                    end_index = self_parser._paste_buffer.index(end_mark)
                    paste_content = self_parser._paste_buffer[:end_index]
                    self_parser.feed_key_callback(_PtKeyPress(_PtKeys.BracketedPaste, paste_content))
                    self_parser._in_bracketed_paste = False
                    remaining = self_parser._paste_buffer[end_index + len(end_mark):]
                    self_parser._paste_buffer = ""
                    self_parser._hermes_bp_start = None
                    if remaining:
                        _patched_vt100_feed(self_parser, remaining)
                else:
                    bp_start = getattr(self_parser, "_hermes_bp_start", None)
                    now = time.monotonic()
                    if bp_start is None:
                        self_parser._hermes_bp_start = now
                    elif now - bp_start > _BP_TIMEOUT_S:
                        paste_content = self_parser._paste_buffer
                        self_parser._in_bracketed_paste = False
                        self_parser._paste_buffer = ""
                        self_parser._hermes_bp_start = None
                        if paste_content:
                            self_parser.feed_key_callback(_PtKeyPress(_PtKeys.BracketedPaste, paste_content))
                            logger.warning(
                                "Bracketed-paste timeout (%.1fs) — flushed %d bytes "
                                "without end mark. Terminal may have dropped ESC[201~ "
                                "(see #16263).",
                                now - bp_start, len(paste_content),
                            )
            else:
                # Re-inlined: calling the original would double-buffer after entering paste mode.
                for i, c in enumerate(data):
                    if self_parser._in_bracketed_paste:
                        _patched_vt100_feed(self_parser, data[i:])
                        break
                    self_parser._input_parser.send(c)

        _vt100_mod.Vt100Parser.feed = _patched_vt100_feed
        _vt100_mod._hermes_bp_timeout_patched = True
        logger.debug("Applied Vt100Parser bracketed-paste timeout patch (#16263)")
    except Exception as exc:  # noqa: BLE001 — defensive: never break startup
        logger.debug("Bracketed-paste timeout patch skipped: %s", exc)


# CPR replies (``ESC[<row>;<col>R``) can race past the input parser under resize storms
# and land as literal text; the ``^[[...R`` form appears when a filter stripped the ESC.
# Cursor Position Report (CPR / DSR) response, format ``ESC[<row>;<col>R``. prompt_toolkit's _on_resize() +
# renderer send ``ESC[6n`` queries to the terminal; under resize storms or tab switches the terminal's reply
# can race past the input parser and end up in the input buffer as literal text (see issue #14692). Also
# matches the visible-form ``^[[<row>;<col>R`` that appears when the ESC byte was stripped by a prior
# filter.
_DSR_CPR_ESC_RE = re.compile(r"\x1b\[\d+;\d+R")
_DSR_CPR_VISIBLE_RE = re.compile(r"\^\[\[\d+;\d+R")
_SGR_MOUSE_ESC_RE = re.compile(r"\x1b\[<\d+;\d+;\d+[Mm]")
_SGR_MOUSE_VISIBLE_RE = re.compile(r"\^\[\[<\d+;\d+;\d+[Mm]")
# Bare "<btn;col;rowM" fragments; deliberately broad, they are almost never intentional input.
_SGR_MOUSE_BARE_RE = re.compile(r"<\d+;\d+;\d+[Mm]")
_TERMINAL_INPUT_MODE_RESET_SEQ = (
    "\x1b[?1006l\x1b[?1003l\x1b[?1002l\x1b[?1000l"  # mouse: SGR, any-motion, button-motion, click
    "\x1b[?1004l"  # focus events
    "\x1b[?2004l"  # bracketed paste
    "\x1b[?1049l"  # leave alt screen
    "\x1b[<u"  # pop kitty keyboard mode
    "\x1b[>4m"  # reset modifyOtherKeys
    "\x1b[0m\x1b[?25h"  # reset attributes, show cursor
)
_KITTY_KEYBOARD_PUSH_SEQ = "\x1b[>1u"
_MODIFY_OTHER_KEYS_SEQ = "\x1b[>4;2m"
_EXTENDED_ENTER_KEYS_SEQ = _KITTY_KEYBOARD_PUSH_SEQ + _MODIFY_OTHER_KEYS_SEQ


_BACKSLASH_LINE_CONTINUATION_RE = re.compile(r"\\[ \t]*$")


def _is_ghostty_terminal(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the terminal is Ghostty (either detection path).

    Ghostty must be pushed ONLY modifyOtherKeys, not the Kitty keyboard
    protocol: its Kitty disambiguate-mode implementation strips the Alt
    modifier from the Backspace key, so Option+Backspace arrives as bare
    \\x7f instead of the CSI-u form ``\\x1b[127;3u`` the protocol calls for
    (upstream Ghostty bug), breaking backward-kill-word (#87630
    regression).  Ghostty implements modifyOtherKeys correctly (it then
    emits ``\\x1b[27;3;127~``, which the alias table also maps).

    Matches exactly the two conditions that admit Ghostty through
    ``_terminal_supports_extended_enter_keys``.
    """
    if env is None:
        env = os.environ
    return (
        (env.get("TERM_PROGRAM") or "").strip() == "ghostty"
        or (env.get("TERM") or "").strip().lower() == "xterm-ghostty"
    )


def _terminal_supports_extended_enter_keys(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether it is safe/useful to request modified Enter key reporting.

    Ghostty gets ONLY modifyOtherKeys: its Kitty disambiguate mode strips Alt from
    Backspace (upstream bug), breaking backward-kill-word.

    Ghostty implements modifyOtherKeys correctly (it then emits ``\\x1b[27;3;127~``, which the alias table
    also maps). See #87630.
    """
    env = os.environ if env is None else env
    return (env.get("TERM_PROGRAM") or "").strip() == "ghostty" or (env.get("TERM") or "").strip().lower() == "xterm-ghostty"


def _terminal_supports_extended_enter_keys(env: Optional[Mapping[str, str]] = None) -> bool:
    """Allowlist of terminals where requesting modified-Enter reporting is safe (aligned with the Ink TUI)."""
    env = os.environ if env is None else env
    term_program = (env.get("TERM_PROGRAM") or "").strip()
    term = (env.get("TERM") or "").strip().lower()
    return bool(
        env.get("WT_SESSION")
        or term_program in {"iTerm.app", "WezTerm", "ghostty", "vscode"}
        or env.get("KITTY_WINDOW_ID") or "kitty" in term
        or term == "xterm-ghostty"
        or term.startswith("tmux") or term_program.lower() == "tmux"
    )


def _enable_extended_enter_keys(output=None, env: Optional[Mapping[str, str]] = None) -> bool:
    """Ask allowlisted terminals to report modified keys distinctly.

    Writes the Kitty keyboard protocol push (CSI >1u, disambiguate mode) AND
    xterm modifyOtherKeys level 2 (CSI >4;2m), mirroring the Ink TUI —
    terminals honor whichever protocol they implement (except Ghostty, which
    gets only modifyOtherKeys; see the Ghostty exception below).  Both are
    needed:
    kitty-the-terminal removed modifyOtherKeys support entirely (it only
    speaks its own protocol), while tmux/VS Code only accept modifyOtherKeys.

    Under either protocol the terminal re-encodes modified keys as escape
    sequences — Kitty disambiguate mode as ``ESC[<codepoint>;<mod>u`` (plus
    the Esc key as ``ESC[27u``), modifyOtherKeys=2 as
    ``ESC[27;<mod>;<codepoint>~``.  Stock prompt_toolkit 3.x maps almost
    none of these, which is why the CSI >1u push was temporarily removed in
    #87074 (Ctrl+C arrived as ``ESC[99;5u`` and died, #56684).
    ``install_modify_other_keys_aliases()`` (called at CLI startup from
    ``hermes_cli.pt_input_extras``) now populates ``ANSI_SEQUENCES`` with the
    full Ctrl/Alt/Shift/multi-modifier and functional-key tables under BOTH
    formats, so every existing key binding continues to fire — including
    Ctrl+C, which is handled by prompt_toolkit's ``c-c`` binding (raw mode
    clears ISIG, so the kernel INTR path was never in play for the CLI).

    Ghostty exception: pushes only modifyOtherKeys — see
    ``_is_ghostty_terminal`` for the full rationale (#87630).

    The exit reset sequence pops/resets both modes, so this is safe across
    normal exits, Ctrl+C, and SIGTERM cleanup.
    """
    if not _terminal_supports_extended_enter_keys(env):
        return False
    # Ghostty exception: only modifyOtherKeys — see _is_ghostty_terminal.
    seq = _MODIFY_OTHER_KEYS_SEQ if _is_ghostty_terminal(env) else _EXTENDED_ENTER_KEYS_SEQ
    try:
        target = output
        if target is not None and hasattr(target, "write_raw"):
            target.write_raw(seq)
            target.flush()
            return True
        stream = sys.stdout
        if stream is not None and stream.isatty():
            stream.write(seq)
            stream.flush()
            return True
    except Exception:
        pass
    return False


def _cli_multiline_shortcuts_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """``display.cli_multiline_shortcuts`` (default on: Ctrl+J = newline; off restores the legacy c-j submit)."""
    if config is None:
        config = CLI_CONFIG
    display = config.get("display") if isinstance(config, dict) else None
    value = display.get("cli_multiline_shortcuts", True) if isinstance(display, dict) else True
    if isinstance(value, bool):
        return value
    return not (isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off", "disabled"})


def _is_backslash_line_continuation(text: str) -> bool:
    """True when Enter should turn a trailing backslash into a newline."""
    return bool(_BACKSLASH_LINE_CONTINUATION_RE.search(text or ""))


def _apply_backslash_line_continuation(text: str) -> str:
    """Replace a trailing ``\\`` marker with an actual newline."""
    return _BACKSLASH_LINE_CONTINUATION_RE.sub("", text or "") + "\n"


def _preserve_ctrl_enter_newline() -> bool:
    """Environments delivering Ctrl+Enter as bare LF (Windows Terminal, WSL, SSH, Ghostty): c-j must stay newline.

    See issue #22379.
    """
    env = os.environ
    if (
        sys.platform == "win32"
        or any(env.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "WT_SESSION",
                                    "GHOSTTY_RESOURCES_DIR", "GHOSTTY_BIN_DIR"))
        or env.get("TERM", "").lower() == "xterm-ghostty" or env.get("TERM_PROGRAM", "").lower() == "ghostty"
        or "microsoft" in env.get("WSL_DISTRO_NAME", "").lower()
    ):
        return True
    # WSL env vars can be scrubbed under sudo; also peek /proc.
    for p in ("/proc/version", "/proc/sys/kernel/osrelease"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if "microsoft" in f.read().lower():
                    return True
        except OSError:
            continue
    return False


def _bind_prompt_submit_keys(kb, handler, *, multiline_shortcuts_enabled: Optional[bool] = None) -> None:
    """Enter always submits; c-j submits only with multiline shortcuts off AND where Ctrl+Enter isn't c-j.

    Even when the setting is disabled, environments where Ctrl+Enter is known to arrive as c-j (Windows,
    WSL, SSH, Windows Terminal, Ghostty) keep c-j reserved for newline; otherwise Ctrl+Enter submits instead
    of composing. See _preserve_ctrl_enter_newline() and issue #22379.
    """
    if multiline_shortcuts_enabled is None:
        multiline_shortcuts_enabled = _cli_multiline_shortcuts_enabled()
    kb.add("enter")(handler)
    if sys.platform != "win32" and not multiline_shortcuts_enabled and not _preserve_ctrl_enter_newline():
        kb.add("c-j")(handler)


def _disable_prompt_toolkit_cpr_warning(app) -> None:
    """Let prompt_toolkit fall back from CPR without printing into the prompt."""
    with suppress(Exception):
        app.renderer.cpr_not_supported_callback = None


def _terminal_may_leak_cpr() -> bool:
    """Suppress prompt_toolkit CPR queries (delayed replies leak into input); Windows keeps pt's default.

    Delayed CPR replies (``ESC[<row>;<col>R`` / visible ``^[[<row>;<col>R``) leak into the status line and
    can freeze input when the reply is slow (#13870 on SSH/slow PTYs). The same race hits local POSIX TTYs
    under heavy subagent / status-line load — see ``tests/hermes_cli/test_cpr_local_leak.py``.
    """
    return os.environ.get("PROMPT_TOOLKIT_NO_CPR", "") == "1" or sys.platform != "win32"


def _build_cpr_disabled_output(stdout):
    """Vt100_Output with ``enable_cpr=False`` (``from_pty()`` doesn't expose it), or None on failure.

    prompt_toolkit's renderer sends ``ESC[6n`` (Device Status Report) to learn the cursor row before
    painting in non-fullscreen mode; the terminal replies ``ESC[<row>;<col>R``. When that reply is delayed
    it races into the display as raw ``^[[39;1R`` and can stall the renderer's pending-CPR future (#13870;
    also local POSIX under heavy subagent load).
    """
    try:
        import io as _io
        from prompt_toolkit.output.vt100 import Vt100_Output, _get_size
        from prompt_toolkit.data_structures import Size

        def _get_term_size():
            rows = columns = None
            try:
                rows, columns = _get_size(stdout.fileno())
            except (OSError, _io.UnsupportedOperation, AttributeError, ValueError):
                pass
            return Size(rows=rows or 24, columns=columns or 80)

        return Vt100_Output(stdout, _get_term_size, enable_cpr=False)
    except Exception:
        return None


def _select_classic_cli_pt_output(stdout):
    """CPR-disabled ``Vt100_Output`` when CPR may leak, else None (Application keeps pt's default)."""
    return _build_cpr_disabled_output(stdout) if _terminal_may_leak_cpr() else None


def _strip_leaked_terminal_responses_with_meta(text: str) -> tuple[str, bool]:
    """Strip leaked CPR replies and mouse-report fragments -> ``(cleaned, had_mouse_reports)``."""
    if not text:
        return text, False

    had_mouse_reports = False
    for present, cpr_re, mouse_re in (
        ("\x1b[" in text, _DSR_CPR_ESC_RE, _SGR_MOUSE_ESC_RE),
        ("^[" in text, _DSR_CPR_VISIBLE_RE, _SGR_MOUSE_VISIBLE_RE),
        ("<" in text and ";" in text and ("M" in text or "m" in text), None, _SGR_MOUSE_BARE_RE),
    ):
        if not present:
            continue
        if cpr_re is not None:
            text = cpr_re.sub("", text)
        text, count = mouse_re.subn("", text)
        had_mouse_reports = had_mouse_reports or count > 0
    return text, had_mouse_reports


def _estimate_tui_input_height(
    lines: list[str] | tuple[str, ...], prompt_text: str, terminal_columns: int, *, max_height: int = 8,
) -> int:
    """Input rows from live terminal cells; the BeforeInput prompt consumes cells only on line 0.

    Never substitute a fake wide fallback: a mis-sized TextArea leaves stale cells at the bottom.
    """
    try:
        from prompt_toolkit.utils import get_cwidth
    except Exception:
        get_cwidth = lambda value: len(value or "")  # type: ignore[assignment]

    columns = max(1, _int_or(terminal_columns or 0, 0))
    prompt_width = max(0, get_cwidth(prompt_text or ""))

    visual_lines = 0
    for index, line in enumerate(lines or [""]):
        display_width = get_cwidth(line or "") + (prompt_width if index == 0 else 0)
        visual_lines += max(1, -(-display_width // columns))

    return min(max(visual_lines, 1), max(1, int(max_height or 1)))


def _status_bar_visible_from_display_config(display_config: object) -> bool:
    """Initial status-bar visibility; both YAML ``off`` (False) and strings like ``"hidden"`` mean off."""
    if not isinstance(display_config, dict):
        display_config = {}
    statusbar_config = display_config.get("statusbar", display_config.get("tui_statusbar", "top"))
    if isinstance(statusbar_config, str):
        return statusbar_config.strip().lower() not in {"0", "false", "hidden", "no", "off"}
    return statusbar_config is not False


def _collect_query_images(query: str | None, image_arg: str | None = None) -> tuple[str, list[Path]]:
    """Collect local image attachments for single-query CLI flows."""
    message = query or ""
    images: list[Path] = []

    if isinstance(message, str):
        dropped = _detect_file_drop(message)
        if dropped and dropped.get("is_image"):
            images.append(dropped["path"])
            message = dropped["remainder"] or f"[User attached image: {dropped['path'].name}]"

    if image_arg:
        explicit_path = _resolve_attachment_path(image_arg)
        if explicit_path is None:
            raise ValueError(f"Image file not found: {image_arg}")
        if explicit_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Not a supported image file: {explicit_path}")
        images.append(explicit_path)

    return message, list(dict.fromkeys(images))


# OSC sequences (e.g. OSC-8 links): pt's ANSI parser strips the ESC but leaks the payload as text.
_OSC_ESCAPE_RE = re.compile(r"\x1b\][\s\S]*?(?:\x07|\x1b\\)")


class ChatConsole:
    """Rich Console drop-in routing rendered ANSI through ``_cprint`` so colors survive patch_stdout."""

    def __init__(self):
        from io import StringIO
        self._buffer = StringIO()
        self._inner = Console(file=self._buffer, force_terminal=True, color_system="truecolor", highlight=False)

    def print(self, *args, **kwargs):
        self._buffer.seek(0)
        self._buffer.truncate()
        self._inner.width = shutil.get_terminal_size((80, 24)).columns
        self._inner.print(*args, **kwargs)
        for line in _OSC_ESCAPE_RE.sub("", self._buffer.getvalue()).rstrip("\n").split("\n"):
            _cprint(line)

    @contextmanager
    def status(self, *_args, **_kwargs):
        """No-op ``console.status`` so slash helpers don't duplicate ``_busy_command()``'s indicator."""
        yield self



def _build_compact_banner() -> str:
    """Build a compact banner that fits the current terminal width."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        _skin = get_active_skin()
    except Exception:
        _skin = None

    def _color(key, default):
        return _skin.get_color(key, default) if _skin else default

    border_color = _color("banner_border", "#FFD700")
    title_color = _color("banner_title", "#FFBF00")
    dim_color = _color("banner_dim", "#B8860B")

    if (getattr(_skin, "name", "default") if _skin else "default") == "default":
        tiny_line = "☤ NOUS HERMES"
    else:
        tiny_line = _skin.get_branding("agent_name", "Hermes Agent") if _skin else "Hermes Agent"
    line1 = f"{tiny_line} - AI Agent Framework"

    if os.environ.get("HERMES_FAST_STARTUP_BANNER") == "1":
        from hermes_cli import __release_date__ as _release_date
        from hermes_cli import __version__ as _version

        version_line = f"Hermes Agent v{_version} ({_release_date})"
    else:
        version_line = format_banner_version_label()

    w = min(shutil.get_terminal_size().columns - 2, 88)
    if w < 30:
        return f"\n[{title_color}]{tiny_line}[/] [dim {dim_color}]- Nous Research[/]\n"

    inner = w - 2  # inside the box border
    bar = "═" * w
    content_width = inner - 2

    line1 = line1[:content_width].ljust(content_width)
    line2 = version_line[:content_width].ljust(content_width)

    return (
        f"\n[bold {border_color}]╔{bar}╗[/]\n"
        f"[bold {border_color}]║[/] [{title_color}]{line1}[/] [bold {border_color}]║[/]\n"
        f"[bold {border_color}]║[/] [dim {dim_color}]{line2}[/] [bold {border_color}]║[/]\n"
        f"[bold {border_color}]╚{bar}╝[/]\n"
    )


def _looks_like_slash_command(text: str) -> bool:
    """``/help`` yes, ``/Users/x/file.md`` no: a command's first word has no further ``/``."""
    if not text or not text.startswith("/"):
        return False
    return "/" not in text.split()[0][1:]


_skill_commands = None
_skill_bundles = None


def _slash_args(cmd: str) -> str:
    """Text after the slash-command word, stripped ("" when absent)."""
    parts = cmd.split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _ensure_skill_commands() -> dict:
    global _skill_commands
    if _skill_commands is None:
        from agent.skill_commands import scan_skill_commands

        _skill_commands = scan_skill_commands()
    return _skill_commands


def get_skill_commands() -> dict:
    return _ensure_skill_commands()


build_skill_invocation_message = _lazy_shim("agent.skill_commands", "build_skill_invocation_message")
build_preloaded_skills_prompt = _lazy_shim("agent.skill_commands", "build_preloaded_skills_prompt")


def get_skill_bundles() -> dict:
    global _skill_bundles
    if _skill_bundles is None:
        from agent.skill_bundles import get_skill_bundles as _impl

        _skill_bundles = _impl()
    return _skill_bundles


build_bundle_invocation_message = _lazy_shim("agent.skill_bundles", "build_bundle_invocation_message")


def _get_plugin_cmd_handler_names() -> set:
    """Return plugin command names (without slash prefix) for dispatch matching."""
    try:
        from hermes_cli.plugins import get_plugin_commands
        return set(get_plugin_commands().keys())
    except Exception:
        return set()


def _parse_skills_argument(skills: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a CLI skills flag into a deduplicated list of skill identifiers."""
    if not skills:
        return []
    raw_values = [str(item) for item in skills if item is not None] if isinstance(skills, (list, tuple)) else [str(skills)]
    parts = (p.strip() for raw in raw_values for p in raw.split(","))
    return list(dict.fromkeys(p for p in parts if p))


def save_config_value(key_path: str, value: any) -> bool:
    """Persist dot-separated ``key_path`` = value into HERMES_HOME/config.yaml; True on success.

    Never the repo's cli-config.yaml: no config reader loads it, so the value would vanish.
    """
    config_path = get_hermes_home() / 'config.yaml'

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        from utils import atomic_roundtrip_yaml_update
        atomic_roundtrip_yaml_update(config_path, key_path, value)
        try:  # owner-only: config files contain API keys
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass
        # Same unpinned-cron notice as `hermes config set` for every model switch.
        from hermes_cli.config import warn_unpinned_cron_jobs_after_model_config_change

        warn_unpinned_cron_jobs_after_model_config_change(key_path, value)
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
        return False


def _normalize_moa_model(model: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """``moa:<preset>`` -> ``("moa", preset)`` (same routing as ``/moa``); anything else -> ``(None, model)``.

    Returns ``("moa", "<preset>")`` when *model* selects the MoA virtual provider, otherwise ``(None,
    model)`` unchanged. This gives non-interactive ``hermes chat -Q -m moa:<preset>`` the same routing the
    interactive ``/moa`` command and the model picker already use: ``resolve_runtime_provider`` handles
    ``requested_provider == "moa"`` and ``agent_init`` builds the MoAClient off ``provider == "moa"``.
    Without this the raw ``moa:<preset>`` string is sent to the real provider and rejected with a 401/400
    "model not supported" (#56828).
    """
    if isinstance(model, str) and model.strip().lower().startswith("moa:"):
        preset = model.strip().split(":", 1)[1].strip()
        if preset:
            return "moa", preset
    return None, model

_split_model_config_default = _lazy_shim("hermes_cli.config", "split_model_config_default", "_split_model_config_default")


class _VoiceInputMessage:
    """Sentinel for voice-transcribed input so the concise voice prefix never applies to typed text.

    Distinguishes STT output from manually typed text while voice mode is active, so the
    concise-voice-response prefix is applied only to messages that actually came from the microphone
    (#65827).
    """

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


class _SeededQueryMessage:
    """Sentinel for a ``-q`` prompt seeded into an interactive session; treated LITERALLY (no slash/!/file-drop)."""

    __slots__ = ("text", "images")

    def __init__(self, text: str, images=None):
        self.text = text or ""
        self.images = list(images or [])

    def __str__(self) -> str:
        return self.text


def _should_seed_interactive(query, image, quiet: bool, oneshot: bool) -> bool:
    """``-q`` seeds an interactive session only on a real TTY without ``--oneshot``/``-Q`` (automation answers and exits)."""
    if not (query or image) or oneshot or quiet:
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _panel_box_width(title: str, content_lines: list[str], min_width: int = 46, max_width: int = 76) -> int:
    """Stable TUI panel width wide enough for the title and content (incl. borders)."""
    term_cols = shutil.get_terminal_size((100, 20)).columns
    longest = max([len(title)] + [len(line) for line in content_lines] + [min_width - 4])
    inner = min(max(longest + 4, min_width - 2), max_width - 2, max(24, term_cols - 6))
    return inner + 2  # leading/trailing space inside the borders


def _wrap_panel_text(text: str, width: int, subsequent_indent: str = "", *, keep_ws: bool = False) -> list[str]:
    """Wrap panel text; ``keep_ws`` preserves whitespace (command/detail previews)."""
    kw = dict(replace_whitespace=False, drop_whitespace=False) if keep_ws else dict(break_long_words=False, break_on_hyphens=False)
    wrapped = textwrap.wrap(text, width=max(8, width), subsequent_indent=subsequent_indent, **kw)
    return wrapped or [""]


_wrap_panel_text_keep_ws = functools.partial(_wrap_panel_text, keep_ws=True)


def _append_panel_line(lines, border_style: str, content_style: str, text: str, box_width: int) -> None:
    lines.extend(((border_style, "│ "), (content_style, text.ljust(max(0, box_width - 2))), (border_style, " │\n")))


def _append_blank_panel_line(lines, border_style: str, box_width: int) -> None:
    lines.append((border_style, "│" + (" " * box_width) + "│\n"))


@dataclass
class _ChatTurn:
    """Per-turn state shared by the ``chat()`` phases and the agent worker thread.

    ``result`` is written by the worker and read after the join; ``tts_normal_exit`` is
    set only when the TTS worker drained on its own so the last sentence is never cut.
    """

    result: Optional[dict] = None
    use_streaming_tts: bool = False
    box_opened: bool = False
    thinking_started: bool = False
    text_queue: Optional[queue.Queue] = None
    tts_thread: Optional[threading.Thread] = None
    stream_callback: Optional[Any] = None
    stop_event: Optional[threading.Event] = None
    tts_normal_exit: bool = False
    voice_prefix: str = ""
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin


_PASTE_REF_RE = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')


class HermesCLI(CLIProcessNotificationsMixin, CLIAgentSetupMixin, CLICommandsMixin, CLIBillingMixin, CLITuiMixin, CLIStatusBarMixin, CLIVoiceMixin, CLIModelSwitchMixin, CLISessionMixin, CLIStreamMixin, CLIModalMixin, CLITerminalMixin, CLIInfoMixin, CLILoopsMixin, CLIChatTurnMixin):
    """Interactive REPL for the Hermes Agent."""

    # Seeded -q first message (see _should_seed_interactive); run() re-creates
    # _pending_input, so it is enqueued only after the fresh queue exists.
    _seeded_first_message: Optional["_SeededQueryMessage"] = None
    # Inspection surfaces (banner, /tools, status line) read this on partially built instances too.
    disabled_toolsets: Optional[List[str]] = None

    def __init__(
        self,
        model: str = None,
        toolsets: List[str] = None,
        provider: str = None,
        reasoning: str = None,
        api_key: str = None,
        base_url: str = None,
        max_turns: int = None,
        run_budget: float = None,
        verbose: Optional[bool] = None,
        compact: bool = False,
        resume: str = None,
        checkpoints: bool = False,
        pass_session_id: bool = False,
        ignore_rules: bool = False,
    ):
        """CLI args win over config; ``reasoning`` is per-run only; ``resume`` restores history from SQLite."""
        self._init_display_options(verbose, compact)
        self._init_model_routing(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget,
                                 checkpoints, pass_session_id, ignore_rules)
        self._init_runtime_state(resume)

    def _init_display_options(self, verbose, compact):
        """Display-related config: compact/tool-progress/focus view, bells, streaming, previews, stream buffers."""
        self.console = Console()
        self.config = CLI_CONFIG
        display = CLI_CONFIG["display"]
        self.compact = compact if compact is not None else display.get("compact", False)
        # tool_progress: "off" | "new" | "all" | "verbose"; YAML 1.1 parses bare `off` as False.
        _raw_tp = display.get("tool_progress", "all")
        self.tool_progress_mode = "off" if _raw_tp is False else str(_raw_tp)
        # focus_view (/focus) is display-only: snaps tool_progress to "off" (stashing the
        # pre-focus mode for /focus off); never changes what is sent to the model.
        self._focus_view_enabled = bool(display.get("focus_view", False))
        self._focus_saved_tool_progress = self._focus_last_counted_tool = None
        self._focus_hidden_lines = 0
        if self._focus_view_enabled:
            from hermes_cli.focus_view import FOCUS_TOOL_PROGRESS_MODE, normalize_tool_progress_mode

            self._focus_saved_tool_progress = normalize_tool_progress_mode(self.tool_progress_mode)
            self.tool_progress_mode = FOCUS_TOOL_PROGRESS_MODE
        self.resume_display = display.get("resume_display", "full")  # "full" | "minimal"
        self.bell_on_complete = display.get("bell_on_complete", False)
        self.bell_on_prompt = display.get("bell_on_prompt", False)  # bell when a blocking modal opens
        self.show_reasoning = display.get("show_reasoning", True)
        self.reasoning_full = display.get("reasoning_full", False)
        _configure_output_history(
            enabled=display.get("persistent_output", True),
            max_lines=display.get("persistent_output_max_lines", 200),
        )
        # busy_input_mode: "interrupt" (redirect the run) | "queue" (next turn) | "steer" (inject mid-run).
        _bim = str(display.get("busy_input_mode", "interrupt")).strip().lower()
        self.busy_input_mode = _bim if _bim in ("queue", "steer") else "interrupt"

        # verbose ONLY controls global DEBUG logging; tool_progress="verbose" is independent
        # (coupling them spewed every module's DEBUG logs to the console).
        self.verbose = bool(verbose) if verbose is not None else False

        self.streaming_enabled = display.get("streaming", False)
        self.show_timestamps = display.get("timestamps", False)
        self.timestamp_format = display.get("timestamp_format", "%H:%M")
        _frm = str(display.get("final_response_markdown", "strip")).strip().lower()
        self.final_response_markdown = _frm if _frm in {"render", "strip", "raw"} else "strip"

        self._inline_diffs_enabled = display.get("inline_diffs", True)

        # Per-turn accounting: CLI-only chrome riding the tool-progress feed.
        self._turn_summary_enabled = bool(display.get("turn_summary", True))
        self._spinner_token_flow_enabled = bool(display.get("spinner_token_flow", True))
        self._turn_summary_collector = None
        self._turn_summary_start = 0.0
        self._turn_token_baseline = 0
        self._interactive_turn = False  # only run()-loop turns; keeps the summary line off -Q

        _ump = display.get("user_message_preview", {})
        _ump = _ump if isinstance(_ump, dict) else {}
        self.user_message_preview_first_lines = max(1, _int_or(_ump.get("first_lines", 2), 2))
        self.user_message_preview_last_lines = max(0, _int_or(_ump.get("last_lines", 2), 2))

        # Streaming display state
        self._stream_buf = ""  # partial line buffer
        self._reasoning_preview_buf = ""  # coalesces tiny reasoning chunks
        self._stream_started = self._stream_box_opened = self._stream_box_live = False
        self._held_status_lines: list[str] = []  # agent status lines parked while a box streams
        # Possible markdown-table lines held until the block ends for wcwidth-aware re-padding.
        self._stream_table_buf: list[str] = []
        self._in_stream_table = False
        self._pending_edit_snapshots = {}
        self._last_input_mode_recovery = self._last_termios_drift_check = None  # None = never; monotonic epoch is arbitrary
        self._input_mode_recovery_notice_shown = self._termios_drift_notice_shown = False

    def _init_model_routing(self, model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget, checkpoints, pass_session_id, ignore_rules):
        """Resolve model/provider/base_url, turn limits, toolsets, checkpoints, prompt/personality, reasoning + routing config."""
        self._init_model_and_provider(model, provider, api_key, base_url)
        self._init_turn_limits(max_turns, run_budget)
        self._init_toolsets(toolsets)
        self._init_checkpoints_and_rules(checkpoints, pass_session_id, ignore_rules)
        self._init_prompt_and_reasoning(reasoning)

    def _init_model_and_provider(self, model, provider, api_key, base_url):
        """Priority: CLI args > env vars > config file."""
        # LLM_MODEL/OPENAI_MODEL env vars are deliberately NOT checked (multi-agent setups
        # would stomp each other through the environment).
        _model_config = CLI_CONFIG["model"]
        # A dict-valued default carries its own provider, which must feed requested_provider
        # instead of being replaced by the merged model.provider (typically "auto").
        _config_model, _nested_provider = _split_model_config_default(
            _model_config.get("default") or _model_config.get("model") or ""
        )
        # resume must not clobber an explicit -m with the session's stored model.
        self._explicit_model_override = bool(model)
        self.model = model or _config_model or ""
        _cfg_provider = _model_config.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER")
        _startup_provider_override = _startup_base_url_override = _startup_api_key_override = ""
        if self.model:
            from hermes_cli.model_switch import resolve_startup_model_route

            _startup_route = resolve_startup_model_route(
                self.model,
                explicit_provider=provider or "",
                current_provider=(provider or _nested_provider or _cfg_provider or ""),
                user_providers=CLI_CONFIG.get("providers"),
                custom_providers=CLI_CONFIG.get("custom_providers"),
            )
            if _startup_route is not None:
                self.model = _startup_route.model
                _startup_provider_override = _startup_route.provider
                _startup_base_url_override = _startup_route.base_url
                _startup_api_key_override = _startup_route.api_key
        # ``moa:<preset>`` selects the MoA virtual provider before provider resolution so the
        # real provider never sees the unknown model; the prefix wins over --provider.
        # A ``moa:<preset>`` model string selects the MoA virtual provider in one shot (parity with
        # interactive ``/moa`` and the model picker). See #56828.
        _moa_provider_override, self.model = _normalize_moa_model(self.model)

        if self.model == "":  # auto-detect from a local server
            _base_url = _model_config.get("base_url") or ""
            if base_url_hostname(_base_url) in ("localhost", "127.0.0.1"):
                from hermes_cli.runtime_provider import _auto_detect_local_model
                self.model = _auto_detect_local_model(_base_url) or self.model
        # Provider normalisation may silently override the default but must warn for an
        # explicit choice (a config model equal to the global fallback is NOT explicit).
        self._model_is_default = not model and not _config_model

        # --api-key wins; otherwise a URL-bearing startup alias carries its own credential.
        # See #28660.
        self._explicit_api_key = api_key or _startup_api_key_override or None
        self._explicit_base_url = base_url

        # Resolved lazily at use-time via _ensure_runtime_credentials().
        self.requested_provider = (
            _moa_provider_override or provider or _startup_provider_override or _nested_provider
            or _cfg_provider or "auto"
        )
        # `--provider <custom>` without `-m` uses that entry's default_model, else the global
        # default goes to the custom endpoint and the compressor gets the wrong context length.
        # Explicit `-m` still wins. See #86978.
        if not model and provider:
            try:
                from hermes_cli.runtime_provider import _get_named_custom_provider

                _named_custom = _get_named_custom_provider(provider)
            except Exception as exc:
                logger.warning(
                    "Could not resolve --provider %s default model; keeping global model.default (%s)",
                    provider, exc,
                )
                _named_custom = None
            _provider_default = str((_named_custom or {}).get("model") or "").strip()
            if _provider_default:
                self.model = _provider_default
                self._model_is_default = False
        self._provider_source: Optional[str] = None
        self.provider = self.requested_provider
        self.api_mode = "chat_completions"
        self.acp_command: Optional[str] = None
        self.acp_args: list[str] = []
        self.base_url = (
            base_url or _startup_base_url_override or _model_config.get("base_url", "")
            or os.getenv("OPENROUTER_BASE_URL", "")
        ) or None
        # Match key to resolved base_url: OpenRouter URL → prefer OPENROUTER_API_KEY,
        # custom endpoint → prefer OPENAI_API_KEY (issue #560).
        # Note: _ensure_runtime_credentials() re-resolves this before first use.
        if self.base_url and base_url_host_matches(self.base_url, "openrouter.ai"):
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        # Max turns priority: CLI arg > config file > env var > default
        # All paths go through resolve_turn_limit() so that agent.max_turns
        # accepts "none"/"unlimited" (→ sys.maxsize) in addition to ints.
        # See hermes_cli.config.resolve_turn_limit for the full spelling table.
        from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
        if max_turns is not None:  # CLI arg was explicitly set
            self.max_turns = _resolve_turn_limit(max_turns)
        elif CLI_CONFIG["agent"].get("max_turns") is not None:
            self.max_turns = _resolve_turn_limit(CLI_CONFIG["agent"]["max_turns"])
        elif CLI_CONFIG.get("max_turns") is not None:  # Backwards compat: root-level max_turns
            # KEEP (evaluated for the v12 support-floor cleanup, July 2026):
            # no versioned config migration ever rewrote root-level max_turns
            # to agent.max_turns on disk — only load-time normalization
            # (_normalize_max_turns_config) folds it, and configs read through
            # other paths may bypass it. This fallback is therefore the only
            # safety net for configs that still carry the root key.
            self.max_turns = _resolve_turn_limit(CLI_CONFIG["max_turns"])
        else:
            # Env var bridge (set by gateway/run.py from config.yaml, or by the
            # user directly). Empty/unset → default (unlimited).
            self.max_turns = _resolve_turn_limit(os.getenv("HERMES_MAX_ITERATIONS"))

        # Wall-clock run budget: CLI flag wins over config; both optional.
        # None keeps the feature fully off (AIAgent stays dormant).
        if run_budget is not None:
            self.run_budget_seconds = run_budget
        else:
            self.run_budget_seconds = CLI_CONFIG["agent"].get("run_budget_seconds")

        # Parse and validate toolsets
        self.enabled_toolsets = toolsets
        from agent.skill_utils import parse_config_string_list

        self.disabled_toolsets = parse_config_string_list(CLI_CONFIG["agent"].get("disabled_toolsets"))

        if toolsets and "all" not in toolsets and "*" not in toolsets:
            # MCP server names only resolve after discover_mcp_tools runs; skip them here.
            mcp_names = set((CLI_CONFIG.get("mcp_servers") or {}).keys())
            invalid = [t for t in toolsets if not validate_toolset(t) and t not in mcp_names]
            if invalid:
                self._console_print(f"[bold red]Warning: Unknown toolsets: {', '.join(invalid)}[/]")

    def _init_checkpoints_and_rules(self, checkpoints, pass_session_id, ignore_rules):
        cp_cfg = CLI_CONFIG.get("checkpoints", {})
        if isinstance(cp_cfg, bool):
            cp_cfg = {"enabled": cp_cfg}
        self.checkpoints_enabled = checkpoints or cp_cfg.get("enabled", False)
        self.checkpoint_max_snapshots = cp_cfg.get("max_snapshots", 20)
        self.checkpoint_max_total_size_mb = cp_cfg.get("max_total_size_mb", 500)
        self.checkpoint_max_file_size_mb = cp_cfg.get("max_file_size_mb", 10)
        self.pass_session_id = pass_session_id
        # --ignore-rules: AIAgent skips context files (AGENTS.md/SOUL.md/...) and memory.
        self.ignore_rules = ignore_rules or is_truthy_value(os.environ.get("HERMES_IGNORE_RULES"))

    def _init_prompt_and_reasoning(self, reasoning):
        """Ephemeral system prompt/prefill, reasoning + service tier, OpenRouter routing knobs, fallback chain."""
        # Env var wins, then hermes_cli.personality (single owner of overlay resolution).
        from hermes_cli.personality import available_personalities, resolve_ephemeral_system_prompt

        self.system_prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "") or resolve_ephemeral_system_prompt(CLI_CONFIG)
        self.personalities = available_personalities(CLI_CONFIG)

        self.prefill_messages = _load_prefill_messages(_resolve_prefill_messages_file(CLI_CONFIG))

        # Per-model override > global reasoning_effort.
        # Reasoning config (OpenRouter reasoning effort level) Per-model override > global reasoning_effort
        # — resolved through the shared chokepoint in hermes_constants (Closes #21256).
        from hermes_constants import resolve_reasoning_config
        self.reasoning_config = resolve_reasoning_config(CLI_CONFIG, self.model)
        # --reasoning wins for this run only (never persisted); unparseable -> warn and ignore.
        if reasoning is not None and str(reasoning).strip():
            _cli_reasoning = _parse_reasoning_config(reasoning)
            if _cli_reasoning is None:
                logger.warning("Unknown --reasoning '%s', keeping the configured level", reasoning)
            else:
                self.reasoning_config = _cli_reasoning
        self.service_tier = _parse_service_tier_config(CLI_CONFIG["agent"].get("service_tier", ""))

        pr = CLI_CONFIG.get("provider_routing", {}) or {}
        self._provider_sort = pr.get("sort")
        self._providers_only = pr.get("only")
        self._providers_ignore = pr.get("ignore")
        self._providers_order = pr.get("order")
        self._provider_require_params = pr.get("require_parameters", False)
        self._provider_data_collection = pr.get("data_collection")

        # OpenRouter Pareto Code router coding-score floor; out-of-range = unset.
        _raw_score = (CLI_CONFIG.get("openrouter", {}) or {}).get("min_coding_score")
        self._openrouter_min_coding_score: Optional[float] = None
        if _raw_score not in {None, ""}:
            try:
                _f = float(_raw_score)
                if 0.0 <= _f <= 1.0:
                    self._openrouter_min_coding_score = _f
            except (TypeError, ValueError):
                pass

        self._fallback_model = get_fallback_chain(CLI_CONFIG)

    def _init_runtime_state(self, resume):
        """Session store + all per-run mutable state (queues, overlays, pet/voice/status-bar fields)."""
        # A signature change across turns (/model, credential rotation) rebuilds the agent.
        self._active_agent_route_signature = None
        self.agent: Optional[Any] = None  # initialized on first use
        self._tool_callbacks_installed = self._tirith_security_checked = False
        self._app = None  # prompt_toolkit Application (set in run())

        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        # Per-prompt elapsed timer shown in the status bar.
        self._prompt_start_time: Optional[float] = None
        self._prompt_duration: float = 0.0
        self._last_turn_finished_at: Optional[float] = None
        self._init_session_store()
        self._pending_title: Optional[str] = None
        self._resumed = bool(resume)
        self.session_id = resume or new_session_id(self.session_start)
        getattr(self, "_write_terminal_breadcrumb", lambda: None)()

        self._history_file = _hermes_home / ".hermes_history"
        self._last_invalidate: float | None = None  # throttles UI repaints (None = never; monotonic epoch is arbitrary)
        self._init_ui_state()

    def _init_session_store(self):
        """Open the session store early (so /title works before the first message) + opportunistic maintenance."""
        self._session_db = None
        self._session_db_unavailable = False
        try:
            # Registry handle, not a bare SessionDB(): goals/loops/heartbeat acquire the same
            # path a moment later from the REPL thread, and a second writer repeats the full
            # open (the /proc-wide deleted-WAL scan, ~4k readlinks) while the render thread
            # holds the GIL — that repeat was the post-banner freeze before the first prompt.
            from hermes_state_registry import acquire
            self._session_db = acquire()
        except Exception as e:
            # Without a store the transcript is NOT persisted while the chat looks healthy,
            # so surface it prominently rather than only logging.
            # #41386: a failed session store means the transcript is NOT persisted to state.db — the live
            # chat looks healthy but resume later shows a truncated/empty session. A buried log line is not
            # enough; surface it prominently so the user knows persistence is off for this run and can fix
            # the store before relying on resume.
            self._session_db_unavailable = True
            logger.warning("Failed to initialize SessionDB — session will NOT be indexed for search: %s", e)
            from hermes_state_user_copy import describe_storage_failure, storage_failure_details
            failure = describe_storage_failure(e)
            try:
                Console(stderr=True).print(
                    "[bold yellow]⚠ Session store unavailable[/bold yellow] — "
                    "this conversation will [bold]NOT be saved[/bold] and cannot be resumed later. "
                    "Searching past sessions is also disabled.\n"
                    f"  Reason: {failure.gloss}.\n"
                    f"  {failure.action}\n"
                    f"  [dim]Details: {storage_failure_details(e)}[/dim]"
                )
            except Exception:
                print(
                    "WARNING: Session store unavailable — this conversation will NOT be "
                    f"saved and cannot be resumed later. Reason: {failure.gloss}. {failure.action}"
                )
        _run_state_db_auto_maintenance(self._session_db)
        _run_checkpoint_auto_maintenance()

    def _init_ui_state(self):
        """Per-run mutable UI state; must exist before any chat() call since -q never goes through run()."""
        self._pending_input = queue.Queue()
        self._interrupt_queue = queue.Queue()
        self._agent_running = self._should_exit = False
        self._last_turn_interrupted = False  # /goal never auto-queues on a Ctrl+C'd turn
        self._terminal_io_broken = False  # stdout EIO: freeze UI paints instead of spinning
        self._delete_session_on_exit = False  # /exit --delete
        # /update: relaunch() runs from run() after prompt_toolkit restored terminal modes.
        # /exit --delete: when True, the current session's SQLite history and on-disk transcripts are
        # deleted during shutdown. Set by process_command() when the user runs /exit --delete or /quit
        # --delete. Ported from google-gemini/gemini-cli#19332.
        self._pending_relaunch: list[str] | None = None
        self._last_ctrl_c_time = 0
        # Blocking-prompt overlays (clarify / sudo / approval / slash-confirm / model picker).
        self._clarify_state = self._clarify_multi_base = None
        self._clarify_freetext = False
        self._clarify_deadline = 0
        self._clarify_multi_base = None
        self._clarify_prefill = ""
        self._sudo_state = None
        self._sudo_deadline = 0
        self._modal_input_snapshot = None
        self._approval_state = None
        self._approval_deadline = 0
        self._approval_lock = threading.Lock()
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0
        self._model_picker_state = None
        # Rotating task-oriented composer placeholder (C-09), chosen once per
        # session so it stays stable while the empty input box is on screen.
        try:
            from hermes_cli.tips import get_random_composer_placeholder
            self._composer_placeholder = get_random_composer_placeholder()
        except Exception:
            self._composer_placeholder = ""
        self._command_palette_state = None
        # Armed when a bare `/resume` prints the recent-sessions list so the
        # very next bare numeric input (e.g. `3`) resolves to that session.
        # Holds the exact list used for index resolution; one-shot (cleared on
        # the next submitted input, whether it's the selection or anything
        # else). See #34584.
        self._pending_resume_sessions = None
        # One-shot agent seed set by a slash handler (e.g. /blueprint <name>)
        # that wants its output run as the next agent turn. Consumed and cleared
        # by the interactive loop immediately after process_command() returns.
        self._pending_agent_seed = None
        self._secret_state = None
        self._secret_deadline = 0
        self._tool_start_time: float = 0.0
        self._pending_tool_info: dict = {}  # function_name -> [(preview, args)] for stacked scrollback
        self._spinner_text = self._command_status = ""
        self._last_scrollback_tool: str = ""  # "new" mode dedup
        self._command_running = self._command_blocks_input = False
        # Petdex mascot (display.pet): kitty placeholders on kitty/Ghostty, half-blocks elsewhere.
        self._pet_renderer = self._pet_anim_thread = None
        self._pet_slug = self._pet_kitty_pending = ""
        self._pet_enabled = self._pet_anim_running = False
        self._pet_cols: int = 18
        self._pet_scale: float = 0.7
        self._pet_frames_cache: dict = {}
        self._pet_kitty_cache: dict = {}
        self._pet_kitty_image_id = self._pet_frame_idx = 0
        self._pet_lock = threading.Lock()
        self._pet_cfg_checked = self._pet_event_until = 0.0
        self._pet_event: str = ""
        self._pet_reasoning = self._pet_turn_error = False
        self._attached_images: list[Path] = []
        self._image_counter = 0
        # Ctrl+S prompt stash; in-memory only because drafts routinely contain secrets.
        from hermes_cli.prompt_stash import PromptStash as _PromptStash
        self._prompt_stash = _PromptStash()
        self.preloaded_skills: list[str] = []
        self._startup_skills_line_shown = False
        # skills.auto_load rendered in the preload thread; None until joined. Handed to every
        # agent this CLI builds so the prompt bytes never depend on when the agent was created.
        self._auto_load_skills_result: Optional[tuple] = None
        # Background --skills preload, joined by finalize_preloaded_skills before any agent is built.
        self._preload_skills_thread: Optional[threading.Thread] = None
        self._preload_skills_result: Optional[tuple] = None
        self._preload_skills_error: Optional[BaseException] = None
        self._preload_skills_requested: list = []
        self._preload_skills_finalized = False
        self._active_session_lease = None

        # Voice mode state (also reinitialized inside run() for interactive TUI).
        self._voice_lock = threading.Lock()
        self._voice_mode = self._voice_tts = self._voice_recording = False
        self._voice_processing = self._voice_continuous = False
        self._voice_recorder = self._voice_tts_stop = None
        self._voice_tts_done = threading.Event()
        self._voice_tts_done.set()
        self._voice_barge_capture = threading.Event()  # barge monitor is capturing the interruption
        self._voice_last_tts_text = ""  # echo guard
        self._voice_barge_phase = None  # "generation" | "playback"

        self._status_bar_visible = _status_bar_visible_from_display_config(CLI_CONFIG.get("display"))
        self._battery_visible = bool(CLI_CONFIG["display"].get("battery", False))
        # Vi/vim editing mode for the input composer (display.vim_mode, config-only).
        # Off by default: prompt_toolkit's standard emacs bindings.
        self._vim_mode = bool(CLI_CONFIG["display"].get("vim_mode", False))
        # Hide rules + status bar until the next input after a resize, so SIGWINCH cannot
        # stamp a fresh status bar over one the terminal just reflowed into scrollback.
        self._status_bar_suppressed_after_resize = self._resize_recovery_pending = False
        self._resize_recovery_lock = threading.Lock()
        self._resize_recovery_timer = self._status_bar_unsuppress_timer = None  # latter: debounced un-suppress
        self._last_resize_width = None  # width change (reflow, needs viewport clear) vs rows-only

        self._background_tasks: Dict[str, threading.Thread] = {}
        self._background_task_counter = 0

        # Cache-hit baseline, reset on model switch / compression so the bar shows the current regime.
        self._cache_hit_baseline_prompt = self._cache_hit_baseline_read = self._cache_hit_baseline_compressions = 0
        self._cache_hit_baseline_model: Optional[str] = None

    def _claim_active_session(self, surface: str = "cli", *, stderr: bool = False) -> bool:
        """Claim a global active-session slot for this CLI process."""
        if self._active_session_lease is not None:
            return True
        try:
            from hermes_cli.active_sessions import format_refusal_stderr, try_acquire_active_session

            lease, message = try_acquire_active_session(
                session_id=self.session_id,
                surface=surface,
                config=self.config,
                # Writer identity: a re-claim by this process replaces its own entry.
                # See #94595.
                metadata={"live_session_id": str(self.session_id)},
            )
        except Exception as exc:
            logger.warning("Failed to claim active session slot: %s", exc)
            return True
        if message:
            print(format_refusal_stderr(message), file=sys.stderr) if stderr else self._console_print(f"[bold red]{message}[/]")
            return False
        self._active_session_lease = lease
        with suppress(Exception):
            atexit.register(self._release_active_session)
        return True

    def _release_active_session(self) -> None:
        lease = getattr(self, "_active_session_lease", None)
        if lease is None:
            return
        try:
            lease.release()
        except Exception:
            logger.debug("Failed to release active session slot", exc_info=True)
        finally:
            self._active_session_lease = None

    _PET_FRAME_INTERVAL = 0.16
    _PET_CFG_INTERVAL = 2.5

    def _pet_resolve_config(self) -> None:
        """(Re)resolve the active pet from config — picks up live enable/disable/

        switch made via ``/pet`` or ``hermes pets`` without a restart, mirroring
        the TUI's steady poll. Cheap and fail-open: any problem disables the pet.
        """
        try:
            from agent.pet import constants, store
            from agent.pet.render import PetRenderer
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}

            from utils import is_truthy_value

            enabled = is_truthy_value(pet_cfg.get("enabled"), default=False)
            slug = str(pet_cfg.get("slug", "") or "")
            scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
            cols = constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))

            if not enabled:
                with self._pet_lock:
                    self._pet_enabled = False
                    self._pet_renderer = None
                    self._pet_frames_cache.clear()
                return

            pet = store.resolve_active_pet(slug)
            if pet is None or not pet.exists:
                with self._pet_lock:
                    self._pet_enabled = False
                    self._pet_renderer = None
                    self._pet_frames_cache.clear()
                return

            with self._pet_lock:
                # Rebuild only when the resolved pet or geometry changes.
                if (
                    self._pet_renderer is None
                    or self._pet_slug != pet.slug
                    or self._pet_cols != cols
                    or self._pet_scale != scale
                ):
                    self._pet_renderer = PetRenderer(
                        str(pet.spritesheet), mode="unicode", scale=scale, unicode_cols=cols
                    )
                    self._pet_slug = pet.slug
                    self._pet_cols = cols
                    self._pet_scale = scale
                    self._pet_frames_cache.clear()
                    self._pet_frame_idx = 0
                self._pet_enabled = True
        except Exception:
            with self._pet_lock:
                self._pet_enabled = False
                self._pet_renderer = None

    def _pet_flash(self, state: str, secs: float = 1.6) -> None:
        """Briefly force a transient reaction (wave/jump/failed) before resting."""
        self._pet_event = state
        self._pet_event_until = time.monotonic() + secs

    def _on_reaction(self, kind: str) -> None:
        """User affection (ily / <3 / good bot), core-detected — the pet's share
        of the vibe signal that plays hearts on the TUI/desktop. Flash a celebrate."""
        if kind == "vibe":
            self._pet_flash("jump")

    def _pet_react_turn_end(self) -> None:
        """Flash the end-of-turn beat: failed on error, jump on a finished plan, else wave."""
        if not self._pet_enabled:
            return
        from agent.pet.state import todos_all_done

        if self._pet_turn_error:
            self._pet_flash("failed")
            return
        try:
            store = getattr(self.agent, "_todo_store", None)
            done = todos_all_done(store.read()) if store else False
        except Exception:
            done = False
        self._pet_flash("jump" if done else "wave")

    def _derive_pet_state(self) -> str:
        """Map current CLI activity to a pet animation state.

        A transient reaction beat (wave/jump/failed) wins while it's live;
        otherwise the steady state comes from the shared
        :func:`agent.pet.state.derive_pet_state` so the CLI can't drift from the
        TUI/desktop priority order.
        """
        if self._pet_event and time.monotonic() < self._pet_event_until:
            return self._pet_event
        self._pet_event = ""
        from agent.pet.state import derive_pet_state

        # A live blocking modal (approval / clarify / sudo / secret / slash
        # confirm) means the agent is paused on the user → the `waiting` pose,
        # which outranks the in-flight signals in derive_pet_state.
        awaiting_input = bool(
            self._approval_state
            or self._clarify_state
            or self._sudo_state
            or self._secret_state
            or getattr(self, "_slash_confirm_state", None)
        )

        return derive_pet_state(
            awaiting_input=awaiting_input,
            busy=getattr(self, "_agent_running", False),
            reasoning=self._pet_reasoning,
        ).value

    def _pet_frames_for(self, state: str) -> list:
        """Return (and cache) the half-block grids for one state."""
        cached = self._pet_frames_cache.get(state)
        if cached is not None:
            return cached
        renderer = self._pet_renderer
        if renderer is None:
            return []
        try:
            count = renderer.frame_count(state) or 1
            grids = [renderer.cells(state, i, cols=self._pet_cols) for i in range(count)]
        except Exception:
            grids = []
        self._pet_frames_cache[state] = grids
        return grids

    def _pet_fragments(self):
        """Return prompt_toolkit FormattedText for the current pet frame, or []."""
        with self._pet_lock:
            if not self._pet_enabled or self._pet_renderer is None:
                return []
            state = self._derive_pet_state()
            grids = self._pet_frames_for(state)
            if not grids:
                return []
            grid = grids[self._pet_frame_idx % len(grids)]

        frags = []
        for y, row in enumerate(grid):
            if y:
                frags.append(("", "\n"))
            for top, bottom in row:
                tr, tg, tb, ta = top
                br, bg, bb, ba = bottom
                top_op = ta >= 32
                bot_op = ba >= 32
                if not top_op and not bot_op:
                    frags.append(("", " "))
                elif top_op and bot_op:
                    frags.append((f"fg:#{tr:02x}{tg:02x}{tb:02x} bg:#{br:02x}{bg:02x}{bb:02x}", "▀"))
                elif top_op:
                    # Upper half only — leave the lower half the terminal's bg
                    # instead of painting it black (cleaner on light themes).
                    frags.append((f"fg:#{tr:02x}{tg:02x}{tb:02x}", "▀"))
                else:
                    frags.append((f"fg:#{br:02x}{bg:02x}{bb:02x}", "▄"))
        return frags

    def _pet_widget_height(self) -> int:
        """Visible rows for the pet window — 0 collapses it when no pet shows."""
        with self._pet_lock:
            if not self._pet_enabled or self._pet_renderer is None:
                return 0
            grids = self._pet_frames_for(self._derive_pet_state())
            if not grids or not grids[0]:
                return 0
            return len(grids[0])

    def _pet_anim_loop(self) -> None:
        """Advance the frame + invalidate on a timer while a pet is enabled."""
        while self._pet_anim_running:
            time.sleep(self._PET_FRAME_INTERVAL)
            if getattr(self, "_terminal_io_broken", False):
                self._pet_anim_running = False
                break
            now = time.monotonic()
            if now - self._pet_cfg_checked >= self._PET_CFG_INTERVAL:
                self._pet_cfg_checked = now
                self._pet_resolve_config()
            if not self._pet_enabled:
                continue
            with self._pet_lock:
                self._pet_frame_idx += 1
            app = getattr(self, "_app", None)
            if app is not None:
                try:
                    app.invalidate()
                except OSError as exc:
                    if getattr(exc, "errno", None) == errno.EIO:
                        self._mark_terminal_io_broken("pet_anim")
                        break
                except Exception:
                    pass

    def _pet_start_anim(self) -> None:
        if self._pet_anim_running:
            return
        self._pet_resolve_config()
        self._pet_anim_running = True
        self._pet_anim_thread = threading.Thread(target=self._pet_anim_loop, daemon=True)
        self._pet_anim_thread.start()

    def _pet_stop_anim(self) -> None:
        self._pet_anim_running = False
        thread = self._pet_anim_thread
        if thread is not None:
            thread.join(timeout=0.3)
        self._pet_anim_thread = None

    def _voice_record_key_label(self) -> str:
        """Return the configured voice push-to-talk key formatted for UI.

        Shared helper so every voice-facing status line / placeholder /
        recording hint advertises the SAME label as the registered
        prompt_toolkit binding.

        Cached at startup (see ``set_voice_record_key_cache``) rather
        than re-read per render. Two reasons (Copilot round-13 on
        #19835):

        * The prompt_toolkit binding is registered once at session
          start via ``@kb.add(_voice_key)``; re-reading config per
          render meant the status bar could advertise a new shortcut
          after a config edit while the actual binding was still the
          startup chord — exactly the display/binding drift this PR
          is trying to eliminate.
        * The label is on the hot render path (status bar + composer
          placeholder invalidated every 150ms during recording), so
          reading config on every call added avoidable UI overhead.
        """
        return getattr(self, "_voice_record_key_display_cache", None) or "Ctrl+B"

    def set_voice_record_key_cache(self, raw_key: object) -> None:
        """Populate the voice label cache from a raw ``voice.record_key``.

        Called at CLI startup after the prompt_toolkit binding is
        registered so the cached label always matches the live binding.
        """
        try:
            from hermes_cli.voice import format_voice_record_key_for_status
            self._voice_record_key_display_cache = format_voice_record_key_for_status(raw_key)
        except Exception:
            self._voice_record_key_display_cache = "Ctrl+B"

    def _get_voice_status_fragments(self, width: Optional[int] = None):
        """Return the voice status bar fragments for the interactive TUI."""
        width = width or self._get_tui_terminal_width()
        compact = self._use_minimal_tui_chrome(width=width)
        label = self._voice_record_key_label()
        if self._voice_recording:
            if compact:
                return [("class:voice-status-recording", " ● REC ")]
            return [("class:voice-status-recording", f" ● REC  {label} to stop ")]
        if self._voice_processing:
            if compact:
                return [("class:voice-status", " ◉ STT ")]
            return [("class:voice-status", " ◉ Transcribing... ")]
        if compact:
            return [("class:voice-status", f" 🎤 {label} ")]
        tts = " | TTS on" if self._voice_tts else ""
        cont = " | Continuous" if self._voice_continuous else ""
        return [("class:voice-status", f" 🎤 Voice mode{tts}{cont}  —  {label} to record ")]

    @staticmethod
    def _status_bar_goal_segment(snapshot: Dict[str, Any]) -> str:
        """Return the ``⊙ goal 3/20`` segment, or ``""`` when no goal is active.

        Active-goal-only by design: paused/done goals don't occupy status-bar
        real estate (they already print their own glyph lines in the thread).
        """
        if not snapshot.get("goal_active"):
            return ""
        used = snapshot.get("goal_turns_used") or 0
        max_turns = snapshot.get("goal_max_turns") or 0
        if max_turns:
            return f"⊙ goal {used}/{max_turns}"
        return "⊙ goal"

    def _build_status_bar_text(self, width: Optional[int] = None) -> str:
        """Return a compact one-line session status string for the TUI footer."""
        try:
            snapshot = self._get_status_bar_snapshot()
            if width is None:
                width = self._get_tui_terminal_width()
            percent = snapshot["context_percent"]
            percent_label = f"{percent}%" if percent is not None else "--"
            duration_label = snapshot["duration"]
            battery_label = snapshot.get("battery_label") or ""
            battery_prefix = f"{battery_label} │ " if battery_label else ""
            focus_label = snapshot.get("focus_label") or ""
            session_title = snapshot.get("session_title") or ""

            yolo_active = self._is_session_yolo_active()
            goal_segment = self._status_bar_goal_segment(snapshot)
            if width < 52:
                text = f"{battery_prefix}⚕ {snapshot['model_short']} · {duration_label}"
                if goal_segment:
                    text += f" · {goal_segment}"
                if focus_label:
                    text += f" · {focus_label}"
                if yolo_active:
                    text += " · ⚠ YOLO"
                return self._right_align_status_title(text, session_title, width)
            if width < 76:
                parts = [f"⚕ {snapshot['model_short']}", percent_label]
                if battery_label:
                    parts.insert(0, battery_label)
                compressions = snapshot.get("compressions", 0)
                if compressions:
                    parts.append(f"🗜️ {compressions}")
                bg_count = snapshot.get("active_background_tasks", 0)
                if bg_count:
                    parts.append(f"▶ {bg_count}")
                bg_proc_count = snapshot.get("active_background_processes", 0)
                if bg_proc_count:
                    parts.append(f"⚙ {bg_proc_count}")
                bg_subagent_count = snapshot.get("active_background_subagents", 0)
                if bg_subagent_count:
                    parts.append(f"⛓ {bg_subagent_count}")
                if goal_segment:
                    parts.append(goal_segment)
                parts.append(duration_label)
                if focus_label:
                    parts.append(focus_label)
                if yolo_active:
                    parts.append("⚠ YOLO")
                return self._right_align_status_title(" · ".join(parts), session_title, width)

            if snapshot["context_length"]:
                ctx_total = _format_context_length(snapshot["context_length"])
                ctx_used = format_token_count_compact(snapshot["context_tokens"])
                context_label = f"{ctx_used}/{ctx_total}"
            else:
                context_label = "ctx --"

            compressions = snapshot.get("compressions", 0)
            parts = [f"⚕ {snapshot['model_short']}", context_label, percent_label]
            if battery_label:
                parts.insert(0, battery_label)
            if compressions:
                parts.append(f"🗜️ {compressions}")
            bg_count = snapshot.get("active_background_tasks", 0)
            if bg_count:
                parts.append(f"▶ {bg_count}")
            bg_proc_count = snapshot.get("active_background_processes", 0)
            if bg_proc_count:
                parts.append(f"⚙ {bg_proc_count}")
            bg_subagent_count = snapshot.get("active_background_subagents", 0)
            if bg_subagent_count:
                parts.append(f"⛓ {bg_subagent_count}")
            if goal_segment:
                parts.append(goal_segment)
            parts.append(duration_label)
            prompt_elapsed = snapshot.get("prompt_elapsed")
            if prompt_elapsed:
                parts.append(prompt_elapsed)
            idle_since = snapshot.get("idle_since")
            if idle_since:
                parts.append(idle_since)
            if focus_label:
                parts.append(focus_label)
            if yolo_active:
                parts.append("⚠ YOLO")
            return self._right_align_status_title(" │ ".join(parts), session_title, width)
        except Exception:
            return f"⚕ {self.model if getattr(self, 'model', None) else 'Hermes'}"

    def _get_status_bar_fragments(self):
        if not self._status_bar_visible or getattr(self, '_model_picker_state', None) or getattr(self, '_command_palette_state', None):
            return []
        try:
            snapshot = self._get_status_bar_snapshot()
            # Use prompt_toolkit's own terminal width when running inside the
            # TUI — shutil.get_terminal_size() can return stale or fallback
            # values (especially on SSH) that differ from what prompt_toolkit
            # actually renders, causing the fragments to overflow to a second
            # line and produce duplicated status bar rows over long sessions.
            width = self._get_tui_terminal_width()
            duration_label = snapshot["duration"]
            yolo_active = self._is_session_yolo_active()
            goal_segment = self._status_bar_goal_segment(snapshot)
            battery_label = snapshot.get("battery_label") or ""
            battery_style = self._battery_status_style(snapshot.get("battery_category", "dim"))
            focus_label = snapshot.get("focus_label") or ""
            session_title = snapshot.get("session_title") or ""

            if width < 52:
                frags = [
                    ("class:status-bar", " ⚕ "),
                    ("class:status-bar-strong", snapshot["model_short"]),
                    ("class:status-bar-dim", " · "),
                    ("class:status-bar-dim", duration_label),
                ]
                if goal_segment:
                    frags.append(("class:status-bar-dim", " · "))
                    frags.append(("class:status-bar-strong", goal_segment))
                if focus_label:
                    frags.append(("class:status-bar-dim", " · "))
                    frags.append(("class:status-bar-strong", focus_label))
                if yolo_active:
                    frags.append(("class:status-bar-dim", " · "))
                    frags.append(("class:status-bar-yolo", "⚠ YOLO"))
                frags.append(("class:status-bar", " "))
            else:
                percent = snapshot["context_percent"]
                percent_label = f"{percent}%" if percent is not None else "--"
                if width < 76:
                    compressions = snapshot.get("compressions", 0)
                    bg_count = snapshot.get("active_background_tasks", 0)
                    bg_proc_count = snapshot.get("active_background_processes", 0)
                    bg_subagent_count = snapshot.get("active_background_subagents", 0)
                    frags = [
                        ("class:status-bar", " ⚕ "),
                        ("class:status-bar-strong", snapshot["model_short"]),
                        ("class:status-bar-dim", " · "),
                        (self._status_bar_context_style(percent), percent_label),
                    ]
                    if compressions:
                        frags.append(("class:status-bar-dim", " · "))
                        frags.append((self._compression_count_style(compressions), f"🗜️ {compressions}"))
                    if bg_count:
                        frags.append(("class:status-bar-dim", " · "))
                        frags.append(("class:status-bar-strong", f"▶ {bg_count}"))
                    if bg_proc_count:
                        frags.append(("class:status-bar-dim", " · "))
                        frags.append(("class:status-bar-strong", f"⚙ {bg_proc_count}"))
                    if bg_subagent_count:
                        frags.append(("class:status-bar-dim", " · "))
                        frags.append(("class:status-bar-strong", f"⛓ {bg_subagent_count}"))
                    if goal_segment:
                        frags.append(("class:status-bar-dim", " · "))
                        frags.append(("class:status-bar-strong", goal_segment))
                    frags.extend([
                        ("class:status-bar-dim", " · "),
                        ("class:status-bar-dim", duration_label),
                    ])
                    if focus_label:
                        frags.append(("class:status-bar-dim", " · "))
                        frags.append(("class:status-bar-strong", focus_label))
                    if yolo_active:
                        frags.append(("class:status-bar-dim", " · "))
                        frags.append(("class:status-bar-yolo", "⚠ YOLO"))
                    frags.append(("class:status-bar", " "))
                else:
                    if snapshot["context_length"]:
                        ctx_total = _format_context_length(snapshot["context_length"])
                        ctx_used = format_token_count_compact(snapshot["context_tokens"])
                        context_label = f"{ctx_used}/{ctx_total}"
                    else:
                        context_label = "ctx --"

                    bar_style = self._status_bar_context_style(percent)
                    compressions = snapshot.get("compressions", 0)
                    bg_count = snapshot.get("active_background_tasks", 0)
                    bg_proc_count = snapshot.get("active_background_processes", 0)
                    bg_subagent_count = snapshot.get("active_background_subagents", 0)
                    frags = [
                        ("class:status-bar", " ⚕ "),
                        ("class:status-bar-strong", snapshot["model_short"]),
                        ("class:status-bar-dim", " │ "),
                        ("class:status-bar-dim", context_label),
                        ("class:status-bar-dim", " │ "),
                        (bar_style, self._build_context_bar(percent)),
                        ("class:status-bar-dim", " "),
                        (bar_style, percent_label),
                    ]
                    if compressions:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append((self._compression_count_style(compressions), f"🗜️ {compressions}"))
                    if bg_count:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append(("class:status-bar-strong", f"▶ {bg_count}"))
                    if bg_proc_count:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append(("class:status-bar-strong", f"⚙ {bg_proc_count}"))
                    if bg_subagent_count:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append(("class:status-bar-strong", f"⛓ {bg_subagent_count}"))
                    if goal_segment:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append(("class:status-bar-strong", goal_segment))
                    frags.extend([
                        ("class:status-bar-dim", " │ "),
                        ("class:status-bar-dim", duration_label),
                    ])
                    # Position 7: per-prompt elapsed timer (live or frozen)
                    prompt_elapsed = snapshot.get("prompt_elapsed")
                    if prompt_elapsed:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append(("class:status-bar-dim", prompt_elapsed))
                    # Position 8: idle time since the last final agent response
                    idle_since = snapshot.get("idle_since")
                    if idle_since:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append(("class:status-bar-dim", idle_since))
                    # Persistent focus-view badge — so the reduced-output mode
                    # is never invisible (mirrors the YOLO badge convention).
                    if focus_label:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append(("class:status-bar-strong", focus_label))
                    if yolo_active:
                        frags.append(("class:status-bar-dim", " │ "))
                        frags.append(("class:status-bar-yolo", "⚠ YOLO"))
                    frags.append(("class:status-bar", " "))

            # Stash indicator (📌 N) — appended after all width tiers so the
            # user always knows a parked draft exists, even on narrow
            # terminals.  Placed before the battery prepend so it stays at the
            # right edge, and it is the first thing the width trim below drops
            # if the bar genuinely cannot fit.
            try:
                stash_indicator = self._prompt_stash.indicator()
            except Exception:
                stash_indicator = ""
            if stash_indicator:
                # Insert before the trailing pad fragment so the bar keeps its
                # one-cell right margin.
                if frags and frags[-1] == ("class:status-bar", " "):
                    frags[-1:-1] = [
                        ("class:status-bar-dim", " · "),
                        ("class:status-bar-strong", stash_indicator),
                    ]
                else:
                    frags.append(("class:status-bar-dim", " · "))
                    frags.append(("class:status-bar-strong", stash_indicator))

            # Battery is the first status-bar element when enabled: prepend it
            # ahead of the leading ⚕ marker in whichever width tier ran above.
            if battery_label:
                frags[0:0] = [
                    ("class:status-bar", " "),
                    (battery_style, battery_label),
                    ("class:status-bar-dim", " │"),
                ]

            frags = self._right_align_status_title_fragments(frags, session_title, width)

            total_width = sum(self._status_bar_display_width(text) for _, text in frags)
            if total_width > width:
                plain_text = "".join(text for _, text in frags)
                trimmed = self._trim_status_bar_text(plain_text, width)
                return [("class:status-bar", trimmed)]
            return frags
        except Exception:
            return [("class:status-bar", f" {self._build_status_bar_text()} ")]

    @staticmethod
    def _fmt_stash_age(stashed_at: float) -> str:
        """Return human-readable age string for a stash entry."""
        import time as _t
        secs = int(_t.monotonic() - stashed_at)
        if secs < 10:
            return "just now"
        if secs < 90:
            return f"{secs}s ago"
        mins = secs // 60
        if mins < 60:
            return f"{mins} min ago"
        return f"{mins // 60}h ago"

    def _render_stash_panel(self, stash_list: list, cursor: int, width: int) -> list:
        """Return prompt_toolkit formatted_text fragments for the stash panel box.

        Every horizontal measurement goes through ``_status_bar_display_width``
        (prompt_toolkit's ``get_cwidth``) rather than ``len()``.  The header
        contains 📌, which is one Python codepoint but two terminal cells; the
        original PR chased that off-by-one through three successive
        "subtract 1 from len()" commits.  Measuring in display cells fixes it
        for real and keeps CJK previews from bleeding past the right border.
        """
        cw = self._status_bar_display_width
        W = max(12, min(width - 4, 80))

        n = len(stash_list)
        hdr_prefix_str = f"╭─ 📌 Stash ({n} item{'s' if n != 1 else ''}) "
        HDR_SUFFIX = " Ctrl+S ─╮"
        FTR_PREFIX = "╰"
        FTR_SUFFIX = " ↑↓ Enter=restore  D=delete  Esc ─╯"

        # On narrow terminals the full hint text is wider than the box itself.
        # Drop to compact affordances rather than letting the frame bleed past
        # the right edge (which is what made the panel look broken).
        if cw(hdr_prefix_str) + cw(HDR_SUFFIX) > W:
            hdr_prefix_str = f"╭─ 📌 {n} "
            HDR_SUFFIX = "─╮"
        if cw(FTR_PREFIX) + cw(FTR_SUFFIX) > W:
            FTR_SUFFIX = " ↑↓ ⏎ D Esc ─╯"
        if cw(FTR_PREFIX) + cw(FTR_SUFFIX) > W:
            FTR_SUFFIX = "─╯"

        hdr_dashes = max(0, W - cw(hdr_prefix_str) - cw(HDR_SUFFIX))
        ftr_dashes = max(0, W - cw(FTR_PREFIX) - cw(FTR_SUFFIX))

        # Row inner width: W minus the two '│' border cells.
        INNER = W - 2

        frags: list = []

        def line(text: str, style: str = "") -> None:
            # Final guard: never emit a line wider than the box, whatever the
            # label lengths worked out to.
            frags.append((style, self._trim_status_bar_text(text, W) + "\n"))

        line(f"{hdr_prefix_str}{'─' * hdr_dashes}{HDR_SUFFIX}", "class:subagent-border")

        for i, item in enumerate(stash_list):
            age = self._fmt_stash_age(item["stashed_at"])
            # Row: " ► [N] {age:<10} {preview} "
            prefix = f" {'►' if i == cursor else ' '} [{i + 1}] {age:<10} "
            if cw(prefix) > INNER - 2:
                prefix = f" {'►' if i == cursor else ' '} [{i + 1}] "
            avail = max(0, INNER - cw(prefix) - 1)
            preview = self._trim_status_bar_text(item.get("preview") or "", avail)
            preview = preview + " " * max(0, avail - cw(preview))
            row = self._trim_status_bar_text(f"│{prefix}{preview} │", W)
            if i == cursor:
                frags.append(("class:subagent-selected", row + "\n"))
            else:
                frags.append(("class:subagent-border", "│"))
                frags.append(("class:subagent-sub", f"{prefix}{preview} "))
                frags.append(("class:subagent-border", "│\n"))

        line(f"{FTR_PREFIX}{'─' * ftr_dashes}{FTR_SUFFIX}", "class:subagent-border")
        return frags

    def _normalize_model_for_provider(self, resolved_provider: str) -> bool:
        """Normalize provider-specific model IDs and routing."""
        current_model = str(self.model or "").strip()
        if isinstance(self.model, dict):
            _m, _ = _split_model_config_default(self.model)
            current_model = _m
        changed = False

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS,
                normalize_model_for_provider,
            )

            if resolved_provider not in _AGGREGATOR_PROVIDERS:
                normalized_model = normalize_model_for_provider(current_model, resolved_provider)
                if normalized_model and normalized_model != current_model:
                    if not self._model_is_default:
                        self._console_print(
                            f"[yellow]⚠️  Normalized model '{current_model}' to '{normalized_model}' for {resolved_provider}.[/]"
                        )
                    self.model = normalized_model
                    current_model = normalized_model
                    changed = True
        except Exception:
            pass

        if resolved_provider == "copilot":
            try:
                from hermes_cli.models import copilot_model_api_mode, normalize_copilot_model_id

                canonical = normalize_copilot_model_id(current_model, api_key=self.api_key)
                if canonical and canonical != current_model:
                    if not self._model_is_default:
                        self._console_print(
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

        from hermes_cli.models import opencode_provider_family

        if opencode_provider_family(resolved_provider) is not None:
            try:
                from hermes_cli.models import normalize_opencode_model_id, opencode_model_api_mode

                canonical = normalize_opencode_model_id(resolved_provider, current_model)
                if canonical and canonical != current_model:
                    if not self._model_is_default:
                        self._console_print(
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
            return changed

        # 1. Strip provider prefix ("openai/gpt-5.4" → "gpt-5.4")
        if "/" in current_model:
            slug = current_model.split("/", 1)[1]
            if not self._model_is_default:
                self._console_print(
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
        self._tool_start_time = 0.0  # clear tool timer when switching to thinking
        self._invalidate()

    def _on_notice(self, notice) -> None:
        """Queue an out-of-band AgentNotice for rendering at the next clean boundary.

        Notices fire from inside the agent turn (cold-start seed during _init_agent,
        per-turn _capture_credits after the API call) — printing immediately races the
        streaming response and the line gets buried behind the prompt (see _cprint's
        bg-thread caveat). So we QUEUE here and flush in _flush_credit_notices(), called
        right after run_conversation returns. Fail-soft: never break the turn.
        """
        try:
            text = getattr(notice, "text", "") or ""
            if not text:
                return
            level = getattr(notice, "level", "info") or "info"
            if not hasattr(self, "_pending_credit_notices"):
                self._pending_credit_notices = []
            self._pending_credit_notices.append((level, text))
        except Exception:
            pass

    def _flush_credit_notices(self) -> None:
        """Print any queued credit notices as level-colored lines. Called at turn end
        (after run_conversation) where _cprint paints cleanly above the prompt."""
        try:
            pending = getattr(self, "_pending_credit_notices", None)
            if not pending:
                return
            self._pending_credit_notices = []
            for level, text in pending:
                color = {
                    "error": "\033[31m",
                    "warn": "\033[33m",
                    "success": "\033[32m",
                    "info": _DIM,
                }.get(level, _DIM)
                _cprint(f"  {color}{text}{_RST}")
        except Exception:
            pass

    def _on_notice_clear(self, key: str) -> None:
        """Notice cleared. The REPL prints lines (no persistent slot to wipe), so
        this drops any still-queued notice with that key is not tracked by key here;
        it's a no-op for rendering — kept so the agent's clear callback is bound
        symmetrically with the show callback (and so future REPL UIs can hook it)."""
        return

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

    def _format_submitted_user_message_preview(self, user_input: str) -> str:
        """Format the submitted user-message scrollback preview."""
        ts_suffix = (
            f" [dim]{datetime.now().strftime(getattr(self, 'timestamp_format', '%H:%M'))}[/]"
            if getattr(self, "show_timestamps", False) else ""
        )
        lines = user_input.split("\n")
        if len(lines) <= 1:
            return f"[bold {_accent_hex()}]●[/] [bold]{_escape(user_input)}[/]{ts_suffix}"

        first_lines = int(getattr(self, "user_message_preview_first_lines", 2))
        last_lines = int(getattr(self, "user_message_preview_last_lines", 2))
        first_lines = max(1, first_lines)
        last_lines = max(0, last_lines)
        head = lines[:first_lines]
        remaining_after_head = max(0, len(lines) - len(head))
        tail_count = min(last_lines, remaining_after_head)
        tail = lines[-tail_count:] if tail_count else []

        hidden_middle_count = len(lines) - len(head) - len(tail)
        if hidden_middle_count < 0:
            hidden_middle_count = 0
            tail = []

        preview_lines = [
            f"[bold {_accent_hex()}]●[/] [bold]{_escape(head[0])}[/]{ts_suffix}"
        ]
        preview_lines.extend(f"[bold]{_escape(line)}[/]" for line in head[1:])

        if hidden_middle_count > 0:
            noun = "line" if hidden_middle_count == 1 else "lines"
            preview_lines.append(f"[dim]... (+{hidden_middle_count} more {noun})[/]")

        preview_lines.extend(f"[bold]{_escape(line)}[/]" for line in tail)
        return "\n".join(preview_lines)

    def _expand_paste_references(self, text: str | None) -> str:
        """Expand [Pasted text #N -> file] placeholders into file contents."""
        if not isinstance(text, str) or "[Pasted text #" not in text:
            return text or ""
        paste_ref_re = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')

        def _expand_ref(match):
            path = Path(match.group(1))
            # Use try/except instead of path.exists() to avoid TOCTOU race:
            # the paste file may be deleted between check and read, causing
            # the input to be silently dropped (#17666).
            try:
                return path.read_text(encoding="utf-8")
            except (OSError, IOError):
                logger.warning("Paste file gone or unreadable, returning placeholder: %s", path)
                return match.group(0)

        return paste_ref_re.sub(_expand_ref, text)

    def _print_user_message_preview(self, user_input: str) -> None:
        """Render a user message using the normal chat scrollback style."""
        ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
        text = str(user_input or "")
        if "\n" in text:
            ChatConsole().print(self._format_submitted_user_message_preview(text))
        else:
            ChatConsole().print(f"[bold {_accent_hex()}]●[/] [bold]{_escape(text)}[/]")

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
        self._reasoning_shown_this_turn = True
        if getattr(self, "_stream_box_opened", False):
            return

        # Open reasoning box on first reasoning token
        if not getattr(self, "_reasoning_box_opened", False):
            self._reasoning_box_opened = True
            w = self._scrollback_box_width()
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
            w = self._scrollback_box_width()
            _cprint(f"{_DIM}└{'─' * (w - 2)}┘{_RST}")
            self._reasoning_box_opened = False

            # Flush any content that was deferred while reasoning was rendering.
            deferred = getattr(self, "_deferred_content", "")
            if deferred:
                self._deferred_content = ""
                self._emit_stream_text(deferred)

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
        _OPEN_TAGS = ("<REASONING_SCRATCHPAD>", "<think>", "<reasoning>", "<THINKING>", "<thinking>", "<thought>")
        _CLOSE_TAGS = ("</REASONING_SCRATCHPAD>", "</think>", "</reasoning>", "</THINKING>", "</thinking>", "</thought>")

        # Append to a pre-filter buffer first
        self._stream_prefilt = getattr(self, "_stream_prefilt", "") + text

        # Check if we're entering a reasoning block.
        # Only match tags that appear at a "block boundary": start of the
        # stream, after a newline (with optional whitespace), or when nothing
        # but whitespace has been emitted on the current line.
        # This prevents false positives when models *mention* tags in prose
        # like "(/think not producing <think> tags)".
        #
        # _stream_last_was_newline tracks whether the last character emitted
        # (or the start of the stream) is a line boundary.  It's True at
        # stream start and set True whenever emitted text ends with '\n'.
        if not hasattr(self, "_stream_last_was_newline"):
            self._stream_last_was_newline = True  # start of stream = boundary

        if not getattr(self, "_in_reasoning_block", False):
            # Case-insensitive matching against a lowercased view so
            # mixed-case tag variants (<Think>, <THINKING>, …) are caught.
            prefilt_lower = self._stream_prefilt.lower()
            for tag in _OPEN_TAGS:
                tag_lower = tag.lower()
                search_start = 0
                while True:
                    idx = prefilt_lower.find(tag_lower, search_start)
                    if idx == -1:
                        break
                    # Check if this is a block boundary position
                    preceding = self._stream_prefilt[:idx]
                    if idx == 0:
                        # At buffer start — only a boundary if we're at
                        # a line start (stream start or last emit ended
                        # with newline)
                        is_block_boundary = getattr(self, "_stream_last_was_newline", True)
                    else:
                        # Find last newline in the buffer before the tag
                        last_nl = preceding.rfind("\n")
                        if last_nl == -1:
                            # No newline in buffer — boundary only if
                            # last emit was a newline AND only whitespace
                            # has accumulated before the tag
                            is_block_boundary = (
                                getattr(self, "_stream_last_was_newline", True)
                                and preceding.strip() == ""
                            )
                        else:
                            # Text between last newline and tag must be
                            # whitespace-only
                            is_block_boundary = preceding[last_nl + 1:].strip() == ""
                    if is_block_boundary:
                        # Emit everything before the tag
                        if preceding:
                            self._emit_stream_text(preceding)
                            self._stream_last_was_newline = preceding.endswith("\n")
                        self._in_reasoning_block = True
                        self._stream_prefilt = self._stream_prefilt[idx + len(tag):]
                        break
                    # Not a block boundary — keep searching after this occurrence
                    search_start = idx + 1
                if getattr(self, "_in_reasoning_block", False):
                    break

            # Could also be a partial open tag at the end — hold it back
            if not getattr(self, "_in_reasoning_block", False):
                # Check for partial tag match at the end (case-insensitive)
                safe = self._stream_prefilt
                for tag in _OPEN_TAGS:
                    tag_lower = tag.lower()
                    for i in range(1, len(tag)):
                        if prefilt_lower.endswith(tag_lower[:i]):
                            safe = self._stream_prefilt[:-i]
                            break
                if safe:
                    self._emit_stream_text(safe)
                    self._stream_last_was_newline = safe.endswith("\n")
                    self._stream_prefilt = self._stream_prefilt[len(safe):]
                return

        # Inside a reasoning block — look for close tag.
        # Keep accumulating _stream_prefilt because close tags can arrive
        # split across multiple tokens (e.g. "</REASONING_SCRATCH" + "PAD>...").
        if getattr(self, "_in_reasoning_block", False):
            prefilt_lower = self._stream_prefilt.lower()
            for tag in _CLOSE_TAGS:
                idx = prefilt_lower.find(tag.lower())
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

        # When show_reasoning is on and reasoning is still rendering,
        # defer content until the reasoning box closes.  This ensures the
        # reasoning block always appears BEFORE the response in the terminal.
        if self.show_reasoning and getattr(self, "_reasoning_box_opened", False):
            self._deferred_content = getattr(self, "_deferred_content", "") + text
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
            if self.show_timestamps:
                label = f"{label} {datetime.now().strftime(getattr(self, 'timestamp_format', '%H:%M'))}"
            w = self._scrollback_box_width()
            fill = w - 2 - HermesCLI._status_bar_display_width(label)
            _cprint(f"\n{_ACCENT}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}")

        self._stream_buf += text

        # Emit complete lines, keep partial remainder in buffer
        _tc = getattr(self, "_stream_text_ansi", "")

        def _emit_one(printed_line: str) -> None:
            _cprint(f"{_STREAM_PAD}{_tc}{printed_line}{_RST}" if _tc else f"{_STREAM_PAD}{printed_line}")

        def _flush_table_buf() -> None:
            buf = self._stream_table_buf
            self._stream_table_buf = []
            self._in_stream_table = False
            if not buf:
                return
            # Strip cell-level markdown (`code`, **bold**, ~~strike~~) FIRST
            # so the realigner pads to the final visible cell width, not
            # the marker-decorated source width.  Otherwise a body row
            # like `` | Bold | `**bold**` | `` lands narrower than its
            # header column once the markers are removed.
            joined = "\n".join(buf)
            if self.final_response_markdown == "strip":
                joined = _strip_markdown_syntax(joined)
            block = realign_markdown_tables(joined, _terminal_width_for_streaming())
            for ln in block.split("\n"):
                _emit_one(ln)

        while "\n" in self._stream_buf:
            line, self._stream_buf = self._stream_buf.split("\n", 1)

            # Hold table-shaped lines in a side-buffer so we can re-pad
            # the whole block once it ends.  Streaming line-by-line, we
            # cannot re-align mid-table without reflowing already-printed
            # rows; the cost is that the user sees the table appear in a
            # single batch when the block closes instead of row-by-row.
            if self._in_stream_table:
                if looks_like_table_row(line) or is_table_divider(line):
                    self._stream_table_buf.append(line)
                    continue
                # Block ended — flush the realigned table, then fall
                # through to print the current (non-table) line.
                _flush_table_buf()
            elif looks_like_table_row(line):
                self._stream_table_buf.append(line)
                self._in_stream_table = True
                continue

            if self.final_response_markdown == "strip":
                line = _strip_markdown_syntax(line)
            _emit_one(line)

        # Long partial lines are emitted ONLY at real newlines — we no
        # longer hard-wrap paragraphs at terminal width ourselves.  Each
        # logical line lands in scrollback as one line; the TERMINAL
        # soft-wraps it visually, and emulators (iTerm2/kitty/VTE/
        # xterm.js/Windows Terminal) rejoin soft-wrapped rows on copy,
        # so highlight-copy yields the original unwrapped text — same
        # outcome as the TUI's selection copy.  (The pre-July-2026 chunk
        # emitter baked real '\n's into every long paragraph, which is
        # exactly what polluted copy/paste.)
        #
        # TTFT perception: while a long opening paragraph accumulates
        # without a newline, mirror its tail into the status-bar spinner
        # line so the user sees tokens arriving instead of a blank box.
        if (
            self._stream_buf
            and not self._in_stream_table
            and not self._stream_buf.lstrip().startswith("|")
            and len(self._stream_buf) >= 80
        ):
            preview = self._stream_buf[-int(_STREAM_PARTIAL_PREVIEW_LEN):]
            cut = preview.find(" ")
            if 0 < cut < len(preview) - 1:
                preview = preview[cut + 1:]
            try:
                self._spinner_text = f"… {preview}"
                self._invalidate()
            except Exception:
                pass

    def _flush_stream(self) -> None:
        """Emit any remaining partial line from the stream buffer and close the box."""
        # If we're still inside a "reasoning block" at end-of-stream, it was
        # a false positive — the model mentioned a tag like <think> in prose
        # but never closed it.  Recover the buffered content as regular text.
        if getattr(self, "_in_reasoning_block", False) and getattr(self, "_stream_prefilt", ""):
            self._in_reasoning_block = False
            self._emit_stream_text(self._stream_prefilt)
            self._stream_prefilt = ""

        # Close reasoning box if still open (in case no content tokens arrived)
        self._close_reasoning_box()

        _tc = getattr(self, "_stream_text_ansi", "")

        # If the stream buffer has a trailing partial line that looks like
        # a table row, fold it into the table buffer so the whole block
        # gets re-aligned together.  Otherwise the final row prints raw
        # (with the model's original under-padded spacing) while the rows
        # above it are aligned.
        if (
            self._stream_buf
            and getattr(self, "_in_stream_table", False)
            and (looks_like_table_row(self._stream_buf) or is_table_divider(self._stream_buf))
        ):
            self._stream_table_buf.append(self._stream_buf)
            self._stream_buf = ""

        # Flush any buffered table rows first so their padding is
        # finalised before the stream remainder lands.
        if getattr(self, "_stream_table_buf", None):
            joined = "\n".join(self._stream_table_buf)
            self._stream_table_buf = []
            self._in_stream_table = False
            if self.final_response_markdown == "strip":
                joined = _strip_markdown_syntax(joined)
            block = realign_markdown_tables(joined, _terminal_width_for_streaming())
            for ln in block.split("\n"):
                _cprint(f"{_STREAM_PAD}{_tc}{ln}{_RST}" if _tc else f"{_STREAM_PAD}{ln}")

        if self._stream_buf:
            line = _strip_markdown_syntax(self._stream_buf) if self.final_response_markdown == "strip" else self._stream_buf
            _cprint(f"{_STREAM_PAD}{_tc}{line}{_RST}" if _tc else f"{_STREAM_PAD}{line}")
            self._stream_buf = ""

        # Close the response box
        if self._stream_box_opened:
            w = self._scrollback_box_width()
            _cprint(f"{_ACCENT}╰{'─' * (w - 2)}╯{_RST}")

    def _reset_stream_state(self) -> None:
        """Reset streaming state before each agent invocation."""
        self._stream_buf = ""
        self._stream_started = False
        self._stream_box_opened = False
        self._stream_text_ansi = ""
        self._stream_prefilt = ""
        self._in_reasoning_block = False
        self._stream_last_was_newline = True
        self._reasoning_box_opened = False
        self._reasoning_buf = ""
        self._reasoning_preview_buf = ""
        self._deferred_content = ""
        self._stream_table_buf = []
        self._in_stream_table = False

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
        if cmd_lower == "/reload-skills" or cmd_lower == "/reload_skills":
            return "Reloading skills..."
        if cmd_lower.startswith("/browser"):
            return "Configuring browser..."
        return "Processing command..."

    def _command_spinner_frame(self) -> str:
        """Return the current spinner frame for slow slash commands."""
        frame_idx = int(time.monotonic() * 10) % len(_COMMAND_SPINNER_FRAMES)
        return _COMMAND_SPINNER_FRAMES[frame_idx]

    @contextmanager
    def _busy_command(self, status: str, *, blocks_input: bool = True):
        """Expose a temporary busy state in the TUI while a slash command runs.

        Most synchronous slash commands must reserve the composer because their
        completion changes the active session state. Manual compression is safe
        to draft through: the queued input is processed against the compacted
        history after the command completes.
        """
        previous_blocks_input = getattr(self, "_command_blocks_input", False)
        self._command_running = True
        self._command_blocks_input = blocks_input
        self._command_status = status
        self._invalidate(min_interval=0.0)
        try:
            print(f"⏳ {status}")
            yield
        finally:
            self._command_running = False
            self._command_blocks_input = previous_blocks_input
            self._command_status = ""
            self._invalidate(min_interval=0.0)

    def _open_external_editor(self, buffer=None) -> bool:
        """Open the active input buffer in an external editor."""
        app = getattr(self, "_app", None)
        if not app:
            _cprint(f"{_DIM}External editor is only available inside the interactive CLI.{_RST}")
            return False
        if self._command_running:
            _cprint(f"{_DIM}Wait for the current command to finish before opening the editor.{_RST}")
            return False
        if self._sudo_state or self._secret_state or self._approval_state or getattr(self, "_slash_confirm_state", None) or self._clarify_state:
            _cprint(f"{_DIM}Finish the active prompt before opening the editor.{_RST}")
            return False
        target_buffer = buffer or getattr(app, "current_buffer", None)
        if target_buffer is None:
            _cprint(f"{_DIM}No active input buffer is available for the external editor.{_RST}")
            return False
        try:
            # Inline pastes so the editor (and the draft it submits) sees real
            # content; skip flag unconditionally so the editor-close text-change
            # doesn't re-collapse it, even when there was nothing to inline.
            self._inline_pastes(target_buffer)
            self._skip_paste_collapse = True
            # Open the editor, then submit the saved draft on a clean exit —
            # matching the TUI's Ctrl+G (openEditor), which sends the buffer
            # instead of requiring a second Enter. Submission in this CLI is
            # driven by the custom `enter` keybinding, NOT the buffer's
            # accept_handler, so validate_and_handle can't route through it;
            # chain a done-callback on the returned Task that re-uses the
            # real submit pipeline via _submit_editor_buffer().
            task = target_buffer.open_in_editor(validate_and_handle=False)
            if task is not None and hasattr(task, "add_done_callback"):
                task.add_done_callback(
                    lambda _t, b=target_buffer: self._submit_editor_buffer(b)
                )
            return True
        except Exception as exc:
            _cprint(f"{_DIM}Failed to open external editor: {exc}{_RST}")
            return False

    def _submit_editor_buffer(self, buffer) -> None:
        """Submit the draft an external editor left in ``buffer``.

        Invoked from the Ctrl+G done-callback so saving the editor sends the
        prompt (TUI parity) instead of leaving it sitting in the input area.
        Mirrors the idle/queue branches of the `enter` keybinding handler:
        an empty save is ignored (never submits a blank turn), a slash command
        is dispatched, otherwise the text is routed through the same input
        queues the normal Enter path uses. Runs on the prompt_toolkit event
        loop via the Task callback, so it must be cheap and non-blocking.
        """
        try:
            text = (getattr(buffer, "text", "") or "").strip()
        except Exception:
            return
        if not text:
            # Editor saved empty / was cleared — match the TUI, which drops
            # an empty draft instead of submitting a blank turn.
            return

        app = getattr(self, "_app", None)

        # `!<command>` shell mode, checked before slash dispatch — matches the
        # Enter path in the input loop so an editor-saved bang command runs
        # locally instead of being sent to the agent.
        try:
            if self.handle_bang_shell(text):
                self._reset_input_buffer(buffer)
                if app is not None:
                    app.invalidate()
                return
        except Exception as exc:
            _cprint(f"  {_DIM}Shell command failed: {exc}{_RST}")
            self._reset_input_buffer(buffer)
            if app is not None:
                app.invalidate()
            return

        # Slash commands: dispatch directly, same as the Enter handler's
        # _looks_like_slash_command branch.
        if _looks_like_slash_command(text):
            try:
                if not self.process_command(text):
                    self._should_exit = True
                    if app is not None and app.is_running:
                        app.exit()
            except Exception as exc:
                _cprint(f"  {_DIM}Command failed: {exc}{_RST}")
            finally:
                self._reset_input_buffer(buffer)
                if app is not None:
                    app.invalidate()
            return

        # Regular prompt: route through the same queues the Enter handler uses.
        if self._agent_running:
            # Agent busy → honour the configured busy-input behaviour by
            # queueing for the next turn (the safe default; interrupt/steer
            # remain reachable via the normal Enter path).
            self._interrupt_queue.put(text) if self.busy_input_mode == "interrupt" else self._pending_input.put(text)
            preview = text[:80] + ("..." if len(text) > 80 else "")
            _cprint(f"  Queued for the next turn: {preview}")
        else:
            self._pending_input.put(text)

        self._reset_input_buffer(buffer)
        if app is not None:
            app.invalidate()

    def _inline_pastes(self, buffer) -> None:
        """Replace collapsed-paste placeholders in ``buffer`` with real content.

        A big paste shows as a compact ``[Pasted text #N -> file]`` placeholder,
        but history recall and the external editor need the actual text — a bare
        reference is useless once the file is gone or on another machine. Inlining
        before ``reset(append_to_history=True)`` also lets prompt_toolkit persist
        the content through its normal path. Sets ``_skip_paste_collapse`` so the
        ensuing text-change doesn't re-collapse it.
        """
        try:
            existing = getattr(buffer, "text", "")
            expanded = self._expand_paste_references(existing)
            if expanded != existing and hasattr(buffer, "text"):
                self._skip_paste_collapse = True
                buffer.text = expanded
                if hasattr(buffer, "cursor_position"):
                    buffer.cursor_position = len(expanded)
        except Exception:
            logger.debug("Failed to inline paste placeholders", exc_info=True)

    def _reset_input_buffer(self, buffer) -> None:
        """Clear an input buffer after a programmatic submit (best-effort)."""
        try:
            buffer.reset(append_to_history=True)
        except Exception:
            try:
                buffer.text = ""
            except Exception:
                pass



    def _install_tool_callbacks(self) -> None:
        """Install tool callbacks that need the live prompt UI."""
        if self._tool_callbacks_installed:
            return
        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)
        set_secret_capture_callback(self._secret_capture_callback)
        from agent.vault_backends.unlock import set_code_prompt_callback, set_save_login_prompt_callback, set_unlock_prompt_callback
        set_unlock_prompt_callback(self._vault_unlock_callback)
        set_save_login_prompt_callback(self._vault_save_login_callback)
        set_code_prompt_callback(self._vault_code_callback)
        self._tool_callbacks_installed = True

    def _ensure_tirith_security(self) -> None:
        """Check tirith availability once before tools can run terminal commands."""
        if self._tirith_security_checked:
            return
        self._tirith_security_checked = True
        try:
            from tools.tirith_security import ensure_installed, is_platform_supported

            if (
                ensure_installed(log_failures=False) is None and is_platform_supported()
                and (self.config.get("security", {}) or {}).get("tirith_enabled", True)
            ):
                _cprint(
                    f"  {_DIM}⚠ tirith security scanner enabled but not available "
                    f"— command scanning will use pattern matching only{_RST}"
                )
        except Exception:
            pass

    def _show_security_advisories(self):
        """Startup banner for unacked security advisories, on stderr (piped stdout stays clean); 24h rate-limited."""
        try:
            from hermes_cli.security_advisories import detect_compromised, startup_banner

            banner = startup_banner(detect_compromised())
            if banner:
                print(banner, file=sys.stderr, flush=True)
        except Exception:
            pass  # never block startup

    def _show_browser_backend_notice(self):
        """Once-per-24h hint when the default Browser Use backend silently fell back to built-in tools."""
        try:
            from tools.browser_use_cli import default_downgrade_notice

            notice = default_downgrade_notice()
            if notice:
                self._console_print(f"[yellow]⚠ {notice}[/yellow]")
        except Exception:
            logger.debug("browser backend notice failed", exc_info=True)

    def finalize_preloaded_skills(self) -> None:
        """Join the background --skills preload and fold it into the prompt (idempotent).

        Raises ``ValueError`` only when EVERY requested skill was unknown.
        """
        if getattr(self, "_preload_skills_finalized", False):
            return
        thread = getattr(self, "_preload_skills_thread", None)
        if thread is None:
            self._preload_skills_finalized = True
            return
        thread.join(timeout=120)
        self._preload_skills_finalized = True
        err = getattr(self, "_preload_skills_error", None)
        if err is not None:
            raise err
        auto_result = getattr(self, "_auto_load_skills_result", None)
        if auto_result and auto_result[2]:
            logger.warning("skills.auto_load: skill(s) not found or disabled, skipped: %s", ", ".join(auto_result[2]))
        # auto_load names first, then explicit -s names that were not already pinned.
        self.preloaded_skills = list(auto_result[1]) if auto_result else []
        result = getattr(self, "_preload_skills_result", None)
        if not result:
            return
        skills_prompt, loaded_skills, missing_skills = result
        if missing_skills:
            missing_display = ", ".join(missing_skills)
            # A typo'd name must not crash a kanban worker; only a fully-missing set fails loudly.
            if loaded_skills:
                logger.warning(
                    "Unknown skill(s) requested, skipping: %s. "
                    "Continuing with: %s. "
                    "List available skills with `hermes skills list`.",
                    missing_display,
                    ", ".join(loaded_skills),
                )
            else:
                raise ValueError(f"Unknown skill(s): {missing_display}")
        if skills_prompt:
            self.system_prompt = "\n\n".join(p for p in (self.system_prompt, skills_prompt) if p).strip()
        self.preloaded_skills += [name for name in loaded_skills if name not in self.preloaded_skills]

    def _show_tool_availability_warnings(self):
        """Warn about toolsets switched off at startup (missing API keys, unusable terminal backend)."""
        try:
            # Runs on a daemon thread on the snapshot fast path: keep the imports to modules the
            # registry walk already loaded plus the pure notices module (a heavy import here races
            # importlib's module locks against the main thread).
            from model_tools import check_tool_availability
            from hermes_cli.tool_availability_notices import (
                current_terminal_backend, filter_to_enabled_toolsets, tool_availability_warning_lines,
            )
            from tools.terminal_tool import terminal_backend_unavailable_reason
            from toolsets import resolve_toolset

            _, unavailable = check_tool_availability()
            # Only toolsets this CLI session actually has. The selection is usually a composite bundle
            # (``hermes-cli``), so expand it to tool names before matching — a raw name comparison
            # matched nothing on a default install and silently dropped the terminal notice.
            unavailable = filter_to_enabled_toolsets(unavailable, self.enabled_toolsets or [], resolve_toolset)
            lines = tool_availability_warning_lines(
                unavailable, terminal_reason=terminal_backend_unavailable_reason(),
                terminal_backend=current_terminal_backend())
            if lines:
                self._console_print()
                for line in lines:
                    self._console_print(line)
        except Exception:
            pass  # Don't crash on import errors
    
    def _show_status(self):
        """Show compact startup status line."""
        # Avoid pulling the full tool registry into the bare Termux prompt path.
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") == "1":
            tool_status = "tools deferred"
        else:
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
            tool_count = len(tools) if tools else 0
            tool_status = f"{tool_count} tools"

        # Format model name (shorten if needed)
        model_short = self.model.split("/")[-1] if "/" in self.model else self.model
        if len(model_short) > 30:
            model_short = model_short[:27] + "..."

        # Get API status indicator
        if self.api_key:
            api_indicator = "[green bold]●[/]"
        else:
            api_indicator = "[red bold]●[/]"

        # Build status line with proper markup — skin-aware colors
        try:
            from hermes_cli.skin_engine import get_active_skin
            skin = get_active_skin()
            separator_color = skin.get_color("banner_dim", "#B8860B")
            accent_color = skin.get_color("ui_accent", "#FFBF00")
            label_color = skin.get_color("ui_label", "#DAA520")
        except Exception:
            separator_color, accent_color, label_color = "#B8860B", "#FFBF00", "cyan"
        toolsets_info = ""
        if self.enabled_toolsets and "all" not in self.enabled_toolsets:
            toolsets_info = f" [dim {separator_color}]·[/] [{label_color}]toolsets: {', '.join(self.enabled_toolsets)}[/]"

        provider_info = f" [dim {separator_color}]·[/] [dim]provider: {self.provider}[/]"
        if self._provider_source:
            provider_info += f" [dim {separator_color}]·[/] [dim]auth: {self._provider_source}[/]"

        self._console_print(
            f"  {api_indicator} [{accent_color}]{model_short}[/] "
            f"[dim {separator_color}]·[/] [bold {label_color}]{tool_status}[/]"
            f"{toolsets_info}{provider_info}"
        )

    def _show_session_status(self):
        """Show gateway-style status for the current CLI session."""
        session_meta = {}
        if self._session_db:
            try:
                session_meta = self._session_db.get_session(self.session_id) or {}
            except Exception:
                session_meta = {}

        title = (session_meta.get("title") or "").strip()

        created_at = self.session_start
        started_at = session_meta.get("started_at")
        if started_at:
            try:
                created_at = datetime.fromtimestamp(float(started_at))
            except Exception:
                created_at = self.session_start

        updated_at = created_at
        for field in ("updated_at", "last_updated_at", "last_activity_at"):
            value = session_meta.get(field)
            if not value:
                continue
            try:
                updated_at = datetime.fromtimestamp(float(value))
                break
            except Exception:
                pass

        agent = getattr(self, "agent", None)
        total_tokens = getattr(agent, "session_total_tokens", 0) or 0
        provider = getattr(self, "provider", None) or "unknown"
        model = getattr(self, "model", None) or "(unknown)"
        is_running = bool(getattr(self, "_agent_running", False))

        # Reasoning level (C-02): resolve the effective effort for display.
        reasoning_label = None
        try:
            rc = getattr(agent, "reasoning_config", None) or getattr(self, "reasoning_config", None)
            if isinstance(rc, dict):
                if rc.get("enabled") is False:
                    reasoning_label = "off"
                elif rc.get("effort"):
                    reasoning_label = str(rc.get("effort"))
            show_r = getattr(self, "show_reasoning", None)
            if reasoning_label:
                reasoning_label += f" (display: {'on' if show_r else 'off'})" if show_r is not None else ""
        except Exception:
            reasoning_label = None

        # Approval mode (C-02).
        approval_label = None
        try:
            from tools.approval import _get_approval_mode, is_approval_bypass_active_for_session
            approval_label = _get_approval_mode()
            try:
                if is_approval_bypass_active_for_session(getattr(self, "session_key", "") or ""):
                    approval_label += " (YOLO bypass active)"
            except Exception:
                pass
        except Exception:
            approval_label = None

        # Context window usage (C-02): reuse the status-bar snapshot which
        # already computes tokens / max / percent.
        ctx_label = None
        try:
            snap = self._get_status_bar_snapshot()
            ctx_tokens = snap.get("context_tokens") or 0
            ctx_max = snap.get("context_length")
            ctx_pct = snap.get("context_percent")
            if ctx_max:
                left = ""
                if isinstance(ctx_pct, (int, float)):
                    left = f"{max(0, 100 - int(ctx_pct))}% left · "
                ctx_label = f"{left}{ctx_tokens:,} / {ctx_max:,} tokens used"
        except Exception:
            ctx_label = None

        lines = [
            "Hermes CLI Status",
            "",
            f"Session ID: {self.session_id}",
            f"Path: {display_hermes_home()}",
        ]
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"Model: {model} ({provider})")
        if reasoning_label:
            lines.append(f"Reasoning: {reasoning_label}")
        if approval_label:
            lines.append(f"Approvals: {approval_label}")
        if ctx_label:
            lines.append(f"Context: {ctx_label}")
        lines.extend([
            f"Created: {created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Last Activity: {updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"Tokens: {total_tokens:,}",
            f"Agent Running: {'Yes' if is_running else 'No'}",
        ])
        self._console_print("\n".join(lines), highlight=False, markup=False)
    
    def _fast_command_available(self) -> bool:
        try:
            from hermes_cli.models import model_supports_fast_mode
        except Exception:
            return False
        agent = getattr(self, "agent", None)
        model = getattr(agent, "model", None) or getattr(self, "model", None)
        return model_supports_fast_mode(model)

    def _command_available(self, slash_command: str) -> bool:
        if slash_command == "/fast":
            return self._fast_command_available()
        return True

    def show_help(self, arg: str = ""):
        """Display help. Bare /help shows categorized core commands with the
        skill list collapsed to one line; /help skills lists all skill
        commands; /help <query> filters commands by substring.
        """
        from hermes_cli.commands import COMMANDS_BY_CATEGORY, HELP_SESSION_SUBGROUPS

        arg = (arg or "").strip()
        skill_commands = _ensure_skill_commands()

        # /help skills — the full skill-command list (kept out of the default
        # view so core commands don't scroll off screen).
        if arg.lower() in ("skills", "skill"):
            if not skill_commands:
                _cprint("\n  No skill commands installed.\n")
                return
            _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} ({len(skill_commands)} installed):")
            for cmd, info in sorted(skill_commands.items()):
                ChatConsole().print(
                    f"    [bold {_accent_hex()}]{cmd:<22}[/] [dim]-[/] {_escape(info['description'])}"
                )
            _cprint("")
            return

        query = arg.lower() if arg else ""

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

        def _emit(cmd: str, desc: str) -> bool:
            if not self._command_available(cmd):
                return False
            if query and query not in cmd.lower() and query not in desc.lower():
                return False
            ChatConsole().print(
                f"    [bold {_accent_hex()}]{cmd:<15}[/] [dim]-[/] {_escape(desc)}"
            )
            return True

        for category, commands in COMMANDS_BY_CATEGORY.items():
            if category == "Session":
                # Split the oversized Session category into readable sub-groups
                # (Session / Context / Background & Automation) in the renderer.
                sub_of: dict[str, str] = {}
                for _sub, _names in HELP_SESSION_SUBGROUPS.items():
                    for _n in _names:
                        sub_of[f"/{_n}"] = _sub
                buckets: dict[str, list[tuple[str, str]]] = {"Session": []}
                for _sub in HELP_SESSION_SUBGROUPS:
                    buckets[_sub] = []
                for cmd, desc in commands.items():
                    buckets[sub_of.get(cmd, "Session")].append((cmd, desc))
                for _sub in ("Session", *HELP_SESSION_SUBGROUPS.keys()):
                    rows = buckets.get(_sub) or []
                    printed_header = False
                    for cmd, desc in rows:
                        if not self._command_available(cmd):
                            continue
                        if query and query not in cmd.lower() and query not in desc.lower():
                            continue
                        if not printed_header:
                            _cprint(f"\n  {_BOLD}── {_sub} ──{_RST}")
                            printed_header = True
                        _emit(cmd, desc)
                continue

            printed_header = False
            for cmd, desc in commands.items():
                if not self._command_available(cmd):
                    continue
                if query and query not in cmd.lower() and query not in desc.lower():
                    continue
                if not printed_header:
                    _cprint(f"\n  {_BOLD}── {category} ──{_RST}")
                    printed_header = True
                _emit(cmd, desc)

        # Skill commands: collapsed to a one-line pointer by default so the
        # 60+ skill entries don't bury the core command reference (C-04).
        if query:
            # In filter mode, DO include matching skill commands inline.
            matched_skills = [
                (cmd, info) for cmd, info in sorted(skill_commands.items())
                if query in cmd.lower() or query in (info.get("description", "").lower())
            ]
            if matched_skills:
                _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} (matching '{arg}'):")
                for cmd, info in matched_skills:
                    ChatConsole().print(
                        f"    [bold {_accent_hex()}]{cmd:<22}[/] [dim]-[/] {_escape(info['description'])}"
                    )
        elif skill_commands:
            _cprint(
                f"\n  ⚡ {_BOLD}Skill Commands{_RST}: {len(skill_commands)} installed "
                f"— {_DIM}/help skills{_RST} to list them"
            )

        _bundles_now = get_skill_bundles()
        if _bundles_now and not query:
            _cprint(f"\n  ▣ {_BOLD}Skill Bundles{_RST} ({len(_bundles_now)} installed):")
            for cmd, info in sorted(_bundles_now.items()):
                skill_count = len(info.get("skills", []))
                desc = info.get("description") or f"Load {skill_count} skills"
                ChatConsole().print(
                    f"    [bold {_accent_hex()}]{cmd:<22}[/] [dim]-[/] "
                    f"{_escape(desc)} [dim]({skill_count} skills)[/]"
                )

        quick_commands = self.config.get("quick_commands", {})
        if quick_commands and not query:
            _cprint(f"\n  ⚡ {_BOLD}Quick Commands{_RST} ({len(quick_commands)} configured):")
            for name, qcmd in sorted(quick_commands.items()):
                desc = qcmd.get("description", qcmd.get("type", ""))
                ChatConsole().print(
                    f"    [bold {_accent_hex()}]{('/' + name):<22}[/] [dim]-[/] {_escape(desc)}"
                )

        if query:
            _cprint(f"\n  {_DIM}Filtered by '{arg}' — run /help for the full list.{_RST}\n")
            return

        _cprint(f"\n  {_DIM}Tip: /help skills lists skill commands · /help <text> filters · Ctrl+P opens the command palette{_RST}")
        _cprint(f"  {_DIM}Multi-line: Ctrl+J, Alt+Enter, or \\\\+Enter for a new line{_RST}")
        _cprint(f"  {_DIM}Draft editor: Ctrl+G (Alt+G in VSCode/Cursor){_RST}")
        if _is_termux_environment():
            _cprint(f"  {_DIM}Attach image: /image {_termux_example_image_path()} or start your prompt with a local image path{_RST}\n")
        else:
            _cprint(f"  {_DIM}Paste image: Alt+V (or /paste){_RST}\n")
    
    def show_tools(self):
        """Display available tools with kawaii ASCII art."""
        # Pre-assembly list: /tools is a discovery/inspection surface, so it
        # must show the full catalog including tools deferred behind the
        # tool_search bridge (users check this to verify an MCP installed).
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True,
                                     skip_tool_search_assembly=True)
        
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
    

    def _handle_whoami_command(self):
        """Display slash-command access for the local CLI surface."""
        import getpass

        try:
            user_name = getpass.getuser() or "?"
        except Exception:
            user_name = "?"

        print()
        print("  You:            cli (local terminal)")
        print(f"  User:           {user_name}")
        print("  Tier:           unrestricted")
        print("  Slash commands: all available")
        print()

    def show_config(self):
        """Display current configuration with kawaii ASCII art."""
        terminal_env = os.getenv("TERMINAL_ENV", "local")
        terminal_cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        terminal_timeout = os.getenv("TERMINAL_TIMEOUT", "60")

        config_path = _hermes_home / 'config.yaml'
        if not config_path.exists():
            config_path = Path(__file__).parent / 'cli-config.yaml'
        config_status = "(loaded)" if config_path.exists() else "(not found)"

        # ``api_key`` may be a callable (Entra ID bearer provider): never invoke it. Prefer the
        # LIVE agent's key: the constructor seeds self.api_key from env before provider
        # resolution, so on non-OpenAI providers it can be another vendor's key.
        from agent.azure_identity_adapter import is_token_provider

        # Prefer the LIVE agent's credential when one exists: HermesCLI's
        # constructor seeds self.api_key from OPENAI/OPENROUTER env vars
        # before provider resolution runs, so on non-OpenAI providers (Nous,
        # Anthropic, ...) the constructor value is a different vendor's key
        # than the one actually authenticating requests. /config displaying
        # an sk-proj-... OpenAI key next to a Nous base URL was the visible
        # symptom (full-surface CLI QA sweep, Aug 2026).
        display_key = self.api_key
        agent = getattr(self, "agent", None)
        if agent is not None and getattr(agent, "api_key", None):
            display_key = agent.api_key
        if is_token_provider(display_key):
            api_key_display = "Microsoft Entra ID"
        elif isinstance(display_key, str) and len(display_key) > 12:
            api_key_display = f"{display_key[:8]}...{display_key[-4:]}"
        else:
            api_key_display = "Not set!"

        title = "(^_^) Configuration"
        width = 50
        pad = width - len(title)
        ssh_target = (
            f"{os.getenv('TERMINAL_SSH_USER', 'not set')}@{os.getenv('TERMINAL_SSH_HOST', 'not set')}"
            f":{os.getenv('TERMINAL_SSH_PORT', '22')}"
        ) if terminal_env == "ssh" else None
        sections = (
            ("Model", (("Model:    ", self.model), ("Base URL: ", self.base_url), ("API Key:  ", api_key_display))),
            ("Terminal", (
                ("Environment: ", terminal_env),
                *((("SSH Target:  ", ssh_target),) if ssh_target else ()),
                ("Working Dir: ", terminal_cwd),
                ("Timeout:     ", f"{terminal_timeout}s"),
            )),
            ("Agent", (
                ("Max Turns: ", self.max_turns),
                ("Toolsets:  ", ", ".join(self.enabled_toolsets) if self.enabled_toolsets else "all"),
                ("Verbose:   ", self.verbose),
            )),
            ("Session", (
                ("Started:    ", self.session_start.strftime("%Y-%m-%d %H:%M:%S")),
                ("Config File:", f"{config_path} {config_status}"),
            )),
        )
        print()
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
            from hermes_cli.session_listing import query_session_listing

            return query_session_listing(
                self._session_db,
                source="cli",
                current_session_id=self.session_id,
                include_all_sources=False,
                include_unnamed=True,
                limit=limit,
                exclude_sources=["kanban", "tool"],
            )
        except Exception:
            return []

    def _show_recent_sessions(self, *, reason: str = "history", limit: int = 10) -> bool:
        """Render recent sessions inline from the active chat TUI.

        Returns True when something was shown, False if no session list was available.
        """
        sessions = self._list_recent_sessions(limit=limit)
        if not sessions:
            return False

        from hermes_cli.main import _relative_time

        _cli_visible_print()
        if reason == "history":
            _cli_visible_print("(._.) No messages in the current chat yet — here are recent sessions you can resume:")
        else:
            _cli_visible_print("  Recent sessions:")
        _cli_visible_print()
        _cli_visible_print(f"  {'#':<3} {'Title':<32} {'Preview':<40} {'Last Active':<13} {'ID'}")
        _cli_visible_print(f"  {'─' * 3} {'─' * 32} {'─' * 40} {'─' * 13} {'─' * 24}")
        for idx, session in enumerate(sessions, start=1):
            title = session.get("title") or "—"
            preview = (session.get("preview") or "")[:38]
            last_active = _relative_time(session.get("last_active"))
            _cli_visible_print(f"  {idx:<3} {title:<32} {preview:<40} {last_active:<13} {session['id']}")
        _cli_visible_print()
        _cli_visible_print("  Use /resume <number>, /resume <session id>, or /resume <session title> to continue.")
        _cli_visible_print("  Example: /resume 2")
        _cli_visible_print()
        return True

    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            if not self._show_recent_sessions(reason="history"):
                _cli_visible_print("(._.) No conversation history yet.")
            return

        preview_limit = 400
        visible_index = 0
        hidden_tool_messages = 0
        show_ts = bool(getattr(self, "show_timestamps", False))

        def _ts_suffix(message: dict) -> str:
            # Messages restored from SessionDB carry a unix `timestamp`; live
            # unsaved turns may not. Only annotate when both the toggle is on
            # and the turn actually has a stored time — never fabricate one.
            if not show_ts:
                return ""
            ts = message.get("timestamp")
            if not ts:
                return ""
            try:
                from datetime import datetime
                return f"  [{datetime.fromtimestamp(float(ts)).strftime(getattr(self, 'timestamp_format', '%H:%M'))}]"
            except (ValueError, OSError, TypeError):
                return ""

        def flush_tool_summary():
            nonlocal hidden_tool_messages
            if not hidden_tool_messages:
                return

            noun = "message" if hidden_tool_messages == 1 else "messages"
            _cli_visible_print("\n  [Tools]")
            _cli_visible_print(f"    ({hidden_tool_messages} tool {noun} hidden)")
            hidden_tool_messages = 0

        _cli_visible_print()
        _cli_visible_print("+" + "-" * 50 + "+")
        _cli_visible_print("|" + " " * 12 + "(^_^) Conversation History" + " " * 11 + "|")
        _cli_visible_print("+" + "-" * 50 + "+")

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
                _cli_visible_print(f"\n  [You #{visible_index}]{_ts_suffix(msg)}")
                _cli_visible_print(
                    f"    {content_text[:preview_limit]}{'...' if len(content_text) > preview_limit else ''}"
                )
                continue

            _cli_visible_print(f"\n  [Hermes #{visible_index}]{_ts_suffix(msg)}")
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
            _cli_visible_print(f"    {preview}{suffix}")

        flush_tool_summary()
        _cli_visible_print()
    
    def _notify_session_boundary(self, event_type: str) -> None:
        """Fire a session-boundary plugin hook (on_session_finalize or on_session_reset).

        Non-blocking — errors are caught and logged.  Safe to call from any
        lifecycle point (shutdown, /new, /reset).
        """
        try:
            from hermes_cli.lifecycle import finalize_session, invoke_hook

            context = {
                "session_id": self.agent.session_id if self.agent else None,
                "platform": getattr(self, "platform", None) or "cli",
                "reason": (
                    "new_session"
                    if event_type == "on_session_reset"
                    else "session_boundary"
                ),
            }
            if event_type == "on_session_finalize":
                finalize_session(**context)
            else:
                invoke_hook(event_type, **context)
        except Exception:
            pass

    def _discard_session_if_empty(self, session_id: Optional[str]) -> bool:
        """Drop a just-ended session row when it never gained content.

        Starting the CLI and immediately quitting (or rotating with /new,
        /clear) used to leave an empty untitled row behind that clutters
        ``/resume`` and ``hermes sessions list``. Delegates the
        check-and-delete to ``SessionDB.delete_session_if_empty``, which
        only removes rows with no messages, no title, and no child
        sessions. Ported from google-gemini/gemini-cli#27770.
        """
        if not self._session_db or not session_id:
            return False
        # In-memory transcript is authoritative: if this CLI object holds
        # conversation messages (flushed to the DB or not), the session is
        # not empty. Protects against pruning a real conversation whose DB
        # flush failed or hasn't happened yet.
        if getattr(self, "conversation_history", None):
            return False
        try:
            from hermes_constants import get_hermes_home as _ghh
            return self._session_db.delete_session_if_empty(
                session_id, sessions_dir=_ghh() / "sessions"
            )
        except Exception:
            logger.debug(
                "Could not prune empty session %s", session_id, exc_info=True
            )
            return False

    def _launch_session_boundary_memory_flush(
        self,
        history_snapshot: list,
        *,
        session_id: Optional[str] = None,
    ) -> Optional[list]:
        """Stage old-session memory extraction so /new stays responsive.

        The context-engine ``on_session_end`` boundary is delivered
        synchronously here: it is cheap (local state clear, no LLM call) and
        ordering-sensitive — it must land before ``reset_session_state()``
        rebinds the engine to the new session.

        The memory-provider half (LLM-bound extraction, seconds) is NOT run
        here. The returned snapshot is handed by ``new_session()`` to
        ``MemoryManager.commit_session_boundary_async`` as a single
        end→switch task on the manager's serialized background worker, so
        extraction can never race the provider rebinding (providers key off
        internal ``_session_id`` state — a late ``on_session_end`` after
        ``on_session_switch`` would misattribute the old transcript to the
        new session).

        Returns the history snapshot to queue, or ``None`` when there is
        nothing to extract (no agent / empty history / no memory manager).
        """
        agent = getattr(self, "agent", None)
        if not agent or not history_snapshot:
            return None

        engine = getattr(agent, "context_compressor", None)
        if engine is not None and hasattr(engine, "on_session_end"):
            try:
                engine.on_session_end(session_id or "", history_snapshot)
            except Exception:
                logger.debug(
                    "Context engine on_session_end failed at /new boundary",
                    exc_info=True,
                )

        # No provider extraction to queue when no memory manager is
        # configured — new_session() falls back to the inline switch path.
        if getattr(agent, "_memory_manager", None) is None:
            return None
        return history_snapshot

    def new_session(self, silent=False, title=None):
        """Start a fresh session with a new session ID and cleared agent state."""
        old_session_id = self.session_id
        _boundary_snapshot = None
        if self.agent and self.conversation_history:
            # Deliver the context-engine boundary synchronously and get back
            # the history snapshot for the deferred provider extraction —
            # queued below (after rotation) so /new never blocks on the
            # LLM-bound extraction call.
            _boundary_snapshot = self._launch_session_boundary_memory_flush(
                list(self.conversation_history),
                session_id=old_session_id,
            )
            self._notify_session_boundary("on_session_finalize")
        elif self.agent:
            # First session or empty history — still finalize the old session
            self._notify_session_boundary("on_session_finalize")

        if self._session_db and old_session_id:
            # Flush any un-persisted messages from the current turn to the
            # old session *before* rotating.  /new can be called mid-turn
            # when _flush_messages_to_session_db() has not yet run — without
            # this, messages generated during the current turn are silently
            # lost on session rotation (#47202).
            if self.agent:
                try:
                    self.agent._flush_messages_to_session_db(
                        self.conversation_history,
                        conversation_history=self.conversation_history,
                    )
                except Exception:
                    pass  # best-effort
            try:
                self._session_db.end_session(old_session_id, "new_session")
            except Exception:
                pass
            # Don't let immediately-rotated empty sessions pile up in
            # /resume and `hermes sessions list` (gemini-cli#27770 port).
            self._discard_session_if_empty(old_session_id)

        self.session_start = datetime.now()
        timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        self.session_id = f"{timestamp_str}_{short_uuid}"
        getattr(self, "_write_terminal_breadcrumb", lambda: None)()
        self.conversation_history = []
        self._pending_title = None
        self._resumed = False
        # /new clears the -m / --model override flag: an explicit CLI model
        # was for the previous session only, not for every session spawned
        # afterwards.
        self._explicit_model_override = False
        self.reasoning_config = _parse_reasoning_config(
            CLI_CONFIG["agent"].get("reasoning_effort", "")
        )
        # /new is a full conversation boundary: session-scoped runtime
        # overrides (/model --session, /fast, one-turn restores) do not carry
        # forward.  Re-derive model/provider and service tier from config.yaml
        # so a session-only switch never leaks into the next session (#48055,
        # #23131).
        self._pending_one_turn_model_restore = None
        self.service_tier = _parse_service_tier_config(
            CLI_CONFIG["agent"].get("service_tier", "")
        )
        _model_config = CLI_CONFIG.get("model", {})
        _raw_default2 = (_model_config.get("default") or _model_config.get("model") or "") if isinstance(_model_config, dict) else (_model_config or "")
        _config_model, _ = _split_model_config_default(_raw_default2)
        if _config_model and _config_model != getattr(self, "model", None):
            _config_provider = (
                _model_config.get("provider", "")
                if isinstance(_model_config, dict)
                else ""
            )
            try:
                from hermes_cli.model_switch import switch_model as _switch_model

                _reset_result = _switch_model(
                    raw_input=_config_model,
                    current_provider=self.provider or "",
                    current_model=self.model or "",
                    current_base_url=self.base_url or "",
                    current_api_key=self.api_key or "",
                    is_global=False,
                    explicit_provider=_config_provider or "",
                )
                if _reset_result.success:
                    if self.agent:
                        self.agent.switch_model(
                            new_model=_reset_result.new_model,
                            new_provider=_reset_result.target_provider,
                            api_key=_reset_result.api_key,
                            base_url=_reset_result.base_url,
                            api_mode=_reset_result.api_mode,
                        )
                    self.model = _reset_result.new_model
                    self.provider = _reset_result.target_provider
                    self.requested_provider = _reset_result.target_provider
                    self._explicit_api_key = _reset_result.api_key
                    self._explicit_base_url = _reset_result.base_url
                    if _reset_result.api_key:
                        self.api_key = _reset_result.api_key
                    if _reset_result.base_url:
                        self.base_url = _reset_result.base_url
                    if _reset_result.api_mode:
                        self.api_mode = _reset_result.api_mode
                    if not silent:
                        _cprint(
                            f"  (model reset to config default: "
                            f"{_reset_result.new_model})"
                        )
            except Exception:
                # Best-effort: an unreachable config default must never block
                # /new. The session keeps the current working model.
                logger.debug("/new model reset to config default failed", exc_info=True)
        _sync_process_session_id(self.session_id)

        if self.agent:
            self.agent.session_id = self.session_id
            self.agent.session_start = self.session_start
            self.agent.reasoning_config = self.reasoning_config
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
                    self.agent._session_db_created = False
                    self._session_db.create_session(
                        session_id=self.session_id,
                        source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                        model=self.model,
                        model_config={
                            "max_iterations": self.max_turns,
                            "reasoning_config": self.reasoning_config,
                        },
                    )
                    self.agent._session_db_created = True
                except Exception:
                    pass
                if title and self._session_db:
                    from hermes_state import SessionDB
                    try:
                        sanitized = SessionDB.sanitize_title(title)
                    except ValueError as e:
                        _cprint(f"  Title rejected: {e}")
                        sanitized = None
                        title = None
                    if sanitized:
                        try:
                            self._session_db.set_session_title(self.session_id, sanitized)
                            self._pending_title = None
                            self._status_bar_title_checked_at = 0.0
                            title = sanitized
                        except ValueError as e:
                            _cprint(f"  {e} — session started untitled.")
                            title = None
                        except Exception:
                            title = None
                    elif title is not None:
                        # sanitize_title returned empty (whitespace-only / unprintable)
                        _cprint("  Title is empty after cleanup — session started untitled.")
                        title = None
            # Notify memory providers that session_id rotated to a fresh
            # conversation. reset=True signals providers to flush accumulated
            # per-session state (_session_turns, _turn_counter, _document_id).
            # Fires BEFORE the plugin on_session_reset hook (shell hooks only
            # see the new id; Python providers see the transition). See #6672.
            #
            # When the old session has history, end-of-session extraction
            # (LLM-bound, seconds) and this switch are queued as ONE task on
            # the memory manager's serialized worker — end strictly before
            # switch, without blocking /new (#16454). With no history there
            # is nothing to extract; switch inline as before.
            try:
                _mm = getattr(self.agent, "_memory_manager", None)
                if _mm is not None:
                    if _boundary_snapshot:
                        _mm.commit_session_boundary_async(
                            _boundary_snapshot,
                            new_session_id=self.session_id,
                            parent_session_id=old_session_id or "",
                            reason="new_session",
                        )
                    else:
                        _mm.on_session_switch(
                            self.session_id,
                            parent_session_id=old_session_id or "",
                            reset=True,
                            reason="new_session",
                        )
            except Exception:
                pass
            self._notify_session_boundary("on_session_reset")

        if not silent:
            if title:
                print(f"(^_^)v New session started: {title}")
            else:
                print("(^_^)v New session started!")


    def _consume_pending_resume_selection(self, text: str) -> bool:
        """Resolve a bare numeric reply that follows a bare ``/resume`` prompt.

        After ``/resume`` (no args) prints the recent-sessions list it arms
        ``self._pending_resume_sessions``. The next submitted input is given
        one chance to be a bare session number (``3``); if so we resume that
        session here. Anything else (another command, free text, blank) simply
        disarms the prompt and is handled normally by the caller.

        Returns True if the input was consumed as a resume selection (caller
        must not treat it as chat); False otherwise. The pending state is
        always one-shot: it is cleared on the first submitted input regardless
        of outcome. See #34584.
        """
        pending = self._pending_resume_sessions
        if not pending:
            return False
        # One-shot: disarm now so a non-matching input can't leave the prompt
        # armed and hijack a later number the user meant as chat.
        self._pending_resume_sessions = None

        if not isinstance(text, str):
            return False
        stripped = text.strip()
        # Only a pure number selects; let "/resume 3", titles, or any other
        # text fall through to normal handling.
        if not stripped.isdigit():
            return False

        index = int(stripped)
        if index < 1 or index > len(pending):
            _cprint(f"  Resume index {index} is out of range.")
            _cprint("  Use /resume with no arguments to see available sessions.")
            return True

        self._handle_resume_command(f"/resume {index}")
        return True


    def save_conversation(self, cmd: str = "/save"):
        """Handle /save — export the current session to json, md, or html.

        Usage: ``/save [json|md|html] [filename] [redact]``

        The snapshot is a convenience export for sharing or off-line
        inspection; every message is already persisted incrementally to the
        SQLite session DB, so the live session remains resumable via
        ``hermes --resume <id>`` regardless of whether the user ever runs
        ``/save``. ``redact`` runs the export through the force-mode secret
        redaction pass before writing.
        """
        from hermes_cli.session_export import (
            SAVE_USAGE,
            normalize_save_format,
            render_session_for_save,
        )

        parts = cmd.split()[1:]
        if not parts:
            print(SAVE_USAGE)
            return
        redact = False
        if parts[-1].lower() in ("redact", "--redact"):
            redact = True
            parts = parts[:-1]
            if not parts:
                print(SAVE_USAGE)
                return

        try:
            fmt = normalize_save_format(parts[0])
        except ValueError as e:
            print(f"(._.) {e}")
            print(SAVE_USAGE)
            return
        filename = parts[1] if len(parts) > 1 else None

        # Prefer the durable DB row (has metadata + tool calls); fall back to
        # the in-memory history for sessions that never touched the DB.
        # getattr: test doubles (SimpleNamespace / object.__new__) may not
        # carry _session_db or session_id.
        session_data = None
        _db = getattr(self, "_session_db", None)
        _sid = getattr(self, "session_id", None)
        if _db and _sid:
            try:
                session_data = _db.export_session(_sid)
            except Exception:
                session_data = None
        if not session_data:
            if not self.conversation_history:
                print("(;_;) No conversation to save.")
                return
            session_data = {
                "id": self.session_id,
                "model": self.model,
                "started_at": self.session_start.timestamp(),
                "messages": self.conversation_history,
            }

        if redact:
            from hermes_cli.session_export_md import redact_session_data

            session_data = redact_session_data(session_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_dir = get_hermes_home() / "sessions" / "saved"
        try:
            saved_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"(x_x) Failed to create save directory {saved_dir}: {e}")
            return
        if filename:
            path = Path(filename).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
        else:
            path = saved_dir / f"hermes_conversation_{timestamp}.{fmt}"

        try:
            content = render_session_for_save(session_data, fmt)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            label = {"json": "JSON", "md": "Markdown", "html": "HTML"}[fmt]
            print(f"(^_^)v Conversation saved to: {path} ({label})")
            if self.session_id:
                print(f"       Resume the live session with: hermes --resume {self.session_id}")
        except Exception as e:
            print(f"(x_x) Failed to save: {e}")

    def _rewind_persisted_user_turn(
        self,
        *,
        warm_history: List[Dict[str, Any]],
        user_ordinal: int,
        warm_live_view: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Bind one warm user ordinal to a durable row and rewind it atomically."""
        if self._session_db is None or not self.session_id:
            raise RuntimeError("session database is unavailable")

        from agent.context_compressor import (
            history_before_user_originated_turn,
            split_user_originated_turn,
            user_originated_turn_view,
        )
        from agent.memory_manager import sanitize_context
        from agent.tool_dispatch_helpers import (
            _is_multimodal_tool_result,
            _multimodal_text_summary,
        )
        from run_agent import _is_ephemeral_scaffolding

        def _persistence_content(content: Any) -> Any:
            """Project warm content exactly as the session DB flush does."""
            if _is_multimodal_tool_result(content):
                return _multimodal_text_summary(content)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif isinstance(part, dict) and part.get("type") in {
                        "image",
                        "image_url",
                        "input_image",
                    }:
                        text_parts.append("[screenshot]")
                return "\n".join(text_parts) if text_parts else None
            return content

        def _comparison_content(message: Dict[str, Any]) -> Any:
            content = _persistence_content(message.get("content"))
            if message.get("role") in {"user", "assistant"} and isinstance(
                content, str
            ):
                return sanitize_context(content).strip()
            return content

        expected_active_ids = self._session_db.get_active_message_ids(
            self.session_id
        )
        durable = self._session_db.get_messages_as_conversation(
            self.session_id,
            include_row_ids=True,
        )
        warm_persistence_history = [
            message
            for message in warm_history
            if not _is_ephemeral_scaffolding(message)
        ]
        warm_user_indices = [
            index
            for index, message in enumerate(warm_persistence_history)
            if user_originated_turn_view(message) is not None
        ]
        durable_user_indices = [
            index
            for index, message in enumerate(durable)
            if user_originated_turn_view(message) is not None
        ]
        if len(durable_user_indices) != len(warm_user_indices):
            raise RuntimeError(
                "session history changed before the rewind could be persisted"
            )
        if user_ordinal < 0 or user_ordinal >= len(durable_user_indices):
            raise RuntimeError("persisted rewind target is no longer available")

        warm_prefix, _ = history_before_user_originated_turn(
            warm_persistence_history, warm_user_indices[user_ordinal]
        )
        durable_target_index = durable_user_indices[user_ordinal]
        durable_target = durable[durable_target_index]
        durable_prefix, durable_live_view = history_before_user_originated_turn(
            durable, durable_target_index
        )
        if _comparison_content(durable_live_view) != _comparison_content(
            warm_live_view
        ):
            raise RuntimeError(
                "session history changed before the rewind could be persisted"
            )
        target_row_id = durable_target.get("_row_id")
        if not isinstance(target_row_id, int):
            raise RuntimeError("persisted rewind target has no row identity")
        scaffold, _ = split_user_originated_turn(durable_target)
        result = self._session_db.rewind_to_message(
            self.session_id,
            target_row_id,
            preserve_compaction_handoff=scaffold is not None,
            expected_active_ids=expected_active_ids,
            expected_target_content=durable_live_view.get("content"),
        )
        if scaffold is not None:
            replacement_id = result.get("replacement_message_id")
            if not isinstance(replacement_id, int) or not durable_prefix:
                raise RuntimeError("rewind did not retain its compaction handoff")
            durable_prefix[-1]["_row_id"] = replacement_id
            durable_prefix[-1]["_db_persisted"] = True
            warm_prefix[-1] = durable_prefix[-1]
        return warm_prefix, durable_live_view, result
    
    def retry_last(self):
        """Retry the last user message by removing the last exchange and re-sending.
        
        Removes the last assistant response (and any tool-call messages) and
        the last user message, then re-sends that user message to the agent.
        Returns the message to re-send, or None if there's nothing to retry.
        """
        if not self.conversation_history:
            print("(._.) No messages to retry.")
            return None
        
        # Walk backwards to the last *real* user message. Timeline bookkeeping
        # rows (display_kind set) are role=user but are not user turns — match
        # CLI resume counting and user_originated_turn_view. Compaction
        # handoffs are excluded too (durable role=user, sometimes without
        # display_kind on legacy sessions; #80622).
        from agent.context_compressor import (
            history_before_user_originated_turn,
            retryable_user_text,
            user_originated_turn_view,
        )
        from agent.memory_manager import sanitize_context
        from run_agent import _is_ephemeral_scaffolding

        warm_history = list(self.conversation_history)

        user_indices = [
            index
            for index, message in enumerate(warm_history)
            if not _is_ephemeral_scaffolding(message)
            and user_originated_turn_view(message) is not None
        ]
        
        if not user_indices:
            print("(._.) No user message found to retry.")
            return None
        last_user_idx = user_indices[-1]
        
        # Resolve a lossless live payload before touching either persistence or
        # memory. A force-user-leading compaction row is one physical carrier:
        # its historical handoff remains in the prefix while only the embedded
        # human ask is retried. Media cannot be replayed by /retry, so fail
        # closed before archiving anything.
        try:
            truncated, live_view = history_before_user_originated_turn(
                warm_history, last_user_idx
            )
            live_content = live_view.get("content")
            if isinstance(live_content, str):
                live_content = sanitize_context(live_content).strip()
            last_message = retryable_user_text(live_content)
        except ValueError as exc:
            print(f"(._.) Cannot retry that message safely: {exc}")
            return None

        # Persist the rewind before publishing the shorter in-memory view.
        # The DB owns the physical carrier split so the archived original and
        # retained scaffold are committed atomically. A plain user row keeps
        # the legacy rewind shape (no replacement scaffold).
        if self._session_db is not None and self.session_id:
            try:
                truncated, _, _ = self._rewind_persisted_user_turn(
                    warm_history=warm_history,
                    user_ordinal=len(user_indices) - 1,
                    warm_live_view=live_view,
                )
            except Exception as exc:
                print(f"(x_x) Retry rewind failed; history was not changed: {exc}")
                return None

        self.conversation_history = truncated
        if self.agent is not None:
            if hasattr(self.agent, "_session_messages"):
                self.agent._session_messages = self.conversation_history
            if hasattr(self.agent, "_last_flushed_db_idx"):
                self.agent._last_flushed_db_idx = len(self.conversation_history)
            if hasattr(self.agent, "_db_flush_scan_prefix"):
                self.agent._db_flush_scan_prefix = self.conversation_history[:]
        
        print(f"(^_^)b Retrying: \"{last_message[:60]}{'...' if len(last_message) > 60 else ''}\"")
        return last_message
    
    def undo_last(self, n: int = 1, prefill: bool = True):
        """Back up N user turns: truncate history, soft-delete on disk, prefill.

        Walks backwards N user messages and discards everything from the
        Nth-from-last user message onward (its assistant response, tool
        calls, etc.). ``n`` defaults to 1 (the last exchange); ``/undo 3``
        backs up three user turns. If ``n`` exceeds the number of user
        turns, it backs up to the oldest one.

        Beyond the in-memory ``conversation_history`` slice, this also:
          • soft-deletes the truncated rows in SessionDB (``active=0``) so
            they're hidden from re-prompts and search but kept for audit;
          • notifies memory providers via ``on_session_switch(rewound=True)``;
          • mirrors /branch's agent surgery (system-prompt invalidation +
            flush-index reset);
          • when ``prefill`` is set and an input buffer is available,
            pre-fills the composer with the backed-up message text so it
            can be edited and resubmitted.

        ``prefill=False`` is used by callers that drive the undo
        programmatically (e.g. checkpoint rollback) and don't want to
        touch the user's input buffer.
        """
        if not self.conversation_history:
            print("(._.) No messages to undo.")
            return

        if n < 1:
            n = 1

        # Walk backwards collecting the indices of the last N *real* user
        # messages (exclude display_kind timeline rows and compaction
        # handoffs — same predicate as user_originated_turn_view, resume
        # turn counting, and /retry; #80622).
        from agent.context_compressor import (
            history_before_user_originated_turn,
            user_originated_turn_view,
        )
        from run_agent import _is_ephemeral_scaffolding

        warm_history = list(self.conversation_history)

        user_indices = [
            index
            for index, message in enumerate(warm_history)
            if not _is_ephemeral_scaffolding(message)
            and user_originated_turn_view(message) is not None
        ]

        if not user_indices:
            print("(._.) No user message found to undo.")
            return

        turns_undone = min(n, len(user_indices))
        target_ordinal = len(user_indices) - turns_undone
        cut_idx = user_indices[target_ordinal]

        removed_count = len(warm_history) - cut_idx
        truncated, live_view = history_before_user_originated_turn(
            warm_history, cut_idx
        )
        removed_text = self._undo_content_to_text(live_view.get("content"))

        # Soft-delete the truncated rows on disk so re-prompts and search
        # see the clean transcript while the rows survive for audit.
        rewound_rows = 0
        if self._session_db is not None and self.session_id:
            try:
                truncated, durable_live_view, result = (
                    self._rewind_persisted_user_turn(
                        warm_history=warm_history,
                        user_ordinal=target_ordinal,
                        warm_live_view=live_view,
                    )
                )
                # Canonicalize the editable prefill before mutation. The raw
                # physical carrier contains the reference summary wrapper.
                durable_text = self._undo_content_to_text(
                    durable_live_view.get("content")
                )
                if durable_text:
                    removed_text = durable_text
                rewound_rows = result.get("rewound_count", 0)
            except Exception as e:
                logger.debug("undo: durable rewind failed: %s", e)
                print(f"(x_x) Undo failed; history was not changed: {e}")
                return

        # Publish only after the durable rewind succeeds (or no store exists).
        self.conversation_history = truncated

        # Agent surgery: invalidate the system-prompt cache and reset the
        # flush index so the next turn re-flushes from the truncated head.
        if self.agent is not None:
            if hasattr(self.agent, "_invalidate_system_prompt"):
                try:
                    self.agent._invalidate_system_prompt()
                except Exception:
                    pass
            if hasattr(self.agent, "_last_flushed_db_idx"):
                try:
                    self.agent._last_flushed_db_idx = len(self.conversation_history)
                except Exception:
                    pass
            if hasattr(self.agent, "_session_messages"):
                self.agent._session_messages = self.conversation_history
            if hasattr(self.agent, "_db_flush_scan_prefix"):
                self.agent._db_flush_scan_prefix = self.conversation_history[:]
            # Notify memory providers — same hook /branch fires, with the
            # rewound flag so per-turn document caches invalidate (#6672, #21910).
            try:
                _mm = getattr(self.agent, "_memory_manager", None)
                if _mm is not None and self.session_id:
                    _mm.on_session_switch(
                        self.session_id,
                        parent_session_id="",
                        reset=False,
                        rewound=True,
                    )
            except Exception:
                pass

        turn_word = "turn" if turns_undone == 1 else "turns"
        msg_count = rewound_rows or removed_count
        print(
            f"(^_^)b Undid {turns_undone} {turn_word} ({msg_count} message(s)). "
            f"Backed up to: \"{removed_text[:60]}{'...' if len(removed_text) > 60 else ''}\""
        )
        remaining = len(self.conversation_history)
        print(f"  {remaining} message(s) remaining in history.")

        # Pre-fill the composer with the backed-up message so the user can
        # edit and resubmit (Claude-Code-style). Editable, not auto-sent.
        if prefill and removed_text:
            self._prefill_input_buffer(removed_text)

    @staticmethod
    def _undo_content_to_text(content) -> str:
        """Flatten message content (str or content-part list) to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(t for t in parts if t)
        return ""

    def _prefill_input_buffer(self, text: str) -> None:
        """Place ``text`` in the active prompt_toolkit buffer, editable."""
        app = getattr(self, "_app", None)
        if app is None:
            return
        try:
            buf = app.current_buffer
            buf.text = text
            if hasattr(buf, "cursor_position"):
                buf.cursor_position = len(text)
            app.invalidate()
        except Exception as e:
            logger.debug("undo: prefill buffer failed: %s", e)
    
    def _run_curses_picker(self, title: str, items: list[str], default_index: int = 0) -> int | None:
        """Run curses_single_select via run_in_terminal so prompt_toolkit handles terminal ownership cleanly."""
        import threading
        from hermes_cli.curses_ui import curses_single_select

        result = [None]

        def _pick():
            result[0] = curses_single_select(title, items, default_index=default_index)

        # run_in_terminal requires an asyncio event loop — only exists in the
        # main prompt_toolkit thread.  If we're in a background thread (e.g.
        # process_loop), fall back to direct curses call.
        in_main_thread = threading.current_thread() is threading.main_thread()

        if self._app and in_main_thread:
            from prompt_toolkit.application import run_in_terminal
            was_visible = self._status_bar_visible
            self._status_bar_visible = False
            self._app.invalidate()
            try:
                run_in_terminal(_pick)
            finally:
                self._status_bar_visible = was_visible
                self._app.invalidate()
        else:
            _pick()

        return result[0]

    def _prompt_text_input(self, prompt_text: str) -> str | None:
        """Prompt for free-text input safely inside or outside prompt_toolkit.

        Mirrors the thread-aware guard in ``_run_curses_picker``: ``run_in_terminal``
        returns a coroutine that must be awaited by the prompt_toolkit event loop,
        which only exists on the main thread.  Slash commands are dispatched from
        the ``process_loop`` daemon thread (see issue #23185), so calling
        ``run_in_terminal`` from there orphans the coroutine — ``_ask`` never runs,
        and user keystrokes leak into the composer instead.  Fall back to a direct
        ``input()`` when we're off the main thread.
        """
        import threading
        result = [None]

        def _ask():
            try:
                result[0] = input(prompt_text).strip() or None
            except (KeyboardInterrupt, EOFError):
                pass

        in_main_thread = threading.current_thread() is threading.main_thread()

        # Slash-worker guard (#23185 / billing auto-reload hang): when a
        # prompt_toolkit app is running but we're on a non-main thread (the
        # process_loop / TUI slash-worker daemon thread), stdin is owned by the
        # event loop / JSON-RPC pipe.  A bare input() there blocks forever until
        # the worker's 45s timeout fires.  We cannot safely prompt off the main
        # thread, so cancel cleanly (None) instead of hanging — mirrors the
        # _stdin_fallback discipline in _prompt_text_input_modal.
        if self._app and not in_main_thread:
            self._invalidate()
            return None

        if self._app and in_main_thread:
            from prompt_toolkit.application import run_in_terminal
            was_visible = self._status_bar_visible
            self._status_bar_visible = False
            self._app.invalidate()
            try:
                run_in_terminal(_ask)
            except Exception:
                # WSL / Warp / certain terminal emulators silently drop the
                # scheduled coroutine.  Fall back to a direct input() so the
                # user's keystrokes don't leak into the agent buffer.
                try:
                    _ask()
                except Exception:
                    pass
            finally:
                self._status_bar_visible = was_visible
                self._app.invalidate()
        else:
            _ask()
        return result[0]

    def _prompt_text_input_modal(
        self,
        *,
        title: str,
        detail: str,
        choices: list[tuple[str, str, str]],
        timeout: float = 120,
    ) -> str | None:
        """Prompt through the prompt_toolkit composer instead of raw input().

        This is for CLI slash-command confirmations.  The old raw input() path
        fought prompt_toolkit's active stdin ownership: in some terminals the
        prompt appeared above the TUI, choices were redrawn later, and Enter
        could be interpreted as EOF/exit.  A first-class modal state keeps the
        choices visible and lets the normal Enter key binding submit the typed
        or highlighted choice.

        **Platform note (Windows — issue #33961):**
        Earlier code bypassed the modal on ``sys.platform == "win32"`` and fell
        back to a raw ``input()`` prompt.  When the confirm was triggered from the
        ``process_loop`` daemon thread (the normal case) that ``input()`` ran off
        the main thread and deadlocked against prompt_toolkit's stdin ownership —
        the user saw a frozen cursor and Ctrl-C was swallowed (bare ``/reset``
        froze; ``/reset now`` worked only because it skips the prompt entirely).

        Native Windows now uses the same path as Linux/macOS: the modal is set up
        on ``self._app.loop`` via ``call_soon_threadsafe`` and answered by the
        normal prompt_toolkit key bindings (the same input channel that already
        handles ordinary typing on Windows).  The raw ``input()`` fallback is kept
        only for the genuinely safe cases: no running app (unit tests /
        non-interactive), no resolvable event loop, or a scheduling failure.
        """
        import threading
        import time as _time

        if not choices:
            return None

        # If prompt_toolkit is not running (unit tests / non-interactive calls),
        # keep the simple stdin fallback.
        if not getattr(self, "_app", None):
            return self._prompt_text_input("Choice [1/2/3]: ")

        try:
            app_loop = self._app.loop
        except Exception:
            app_loop = None

        in_main_thread = threading.current_thread() is threading.main_thread()

        def _stdin_fallback() -> str | None:
            # On native Windows a raw input() from a non-main thread deadlocks
            # against prompt_toolkit's stdin ownership (#33961).  With an app
            # running we cannot safely prompt off the main thread, so cancel
            # cleanly (None) rather than hang the terminal.
            if sys.platform == "win32" and not in_main_thread:
                self._invalidate()
                return None
            return self._prompt_text_input("Choice [1/2/3]: ")

        if not in_main_thread and app_loop is None:
            return _stdin_fallback()

        response_queue = queue.Queue()

        def _setup_modal() -> None:
            self._capture_modal_input_snapshot()
            self._slash_confirm_state = {
                "title": title,
                "detail": detail,
                "choices": choices,
                "selected": 0,
                "response_queue": response_queue,
            }
            self._slash_confirm_deadline = _time.monotonic() + timeout
            self._invalidate()

        def _teardown_modal() -> None:
            self._slash_confirm_state = None
            self._slash_confirm_deadline = 0
            self._restore_modal_input_snapshot()
            self._invalidate()

        def _run_on_app_loop(fn) -> bool:
            if in_main_thread or app_loop is None:
                fn()
                return True
            ready = threading.Event()

            def _wrapped() -> None:
                try:
                    fn()
                finally:
                    ready.set()

            try:
                app_loop.call_soon_threadsafe(_wrapped)
            except Exception:
                return False
            return ready.wait(timeout=5)

        if not _run_on_app_loop(_setup_modal):
            return _stdin_fallback()

        _last_countdown_refresh = _time.monotonic()
        try:
            while True:
                try:
                    result = response_queue.get(timeout=1)
                    _run_on_app_loop(_teardown_modal)
                    return result
                except queue.Empty:
                    remaining = self._slash_confirm_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                    now = _time.monotonic()
                    if now - _last_countdown_refresh >= 5.0:
                        _last_countdown_refresh = now
                        self._invalidate()
        finally:
            if self._slash_confirm_state is not None:
                _run_on_app_loop(_teardown_modal)
        return None

    def _submit_slash_confirm_response(self, value: str | None) -> None:
        state = self._slash_confirm_state
        if not state:
            return
        state["response_queue"].put(value)
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0
        self._invalidate()

    def _normalize_slash_confirm_choice(
        self,
        raw: str | None,
        choices: list[tuple[str, str, str]],
    ) -> str | None:
        if raw is None:
            return None
        choice_raw = raw.strip().lower()
        if not choice_raw:
            return None
        aliases = {
            "1": "once",
            "once": "once",
            "approve": "once",
            "yes": "once",
            "y": "once",
            "ok": "once",
            "2": "always",
            "always": "always",
            "remember": "always",
            "3": "cancel",
            "cancel": "cancel",
            "nevermind": "cancel",
            "no": "cancel",
            "n": "cancel",
        }
        allowed = {choice[0] for choice in choices}
        normalized = aliases.get(choice_raw)
        if normalized in allowed:
            return normalized
        if choice_raw in allowed:
            return choice_raw
        return None

    def _get_slash_confirm_display_fragments(self):
        """Render the /new-/clear-style confirmation panel."""
        state = self._slash_confirm_state
        if not state:
            return []

        title = state.get("title") or "Confirm action"
        detail = state.get("detail") or ""
        choices = state.get("choices") or []
        selected = state.get("selected", 0)

        def _panel_box_width(title_text: str, content_lines: list[str], min_width: int = 56, max_width: int = 86) -> int:
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

        preview_lines = []
        for line in detail.splitlines():
            preview_lines.extend(_wrap_panel_text(line, 72))
        for idx, (_value, label, desc) in enumerate(choices):
            marker = "❯" if idx == selected else " "
            preview_lines.extend(_wrap_panel_text(f"{marker} [{idx + 1}] {label} — {desc}", 72, subsequent_indent="    "))
        preview_lines.append("Type 1/2/3 or use ↑/↓ then Enter. ESC/Ctrl+C cancels.")

        box_width = _panel_box_width(title, preview_lines)
        inner_text_width = max(8, box_width - 2)
        detail_wrapped = []
        for line in detail.splitlines():
            detail_wrapped.extend(_wrap_panel_text(line, inner_text_width))
        choice_wrapped: list[tuple[int, str]] = []
        for idx, (_value, label, desc) in enumerate(choices):
            marker = "❯" if idx == selected else " "
            for wrapped in _wrap_panel_text(f"{marker} [{idx + 1}] {label} — {desc}", inner_text_width, subsequent_indent="    "):
                choice_wrapped.append((idx, wrapped))

        term_rows = shutil.get_terminal_size((100, 24)).lines
        reserved_below = 6
        chrome_full = 6
        available = max(0, term_rows - reserved_below)
        max_detail_rows = max(1, available - chrome_full - len(choice_wrapped))
        max_detail_rows = min(max_detail_rows, 8)
        if len(detail_wrapped) > max_detail_rows:
            keep = max(1, max_detail_rows - 1)
            detail_wrapped = detail_wrapped[:keep] + ["… (detail truncated)"]

        lines = []
        lines.append(('class:approval-border', '╭' + ('─' * box_width) + '╮\n'))
        _append_panel_line(lines, 'class:approval-border', 'class:approval-title', title, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        for wrapped in detail_wrapped:
            _append_panel_line(lines, 'class:approval-border', 'class:approval-desc', wrapped, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        for idx, wrapped in choice_wrapped:
            style = 'class:approval-selected' if idx == selected else 'class:approval-choice'
            _append_panel_line(lines, 'class:approval-border', style, wrapped, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        _append_panel_line(lines, 'class:approval-border', 'class:approval-cmd', 'Type 1/2/3 or use ↑/↓ then Enter. ESC/Ctrl+C cancels.', box_width)
        lines.append(('class:approval-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _build_command_palette_entries(self) -> list:
        """Flat list of (command, description) for the Ctrl+P palette.

        Sourced from the same COMMAND_REGISTRY that backs /help, filtered to
        commands available on this surface, plus installed skill commands.
        Selecting an entry inserts the exact command string — never a fuzzy
        resolution.
        """
        from hermes_cli.commands import COMMANDS_BY_CATEGORY

        entries: list[tuple[str, str, str]] = []  # (command, category, desc)
        for category, commands in COMMANDS_BY_CATEGORY.items():
            for cmd, desc in commands.items():
                if not self._command_available(cmd):
                    continue
                entries.append((cmd, category, desc))
        try:
            for cmd, info in sorted(_ensure_skill_commands().items()):
                entries.append((cmd, "Skill", info.get("description", "")))
        except Exception:
            pass
        return entries

    def _open_command_palette(self) -> None:
        """Open the Ctrl+P fuzzy command palette modal."""
        if getattr(self, "_command_palette_state", None):
            return
        # Don't stack over other modals.
        if (self._model_picker_state or self._clarify_state or self._approval_state
                or self._slash_confirm_state or self._sudo_state or self._secret_state):
            return
        self._capture_modal_input_snapshot()
        self._command_palette_state = {
            "entries": self._build_command_palette_entries(),
            "filter": "",
            "selected": 0,
            "_scroll_offset": 0,
        }
        self._invalidate(min_interval=0.0)

    def _close_command_palette(self) -> None:
        self._command_palette_state = None
        self._restore_modal_input_snapshot()
        self._invalidate(min_interval=0.0)

    def _command_palette_visible_entries(self) -> list:
        """Return (command, category, desc) rows matching the active filter.

        Ranked, command-name-focused matching (a bare subsequence over the
        whole "cmd category desc" string is uselessly permissive — "steer"
        would match 130+ rows via description text). Priority:
          0 exact command match
          1 command startswith query
          2 query substring in command
          3 query subsequence in command
          4 query substring in description
        Rows that match nowhere are dropped. Ties keep registry order.
        """
        state = self._command_palette_state or {}
        entries = state.get("entries") or []
        q = (state.get("filter", "") or "").strip().lower()
        if not q:
            return list(entries)

        def _subseq(needle: str, hay: str) -> bool:
            it = iter(hay)
            return all(ch in it for ch in needle)

        ranked = []
        for order, row in enumerate(entries):
            cmd, _cat, desc = row
            name = cmd.lower().lstrip("/")
            qn = q.lstrip("/")
            desc_l = (desc or "").lower()
            if name == qn:
                rank = 0
            elif name.startswith(qn):
                rank = 1
            elif qn in name:
                rank = 2
            elif _subseq(qn, name):
                rank = 3
            elif q in desc_l:
                rank = 4
            else:
                continue
            ranked.append((rank, order, row))
        ranked.sort(key=lambda t: (t[0], t[1]))
        return [row for (_r, _o, row) in ranked]

    def _handle_command_palette_selection(self) -> None:
        """Insert the selected command into the composer (does not auto-run)."""
        state = self._command_palette_state
        if not state:
            return
        rows = self._command_palette_visible_entries()
        selected = state.get("selected", 0)
        if not (0 <= selected < len(rows)):
            self._close_command_palette()
            return
        cmd = rows[selected][0]  # exact command string, e.g. "/model"
        self._close_command_palette()
        # Prefill the composer so the user can add args / confirm — never
        # auto-execute (a palette pick should be explicit, and many commands
        # take arguments).
        try:
            app = getattr(self, "_app", None)
            if app is not None:
                buf = app.current_buffer
                buf.text = cmd + " "
                buf.cursor_position = len(buf.text)
                self._invalidate(min_interval=0.0)
        except Exception:
            logger.debug("command palette prefill failed", exc_info=True)

    def _open_model_picker(self, providers: list, current_model: str, current_provider: str, user_provs=None, custom_provs=None) -> None:
        """Open prompt_toolkit-native /model picker modal."""
        self._capture_modal_input_snapshot()
        default_idx = next((i for i, p in enumerate(providers) if p.get("is_current")), 0)
        self._model_picker_state = {
            "stage": "provider",
            "providers": providers,
            "selected": default_idx,
            "current_model": current_model,
            "current_provider": current_provider,
            "user_provs": user_provs,
            "custom_provs": custom_provs,
            "filter": "",
        }
        self._invalidate(min_interval=0.0)

    def _confirm_expensive_model_switch(self, result) -> bool:
        """Ask for explicit confirmation before applying costly model switches."""
        if not getattr(result, "success", False):
            return True
        try:
            from hermes_cli.model_selection_guards import combined_selection_warning

            warning = combined_selection_warning(
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or self.base_url or "",
                api_key=result.api_key or self.api_key or "",
                model_info=result.model_info,
            )
        except Exception:
            warning = None
        if warning is None:
            return True

        choices = [
            ("once", "Switch anyway", "Use this model for the current Hermes session."),
            ("cancel", "Cancel", "Keep the current model."),
        ]
        raw = self._prompt_text_input_modal(
            title=f"!!! {warning.title} !!!",
            detail=warning.message,
            choices=choices,
            timeout=120,
        )
        choice = self._normalize_slash_confirm_choice(raw, choices)
        return choice == "once"

    def _confirm_and_apply_model_switch_result(
        self, result, persist_global: bool, custom_providers=None
    ) -> None:
        try:
            if result.success and not self._confirm_expensive_model_switch(result):
                _cprint("  Model switch cancelled.")
                return
            self._apply_model_switch_result(
                result, persist_global, custom_providers=custom_providers
            )
        except Exception as exc:
            _cprint(f"  ✗ Model selection failed: {exc}")

    def _close_model_picker(self) -> None:
        self._model_picker_state = None
        self._restore_modal_input_snapshot()
        self._invalidate(min_interval=0.0)

    def _snapshot_model_runtime(self) -> dict:
        """Capture current CLI and agent model runtime for one-turn restore."""
        agent = getattr(self, "agent", None)
        return {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "_explicit_api_key": getattr(self, "_explicit_api_key", None),
            "_explicit_base_url": getattr(self, "_explicit_base_url", None),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "agent_primary_runtime": copy.deepcopy(
                getattr(agent, "_primary_runtime", None)
            ) if agent is not None else None,
        }

    def _restore_model_runtime_snapshot(self, snapshot: dict | None) -> None:
        """Restore a model runtime captured before a one-turn override."""
        if not snapshot:
            return
        for key in (
            "model",
            "provider",
            "requested_provider",
            "_explicit_api_key",
            "_explicit_base_url",
            "api_key",
            "base_url",
            "api_mode",
        ):
            if key in snapshot:
                setattr(self, key, snapshot.get(key))

        agent = getattr(self, "agent", None)
        if agent is None:
            return

        primary = snapshot.get("agent_primary_runtime")
        if primary and hasattr(agent, "_restore_primary_runtime"):
            try:
                agent._primary_runtime = copy.deepcopy(primary)
                agent._fallback_activated = True
                agent._rate_limited_until = 0
                if agent._restore_primary_runtime():
                    return
            except Exception:
                logger.debug("CLI one-turn model restore via primary runtime failed", exc_info=True)

        if hasattr(agent, "switch_model"):
            try:
                agent.switch_model(
                    new_model=snapshot.get("model", ""),
                    new_provider=snapshot.get("provider", ""),
                    api_key=snapshot.get("api_key", ""),
                    base_url=snapshot.get("base_url", ""),
                    api_mode=snapshot.get("api_mode", ""),
                )
            except Exception as exc:
                logger.warning("CLI one-turn model restore failed: %s", exc)

    @staticmethod
    def _filter_model_picker_entries(entries: list, query: str) -> list:
        """Return (original_index, label) pairs for entries matching ``query``.

        Subsequence ("fuzzy") match, case-insensitive: the query characters
        must appear in order in the label. An empty query matches everything.
        Crucially the returned pairs carry the ORIGINAL index into ``entries``,
        so a selection in the filtered view still resolves to exactly one
        concrete model — filtering only narrows the list, it never introduces
        an ambiguous or fuzzy *resolution* (the anti-"claude→old-model" rule).
        """
        pairs = list(enumerate(entries))
        q = (query or "").strip().lower()
        if not q:
            return pairs

        def _subseq(needle: str, hay: str) -> bool:
            it = iter(hay)
            return all(ch in it for ch in needle)

        out = [(i, e) for (i, e) in pairs if _subseq(q, str(e).lower())]
        return out

    @staticmethod
    def _compute_model_picker_viewport(
        selected: int,
        scroll_offset: int,
        n: int,
        term_rows: int,
        reserved_below: int = 6,
        panel_chrome: int = 6,
        min_visible: int = 3,
    ) -> tuple[int, int]:
        """Resolve (scroll_offset, visible) for the /model picker viewport.

        ``reserved_below`` matches the approval / clarify panels — input area,
        status bar, and separators below the panel. ``panel_chrome`` covers
        this panel's own borders + blanks + hint row. The remaining rows hold
        the scrollable list, with the offset slid to keep ``selected`` on screen.
        """
        max_visible = max(min_visible, term_rows - reserved_below - panel_chrome)
        if n <= max_visible:
            return 0, n
        visible = max_visible
        if selected < scroll_offset:
            scroll_offset = selected
        elif selected >= scroll_offset + visible:
            scroll_offset = selected - visible + 1
        scroll_offset = max(0, min(scroll_offset, n - visible))
        return scroll_offset, visible

    def _clear_persisted_context_for_model_switch(self, result) -> None:
        """Drop a global context pin when its configured owner changes."""
        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.route_identity import should_clear_context_pin

            config = load_config_readonly()
            model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
            if not isinstance(model_cfg, dict) or "context_length" not in model_cfg:
                return
            if should_clear_context_pin(
                model_cfg.get("default") or model_cfg.get("model"),
                result.new_model,
                model_cfg.get("base_url"),
                result.base_url,
                model_cfg.get("provider"),
                result.target_provider,
            ):
                save_config_value("model.context_length", None)
        except Exception:
            save_config_value("model.context_length", None)

    def _apply_model_switch_result(
        self, result, persist_global: bool, custom_providers=None
    ) -> None:
        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return

        if self.agent is not None:
            try:
                from hermes_cli.context_switch_guard import merge_preflight_compression_warning

                # Prefer the fresh inventory list (same source as switch_model /
                # TUI); fall back to the agent-init snapshot.
                _cp = (
                    custom_providers
                    if custom_providers is not None
                    else getattr(self.agent, "_custom_providers", None)
                )
                merge_preflight_compression_warning(
                    result,
                    agent=self.agent,
                    messages=list(self.conversation_history or []),
                    custom_providers=_cp,
                    config_context_length=getattr(self.agent, "_config_context_length", None),
                )
            except Exception as exc:
                logger.debug("preflight-compression switch warning failed: %s", exc)

        old_model = self.model
        # Snapshot the CLI-level credential/runtime fields BEFORE mutating them
        # so a failed in-place agent swap can roll the whole CLI back to the old
        # working model.  Otherwise the broken credentials staged below leak into
        # the next turn's resolution even though the agent itself rolled back
        # (#50163).
        _cli_snapshot = {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "_explicit_api_key": getattr(self, "_explicit_api_key", None),
            "_explicit_base_url": getattr(self, "_explicit_base_url", None),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
        }
        self.model = result.new_model
        self.provider = result.target_provider
        self.requested_provider = result.target_provider
        # Always overwrite explicit overrides so stale credentials from the
        # previous provider (e.g. Ollama api_key/base_url) don't leak into
        # the new provider's credential resolution on the next turn.
        self._explicit_api_key = result.api_key
        self._explicit_base_url = result.base_url
        if result.api_key:
            self.api_key = result.api_key
        if result.base_url:
            self.base_url = result.base_url
        if result.api_mode:
            self.api_mode = result.api_mode

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
                # The agent rolled itself back to the old working model/client.
                # Roll the CLI's own staged fields back too and abort the rest
                # of the commit (note + success print) so a failed switch is a
                # no-op rather than a dead session (#50163).
                for _k, _v in _cli_snapshot.items():
                    setattr(self, _k, _v)
                _cprint(
                    f"  ⚠ Model switch to {result.new_model} failed ({exc}); "
                    f"staying on {old_model}."
                )
                return

        from hermes_cli.model_switch import format_model_for_display
        _display_old = format_model_for_display(old_model)
        _display_new = format_model_for_display(result.new_model)

        self._pending_model_switch_note = (
            f"[Note: model was just switched from {_display_old} to {_display_new} "
            f"via {result.provider_label or result.target_provider}. "
            f"Adjust your self-identification accordingly.]"
        )

        provider_label = result.provider_label or result.target_provider
        _cprint(f"  ✓ Model switched: {_display_new}")
        _cprint(f"    Provider: {provider_label}")

        # Context: always resolve via the provider-aware chain so Codex OAuth,
        # Copilot, and Nous-enforced caps win over the raw models.dev entry
        # (e.g. gpt-5.5 is 1.05M on openai but 272K on Codex OAuth).
        mi = result.model_info
        try:
            from hermes_cli.model_switch import resolve_display_context_length
            ctx = resolve_display_context_length(
                result.new_model,
                result.target_provider,
                base_url=result.base_url or self.base_url or "",
                api_key=result.api_key or self.api_key or "",
                model_info=mi,
                config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None,
                custom_providers=getattr(self.agent, "_custom_providers", None) if self.agent else None,
            )
            if ctx:
                _cprint(f"    Context: {ctx:,} tokens")
        except Exception:
            pass
        if mi:
            if mi.max_output:
                _cprint(f"    Max output: {mi.max_output:,} tokens")
            _cprint(f"    Capabilities: {mi.format_capabilities()}")

        cache_enabled = (
            (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
            or result.api_mode == "anthropic_messages"
        )
        if cache_enabled:
            _cprint("    Prompt caching: enabled")
        if result.warning_message:
            _cprint(f"    ⚠ {result.warning_message}")
        if persist_global:
            HermesCLI._clear_persisted_context_for_model_switch(self, result)
            save_config_value("model.default", result.new_model)
            save_config_value("model.provider", result.target_provider)
            # base_url/api_mode were previously never persisted here, so a
            # global switch left the OLD provider's endpoint/wire-protocol in
            # config.yaml. result.base_url/api_mode are always freshly
            # resolved for the target provider (see model_switch.py), so sync
            # them every time; None clears a value the new provider doesn't
            # need (#25106).
            save_config_value("model.base_url", result.base_url or None)
            save_config_value("model.api_mode", result.api_mode or None)
            _cprint("    Saved to config.yaml (--global)")
        else:
            _cprint("    (session only — add --global to persist)")

        # Persist the switch to this session's row so --resume /
        # session.resume restore it. --global also updates config.yaml
        # (future sessions), but the row still records what THIS session
        # actually runs — otherwise a later resume would restore the stale
        # creation-time model over the user's new global choice.
        HermesCLI._persist_model_switch_to_session(self, result)

    def _handle_model_picker_selection(self, persist_global: bool = False) -> None:
        state = self._model_picker_state
        if not state:
            return
        selected = state.get("selected", 0)
        stage = state.get("stage")
        if stage == "provider":
            providers = state.get("providers") or []
            if selected >= len(providers):
                self._close_model_picker()
                return
            provider_data = providers[selected]
            # Use the curated model list from list_authenticated_providers()
            # (same lists as `hermes model` and gateway pickers).
            # Only fall back to the live provider catalog when the curated
            # list is empty (e.g. user-defined endpoints with no curated list).
            model_list = provider_data.get("models", [])
            if not model_list:
                try:
                    from hermes_cli.models import provider_model_ids
                    live = provider_model_ids(provider_data["slug"])
                    if live:
                        model_list = live
                except Exception:
                    pass
            state["stage"] = "model"
            state["provider_data"] = provider_data
            state["model_list"] = model_list
            state["selected"] = 0
            state["filter"] = ""
            state["_filtered_pairs"] = None
            self._invalidate(min_interval=0.0)
            return
        if stage == "model":
            provider_data = state.get("provider_data") or {}
            model_list = state.get("model_list") or []
            # Map the selected row through the active fuzzy filter so the
            # index lines up with what the picker is currently showing. The
            # filtered pair carries the ORIGINAL index into model_list, so the
            # resolved model is always one concrete, unambiguous entry.
            filtered_pairs = state.get("_filtered_pairs")
            if filtered_pairs is None:
                filtered_pairs = list(enumerate(model_list))
            visible_labels = [e for (_i, e) in filtered_pairs]
            back_idx = len(visible_labels)
            cancel_idx = len(visible_labels) + 1
            if selected == back_idx:
                state["stage"] = "provider"
                state["filter"] = ""
                state["_filtered_pairs"] = None
                state["selected"] = next((i for i, p in enumerate(state.get("providers") or []) if p.get("slug") == provider_data.get("slug")), 0)
                self._invalidate(min_interval=0.0)
                return
            if selected >= cancel_idx:
                self._close_model_picker()
                return
            if 0 <= selected < len(visible_labels):
                from hermes_cli.model_switch import switch_model
                chosen_model = visible_labels[selected]
                result = switch_model(
                    raw_input=chosen_model,
                    current_provider=self.provider or "",
                    current_model=self.model or "",
                    current_base_url=self.base_url or "",
                    current_api_key=self.api_key or "",
                    is_global=persist_global,
                    explicit_provider=provider_data.get("slug"),
                    user_providers=state.get("user_provs"),
                    custom_providers=state.get("custom_provs"),
                )
                # Capture before close — picker state is cleared on close.
                _picker_custom_provs = state.get("custom_provs")
                self._close_model_picker()
                if getattr(self, "_app", None):
                    threading.Thread(
                        target=self._confirm_and_apply_model_switch_result,
                        args=(result, persist_global, _picker_custom_provs),
                        daemon=True,
                    ).start()
                else:
                    self._confirm_and_apply_model_switch_result(
                        result, persist_global, custom_providers=_picker_custom_provs
                    )
                return
            self._close_model_picker()

    def _handle_model_switch(self, cmd_original: str):
        """Handle /model command — switch model.

        Supports:
          /model                              — show current model + usage hints
          /model <name>                       — switch model (this session only)
          /model <name> --once                — switch for the next turn only
          /model <name> --session             — switch for this session only (explicit)
          /model <name> --global              — switch and persist to config.yaml
          /model <name> --provider <provider> — switch provider + model
          /model --provider <provider>        — switch to provider, auto-detect model

        Persistence defaults to off (``model.persist_switch_by_default`` in
        config.yaml, default False — switches are session-scoped). Use
        ``--global`` to persist, or ``--once`` for the next turn only.
        """
        from hermes_cli.model_switch import (
            switch_model,
            parse_model_switch_args,
            resolve_persist_behavior,
        )
        from hermes_cli.providers import get_label

        # Parse args from the original command
        parts = cmd_original.split(None, 1)  # split off '/model'
        raw_args = parts[1].strip() if len(parts) > 1 else ""

        # Parse --provider, --global, --session, --once, and --refresh flags
        # via the shared single-owner parser (hermes_cli.model_switch).
        request = parse_model_switch_args(raw_args)
        model_input = request.target
        explicit_provider = request.explicit_provider
        is_global_flag = request.is_global
        force_refresh = request.force_refresh
        is_session = request.is_session
        one_turn = request.is_once
        if request.errors:
            # CLI decoration: "  ✗ " prefix over the canonical error copy.
            _cprint(f"  ✗ {request.error_messages()[0]}")
            return
        # Resolve the effective persistence once: --global forces persist,
        # --session/--once force session-scope, otherwise defer to
        # model.persist_switch_by_default (defaults to False so /model is
        # session-scoped unless the user opts in).
        persist_global = resolve_persist_behavior(
            is_global_flag, is_session, is_once=one_turn,
            explicit_provider=explicit_provider,
        )

        # --refresh: wipe the on-disk picker cache before building the
        # provider list. Forces a live re-fetch of every authed provider's
        # /v1/models endpoint on this open.
        if force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
                _cprint("  Cleared model picker cache. Refreshing...")
            except Exception:
                pass

        # Single inventory context — replaces the inline config-slice the
        # dashboard / TUI used to duplicate. Overlay live session state
        # via with_overrides (truthy-only) so empty self.* attrs don't
        # clobber disk config.
        from hermes_cli.inventory import build_models_payload, load_picker_context

        try:
            ctx = load_picker_context().with_overrides(
                current_provider=self.provider or "",
                current_model=self.model or "",
                current_base_url=self.base_url or "",
            )
        except Exception:
            ctx = None

        # switch_model() + _open_model_picker still need the raw provider
        # dicts; ConfigContext is the canonical source for both.
        user_provs = ctx.user_providers if ctx is not None else None
        custom_provs = ctx.custom_providers if ctx is not None else None

        # No args at all: open prompt_toolkit-native picker modal
        if not model_input and not explicit_provider:
            model_display = self.model or "unknown"
            provider_display = get_label(self.provider) if self.provider else "unknown"

            try:
                if ctx is None:
                    raise RuntimeError("inventory context unavailable")
                providers = build_models_payload(
                    ctx,
                    probe_custom_providers=force_refresh,
                    probe_current_custom_provider=not force_refresh,
                )["providers"]
            except Exception:
                providers = []

            if not providers:
                _cprint("  No authenticated providers found.")
                _cprint("")
                _cprint("  /model <name>                        switch model (persists)")
                _cprint("  /model <name> --once                 switch for the next turn only")
                _cprint("  /model <name> --session              switch for this session only")
                _cprint("  /model --provider <slug>             switch provider")
                _cprint("  /model --refresh                     re-fetch live model lists")
                return

            self._open_model_picker(
                providers,
                model_display,
                provider_display,
                user_provs=user_provs,
                custom_provs=custom_provs,
            )
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
            user_providers=user_provs,
            custom_providers=custom_provs,
        )

        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return

        if self.agent is not None:
            try:
                from hermes_cli.context_switch_guard import merge_preflight_compression_warning

                merge_preflight_compression_warning(
                    result,
                    agent=self.agent,
                    messages=list(self.conversation_history or []),
                    # Same fresh inventory list passed to switch_model above.
                    custom_providers=custom_provs
                    if custom_provs is not None
                    else getattr(self.agent, "_custom_providers", None),
                    config_context_length=getattr(self.agent, "_config_context_length", None),
                )
            except Exception as exc:
                logger.debug("preflight-compression switch warning failed: %s", exc)

        # Run the confirm + apply sequence off the main thread. The
        # expensive-model confirmation modal blocks the calling thread on a
        # response queue (see _prompt_text_input_modal); running it on the
        # prompt_toolkit main thread freezes TUI rendering, so the modal never
        # appears and the switch silently cancels after the 120s timeout.
        # Mirror the picker path (_handle_model_picker_selection), which
        # already dispatches confirm+apply on a worker thread.
        if getattr(self, "_app", None):
            threading.Thread(
                target=self._confirm_and_apply_cli_model_switch,
                args=(result, persist_global, one_turn, custom_provs),
                daemon=True,
            ).start()
            return
        self._confirm_and_apply_cli_model_switch(
            result, persist_global, one_turn, custom_provs
        )
        return

    def _confirm_and_apply_cli_model_switch(
        self, result, persist_global: bool, one_turn: bool, custom_provs=None
    ) -> None:
        """Confirm an expensive model switch and apply it to CLI state.

        Runs on a worker thread when the TUI is active (see
        _handle_model_switch) so the confirmation modal can render.
        """
        if not self._confirm_expensive_model_switch(result):
            _cprint("  Model switch cancelled.")
            return

        # Apply to CLI state.
        # Update requested_provider so _ensure_runtime_credentials() doesn't
        # overwrite the switch on the next turn (it re-resolves from this).
        old_model = self.model
        _one_turn_restore_snapshot = self._snapshot_model_runtime() if one_turn else None
        # Snapshot CLI-level fields before mutation so a failed in-place swap
        # rolls the whole CLI back to the old working model (#50163).
        _cli_snapshot = {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "_explicit_api_key": getattr(self, "_explicit_api_key", None),
            "_explicit_base_url": getattr(self, "_explicit_base_url", None),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
        }
        self.model = result.new_model
        self.provider = result.target_provider
        self.requested_provider = result.target_provider
        # Always overwrite explicit overrides so stale credentials from the
        # previous provider (e.g. Ollama api_key/base_url) don't leak into
        # the new provider's credential resolution on the next turn.
        self._explicit_api_key = result.api_key
        self._explicit_base_url = result.base_url
        if result.api_key:
            self.api_key = result.api_key
        if result.base_url:
            self.base_url = result.base_url
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
                # Agent rolled itself back; roll the CLI back too and abort so a
                # failed switch is a no-op rather than a dead session (#50163).
                for _k, _v in _cli_snapshot.items():
                    setattr(self, _k, _v)
                _cprint(
                    f"  ⚠ Model switch to {result.new_model} failed ({exc}); "
                    f"staying on {old_model}."
                )
                return

        # Store a note to prepend to the next user message so the model
        # knows a switch occurred (avoids injecting system messages mid-history
        # which breaks providers and prompt caching).
        from hermes_cli.model_switch import format_model_for_display
        _display_old = format_model_for_display(old_model)
        _display_new = format_model_for_display(result.new_model)

        self._pending_model_switch_note = (
            f"[Note: model was just switched from {_display_old} to {_display_new} "
            f"via {result.provider_label or result.target_provider}. "
            f"{'This override applies to the next turn only. ' if one_turn else ''}"
            f"Adjust your self-identification accordingly.]"
        )
        if one_turn:
            self._pending_one_turn_model_restore = _one_turn_restore_snapshot
        else:
            self._pending_one_turn_model_restore = None

        # Display confirmation with full metadata
        provider_label = result.provider_label or result.target_provider
        _cprint(f"  ✓ Model switched: {_display_new}")
        _cprint(f"    Provider: {provider_label}")

        # Context: always resolve via the provider-aware chain so Codex OAuth,
        # Copilot, and Nous-enforced caps win over the raw models.dev entry
        # (e.g. gpt-5.5 is 1.05M on openai but 272K on Codex OAuth).
        mi = result.model_info
        from hermes_cli.model_switch import resolve_display_context_length
        ctx = resolve_display_context_length(
            result.new_model,
            result.target_provider,
            base_url=result.base_url or self.base_url or "",
            api_key=result.api_key or self.api_key or "",
            model_info=mi,
            config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None,
            custom_providers=getattr(self.agent, "_custom_providers", None) if self.agent else None,
        )
        if ctx:
            _cprint(f"    Context: {ctx:,} tokens")
        if mi:
            if mi.max_output:
                _cprint(f"    Max output: {mi.max_output:,} tokens")
            _cprint(f"    Capabilities: {mi.format_capabilities()}")

        # Cache notice
        cache_enabled = (
            (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
            or result.api_mode == "anthropic_messages"
        )
        if cache_enabled:
            _cprint("    Prompt caching: enabled")

        # Warning from validation
        if result.warning_message:
            _cprint(f"    ⚠ {result.warning_message}")

        # Persistence
        if persist_global:
            HermesCLI._clear_persisted_context_for_model_switch(self, result)
            save_config_value("model.default", result.new_model)
            save_config_value("model.provider", result.target_provider)
            # See _apply_model_switch_result above for why base_url/api_mode
            # must be synced on every global switch (#25106).
            save_config_value("model.base_url", result.base_url or None)
            save_config_value("model.api_mode", result.api_mode or None)
            _cprint("    Saved to config.yaml")
        elif one_turn:
            _cprint("    (next turn only — restores after one response)")
        else:
            _cprint("    (session only — add --global to persist)")

        # Persist the switch to this session's row so --resume /
        # session.resume restore it (--global also updates config.yaml but
        # the row still records what THIS session runs; --once is ephemeral
        # and restored after one turn, so it must not touch the row).
        if not one_turn:
            HermesCLI._persist_model_switch_to_session(self, result)

    def _handle_codex_runtime(self, cmd_original: str) -> None:
        """Handle /codex-runtime — toggle the codex app-server runtime opt-in.

        Usage:
            /codex-runtime                       — show current state
            /codex-runtime auto                  — Hermes default (chat_completions)
            /codex-runtime codex_app_server      — hand turns to codex subprocess
            /codex-runtime on / off              — synonyms for the above
        """
        from hermes_cli import codex_runtime_switch as crs

        parts = cmd_original.split(None, 1)
        raw_args = parts[1].strip() if len(parts) > 1 else ""
        new_value, errors = crs.parse_args(raw_args)
        if errors:
            for err in errors:
                _cprint(f"❌ {err}")
            return

        # Load + persist via the existing config helpers
        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:
            _cprint(f"❌ could not load config: {exc}")
            return
        cfg = load_config()

        result = crs.apply(
            cfg,
            new_value,
            persist_callback=(save_config if new_value is not None else None),
        )

        prefix = "✓" if result.success else "✗"
        for line in result.message.splitlines():
            _cprint(f"  {prefix} {line}" if line.startswith("openai_runtime")
                    else f"    {line}")
        if result.success and result.requires_new_session:
            _cprint("    Tip: `/reset` starts a new session immediately.")

    def _should_handle_model_command_inline(self, text: str, has_images: bool = False) -> bool:
        """Return True when /model should be handled immediately on the UI thread."""
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        try:
            from hermes_cli.commands import resolve_command
            base = text.split(None, 1)[0].lower().lstrip('/')
            cmd = resolve_command(base)
            return bool(cmd and cmd.name == "model")
        except Exception:
            return False

    def _should_handle_steer_command_inline(self, text: str, has_images: bool = False) -> bool:
        """Return True when /steer should be dispatched immediately while the agent is running.

        /steer MUST bypass the normal _pending_input → process_loop path when
        the agent is active, because process_loop is blocked inside
        self.chat() for the duration of the run.  By the time the queued
        command is pulled from _pending_input, _agent_running has already
        flipped back to False, and process_command() takes the idle
        fallback — delivering the steer as a next-turn message instead of
        injecting it mid-run.  Dispatching inline on the UI thread calls
        agent.steer() directly, which is thread-safe (uses _pending_steer_lock).
        """
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        if not getattr(self, "_agent_running", False):
            return False
        try:
            from hermes_cli.commands import resolve_command
            base = text.split(None, 1)[0].lower().lstrip('/')
            cmd = resolve_command(base)
            return bool(cmd and cmd.name == "steer")
        except Exception:
            return False

    def _should_handle_background_command_inline(
        self, text: str, has_images: bool = False
    ) -> bool:
        """Return True when /background should be dispatched while the agent runs.

        Same queue problem /steer had. ``/background`` (``/bg``, ``/btw``)
        exists to start independent work *without* waiting for the current
        turn, but a slash command typed while the agent is busy goes into
        ``_pending_input``, and ``process_loop`` is blocked inside
        ``self.chat()`` for the whole run. The background task therefore only
        starts once the foreground turn has finished, which is the one moment
        it was not needed.

        The command's own ``CommandDef`` already declares
        ``busy_policy="dispatch"``; the gateway honours that, the classic CLI
        never consulted it. Dispatching inline on the UI thread starts the
        background session immediately and leaves the foreground turn running
        untouched: no interrupt, no steer.
        """
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        if not getattr(self, "_agent_running", False):
            return False
        try:
            from hermes_cli.commands import resolve_command
            base = text.split(None, 1)[0].lower().lstrip('/')
            cmd = resolve_command(base)
            return bool(cmd and cmd.name == "background")
        except Exception:
            return False

    def _output_console(self):
        """Use prompt_toolkit-safe Rich rendering once the TUI is live."""
        if getattr(self, "_app", None):
            return ChatConsole()
        return self.console

    def _console_print(self, *args, **kwargs):
        """Print through the active command-safe console."""
        self._output_console().print(*args, **kwargs)

    def handle_bang_shell(self, text: str) -> bool:
        """Run a ``!<command>`` submission. Returns True when it was handled.

        Dispatched from the input loop BEFORE slash-command routing and before
        anything is queued for the agent, so a bang command never becomes a
        turn: no user message, no assistant message, no tool result touches
        ``self.conversation_history``. That is what makes ``!`` free — zero
        tokens, and role alternation / prompt caching are untouched by
        construction. The invariant is covered by
        tests/cli/test_bang_shell_mode.py.

        Returns False when the text is not a bang command or when bang mode is
        disabled for this context (gateway/cron), letting the caller fall
        through to normal routing.
        """
        from hermes_cli.bang_shell import (
            USAGE_HINT,
            bang_shell_enabled,
            check_bang_approval,
            is_bang_command,
            parse_bang_command,
            resolve_bang_cwd,
            run_bang_command,
        )

        if not is_bang_command(text):
            return False
        if not bang_shell_enabled():
            # Gateway / cron / API contexts: no composer, no human at a
            # keyboard, and those users already have their own shells. Let the
            # text route normally rather than becoming remote execution.
            return False

        command = parse_bang_command(text)
        if not command:
            # Bare `!` — show what the feature does instead of running an
            # empty shell or sending "!" to the model.
            self._console_print(f"[dim]{USAGE_HINT}[/]")
            return True

        approval = check_bang_approval(command)
        if not approval.get("approved"):
            message = approval.get("message") or (
                f"Command denied: {approval.get('description', 'flagged as dangerous')}"
            )
            self._console_print(f"[bold red]{_escape(str(message))}[/]")
            return True

        cwd = resolve_bang_cwd(getattr(self, "session_id", None))
        exit_code = run_bang_command(
            command,
            cwd=cwd,
            writer=lambda line: self._console_print(_rich_text_from_ansi(line)),
        )
        if exit_code:
            self._console_print(f"[dim]! exited {exit_code}[/]")
        return True

    @staticmethod
    def _resolve_personality_prompt(value) -> str:
        """Accept string or dict personality value; return system prompt string.

        Delegates to hermes_cli.personality (single owner of rendering).
        """
        from hermes_cli.personality import render_personality_prompt

        return render_personality_prompt(value)


    



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
            
            print("  Messaging Platform Configuration:")
            print("  " + "-" * 55)
            
            platform_status = {
                Platform.TELEGRAM: ("Telegram", "TELEGRAM_BOT_TOKEN"),
                Platform.DISCORD: ("Discord", "DISCORD_BOT_TOKEN"),
                Platform.SLACK: ("Slack", "SLACK_BOT_TOKEN"),
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
            print(f"  -- {name} --")
            for label, value in rows:
                print(f"  {label} {value}")
        print()

    # canonical command -> (method name, pass cmd_original?). Absent commands resolve to
    # ``_handle_<name>_command(cmd)``. Looked up via getattr at dispatch time so
    # monkeypatching works. A handler returning False exits the REPL.
    _SLASH_DISPATCH: dict[str, tuple[str, bool]] = {
        "exit": ("_cmd_exit", True), "quit": ("_cmd_exit", True), "help": ("_cmd_help", True),
        "palette": ("_open_command_palette", False), "whoami": ("_handle_whoami_command", False),
        "profile": ("_handle_profile_command", False), "toolsets": ("show_toolsets", False),
        "config": ("show_config", False), "redraw": ("_cmd_redraw", True), "clear": ("_cmd_clear", True),
        "history": ("show_history", False), "title": ("_cmd_title", True), "new": ("_cmd_new", True),
        "model": ("_handle_model_switch", True), "codex-runtime": ("_handle_codex_runtime", True),
        "retry": ("_cmd_retry", True), "prompt": ("_handle_prompt_compose_command", True),
        "undo": ("_cmd_undo", True), "save": ("save_conversation", True), "skills": ("_cmd_skills", True),
        "platforms": ("_show_gateway_status", False), "status": ("_show_session_status", False),
        "context": ("_show_context_breakdown", True), "egress": ("_cmd_egress", True),
        "statusbar": ("_cmd_statusbar", True), "verbose": ("_toggle_verbose", False), "yolo": ("_toggle_yolo", False),
        "compress": ("_manual_compress", True), "subscription": ("_show_subscription", False),
        "topup": ("_show_billing", True), "insights": ("_show_insights", True), "update": ("_cmd_update", True),
        "version": ("_cmd_version", True), "paste": ("_handle_paste_command", False), "reload": ("_cmd_reload", True),
        "reload-mcp": ("_confirm_and_reload_mcp", True), "reload-skills": ("_cmd_reload_skills", True),
        "plugins": ("_cmd_plugins", True), "stop": ("_handle_stop_command", False),
        "agents": ("_handle_agents_command", False), "bg": ("_handle_background_command", True),
        "queue": ("_cmd_queue", True), "steer": ("_cmd_steer", True), "moa": ("_cmd_moa", True),
    }

    @classmethod
    def _slash_handler(cls, canonical: str) -> tuple[str, bool] | None:
        """(method name, pass cmd_original?) for a registered command, else None."""
        entry = cls._SLASH_DISPATCH.get(canonical)
        if entry is None:
            name = f"_handle_{canonical.replace('-', '_')}_command"
            if callable(getattr(cls, name, None)):
                entry = (name, True)
        return entry

    def process_command(self, command: str) -> bool:
        """Dispatch a slash command; returns False to exit the REPL."""
        cmd_lower = command.lower().strip()  # lowercase only for matching; args keep their case
        cmd_original = command.strip()

        # Aliases resolve via the central registry (hermes_cli/commands.py).
        from hermes_cli.commands import resolve_command as _resolve_cmd
        _base_word = cmd_lower.split()[0].lstrip("/")
        _cmd_def = _resolve_cmd(_base_word)
        canonical = _cmd_def.name if _cmd_def else _base_word

        # Observer-only pre_command plugin hook (return values ignored; never raises).
        if _cmd_def is not None:
            from hermes_cli.plugins import fire_pre_command_hook
            fire_pre_command_hook(
                surface="cli", command=canonical, alias_used=_base_word, args_raw=_slash_args(cmd_original),
                session_key=getattr(self, "session_id", None), platform="cli",
            )

        # A bare `/resume` prompt is one-shot: any other command disarms it so a later
        # number isn't swallowed as a stale selection.
        # See #34584.
        if canonical not in {"resume", "sessions"}:
            # Armed when a bare `/resume` prints the recent-sessions list so the very next bare numeric
            # input (e.g. `3`) resolves to that session. Holds the exact list used for index resolution;
            # one-shot (cleared on the next submitted input, whether it's the selection or anything else).
            # See #34584.
            self._pending_resume_sessions = None

        if canonical in {"quit", "exit"}:
            # Parse --delete flag: /exit --delete also removes the current
            # session's transcripts + SQLite history. Ported from
            # google-gemini/gemini-cli#19332.
            _rest = cmd_original.split(None, 1)
            _args = (_rest[1] if len(_rest) > 1 else "").strip().lower()
            if _args in {"--delete", "-d"}:
                self._delete_session_on_exit = True
            elif _args:
                _cprint(f"  {_DIM}✗ Unknown argument: {_escape(_args)}. Use /exit --delete to also remove session history.{_RST}")
                return True
            return False
        elif canonical == "help":
            _help_parts = cmd_original.split(None, 1)
            self.show_help(_help_parts[1].strip() if len(_help_parts) > 1 else "")
        elif canonical == "palette":
            self._open_command_palette()
        elif canonical == "whoami":
            self._handle_whoami_command()
        elif canonical == "profile":
            self._handle_profile_command()
        elif canonical == "tools":
            self._handle_tools_command(cmd_original)
        elif canonical == "toolsets":
            self.show_toolsets()
        elif canonical == "config":
            self.show_config()
        elif canonical == "redraw":
            # Manual recovery for terminal buffer drift from multiplexer
            # tab switches, subshell ``clear``, SSH window restores, etc.
            # See issue #8688 (cmux). Ctrl+L is bound to the same helper.
            self._force_full_redraw()
            _cprint(f"  {_DIM}✓ UI redrawn{_RST}")
        elif canonical == "clear":
            if self._confirm_destructive_slash(
                "clear",
                "This clears the screen and starts a new session.\n"
                "The current conversation history will be discarded.",
                cmd_original=cmd_original,
            ) is None:
                return True  # confirmation cancelled — command handled, keep REPL alive
            self.new_session(silent=True)
            _clear_output_history()
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
                        provider=self.provider,
                    )
                _cprint("  ✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n")
                # Show a random tip on new session
                try:
                    from hermes_cli.tips import get_random_tip
                    _tip = get_random_tip()
                    try:
                        from hermes_cli.skin_engine import get_active_skin
                        _tip_color = get_active_skin().get_color("banner_dim", "#B8860B")
                    except Exception:
                        _tip_color = "#B8860B"
                    cc.print(f"[dim {_tip_color}]✦ Tip: {_tip}[/]")
                except Exception:
                    pass
            else:
                self.show_banner()
                print("  ✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n")
                # Show a random tip on new session
                try:
                    from hermes_cli.tips import get_random_tip
                    _tip = get_random_tip()
                    try:
                        from hermes_cli.skin_engine import get_active_skin
                        _tip_color = get_active_skin().get_color("banner_dim", "#B8860B")
                    except Exception:
                        _tip_color = "#B8860B"
                    self._console_print(f"[dim {_tip_color}]✦ Tip: {_tip}[/]")
                except Exception:
                    pass
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
                            # sanitize_title rejected the input (e.g. too long).
                            # Print that one reason and stop — don't fall
                            # through to the "empty after cleanup" branch and
                            # print a second, contradictory error (SC-05).
                            _cprint(f"  {e}")
                            return True
                        if not new_title:
                            _cprint("  Title is empty after cleanup. Please use printable characters.")
                        elif self._session_db.get_session(self.session_id):
                            # Session exists in DB — set title directly
                            try:
                                if self._session_db.set_session_title(self.session_id, new_title):
                                    self._status_bar_title_checked_at = 0.0
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
                        from hermes_state import format_session_db_unavailable
                        _cprint(f"  {format_session_db_unavailable()}")
                else:
                    _cprint("  Usage: /title <your session title>")
            # Show current title and session ID if no argument given
            elif self._session_db:
                _cprint(f"  Session ID: {self.session_id}")
                session = self._session_db.get_session(self.session_id)
                if session and session.get("title"):
                    _cprint(f"  Title: {session['title']}")
                elif self._pending_title:
                    _cprint(f"  Title (pending): {self._pending_title}")
                else:
                    _cprint("  No title set. Usage: /title <your session title>")
            else:
                from hermes_state import format_session_db_unavailable
                _cprint(f"  {format_session_db_unavailable()}")
        elif canonical == "handoff":
            if not self._handle_handoff_command(cmd_original):
                return False
        elif canonical == "new":
            # Strip inline-skip tokens (now/--yes/-y) before deriving the title
            # so "/new now My Session" yields title="My Session" instead of
            # title="now My Session". See _split_destructive_skip.
            _new_args, _ = self._split_destructive_skip(cmd_original)
            title = _new_args.strip() or None
            if self._confirm_destructive_slash(
                "new",
                "This starts a fresh session.\n"
                "The current conversation history will be discarded.",
                cmd_original=cmd_original,
            ) is None:
                return True  # confirmation cancelled — command handled, keep REPL alive
            self.new_session(title=title)
        elif canonical == "resume":
            self._handle_resume_command(cmd_original)
        elif canonical == "sessions":
            self._handle_sessions_command(cmd_original)
        elif canonical == "model":
            self._handle_model_switch(cmd_original)
        elif canonical == "codex-runtime":
            self._handle_codex_runtime(cmd_original)

        elif canonical == "personality":
            # Use original case (handler lowercases the personality name itself)
            self._handle_personality_command(cmd_original)
        elif canonical == "pet":
            self._handle_pet_command(cmd_original)

        elif canonical == "hatch":
            self._handle_hatch_command(cmd_original)
        elif canonical == "retry":
            retry_msg = self.retry_last()
            if retry_msg and hasattr(self, '_pending_input'):
                # Re-queue the message so process_loop sends it to the agent
                self._pending_input.put(retry_msg)
        elif canonical == "prompt":
            self._handle_prompt_compose_command(cmd_original)
        elif canonical == "undo":
            # Parse optional turn count: "/undo" → 1, "/undo 3" → 3.
            _undo_n = 1
            _undo_parts = cmd_original.split()
            if len(_undo_parts) > 1:
                try:
                    _undo_n = int(_undo_parts[1])
                except ValueError:
                    print(f"(._.) Invalid count {_undo_parts[1]!r} — use /undo or /undo N.")
                    return True  # bad arg — command handled, keep the REPL alive
                if _undo_n < 1:
                    _undo_n = 1
            # Nothing to undo → say so immediately; don't pop a destructive
            # confirmation dialog for a guaranteed no-op (SC-06).
            if not self.conversation_history:
                print("(._.) No messages to undo.")
                return True
            _undo_desc = (
                "This removes the last user/assistant exchange from history."
                if _undo_n == 1
                else f"This removes the last {_undo_n} user turns from history."
            )
            if self._confirm_destructive_slash(
                "undo",
                _undo_desc,
                cmd_original=cmd_original,
            ) is None:
                return True  # confirmation cancelled — command handled, keep REPL alive
            self.undo_last(_undo_n)
        elif canonical == "branch":
            self._handle_branch_command(cmd_original)
        elif canonical == "worktree":
            self._handle_worktree_command(cmd_original)
        elif canonical == "save":
            self.save_conversation(cmd_original)
        elif canonical == "cron":
            self._handle_cron_command(cmd_original)
        elif canonical == "suggestions":
            self._handle_suggestions_command(cmd_original)
        elif canonical == "blueprint":
            self._handle_blueprint_command(cmd_original)
        elif canonical == "curator":
            self._handle_curator_command(cmd_original)
        elif canonical == "kanban":
            self._handle_kanban_command(cmd_original)
        elif canonical == "skills":
            with self._busy_command(self._slow_command_status(cmd_original)):
                self._handle_skills_command(cmd_original)
        elif canonical == "learn":
            self._handle_learn_command(cmd_original)
        elif canonical == "init":
            self._handle_init_command(cmd_original)
        elif canonical == "memory":
            self._handle_memory_command(cmd_original)
        elif canonical == "platforms":
            self._show_gateway_status()
        elif canonical == "status":
            self._show_session_status()
        elif canonical == "context":
            self._show_context_breakdown(cmd_original)
        elif canonical == "egress":
            from hermes_cli.slash_exec import CommandContext, execute_command

            self._console_print(
                execute_command("egress", CommandContext(surface="cli")).text,
                highlight=False, markup=False,
            )
        elif canonical == "statusbar":
            self._status_bar_visible = not self._status_bar_visible
            state = "visible" if self._status_bar_visible else "hidden"
            self._console_print(f"  Status bar {state}")
        elif canonical == "diff":
            self._handle_diff_command(cmd_original)
        elif canonical == "battery":
            self._handle_battery_command(cmd_original)
        elif canonical == "timestamps":
            self._handle_timestamps_command(cmd_original)
        elif canonical == "verbose":
            self._toggle_verbose()
        elif canonical == "focus":
            self._handle_focus_command(cmd_original)
        elif canonical == "footer":
            self._handle_footer_command(cmd_original)
        elif canonical == "yolo":
            self._toggle_yolo()
        elif canonical == "approvals":
            self._handle_approvals_command(cmd_original)
        elif canonical == "reasoning":
            self._handle_reasoning_command(cmd_original)
        elif canonical == "fast":
            self._handle_fast_command(cmd_original)
        elif canonical == "compress":
            self._manual_compress(cmd_original)
        elif canonical == "usage":
            self._handle_usage_command(cmd_original)
        elif canonical == "subscription":
            self._show_subscription()
        elif canonical == "topup":
            self._show_billing(cmd_original)
        elif canonical == "insights":
            self._show_insights(cmd_original)
        elif canonical == "copy":
            self._handle_copy_command(cmd_original)
        elif canonical == "debug":
            self._handle_debug_command(cmd_original)
        elif canonical == "update":
            if self._handle_update_command():
                return False
        elif canonical == "version":
            from hermes_cli.main import _print_version_info

            _print_version_info(check_updates=True)
        elif canonical == "paste":
            self._handle_paste_command()
        elif canonical == "image":
            self._handle_image_command(cmd_original)
        elif canonical == "reload":
            from hermes_cli.config import reload_env
            count = reload_env()
            print(f"  Reloaded .env ({count} var(s) updated)")
        elif canonical == "reload-mcp":
            # Interactive reload: confirm first (unless the user has opted out).
            # The auto-reload path (file watcher) calls _reload_mcp directly
            # without this confirmation.
            self._confirm_and_reload_mcp(cmd_original)
        elif canonical == "reload-skills":
            with self._busy_command(self._slow_command_status(cmd_original)):
                self._reload_skills()
        elif canonical == "bundles":
            self._handle_bundles_command(cmd_original)
        elif canonical == "browser":
            self._handle_browser_command(cmd_original)
        elif canonical == "plugins":
            try:
                # Discover from disk (bundled + user), matching `hermes plugins
                # list` — so installed-but-not-enabled plugins are visible here
                # too. The plugin manager only knows about *loaded* plugins, so
                # using it alone made freshly-installed, not-yet-enabled plugins
                # look like "nothing installed".
                from hermes_cli.plugins_cmd import (
                    _discover_all_plugins,
                    _get_disabled_set,
                    _get_enabled_set,
                    _plugin_status,
                )

                entries = _discover_all_plugins()
                enabled = _get_enabled_set()
                disabled = _get_disabled_set()

                # `/plugins` is a quick glance — default to user-installed
                # plugins (what the user actually added). Bundled provider/
                # platform plugins are summarized on one line; the full
                # catalog lives behind `hermes plugins list`.
                user_entries = [e for e in entries if e[3] != "bundled"]
                bundled_count = len(entries) - len(user_entries)

                if not user_entries:
                    print("No user plugins installed.")
                    print("  Install one: hermes plugins install owner/repo")
                    print(f"  Or drop a plugin directory into {display_hermes_home()}/plugins/")
                    if bundled_count:
                        print(f"  ({bundled_count} bundled plugins available — see: hermes plugins list)")
                else:
                    # Loaded-plugin details (tools/hooks/commands counts, errors)
                    # keyed by name, when available.
                    loaded: dict = {}
                    try:
                        from hermes_cli.plugins import get_plugin_manager
                        for p in get_plugin_manager().list_plugins():
                            loaded[p["name"]] = p
                    except Exception:
                        loaded = {}

                    print(f"User plugins ({len(user_entries)}):")
                    for name, version, _desc, source, _dir, key in sorted(user_entries):
                        state = _plugin_status(name, enabled, disabled, key=key)
                        glyph = {"enabled": "✓", "disabled": "✗"}.get(state, "○")
                        ver = f" v{version}" if version else ""
                        info = loaded.get(name) or {}
                        bits = []
                        if info.get("tools"):
                            bits.append(f"{info['tools']} tools")
                        if info.get("hooks"):
                            bits.append(f"{info['hooks']} hooks")
                        if info.get("commands"):
                            bits.append(f"{info['commands']} commands")
                        detail = f" ({', '.join(bits)})" if bits else ""
                        label = "" if state == "enabled" else f" [{state}]"
                        error = f" — {info['error']}" if info.get("error") else ""
                        print(f"  {glyph} {name}{ver}{label}{detail}{error}")
                    if bundled_count:
                        print(f"  (+{bundled_count} bundled — see: hermes plugins list)")
                    print("  Enable/disable: hermes plugins enable/disable <name>")
            except Exception as e:
                print(f"Plugin system error: {e}")
        elif canonical == "rollback":
            self._handle_rollback_command(cmd_original)
        elif canonical == "snapshot":
            self._handle_snapshot_command(cmd_original)
        elif canonical == "export":
            self._handle_export_command(cmd_original)
        elif canonical == "import":
            self._handle_import_command(cmd_original)
        elif canonical == "stop":
            self._handle_stop_command()
        elif canonical == "agents":
            self._handle_agents_command()
        elif canonical == "journey":
            self._handle_journey_command(cmd_original)
        elif canonical == "background":
            self._handle_background_command(cmd_original)
        elif canonical == "queue":
            # Extract prompt after "/queue " or "/q "
            parts = cmd_original.split(None, 1)
            payload = parts[1].strip() if len(parts) > 1 else ""
            payload = self._expand_paste_references(payload)
            if not payload:
                _cprint("  Usage: /queue <prompt>")
            else:
                self._pending_input.put(payload)
                if self._agent_running:
                    _cprint(f"  Queued for the next turn: {payload[:80]}{'...' if len(payload) > 80 else ''}")
                else:
                    _cprint(f"  Queued: {payload[:80]}{'...' if len(payload) > 80 else ''}")
        elif canonical == "steer":
            # Inject a message after the next tool call without interrupting.
            # If the agent is actively running, push the text into the agent's
            # pending_steer slot — the drain hook in _execute_tool_calls_*
            # will append it to the next tool result's content. If no agent
            # is running, fall back to queue semantics (same as /queue).
            parts = cmd_original.split(None, 1)
            payload = parts[1].strip() if len(parts) > 1 else ""
            if not payload:
                _cprint("  Usage: /steer <prompt>")
            elif self._agent_running and self.agent is not None and hasattr(self.agent, "steer"):
                try:
                    accepted = self.agent.steer(payload)
                except Exception as exc:
                    _cprint(f"  Steer failed: {exc}")
                else:
                    if accepted:
                        _cprint(f"  ⏩ Steer queued — arrives after the next tool call: {payload[:80]}{'...' if len(payload) > 80 else ''}")
                    else:
                        _cprint("  Steer rejected (empty payload).")
            else:
                # No active run — treat as a normal next-turn message.
                self._pending_input.put(payload)
                _cprint(f"  No agent running; queued as next turn: {payload[:80]}{'...' if len(payload) > 80 else ''}")
        elif canonical == "goal":
            self._handle_goal_command(cmd_original)
        elif canonical == "heartbeat":
            self._handle_heartbeat_command(cmd_original)
        elif canonical == "refine":
            self._handle_refine_command(cmd_original)
        elif canonical == "review":
            self._handle_review_command(cmd_original)
        elif canonical == "loop":
            self._handle_loop_command(cmd_original)
        elif canonical == "moa":
            # /moa is one-shot sugar only: run a single prompt through the
            # default MoA preset, then restore the prior model. To *switch* to a
            # MoA preset for the session, pick it from the model picker (MoA
            # presets surface as a virtual "Mixture of Agents" provider).
            from hermes_cli.moa_config import (
                moa_usage,
                normalize_moa_config,
            )

            parts = cmd_original.split(None, 1)
            payload = parts[1].strip() if len(parts) > 1 else ""
            if not payload:
                _cprint(f"  {moa_usage()}")
                return True
            moa_cfg = self.config.get("moa") if isinstance(self.config, dict) else {}
            normalized = normalize_moa_config(moa_cfg)
            preset = normalized["default_preset"]
            self._pending_moa_restore_model = {
                "requested_provider": getattr(self, "requested_provider", None),
                "provider": getattr(self, "provider", None),
                "model": getattr(self, "model", None),
                "api_key": getattr(self, "api_key", None),
                "base_url": getattr(self, "base_url", None),
                "api_mode": getattr(self, "api_mode", None),
            }
            self.requested_provider = "moa"
            self.provider = "moa"
            self.model = preset
            self.api_key = "moa-virtual-provider"
            self.base_url = "moa://local"
            self.api_mode = "chat_completions"
            self.agent = None
            self._pending_moa_disable_after_turn = True
            self._pending_agent_seed = payload
            _cprint(f"  MoA one-shot queued with preset {preset}; previous model will be restored after this turn.")
        elif canonical == "subgoal":
            self._handle_subgoal_command(cmd_original)
        elif canonical == "skin":
            self._handle_skin_command(cmd_original)
        elif canonical == "voice":
            self._handle_voice_command(cmd_original)
        elif canonical == "wake":
            self._handle_wake_command(cmd_original)
        elif canonical == "busy":
            self._handle_busy_command(cmd_original)
        elif canonical == "indicator":
            self._handle_indicator_command(cmd_original)
        else:
            return self._expand_slash_prefix(cmd_original, cmd_lower, skill_commands, skill_bundles)
        return True

    def _run_quick_command(self, base_cmd: str, qcmd: dict, user_args: str) -> bool:
        """User-defined quick command (config.yaml): ``exec`` runs a shell snippet, ``alias`` re-dispatches."""
        qtype = qcmd.get("type")
        if qtype == "alias":
            target = qcmd.get("target", "").strip()
            if target:
                target = target if target.startswith("/") else f"/{target}"
                return self.process_command(f"{target} {user_args}".strip())
            self._console_print(f"[bold red]Quick command '{base_cmd}' has no target defined[/]")
            return True
        if qtype != "exec":
            self._console_print(f"[bold red]Quick command '{base_cmd}' has unsupported type (supported: 'exec', 'alias')[/]")
            return True
        import subprocess
        exec_cmd = qcmd.get("command", "")
        if not exec_cmd:
            self._console_print(f"[bold red]Quick command '{base_cmd}' has no command defined[/]")
            return True
        try:
            # shell=True is intentional (user-authored config snippets, never LLM controlled);
            # the env is sanitized because this process holds every API key.
            from tools.environments.local import build_subprocess_env
            from hermes_cli._subprocess_compat import windows_hide_flags
            result = subprocess.run(
                exec_cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=30, env=build_subprocess_env(),
                creationflags=windows_hide_flags(),  # no console flash on Windows (#56747)
            )
            # See #56747.
            output = result.stdout.strip() or result.stderr.strip()
            if output:
                from agent.redact import redact_sensitive_text
                self._console_print(_rich_text_from_ansi(redact_sensitive_text(output)))
            else:
                self._console_print("[dim]Command returned no output[/]")
        except subprocess.TimeoutExpired:
            self._console_print("[bold red]Quick command timed out (30s)[/]")
        except Exception as e:
            self._console_print(f"[bold red]Quick command error: {e}[/]")
        return True

    def _run_plugin_slash_command(self, base_cmd: str, user_args: str) -> None:
        from hermes_cli.plugins import get_plugin_command_handler, resolve_plugin_command_result

        plugin_handler = get_plugin_command_handler(base_cmd.lstrip("/"))
        if not plugin_handler:
            return
        try:
            result = resolve_plugin_command_result(plugin_handler(user_args))
            if result:
                _cprint(str(result))
        except Exception as e:
            _cprint(f"\033[1;31mPlugin command error: {e}{_RST}")

    def _queue_skill_message(self, msg) -> None:
        if hasattr(self, '_pending_input'):
            self._pending_input.put(msg)

    def _run_skill_bundle_command(self, base_cmd: str, bundle_info: dict, user_instruction: str) -> None:
        """``/<bundle>`` loads several skills at once (bundles win over same-named skills)."""
        bundle_result = build_bundle_invocation_message(base_cmd, user_instruction, task_id=self.session_id)
        if not bundle_result:
            ChatConsole().print(f"[bold red]Failed to load bundle for {base_cmd}[/]")
            return
        msg, loaded_names, missing = bundle_result
        self._queue_loaded_skills(msg, f"Loading bundle: {bundle_info['name']} ({len(loaded_names)} skills)", missing)

    def _queue_loaded_skills(self, msg, label: str, missing) -> None:
        print(f"\n⚡ {label}")
        if missing:
            ChatConsole().print(f"[yellow]Skipped missing skills: {', '.join(missing)}[/]")
        self._queue_skill_message(msg)

    def _run_skill_slash_command(self, base_cmd: str, skill_info: dict, rest: str) -> None:
        """``/<skill> ...``; stacked ``/skill-a /skill-b do XYZ`` loads every leading skill (up to 5)."""
        from agent.skill_commands import build_stacked_skill_invocation_message, split_stacked_skill_commands

        extra_keys, user_instruction = split_stacked_skill_commands(rest)
        if extra_keys:
            stacked_result = build_stacked_skill_invocation_message(
                [base_cmd, *extra_keys], user_instruction, task_id=self.session_id,
            )
            if not stacked_result:
                ChatConsole().print(f"[bold red]Failed to load stacked skills for {base_cmd}[/]")
                return
            msg, loaded_names, missing = stacked_result
            self._queue_loaded_skills(
                msg, f"Loading {len(loaded_names)} stacked skills: {', '.join(loaded_names)}", missing
            )
            return
        msg = build_skill_invocation_message(base_cmd, rest, task_id=self.session_id)
        if msg:
            self._queue_loaded_skills(msg, f"Loading skill: {skill_info['name']}", None)
        else:
            ChatConsole().print(f"[bold red]Failed to load skill for {base_cmd}[/]")

    def _expand_slash_prefix(self, cmd_original: str, cmd_lower: str, skill_commands, skill_bundles) -> bool:
        """Unique-prefix expansion against built-in COMMANDS + skill commands/bundles (agrees with tab-completion)."""
        from hermes_cli.commands import COMMANDS
        typed_base = cmd_lower.split()[0]
        all_known = set(COMMANDS) | set(skill_commands) | set(skill_bundles)
        matches = [c for c in all_known if c.startswith(typed_base)]
        if len(matches) > 1:
            if typed_base in matches:
                matches = [typed_base]
            else:
                # Unique shortest match wins: /qui -> /quit (5) over /quint-pipeline (15)
                min_len = min(len(c) for c in matches)
                shortest = [c for c in matches if len(c) == min_len]
                if len(shortest) == 1:
                    matches = shortest
        if len(matches) == 1 and matches[0] != typed_base:
            # Expand to the full name, preserving arguments.
            return self.process_command(matches[0] + cmd_original.strip()[len(typed_base):])
        if len(matches) > 1:
            _cprint(f"{_ACCENT}Ambiguous command: {cmd_lower}{_RST}")
            _cprint(f"{_DIM}Did you mean: {', '.join(sorted(matches))}?{_RST}")
        else:
            # Exact token with no handler (never re-dispatch the same token: recursion), or no match.
            from hermes_cli.cli_unknown_command import unknown_command_lines
            lead, pointer = unknown_command_lines(cmd_lower, all_known)
            _cprint(f"\033[1;31m{lead}{_RST}")
            _cprint(f"{_DIM}{_ACCENT}{pointer}{_RST}")
        return True

    def _drain_interrupt_queue_to_pending_input(self) -> None:
        """Move stray ``_interrupt_queue`` messages into ``_pending_input`` after every turn.

        Busy-time input lands in ``_interrupt_queue`` and is only drained by the explicit
        interrupt path; a turn that finishes naturally would otherwise strand it and the
        CLI appears to hang. Never raises.

        Called once at the end of every turn from ``process_loop``'s ``finally`` block. Catches and swallows
        ``Exception`` because the drain must never break the main loop. (#20271)
        """
        try:
            while not self._interrupt_queue.empty():
                stray = self._interrupt_queue.get_nowait()
                if stray:
                    self._pending_input.put(stray)
        except Exception:
            pass

        # If the turn was user-interrupted (Ctrl+C), auto-pause the goal
        # and bail. The judge call would almost always return "continue"
        # on the partial output and immediately re-queue another turn,
        # which is exactly what the user cancelled. Pausing (rather than
        # silently skipping) is the observable, recoverable behavior.
        if getattr(self, "_last_turn_interrupted", False):
            try:
                mgr.pause(reason="user-interrupted (Ctrl+C)")
            except Exception as exc:
                logging.debug("goal pause-on-interrupt failed: %s", exc)
            _cprint(
                f"  {_DIM}⏸ Goal paused — turn was interrupted. "
                f"Use /goal resume to continue, or /goal clear to stop.{_RST}"
            )
            return

        # Extract the agent's final response for this turn.
        last_response = ""
        try:
            hist = self.conversation_history or []
            for msg in reversed(hist):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Multimodal content — flatten text parts.
                        parts = [
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") in {"text", "output_text"}
                        ]
                        last_response = "\n".join(t for t in parts if t)
                    else:
                        last_response = str(content or "")
                    break
        except Exception:
            last_response = ""

        # Skip judging on empty/whitespace-only responses. These are almost
        # always transient failures (API error, empty stream) where the
        # judge would say "continue" and trip the consecutive-parse-failures
        # backstop unnecessarily. Mirrors the gateway guard.
        if not last_response.strip():
            return

        try:
            from hermes_cli.goals import gather_background_processes as _gather_bg
            _bg_procs = _gather_bg()
        except Exception:
            _bg_procs = None

        decision = mgr.evaluate_after_turn(
            last_response,
            user_initiated=True,
            background_processes=_bg_procs,
        )
        msg = decision.get("message") or ""
        if msg:
            _cprint(f"  {msg}")

        if decision.get("should_continue"):
            prompt = decision.get("continuation_prompt")
            if prompt:
                try:
                    self._pending_input.put(prompt)
                except Exception as exc:
                    logging.debug("goal continuation enqueue failed: %s", exc)



    def _toggle_verbose(self):
        """Cycle tool progress mode: off → new → all → verbose → off.

        Tool-progress display (full args / results / think blocks at the
        ``verbose`` step) is INDEPENDENT of global DEBUG logging.  Cycling
        through here does not change ``self.verbose`` or the agent's
        ``verbose_logging`` / ``quiet_mode`` — those remain under the
        explicit ``-v``/``--verbose`` flag and the ``/verbose-logging``
        toggle.  See PR #6a1aa420e for the history that decoupled them.
        """
        cycle = ["off", "new", "all", "verbose"]
        try:
            idx = cycle.index(self.tool_progress_mode)
        except ValueError:
            idx = 2  # default to "all"
        self.tool_progress_mode = cycle[(idx + 1) % len(cycle)]

        # /verbose is the explicit tool-progress control, so cycling it takes
        # ownership of the mode back from focus view. Leaving _focus_view_enabled
        # set would show a "focus" status-bar badge and hidden-line counts while
        # tool lines were visibly printing. Display-only state change.
        if getattr(self, "_focus_view_enabled", False):
            self._focus_view_enabled = False
            self._focus_saved_tool_progress = None
            self._focus_hidden_lines = 0
            self._focus_last_counted_tool = None
            try:
                from hermes_cli.focus_view import FOCUS_CONFIG_KEY

                save_config_value(FOCUS_CONFIG_KEY, False)
            except Exception:
                pass

        if self.agent:
            self.agent.reasoning_callback = self._current_reasoning_callback()
            # Keep the live agent's tool_progress_mode in sync so the
            # tool_executor rendering path reflects the new mode this turn,
            # without waiting for an agent rebuild.
            self.agent.tool_progress_mode = self.tool_progress_mode

        # Use raw ANSI codes via _cprint so the output is routed through
        # prompt_toolkit's renderer.  self.console.print() with Rich markup
        # writes directly to stdout which patch_stdout's StdoutProxy mangles
        # into garbled sequences like '?[33mTool progress: NEW?[0m' (#2262).
        from hermes_cli.colors import Colors as _Colors
        labels = {
            "off": f"{_Colors.DIM}Tool progress: OFF{_Colors.RESET} — silent mode, just the final response.",
            "new": f"{_Colors.YELLOW}Tool progress: NEW{_Colors.RESET} — show each new tool (skip repeats).",
            "all": f"{_Colors.GREEN}Tool progress: ALL{_Colors.RESET} — show every tool call.",
            "verbose": f"{_Colors.BOLD}{_Colors.GREEN}Tool progress: VERBOSE{_Colors.RESET} — full args, results, and think blocks.",
        }
        _cprint(labels.get(self.tool_progress_mode, ""))

    def _write_terminal_breadcrumb(self) -> None:
        """Record this terminal's live session for bare ``hermes -c``.

        Called at session start and whenever ``self.session_id`` is
        reassigned mid-run (/new, /branch, auto-compression rotation) so a
        later bare ``-c`` in THIS terminal resumes THIS conversation's live
        tip. Best-effort — never raises, no-op without a terminal identity
        or when session.terminal_continue is false.
        """
        try:
            from hermes_cli.terminal_breadcrumbs import write_breadcrumb

            write_breadcrumb(self.session_id)
        except Exception:
            pass

    def _transfer_session_yolo(self, old_session_id: str, new_session_id: str) -> None:
        """Move YOLO bypass state from an old session key to a new one.

        Called whenever ``self.session_id`` is reassigned mid-run — ``/branch``
        forks into a new session, and auto-compression rotates the agent's
        session id into a fresh continuation session. Without this transfer
        the user's ``/yolo ON`` toggle would silently revert on the very next
        turn (the same UX failure mode that motivated this entire fix), since
        ``_session_yolo`` is keyed by session id.

        Mirrors ``tui_gateway/server.py`` (~line 1297-1305) which performs the
        same transfer for the TUI's session-rename path. No-op when YOLO
        wasn't enabled or when the ids match.
        """
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return
        try:
            from tools.approval import (
                disable_session_yolo,
                enable_session_yolo,
                is_session_yolo_enabled,
            )
        except Exception:
            return
        if is_session_yolo_enabled(old_session_id):
            enable_session_yolo(new_session_id)
            disable_session_yolo(old_session_id)
            # Carry the persisted flag onto the continuation row so a later
            # `hermes --resume <new_id>` restores the bypass too. getattr
            # guard: tests call this unbound against a minimal stand-in.
            _persist = getattr(self, "_persist_session_yolo", None)
            if _persist:
                _persist(new_session_id, True)

    def _is_session_yolo_active(self) -> bool:
        """Whether YOLO bypass is currently enabled for this CLI session.

        Reads from ``tools.approval._session_yolo`` (the same set that
        ``enable_session_yolo`` / ``disable_session_yolo`` write to) so the
        status bar reflects the actual bypass state instead of a stale env
        var. Also honors the process-start ``--yolo`` flag, which freezes
        ``HERMES_YOLO_MODE`` into ``_YOLO_MODE_FROZEN`` before tool imports
        happen.
        """
        try:
            from tools.approval import (
                _YOLO_MODE_FROZEN,
                is_session_yolo_enabled,
            )
        except Exception:
            return False
        if _YOLO_MODE_FROZEN:
            return True
        # Use ``getattr`` so test fixtures that build a CLI via ``__new__``
        # (skipping ``__init__``) don't trip an AttributeError here; the
        # status-bar builders swallow exceptions silently but lose every
        # field after the failure.
        session_key = getattr(self, "session_id", None) or "default"
        return is_session_yolo_enabled(session_key)

    def _toggle_yolo(self):
        """Toggle YOLO mode — skip all dangerous command approval prompts.

        Per-session toggle that mirrors the gateway and TUI ``/yolo`` handlers
        (see ``gateway/run.py:_handle_yolo_command`` and
        ``tui_gateway/server.py`` key=="yolo"). We deliberately do NOT mutate
        ``HERMES_YOLO_MODE`` here — that env var is read once at module import
        time into ``tools.approval._YOLO_MODE_FROZEN`` to keep prompt-injected
        skills from flipping the bypass mid-session, so setting it after CLI
        startup is a silent no-op. Routing through ``enable_session_yolo`` /
        ``disable_session_yolo`` gives the same auditable, per-session bypass
        the other surfaces have. ``run_conversation`` binds
        ``self.session_id`` as the active approval session key via
        ``set_current_session_key`` so the bypass takes effect on the very
        next dangerous command in this run.
        """
        from hermes_cli.colors import Colors as _Colors
        from tools.approval import (
            _YOLO_MODE_FROZEN,
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        # Process-level YOLO (--yolo flag / HERMES_YOLO_MODE at startup) is
        # frozen into tools.approval at import time and cannot be disabled by
        # the session toggle. Before this guard, /yolo printed "YOLO mode OFF —
        # dangerous commands will require approval" while every command kept
        # auto-approving (the frozen flag short-circuits the approval gate
        # ahead of the session check) — a false safety claim. Say the truth
        # instead of toggling a bypass that has no effect.
        if _YOLO_MODE_FROZEN:
            _cprint(
                f"  ⚡ YOLO is {_Colors.BOLD}{_Colors.RED}locked ON{_Colors.RESET}"
                " for this process (started with --yolo / HERMES_YOLO_MODE)."
                " /yolo cannot disable it — restart without the flag to"
                " re-enable approvals."
            )
            return

        session_key = self.session_id or "default"
        # ``getattr`` guard: tests exercise this method unbound against a
        # minimal stand-in object (see tests/cli/test_cli_yolo_toggle.py);
        # persistence is best-effort either way.
        _persist = getattr(self, "_persist_session_yolo", None)
        if is_session_yolo_enabled(session_key):
            disable_session_yolo(session_key)
            if _persist:
                _persist(session_key, False)
            _cprint(
                f"  ⚠ YOLO mode {_Colors.BOLD}{_Colors.RED}OFF{_Colors.RESET}"
                " — dangerous commands will require approval."
            )
        else:
            enable_session_yolo(session_key)
            if _persist:
                _persist(session_key, True)
            _cprint(
                f"  ⚡ YOLO mode {_Colors.BOLD}{_Colors.GREEN}ON{_Colors.RESET}"
                " — all commands auto-approved. Use with caution."
            )

    def _persist_session_yolo(self, session_key: str, enabled: bool) -> None:
        """Persist the YOLO flag to the session row so --resume restores it.

        Best-effort: the in-memory toggle is authoritative for this process;
        persistence only affects a future ``hermes --resume``. Skipped when the
        session store is unavailable or the row doesn't exist yet (the row is
        created lazily on the first turn — ``_toggle_yolo`` before any chat
        writes nothing, and the launch-time ``--yolo`` flag is carried into the
        creation-time model_config instead).
        """
        db = getattr(self, "_session_db", None)
        if db is None or not session_key or session_key == "default":
            return
        try:
            db.set_session_yolo(session_key, enabled)
        except Exception:
            pass




    def _on_reasoning(self, reasoning_text: str):
        """Callback for intermediate reasoning display during tool-call loops."""
        if not reasoning_text:
            return
        self._reasoning_preview_buf = getattr(self, "_reasoning_preview_buf", "") + reasoning_text
        self._flush_reasoning_preview(force=False)

    def _manual_compress(self, cmd_original: str = ""):
        """Manually trigger context compression on the current conversation.

        Two modes:

        * ``/compress [<focus>]`` — compress the *whole* history. An
          optional focus topic guides the summariser to preserve
          information related to *focus* while being more aggressive
          about discarding everything else.  Inspired by Claude Code's
          ``/compact <focus>`` feature.
        * ``/compress here [N]`` — boundary-aware compression. Summarize
          everything *except* the most recent ``N`` exchanges (default
          2), which are preserved verbatim. Inspired by Claude Code's
          Rewind "Summarize up to here" action (v2.1.139, May 2026,
          https://code.claude.com/docs/en/whats-new/2026-w20). Lets the
          user pick the compression boundary instead of leaving it to
          the automatic token-budget heuristic.
        """
        if not self.conversation_history or len(self.conversation_history) < 4:
            print("(._.) Not enough conversation to compress (need at least 4 messages).")
            return

        if not self.agent:
            print("(._.) No active agent -- send a message first.")
            return

        # No compression_enabled gate here: the config flag disables
        # *automatic* compaction only. Manual /compress is an explicit user
        # action — the context-overflow error path (conversation_loop.py)
        # directs users here when auto-compaction is off, and the gateway's
        # /compress handler has never gated on the flag.

        from hermes_cli.partial_compress import (
            extract_compress_flags,
            parse_partial_compress_args,
            rejoin_compressed_head_and_tail,
            split_history_for_partial_compress,
            summarize_compress_preview,
        )
        from agent.conversation_compression import (
            finalize_context_engine_compression_notification,
        )

        # Args after the command word (e.g. "/compress here 3" -> "here 3").
        raw_args = ""
        if cmd_original:
            _parts = cmd_original.strip().split(None, 1)
            if len(_parts) > 1:
                raw_args = _parts[1].strip()

        # Strip --preview/--dry-run/--aggressive before positional parsing
        # so the flags coexist with 'here [N]' / focus-topic forms.
        raw_args, preview, aggressive = extract_compress_flags(raw_args)
        partial, keep_last, focus_topic = parse_partial_compress_args(raw_args)
        focus_topic = focus_topic or ""

        if aggressive:
            # LLM-free hard truncation is not supported: it would need its
            # own transcript-persistence path outside the guarded
            # _compress_context rotation machinery. Surface that instead of
            # silently mis-parsing the flag as a focus topic.
            print("(._.) --aggressive is not supported; use '/compress here [N]' "
                  "to keep only recent exchanges, or /undo to drop turns.")
            if not preview:
                return

        if preview:
            from agent.model_metadata import estimate_request_tokens_rough
            _sys_prompt = getattr(self.agent, "_cached_system_prompt", "") or ""
            _tools = getattr(self.agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(
                self.conversation_history,
                system_prompt=_sys_prompt,
                tools=_tools,
            )
            report = summarize_compress_preview(
                self.conversation_history,
                partial,
                keep_last,
                focus_topic or None,
                approx_tokens,
            )
            for line in report["lines"]:
                print(f"🗜️  {line}")
            return

        original_count = len(self.conversation_history)
        with self._busy_command("Compressing context...", blocks_input=False):
            try:
                from agent.model_metadata import estimate_request_tokens_rough
                from agent.manual_compression_feedback import summarize_manual_compression
                original_history = list(self.conversation_history)

                # Boundary-aware split: only the head is summarized; the
                # most recent `keep_last` exchanges ride along verbatim.
                tail: list = []
                head = original_history
                if partial:
                    head, tail = split_history_for_partial_compress(
                        original_history, keep_last
                    )
                    if not tail:
                        # Split degenerated (everything would be kept, or
                        # no head left to compress). Fall back to full
                        # compression so the user still gets an action.
                        partial = False
                        head = original_history

                # Include system prompt + tool schemas in the estimate —
                # a transcript-only number understates real request pressure
                # and can even appear to grow after compression because a
                # dense handoff summary replaces many short turns (#6217).
                _sys_prompt = getattr(self.agent, "_cached_system_prompt", "") or ""
                _tools = getattr(self.agent, "tools", None) or None
                approx_tokens = estimate_request_tokens_rough(
                    original_history,
                    system_prompt=_sys_prompt,
                    tools=_tools,
                )
                if partial:
                    print(f"🗜️  Summarizing up to here: compressing {len(head)} of "
                          f"{original_count} messages (~{approx_tokens:,} tokens), "
                          f"keeping last {keep_last} exchange(s) verbatim...")
                elif focus_topic:
                    print(f"🗜️  Compressing {original_count} messages (~{approx_tokens:,} tokens), "
                          f"focus: \"{focus_topic}\"...")
                else:
                    print(f"🗜️  Compressing {original_count} messages (~{approx_tokens:,} tokens)...")

                # Pass None as system_message so _compress_context rebuilds
                # the system prompt from scratch via _build_system_prompt(None).
                # Passing _cached_system_prompt caused duplication because
                # _build_system_prompt appends system_message to prompt_parts
                # which already contain the agent identity — resulting in the
                # identity block appearing twice (issue #15281).
                compressed, _ = self.agent._compress_context(
                    head,
                    None,
                    approx_tokens=approx_tokens,
                    focus_topic=focus_topic or None,
                    force=True,
                    defer_context_engine_notification=True,
                )

                # If _compress_context returned unchanged because a
                # concurrent compression lock is held, tell the user
                # clearly instead of showing the misleading
                # "No changes from compression" no-op text. The wording
                # distinguishes a confirmed holder from an unconfirmed
                # acquisition failure (describe_compression_lock_skip).
                # Type-pinned check (is True / str): the flag's only real
                # values are None/True/holder-string, and a bare getattr
                # truthiness test is fooled by MagicMock auto-attributes on
                # test-double agents (skill pitfall: MagicMock vs hasattr).
                _lock_skip_signal = getattr(
                    self.agent, "_compression_skipped_due_to_lock", None
                )
                if _lock_skip_signal is True or isinstance(_lock_skip_signal, str):
                    from agent.manual_compression_feedback import (
                        describe_compression_lock_skip,
                    )
                    print(
                        "  "
                        + describe_compression_lock_skip(
                            self.agent._compression_skipped_due_to_lock
                        )
                    )
                    self.agent._compression_skipped_due_to_lock = None
                    # No boundary was committed on a lock-skip; discard the
                    # deferred context-engine notification (exactly-once).
                    finalize_context_engine_compression_notification(
                        self.agent,
                        committed=False,
                    )
                    return

                if partial and tail:
                    compressed = rejoin_compressed_head_and_tail(compressed, tail)
                self.conversation_history = compressed
                # _compress_context ends the old session and creates a new child
                # session on the agent (run_agent.py::_compress_context). Sync the
                # CLI's session_id so /status, /resume, exit summary, and title
                # generation all point at the live continuation session, not the
                # ended parent. Without this, subsequent end_session() calls target
                # the already-closed parent and the child is orphaned.
                if (
                    getattr(self.agent, "session_id", None)
                    and self.agent.session_id != self.session_id
                ):
                    self.session_id = self.agent.session_id
                    getattr(self, "_write_terminal_breadcrumb", lambda: None)()
                    self._pending_title = None
                    # Manual /compress replaces conversation_history with a new
                    # compressed handoff for the child session. Persist it from
                    # offset 0 so resume can recover the continuation after exit.
                    self.agent._flush_messages_to_session_db(self.conversation_history, None)
                finalize_context_engine_compression_notification(
                    self.agent,
                    committed=True,
                )
                new_tokens = estimate_request_tokens_rough(
                    self.conversation_history,
                    system_prompt=_sys_prompt,
                    tools=_tools,
                )
                summary = summarize_manual_compression(
                    original_history,
                    self.conversation_history,
                    approx_tokens,
                    new_tokens,
                    compression_state=getattr(
                        self.agent, "context_compressor", None
                    ),
                )
                if (
                    summary.get("aborted")
                    or summary.get("fallback_used")
                    or summary.get("refused_would_grow")
                ):
                    icon = "⚠️"
                else:
                    icon = "🗜️" if summary["noop"] else "✅"
                print(f"  {icon} {summary['headline']}")
                print(f"     {summary['token_line']}")
                if summary["note"]:
                    print(f"     {summary['note']}")

            except Exception as e:
                finalize_context_engine_compression_notification(
                    self.agent,
                    committed=False,
                )
                print(f"  ❌ Compression failed: {e}")



    def _handle_usage_command(self, cmd_original: str):
        """Dispatch `/usage [reset [--force]]`.

        Bare `/usage` keeps the classic display. `/usage reset` redeems one
        banked Codex rate-limit reset credit (guarded: refuses when limits
        aren't exhausted unless --force).
        """
        parts = cmd_original.split()
        args = [p.lower() for p in parts[1:]]
        if args and args[0] == "reset":
            self._usage_reset(force="--force" in args[1:])
            return
        if args:
            print(f"  Unknown /usage subcommand: {' '.join(parts[1:])}. Try /usage or /usage reset [--force].")
            return
        self._show_usage()

    def _usage_reset(self, force: bool = False):
        """`/usage reset [--force]` — redeem one banked Codex reset credit."""
        provider = (
            (getattr(self.agent, "provider", None) if self.agent else None)
            or getattr(self, "provider", None)
        )
        normalized = str(provider or "").strip().lower()
        if normalized != "openai-codex":
            print("  Banked usage resets are only available on the openai-codex provider.")
            print("  Switch with `/model` or `hermes auth` first.")
            return
        base_url = (getattr(self.agent, "base_url", None) if self.agent else None) or getattr(self, "base_url", None)
        api_key = (getattr(self.agent, "api_key", None) if self.agent else None) or getattr(self, "api_key", None)

        from agent.account_usage import redeem_codex_reset_credit

        print("  ⏳ Checking banked reset credits...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            try:
                result = _pool.submit(
                    redeem_codex_reset_credit,
                    base_url=base_url,
                    api_key=api_key,
                    force=force,
                ).result(timeout=45.0)
            except concurrent.futures.TimeoutError:
                print("  ❌ Timed out talking to the Codex backend — try again shortly.")
                return
        print(f"  {result.message}")

    def _show_context_breakdown(self, cmd_original: str = ""):
        """`/context [all]` — visual context-window usage breakdown.

        Renders a 5×20 glyph block grid (each cell ≈ 1% of the model context
        window) plus an estimated per-category table: system prompt, tool
        definitions, rules, skills index, MCP, subagents, memory, and the
        conversation itself — versus free space. `/context all` appends the
        expanded per-skill and per-toolset cost listings.

        Read-only: same chars/4 estimation engine as the desktop context
        popover (agent.context_breakdown) — no provider calls, no prompt-cache
        impact.
        """
        if not self.agent:
            print("  (._.) No active agent -- send a message first.")
            return

        args = cmd_original.split(maxsplit=1)[1].strip().lower() if " " in cmd_original else ""
        expanded = args in {"all", "full", "details"}

        from agent.context_breakdown import (
            compute_context_details,
            compute_session_context_breakdown,
            render_context_breakdown_lines,
        )

        try:
            payload = compute_session_context_breakdown(
                self.agent, self.conversation_history
            )
        except Exception as e:
            print(f"  (._.) Could not compute context breakdown: {e}")
            return

        details = None
        if expanded:
            try:
                details = compute_context_details(self.agent)
            except Exception:
                details = {"skills": [], "toolsets": []}

        model = payload.get("model") or self.model
        print()
        print(f"  🧠 Context Usage — {model}")
        print()
        for line in render_context_breakdown_lines(payload, details=details, grid=True):
            print(f"  {line}")
        print()

    def _show_usage(self):
        """Rate limits + session token usage (when a live agent exists) + Nous credits.

        The Nous credits block is agent-independent (a portal fetch), so it runs even
        with no live agent — important for the TUI, where /usage runs in a slash-worker
        subprocess that resumes the session WITHOUT building an agent (self.agent is None),
        which would otherwise early-return before any credits showed.
        """
        if not self.agent:
            if self._print_nous_credits_block():
                self._print_usage_cta()
            else:
                print("(._.) No active agent -- send a message first.")
            return

        agent = self.agent
        calls = agent.session_api_calls

        if calls == 0:
            if self._print_nous_credits_block():
                self._print_usage_cta()
            else:
                print("(._.) No API calls made yet in this session.")
            return

        # ── Rate limits (shown first when available) ────────────────
        rl_state = agent.get_rate_limit_state()
        if rl_state and rl_state.has_data:
            from agent.rate_limit_tracker import format_rate_limit_display
            print()
            print(format_rate_limit_display(rl_state))
            print()

        # ── Session token usage ─────────────────────────────────────
        input_tokens = getattr(agent, "session_input_tokens", 0) or 0
        output_tokens = getattr(agent, "session_output_tokens", 0) or 0
        reasoning_tokens = getattr(agent, "session_reasoning_tokens", 0) or 0
        prompt = agent.session_prompt_tokens
        completion = agent.session_completion_tokens
        total = agent.session_total_tokens

        compressor = agent.context_compressor
        last_prompt = compressor.last_prompt_tokens if compressor.last_prompt_tokens > 0 else 0
        ctx_len = compressor.context_length
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        compressions = compressor.compression_count

        msg_count = len(self.conversation_history)
        elapsed = format_duration_compact((datetime.now() - self.session_start).total_seconds())

        print("  📊 Session Token Usage")
        print(f"  {'─' * 40}")
        print(f"  Model:                     {agent.model}")
        print(f"  Input tokens:              {input_tokens:>10,}")
        print(f"  Output tokens:             {output_tokens:>10,}")
        if reasoning_tokens:
            print(f"  ↳ Reasoning (subset):      {reasoning_tokens:>10,}")
        print(f"  Prompt tokens (total):     {prompt:>10,}")
        print(f"  Completion tokens:         {completion:>10,}")
        print(f"  Total tokens:              {total:>10,}")
        print(f"  API calls:                 {calls:>10,}")
        print(f"  Session duration:          {elapsed:>10}")
        print(f"  {'─' * 40}")
        print(f"  Current context:  {last_prompt:,} / {ctx_len:,} ({pct:.0f}%)")
        print(f"  Messages:         {msg_count}")
        print(f"  Compressions:     {compressions}")

        # Account limits -- fetched off-thread with a hard timeout so slow
        # provider APIs don't hang the prompt.
        provider = getattr(agent, "provider", None) or getattr(self, "provider", None)
        base_url = getattr(agent, "base_url", None) or getattr(self, "base_url", None)
        api_key = getattr(agent, "api_key", None) or getattr(self, "api_key", None)
        # Lazy import — pulls the OpenAI SDK chain, only needed here.
        from agent.account_usage import fetch_account_usage, render_account_usage_lines
        account_snapshot = None
        if provider:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                try:
                    account_snapshot = _pool.submit(
                        fetch_account_usage, provider,
                        base_url=base_url, api_key=api_key,
                    ).result(timeout=10.0)
                except (concurrent.futures.TimeoutError, Exception):
                    account_snapshot = None
        account_lines = [f"  {line}" for line in render_account_usage_lines(account_snapshot)]
        if account_lines:
            print()
            for line in account_lines:
                print(line)

        # Nous credits magnitudes + monthly-grant gauge (agent-independent — also
        # runs at the no-agent / no-calls early-returns above). See the helper.
        if self._print_nous_credits_block():
            self._print_usage_cta()

        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            for noisy in ('openai', 'openai._base_client', 'httpx', 'httpcore', 'asyncio', 'hpack', 'grpc', 'modal'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.INFO)
            # NOTE: We deliberately do NOT raise per-logger levels for
            # tools/run_agent/etc. in quiet mode. Setting logger.setLevel
            # above the file handler level filters records before they
            # reach handlers, so agent.log / errors.log lose visibility
            # into stream-retry events, credential rotations, etc.
            # Console quietness is enforced by hermes_logging not
            # installing a console StreamHandler in non-verbose mode.

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
            elif parts[i].isdigit():
                days = int(parts[i])
                i += 1
            else:
                i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            db = SessionDB()
            try:
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                print(engine.format_terminal(report))
            finally:
                db.close()
        except Exception as e:
            print(f"  Error generating insights: {e}")

    def _check_config_mcp_changes(self) -> None:
        """Detect mcp_servers changes in config.yaml and react.

        Called from process_loop every CONFIG_WATCH_INTERVAL seconds.
        Compares config.yaml mtime + mcp_servers section against the last
        known state.  When a change is detected:

        * By default (``mcp.auto_reload_on_config_change: true``) it
          auto-triggers ``_reload_mcp()`` and informs the user — legacy
          behaviour from #1474.
        * When opted out (``mcp.auto_reload_on_config_change: false``) it
          does NOT reload.  Instead it notifies the user that the config
          changed and that they can apply it with ``/reload-mcp`` — while
          warning that ``/reload-mcp`` rebuilds the tool surface and
          **invalidates the provider prompt cache** (the next message
          re-sends the full input prefix, expensive on long-context /
          high-reasoning models).  This stops silent cache-breaking reloads
          when config.yaml is rewritten frequently by external tooling or
          other Hermes instances.
        """

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
        # Expand ${VAR} templates so the comparison is consistent with the
        # init snapshot (self._config_mcp_servers), which was populated from
        # the deep-merged + expanded config.  Without this, any
        # save_config_value() that rewrites config.yaml (even for unrelated
        # keys) triggers a false-positive MCP reload because the raw yaml
        # still has "${POWERMEM_API_KEY}" while the snapshot has the
        # expanded value.
        from hermes_cli.config import _expand_env_vars
        new_mcp = _expand_env_vars(new_mcp)
        if new_mcp == self._config_mcp_servers:
            return  # mcp_servers unchanged (some other section was edited)

        # Detected a change in the mcp_servers section.  By default we
        # auto-reload (legacy behaviour), but if the user has opted out we
        # notify instead of reloading — because every reload rebuilds the
        # agent tool surface and INVALIDATES the provider prompt cache (the
        # next message re-sends the full input prefix, which is expensive on
        # long-context / high-reasoning models).
        #
        # The toggle is the top-level ``mcp.auto_reload_on_config_change``
        # key (see DEFAULT_CONFIG).  Read it from the config we just parsed
        # so the user can flip it in the same edit that changes mcp_servers;
        # missing key means default-on.
        _mcp_cfg = new_cfg.get("mcp")
        _auto = (
            _mcp_cfg.get("auto_reload_on_config_change", True)
            if isinstance(_mcp_cfg, dict)
            else True
        )

        self._config_mcp_servers = new_mcp

        if not _auto:
            # Notify the user that the config changed but do NOT auto-reload.
            # They can apply the new settings on their own terms with
            # /reload-mcp — which we explicitly warn may invalidate the cache.
            print()
            print("🔄 MCP server config changed — reload skipped (auto-reload disabled).")
            print("   New settings are NOT applied yet. To apply them now, run:")
            print("     /reload-mcp")
            print("   ⚠️  Note: /reload-mcp rebuilds the tool set and invalidates the")
            print("   provider prompt cache (next message re-sends full input tokens).")
            return

        # Notify user and reload.  Run in a separate thread with a hard
        # timeout so a hung MCP server cannot block the process_loop
        # indefinitely (which would freeze the entire TUI).
        print()
        print("🔄 MCP server config changed — reloading connections...")
        _reload_thread = threading.Thread(
            target=self._reload_mcp, daemon=True
        )
        _reload_thread.start()
        # Do NOT join here — process_loop calls this from its idle branch, so a
        # blocking join would freeze input consumption for up to 30s (and a hung
        # MCP server could block far longer). The reload runs purely in the
        # background daemon thread, which reports its own progress/completion
        # status via print() inside _reload_mcp().

    # Inline-skip tokens that bypass the destructive-slash confirmation modal.
    # A general escape hatch for non-interactive use (scripting/automation) and
    # for the degraded path where the modal can't be marshaled onto the app loop
    # — lets users self-serve without flipping approvals.destructive_slash_confirm
    # in config. (Native Windows now drives the modal normally — see #33961.)
    _DESTRUCTIVE_SKIP_TOKENS = frozenset({"now", "--yes", "-y"})

    def _tui_process_loop(self):
        """REPL worker thread: drain ``_pending_input``, run idle housekeeping, dispatch each input."""
        while not self._should_exit:
            try:
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

    def _reload_skills(self) -> None:
        """Reload skills: rescan ~/.hermes/skills/ and queue a note for the
        next user turn.

        Skills don't need to live in the system prompt for the model to use
        them (they're invoked via ``/skill-name``, ``skills_list``, or
        ``skill_view`` at runtime), so this does NOT clear the prompt cache.
        It rescans the slash-command map, prints the diff for the user, and
        — if any skills were added or removed — queues a one-shot note that
        gets prepended to the next user message. This preserves message
        alternation (no phantom user turn injected out of band) and keeps
        prompt caching intact.
        """
        try:
            from agent.skill_commands import reload_skills, get_skill_commands

            if not self._command_running:
                print("🔄 Reloading skills...")

            result = reload_skills()

            # Sync cli.py's module-level _skill_commands so all consumers
            # (help display, command dispatch, Tab-completion lambda) see the
            # updated dict without needing to restart the session.
            global _skill_commands
            _skill_commands = get_skill_commands()
            added = result.get("added", [])      # [{"name", "description"}, ...]
            removed = result.get("removed", [])  # [{"name", "description"}, ...]
            total = result.get("total", 0)

            if not added and not removed:
                print("  No new skills detected.")
                print(f"  📚 {total} skill(s) available")
                return

            def _fmt_line(item: dict) -> str:
                nm = item.get("name", "")
                desc = item.get("description", "")
                return f"    - {nm}: {desc}" if desc else f"    - {nm}"

            if added:
                print("  ➕ Added Skills:")
                for item in added:
                    print(f"  {_fmt_line(item)}")
            if removed:
                print("  ➖ Removed Skills:")
                for item in removed:
                    print(f"  {_fmt_line(item)}")
            print(f"  📚 {total} skill(s) available")

            # Queue a one-shot note for the NEXT user turn. The CLI's agent
            # loop prepends ``_pending_skills_reload_note`` (if set) to the
            # API-call-local message at ~L8770, then clears it — same
            # pattern as ``_pending_model_switch_note``. Nothing is written
            # to conversation_history here, so message alternation stays
            # intact and no out-of-band user turn is persisted.
            #
            # Format matches how the system prompt renders pre-existing
            # skills (``    - name: description``) so the model reads the
            # diff in the same shape as its original skill catalog.
            sections = ["[USER INITIATED SKILLS RELOAD:"]
            if added:
                sections.append("")
                sections.append("Added Skills:")
                for item in added:
                    sections.append(_fmt_line(item))
            if removed:
                sections.append("")
                sections.append("Removed Skills:")
                for item in removed:
                    sections.append(_fmt_line(item))
            sections.append("")
            sections.append("Use skills_list to see the updated catalog.]")
            self._pending_skills_reload_note = "\n".join(sections)

        except Exception as e:
            print(f"  ❌ Skills reload failed: {e}")

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
        spinner and next response).

        On tool.started, records a monotonic timestamp so get_spinner_text()
        can show a live elapsed timer (the TUI poll loop already invalidates
        every ~0.15s, so the counter updates automatically).

        When tool_progress_mode is "all" or "new", also prints a persistent
        stacked line to scrollback on tool.completed so users can see the
        full history of tool calls (not just the current one in the spinner).
        """
        # MoA reference-model outputs: render each reference's answer as a
        # labelled thinking-style block BEFORE the aggregator acts, so the user
        # sees the mixture-of-agents process instead of a silent pause. These
        # are display-only events emitted by the MoA facade (agent_init relay);
        # they never enter message history.
        if event_type == "moa.reference":
            label = function_name or "reference"
            text = preview or ""
            idx = kwargs.get("moa_index")
            count = kwargs.get("moa_count")
            header = f"Reference {idx}/{count} — {label}" if idx and count else f"Reference — {label}"
            try:
                self._flush_reasoning_preview(force=True)
            except Exception:
                pass
            _cprint(f"  {_DIM}┊ ◇ {header}{_RST}")
            try:
                self._emit_reasoning_preview(text)
            except Exception:
                # Fallback: print the raw text dimmed if the preview helper fails.
                if text.strip():
                    _cprint(f"  {_DIM}{text.strip()}{_RST}")
            self._invalidate()
            return
        if event_type == "moa.aggregating":
            agg = function_name or ""
            self._spinner_text = f"◆ aggregating ({agg})" if agg else "◆ aggregating"
            self._invalidate()
            return

        # Feed the pet: tools mean "running" (not reasoning); a failed tool
        # latches the turn so it ends on a sulk.
        if event_type == "tool.started":
            self._pet_reasoning = False
        elif event_type == "tool.completed" and kwargs.get("is_error"):
            self._pet_turn_error = True
        elif event_type and event_type.startswith("reasoning"):
            self._pet_reasoning = True

        if event_type == "tool.completed":
            self._tool_start_time = 0.0
            # Per-turn accounting: this feed already sees every tool call with
            # its result, so the summary line needs no agent-loop state.
            self._turn_summary_record(
                function_name, kwargs.get("result"), kwargs.get("is_error", False)
            )
            # Focus view: count the scrollback line we are NOT printing, so the
            # post-turn recovery line can report how much was hidden. Counted
            # against the pre-focus tool-progress mode, so a user who already
            # had /verbose off is never told focus hid something it didn't.
            if getattr(self, "_focus_view_enabled", False):
                try:
                    self._note_focus_hidden_line(function_name or "")
                except Exception:
                    pass
            # Print stacked scrollback line for "new" / "all" / "verbose" modes.
            # "verbose" was previously omitted here, so non-streaming model
            # calls (MoA aggregator, copilot-acp) rendered each tool only into
            # the transient spinner line — which overwrites itself, so no
            # scrollable tool history accumulated. Streaming models hid the bug
            # because _on_tool_gen_start commits a "preparing" line per tool;
            # non-streaming calls never emit that, leaving verbose mode with no
            # committed line at all. "verbose" is strictly more than "all", so
            # it must commit at least the same line.
            if function_name and self.tool_progress_mode in {"new", "all", "verbose"}:
                duration = kwargs.get("duration", 0.0)
                # Pop stored args from tool.started for this function
                stored = self._pending_tool_info.get(function_name)
                stored_args = stored.pop(0) if stored else {}
                if stored is not None and not stored:
                    del self._pending_tool_info[function_name]
                # "new" mode: skip consecutive repeats of the same tool
                if self.tool_progress_mode == "new" and function_name == self._last_scrollback_tool:
                    self._invalidate()
                    return
                self._last_scrollback_tool = function_name
                try:
                    from agent.display import get_cute_tool_message
                    line = get_cute_tool_message(function_name, stored_args, duration, result=kwargs.get("result"))
                    _cprint(f"  {line}")
                except Exception:
                    pass
                # First-touch onboarding: on the first tool in this process
                # that takes longer than the threshold while we're in the
                # noisiest progress mode, print a one-time hint about
                # /verbose.  Latched on self so it fires at most once per
                # process; persisted to config.yaml so it never fires again
                # across processes either.
                try:
                    if (
                        not getattr(self, "_long_tool_hint_fired", False)
                        and self.tool_progress_mode == "all"
                        and duration >= 30.0
                    ):
                        from agent.onboarding import (
                            TOOL_PROGRESS_FLAG,
                            is_seen,
                            mark_seen,
                            tool_progress_hint_cli,
                        )
                        if not is_seen(CLI_CONFIG, TOOL_PROGRESS_FLAG):
                            self._long_tool_hint_fired = True
                            _cprint(f"  {_DIM}{tool_progress_hint_cli()}{_RST}")
                            mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
                            CLI_CONFIG.setdefault("onboarding", {}).setdefault("seen", {})[TOOL_PROGRESS_FLAG] = True
                except Exception:
                    pass
            self._invalidate()
            return
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
            self._tool_start_time = time.monotonic()
            # Store args for stacked scrollback line on completion
            self._pending_tool_info.setdefault(function_name, []).append(
                function_args if function_args is not None else {}
            )
            self._invalidate()

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
        # A top-level delegate_task dispatches in the background and re-enters as
        # a fresh turn when done. Say so once — no spinner, nothing to poll — so
        # the idle prompt doesn't read as "nothing happened" (⛓ tracks the work).
        if function_name == "delegate_task":
            try:
                parsed = json.loads(function_result) if isinstance(function_result, str) else (function_result or {})
            except Exception:
                parsed = {}
            if isinstance(parsed, dict) and parsed.get("status") == "dispatched" and parsed.get("mode") == "background":
                n = parsed.get("count") or 1
                noun, tail = ("task", "it finishes") if n == 1 else (f"{n} tasks", "they finish")
                try:
                    _cprint(f"\033[2m\u21a9 Background {noun} running — I'll resume when {tail}. Keep chatting.\033[0m")
                except Exception:
                    pass
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
        from tools.voice_mode import create_audio_recorder, check_voice_requirements

        reqs = check_voice_requirements()
        if not reqs["audio_available"]:
            if _is_termux_environment():
                details = reqs.get("details", "")
                if "Termux:API Android app is not installed" in details:
                    raise RuntimeError(
                        "Termux:API command package detected, but the Android app is missing.\n"
                        "Install/update the Termux:API Android app, then retry /voice on.\n"
                        "Fallback: pkg install python-numpy portaudio && python -m pip install sounddevice"
                    )
                raise RuntimeError(
                    "Voice mode requires either Termux:API microphone access or Python audio libraries.\n"
                    "Option 1: pkg install termux-api and install the Termux:API Android app\n"
                    "Option 2: pkg install python-numpy portaudio && python -m pip install sounddevice"
                )
            raise RuntimeError(
                "Voice mode requires sounddevice and numpy.\n"
                f"Install with: {sys.executable} -m pip install sounddevice numpy"
            )
        if not reqs.get("stt_available", reqs.get("stt_key_set")):
            raise RuntimeError(
                "Voice mode requires an STT provider for transcription.\n"
                "Option 1: uv pip install faster-whisper  "
                "(free, local; `pip install faster-whisper` also works if pip is on PATH)\n"
                "Option 2: Set GROQ_API_KEY (free tier)\n"
                "Option 3: Set VOICE_TOOLS_OPENAI_KEY (paid)"
            )

        # Prevent double-start from concurrent threads (atomic check-and-set)
        with self._voice_lock:
            if self._voice_recording:
                return
            self._voice_recording = True

        # Load silence detection params from config. Shape-safe: a
        # hand-edited ``voice: true`` / ``voice: cmd+b`` leaves
        # ``load_config()['voice']`` as a non-dict; coerce to {} so
        # continuous recording falls back to the documented defaults
        # instead of crashing on ``.get()``.
        voice_cfg: dict = {}
        try:
            from hermes_cli.config import load_config
            _cfg = load_config().get("voice")
            voice_cfg = _cfg if isinstance(_cfg, dict) else {}
        except Exception:
            pass

        # Recorder creation can fail (no input device, PortAudio init error).
        # Reset the flag on failure or _voice_recording stays True forever and
        # every future voice start is silently skipped by the guard above.
        if self._voice_recorder is None:
            try:
                self._voice_recorder = create_audio_recorder()
            except Exception:
                with self._voice_lock:
                    self._voice_recording = False
                raise

        # Apply config-driven silence params (numeric-guarded so YAML
        # scalar corruption doesn't break recording start-up).
        #
        # ``bool`` is explicitly excluded from the numeric check — in
        # Python bool is a subclass of int, so a hand-edited
        # ``silence_threshold: true`` would otherwise be forwarded as
        # ``1`` instead of falling back to the 200 default (Copilot
        # round-12 on #19835).
        _threshold = voice_cfg.get("silence_threshold")
        _duration = voice_cfg.get("silence_duration")
        self._voice_recorder._silence_threshold = (
            _threshold if isinstance(_threshold, (int, float)) and not isinstance(_threshold, bool) else 200
        )
        self._voice_recorder._silence_duration = (
            _duration if isinstance(_duration, (int, float)) and not isinstance(_duration, bool) else 3.0
        )
        # voice.max_recording_seconds — hard cap on a single recording's length.
        # Same numeric guard as the silence params (bool excluded: a hand-edited
        # ``max_recording_seconds: true`` must not become ``1`` — it falls back
        # to the documented 120 default, mirroring the silence-param handling).
        # An explicit numeric value <= 0 disables the cap. Previously this
        # documented key was never read (dead config); wiring it here makes it
        # take effect.
        _max_rec = voice_cfg.get("max_recording_seconds")
        self._voice_recorder._max_recording_seconds = (
            (_max_rec if _max_rec > 0 else 0.0)
            if isinstance(_max_rec, (int, float)) and not isinstance(_max_rec, bool)
            else 120.0
        )

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
        if self._voice_beeps_enabled():
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
        _label = self._voice_record_key_label()
        if getattr(self._voice_recorder, "supports_silence_autostop", True):
            _recording_hint = f"auto-stops on silence | {_label} to stop & exit continuous"
        elif _is_termux_environment():
            _recording_hint = f"Termux:API capture | {_label} to stop"
        else:
            _recording_hint = f"{_label} to stop"
        _cprint(f"\n{_ACCENT}● Recording...{_RST} {_DIM}({_recording_hint}){_RST}")

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

    def _voice_stt_model(self) -> Optional[str]:
        """STT model override from config, or None for the provider default.

        For the local provider, prefer stt.local.model (default ``base``) so the
        CLI passes a real model name into the local STT backend.
        """
        try:
            from hermes_cli.config import load_config
            stt_config = load_config().get("stt", {})
            if not isinstance(stt_config, dict):
                return None
            provider = str(stt_config.get("provider") or "").strip().lower()
            if provider == "local":
                local_config = stt_config.get("local") or {}
                if not isinstance(local_config, dict):
                    local_config = {}
                return local_config.get("model") or "base"
            return stt_config.get("model")
        except Exception:
            return None

    def _voice_stt_provider(self) -> str:
        """Configured STT provider name (lowercased), or empty string."""
        try:
            from hermes_cli.config import load_config
            stt_config = load_config().get("stt", {})
            if not isinstance(stt_config, dict):
                return ""
            return str(stt_config.get("provider") or "").strip().lower()
        except Exception:
            return ""

    def _voice_restart_recording_async(self) -> None:
        """Restart continuous-mode recording off-thread (start() can block)."""
        def _restart_recording():
            try:
                self._voice_start_recording()
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()
            except Exception as e:
                _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
        threading.Thread(target=_restart_recording, daemon=True).start()

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
        transcription_failed = False
        wav_path = None
        try:
            if self._voice_recorder is None:
                return

            wav_path = self._voice_recorder.stop()

            # Audio cue: double beep after stream stopped (no CoreAudio conflict)
            if self._voice_beeps_enabled():
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

            stt_model = self._voice_stt_model()
            if self._voice_stt_provider() == "local":
                _cprint(
                    f"{_DIM}Preparing local STT model '{stt_model}' "
                    f"(first use may download it from Hugging Face)...{_RST}"
                )
            else:
                _cprint(f"{_DIM}Transcribing...{_RST}")

            from tools.voice_mode import transcribe_recording
            result = transcribe_recording(wav_path, model=stt_model)

            if result.get("success") and result.get("transcript", "").strip():
                transcript = result["transcript"].strip()
                from tools.voice_mode import is_voice_stop_phrase
                if is_voice_stop_phrase(transcript):
                    # Bare "stop" (or configured phrase) ends the voice chat
                    # instead of being sent to the agent.
                    _cprint(f"{_DIM}Stop phrase detected — ending voice chat.{_RST}")
                    self._disable_voice_mode()
                    return
                self._attached_images.clear()
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()
                self._pending_input.put(_VoiceInputMessage(transcript))
                submitted = True
            elif result.get("success"):
                _cprint(f"{_DIM}No speech detected.{_RST}")
            else:
                error = result.get("error", "Unknown error")
                _cprint(f"\n{_DIM}Transcription failed: {error}{_RST}")
                transcription_failed = True

        except Exception as e:
            _cprint(f"\n{_DIM}Voice processing error: {e}{_RST}")
            transcription_failed = wav_path is not None
        finally:
            with self._voice_lock:
                self._voice_processing = False
            if hasattr(self, '_app') and self._app:
                self._app.invalidate()
            # Clean up temp file unless transcription failed. On failure, keep
            # the source recording so long dictation is not lost.
            try:
                if wav_path and os.path.isfile(wav_path):
                    if transcription_failed:
                        _cprint(f"{_DIM}Recording preserved at: {wav_path}{_RST}")
                    else:
                        os.unlink(wav_path)
            except Exception:
                pass

            # Track consecutive no-speech cycles to avoid infinite restart loops.
            # While the agent is mid-turn or TTS is speaking, the user is
            # CORRECTLY silent (waiting/listening) — those cycles must not
            # count, or a multi-minute tool run ends the voice chat under
            # the user. The stop phrase and barge-in still work during the
            # hold (they run on their own paths above).
            stop_continuous_restart = False
            _tts_done = getattr(self, "_voice_tts_done", None)
            _activity_hold = bool(
                getattr(self, "_agent_running", False)
                or (_tts_done is not None and not _tts_done.is_set())
            )
            if not submitted:
                if _activity_hold:
                    pass  # held: keep listening without counting the cycle
                else:
                    self._no_speech_count = getattr(self, '_no_speech_count', 0) + 1
                    if self._no_speech_count >= 3:
                        self._voice_continuous = False
                        self._no_speech_count = 0
                        _cprint(f"{_DIM}No speech detected 3 times, continuous mode stopped.{_RST}")
                        stop_continuous_restart = True
            else:
                self._no_speech_count = 0

            # If no transcript was submitted but continuous mode is active,
            # restart recording so the user can keep talking.
            # (When transcript IS submitted, process_loop handles restart
            # after chat() completes.)
            if (
                self._voice_continuous
                and not submitted
                and not self._voice_recording
                and not stop_continuous_restart
            ):
                self._voice_restart_recording_async()

    def _voice_speak_response_async(self, text: str) -> None:
        """Schedule TTS and mark it pending before continuous recording can restart."""
        if not self._voice_tts or not text:
            return
        self._voice_tts_done.clear()
        threading.Thread(
            target=self._voice_speak_response,
            args=(text,),
            daemon=True,
        ).start()
        # Spoken barge-in must work on the whole-file fallback path too. The
        # full-duplex agent-turn listener normally already covers playback
        # (armed at turn start in chat()); this arm is an idempotent safety
        # net for speak calls outside a chat turn — the listener refuses to
        # double-arm via _voice_fd_active.
        if self._voice_continuous:
            threading.Thread(
                target=self._voice_full_duplex_listener,
                daemon=True,
            ).start()

    def _voice_speak_response(self, text: str):
        """Speak the agent's response aloud using TTS (runs in background thread)."""
        if not self._voice_tts:
            return
        self._voice_tts_done.clear()
        try:
            from tools.tts_tool import text_to_speech_tool
            from tools.voice_mode import play_audio_file

            # Strip markdown and non-speech content for cleaner TTS via the
            # shared cleaner (tools/tts_text_normalize): markdown, emoji,
            # ⋗ blocks, verifier footer, units, newline flattening.
            # The TTS tool owns provider request limits and long-form chunking.
            try:
                from tools.tts_text_normalize import prepare_spoken_text
                tts_text = prepare_spoken_text(text, max_chars=None)
            except Exception:
                # Legacy fallback pipeline — keep voice replies best-effort.
                tts_text = re.sub(r'```[\s\S]*?```', ' ', text)   # fenced code blocks
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
            self._voice_last_tts_text = tts_text

            # Use MP3 output for CLI playback (afplay doesn't handle OGG well).
            # The TTS tool may auto-convert MP3->OGG, but the original MP3 remains.
            os.makedirs(os.path.join(tempfile.gettempdir(), "hermes_voice"), exist_ok=True)
            mp3_path = os.path.join(
                tempfile.gettempdir(), "hermes_voice",
                f"tts_{time.strftime('%Y%m%d_%H%M%S')}.mp3",
            )

            raw_result = text_to_speech_tool(text=tts_text, output_path=mp3_path)
            try:
                tts_result = json.loads(raw_result) if isinstance(raw_result, str) else {}
            except Exception:
                tts_result = {}

            # The tool result is authoritative — it may return multiple files
            # for long-form chunked output. Play each in order.
            play_paths = tts_result.get("file_paths") or [
                tts_result.get("file_path") or mp3_path
            ]
            for play_path in play_paths if tts_result.get("success") else []:
                if os.path.isfile(play_path) and os.path.getsize(play_path) > 0:
                    play_audio_file(play_path)
            # Clean up all generated files (play_paths + mp3_path + ogg variants)
            cleanup_paths = set(play_paths + [mp3_path, mp3_path.rsplit(".", 1)[0] + ".ogg"])
            for path in cleanup_paths:
                if os.path.isfile(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        except Exception as e:
            logger.warning("Voice TTS playback failed: %s", e)
            _cprint(f"{_DIM}TTS playback failed: {e}{_RST}")
        finally:
            self._voice_tts_done.set()


    def _voice_full_duplex_listener(self) -> None:
        """Full-duplex agent-turn listener: mic live for the WHOLE turn.

        Armed at utterance-submit (chat() start in continuous voice mode) and
        disarmed when the turn is fully done (agent finished + TTS played).
        Replaces the old per-playback ``_voice_barge_in_monitor``, which only
        listened while TTS audio was playing — during LLM generation the mic
        was dead, so the user could not interject by voice at all (and the
        playback monitor calibrated against its own speaker bleed, making
        the trigger unreachable; see tools.voice_mode.full_duplex_listen).

        Phase behaviour:

        * generation (no TTS audio yet): speech interrupts the in-flight
          agent turn via ``self.agent.interrupt()`` — the same seam the
          typed/Ctrl+C interrupt uses — and the captured utterance is
          submitted as the next message.
        * playback: speech cuts TTS (pipeline stop event + stop_playback)
          and the interruption is captured with pre-roll and submitted.

        The stop phrase ends the voice chat in BOTH phases (a stop during
        generation means "stop everything": the turn is already interrupted
        at trip time, then ``_voice_submit_barge_utterance`` disables voice
        mode).
        """
        fd_active = getattr(self, "_voice_fd_active", None)
        if fd_active is None:
            fd_active = threading.Event()
            self._voice_fd_active = fd_active
        if fd_active.is_set():
            return  # one listener owns the mic for this turn
        fd_active.set()
        try:
            from hermes_cli.config import load_config
            voice_cfg = load_config().get("voice") or {}
            if not (isinstance(voice_cfg, dict) and voice_cfg.get("barge_in", True)):
                return
            from tools.voice_mode import (
                full_duplex_listen,
                is_audio_output_active,
                stop_playback,
            )

            try:
                _mult = float(voice_cfg.get("barge_in_threshold_multiplier", 0) or 0)
            except (TypeError, ValueError):
                _mult = 0.0
            try:
                _grace_ms = int(float(voice_cfg.get("barge_in_grace_seconds", 0.5)) * 1000)
            except (TypeError, ValueError):
                _grace_ms = 500

            tts_done = getattr(self, "_voice_tts_done", None)

            def _should_stop() -> bool:
                if not (getattr(self, "_voice_mode", False) and getattr(self, "_voice_continuous", False)):
                    return True
                if getattr(self, "_agent_running", False):
                    return False
                # Agent finished — keep listening until TTS fully played.
                if tts_done is not None and not tts_done.is_set():
                    return False
                return not is_audio_output_active()

            def _on_trigger(phase: str) -> None:
                # Latch BEFORE cutting anything: suppresses process_loop's
                # auto-restart until the capture is submitted.
                self._voice_barge_capture.set()
                self._voice_barge_phase = phase
                if phase == "playback":
                    logger.debug(
                        "TTS CUT: full-duplex listener tripped during playback"
                    )
                    from tools.tts_streaming import mark_speech_interrupted
                    mark_speech_interrupted()
                    _pipe_stop = getattr(self, "_voice_tts_stop", None)
                    if _pipe_stop is not None:
                        _pipe_stop.set()
                    stop_playback()
                else:
                    # Generation phase: no audio to cut — interrupt the
                    # in-flight agent turn (same seam as typed interrupt).
                    logger.debug(
                        "full-duplex listener tripped during generation — "
                        "interrupting agent turn"
                    )
                    _pipe_stop = getattr(self, "_voice_tts_stop", None)
                    if _pipe_stop is not None:
                        _pipe_stop.set()  # never let the stale reply speak
                    try:
                        if self.agent is not None and getattr(self, "_agent_running", False):
                            _cprint(f"\n{_DIM}🎤 Voice interjection — interrupting…{_RST}")
                            self.agent.interrupt()
                    except Exception as e:
                        logger.debug("voice interjection interrupt failed: %s", e)

            wav_path = full_duplex_listen(
                _should_stop,
                is_playing=is_audio_output_active,
                on_trigger=_on_trigger,
                multiplier=_mult or None,
                grace_ms=max(0, _grace_ms),
            )
            if wav_path and self._voice_barge_capture.is_set():
                self._voice_submit_barge_utterance(wav_path)
            else:
                self._voice_barge_capture.clear()
        except Exception as e:
            self._voice_barge_capture.clear()
            logger.debug("Voice full-duplex listener failed: %s", e)
        finally:
            fd_active.clear()

    def _voice_submit_barge_utterance(self, wav_path: str) -> None:
        """Transcribe a barge-captured interruption and queue it as the next turn."""
        submitted = False
        try:
            from tools.voice_mode import transcribe_recording
            result = transcribe_recording(wav_path, model=self._voice_stt_model())
            transcript = (result.get("transcript") or "").strip() if result.get("success") else ""
            if transcript:
                from tools.voice_mode import is_voice_stop_phrase
                if is_voice_stop_phrase(transcript):
                    _cprint(f"\n{_DIM}Stop phrase detected — ending voice chat.{_RST}")
                    self._disable_voice_mode()
                    return
                # Fail-closed echo guard (#75780): a playback-phase capture
                # has no acoustic echo cancellation, so speaker bleed alone
                # can trip the barge trigger. If the transcript is a close
                # match for what Hermes just spoke, treat it as self-capture
                # instead of queuing it as a user turn.
                if getattr(self, "_voice_barge_phase", None) == "playback":
                    from tools.voice_mode import is_tts_echo
                    if is_tts_echo(transcript, getattr(self, "_voice_last_tts_text", "")):
                        logger.debug(
                            "Dropping playback-phase barge transcript as TTS echo: %r",
                            transcript,
                        )
                        _cprint(f"\n{_DIM}Ignored likely TTS echo (not queued).{_RST}")
                        return
                self._pending_input.put(_VoiceInputMessage(transcript))
                submitted = True
            elif not result.get("success"):
                _cprint(f"\n{_DIM}Transcription failed: {result.get('error', 'Unknown error')}{_RST}")
        except Exception as e:
            _cprint(f"\n{_DIM}Voice processing error: {e}{_RST}")
        finally:
            try:
                if os.path.isfile(wav_path):
                    os.unlink(wav_path)
            except OSError:
                pass
            self._voice_barge_capture.clear()
            self._voice_barge_phase = None
            # No usable transcript: hand the mic back to the normal loop.
            if not submitted and self._voice_mode and self._voice_continuous and not self._voice_recording:
                self._voice_restart_recording_async()

    def _voice_beeps_enabled(self) -> bool:
        """Return whether CLI voice mode should play record start/stop beeps."""
        try:
            from hermes_cli.config import load_config
            from utils import is_truthy_value
            voice_cfg = load_config().get("voice", {})
            if isinstance(voice_cfg, dict):
                # is_truthy_value handles quoted YAML strings like "false"
                # which bool() would misread as True (#49883).
                return is_truthy_value(voice_cfg.get("beep_enabled", True), default=True)
        except Exception:
            pass
        return True

    def _enable_voice_mode(self):
        """Enable voice mode after checking requirements."""
        if self._voice_mode:
            _cprint(f"{_DIM}Voice mode is already enabled.{_RST}")
            return

        from tools.voice_mode import check_voice_requirements, detect_audio_environment

        # Environment detection -- warn and block in incompatible environments
        env_check = detect_audio_environment()
        if not env_check["available"]:
            _cprint(f"\n{_ACCENT}Voice mode unavailable in this environment:{_RST}")
            for warning in env_check["warnings"]:
                _cprint(f"  {_DIM}{warning}{_RST}")
            return

        reqs = check_voice_requirements()
        if not reqs["available"]:
            _cprint(f"\n{_ACCENT}Voice mode requirements not met:{_RST}")
            for line in reqs["details"].split("\n"):
                _cprint(f"  {_DIM}{line}{_RST}")
            if reqs["missing_packages"]:
                if _is_termux_environment():
                    _cprint(f"\n  {_BOLD}Option 1: pkg install termux-api{_RST}")
                    _cprint(f"  {_DIM}Then install/update the Termux:API Android app for microphone capture{_RST}")
                    _cprint(f"  {_BOLD}Option 2: pkg install python-numpy portaudio && python -m pip install sounddevice{_RST}")
                else:
                    _cprint(f"\n  {_BOLD}Install: {sys.executable} -m pip install {' '.join(reqs['missing_packages'])}{_RST}")
            return

        with self._voice_lock:
            self._voice_mode = True

        # Check config for auto_tts (shape-safe — malformed ``voice:`` YAML
        # leaves ``voice_config`` as a non-dict, so guard before .get()).
        try:
            from hermes_cli.config import load_config
            _raw_voice = load_config().get("voice")
            voice_config = _raw_voice if isinstance(_raw_voice, dict) else {}
            if voice_config.get("auto_tts", False):
                with self._voice_lock:
                    self._voice_tts = True
        except Exception:
            pass

        # Voice mode instruction is injected as a user message prefix (not a
        # system prompt change) to avoid invalidating the prompt cache.  See
        # _voice_message_prefix property and its usage in _process_message().

        tts_status = " (TTS enabled)" if self._voice_tts else ""
        # Use the startup-pinned cache so the advertised shortcut always
        # matches the live prompt_toolkit binding — reading live config
        # here would drift after a mid-session config edit (Copilot
        # round-14 on #19835, same class as round-13).
        _ptt_display = self._voice_record_key_label()
        _cprint(f"\n{_ACCENT}Voice mode enabled{tts_status}{_RST}")
        _cprint(f"  {_DIM}{_ptt_display} to start/stop recording{_RST}")
        # Spoken-stop hint sourced from voice.stop_phrases (first entry); the
        # helper returns "" when stop phrases are disabled — show no hint then.
        try:
            from tools.voice_mode import voice_stop_hint
            _stop_hint = voice_stop_hint()
        except Exception:
            _stop_hint = ""
        if _stop_hint:
            _cprint(f"  {_DIM}{_stop_hint}{_RST}")
        _cprint(f"  {_DIM}/voice tts  to toggle speech output{_RST}")
        _cprint(f"  {_DIM}/voice off  to disable voice mode{_RST}")

    def _typed_voice_stop(self, user_input) -> bool:
        """Typed bare stop phrase during an active voice chat ends the chat.

        Saying "stop" ends the voice chat (PR #73106); TYPING the same bare
        stop phrase while voice mode is on must behave identically instead of
        sending "stop" to the agent as a turn. Guarded on voice mode being ON
        — typed "stop" outside voice chat passes through to the agent exactly
        as before. Reuses ``is_voice_stop_phrase`` (same config
        ``voice.stop_phrases``, same exact-match semantics), so longer typed
        messages containing "stop" are never swallowed.
        """
        if not isinstance(user_input, str):
            return False
        with self._voice_lock:
            voice_on = self._voice_mode or self._voice_continuous
        if not voice_on:
            return False
        try:
            from tools.voice_mode import is_voice_stop_phrase
            if not is_voice_stop_phrase(user_input):
                return False
        except Exception:
            return False
        _cprint(f"\n{_DIM}Stop phrase typed — ending voice chat.{_RST}")
        self._disable_voice_mode()
        return True

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

        # Stop any active TTS playback (file player + streaming pipeline)
        try:
            if self._voice_tts_stop is not None:
                logger.info("TTS CUT: _disable_voice_mode setting stop event")
                self._voice_tts_stop.set()
            from tools.voice_mode import stop_playback
            stop_playback()
        except Exception:
            pass
        self._voice_tts_done.set()

        _cprint(f"\n{_DIM}Voice mode disabled.{_RST}")

    # ── Wake word ("Hey Hermes") ─────────────────────────────────────────
    #
    # An always-on hotword listener (tools/wake_word.py) that, on detecting
    # the wake phrase, starts a fresh session and captures one utterance via
    # the existing voice pipeline — the "Hey Siri" pattern, fully on-device.
    #
    # The detector holds the microphone, so it must be paused while a voice
    # turn records (two input streams on one device is unreliable). On wake we
    # pause it and mark the system suspended; a lightweight watchdog resumes it
    # once the turn finishes and the CLI is idle again — covering every exit
    # path (transcript submitted, no speech, or transcription error) without
    # threading resume logic through the voice machinery.

    def _maybe_start_wake_word(self):
        """Start the wake-word listener at CLI startup if this surface is eligible."""
        try:
            from tools.wake_word import wake_surface_enabled
            if not wake_surface_enabled("cli"):
                return
        except Exception:
            return
        self._start_wake_word_listener(announce=True)

    def _start_wake_word_listener(self, announce: bool = False) -> bool:
        """Build + start the hotword detector. Returns True on success."""
        try:
            from tools.wake_word import (
                check_wake_word_requirements,
                load_wake_word_config,
                owns_listener,
                start_listening,
            )
        except Exception as e:
            if announce:
                _cprint(f"{_DIM}Wake word unavailable: {e}{_RST}")
            return False

        if getattr(self, "_wake_word_active", False) and owns_listener(self):
            if announce:
                _cprint(f"{_DIM}Wake word is already listening.{_RST}")
            return True
        self._wake_word_active = False

        cfg = load_wake_word_config()
        reqs = check_wake_word_requirements(cfg)
        if not reqs["available"]:
            if announce:
                _cprint(f"\n{_ACCENT}Wake word requirements not met:{_RST}")
                if reqs.get("hint"):
                    _cprint(f"  {_DIM}{reqs['hint']}{_RST}")
            return False

        if announce and not reqs.get("deps_available", True):
            # Fresh install: the engine constructor lazy-installs its deps
            # (onnxruntime is a large wheel) — tell the user why this is slow.
            _cprint(f"{_DIM}Installing wake word engine (first use — this may take a minute)...{_RST}")

        self._wake_start_new_session = bool(cfg.get("start_new_session", True))
        try:
            start_listening(self._on_wake_word, owner=self, config=cfg)
        except Exception as e:
            if announce:
                _cprint(f"\n{_DIM}Failed to start wake word: {e}{_RST}")
            return False

        self._wake_word_active = True
        self._wake_suspended = False
        global _cli_wake_owner
        _cli_wake_owner = self
        self._start_wake_watchdog()
        if announce:
            _cprint(f"\n{_ACCENT}Wake word listening{_RST} "
                    f"{_DIM}(say \"{reqs['phrase']}\" — /wake off to stop){_RST}")
        return True

    def _stop_wake_word_listener(self, announce: bool = False):
        """Stop and tear down the hotword detector."""
        global _cli_wake_owner
        was_active = getattr(self, "_wake_word_active", False)
        self._wake_word_active = False
        self._wake_suspended = False
        try:
            from tools.wake_word import stop_listening
            stop_listening(owner=self)
        except Exception:
            pass
        if _cli_wake_owner is self:
            _cli_wake_owner = None
        if announce:
            if was_active:
                _cprint(f"{_DIM}Wake word stopped.{_RST}")
            else:
                _cprint(f"{_DIM}Wake word is not running.{_RST}")

    def _on_wake_word(self):
        """Fired after the detector hears the wake phrase."""
        if getattr(self, "_should_exit", False):
            return
        # Ignore wake while a turn is in flight or the mic is already in use.
        if self._agent_running or self._voice_recording or getattr(self, "_voice_processing", False):
            return

        # Release the mic so STT can capture the command utterance.
        try:
            from tools.wake_word import pause_listening
            if not pause_listening(owner=self):
                self._wake_word_active = False
                return
        except Exception as e:
            logger.debug("wake word pause failed: %s", e)
            return
        self._wake_suspended = True

        # Multi-profile routing: the CLI is a single-profile process, so a
        # phrase enrolled by ANOTHER profile can't be routed here — print the
        # switch command and re-arm rather than answering as the wrong profile.
        try:
            from tools.wake_word import get_last_match
            _match = get_last_match()
        except Exception:
            _match = None
        if _match and _match[1]:
            from tools.wake_word import _active_profile_name
            if _match[1] != _active_profile_name():
                _cprint(f"\n{_DIM}Wake phrase for profile '{_match[1]}' — "
                        f"run: hermes -p {_match[1]}{_RST}")
                self._wake_suspended = True  # watchdog resumes the listener
                return

        _cprint(f"\n{_ACCENT}✦ Wake word detected — listening...{_RST}")
        if getattr(self, "_app", None):
            try:
                self._app.invalidate()
            except Exception:
                pass

        if getattr(self, "_wake_start_new_session", True):
            try:
                self.new_session(silent=True)
            except Exception as e:
                logger.debug("wake word new_session failed: %s", e)

        # Single-utterance capture (not continuous) via the voice pipeline;
        # VAD auto-stop transcribes and queues the transcript for process_loop.
        with self._voice_lock:
            self._voice_mode = True
        self._voice_continuous = False
        try:
            self._voice_start_recording()
        except Exception as e:
            _cprint(f"{_DIM}Wake capture failed: {e}{_RST}")
            # Leave _wake_suspended set; the watchdog resumes once idle.

    def _start_wake_watchdog(self):
        """Resume the paused detector when the CLI returns to a stable idle."""
        if getattr(self, "_wake_watchdog_started", False):
            return
        self._wake_watchdog_started = True

        def _loop():
            idle_polls = 0
            try:
                while getattr(self, "_wake_word_active", False) and not getattr(self, "_should_exit", False):
                    time.sleep(0.25)
                    if not getattr(self, "_wake_suspended", False):
                        idle_polls = 0
                        continue
                    busy = (
                        self._agent_running
                        or self._voice_recording
                        or getattr(self, "_voice_processing", False)
                        or not self._pending_input.empty()
                    )
                    if busy:
                        idle_polls = 0
                        continue
                    # Require a few consecutive idle polls (~0.75s) so we don't
                    # resume in the gap between VAD stop and the agent starting.
                    idle_polls += 1
                    if idle_polls >= 3:
                        idle_polls = 0
                        try:
                            from tools.wake_word import resume_listening
                            if resume_listening(owner=self):
                                self._wake_suspended = False
                            else:
                                self._wake_word_active = False
                        except Exception as e:
                            logger.debug("wake word resume failed: %s", e)
            finally:
                self._wake_watchdog_started = False

        threading.Thread(target=_loop, daemon=True, name="wake-watchdog").start()

    def _show_wake_word_status(self):
        """Show current wake-word listener status."""
        from tools.wake_word import (
            audio_is_silent,
            check_wake_word_requirements,
            is_listening,
            load_wake_word_config,
            owns_listener,
        )

        cfg = load_wake_word_config()
        reqs = check_wake_word_requirements(cfg)
        owned = owns_listener(self)
        state = "LISTENING" if owned and is_listening() else "PAUSED" if owned else "OFF"

        _cprint(f"\n{_BOLD}Wake Word Status{_RST}")
        _cprint(f"  State:       {state}")
        _cprint(f"  Phrase:      \"{reqs['phrase']}\"")
        _cprint(f"  Provider:    {reqs['provider']}")
        _cprint(f"  Surface:     {cfg.get('surface', 'auto')}")
        _cprint(f"  New session: {'yes' if cfg.get('start_new_session', True) else 'no'}")
        if state == "LISTENING" and audio_is_silent():
            _cprint(f"  {_ACCENT}⚠ Microphone delivers only silence — the listener can't hear anything.{_RST}")
            _cprint(f"  {_DIM}On macOS: System Settings > Privacy & Security > Microphone — allow your"
                    f" terminal/Hermes, then /wake off + /wake on.{_RST}")
        if not reqs["available"] and reqs.get("hint"):
            _cprint(f"  {_DIM}{reqs['hint']}{_RST}")
        if not owned:
            _cprint(f"  {_DIM}Enable with /wake on{_RST}")

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

        _cprint(f"{_ACCENT}Voice TTS {status}.{_RST}")

    def _show_voice_status(self):
        """Show current voice mode status."""
        from tools.voice_mode import check_voice_requirements

        reqs = check_voice_requirements()

        _cprint(f"\n{_BOLD}Voice Mode Status{_RST}")
        _cprint(f"  Mode:      {'ON' if self._voice_mode else 'OFF'}")
        _cprint(f"  TTS:       {'ON' if self._voice_tts else 'OFF'}")
        _cprint(f"  Recording: {'YES' if self._voice_recording else 'no'}")
        # Display the startup-pinned label so /voice status always
        # matches the live prompt_toolkit binding (Copilot round-14 on
        # #19835, same class as round-13). Reading live config here
        # would drift after a mid-session config edit.
        _cprint(f"  Record key: {self._voice_record_key_label()}")
        _cprint(f"\n  {_BOLD}Requirements:{_RST}")
        for line in reqs["details"].split("\n"):
            _cprint(f"    {line}")

    def _persist_prompt_summary(self, icon: str, label: str, detail: str, outcome: str) -> None:
        """Print a one-line scrollback summary of a resolved modal prompt.

        Modal panels (approval / clarify) live in the prompt_toolkit layout and
        vanish on the next repaint, so the question and the decision leave no
        trace in the terminal scrollback. When display.persist_prompts is on
        (default), emit a dim single line after the prompt resolves so the
        decision survives in chat history.
        """
        if not CLI_CONFIG.get("display", {}).get("persist_prompts", True):
            return
        detail = " ".join(detail.split())
        if len(detail) > 120:
            detail = detail[:119] + "…"
        outcome = " ".join(outcome.split())
        if len(outcome) > 120:
            outcome = outcome[:119] + "…"
        _cprint(f"\n{_DIM}{icon} {label}: {detail} → {outcome}{_RST}")

    def _clarify_callback(self, question, choices, multi_select=False, questions=None):
        """
        Platform callback for the clarify tool. Called from the agent thread.

        Sets up the interactive selection UI (or freetext prompt for open-ended
        questions), then blocks until the user responds via the prompt_toolkit
        key bindings.  If no response arrives within the configured timeout the
        question is dismissed and the agent is told to decide on its own.

        When ``multi_select`` is True, shows checkboxes and the user can
        select multiple options with Space, confirming with Enter.

        When ``questions`` is a non-empty list (batch clarify, issue #18450),
        the panel switches to the A-compact multi-question layout and the
        return value is a dict ``{"answers": {qid: raw_answer}}`` (plus
        ``"timed_out": True`` when the deadline expired with only partial
        answers). The single-question path below is unchanged.
        """
        import time as _time

        from tools.clarify_gateway import resolve_clarify_timeout

        if questions:
            return self._clarify_callback_batch(questions)

        # Canonical clarify timeout, shared with the gateway/TUI path. `<= 0`
        # means unlimited (never auto-skip mid-think) → a null deadline.
        timeout = resolve_clarify_timeout(CLI_CONFIG)
        response_queue = queue.Queue()
        is_open_ended = not choices
        # multi-select support: only active when multi_select is True and choices exist
        effective_multi = multi_select and not is_open_ended

        self._clarify_state = {
            "question": question,
            "choices": choices if not is_open_ended else [],
            "selected": 0,
            # multi-select support
            "multi_select": effective_multi,
            "selected_indices": set() if effective_multi else None,
            "response_queue": response_queue,
        }
        self._clarify_deadline = None if timeout <= 0 else _time.monotonic() + timeout
        # Open-ended questions skip straight to freetext input
        self._clarify_freetext = is_open_ended
        self._clarify_multi_base = None

        # Trigger an immediate prompt_toolkit repaint from this (non-main)
        # thread. Modal prompts must paint at once and must not be gated by the
        # _invalidate throttle / resize guard — see _paint_now / _invalidate (#41098).
        self._paint_now()

        # Poll for the user's response. The countdown in the hint line updates
        # on each repaint; refresh it once a second so the timer stays visible
        # while we wait. Selection changes (↑/↓) trigger instant repaints via
        # the key bindings.
        _last_countdown_refresh = _time.monotonic()
        while True:
            try:
                result = response_queue.get(timeout=1)
                self._clarify_deadline = None
                self._persist_prompt_summary("?", "Clarify", question, str(result))
                return result
            except queue.Empty:
                # None deadline = unlimited: never auto-skip, just keep polling.
                if self._clarify_deadline is not None:
                    remaining = self._clarify_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                now = _time.monotonic()
                if now - _last_countdown_refresh >= 1.0:
                    _last_countdown_refresh = now
                    self._paint_now()

        # Timed out — tear down the UI and let the agent decide
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = None
        self._clarify_multi_base = None
        self._paint_now()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")
        return (
            "The user did not provide a response within the time limit. "
            "Use your best judgement to make the choice and proceed."
        )

    # --- Batch clarify (multi-question, issue #18450) -----------------------

    def _clarify_batch_set_active(self, state, index) -> None:
        """Point the batch clarify panel at question ``index``.

        Mirrors the active question's data into the flat keys the existing
        single-question keybindings and renderer read (``question``,
        ``choices``, ``selected``, ``multi_select``, ``selected_indices``),
        so ↑/↓/Space/number keys operate on the active question unchanged.
        Open-ended questions drop straight into freetext, matching the
        single-question path. Re-visiting an answered question restores the
        cursor to the earlier selection (choice answers highlight their row,
        an "Other" answer highlights the Other row) so the user can see and
        edit what they picked.
        """
        questions_list = state["questions"]
        index = max(0, min(index, len(questions_list) - 1))
        entry = questions_list[index]
        state["active"] = index
        state["question"] = entry["question"]
        state["choices"] = entry["choices"] or []
        state["selected"] = 0
        state["multi_select"] = bool(entry["multi_select"])
        state["selected_indices"] = set() if entry["multi_select"] else None
        self._clarify_freetext = not entry["choices"]
        self._clarify_multi_base = None
        # Restore the earlier answer's cursor/checkbox position on re-visit.
        meta = (state.get("answer_meta") or {}).get(entry["qid"])
        choices = entry["choices"] or []
        if meta is None:
            return
        if meta.get("kind") == "choice":
            answer = state["answers"].get(entry["qid"])
            if answer in choices:
                state["selected"] = choices.index(answer)
        elif meta.get("kind") == "other":
            state["selected"] = len(choices)
        elif meta.get("kind") == "multi":
            checked = set()
            for label in meta.get("choices") or []:
                if label in choices:
                    checked.add(choices.index(label))
            if meta.get("other_text"):
                checked.add(len(choices))
            state["selected_indices"] = checked

    def _clarify_batch_lock(self, state, answer, meta=None) -> None:
        """Lock ``answer`` for the active batch question and advance.

        Overwrites any earlier answer for the same question (locked answers
        stay editable until the batch completes). ``meta`` records how the
        answer was produced ({"kind": "choice"|"other"|"multi", ...}) so a
        re-visit can restore the cursor and prefill an "Other" edit. Advances
        ``active`` to the next unanswered question; when every question has
        an answer, puts the answers dict on the response queue and tears down
        the panel.
        """
        entry = state["questions"][state["active"]]
        state["answers"][entry["qid"]] = answer
        state.setdefault("answer_meta", {})[entry["qid"]] = meta or {"kind": "choice"}
        self._persist_prompt_summary("?", "Clarify", entry["question"], str(answer))
        total = len(state["questions"])
        for offset in range(1, total + 1):
            candidate = (state["active"] + offset) % total
            if state["questions"][candidate]["qid"] not in state["answers"]:
                self._clarify_batch_set_active(state, candidate)
                return
        # Every question answered — resolve the batch.
        try:
            state["response_queue"].put(dict(state["answers"]))
        except Exception:
            pass
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_multi_base = None

    def _clarify_batch_enter(self, state) -> None:
        """Enter in batch choice mode: lock the active question's selection.

        Multi-select questions lock a JSON array string of the checked
        labels (the tool core parses it via ``_parse_multi_select_response``).
        Selecting "Other" switches to freetext; the freetext submit path
        locks the typed answer. Entering "Other" on a question whose earlier
        answer was typed prefills the composer with that text for editing.
        """
        choices = state.get("choices") or []
        selected = state.get("selected", 0)
        entry = state["questions"][state["active"]]
        meta = (state.get("answer_meta") or {}).get(entry["qid"]) or {}
        if state.get("multi_select"):
            indices = state.get("selected_indices") or set()
            sorted_idx = sorted(indices)
            selected_choices = [choices[i] for i in sorted_idx if i < len(choices)]
            other_checked = len(choices) in sorted_idx
            if other_checked:
                # Stash the checked real choices (possibly none) so the
                # freetext submit appends the typed answer to the array.
                self._clarify_multi_base = selected_choices
                self._clarify_freetext = True
                self._clarify_prefill = meta.get("other_text") or ""
                return
            self._clarify_batch_lock(
                state,
                json.dumps(selected_choices, ensure_ascii=False),
                meta={"kind": "multi", "choices": selected_choices, "other_text": ""},
            )
            return
        if selected < len(choices):
            self._clarify_batch_lock(
                state, choices[selected], meta={"kind": "choice"}
            )
            return
        # "Other" highlighted → switch to freetext; prefill an earlier typed
        # answer so Enter on an answered Other edits instead of retyping.
        self._clarify_freetext = True
        self._clarify_prefill = (
            meta.get("other_text") or "" if meta.get("kind") == "other" else ""
        )

    def _clarify_callback_batch(self, questions):
        """Batch clarify panel (A-compact): all questions, one active.

        Blocks on the response queue like the single-question path. Returns
        ``{"answers": {qid: raw_answer}}`` when every question is locked, the
        same dict plus ``"timed_out": True`` when the deadline expires with
        partial (or zero) answers, and passes a cancel string through
        unchanged so the tool core resolves the batch empty.
        """
        import time as _time

        from tools.clarify_gateway import resolve_clarify_timeout

        timeout = resolve_clarify_timeout(CLI_CONFIG)
        response_queue = queue.Queue()

        state = {
            "questions": list(questions),
            "answers": {},
            "answer_meta": {},
            "active": 0,
            "response_queue": response_queue,
            # Flat keys mirroring the active question — filled by
            # _clarify_batch_set_active below.
            "question": "",
            "choices": [],
            "selected": 0,
            "multi_select": False,
            "selected_indices": None,
        }
        self._clarify_state = state
        self._clarify_batch_set_active(state, 0)
        self._clarify_deadline = None if timeout <= 0 else _time.monotonic() + timeout
        self._paint_now()

        _last_countdown_refresh = _time.monotonic()
        while True:
            try:
                result = response_queue.get(timeout=1)
                self._clarify_deadline = None
                if isinstance(result, dict):
                    return {"answers": result}
                # Cancel path (Ctrl+C teardown) posts a plain string — pass
                # it through so the tool core resolves the batch empty.
                return result
            except queue.Empty:
                if self._clarify_deadline is not None:
                    remaining = self._clarify_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                now = _time.monotonic()
                if now - _last_countdown_refresh >= 1.0:
                    _last_countdown_refresh = now
                    self._paint_now()

        # Timed out — keep the answers locked so far and flag the timeout.
        partial = dict(state["answers"])
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = None
        self._clarify_multi_base = None
        self._paint_now()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — locked answers returned){_RST}")
        return {"answers": partial, "timed_out": True}

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

        self._capture_modal_input_snapshot()
        self._sudo_state = {
            "response_queue": response_queue,
        }
        self._sudo_deadline = _time.monotonic() + timeout

        # Modal prompt — paint immediately, bypassing the throttle/resize guard
        # so the prompt can't be dropped and time out unseen (#41098).
        self._paint_now()

        while True:
            try:
                result = response_queue.get(timeout=1)
                self._sudo_state = None
                self._sudo_deadline = 0
                self._restore_modal_input_snapshot()
                self._paint_now()
                if result:
                    _cprint(f"\n{_DIM}  ✓ Password received (cached for session){_RST}")
                else:
                    _cprint(f"\n{_DIM}  ⏭ Skipped{_RST}")
                return result
            except queue.Empty:
                remaining = self._sudo_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                self._paint_now()

        self._sudo_state = None
        self._sudo_deadline = 0
        self._restore_modal_input_snapshot()
        self._paint_now()
        _cprint(f"\n{_DIM}  ⏱ Timeout — continuing without sudo{_RST}")
        return ""

    def _approval_callback(self, command: str, description: str,
                           *, allow_permanent: bool = True,
                           allow_session: bool = True,
                           smart_denied: bool = False) -> str:
        """
        Prompt for dangerous command approval through the prompt_toolkit UI.

        Called from the agent thread. Shows a selection UI similar to clarify
        with choices: once / session / always / deny. Smart DENY owner
        overrides show only once / deny, as do gates that re-ask every time
        (allow_session=False). When allow_permanent is False for another
        reason (for example tirith), only 'always' is hidden.
        Long commands also get a 'view' option so the full command can be
        expanded before deciding.

        Uses _approval_lock to serialize concurrent requests (e.g. from
        parallel delegation subtasks) so each prompt gets its own turn
        and the shared _approval_state / _approval_deadline aren't clobbered.
        """
        import time as _time

        with self._approval_lock:
            timeout = int(CLI_CONFIG.get("approvals", {}).get("timeout", 300))
            response_queue = queue.Queue()

            self._approval_state = {
                "command": command,
                "description": description,
                "choices": self._approval_choices(
                    command,
                    allow_permanent=allow_permanent,
                    allow_session=allow_session,
                    smart_denied=smart_denied,
                ),
                "selected": 0,
                "response_queue": response_queue,
            }
            self._approval_deadline = _time.monotonic() + timeout

            # Modal prompt — paint immediately, bypassing the throttle/resize
            # guard. A throttled paint here can be silently dropped (250ms
            # window collision or in-flight resize), leaving the panel unseen so
            # the command is denied on timeout without the user ever seeing it
            # (#41098). The countdown refreshes below paint the same way.
            self._paint_now()

            _last_countdown_refresh = _time.monotonic()
            while True:
                try:
                    result = response_queue.get(timeout=1)
                    self._approval_state = None
                    self._approval_deadline = 0
                    self._paint_now()
                    _outcome_labels = {
                        "once": "allowed once",
                        "session": "allowed for session",
                        "always": "added to allowlist",
                        "deny": "denied",
                    }
                    self._persist_prompt_summary(
                        "⚠", "Approval", command,
                        _outcome_labels.get(result, str(result)),
                    )
                    return result
                except queue.Empty:
                    remaining = self._approval_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                    now = _time.monotonic()
                    if now - _last_countdown_refresh >= 1.0:
                        _last_countdown_refresh = now
                        self._paint_now()

            self._approval_state = None
            self._approval_deadline = 0
            self._paint_now()
            _cprint(f"\n{_DIM}  ⏱ Timeout — denying command{_RST}")
            self._persist_prompt_summary(
                "⚠", "Approval", command, "timed out (no response)",
            )
            return "timeout"

    def _approval_choices(self, command: str, *, allow_permanent: bool = True,
                          allow_session: bool = True,
                          smart_denied: bool = False) -> list[str]:
        """Return approval choices for a dangerous command prompt."""
        if smart_denied or not allow_session:
            choices = ["once", "deny"]
        else:
            choices = ["once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]
        if len(command) > 70:
            choices.append("view")
        return choices

    def _computer_use_approval_callback(self, action: str, args: dict, summary: str) -> str:
        """Adapt the generic approval UI for the computer_use tool.

        The computer_use handler expects verdicts of the form
        `approve_once` | `approve_session` | `always_approve` | `deny`.
        The CLI's built-in approval UI returns `once` | `session` | `always`
        | `deny`. Translate between the two.
        """
        # Build a command-ish string so the existing UI renders something
        # meaningful. `summary` is already a one-line human description.
        verdict = self._approval_callback(
            command=f"computer_use: {summary}",
            description=f"Allow computer_use to perform `{action}`?",
        )
        return {
            "once": "approve_once",
            "session": "approve_session",
            "always": "always_approve",
            "deny": "deny",
            "timeout": "timeout",
        }.get(verdict, "deny")

    def _handle_approval_selection(self) -> None:
        """Process the currently selected dangerous-command approval choice."""
        state = self._approval_state
        if not state:
            return

        selected = state.get("selected", 0)
        choices = state.get("choices")
        if not isinstance(choices, list):
            choices = []
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
        """Render the dangerous-command approval panel for the prompt_toolkit UI.

        Layout priority: title + command + choices must always render, even if
        the terminal is short or the description is long. Description is placed
        at the bottom of the panel and gets truncated to fit the remaining row
        budget. This prevents HSplit from clipping approve/deny off-screen when
        tirith findings produce multi-paragraph descriptions or when the user
        runs in a compact terminal pane.
        """
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
        cmd_display = command
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

        # Pre-wrap the mandatory content — command + choices must always render.
        cmd_wrapped = _wrap_panel_text(cmd_display, inner_text_width)
        if not show_full and "view" in choices and len(cmd_wrapped) > 4:
            cmd_wrapped = cmd_wrapped[:3] + _wrap_panel_text(
                "… (choose Show full command)",
                inner_text_width,
            )

        # (choice_index, wrapped_line) so we can re-apply selected styling below
        choice_wrapped: list[tuple[int, str]] = []
        for i, choice in enumerate(choices):
            label = choice_labels.get(choice, choice)
            # Show number prefix for quick selection (1-9 for items 1-9, 0 for 10th item)
            if i < 9:
                num_prefix = str(i + 1)
            elif i == 9:
                num_prefix = '0'
            else:
                num_prefix = ' '  # No number for items beyond 10th
            if i == selected:
                prefix = f'❯ {num_prefix}. '
            else:
                prefix = f'  {num_prefix}. '
            for wrapped in _wrap_panel_text(f"{prefix}{label}", inner_text_width, subsequent_indent="    "):
                choice_wrapped.append((i, wrapped))

        # Budget vertical space so HSplit never clips the command or choices.
        # Panel chrome (full layout with separators):
        #   top border + title + blank_after_title
        #   + blank_between_cmd_choices + bottom border = 5 rows.
        # In tight terminals we collapse to:
        #   top border + title + bottom border = 3 rows (no blanks).
        #
        # reserved_below: rows consumed below the approval panel by the
        # spinner/tool-progress line, status bar, input area, separators, and
        # prompt symbol. Measured at ~6 rows during live PTY approval prompts;
        # budget 6 so we don't overestimate the panel's room.
        term_rows = shutil.get_terminal_size((100, 24)).lines
        chrome_full = 5
        chrome_tight = 3
        reserved_below = 6

        available = max(0, term_rows - reserved_below)
        mandatory_full = chrome_full + len(cmd_wrapped) + len(choice_wrapped)

        # If the full-chrome panel doesn't fit, drop the separator blanks.
        # This keeps the command and every choice on-screen in compact terminals.
        use_compact_chrome = mandatory_full > available
        chrome_rows = chrome_tight if use_compact_chrome else chrome_full

        # If the command itself is too long to leave room for choices (e.g. user
        # hit "view" on a multi-hundred-character command), truncate it so the
        # approve/deny buttons still render. Keep at least 1 row of command.
        max_cmd_rows = max(1, available - chrome_rows - len(choice_wrapped))
        if len(cmd_wrapped) > max_cmd_rows:
            keep = max(1, max_cmd_rows - 1) if max_cmd_rows > 1 else 1
            cmd_wrapped = cmd_wrapped[:keep] + _wrap_panel_text(
                "… (command truncated — use /logs or /debug for full text)",
                inner_text_width,
            )

        # Allocate any remaining rows to description. The extra -1 in full mode
        # accounts for the blank separator between choices and description.
        mandatory_no_desc = chrome_rows + len(cmd_wrapped) + len(choice_wrapped)
        desc_sep_cost = 0 if use_compact_chrome else 1
        available_for_desc = available - mandatory_no_desc - desc_sep_cost
        # Even on huge terminals, cap description height so the panel stays compact.
        available_for_desc = max(0, min(available_for_desc, 10))

        desc_wrapped = _wrap_panel_text(description, inner_text_width) if description else []
        if available_for_desc < 1 or not desc_wrapped:
            desc_wrapped = []
        elif len(desc_wrapped) > available_for_desc:
            keep = max(1, available_for_desc - 1)
            desc_wrapped = desc_wrapped[:keep] + ["… (description truncated)"]

        # Render: title → command → choices → description (description last so
        # any remaining overflow clips from the bottom of the least-critical
        # content, never from the command or choices). Use compact chrome (no
        # blank separators) when the terminal is tight.
        lines = []
        lines.append(('class:approval-border', '╭' + ('─' * box_width) + '╮\n'))
        _append_panel_line(lines, 'class:approval-border', 'class:approval-title', title, box_width)
        if not use_compact_chrome:
            _append_blank_panel_line(lines, 'class:approval-border', box_width)

        for wrapped in cmd_wrapped:
            _append_panel_line(lines, 'class:approval-border', 'class:approval-cmd', wrapped, box_width)
        if not use_compact_chrome:
            _append_blank_panel_line(lines, 'class:approval-border', box_width)

        for i, wrapped in choice_wrapped:
            style = 'class:approval-selected' if i == selected else 'class:approval-choice'
            _append_panel_line(lines, 'class:approval-border', style, wrapped, box_width)

        if desc_wrapped:
            if not use_compact_chrome:
                _append_blank_panel_line(lines, 'class:approval-border', box_width)
            for wrapped in desc_wrapped:
                _append_panel_line(lines, 'class:approval-border', 'class:approval-desc', wrapped, box_width)

        lines.append(('class:approval-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines

    def _secret_capture_callback(self, var_name: str, prompt: str, metadata=None) -> dict:
        return prompt_for_secret(self, var_name, prompt, metadata)

    def _capture_modal_input_snapshot(self) -> None:
        """Temporarily clear the input buffer and save the user's in-progress draft."""
        if self._modal_input_snapshot is not None or not getattr(self, "_app", None):
            return
        try:
            buf = self._app.current_buffer
            self._modal_input_snapshot = {
                "text": buf.text,
                "cursor_position": buf.cursor_position,
            }
            buf.reset()
        except Exception:
            self._modal_input_snapshot = None

    def _restore_modal_input_snapshot(self) -> None:
        """Restore any draft text that was present before a modal prompt opened."""
        snapshot = self._modal_input_snapshot
        self._modal_input_snapshot = None
        if not snapshot or not getattr(self, "_app", None):
            return
        try:
            buf = self._app.current_buffer
            buf.text = snapshot.get("text", "")
            buf.cursor_position = min(snapshot.get("cursor_position", 0), len(buf.text))
        except Exception:
            pass

    def _clear_active_overlays_for_interrupt(self) -> None:
        """Drain and clear every input-blocking overlay left by an interrupted agent.

        approval/clarify/sudo/secret prompts each block a worker thread on a
        ``response_queue.get()``.  When the agent is interrupted the worker
        thread is torn down, but the overlay's state dict stays set — leaving
        the CLI input gated (``read_only`` condition + keypress filter) with no
        thread servicing the prompt.  The result is a frozen terminal until the
        prompt's own timeout expires.  Push a terminal value onto each queue so
        any still-blocked thread unblocks cleanly, then nil the state out and
        restore the user's pre-modal draft (#14026).

        Safe default per prompt: approval -> "deny", clarify/sudo/secret ->
        cancel (None / empty).  Each step is wrapped so a dead queue can't
        prevent clearing the others.
        """
        if self._approval_state:
            try:
                self._approval_state["response_queue"].put("deny")
            except Exception:
                pass
            self._approval_state = None
        if self._clarify_state:
            try:
                self._clarify_state["response_queue"].put(
                    "The user cancelled. Use your best judgement to proceed."
                )
            except Exception:
                pass
            self._clarify_state = None
            self._clarify_freetext = False
            self._clarify_multi_base = None
        if self._sudo_state:
            try:
                self._sudo_state["response_queue"].put("")
            except Exception:
                pass
            self._sudo_state = None
            self._sudo_deadline = 0
            self._restore_modal_input_snapshot()
        if self._secret_state:
            try:
                self._cancel_secret_capture()
            except Exception:
                self._secret_state = None

    def _submit_secret_response(self, value: str) -> None:
        if not self._secret_state:
            return
        self._secret_state["response_queue"].put(value)
        self._secret_state = None
        self._secret_deadline = 0
        # Modal teardown — paint directly so the secret panel clears at once and
        # isn't held by the _invalidate throttle/resize guard (#41098).
        self._paint_now()

    def _cancel_secret_capture(self) -> None:
        self._submit_secret_response("")

    def _clear_secret_input_buffer(self) -> None:
        if getattr(self, "_app", None):
            try:
                self._app.current_buffer.reset()
            except Exception:
                pass

    def chat(self, message, images: list = None, voice_input: bool = False) -> Optional[str]:
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
            voice_input: True when the message came from voice transcription
                (gates the concise voice-response prefix, #65827)
            
        Returns:
            The agent's response, or None on error
        """
        # Single-query and direct chat callers do not go through run(), so
        # register secure secret capture here as well.
        set_secret_capture_callback(self._secret_capture_callback)

        # Reset the per-turn interrupt flag. Any subsequent path that
        # discovers an interrupt (below, after run_conversation) will flip
        # this to True. Early returns (credential refresh failure, etc.)
        # leave it False, which is correct — those aren't user interrupts.
        self._last_turn_interrupted = False

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
            request_overrides=turn_route.get("request_overrides"),
        ):
            return None
        agent = self.agent
        if agent is None:
            return None

        # Route image attachments based on the active model's vision capability.
        # "native" → pass pixels as OpenAI-style content parts (adapters
        #            translate for Anthropic/Gemini/Bedrock).
        # "text"   → pre-analyze each image with vision_analyze and prepend the
        #            description as text — works with non-vision models.
        # See agent/image_routing.py for the decision table.
        if images:
            try:
                from agent.image_routing import (
                    build_native_content_parts,
                    decide_image_input_mode,
                )
                from hermes_cli.config import load_config

                _img_model, _img_provider = "", ""
                if isinstance(self.model, dict):
                    _img_model, _ = _split_model_config_default(self.model)
                else:
                    _img_model = str(self.model or "")
                if isinstance(self.provider, dict):
                    _, _img_provider = _split_model_config_default(self.provider)
                else:
                    _img_provider = str(self.provider or "")
                _img_mode = decide_image_input_mode(
                    _img_provider.strip(),
                    _img_model.strip(),
                    load_config(),
                    requested_provider=(self.requested_provider or "").strip(),
                )
            except Exception as _img_exc:
                logging.debug("image_routing decision failed, defaulting to text: %s", _img_exc)
                _img_mode = "text"

            if _img_mode == "native":
                try:
                    _text_for_parts = message if isinstance(message, str) else ""
                    _img_str_paths = [str(p) for p in images]
                    _parts, _skipped = build_native_content_parts(
                        _text_for_parts,
                        _img_str_paths,
                    )
                    if _skipped:
                        _cprint(
                            f"  {_DIM}⚠ skipped {len(_skipped)} unreadable image path(s){_RST}"
                        )
                    if any(p.get("type") == "image_url" for p in _parts):
                        _img_names = ", ".join(Path(p).name for p in _img_str_paths)
                        _cprint(
                            f"  {_DIM}📎 attaching {len(images)} image(s) natively "
                            f"(model supports vision): {_img_names}{_RST}"
                        )
                        message = _parts
                    else:
                        # All images unreadable — fall back to text enrichment.
                        message = self._preprocess_images_with_vision(
                            message if isinstance(message, str) else "", images
                        )
                except Exception as _img_exc:
                    logging.warning("native image attach failed, falling back to text: %s", _img_exc)
                    message = self._preprocess_images_with_vision(
                        message if isinstance(message, str) else "", images
                    )
            else:
                message = self._preprocess_images_with_vision(
                    message if isinstance(message, str) else "", images
                )

        # Expand @ context references (e.g. @file:main.py, @diff, @folder:src/)
        if isinstance(message, str) and "@" in message:
            try:
                from agent.context_references import preprocess_context_references
                from agent.model_metadata import get_model_context_length
                _ctx_len = get_model_context_length(
                    self.model, base_url=self.base_url or "", api_key=self.api_key or "",
                    provider=self.provider or "",
                    config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None)
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
                if isinstance(e, OSError) and e.errno == errno.EIO:
                    self._mark_terminal_io_broken("process_loop")
                    logger.warning("process_loop EIO — freezing UI paints (#81521): %s", e)
                    continue
                logger.warning("process_loop unhandled error (msg may be lost): %s", e)

    def _tui_idle_tick(self):
        """Idle housekeeping between inputs (agent not running)."""
        self._check_config_mcp_changes()  # auto-reload MCP on mcp_servers change
        # Termios drift heal first: a drifted tty makes the CLI look dead while the loop is healthy.
        for step in (
            self._check_termios_drift,
            lambda: self._drain_process_notifications("cli-idle"),
            self._maybe_fire_loop_tick,
            self._maybe_resume_parked_goal,
        ):
            with suppress(Exception):
                step()

    def _tui_process_one_input(self, user_input):
        """Route one submitted input: file drop, /resume pick, ! shell, slash command, or a chat turn."""
        from tools.process_registry_notifications import TimelineNotification
        user_input, is_voice_input, is_seeded_query = self._tui_unwrap_input(user_input)
        if not user_input:
            return
        notification_preview = user_input if isinstance(user_input, TimelineNotification) else None
        self._status_bar_suppressed_after_resize = False  # input ends post-resize suppression

        submit_images = []
        if isinstance(user_input, tuple):
            user_input, submit_images = user_input

        if isinstance(user_input, str):
            user_input = _strip_leaked_bracketed_paste_wrappers(user_input)
            user_input, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(user_input)
            if _had_mouse_reports:
                self._recover_terminal_input_modes(reason="mouse reports leaked into submitted input")

        # A typed bare stop phrase ends an active voice chat (transcripts are checked earlier).
        if not is_voice_input and self._typed_voice_stop(user_input):
            return

        # File drops are detected before any dispatch; seeded -q prompts are literal text.
        _file_drop = _detect_file_drop(user_input) if isinstance(user_input, str) and not is_seeded_query else None
        if _file_drop:
            _drop_path = _file_drop["path"]
            _remainder = _file_drop["remainder"]
            if _file_drop["is_image"]:
                submit_images.append(_drop_path)
                user_input = _remainder or f"[User attached image: {_drop_path.name}]"
                _cprint(f"  📎 Auto-attached image: {_drop_path.name}")
            else:
                _cprint(f"  📄 Detected file: {_drop_path.name}")
                user_input = f"[User attached file: {_drop_path}]" + (f"\n{_remainder}" if _remainder else "")
        elif isinstance(user_input, str):
            # A bare number right after a bare `/resume` selects that session (never sent to the agent).
            if self._pending_resume_sessions and self._consume_pending_resume_selection(user_input):
                return
            if not is_seeded_query:
                if self.handle_bang_shell(user_input):
                    return
                if _looks_like_slash_command(user_input):
                    user_input = self._tui_run_slash_input(user_input)
                    if user_input is None:
                        return

        if isinstance(user_input, str) and _PASTE_REF_RE.search(user_input):
            user_input = self._expand_paste_references(user_input)
        print()
        self._print_user_message_preview(notification_preview or user_input)

        if submit_images:
            n = len(submit_images)
            _cprint(f"  {_DIM}📎 {n} image{'s' if n > 1 else ''} attached{_RST}")

        self._agent_running = self._interactive_turn = True
        self._pet_turn_error = self._pet_reasoning = False
        self._turn_summary_begin()
        self._app.invalidate()
        try:
            self.chat(notification_preview or user_input, images=submit_images or None, voice_input=is_voice_input)
        finally:
            self._tui_after_turn()

    def _tui_run_slash_input(self, user_input: str):
        """Dispatch a slash command. Returns the pending agent seed to run as a chat turn, else None."""
        _cprint(f"\n⚙️  {user_input}")
        try:
            if not self.process_command(user_input):
                self._should_exit = True
                if self._app.is_running:
                    self._app.exit()
        except KeyboardInterrupt:
            # Ctrl+C during a slow slash command returns to the prompt instead of exiting.
            _cprint("\n[dim]Command interrupted.[/dim]")
            return None
        _seed, self._pending_agent_seed = self._pending_agent_seed, None
        return _seed or None

    def _tui_after_turn(self):
        """Post-turn bookkeeping after chat() returns (normal, error, or interrupt)."""
        self._agent_running = self._pet_reasoning = False
        self._spinner_text = self._last_scrollback_tool = ""
        self._tool_start_time = 0.0
        self._pending_tool_info.clear()
        self._pet_react_turn_end()
        self._turn_summary_emit()
        self._interactive_turn = False
        self._app.invalidate()

        # After an interrupt the renderer may have drifted (leaked CPR text, VT100 parser
        # stalled mid-escape): drain stray bytes and force a clean redraw.
        if self._last_turn_interrupted:
            self._recover_terminal_after_interrupt()

        # Re-queue any messages that arrived in _interrupt_queue while the agent was running and were never
        # claimed by the explicit interrupt path. See _drain_interrupt_queue_to_pending_input for the full
        # rationale. Regression of #17666 / #18760 — the drain block from the original PR #17939 was
        # deferred as "worth its own review" and never re-landed (#20271).
        self._drain_interrupt_queue_to_pending_input()

        # /goal continuation (queued user input still preempts), then /loop tick completion.
        for hook, what in (
            (self._maybe_continue_goal_after_turn, "goal continuation"),
            (self._maybe_complete_loop_tick_after_turn, "loop completion"),
        ):
            try:
                hook()
            except Exception as _exc:
                logging.debug("%s hook failed: %s", what, _exc)

        # Continuous voice: restart recording off-thread (beep + recorder start would block process_loop).
        if self._voice_mode and self._voice_continuous and not self._voice_recording:
            def _restart_recording():
                try:
                    if self._voice_tts:
                        self._voice_tts_done.wait(timeout=60)
                        time.sleep(0.3)
                    # A barge-in capture already owns the mic and submits the interruption itself.
                    if self._voice_barge_capture.is_set():
                        return
                    self._voice_start_recording()
                    self._app.invalidate()
                except Exception as e:
                    _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
            threading.Thread(target=_restart_recording, daemon=True).start()

        with suppress(Exception):
            self._drain_process_notifications("cli-post-turn")

    def _tui_signal_handler(self, signum, frame):
        """SIGHUP/SIGTERM -> graceful shutdown.

        The agent is hard-interrupted first so its daemon thread can kill the tool's
        setsid subprocess group before the main thread unwinds (else an orphan child).
        ``logger.debug`` is guarded: logging is not reentrant-safe and a shutdown race
        can raise ``KeyError`` inside the handler, bypassing prompt_toolkit's unwind.
        """
        with suppress(Exception):
            logger.debug("Received signal %s, triggering graceful shutdown", signum)
        # Arm the backstop IMMEDIATELY: if the unwind wedges, _run_cleanup never arms its own.
        # Shutdown intent is now unambiguous — arm the exit backstop IMMEDIATELY, before the graceful unwind
        # below. If any step of that unwind wedges (main thread parked in a syscall, prompt_toolkit teardown
        # never returning), _run_cleanup never runs and would never arm its own watchdog — leaving a "dead"
        # CLI alive for minutes (#65998 class).
        # Arm the exit backstop now that shutdown intent is unambiguous — covers wedges in the unwind below
        # that would otherwise leave the process alive with no watchdog (#65998 class).
        _arm_exit_watchdog_on_shutdown_signal()
        if self._agent_running:
            _interrupt_agent_for_signal(self.agent, signum)
        # Prefer app.exit() over raising KeyboardInterrupt: a KBI from a signal handler
        # lands in a pt Task ("Unhandled exception in event loop" + "Press ENTER to
        # continue..."); call_soon_threadsafe lets the loop unwind normally.
        try:
            from prompt_toolkit.application.current import get_app_or_none
            _app = get_app_or_none()
            _loop = getattr(_app, "loop", None)
            if _loop is not None:
                _loop.call_soon_threadsafe(_app.exit)
                return  # clean unwind — no traceback, no ENTER pause
        except Exception:
            pass
        raise KeyboardInterrupt()  # fallback for non-prompt_toolkit contexts

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
        slash_confirm_widget=None,
        clarify_widget,
        model_picker_widget=None,
        command_palette_widget=None,
        spinner_widget=None,
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
            item for item in [
                Window(height=0),
                sudo_widget,
                secret_widget,
                approval_widget,
                slash_confirm_widget,
                clarify_widget,
                model_picker_widget,
                command_palette_widget,
                spinner_widget,
                spacer,
                *self._get_extra_tui_widgets(),
                getattr(self, "_pet_widget", None),
                getattr(self, "_stash_panel_widget", None),
                status_bar,
                input_rule_top,
                image_bar,
                input_area,
                input_rule_bot,
                voice_status_bar,
                completions_menu,
            ] if item is not None
        ]

    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        if not self._claim_active_session("cli"):
            return

        # Detect light/dark terminal mode now (before pt grabs the tty).
        # Caches the result so subsequent _hex_to_ansi / style calls
        # don't risk re-querying mid-render.
        try:
            _detect_light_mode()
        # Scroll the cursor to the last row so banner, responses and prompt pin to the bottom.
        with suppress(Exception):
            _term_lines = shutil.get_terminal_size().lines
            if _term_lines > 2:
                print("\n" * (_term_lines - 1), end="", flush=True)

        self.show_banner()
        self._show_security_advisories()
        self._show_browser_backend_notice()

        # First-run: an unconfigured install routes into provider onboarding instead of
        # a chat that spins ~30s and fails with a provider-specific error. TTY only.
        try:
            if sys.stdin.isatty() and not self._runtime_credentials_ready():
                self._offer_first_run_setup()
        except Exception:
            logger.debug("first-run setup offer failed", exc_info=True)

        if self._resumed and self._preload_resumed_session():
            self._display_resumed_history()

        _welcome_skin = None  # stays None when the skin engine failed
        _welcome_text = "Welcome to Hermes Agent! Type your message or /help for commands."
        _welcome_color = "#FFF8DC"
        try:
            from hermes_cli.skin_engine import get_active_skin
            _welcome_skin = get_active_skin()
            _welcome_text = _welcome_skin.get_branding("welcome", _welcome_text)
            _welcome_color = _welcome_skin.get_color("banner_text", _welcome_color)
        except Exception:
            pass
        self._console_print(f"[{_welcome_color}]{_welcome_text}[/]")

        self._tui_startup_prewarm_and_warnings(_welcome_skin)
        self._print_random_tip()

        self._tui_startup_background_maintenance()
        # Before the background preload is folded in (at agent init), show the REQUESTED names.
        _skills_for_line = self.preloaded_skills or list(self._preload_skills_requested or [])
        if _skills_for_line and not self._startup_skills_line_shown:
            self._console_print(f"[bold {_accent_hex()}]Activated skills:[/] {', '.join(_skills_for_line)}")
            self._startup_skills_line_shown = True
        self._console_print()

    def _tui_startup_prewarm_and_warnings(self, _welcome_skin):
        """Idle-window prewarms (picker cache, agent runtime imports) plus the redaction-off and OpenClaw-residue banners."""
        # Warm the /model picker cache off-thread (else its first open blocks ~1-2s).
        with suppress(Exception):
            from hermes_cli.model_switch_providers import prewarm_picker_cache_async
            prewarm_picker_cache_async()

        # Pre-import the agent runtime (~1.5s: run_agent + OpenAI SDK) off-thread; the import
        # lock makes an early submit block on the remaining work rather than redo it.
        # Skipped when Termux defers agent startup on purpose.
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            def _prewarm_agent_runtime() -> None:
                try:
                    import run_agent  # noqa: F401  (imports model_tools + tool registry)
                    import openai  # noqa: F401
                except Exception:
                    logger.debug("agent runtime pre-import failed", exc_info=True)

            threading.Thread(target=_prewarm_agent_runtime, name="agent-runtime-prewarm", daemon=True).start()

        # Redaction is ON by default; be loud when the operator turned it off.
        with suppress(Exception):
            # The redactor snapshots its state at import time so any toggle now won't affect the running
            # process — we just want the operator to see that they're running without the safety net. See
            # #17691.
            _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
            if _redact_raw.lower() not in {"1", "true", "yes", "on"}:
                self._console_print(
                    "[bold red]⚠  Secret redaction is DISABLED[/] "
                    f"(HERMES_REDACT_SECRETS={_redact_raw}). "
                    "API keys and tokens may appear verbatim in chat output, "
                    "session JSONs, and logs. Set "
                    "[cyan]security.redact_secrets: true[/] in config.yaml "
                    "to re-enable."
                )
        # One-time banner when ~/.openclaw/ is left over from a migration.
        try:
            from agent.onboarding import (
                OPENCLAW_RESIDUE_FLAG, detect_openclaw_residue, is_seen, mark_seen, openclaw_residue_hint_cli,
            )
            if not is_seen(self.config, OPENCLAW_RESIDUE_FLAG) and detect_openclaw_residue():
                try:
                    _resid_color = _welcome_skin.get_color("banner_dim", "#B8860B")
                except Exception:
                    _resid_color = "#B8860B"
                self._console_print(f"[{_resid_color}]{openclaw_residue_hint_cli()}[/]")
                try:
                    from hermes_cli.config import get_config_path as _get_cfg_path_resid
                    mark_seen(_get_cfg_path_resid(), OPENCLAW_RESIDUE_FLAG)
                except Exception:
                    pass  # banner fires again next session
        except Exception:
            pass

    def _tui_startup_background_maintenance(self):
        """Best-effort startup passes: curator skill maintenance, personal + org skill sync."""
        with suppress(Exception):
            from agent.curator import maybe_run_curator
            maybe_run_curator(
                idle_for_seconds=float("inf"),  # CLI startup = fully idle
                on_summary=lambda msg: self._console_print(f"[dim #6b7684]💾 {msg}[/]"),
            )

        # Skill sync (personal, then org-shared): inert unless the access gate is open
        # and a sync base URL is configured. The org pull is gated on a real org role on
        # the token (only issued for multi-member orgs), so a solo account never hits
        # the network here. Both fail-quiet.
        try:
            from tools.skills_sync_client import maybe_pull_skills
            from tools.skills_sync_client_org import maybe_pull_org_skills
        except Exception:
            return
        for pull in (maybe_pull_skills, maybe_pull_org_skills):
            with suppress(Exception):
                pull()

        # Org-shared skills — pull the organisation's approved set into the
        # read-only mirror. Gated on real org membership: resolve_org_identity
        # requires an org role on the token, which is only issued for
        # multi-member organisations, so a solo account never reaches the
        # network here. Fail-quiet, exactly like the personal pull above.
        try:
            from tools.skills_sync_client import maybe_pull_org_skills
            maybe_pull_org_skills()
        except Exception:
            pass
        _skills_for_line = self.preloaded_skills or list(
            getattr(self, "_preload_skills_requested", []) or []
        )
        if _skills_for_line and not self._startup_skills_line_shown:
            # When the background --skills preload hasn't been folded in yet
            # (it joins at agent init), show the REQUESTED names — identical
            # to the loaded set except for typo'd names, which warn later.
            skills_label = ", ".join(_skills_for_line)
            self._console_print(
                f"[bold {_accent_hex()}]Activated skills:[/] {skills_label}"
            )
            self._startup_skills_line_shown = True
        self._console_print()
        
        # State for async operation
        self._agent_running = False
        self._pending_input = queue.Queue()     # For normal input (commands + new queries)
        self._interrupt_queue = queue.Queue()   # For messages typed while agent is running
        # See constructor note. Mirrored here for the run() path that skips
        # the earlier __init__ branch.
        self._last_turn_interrupted = False
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
        self._modal_input_snapshot = None

        # Dangerous command approval state (similar mechanism to clarify)
        self._approval_state = None     # dict with command, description, choices, selected, response_queue
        self._approval_deadline = 0
        self._approval_lock = threading.Lock()  # serialize concurrent approval prompts (delegation race fix)

        # Destructive slash-command confirmation state (/new, /clear, /undo).
        # These prompts are answered through the prompt_toolkit composer, not
        # raw input(), so the option labels stay visible and Enter does not EOF
        # the whole app.
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0

        # Slash command loading state
        self._command_running = False
        self._command_blocks_input = False
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
        self._voice_tts_stop = None  # active streaming pipeline's stop event
        self._voice_barge_capture = threading.Event()  # barge monitor is capturing the interruption
        self._voice_last_tts_text = ""  # most recently spoken TTS text (echo guard, #75780)
        self._voice_barge_phase = None  # "generation" or "playback" phase of the last barge trip

        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            self._install_tool_callbacks()

        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            self._ensure_tirith_security()
        
        # Key bindings for the input area
        kb = KeyBindings()

        _multiline_shortcuts_enabled = _cli_multiline_shortcuts_enabled(self.config or CLI_CONFIG)

        from prompt_toolkit.keys import Keys as _IgnoreKeys

        @kb.add(_IgnoreKeys.Ignore, eager=True)
        def handle_ignored_terminal_sequence(event):
            """Consume parser-level ignored terminal sequences before self-insert.

            install_ignored_terminal_sequences() in hermes_cli.pt_input_extras
            registers focus reports (CSI I / CSI O) as Keys.Ignore at the
            VT100 parser level. Without this no-op binding the default
            self-insert path would still fire and the bytes would land in
            the buffer.

            Focus-in (CSI I) additionally schedules a rate-limited full
            repaint: while the tab/window was hidden the emulator may have
            coalesced output or repainted the surface, so prompt_toolkit's
            incremental diff would stack a fresh copy of the prompt chrome
            on top of the stale one (#60920 focus-regain variant, #25337).
            """
            try:
                for press in getattr(event, "key_sequence", None) or ():
                    if getattr(press, "data", None) == "\x1b[I":
                        self._schedule_focus_regain_redraw()
                        break
            except Exception:
                pass
            return None

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

            # --- Slash-command confirmation: submit typed or highlighted choice ---
            if self._slash_confirm_state:
                text = event.app.current_buffer.text.strip()
                choices = self._slash_confirm_state.get("choices") or []
                choice = self._normalize_slash_confirm_choice(text, choices) if text else None
                if choice is None:
                    selected = self._slash_confirm_state.get("selected", 0)
                    if 0 <= selected < len(choices):
                        choice = choices[selected][0]
                self._submit_slash_confirm_response(choice or "cancel")
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # --- /model picker modal ---
            if self._model_picker_state:
                try:
                    # Picker selections follow the same session-scoped default
                    # as /model <name>; honour model.persist_switch_by_default.
                    from hermes_cli.model_switch import resolve_persist_behavior

                    self._handle_model_picker_selection(
                        persist_global=resolve_persist_behavior(False, False)
                    )
                except Exception as _exc:
                    _cprint(f"  ✗ Model selection failed: {_exc}")
                    self._close_model_picker()
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # --- Clarify freetext mode: user typed their own answer ---
            if self._clarify_freetext and self._clarify_state:
                text = event.app.current_buffer.text.strip()
                if text:
                    state = self._clarify_state
                    # Batch mode: lock the typed answer for the active question
                    if state.get("questions"):
                        base = getattr(self, '_clarify_multi_base', None)
                        if base is not None:
                            # Multi-select "Other": append the typed answer to
                            # the checked labels as a JSON array string.
                            answer = json.dumps(base + [text], ensure_ascii=False)
                            meta = {"kind": "multi", "choices": list(base), "other_text": text}
                            self._clarify_multi_base = None
                        else:
                            answer = text
                            meta = {"kind": "other", "other_text": text}
                        self._clarify_freetext = False
                        self._clarify_prefill = ""
                        self._clarify_batch_lock(state, answer, meta=meta)
                        event.app.current_buffer.reset()
                        event.app.invalidate()
                        return
                    # multi-select: prepend previously checked real choices
                    base = getattr(self, '_clarify_multi_base', None)
                    if base:
                        text = ", ".join(base) + ", " + text
                        self._clarify_multi_base = None
                    self._clarify_state["response_queue"].put(text)
                    self._clarify_state = None
                    self._clarify_freetext = False
                    event.app.current_buffer.reset()
                    event.app.invalidate()
                return

            # --- Clarify choice mode: confirm the highlighted selection ---
            if self._clarify_state and not self._clarify_freetext:
                state = self._clarify_state
                # Batch mode: Enter locks the active question's answer and
                # advances to the next unanswered question.
                if state.get("questions"):
                    self._clarify_batch_enter(state)
                    # Editing an earlier "Other" answer: prefill the composer
                    # with the previously typed text.
                    if self._clarify_freetext and self._clarify_prefill:
                        event.app.current_buffer.text = self._clarify_prefill
                        event.app.current_buffer.cursor_position = len(self._clarify_prefill)
                        self._clarify_prefill = ""
                    event.app.invalidate()
                    return
                selected = state["selected"]
                choices = state.get("choices") or []
                # multi-select support: submit comma-joined list of checked choices
                if state.get("multi_select"):
                    indices = state.get("selected_indices")
                    if not indices:
                        # Nothing checked → submit empty string (parses to [])
                        state["response_queue"].put("")
                        self._clarify_state = None
                        event.app.invalidate()
                        return
                    sorted_idx = sorted(indices)
                    selected_choices = [choices[i] for i in sorted_idx if i < len(choices)]
                    other_checked = len(choices) in sorted_idx
                    if other_checked and selected_choices:
                        # "Other" + real choices: store base choices, switch to freetext
                        # so the user can type a custom answer that gets appended
                        self._clarify_multi_base = selected_choices
                        self._clarify_freetext = True
                        event.app.invalidate()
                        return
                    if selected_choices:
                        state["response_queue"].put(", ".join(selected_choices))
                        self._clarify_state = None
                        event.app.invalidate()
                        return
                    # Only "Other" was checked → switch to freetext
                    self._clarify_freetext = True
                    event.app.invalidate()
                    return
                # Original single-select behavior: submit the highlighted choice
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
            raw_text = event.app.current_buffer.text
            if (
                _multiline_shortcuts_enabled
                and event.app.current_buffer.cursor_position == len(raw_text)
                and _is_backslash_line_continuation(raw_text)
            ):
                continued = _apply_backslash_line_continuation(raw_text)
                event.app.current_buffer.text = continued
                event.app.current_buffer.cursor_position = len(continued)
                event.app.invalidate()
                return
            text = raw_text.strip()
            has_images = bool(self._attached_images)
            if text or has_images:
                # Handle /model directly on the UI thread so interactive pickers
                # can safely use prompt_toolkit terminal handoff helpers.
                if self._should_handle_model_command_inline(text, has_images=has_images):
                    if not self.process_command(text):
                        self._should_exit = True
                        if event.app.is_running:
                            event.app.exit()
                    event.app.current_buffer.reset(append_to_history=True)
                    # Force a repaint: process_command() prints through
                    # patch_stdout (scrolls output above the prompt) and never
                    # invalidates the app, so the just-cleared input area can
                    # keep showing the submitted text until some unrelated
                    # redraw fires. Every other early-return branch in this
                    # handler invalidates after reset — match them.
                    event.app.invalidate()
                    return

                # Handle /steer while the agent is running immediately on the
                # UI thread.  Queuing through _pending_input would deadlock the
                # steer until after the agent loop finishes (process_loop is
                # blocked inside self.chat()), which turns /steer into a
                # post-run next-turn message — defeating mid-run injection.
                # agent.steer() is thread-safe (holds _pending_steer_lock).
                if self._should_handle_steer_command_inline(text, has_images=has_images):
                    self.process_command(text)
                    event.app.current_buffer.reset(append_to_history=True)
                    # Force a repaint after clearing the buffer.  /steer is
                    # dispatched mid-run while the agent streams output through
                    # patch_stdout; process_command() never invalidates the
                    # app, so without this the submitted "/steer <text>" can
                    # linger in the input area (looking unsent) and invite an
                    # accidental re-submit. See issue #34569.
                    event.app.invalidate()
                    return

                # Same treatment for /background (/bg, /btw) while the agent is
                # running.  Queuing it defeats the entire point of the command:
                # process_loop is blocked inside self.chat(), so the background
                # task would only start once the foreground turn it was meant to
                # run alongside has already finished (#75221).  The foreground
                # turn is left alone: no interrupt, no steer.
                if self._should_handle_background_command_inline(
                    text, has_images=has_images
                ):
                    self.process_command(text)
                    event.app.current_buffer.reset(append_to_history=True)
                    # Repaint for the same reason as the /steer branch above:
                    # process_command() prints through patch_stdout and never
                    # invalidates the app, so the submitted text can linger in
                    # the input area looking unsent.
                    event.app.invalidate()
                    return

                # Snapshot and clear attached images
                images = list(self._attached_images)
                self._attached_images.clear()
                event.app.invalidate()
                # Bundle text + images as a tuple when images are present
                payload = (text, images) if images else text
                # A bang command is treated like a slash command while the
                # agent is busy: it must never be routed into steer/redirect
                # (which would inject `!git status` into the model's context as
                # a prompt). It queues and runs locally once the loop drains.
                _is_local_dispatch = bool(text) and (
                    _looks_like_slash_command(text) or text.strip().startswith("!")
                )
                if self._agent_running and not _is_local_dispatch:
                    _effective_mode = self.busy_input_mode
                    redirected = False
                    if _effective_mode == "steer":
                        # Route Enter through /steer — inject mid-run after the
                        # next tool call.  Images can't ride along (steer only
                        # appends text), so fall back to queue when images are
                        # attached.  If the agent lacks steer() or rejects the
                        # payload, also fall back to queue so nothing is lost.
                        if images or not text:
                            _effective_mode = "queue"
                        else:
                            accepted = False
                            try:
                                if self.agent is not None and hasattr(self.agent, "steer"):
                                    accepted = bool(self.agent.steer(text))
                            except Exception as exc:
                                _cprint(f"  {_DIM}Steer failed ({exc}) — queued for next turn.{_RST}")
                                accepted = False
                            if accepted:
                                preview = text[:80] + ("..." if len(text) > 80 else "")
                                _cprint(f"  {_ACCENT}⏩ Steered: '{preview}'{_RST}")
                            else:
                                _effective_mode = "queue"
                    if _effective_mode == "queue":
                        # Queue for the next turn instead of interrupting
                        self._pending_input.put(payload)
                        preview = text if text else f"[{len(images)} image{'s' if len(images) != 1 else ''} attached]"
                        _cprint(f"  Queued for the next turn: {preview[:80]}{'...' if len(preview) > 80 else ''}")
                    elif _effective_mode == "interrupt":
                        if not images and text:
                            try:
                                if (
                                    self.agent is not None
                                    and getattr(
                                        self.agent,
                                        "_supports_active_turn_redirect",
                                        False,
                                    )
                                    is True
                                    and hasattr(self.agent, "redirect")
                                ):
                                    redirected = bool(self.agent.redirect(text))
                            except Exception:
                                redirected = False
                        if redirected:
                            preview = text[:80] + ("..." if len(text) > 80 else "")
                            _cprint(f"  {_ACCENT}↪ Redirected current turn: '{preview}'{_RST}")
                        else:
                            # Compatibility path for older agents, multimodal
                            # follow-ups, or a turn that finished in the race.
                            self._interrupt_queue.put(payload)
                            try:
                                _dbg = _hermes_home / "interrupt_debug.log"
                                with open(_dbg, "a", encoding="utf-8") as _f:
                                    _f.write(f"{time.strftime('%H:%M:%S')} ENTER: queued interrupt msg={str(payload)[:60]!r}, "
                                             f"agent_running={self._agent_running}\n")
                            except Exception:
                                pass
                    # First-touch onboarding: on the very first busy-while-running
                    # event for this install, print a one-line tip explaining the
                    # /busy knob.  Flag persists to config.yaml and never fires
                    # again.  Guarded for exceptions so onboarding can't break
                    # the input loop.
                    try:
                        from agent.onboarding import (
                            BUSY_INPUT_FLAG,
                            busy_input_hint_cli,
                            is_seen,
                            mark_seen,
                        )
                        if not is_seen(CLI_CONFIG, BUSY_INPUT_FLAG):
                            _hint_mode = "redirect" if redirected else _effective_mode
                            _cprint(f"  {_DIM}{busy_input_hint_cli(_hint_mode)}{_RST}")
                            mark_seen(_hermes_home / "config.yaml", BUSY_INPUT_FLAG)
                            CLI_CONFIG.setdefault("onboarding", {}).setdefault("seen", {})[BUSY_INPUT_FLAG] = True
                    except Exception:
                        pass
                else:
                    self._pending_input.put(payload)
                # History stores real pasted content, not the placeholder, so
                # up-arrow recall restores the actual text.
                self._inline_pastes(event.app.current_buffer)
                event.app.current_buffer.reset(append_to_history=True)

        _bind_prompt_submit_keys(
            kb,
            handle_enter,
            multiline_shortcuts_enabled=_multiline_shortcuts_enabled,
        )
        
        @kb.add('escape', 'enter')
        def handle_alt_enter(event):
            """Alt+Enter inserts a newline for multi-line input.

            Works on mac/Linux/WSL. On Windows Terminal this keystroke is
            intercepted at the terminal layer (toggles fullscreen) and never
            reaches here — Windows users get newline via Ctrl+Enter instead
            (bound below as c-j, since WT delivers Ctrl+Enter as LF).
            """
            event.current_buffer.insert_text('\n')

        if _multiline_shortcuts_enabled or _preserve_ctrl_enter_newline():
            @kb.add('c-j')
            def handle_ctrl_enter_newline(event):
                """Ctrl+J inserts a newline for multi-line input.

                This is enabled by default to match Claude Code / Codex /
                OpenCode behavior. On Windows Terminal and similar environments,
                Ctrl+Enter is delivered as the same c-j key code, so this also
                covers Ctrl+Enter there. Set display.cli_multiline_shortcuts:
                false to restore legacy c-j submit behavior on unusual POSIX
                PTYs where plain Enter arrives as LF.
                """
                event.current_buffer.insert_text('\n')

        # VSCode/Cursor bind Ctrl+G to "Find Next" at the editor level, so
        # the keystroke never reaches the embedded terminal. Alt+G is unbound
        # in those IDEs and arrives here as ('escape', 'g') — register it as
        # a fallback so the editor handoff works inside Cursor/VSCode too.
        _editor_filter = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._sudo_state and not self._secret_state
        )

        @kb.add('c-g', filter=_editor_filter)
        @kb.add('escape', 'g', filter=_editor_filter)
        def handle_open_in_editor(event):
            """Ctrl+G (or Alt+G in VSCode/Cursor) opens the current draft in an external editor."""
            cli_ref._open_external_editor(event.current_buffer)

        # --- Ctrl+S prompt stash -------------------------------------------
        # Park a half-written draft, send something else, then bring the draft
        # back.  Suppressed while a modal prompt owns the composer (sudo /
        # secret / approval / clarify) so Ctrl+S can't stash a password.
        _stash_filter = Condition(
            lambda: not cli_ref._clarify_state
            and not cli_ref._approval_state
            and not cli_ref._sudo_state
            and not cli_ref._secret_state
            and not cli_ref._slash_confirm_state
            and not cli_ref._model_picker_state
        )
        _stash_panel_filter = Condition(
            lambda: cli_ref._prompt_stash.panel_open and bool(len(cli_ref._prompt_stash))
        )

        def _restore_stash_payload(event, payload) -> None:
            """Put a popped (text, images) payload back into the composer."""
            if not payload:
                return
            text, images = payload
            buf = event.app.current_buffer
            buf.text = text
            buf.cursor_position = len(text)
            if images:
                # Restore attachments the draft was carrying.  Extend rather
                # than replace: the user may have attached something new since
                # the stash was taken and silently dropping it would be data
                # loss.
                for img in images:
                    if img not in cli_ref._attached_images:
                        cli_ref._attached_images.append(img)

        @kb.add('c-s', filter=_stash_filter)
        def handle_prompt_stash(event):
            """Ctrl+S: stash the current draft, or restore/browse a stashed one.

            - Composer has content → push it onto the stash and clear the input.
            - Composer empty, one stashed draft → pop it straight back.
            - Composer empty, several stashed → open the browse panel.
            - Browse panel open → close it.

            Pushing onto a stack (rather than a single slot) is what makes
            repeated Ctrl+S safe: a second stash never silently overwrites the
            first, both stay reachable in the panel.
            """
            from hermes_cli.prompt_stash import (
                ACTION_OPEN_PANEL,
                ACTION_RESTORED,
                ACTION_STASHED,
                resolve_ctrl_s,
            )

            buf = event.app.current_buffer
            action, payload = resolve_ctrl_s(
                cli_ref._prompt_stash, buf.text, cli_ref._attached_images
            )

            if action == ACTION_STASHED:
                # reset() (not `text = ""`) so completion state, selection, and
                # the undo stack are cleared along with the text.
                buf.reset()
                cli_ref._attached_images.clear()
            elif action == ACTION_RESTORED:
                _restore_stash_payload(event, payload)
            elif action == ACTION_OPEN_PANEL:
                pass  # resolve_ctrl_s already flipped panel_open

            event.app.invalidate()

        @kb.add('up', filter=_stash_panel_filter, eager=True)
        def handle_stash_panel_up(event):
            cli_ref._prompt_stash.move_cursor(-1)
            event.app.invalidate()

        @kb.add('down', filter=_stash_panel_filter, eager=True)
        def handle_stash_panel_down(event):
            cli_ref._prompt_stash.move_cursor(1)
            event.app.invalidate()

        @kb.add('enter', filter=_stash_panel_filter, eager=True)
        def handle_stash_panel_restore(event):
            """Enter in the browse panel restores the highlighted draft."""
            payload = cli_ref._prompt_stash.restore_at_cursor()
            _restore_stash_payload(event, payload)
            event.app.invalidate()

        @kb.add('d', filter=_stash_panel_filter, eager=True)
        @kb.add('D', filter=_stash_panel_filter, eager=True)
        def handle_stash_panel_delete(event):
            """D in the browse panel discards the highlighted draft."""
            cli_ref._prompt_stash.delete_at_cursor()
            event.app.invalidate()

        @kb.add('escape', filter=_stash_panel_filter, eager=True)
        def handle_stash_panel_close(event):
            cli_ref._prompt_stash.close_panel()
            event.app.invalidate()

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

        # multi-select support: Space toggles the checkbox at the current cursor position
        @kb.add('space', filter=Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext and self._clarify_state.get("multi_select")))
        def clarify_toggle(event):
            if self._clarify_state:
                selected = self._clarify_state["selected"]
                indices = self._clarify_state.get("selected_indices", set())
                if selected in indices:
                    indices.discard(selected)
                else:
                    indices.add(selected)
                event.app.invalidate()

        # Batch clarify: Tab cycles the active question (any-order answering;
        # moving onto an answered question lets the user re-answer it before
        # the batch completes). Registered after the generic tab handler so
        # this filtered binding wins while the batch panel is open.
        @kb.add('tab', filter=Condition(lambda: bool(self._clarify_state) and bool(self._clarify_state.get("questions")) and not self._clarify_freetext), eager=True)
        def clarify_batch_tab(event):
            state = self._clarify_state
            if state and state.get("questions"):
                self._clarify_batch_set_active(
                    state, (state["active"] + 1) % len(state["questions"])
                )
                event.app.invalidate()

        # Shift-Tab walks backwards through the questions.
        @kb.add('s-tab', filter=Condition(lambda: bool(self._clarify_state) and bool(self._clarify_state.get("questions")) and not self._clarify_freetext), eager=True)
        def clarify_batch_backtab(event):
            state = self._clarify_state
            if state and state.get("questions"):
                self._clarify_batch_set_active(
                    state, (state["active"] - 1) % len(state["questions"])
                )
                event.app.invalidate()

        # Number keys for quick clarify selection (1-9, 0 for 10th item)
        def _make_clarify_number_handler(idx):
            def handler(event):
                if self._clarify_state and not self._clarify_freetext:
                    choices = self._clarify_state.get("choices") or []
                    # multi-select support: number keys toggle checkboxes instead of submitting
                    if self._clarify_state.get("multi_select"):
                        if idx < len(choices):
                            indices = self._clarify_state.get("selected_indices", set())
                            if idx in indices:
                                indices.discard(idx)
                            else:
                                indices.add(idx)
                            event.app.invalidate()
                        elif idx == len(choices):
                            # Toggle "Other" in multi-select mode
                            indices = self._clarify_state.get("selected_indices", set())
                            if idx in indices:
                                indices.discard(idx)
                            else:
                                indices.add(idx)
                            event.app.invalidate()
                        return
                    # Original single-select: number keys submit directly
                    # Map index to choice (treating "Other" as the last option)
                    if idx < len(choices):
                        # Batch mode: lock the numbered choice for the active
                        # question instead of resolving the whole prompt.
                        if self._clarify_state.get("questions"):
                            self._clarify_batch_lock(self._clarify_state, choices[idx])
                            event.app.invalidate()
                            return
                        # Select a numbered choice
                        self._clarify_state["response_queue"].put(choices[idx])
                        self._clarify_state = None
                        self._clarify_freetext = False
                        event.app.invalidate()
                    elif idx == len(choices):
                        # Select "Other" option
                        self._clarify_freetext = True
                        event.app.invalidate()
            return handler

        for _num in range(10):
            # 1-9 select items 0-8, 0 selects item 9 (10thitem)
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext))(_make_clarify_number_handler(_idx))

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

        # --- Slash-command confirmation: arrow-key navigation ---
        @kb.add('up', filter=Condition(lambda: bool(self._slash_confirm_state)))
        def slash_confirm_up(event):
            if self._slash_confirm_state:
                self._slash_confirm_state["selected"] = max(0, self._slash_confirm_state.get("selected", 0) - 1)
                event.app.invalidate()

        @kb.add('down', filter=Condition(lambda: bool(self._slash_confirm_state)))
        def slash_confirm_down(event):
            if self._slash_confirm_state:
                max_idx = len(self._slash_confirm_state.get("choices") or []) - 1
                self._slash_confirm_state["selected"] = min(max_idx, self._slash_confirm_state.get("selected", 0) + 1)
                event.app.invalidate()

        # --- /model picker: arrow-key navigation ---
        @kb.add('up', filter=Condition(lambda: bool(self._model_picker_state)))
        def model_picker_up(event):
            if self._model_picker_state:
                self._model_picker_state["selected"] = max(0, self._model_picker_state.get("selected", 0) - 1)
                event.app.invalidate()

        @kb.add('down', filter=Condition(lambda: bool(self._model_picker_state)))
        def model_picker_down(event):
            state = self._model_picker_state
            if not state:
                return
            if state.get("stage") == "provider":
                max_idx = len(state.get("providers") or [])
            else:
                # +1 for "← Back" and Cancel over the filtered visible rows.
                _fp = state.get("_filtered_pairs")
                _visible = len(_fp) if _fp is not None else len(state.get("model_list") or [])
                max_idx = _visible + 1
            state["selected"] = min(max_idx, state.get("selected", 0) + 1)
            event.app.invalidate()

        def _model_picker_typing_active() -> bool:
            # Type-to-filter is only live on the model stage (concrete list).
            st = self._model_picker_state
            return bool(st) and st.get("stage") == "model"

        def _make_model_filter_char_handler(ch: str):
            def handler(event):
                st = self._model_picker_state
                if not st or st.get("stage") != "model":
                    return
                st["filter"] = (st.get("filter", "") or "") + ch
                st["selected"] = 0
                st["_scroll_offset"] = 0
                event.app.invalidate()
            return handler

        # Printable ASCII (space through ~) narrows the model list as you type.
        import string as _string
        for _ch in _string.digits + _string.ascii_letters + "-_.:/ ":
            kb.add(_ch, filter=Condition(_model_picker_typing_active))(
                _make_model_filter_char_handler(_ch)
            )

        @kb.add('backspace', filter=Condition(_model_picker_typing_active))
        def model_picker_filter_backspace(event):
            st = self._model_picker_state
            if not st:
                return
            cur = st.get("filter", "") or ""
            st["filter"] = cur[:-1]
            st["selected"] = 0
            st["_scroll_offset"] = 0
            event.app.invalidate()

        @kb.add('escape', filter=Condition(lambda: bool(self._model_picker_state)), eager=True)
        def model_picker_escape(event):
            """ESC clears an active filter first, else closes the picker."""
            st = self._model_picker_state
            if st and st.get("stage") == "model" and (st.get("filter") or ""):
                st["filter"] = ""
                st["selected"] = 0
                st["_scroll_offset"] = 0
                event.app.invalidate()
                return
            self._close_model_picker()
            event.app.current_buffer.reset()
            event.app.invalidate()

        # --- Ctrl+P command palette keybindings ---
        def _palette_active() -> bool:
            return bool(self._command_palette_state)

        @kb.add('c-p', filter=Condition(
            lambda: not self._command_palette_state
            and not self._model_picker_state and not self._clarify_state
            and not self._approval_state and not self._slash_confirm_state
            and not self._sudo_state and not self._secret_state
        ))
        def open_command_palette(event):
            self._open_command_palette()
            event.app.invalidate()

        @kb.add('up', filter=Condition(_palette_active))
        def command_palette_up(event):
            st = self._command_palette_state
            if st:
                st["selected"] = max(0, st.get("selected", 0) - 1)
                event.app.invalidate()

        @kb.add('down', filter=Condition(_palette_active))
        def command_palette_down(event):
            st = self._command_palette_state
            if st:
                n = st.get("_visible_count", len(self._command_palette_visible_entries()))
                st["selected"] = min(max(0, n - 1), st.get("selected", 0) + 1)
                event.app.invalidate()

        @kb.add('enter', filter=Condition(_palette_active))
        def command_palette_enter(event):
            self._handle_command_palette_selection()
            event.app.invalidate()

        @kb.add('backspace', filter=Condition(_palette_active))
        def command_palette_backspace(event):
            st = self._command_palette_state
            if st:
                st["filter"] = (st.get("filter", "") or "")[:-1]
                st["selected"] = 0
                st["_scroll_offset"] = 0
                event.app.invalidate()

        @kb.add('escape', filter=Condition(_palette_active), eager=True)
        def command_palette_escape(event):
            self._close_command_palette()
            event.app.invalidate()

        def _make_palette_char_handler(ch: str):
            def handler(event):
                st = self._command_palette_state
                if not st:
                    return
                st["filter"] = (st.get("filter", "") or "") + ch
                st["selected"] = 0
                st["_scroll_offset"] = 0
                event.app.invalidate()
            return handler

        import string as _pstring
        for _pch in _pstring.digits + _pstring.ascii_letters + "-_.:/ ":
            kb.add(_pch, filter=Condition(_palette_active))(_make_palette_char_handler(_pch))

        # Number keys for quick approval selection (1-9, 0 for 10th item)
        def _make_approval_number_handler(idx):
            def handler(event):
                if self._approval_state and idx < len(self._approval_state["choices"]):
                    self._approval_state["selected"] = idx
                    self._handle_approval_selection()
                    event.app.invalidate()
            return handler

        for _num in range(10):
            # 1-9 select items 0-8, 0 selects item 9 (10th item)
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=Condition(lambda: bool(self._approval_state)))(_make_approval_number_handler(_idx))

        # Number keys for quick slash-confirm selection (1-9, 0 for 10th item)
        def _make_slash_confirm_number_handler(idx):
            def handler(event):
                if self._slash_confirm_state and idx < len(self._slash_confirm_state.get("choices") or []):
                    choice = self._slash_confirm_state["choices"][idx][0]
                    self._submit_slash_confirm_response(choice)
                    event.app.current_buffer.reset()
                    event.app.invalidate()
            return handler

        for _num in range(10):
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=Condition(lambda: bool(self._slash_confirm_state)))(_make_slash_confirm_number_handler(_idx))

        # --- History navigation: up/down browse history in normal input mode ---
        # The TextArea is multiline, so by default up/down only move the cursor.
        # Buffer.auto_up/auto_down handle both: cursor movement when multi-line,
        # history browsing when on the first/last line (or single-line input).
        _normal_input = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._slash_confirm_state and not self._sudo_state and not self._secret_state and not self._model_picker_state and not self._command_palette_state
        )

        def _recall_without_recollapse(buf, move):
            """Run a history-navigation move, suppressing paste-collapse.

            Recalled history can hold the full text of a paste that was
            collapsed to a placeholder at submit time. Loading it back into the
            buffer looks exactly like a fresh large paste to ``_on_text_changed``
            and would be re-collapsed. Set the skip flag around the move; if the
            move didn't change the text (plain cursor movement), clear the flag
            so a later real paste still collapses.
            """
            before = buf.text
            self._skip_paste_collapse = True
            move()
            if buf.text == before:
                self._skip_paste_collapse = False

        @kb.add('up', filter=_normal_input)
        def history_up(event):
            """Up arrow: browse history when on first line, else move cursor up."""
            buf = event.app.current_buffer
            _recall_without_recollapse(buf, lambda: buf.auto_up(count=event.arg))

        @kb.add('down', filter=_normal_input)
        def history_down(event):
            """Down arrow: browse history when on last line, else move cursor down."""
            buf = event.app.current_buffer
            _recall_without_recollapse(buf, lambda: buf.auto_down(count=event.arg))

        @kb.add('c-l')
        def handle_ctrl_l(event):
            """Ctrl+L: force a clean full-screen repaint.

            Recovers the UI after external terminal buffer drift — tmux /
            cmux tab switches, ``clear`` from a subshell, SSH window
            restores, etc. — that prompt_toolkit can't detect on its own.
            Matches the universal bash/zsh/fish/vim/htop convention.
            """
            self._force_full_redraw()

        @kb.add('c-c')
        def handle_ctrl_c(event):
            """Handle Ctrl+C - cancel interactive prompts, interrupt agent, or exit.
            
            Priority:
            0. Cancel active voice recording
            1. Cancel active sudo/approval/clarify prompt
            2. Interrupt the running agent (first press)
            3. Force exit (second press within 2s, or when idle)
            """
            now = time.time()

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

            # Cancel slash confirmation prompt (foreground UI, not an
            # agent-blocking overlay — cancel and stop here).
            if self._slash_confirm_state:
                self._submit_slash_confirm_response("cancel")
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # Cancel /model picker (foreground UI — cancel and stop here).
            if self._model_picker_state:
                self._close_model_picker()
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # Cancel command palette (foreground UI — cancel and stop here).
            if self._command_palette_state:
                self._close_command_palette()
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # Clear all agent-blocking overlays (approval/clarify/sudo/secret)
            # in one shot.  We do NOT return after clearing — we fall through so
            # that if the agent is also running we fire the interrupt on the same
            # Ctrl+C press.  This fixes the case where a stale/orphaned overlay
            # (left behind by a previous interrupt) consumes the press without
            # ever reaching the agent-interrupt branch, leaving the chat frozen
            # (#14026).
            _overlay_cleared = bool(
                self._sudo_state
                or self._secret_state
                or self._approval_state
                or self._clarify_state
            )
            if _overlay_cleared:
                self._clear_active_overlays_for_interrupt()
                event.app.current_buffer.reset()
                event.app.invalidate()

            # If we only cleared overlays and the agent is NOT running, stop here
            # (don't fall through to the interrupt/exit path).
            if _overlay_cleared and not (self._agent_running and self.agent):
                return

            if self._agent_running and self.agent:
                if now - self._last_ctrl_c_time < 2.0:
                    print("\n⚡ Force exiting...")
                    self._should_exit = True
                    event.app.exit()
                    return
                
                self._last_ctrl_c_time = now
                print("\n⚡ Interrupting agent... (press Ctrl+C again to force exit)")
                request_hard_interrupt(self.agent)
            # If there's text or images, clear them (like bash).
            # If everything is already empty, exit.
            elif event.app.current_buffer.text or self._attached_images:
                event.app.current_buffer.reset()
                self._attached_images.clear()
                event.app.invalidate()
            else:
                self._should_exit = True
                event.app.exit()

        # Ctrl+Shift+C: no binding needed. Terminal emulators (GNOME Terminal,
        # iTerm2, kitty, Windows Terminal, etc.) intercept Ctrl+Shift+C before
        # the keystroke reaches the application's stdin — prompt_toolkit never
        # sees it, and prompt_toolkit's key spec parser doesn't even recognise
        # 'c-S-c' anyway (the Shift modifier is meaningless on control-sequence
        # keys). #19884 added a handler for this; #19895 patched the resulting
        # startup crash with try/except. Both were based on a misreading of how
        # terminal key events propagate. Deleting the dead handler outright.

        @kb.add('c-q')  # Ctrl+Q
        def handle_ctrl_q(event):
            """Alternative interrupt/exit shortcut (Ctrl+Q).

            Behaves like Ctrl+C: cancels active prompts, interrupts the
            running agent, or clears the input buffer. Does not support
            the double-press 'force exit' feature of Ctrl+C.
            """
            # Cancel active voice recording.
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

            # Cancel slash confirmation prompt (foreground UI — cancel and stop).
            if self._slash_confirm_state:
                self._submit_slash_confirm_response("cancel")
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # Cancel /model picker (foreground UI — cancel and stop).
            if self._model_picker_state:
                self._close_model_picker()
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

            # Clear all agent-blocking overlays in one shot, then fall through to
            # the agent-interrupt branch so a single Ctrl+Q both clears a stale
            # overlay and interrupts a still-running agent (#14026).
            _overlay_cleared = bool(
                self._sudo_state
                or self._secret_state
                or self._approval_state
                or self._clarify_state
            )
            if _overlay_cleared:
                self._clear_active_overlays_for_interrupt()
                event.app.current_buffer.reset()
                event.app.invalidate()

            if _overlay_cleared and not (self._agent_running and self.agent):
                return

            if self._agent_running and self.agent:
                print("\n⚡ Interrupting agent...")
                request_hard_interrupt(self.agent)
            elif event.app.current_buffer.text or self._attached_images:
                event.app.current_buffer.reset()
                self._attached_images.clear()
                event.app.invalidate()
            else:
                self._should_exit = True
                event.app.exit()

        @kb.add('c-d')
        def handle_ctrl_d(event):
            """Ctrl+D: delete char under cursor (standard readline behaviour).
            Only exit when the input is empty — same as bash/zsh. Pending
            attached images count as input and block the EOF-exit so the
            user doesn't lose them silently.
            """
            buf = event.app.current_buffer
            if buf.text:
                buf.delete()
            elif self._attached_images:
                # Empty text but pending attachments — no-op, don't exit.
                return
            else:
                self._should_exit = True
                event.app.exit()

        _modal_prompt_active = Condition(
            lambda: bool(self._secret_state or self._sudo_state or self._slash_confirm_state)
        )

        @kb.add('escape', filter=_modal_prompt_active, eager=True)
        def handle_escape_modal(event):
            """ESC cancels active secret/sudo prompts."""
            if self._secret_state:
                self._cancel_secret_capture()
                event.app.current_buffer.reset()
                event.app.invalidate()
                return
            if self._sudo_state:
                self._sudo_state["response_queue"].put("")
                self._sudo_state = None
                event.app.invalidate()
                return
            if self._slash_confirm_state:
                self._submit_slash_confirm_response("cancel")
                event.app.current_buffer.reset()
                event.app.invalidate()
                return

        @kb.add('escape', 'escape', filter=~_modal_prompt_active)
        def handle_double_escape(event):
            """Double ESC: discard the current draft and any attached images.

            Matches Claude Code / Gemini CLI, where double-Esc is the
            clear-the-composer gesture. It works while the agent is
            streaming, which is the gap Ctrl+C leaves: Ctrl+C interrupts a
            running turn and only clears the draft when idle, so mid-stream
            there was no way to discard a half-typed prompt.

            The draft is appended to history first, so Up recalls it — the
            same undo affordance Claude Code provides, and the reason this
            is safe to bind to a key pressed by reflex.

            Single ESC is the prefix for Alt sequences (escape+enter,
            escape+g, escape+v), so prompt_toolkit's escape-timeout keeps
            those distinct from the double press. Modal prompts bind ESC
            eagerly and are excluded here so cancel still wins.
            """
            buf = event.app.current_buffer
            if not (buf.text or cli_ref._attached_images):
                return
            buf.reset(append_to_history=bool(buf.text))
            cli_ref._attached_images.clear()
            event.app.invalidate()

        @kb.add('c-z')
        def handle_ctrl_z(event):
            """Handle Ctrl+Z - suspend process to background (Unix only)."""
            if sys.platform == 'win32':
                _cprint(f"\n{_DIM}Suspend (Ctrl+Z) is not supported on Windows.{_RST}")
                event.app.invalidate()
                return
            import signal as _sig
            from prompt_toolkit.application import run_in_terminal
            from hermes_cli.skin_engine import get_active_skin
            agent_name = get_active_skin().get_branding("agent_name", "Hermes Agent")
            msg = f"\n{agent_name} has been suspended. Run `fg` to bring {agent_name} back."
            def _suspend():
                os.write(1, msg.encode())
                os.kill(0, _sig.SIGTSTP)
            run_in_terminal(_suspend)

        # Voice push-to-talk key: configurable via config.yaml (voice.record_key)
        # Default: Ctrl+B (avoids conflict with Ctrl+R readline reverse-search).
        # Config spellings (ctrl/control/alt/option/opt) are normalized to
        # prompt_toolkit's c-x / a-x format via ``normalize_voice_record_key_for_prompt_toolkit``
        # so the same config value binds identically in the TUI and CLI
        # (Copilot round-9 review on #19835). ``super``/``win``/``windows``
        # configs silently fall back to the default here since prompt_toolkit
        # has no super modifier — log a warning so users notice the
        # TUI/CLI split instead of a silent mismatch (round-11).
        _raw_key: object = "ctrl+b"
        try:
            from hermes_cli.config import load_config
            from hermes_cli.voice import (
                normalize_voice_record_key_for_prompt_toolkit,
                pt_key_to_sequence,
                voice_record_key_from_config,
            )
            _raw_key = voice_record_key_from_config(load_config())
            _voice_key = normalize_voice_record_key_for_prompt_toolkit(_raw_key)
            if (
                isinstance(_raw_key, str)
                and _raw_key.strip().lower().split("+", 1)[0].strip() in {"super", "win", "windows"}
                and _voice_key == "c-b"
            ):
                logger.warning(
                    "voice.record_key %r uses a TUI-only modifier (super/win); "
                    "CLI fell back to Ctrl+B. Use ctrl+<key> or alt+<key> for "
                    "cross-runtime parity.",
                    _raw_key,
                )
        except Exception:
            _voice_key = "c-b"

        # Cache the UI label here — same ``_raw_key`` that drives the
        # prompt_toolkit binding below. Every status / placeholder /
        # recording-hint render reads this cached value so display can
        # never drift from the live keybinding even if the user edits
        # voice.record_key mid-session (Copilot round-13 on #19835).
        self.set_voice_record_key_cache(_raw_key)

        @kb.add(*pt_key_to_sequence(_voice_key))
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
                # Allow disarming continuous mode even when the agent is
                # running or transcribing — otherwise the user is stuck in
                # an auto-restart loop until /voice off (#67545).
                if cli_ref._agent_running or cli_ref._voice_processing:
                    with cli_ref._voice_lock:
                        cli_ref._voice_continuous = False
                    event.app.invalidate()
                    return
                # Guard: don't START recording during interactive prompts
                if cli_ref._clarify_state or cli_ref._sudo_state or cli_ref._approval_state or cli_ref._slash_confirm_state:
                    return

                # Interrupt TTS if playing, so user can start talking.
                # stop_playback() is fast (just terminates a subprocess);
                # the stop event drains the streaming pipeline if one is live.
                if not cli_ref._voice_tts_done.is_set():
                    try:
                        logger.info("TTS CUT: record key handler cutting TTS")
                        from tools.tts_streaming import mark_speech_interrupted
                        mark_speech_interrupted()
                        if cli_ref._voice_tts_stop is not None:
                            cli_ref._voice_tts_stop.set()
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
            triggers this with the pasted text. We only auto-attach a
            clipboard image for image-only/empty paste gestures so text
            pastes and dictation do not accidentally attach stale images.

            Large pastes (5+ lines) are collapsed to a file reference
            placeholder while preserving any existing user text in the
            buffer.
            """
            # Diagnostic canary: measure how long the paste handler blocks
            # the prompt_toolkit event loop. If this exceeds ~500ms we log
            # it so recurring "CLI freezes on paste" reports (issue #16263,
            # macOS Tahoe 26 + iTerm2/Ghostty) arrive with data attached.
            _paste_handler_start = time.perf_counter()
            _paste_raw_size = len(event.data or "")
            pasted_text = event.data or ""
            # Normalise line endings — Windows \r\n and old Mac \r both become \n
            # so the 5-line collapse threshold and display are consistent.
            pasted_text = pasted_text.replace('\r\n', '\n').replace('\r', '\n')
            pasted_text = _strip_leaked_bracketed_paste_wrappers(pasted_text)
            pasted_text, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(pasted_text)
            if _had_mouse_reports:
                self._recover_terminal_input_modes(reason="mouse reports leaked into bracketed paste payload")
            if _should_auto_attach_clipboard_image_on_paste(pasted_text) and self._try_attach_clipboard_image():
                event.app.invalidate()
            if pasted_text:
                # Sanitize surrogate characters (e.g. from Word/Google Docs paste) before writing
                from run_agent import _sanitize_surrogates
                pasted_text = _sanitize_surrogates(pasted_text)
                line_count = pasted_text.count('\n')
                buf = event.current_buffer
                threshold = self.config.get("paste_collapse_threshold", 5)
                char_threshold = self.config.get("paste_collapse_char_threshold", 2000)
                lines_hit = threshold > 0 and line_count >= threshold
                chars_hit = char_threshold > 0 and len(pasted_text) >= char_threshold
                if (lines_hit or chars_hit) and not buf.text.strip().startswith('/'):
                    _paste_counter[0] += 1
                    paste_dir = _hermes_home / "pastes"
                    paste_dir.mkdir(parents=True, exist_ok=True)
                    paste_file = paste_dir / f"paste_{_paste_counter[0]}_{datetime.now().strftime('%H%M%S')}.txt"
                    paste_file.write_text(pasted_text, encoding="utf-8")
                    logger.info("Collapsed paste #%d: %d lines, %d chars -> %s", _paste_counter[0], line_count + 1, len(pasted_text), paste_file)
                    placeholder = f"[Pasted text #{_paste_counter[0]}: {line_count + 1} lines \u2192 {paste_file}]"
                    prefix = ""
                    if buf.cursor_position > 0 and buf.text[buf.cursor_position - 1] != '\n':
                        prefix = "\n"
                    _paste_just_collapsed[0] = True
                    buf.insert_text(prefix + placeholder)
                else:
                    buf.insert_text(pasted_text)
            _paste_handler_elapsed_ms = (time.perf_counter() - _paste_handler_start) * 1000.0
            if _paste_handler_elapsed_ms > 500.0:
                logger.warning(
                    "Slow bracketed-paste handler: %.1fms to process %d bytes "
                    "(%d lines) on %s. If the input becomes unresponsive after "
                    "this, attach this log line to the bug report.",
                    _paste_handler_elapsed_ms,
                    _paste_raw_size,
                    pasted_text.count('\n') + 1 if pasted_text else 0,
                    sys.platform,
                )

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

        # Create the input area with multiline (Alt+Enter), autocomplete, and paste handling
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import ThreadedCompleter


        _completer = SlashCommandCompleter(
            skill_commands_provider=lambda: get_skill_commands(),
            command_filter=cli_ref._command_available,
            skill_bundles_provider=lambda: get_skill_bundles(),
        )
        input_area = TextArea(
            height=Dimension(min=1, max=8, preferred=1),
            prompt=get_prompt,
            style='class:input-area',
            multiline=True,
            wrap_lines=True,
            read_only=Condition(lambda: bool(cli_ref._command_blocks_input)),
            history=FileHistory(str(self._history_file)),
            # complete_while_typing fires the completer on every keystroke. The
            # completer does blocking work — fuzzy @-file indexing shells out to
            # rg/fd (up to a 2s timeout) and path completion hits os.listdir/stat
            # — so running it inline would stall the render loop on each key (very
            # noticeable on WSL2/slow filesystems). ThreadedCompleter moves it off
            # the UI event loop, keeping typing responsive.
            completer=ThreadedCompleter(_completer),
            complete_while_typing=True,
            auto_suggest=SlashCommandAutoSuggest(
                history_suggest=AutoSuggestFromHistory(),
                completer=_completer,
            ),
        )
        # Keep prompt_toolkit on its simple tempfile path. Setting
        # buffer.tempfile = "prompt.md" triggers its complex-tempfile branch,
        # which tries to mkdir() the mkdtemp() directory again and raises
        # EEXIST. The suffix keeps markdown highlighting without that bug.
        input_area.buffer.tempfile_suffix = '.md'

        # Dynamic height: accounts for both explicit newlines AND visual
        # wrapping of long lines so the input area always fits its content.
        def _input_height():
            try:
                from prompt_toolkit.application import get_app

                doc = input_area.buffer.document
                try:
                    terminal_columns = get_app().output.get_size().columns
                except Exception:
                    terminal_columns = shutil.get_terminal_size((80, 24)).columns
                return _estimate_tui_input_height(
                    doc.lines,
                    self._get_tui_prompt_text(),
                    terminal_columns,
                )
            except Exception:
                return 1

        input_area.window.height = _input_height

        # Paste collapsing: detect large pastes and save to temp file
        _paste_counter = [0]
        _prev_text_len = [0]
        _prev_newline_count = [0]
        _paste_just_collapsed = [False]
        self._skip_paste_collapse = False

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
            text = _strip_leaked_bracketed_paste_wrappers(buf.text)
            text, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(text)
            if _had_mouse_reports:
                self._recover_terminal_input_modes(reason="mouse reports leaked into prompt buffer")
            if text != buf.text:
                cursor = min(buf.cursor_position, len(text))
                _paste_just_collapsed[0] = True
                buf.text = text
                buf.cursor_position = cursor
                _prev_text_len[0] = len(text)
                _prev_newline_count[0] = text.count('\n')
                return
            chars_added = len(text) - _prev_text_len[0]
            _prev_text_len[0] = len(text)
            if _paste_just_collapsed[0] or self._skip_paste_collapse:
                _paste_just_collapsed[0] = False
                self._skip_paste_collapse = False
                _prev_newline_count[0] = text.count('\n')
                return
            line_count = text.count('\n')
            newlines_added = line_count - _prev_newline_count[0]
            _prev_newline_count[0] = line_count
            is_paste = chars_added > 1 or newlines_added >= 4
            threshold = self.config.get("paste_collapse_threshold_fallback", 5)
            char_threshold = self.config.get("paste_collapse_char_threshold", 2000)
            lines_hit = threshold > 0 and line_count >= threshold
            chars_hit = char_threshold > 0 and len(text) >= char_threshold
            if (lines_hit or chars_hit) and is_paste and not text.startswith('/'):
                _paste_counter[0] += 1
                paste_dir = _hermes_home / "pastes"
                paste_dir.mkdir(parents=True, exist_ok=True)
                paste_file = paste_dir / f"paste_{_paste_counter[0]}_{datetime.now().strftime('%H%M%S')}.txt"
                paste_file.write_text(text, encoding="utf-8")
                logger.info("Collapsed paste #%d: %d lines, %d chars -> %s (fallback)", _paste_counter[0], line_count + 1, len(text), paste_file)
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
                _label = cli_ref._voice_record_key_label()
                return f"recording... {_label} to stop, Ctrl+C to cancel"
            if cli_ref._voice_processing:
                return "transcribing..."
            if cli_ref._sudo_state:
                return "type password (hidden), Enter to submit · ESC to skip"
            if cli_ref._secret_state:
                return "type secret (hidden), Enter to submit · ESC to skip"
            if cli_ref._approval_state:
                return ""
            if cli_ref._slash_confirm_state:
                return "type 1/2/3, or use ↑/↓ then Enter"
            if cli_ref._clarify_freetext:
                return "type your answer here and press Enter"
            if cli_ref._clarify_state:
                return ""
            if cli_ref._command_running:
                frame = cli_ref._command_spinner_frame()
                status = cli_ref._command_status or "Processing command..."
                return f"{frame} {status}"
            if cli_ref._agent_running:
                return "msg=interrupt · /queue · /bg · /steer · Ctrl+C cancel"
            if cli_ref._voice_mode:
                _label = cli_ref._voice_record_key_label()
                return f"type or {_label} to record"
            # Advertise a parked draft so the stash can never be silently
            # forgotten — the composer itself tells you how to get it back.
            _stash_hint = ""
            try:
                _stash_hint = cli_ref._prompt_stash.placeholder_hint()
            except Exception:
                _stash_hint = ""
            if _stash_hint:
                return _stash_hint
            # Idle + empty composer: show a rotating task-oriented example to
            # nudge the user toward a high-value first action (C-09). Chosen
            # once per session (self._composer_placeholder) so it stays stable
            # while being read, not flickering every render.
            return getattr(cli_ref, "_composer_placeholder", "") or ""

        input_area.control.input_processors.append(_PlaceholderProcessor(_get_placeholder))

        # Hint line above input: shown only for interactive prompts that need
        # extra instructions (sudo countdown, approval navigation, clarify).
        # The agent-running interrupt hint is now an inline placeholder above.
        def get_hint_text():
            if cli_ref._sudo_state:
                remaining = max(0, int(cli_ref._sudo_deadline - time.monotonic()))
                return [
                    ('class:hint', '  password hidden · Enter to skip'),
                    ('class:clarify-countdown', f'  ({remaining}s)'),
                ]

            if cli_ref._secret_state:
                remaining = max(0, int(cli_ref._secret_deadline - time.monotonic()))
                return [
                    ('class:hint', '  secret hidden · Enter to skip'),
                    ('class:clarify-countdown', f'  ({remaining}s)'),
                ]

            if cli_ref._approval_state:
                remaining = max(0, int(cli_ref._approval_deadline - time.monotonic()))
                return [
                    ('class:hint', '  ↑/↓ to select, Enter to confirm'),
                    ('class:clarify-countdown', f'  ({remaining}s)'),
                ]

            if cli_ref._slash_confirm_state:
                remaining = max(0, int(cli_ref._slash_confirm_deadline - time.monotonic()))
                return [
                    ('class:hint', '  type 1/2/3, or ↑/↓ to select, Enter to confirm'),
                    ('class:clarify-countdown', f'  ({remaining}s)'),
                ]

            if cli_ref._clarify_state:
                # None deadline = unlimited wait → hide the countdown entirely.
                if cli_ref._clarify_deadline is None:
                    countdown = ''
                else:
                    remaining = max(0, int(cli_ref._clarify_deadline - time.monotonic()))
                    countdown = f'  ({remaining}s)'
                if cli_ref._clarify_freetext:
                    return [
                        ('class:hint', '  type your answer and press Enter'),
                        ('class:clarify-countdown', countdown),
                    ]
                if cli_ref._clarify_state.get("questions"):
                    return [
                        ('class:hint', '  ↑/↓ to select, Enter to lock, Tab next question'),
                        ('class:clarify-countdown', countdown),
                    ]
                return [
                    ('class:hint', '  ↑/↓ to select, Enter to confirm'),
                    ('class:clarify-countdown', countdown),
                ]

            if cli_ref._command_running:
                frame = cli_ref._command_spinner_frame()
                detail = "input temporarily disabled" if cli_ref._command_blocks_input else "input stays active; Enter queues"
                return [
                    ('class:hint', f'  {frame} command in progress · {detail}'),
                ]

            return []

        def get_hint_height():
            if cli_ref._sudo_state or cli_ref._secret_state or cli_ref._approval_state or cli_ref._slash_confirm_state or cli_ref._clarify_state or cli_ref._command_running:
                return 1
            # Keep a spacer while the agent runs on roomy terminals, but reclaim
            # the row on narrow/mobile screens where every line matters.
            return cli_ref._agent_spacer_height()

        def get_spinner_text():
            spinner_line = cli_ref._render_spinner_text()
            if not spinner_line:
                return []
            return [('class:hint', spinner_line)]

        def get_spinner_height():
            return cli_ref._spinner_widget_height()

        spinner_widget = Window(
            content=FormattedTextControl(get_spinner_text),
            height=get_spinner_height,
            wrap_lines=True,
        )

        # Petdex mascot — right-aligned half-block sprite above the prompt,
        # mirroring the TUI's PetPane. Collapses to height 0 when no pet is
        # enabled, so it's a no-op for everyone else. The _pet_anim_loop thread
        # advances frames + invalidates; align=RIGHT pins it to the edge.
        self._pet_widget = Window(
            content=FormattedTextControl(self._pet_fragments),
            height=self._pet_widget_height,
            align=WindowAlign.RIGHT,
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

        def _get_clarify_batch_display(state):
            """Build styled text for the batch (multi-question) clarify panel.

            A-compact layout mirroring the TUI: a "N questions" header, one
            status line per question (✓ answered → answer / ▸ active /
            · pending), and the active question's numbered choices (+ Other)
            expanded directly beneath its status line.
            """
            questions_list = state.get("questions") or []
            answers = state.get("answers") or {}
            active = state.get("active", 0)
            choices = state.get("choices") or []
            selected = state.get("selected", 0)
            multi_select = state.get("multi_select", False)
            selected_indices = state.get("selected_indices", set()) if multi_select else set()

            title = "Hermes needs your input"
            header = f"{len(questions_list)} questions"

            def _status_rows(width):
                """(style, text) rows for the status list + expanded active question."""
                rows = []
                answer_meta = state.get("answer_meta") or {}
                for idx, entry in enumerate(questions_list):
                    answered = entry["qid"] in answers
                    if answered:
                        marker = "✓"
                    elif idx == active:
                        marker = "▸"
                    else:
                        marker = "·"
                    label = f"{marker} {entry['question']}"
                    row_style = 'class:clarify-selected' if idx == active else 'class:clarify-choice'
                    for wrapped in _wrap_panel_text(label, width, subsequent_indent="  "):
                        rows.append((row_style, wrapped))
                    if answered:
                        # The locked answer on its own line, in its own color,
                        # so the current answer stays readable while walking
                        # the list with Tab/Shift-Tab.
                        for wrapped in _wrap_panel_text(
                            f"    {answers[entry['qid']]}", width, subsequent_indent="    "
                        ):
                            rows.append(('class:clarify-answer', wrapped))
                    if idx != active:
                        continue
                    # Expanded active question: numbered choices + Other.
                    for i, choice in enumerate(choices):
                        num_prefix = str(i + 1) if i < 9 else ('0' if i == 9 else ' ')
                        if multi_select:
                            cb = "[x]" if i in selected_indices else "[ ]"
                            cursor = "❯" if i == selected and not cli_ref._clarify_freetext else " "
                            prefix = f"  {cursor} {cb} {num_prefix}. "
                        else:
                            cursor = "❯" if i == selected and not cli_ref._clarify_freetext else " "
                            prefix = f"  {cursor} {num_prefix}. "
                        style = 'class:clarify-selected' if i == selected and not cli_ref._clarify_freetext else 'class:clarify-choice'
                        for wrapped in _wrap_panel_text(f"{prefix}{choice}", width, subsequent_indent="      "):
                            rows.append((style, wrapped))
                    if choices:
                        other_idx = len(choices)
                        other_num = other_idx + 1
                        other_num_prefix = str(other_num) if other_num < 10 else ('0' if other_num == 10 else ' ')
                        if multi_select:
                            cb = "[x]" if other_idx in selected_indices else "[ ]"
                            mid = f"{cb} {other_num_prefix}"
                        else:
                            mid = other_num_prefix
                        # An earlier typed answer stays visible next to Other;
                        # Enter on it edits (the composer is prefilled).
                        meta = answer_meta.get(entry["qid"]) or {}
                        other_text = meta.get("other_text") or ""
                        other_suffix = f"Other: {other_text}" if other_text else None
                        if cli_ref._clarify_freetext:
                            other_label = f"  ❯ {mid}. " + (other_suffix or "Other (type below)")
                            other_style = 'class:clarify-active-other'
                        elif selected == other_idx:
                            other_label = f"  ❯ {mid}. " + (other_suffix or "Other (type your answer)")
                            other_style = 'class:clarify-selected'
                        else:
                            other_label = f"    {mid}. " + (other_suffix or "Other (type your answer)")
                            other_style = 'class:clarify-choice'
                        for wrapped in _wrap_panel_text(other_label, width, subsequent_indent="      "):
                            rows.append((other_style, wrapped))
                    elif cli_ref._clarify_freetext:
                        for wrapped in _wrap_panel_text(
                            "  Type your answer in the prompt below, then press Enter.", width
                        ):
                            rows.append(('class:clarify-active-other', wrapped))
                return rows

            preview_rows = _status_rows(60)
            box_width = _panel_box_width(title, [header] + [text for _, text in preview_rows])
            inner_text_width = max(8, box_width - 2)
            rows = _status_rows(inner_text_width)

            lines = []
            lines.append(('class:clarify-border', '╭─ '))
            lines.append(('class:clarify-title', title))
            lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
            _append_panel_line(lines, 'class:clarify-border', 'class:clarify-question', header, box_width)
            for style, text in rows:
                _append_panel_line(lines, 'class:clarify-border', style, text, box_width)
            lines.append(('class:clarify-border', '╰' + ('─' * box_width) + '╯\n'))
            return lines

        def _get_clarify_display():
            """Build styled text for the clarify question/choices panel.

            Layout priority: choices + Other option must always render even if
            the question is very long. The question is budgeted to leave enough
            rows for the choices and trailing chrome; anything over the budget
            is truncated with a marker.
            """
            state = cli_ref._clarify_state
            if not state:
                return []
            if state.get("questions"):
                return _get_clarify_batch_display(state)

            question = state["question"]
            choices = state.get("choices") or []
            selected = state.get("selected", 0)
            # multi-select support
            multi_select = state.get("multi_select", False)
            selected_indices = state.get("selected_indices", set()) if multi_select else set()
            preview_lines = _wrap_panel_text(question, 60)
            for i, choice in enumerate(choices):
                # Show number prefix for quick selection (1-9 for items 1-9, 0 for 10th item)
                if i < 9:
                    num_prefix = str(i + 1)
                elif i == 9:
                    num_prefix = '0'
                else:
                    num_prefix = ' '
                if multi_select:
                    cb = "[x]" if i in selected_indices else "[ ]"
                    if i == selected and not cli_ref._clarify_freetext:
                        prefix = f"❯ {cb} {num_prefix}. "
                    else:
                        prefix = f"  {cb} {num_prefix}. "
                elif i == selected and not cli_ref._clarify_freetext:
                    prefix = f"❯ {num_prefix}. "
                else:
                    prefix = f"  {num_prefix}. "
                preview_lines.extend(_wrap_panel_text(f"{prefix}{choice}", 60, subsequent_indent="    "))
            # "Other" option in preview
            other_num = len(choices) + 1
            if other_num < 10:
                other_num_prefix = str(other_num)
            elif other_num == 10:
                other_num_prefix = '0'
            else:
                other_num_prefix = ' '
            other_idx_val = len(choices)
            if multi_select:
                cb = "[x]" if other_idx_val in selected_indices else "[ ]"
                other_label = (
                    f"❯ {cb} {other_num_prefix}. Other (type below)" if cli_ref._clarify_freetext
                    else f"❯ {cb} {other_num_prefix}. Other (type your answer)" if selected == other_idx_val
                    else f"  {cb} {other_num_prefix}. Other (type your answer)"
                )
            else:
                other_label = (
                    f"❯ {other_num_prefix}. Other (type below)" if cli_ref._clarify_freetext
                    else f"❯ {other_num_prefix}. Other (type your answer)" if selected == len(choices)
                    else f"  {other_num_prefix}. Other (type your answer)"
                )
            preview_lines.extend(_wrap_panel_text(other_label, 60, subsequent_indent="    "))
            box_width = _panel_box_width("Hermes needs your input", preview_lines)
            inner_text_width = max(8, box_width - 2)

            # Pre-wrap choices + Other option — these are mandatory.
            choice_wrapped: list[tuple[int, str]] = []
            if choices:
                for i, choice in enumerate(choices):
                    # Show number prefix for quick selection (1-9 for items 1-9, 0 for 10th item)
                    if i < 9:
                        num_prefix = str(i + 1)
                    elif i == 9:
                        num_prefix = '0'
                    else:
                        num_prefix = ' '
                    # multi-select support: add checkbox after cursor indicator
                    if multi_select:
                        cb = "[x]" if i in selected_indices else "[ ]"
                        if i == selected and not cli_ref._clarify_freetext:
                            prefix = f'❯ {cb} {num_prefix}. '
                        else:
                            prefix = f'  {cb} {num_prefix}. '
                    elif i == selected and not cli_ref._clarify_freetext:
                        prefix = f'❯ {num_prefix}. '
                    else:
                        prefix = f'  {num_prefix}. '
                    for wrapped in _wrap_panel_text(f"{prefix}{choice}", inner_text_width, subsequent_indent="    "):
                        choice_wrapped.append((i, wrapped))
                # Trailing Other row(s)
                other_idx = len(choices)
                other_num = other_idx + 1
                if other_num < 10:
                    other_num_prefix = str(other_num)
                elif other_num == 10:
                    other_num_prefix = '0'
                else:
                    other_num_prefix = ' '
                # multi-select support: add checkbox to Other option
                if multi_select:
                    cb = "[x]" if other_idx in selected_indices else "[ ]"
                    if selected == other_idx and not cli_ref._clarify_freetext:
                        other_label_mand = f'❯ {cb} {other_num_prefix}. Other (type your answer)'
                    elif cli_ref._clarify_freetext:
                        other_label_mand = f'❯ {cb} {other_num_prefix}. Other (type below)'
                    else:
                        other_label_mand = f'  {cb} {other_num_prefix}. Other (type your answer)'
                else:
                    if selected == other_idx and not cli_ref._clarify_freetext:
                        other_label_mand = f'❯ {other_num_prefix}. Other (type your answer)'
                    elif cli_ref._clarify_freetext:
                        other_label_mand = f'❯ {other_num_prefix}. Other (type below)'
                    else:
                        other_label_mand = f'  {other_num_prefix}. Other (type your answer)'
                other_wrapped = _wrap_panel_text(other_label_mand, inner_text_width, subsequent_indent="    ")
            elif cli_ref._clarify_freetext:
                # Freetext-only mode: the guidance line takes the place of choices.
                other_wrapped = _wrap_panel_text(
                    "Type your answer in the prompt below, then press Enter.",
                    inner_text_width,
                )
            else:
                other_wrapped = []

            # Budget the question so mandatory rows always render.
            # Chrome layouts:
            #   full : top border + blank_after_title + blank_after_question
            #          + blank_before_bottom + bottom border = 5 rows
            #   tight: top border + bottom border = 2 rows (drop all blanks)
            #
            # reserved_below matches the approval-panel budget (~6 rows for
            # spinner/tool-progress + status + input + separators + prompt).
            term_rows = shutil.get_terminal_size((100, 24)).lines
            chrome_full = 5
            chrome_tight = 2
            reserved_below = 6

            available = max(0, term_rows - reserved_below)
            # The compact decision must reserve room for at least one question
            # row on top of the choices, otherwise full chrome (3 blank
            # separators) gets kept when there is no room for it and the panel
            # overflows the viewport — HSplit then clips the panel's tail,
            # silently dropping the choices (the reported bug).
            mandatory_full = chrome_full + 1 + len(choice_wrapped) + len(other_wrapped)

            use_compact_chrome = mandatory_full > available
            chrome_rows = chrome_tight if use_compact_chrome else chrome_full

            max_question_rows = max(1, available - chrome_rows - len(choice_wrapped) - len(other_wrapped))
            max_question_rows = min(max_question_rows, 12)  # soft cap on huge terminals

            # When the choices alone (plus compact chrome) already exceed the
            # viewport, drop the question entirely — the choices are the only
            # thing the user must see to make a selection. Without this the
            # question would still claim its 1-row floor above and push the
            # tail of the choices off-screen (HSplit clips the overflow).
            choices_overflow = chrome_rows + len(choice_wrapped) + len(other_wrapped) >= available
            if choices_overflow:
                max_question_rows = 0

            question_wrapped = _wrap_panel_text(question, inner_text_width)
            if max_question_rows <= 0:
                question_wrapped = []
            elif len(question_wrapped) > max_question_rows:
                # The truncation marker is itself a row, so it must count
                # against the budget. With a 1-row budget there is no room for
                # both a question line and the marker — show the marker alone
                # so the rendered question never exceeds max_question_rows.
                keep = max(0, max_question_rows - 1)
                question_wrapped = question_wrapped[:keep] + ["… (question truncated)"]

            lines = []
            # Box top border
            lines.append(('class:clarify-border', '╭─ '))
            lines.append(('class:clarify-title', 'Hermes needs your input'))
            lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len("Hermes needs your input") - 3)) + '╮\n'))
            if not use_compact_chrome:
                _append_blank_panel_line(lines, 'class:clarify-border', box_width)

            # Question text (bounded)
            for wrapped in question_wrapped:
                _append_panel_line(lines, 'class:clarify-border', 'class:clarify-question', wrapped, box_width)
            if not use_compact_chrome:
                _append_blank_panel_line(lines, 'class:clarify-border', box_width)

            if cli_ref._clarify_freetext and not choices:
                for wrapped in other_wrapped:
                    _append_panel_line(lines, 'class:clarify-border', 'class:clarify-choice', wrapped, box_width)
                if not use_compact_chrome:
                    _append_blank_panel_line(lines, 'class:clarify-border', box_width)

            if choices:
                # Multiple-choice mode: show selectable options
                for i, wrapped in choice_wrapped:
                    style = 'class:clarify-selected' if i == selected and not cli_ref._clarify_freetext else 'class:clarify-choice'
                    _append_panel_line(lines, 'class:clarify-border', style, wrapped, box_width)

                # "Other" option (trailing row(s), only shown when choices exist)
                other_idx = len(choices)
                # Calculate number prefix for "Other" option
                other_num = other_idx + 1
                if other_num < 10:
                    other_num_prefix = str(other_num)
                elif other_num == 10:
                    other_num_prefix = '0'
                else:
                    other_num_prefix = ' '
                
                if selected == other_idx and not cli_ref._clarify_freetext:
                    other_style = 'class:clarify-selected'
                elif cli_ref._clarify_freetext:
                    other_style = 'class:clarify-active-other'
                else:
                    other_style = 'class:clarify-choice'
                for wrapped in other_wrapped:
                    _append_panel_line(lines, 'class:clarify-border', other_style, wrapped, box_width)

            if not use_compact_chrome:
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
            body = 'Enter secret below (hidden), ESC or Ctrl+C to skip'
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

        def _get_slash_confirm_display():
            return cli_ref._get_slash_confirm_display_fragments()

        slash_confirm_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_slash_confirm_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._slash_confirm_state is not None),
        )

        # --- /model picker: display widget ---
        def _get_model_picker_display():
            state = cli_ref._model_picker_state
            if not state:
                return []
            stage = state.get("stage", "provider")
            if stage == "provider":
                title = "⚙ Model Picker — Select Provider"
                choices = []
                _providers = state.get("providers")
                for p in _providers if isinstance(_providers, list) else []:
                    count = p.get("total_models", len(p.get("models", [])))
                    label = f"{p['name']} ({count} model{'s' if count != 1 else ''})"
                    if p.get("is_current"):
                        label += "  ← current"
                    choices.append(label)
                choices.append("Cancel")
                hint = f"Current: {state.get('current_model', 'unknown')} on {state.get('current_provider', 'unknown')}"
            else:
                provider_data = state.get("provider_data") or {}
                model_list = state.get("model_list") or []
                title = f"⚙ Model Picker — {provider_data.get('name', provider_data.get('slug', 'Provider'))}"
                # Fuzzy filter: narrow the concrete model list by the typed
                # query. Selection still resolves to a real entry (see the
                # filtered_pairs index mapping in the selection handler), so
                # this never introduces an ambiguous model resolution.
                _query = state.get("filter", "") or ""
                filtered_pairs = cli_ref._filter_model_picker_entries(model_list, _query)
                state["_filtered_pairs"] = filtered_pairs
                model_labels = [e for (_i, e) in filtered_pairs]
                choices = list(model_labels) + ["← Back", "Cancel"]
                if _query:
                    hint = (
                        f"Filter: {_query}▏  ({len(model_labels)}/{len(model_list)} match "
                        "— type to narrow, Backspace to clear)"
                    )
                elif model_list:
                    hint = f"Select a model ({len(model_list)} available) — type to filter"
                else:
                    hint = "No models listed for this provider. Use Back or Cancel."

            box_width = _panel_box_width(title, [hint] + choices, min_width=46, max_width=84)
            inner_text_width = max(8, box_width - 6)
            selected = state.get("selected", 0)

            # Scrolling viewport: the panel renders into a Window with no max
            # height, so without limiting visible items the bottom border and
            # any items past the available terminal rows get clipped on long
            # provider catalogs (e.g. Ollama Cloud's 36+ models).
            try:
                from prompt_toolkit.application import get_app
                term_rows = get_app().output.get_size().rows
            except Exception:
                term_rows = shutil.get_terminal_size((100, 24)).lines
            scroll_offset, visible = HermesCLI._compute_model_picker_viewport(
                selected, state.get("_scroll_offset", 0), len(choices), term_rows,
            )
            state["_scroll_offset"] = scroll_offset

            lines = []
            lines.append(('class:clarify-border', '╭─ '))
            lines.append(('class:clarify-title', title))
            lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)
            _append_panel_line(lines, 'class:clarify-border', 'class:clarify-hint', hint, box_width)
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)
            for idx in range(scroll_offset, scroll_offset + visible):
                choice = choices[idx]
                style = 'class:clarify-selected' if idx == selected else 'class:clarify-choice'
                prefix = '❯ ' if idx == selected else '  '
                for wrapped in _wrap_panel_text(prefix + choice, inner_text_width, subsequent_indent='  '):
                    _append_panel_line(lines, 'class:clarify-border', style, wrapped, box_width)
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)
            lines.append(('class:clarify-border', '╰' + ('─' * box_width) + '╯\n'))
            return lines

        model_picker_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_model_picker_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._model_picker_state is not None),
        )

        # --- Ctrl+P command palette: display widget ---
        def _get_command_palette_display():
            state = cli_ref._command_palette_state
            if not state:
                return []
            rows = cli_ref._command_palette_visible_entries()
            state["_visible_count"] = len(rows)
            _query = state.get("filter", "") or ""
            total = len(state.get("entries") or [])
            title = "⚙ Command Palette"
            if _query:
                hint = f"Filter: {_query}▏  ({len(rows)}/{total} match — Enter inserts, Esc cancels)"
            else:
                hint = f"Type to filter {total} commands — ↑/↓ then Enter inserts, Esc cancels"

            labels = [f"{c}  —  {d}" if d else c for (c, _cat, d) in rows]
            if not labels:
                labels = ["(no matching commands)"]
            box_width = _panel_box_width(title, [hint] + labels, min_width=50, max_width=90)
            inner_text_width = max(8, box_width - 6)
            selected = state.get("selected", 0)
            try:
                from prompt_toolkit.application import get_app
                term_rows = get_app().output.get_size().rows
            except Exception:
                term_rows = shutil.get_terminal_size((100, 24)).lines
            scroll_offset, visible = HermesCLI._compute_model_picker_viewport(
                selected, state.get("_scroll_offset", 0), len(labels), term_rows,
            )
            state["_scroll_offset"] = scroll_offset

            lines = []
            lines.append(('class:clarify-border', '╭─ '))
            lines.append(('class:clarify-title', title))
            lines.append(('class:clarify-border', ' ' + ('─' * max(0, box_width - len(title) - 3)) + '╮\n'))
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)
            _append_panel_line(lines, 'class:clarify-border', 'class:clarify-hint', hint, box_width)
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)
            for idx in range(scroll_offset, min(scroll_offset + visible, len(labels))):
                label = labels[idx]
                style = 'class:clarify-selected' if idx == selected else 'class:clarify-choice'
                prefix = '❯ ' if idx == selected else '  '
                for wrapped in _wrap_panel_text(prefix + label, inner_text_width, subsequent_indent='    '):
                    _append_panel_line(lines, 'class:clarify-border', style, wrapped, box_width)
            _append_blank_panel_line(lines, 'class:clarify-border', box_width)
            lines.append(('class:clarify-border', '╰' + ('─' * box_width) + '╯\n'))
            return lines

        command_palette_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_command_palette_display),
                wrap_lines=True,
            ),
            filter=Condition(lambda: cli_ref._command_palette_state is not None),
        )

        # Horizontal rules above and below the input.
        # On narrow/mobile terminals we keep the top separator for structure but
        # hide the bottom one to recover a full row for conversation content.
        input_rule_top = Window(
            char='─',
            height=lambda: cli_ref._tui_input_rule_height("top"),
            style='class:input-rule',
        )
        input_rule_bot = Window(
            char='─',
            height=lambda: cli_ref._tui_input_rule_height("bottom"),
            style='class:input-rule',
        )

        # Image attachment indicator — shows badges like [📎 Image #1] above input
        cli_ref = self

        def _get_image_bar():
            if not cli_ref._attached_images:
                return []
            badges = _format_image_attachment_badges(
                cli_ref._attached_images,
                cli_ref._image_counter,
            )
            return [("class:image-badge", f" {badges} ")]

        image_bar = Window(
            content=FormattedTextControl(_get_image_bar),
            height=Condition(lambda: bool(cli_ref._attached_images)),
        )

        # Persistent voice mode status bar (visible only when voice mode is on)
        def _get_voice_status():
            return cli_ref._get_voice_status_fragments()

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
            filter=Condition(
                lambda: cli_ref._status_bar_visible
                and not getattr(cli_ref, "_status_bar_suppressed_after_resize", False)
            ),
        )

        # Stash browse panel — appears just above the status bar when the user
        # presses Ctrl+S on an empty composer with 2+ stashed drafts.
        def _get_stash_panel_display():
            try:
                _stash = cli_ref._prompt_stash
                return cli_ref._render_stash_panel(
                    _stash.panel_rows(),
                    _stash.panel_cursor,
                    cli_ref._get_tui_terminal_width(),
                )
            except Exception:
                return []

        self._stash_panel_widget = ConditionalContainer(
            Window(
                FormattedTextControl(_get_stash_panel_display),
                wrap_lines=False,
            ),
            filter=Condition(
                lambda: cli_ref._prompt_stash.panel_open
                and bool(len(cli_ref._prompt_stash))
            ),
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
                    slash_confirm_widget=slash_confirm_widget,
                    clarify_widget=clarify_widget,
                    model_picker_widget=model_picker_widget,
                    command_palette_widget=command_palette_widget,
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
            # Input area / prompt: empty style strings inherit the
            # terminal's default foreground/background, so the typed
            # text is readable in both light and dark Terminal.app
            # color schemes.  (Hardcoding a near-white #FFF8DC made
            # input invisible on light backgrounds.)
            'input-area': '',
            'placeholder': '#888888 italic',
            'prompt': '',
            'prompt-working': '#888888 italic',
            'hint': '#888888 italic',
            'status-bar': 'bg:#1a1a2e #C0C0C0',
            'status-bar-strong': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-dim': 'bg:#1a1a2e #8B8682',
            'status-bar-good': 'bg:#1a1a2e #8FBC8F bold',
            'status-bar-warn': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-bad': 'bg:#1a1a2e #FF8C00 bold',
            'status-bar-critical': 'bg:#1a1a2e #FF6B6B bold',
            'status-bar-yolo': 'bg:#1a1a2e #FF4444 bold',
            'status-bar-session-title': 'bg:#FFD700 #1a1a2e bold',
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
            'clarify-answer': '#98FB98',
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

        # Select CPR-disabled output when _terminal_may_leak_cpr() says so
        # (POSIX local + SSH; Windows keeps PT default — see helper docs).
        # None falls back to prompt_toolkit's default output; input scrubbing
        # in _strip_leaked_terminal_responses still guards residual leaks.
        _cpr_disabled_output = _select_classic_cli_pt_output(sys.stdout)

        # Kitty placeholders encode the image id in exact foreground RGB, so the whole app
        # runs 24-bit there. ColorDepth is imported lazily for tests that stub prompt_toolkit.
        extra_kw = {}
        if pet_render.supports_kitty_placeholders():
            from prompt_toolkit.output import ColorDepth

            extra_kw["color_depth"] = ColorDepth.DEPTH_24_BIT
        if _cpr_disabled_output is not None:
            extra_kw["output"] = _cpr_disabled_output
        if _STEADY_CURSOR is not None:
            extra_kw["cursor"] = _STEADY_CURSOR
        if EditingMode is not None:
            # Vi editing mode when display.vim_mode is on.
            # EMACS is prompt_toolkit's own default, so non-opted-in behaviour is unchanged.
            extra_kw["editing_mode"] = EditingMode.VI if self._vim_mode else EditingMode.EMACS
        return Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
            # 0 (default) avoids fighting terminal auto-scroll in non-fullscreen mode.
            refresh_interval=float(CLI_CONFIG.get("display", {}).get("cli_refresh_interval", 0)),
            # Erase the bottom chrome on exit instead of freezing a copy into scrollback.
            # Without this, prompt_toolkit's render_as_done teardown repaints the chrome one last time and
            # leaves it stranded above the exit summary — so a dead status bar + empty prompt sit between
            # the conversation transcript and the "Resume this session" block, and stack with the next
            # session's UI on resume (#38252). The actual conversation transcript is printed through
            # patch_stdout into normal scrollback and is unaffected; only the managed chrome is erased.
            # Applies to every exit path (/exit, /quit, EOF, Ctrl+C).
            erase_when_done=True,
            **extra_kw,
        )

    def _tui_install_signal_handlers(self):
        """SIGTERM/SIGHUP -> graceful shutdown; Windows absorbs SIGINT (see body)."""
        try:
            import signal as _signal
            _signal.signal(_signal.SIGTERM, self._tui_signal_handler)
            if hasattr(_signal, 'SIGHUP'):
                _signal.signal(_signal.SIGHUP, self._tui_signal_handler)

            # Windows: absorb SIGINT. Win32 delivers spurious CTRL_C_EVENT when children spawn
            # from background threads, which would unwind app.run() mid-turn. Real Ctrl+C is
            # bound by prompt_toolkit. Never call agent.interrupt() here (fake user message).
            if sys.platform == "win32":
                _signal.signal(_signal.SIGINT, lambda signum, frame: None)
        except Exception:
            pass  # restricted environments

    def _tui_stdin_usable(self) -> bool:
        """Validate fd 0 before prompt_toolkit starts; on macOS fall back to a select() loop when kqueue can't watch it (uv-managed Python)."""
        try:
            os.fstat(0)
        except OSError:
            print(
                "Error: stdin (fd 0) is not available.\n"
                "This can happen with certain Python installations (e.g. uv-managed cPython on macOS).\n"
                "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
            )
            return False
        if sys.platform == "darwin":
            import selectors as _selectors
            try:
                if hasattr(_selectors, "KqueueSelector"):
                    _kq = _selectors.KqueueSelector()
                    try:
                        _kq.register(0, _selectors.EVENT_READ)
                        _kq.unregister(0)
                    finally:
                        _kq.close()
            except (OSError, ValueError, KeyError):
                import asyncio as _aio_probe

                class _SelectEventLoopPolicy(_aio_probe.DefaultEventLoopPolicy):
                    def new_event_loop(self):
                        return _aio_probe.SelectorEventLoop(_selectors.SelectSelector())

                _aio_probe.set_event_loop_policy(_SelectEventLoopPolicy())
        return True

    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        if not self._claim_active_session("cli"):
            return

        self._tui_print_startup()
        self._tui_init_run_state()
        kb = self._tui_build_key_bindings()
        layout, style = self._tui_build_layout(kb)

        app = self._tui_build_application(layout, kb, style)
        _disable_prompt_toolkit_cpr_warning(app)
        app.after_render += self._pet_flush_kitty_frame
        self._app = app

        # Ghost status-bar lines on resize: pt's renderer scrolls the terminal after each
        # paint, pushing chrome into scrollback where a column-shrink reflows it into
        # duplicates. Wrapping _output_screen_diff keeps its reserve-space branch from firing.
        try:
            # Background: prompt_toolkit's renderer (renderer.py L232-242) explicitly moves the cursor to
            # the bottom of the canvas after painting "to make sure the terminal scrolls up, even when the
            # lower lines of the canvas just contain whitespace". In non-fullscreen mode this scrolls chrome
            # content (status bar, input rules) into terminal scrollback on every render. When the terminal
            # column-shrinks, the emulator reflows the previously rendered full-width rows into multiple
            # narrower rows that get pushed up — leaving ghost duplicates AND polluting scrollback. Same
            # issue as pt #29 (open since 2014), #1675, #1933. Surgical fix: wrap _output_screen_diff so
            # that when its internal `if current_height > previous_screen.height` branch fires (the one that
            # does the bottom-cursor-move), we make it fall through by inflating previous_screen.height
            # first.
            import prompt_toolkit.renderer as _pt_renderer
            from prompt_toolkit.renderer import _output_screen_diff as _orig_osd

            if not getattr(_pt_renderer, "_hermes_osd_patched", False):
                _pt_renderer._output_screen_diff = functools.partial(
                    _hermes_call_output_screen_diff, _orig_osd
                )
                _pt_renderer._hermes_osd_patched = True
        except Exception:
            pass

        _apply_bracketed_paste_timeout_patch()

        self._install_resize_recovery(app)

        threading.Thread(target=self._tui_spinner_loop, daemon=True).start()
        threading.Thread(target=self._tui_process_loop, daemon=True).start()
        # Wake word listener off-thread so a first-run engine install never blocks the prompt.
        threading.Thread(target=self._tui_wake_startup, daemon=True, name="wake-startup").start()

        atexit.register(_run_cleanup)
        self._tui_install_signal_handlers()

        if not self._tui_stdin_usable():
            _run_cleanup()
            self._print_exit_summary()
            return

        try:
            with patch_stdout():
                try:
                    # run_in_terminal() may return either: • a coroutine / Future (prompt_toolkit ≥ 3.0) —
                    # must be scheduled via ensure_future so the coroutine is actually awaited; calling it
                    # bare would leave it unawaited and silently drop the output (fixes #23185 Bug A). •
                    # None (some mocks / older PT builds) — just call the inner function directly since PT
                    # already executed it synchronously. Do NOT fall back to a bare _pt_print when
                    # ensure_future raises, because run_in_terminal already invoked the lambda in that case
                    # (the mock path), which would double-print the line.
                    import asyncio as _aio
                    _aio.get_running_loop().set_exception_handler(self._tui_suppress_closed_loop_errors)
                except Exception:
                    pass  # no running loop -- nothing to patch
                # Record that the app enables focus reporting + mouse tracking so _run_cleanup
                # resets them; extended key modes are popped by the same reset.
                # When multiline shortcuts are on, also ask supported terminals (e.g. iTerm2) to report
                # modified keys distinctly (kitty protocol + modifyOtherKeys); the cleanup reset pops both
                # modes. See #36823.
                _mark_tui_input_modes_active()
                if self._tui_multiline_shortcuts:
                    _enable_extended_enter_keys(app.output)
                self._pet_start_anim()
                app.run()
        except (EOFError, KeyboardInterrupt, BrokenPipeError):
            pass
        except (KeyError, OSError) as _stdin_err:
            # Selector registration failures from broken stdin and I/O errors from a
            # broken stdout during interrupt (EIO is suppressed).
            _errno = getattr(_stdin_err, "errno", None) if isinstance(_stdin_err, OSError) else None
            _msg = str(_stdin_err)
            if _errno == errno.EIO:
                pass
            elif _errno in {errno.EINVAL, errno.EBADF} or any(
                s in _msg for s in ("is not registered", "Bad file descriptor", "Invalid argument")
            ):
                print(
                    f"\nError: stdin is not usable ({_stdin_err}).\n"
                    "This can happen with certain Python installations (e.g. uv-managed cPython on macOS)\n"
                    "where kqueue cannot register fd 0.\n"
                    "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
                )
            else:
                raise
        finally:
            self._tui_shutdown()

        # /update relaunch happens here, after prompt_toolkit restored terminal modes, on the
        # main thread (the process_loop thread would skip cleanup / only exit itself on Windows).
        if self._pending_relaunch:
            from hermes_cli.relaunch import relaunch
            relaunch(self._pending_relaunch, preserve_inherited=False)

    def _tui_shutdown(self):
        """Teardown after the app exits: interrupt agent, stop voice/pet, persist + close session, cleanup, exit summary."""
        self._should_exit = True
        self._pet_stop_anim()
        # Without this line the terminal sits silent through the whole cleanup window.
        with suppress(Exception):
            print(f"{_DIM}Shutting down… (finalizing session){_RST}", flush=True)
        if self.agent and self._agent_running:
            with suppress(Exception):
                request_hard_interrupt(self.agent)
        if self._voice_recorder:
            with suppress(Exception):
                self._voice_recorder.shutdown()
            self._voice_recorder = None
        with suppress(Exception):
            from tools.voice_mode import cleanup_temp_recordings
            cleanup_temp_recordings()
        from agent.vault_backends.unlock import (lock as _vault_lock, set_code_prompt_callback,
                                                 set_save_login_prompt_callback, set_unlock_prompt_callback)
        for _unset in (set_sudo_password_callback, set_approval_callback, set_secret_capture_callback,
                       set_unlock_prompt_callback, set_save_login_prompt_callback, set_code_prompt_callback):
            _unset(None)
        _vault_lock()  # session tokens for external password managers die with the session
        # On SIGHUP/SIGTERM the agent thread may be reaped before its own persistence runs.
        self._persist_active_session_before_close()

        if self._session_db and self.agent:
            try:
                self._session_db.end_session(self.agent.session_id, "cli_close")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Could not close session in DB: %s", e)
            if not self._delete_session_on_exit:
                # Drop the empty row of a start-and-quit session so /resume stays clean.
                try:
                    self._discard_session_if_empty(self.agent.session_id)
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not prune empty session: %s", e)
            else:
                # /exit --delete: remove transcripts + SQLite history.
                try:
                    _sid = self.agent.session_id
                    if self._session_db.delete_session(_sid, sessions_dir=get_hermes_home() / "sessions"):
                        _cprint(f"  {_DIM}✓ Session {_escape(_sid)} deleted{_RST}")
                    else:
                        _cprint(f"  {_DIM}✗ Session {_escape(_sid)} not found for deletion{_RST}")
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not delete session on exit: %s", e)
        # run_conversation() fires on_session_end on normal completion; only fire here mid-turn.
        if self.agent and self._agent_running:
            _invoke_interrupted_session_end(self.agent, self.agent.session_id, "shutdown")
        _run_cleanup()
        self._print_exit_summary()
        self._release_active_session()


def _int_or(value, default: int) -> int:
    """``int(value)``, or ``default`` when it does not parse."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _interrupt_agent_for_signal(agent, signum) -> None:
    """Hard-interrupt ``agent`` for a shutdown signal, then sleep ``HERMES_SIGTERM_GRACE`` (1.5 s).

    The grace lets the agent thread kill the tool's setsid subprocess group before the
    main thread unwinds (else an orphan child). Never raises.
    """
    try:
        if agent is not None:
            request_hard_interrupt(agent, f"received signal {signum}")
            _grace = _float_env("HERMES_SIGTERM_GRACE", 1.5)
            if _grace > 0:
                time.sleep(_grace)
    except Exception:
        pass  # never block signal handling


def _run_kanban_goal_loop_q(cli: "HermesCLI", first_response: str) -> None:
    """Drive a kanban goal_mode worker through ``goals.run_kanban_goal_loop`` after its first turn.

    The caller swallows all errors: a broken loop must never wedge a worker.
    """
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        return
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    worker_run_id = _int_or(raw_run_id, None) if raw_run_id else None
    if raw_run_id and worker_run_id is None:
        logger.warning("invalid HERMES_KANBAN_RUN_ID=%r", raw_run_id)

    from hermes_cli import kanban_db as _kb
    from hermes_cli import kanban_db_connect as _kbc
    from hermes_cli.goals import run_kanban_goal_loop as _run_loop, DEFAULT_MAX_TURNS as _DEF_TURNS

    # Goal text = title + body (the acceptance criteria the judge evaluates against).
    with _kbc.connect_closing() as conn:
        task = _kb.get_task(conn, task_id)
    if task is None:
        return

    goal_text = "\n\n".join(p for p in (task.title or "", task.body) if p).strip()
    if not goal_text:
        return

    def _run_turn(prompt: str) -> str:
        result = cli.agent.run_conversation(user_message=prompt, conversation_history=cli.conversation_history)
        _sync_cli_session_id_from_agent(cli)
        resp = result.get("final_response", "") if isinstance(result, dict) else str(result)
        if resp:
            print(resp)
        return resp or ""

    def _task_status() -> "str | None":
        with _kbc.connect_closing() as c:
            return _kb.goal_run_status(c, task_id, worker_run_id)

    def _block(reason: str) -> None:
        with _kbc.connect_closing() as c:
            _kb.block_task(c, task_id, reason=reason, expected_run_id=worker_run_id)

    _run_loop(
        task_id=task_id, goal_text=goal_text, run_turn=_run_turn, task_status_fn=_task_status, block_fn=_block,
        max_turns=task.goal_max_turns or _DEF_TURNS, first_response=first_response or "",
        log=lambda m: logger.info("%s", m),
    )


def _sync_cli_session_id_from_agent(cli) -> None:
    """Keep ``cli.session_id`` in sync when mid-run compression rotated the agent's session."""
    if getattr(cli.agent, "session_id", None) and cli.agent.session_id != cli.session_id:
        cli.session_id = cli.agent.session_id


def _run_quiet_single_query(cli, effective_query, emitter=None):
    """Quiet (-Q) one-shot turn: run, print the response (stderr for errors/session_id), then sys.exit with the automation exit code.
    With a ``StreamJsonEmitter`` the final answer and the exit line become the terminal ``result`` JSONL record instead.
    HERMES_TURN_AUTHOR (set only by a bot-to-bot dispatcher) is consumed here so tool subprocesses do not inherit it.
    Nested Bot Mode notifies bind this session's key (not the dispatcher's) and resume in-process
    before stdout is printed, so a teammate reply is the quiet run's final answer rather than a
    stranded receipt."""
    from agent.interrupt_compat import _accepts_keyword
    from agent.turn_author import take_turn_author_from_env
    from hermes_cli.quiet_single_query import (
        bind_quiet_session_key, continue_quiet_notify_completions, quiet_notify_linger_seconds,
    )

    author = take_turn_author_from_env()
    author_kwargs = {"turn_author": author} if author is not None and _accepts_keyword(cli.agent.run_conversation, "turn_author") else {}
    with bind_quiet_session_key(getattr(cli, "session_id", "") or "default"):
        try:
            result = cli.agent.run_conversation(
                user_message=effective_query, conversation_history=cli.conversation_history, **author_kwargs,
            )
        except KeyboardInterrupt:
            _emit_interrupted_session_end(cli, reason="keyboard_interrupt")
            if emitter is not None:
                sys.exit(emitter.emit_result({"failed": True, "error": "Interrupted"}, session_id=cli.session_id or "", exit_code=130))
            print(f"\nsession_id: {cli.session_id}", file=sys.stderr)
            sys.exit(130)
        # The exit line below reports session_id to stderr for automation wrappers;
        # without this sync it would point at the ended parent after compression.
        _sync_cli_session_id_from_agent(cli)
        if isinstance(result, dict) and not result.get("failed"):
            history = result.get("messages") or cli.conversation_history

            def _follow_up(text):
                nonlocal history
                follow = cli.agent.run_conversation(
                    user_message=text, conversation_history=history, **author_kwargs,
                )
                if isinstance(follow, dict) and follow.get("messages"):
                    history = follow["messages"]
                # Same sync contract as the main turn: a compression rotation during a
                # follow-up must not leave a stale id on the exit line / drain key.
                _sync_cli_session_id_from_agent(cli)
                return follow

            # One shared linger budget for the whole run: the loop below and the later
            # _wait_for_oneshot_background_completions pass must not each wait the full
            # oneshot_completion_wait_seconds on the same stuck notify_on_complete child.
            # Flagged after the loop (finally-equivalent): the wait is the loop's first
            # statement, so anything raising past that point has consumed budget the
            # finalize pass must not re-wait.
            try:
                continued = continue_quiet_notify_completions(
                    getattr(cli, "session_id", "") or "",
                    _follow_up,
                    owns_event=getattr(cli, "_owns_process_notification", None),
                    linger_budget=quiet_notify_linger_seconds(),
                )
            finally:
                cli._quiet_notify_linger_done = True
            if isinstance(continued, dict):
                result = continued
        response = result.get("final_response", "") if isinstance(result, dict) else str(result)
    # Surface backend errors that produced no visible output (e.g. invalid model slug
    # -> provider 4xx) on stderr so piped stdout stays clean.
    if emitter is not None:
        pass  # the result record below carries text/error; nothing else may touch stdout
    elif (
        not response and isinstance(result, dict) and result.get("error")
        and (result.get("failed") or result.get("partial"))
    ):
        print(f"Error: {result['error']}", file=sys.stderr)
    elif response:
        print(response)

    # Kanban goal_mode: keep working in THIS session until a judge agrees the card is
    # done, the worker terminates it, or the turn budget runs out (sticky block).
    if os.environ.get("HERMES_KANBAN_GOAL_MODE") == "1":
        try:
            _run_kanban_goal_loop_q(cli, response)
        except Exception as _goal_exc:
            logger.debug("kanban goal loop failed: %s", _goal_exc)

    if emitter is None:
        print(f"\nsession_id: {cli.session_id}", file=sys.stderr)

    # Exit code 0/1 for automation wrappers. Kanban workers that failed purely on
    # rate-limit/billing exit with the EX_TEMPFAIL sentinel so the dispatcher releases
    # the task without counting a failure (a quota window must not trip the breaker).
    _exit_code = 0
    if isinstance(result, dict) and result.get("failed"):
        _exit_code = 1
        if os.environ.get("HERMES_KANBAN_TASK") and result.get("failure_reason") in ("rate_limit", "billing"):
            try:
                from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE as _RL_CODE
                _exit_code = _RL_CODE
            except Exception:
                _exit_code = 1
    if emitter is not None:
        _exit_code = emitter.emit_result(result, session_id=cli.session_id or "", exit_code=_exit_code)
    sys.exit(_exit_code)


def _route_single_query_images(cli, query, effective_query, single_query_images, single_query_image_urls):
    """Attach one-shot images natively when the model supports vision, else pre-describe them as text."""
    if not (single_query_images or single_query_image_urls):
        return effective_query
    # Same image-routing decision as the interactive path: a vision-capable model
    # (incl. custom-provider models declaring `model.supports_vision: true`) gets
    # native image_url parts; otherwise the text pipeline (vision_analyze
    # pre-description).
    _img_mode = "text"
    _build_parts = None
    try:
        from agent.image_routing import build_native_content_parts as _build_parts  # noqa: F811
        from agent.image_routing import decide_image_input_mode
        from hermes_cli.config import load_config

        _img_mode = decide_image_input_mode(
            (cli.provider or "").strip(), (cli.model or "").strip(), load_config(),
            requested_provider=(cli.requested_provider or "").strip(),
        )
    except Exception:
        _img_mode = "text"

    def _text_fallback():
        # ``_preprocess_images_with_vision`` only knows local files; when only URLs
        # were supplied keep the original query text intact.
        if single_query_images:
            return cli._preprocess_images_with_vision(query, single_query_images, announce=False)
        return effective_query

    if _img_mode != "native" or _build_parts is None:
        return _text_fallback()
    try:
        _parts, _skipped = _build_parts(
            query if isinstance(query, str) else "",
            [str(p) for p in single_query_images],
            image_urls=list(single_query_image_urls) or None,
        )
        if any(p.get("type") == "image_url" for p in _parts):
            return _parts
        return _text_fallback()  # all images unreadable
    except Exception:
        return _text_fallback()


def _collect_kanban_task_images(single_query_images):
    """Kanban workers: image paths/URLs in the task body join the first turn's attachments."""
    single_query_image_urls: list[str] = []
    _kanban_task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not _kanban_task_id:
        return single_query_image_urls
    try:
        from hermes_cli import kanban_db as _kb
        from hermes_cli import kanban_db_connect as _kbc
        from agent.image_routing import extract_image_refs as _extract_refs

        with _kbc.connect_closing() as _conn:
            _task = _kb.get_task(_conn, _kanban_task_id)
        _body = getattr(_task, "body", "") if _task is not None else ""
        if _body:
            _kb_paths, _kb_urls = _extract_refs(_body)
            # Dedupe against any --image the user already passed.
            _seen = {str(p) for p in single_query_images}
            for _p in _kb_paths:
                if _p not in _seen:
                    _seen.add(_p)
                    single_query_images.append(Path(_p))
            single_query_image_urls.extend(_kb_urls)
    except Exception as _exc:
        # Best-effort enrichment; never block worker startup on it.
        logger.debug("kanban image-ref extraction failed: %s", _exc)
    return single_query_image_urls


def _install_single_query_signal_handlers(cli):
    """Route SIGINT/SIGTERM/SIGHUP through agent.interrupt() before unwinding; kanban workers hard-exit.

    A plain KeyboardInterrupt only unwinds the main thread, so tool worker threads
    would orphan the setsid child; the interrupt + grace window lets them kill it.
    """
    import signal as _signal

    def _signal_handler_q(signum, frame):
        logger.debug("Received signal %s in single-query mode", signum)
        _arm_exit_watchdog_on_shutdown_signal()  # covers wedges in the unwind below
        _interrupt_agent_for_signal(getattr(cli, "agent", None), signum)
        # Kanban: a non-daemon worker blocked in _wait_for_process survives KeyboardInterrupt
        # and the dispatcher sees 'running' forever, so os._exit(0) (SIGALRM deadman guards
        # a blocking flush). That skips atexit + the token-drain hook, hence the explicit flush.
        # Kanban worker exit path (#28181): SIGTERM hits a dispatcher-spawned worker that's likely in a
        # non-daemon thread waiting on a child subprocess in _wait_for_process. Raising KeyboardInterrupt
        # only unwinds the main thread; the worker thread keeps running, the process gets reparented to
        # init, and the dispatcher's _pid_alive check returns True forever — task stuck in 'running'
        # indefinitely. Skip the controlled-unwind dance and call os._exit(0) so the kernel reclaims the PID
        # immediately and detect_crashed_workers can reclaim the stale claim on the next tick. Flush logging
        # + stdout/stderr first so the final debug trace isn't lost; SIGALRM deadman guards the flush
        # against any rare blocking-I/O case (the reporter measured flush in <1ms; the alarm is a failsafe,
        # not the common path).
        if os.environ.get("HERMES_KANBAN_TASK"):
            with suppress(Exception):
                if hasattr(_signal, "SIGALRM"):
                    _signal.signal(_signal.SIGALRM, lambda *_: os._exit(0))
                    _signal.alarm(5)
            with suppress(Exception):
                # Durable flush FIRST: memory-provider shutdown inside _run_cleanup can issue aux-LLM calls,
                # and nothing after it may fail in a way that loses the turn (#88583).
                # os._exit(0) skips atexit AND SessionDB's token-drain hook, so flush + finalize the session
                # store here or the worker's turn (and its usage deltas) never become durable (#88583 /
                # #50881 class). Best-effort under the SIGALRM deadman above.
                _flush_one_shot_session_store(cli)
            _flush_logging_and_stdio()
            os._exit(0)
        raise KeyboardInterrupt()
    with suppress(Exception):  # restricted environments
        for _name in ("SIGINT", "SIGTERM", "SIGHUP"):
            if hasattr(_signal, _name):
                _signal.signal(getattr(_signal, _name), _signal_handler_q)


def _build_cli_from_args(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget, verbose, compact, resume, checkpoints, pass_session_id, ignore_rules, skills):
    """Resolve the toolset list (explicit / coding posture / platform default), construct HermesCLI, and start the background skills preload."""
    toolsets_list = None
    if isinstance(toolsets, str) and toolsets:
        toolsets_list = [t.strip() for t in toolsets.split(",")]
    elif isinstance(toolsets, (list, tuple)) and toolsets:
        # Fire may pass multiple --toolsets as a tuple
        toolsets_list = []
        for t in toolsets:
            toolsets_list.extend([x.strip() for x in t.split(",")] if isinstance(t, str) else [str(t)])
    elif not toolsets:
        # Coding posture inside a code workspace, else the shared platform resolver.
        try:
            from agent.coding_context import coding_selection
            toolsets_list = coding_selection(platform="cli", config=CLI_CONFIG)
        except Exception:
            toolsets_list = None
        if toolsets_list is None:
            from hermes_cli.tools_config import _get_platform_tools
            toolsets_list = sorted(_get_platform_tools(CLI_CONFIG, "cli"))

    parsed_skills = _parse_skills_argument(skills)

    try:
        cli = HermesCLI(
            model=model,
            toolsets=toolsets_list,
            provider=provider,
            reasoning=reasoning,
            api_key=api_key,
            base_url=base_url,
            max_turns=max_turns,
            run_budget=run_budget,
            verbose=verbose,
            compact=compact,
            resume=resume,
            checkpoints=checkpoints,
            pass_session_id=pass_session_id,
            ignore_rules=ignore_rules,
        )
    except ImportError as e:
        # Direct `python cli.py` bypasses cmd_chat's partial-update ImportError handler.
        from hermes_constants import emit_partial_update_hint

        if emit_partial_update_hint(e):
            sys.exit(1)
        raise

    # skills.auto_load rides the same background preload as -s; --ignore-rules skips it with
    # the rest of the auto-injected context. Resolved here (not lazily in the agent) so the
    # session id is real for ${HERMES_SESSION_ID} and -s can dedupe against it.
    from agent.skill_commands import build_auto_load_prompt, resolve_auto_load_skills
    auto_load_names = [] if getattr(cli, "ignore_rules", ignore_rules) else resolve_auto_load_skills(CLI_CONFIG)
    if not auto_load_names:
        cli._auto_load_skills_result = ("", [], [])
    if parsed_skills or auto_load_names:
        # Load the skill payloads in the background: skill_view walks the full skills
        # tree per skill (~0.5s for a large library) and the result is only consumed
        # at agent init, not by the banner. finalize_preloaded_skills() joins the
        # thread before any consumer reads cli.system_prompt.
        def _load_preloaded_skills() -> None:
            try:
                if auto_load_names:
                    cli._auto_load_skills_result = build_auto_load_prompt(task_id=cli.session_id, user_config=CLI_CONFIG)
                if parsed_skills:
                    cli._preload_skills_result = build_preloaded_skills_prompt(
                        parsed_skills, task_id=cli.session_id, excluded_loaded_names=set(cli._auto_load_skills_result[1]))
            except Exception as exc:  # surfaced by finalize
                cli._preload_skills_error = exc

        cli._preload_skills_requested = [*auto_load_names, *(s for s in parsed_skills if s not in auto_load_names)]
        cli._preload_skills_thread = threading.Thread(target=_load_preloaded_skills, name="skills-preload", daemon=True)
        cli._preload_skills_thread.start()
    return cli


def _run_legacy_gateway():
    """Legacy `cli.py --gateway` entry: arm the startup watchdog (before importing the gateway graph), then run it."""
    import asyncio
    with suppress(Exception):
        from hermes_startup_watchdog import arm_startup_watchdog
        arm_startup_watchdog()
    from gateway.run import start_gateway
    print("Starting Hermes Gateway (messaging platforms)...")
    asyncio.run(start_gateway())


def _start_worktree_setup(list_tools, list_toolsets, worktree, w):
    """Start isolated-worktree creation (+ tool prewarm) in the background.

    Returns a join callable that publishes ``_active_worktree``/TERMINAL_CWD and
    schedules stale-worktree GC, or None when no worktree is wanted.
    """
    if list_tools or list_toolsets or not (worktree or w or CLI_CONFIG.get("worktree", False)):
        return None
    # Overlap tool discovery with the I/O-bound worktree setup so show_banner() hits a warm
    # cache (~0.4s). Only on the -w path: plain `hermes` has no I/O wait to hide.
    def _prewarm_tools() -> None:
        try:
            import model_tools as _mt
            _mt.get_tool_definitions(quiet_mode=True)
        except Exception:
            logger.debug("tool prewarm failed", exc_info=True)

    threading.Thread(target=_prewarm_tools, name="tool-prewarm", daemon=True).start()
    _sync_base = CLI_CONFIG.get("worktree_sync", True)
    _wt_result: dict = {}

    def _create_worktree() -> None:
        try:
            _wt_result["info"] = _setup_worktree(sync_base=_sync_base)
        except Exception:
            logger.debug("worktree setup failed", exc_info=True)
            _wt_result["info"] = None

    _wt_thread = threading.Thread(target=_create_worktree, name="worktree-setup", daemon=True)
    _wt_thread.start()

    def _worktree_maintenance(repo: str) -> None:
        _prune_stale_worktrees(repo)
        _maintain_pack_health(repo)

    def _join_worktree() -> Optional[Dict[str, str]]:
        _wt_thread.join(timeout=120)
        info = _wt_result.get("info")
        if not info:
            return info
        global _active_worktree
        _active_worktree = info
        os.environ["TERMINAL_CWD"] = info["path"]
        atexit.register(_cleanup_worktree, info)
        # GC stale worktrees AFTER _setup_worktree so they never race on git's worktree
        # metadata (the new tree is immune: <24h age gate + live pid lock); then repack
        # once refs are final so lookups stay fast on multi-agent boxes.
        _repo = _git_repo_root()
        if _repo:
            threading.Thread(target=_worktree_maintenance, args=(_repo,), name="worktree-prune", daemon=True).start()
        return info

    return _join_worktree


def _configure_quiet_agent(agent) -> None:
    """Neutralize every stdout-writing callback so -Q stdout carries only the final response."""
    agent.quiet_mode = True
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None
    agent.reasoning_callback = None
    # The diff/progress callbacks print directly and are gated by neither quiet_mode nor
    # tool_progress_mode, so they must go too; "off" also covers the executor's direct prints.
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    agent.tool_complete_callback = None
    agent.tool_progress_mode = "off"


def _run_single_query_mode(cli, query, image, quiet, oneshot, stream_json: bool = False):
    """``-q``/``--image`` entry: seed an interactive session on a TTY, else run the one-shot turn and exit.
    ``stream_json`` (implies quiet) swaps the plain-text final answer for the JSONL event protocol."""
    if _should_seed_interactive(query, image, quiet, oneshot):
        seeded_query, seeded_images = _collect_query_images(query, image)
        logger.info(
            "Seeding interactive session with -q prompt (%d chars, %d images)",
            len(seeded_query or ""), len(seeded_images),
        )
        cli._seeded_first_message = _SeededQueryMessage(seeded_query, seeded_images)
        return cli.run()
    cli._single_query_mode = True  # agent waits the full MCP cold-start before its only tool snapshot
    # No user can answer approval prompts: the approval gate takes the deterministic path.
    # One-shot mode: no between-turns MCP late-binding refresh, so the agent must wait the full MCP
    # cold-start bound before its first (and only) tool snapshot. See #51316.
    # Mark single-query for the approval gate. cli.py sets HERMES_INTERACTIVE earlier for interactive sudo
    # prompts, but a -q run has NO user waiting to answer approval prompts. The gate reads this marker (via
    # gateway.session_context.get_session_env, which falls back to os.environ when the session-context layer
    # isn't engaged) and takes the deterministic approvals.single_query_mode path instead of waiting the
    # full timeout. See #86878.
    os.environ["HERMES_SINGLE_QUERY_SESSION"] = "1"
    if not cli._claim_active_session("cli", stderr=bool(quiet)):
        sys.exit(1)
    try:
        query, single_query_images = _collect_query_images(query, image)
        single_query_image_urls = _collect_kanban_task_images(single_query_images)
        if quiet:
            # Quiet mode: suppress banner, spinner, tool previews.
            cli.tool_progress_mode = "off"
            emitter = None
            if stream_json:
                # Built BEFORE credentials/agent init so a failed start still closes the protocol
                # (init + result) instead of exiting 1 with an empty stdout.
                from hermes_cli.stream_json import StreamJsonEmitter
                emitter = StreamJsonEmitter(model=getattr(cli, "model", "") or "", session_id=cli.session_id or "")
            if cli._ensure_runtime_credentials():
                effective_query: Any = _route_single_query_images(
                    cli, query, query, single_query_images, single_query_image_urls
                )
                turn_route = cli._resolve_turn_agent_config(effective_query)
                if turn_route["signature"] != cli._active_agent_route_signature:
                    cli.agent = None
                if cli._init_agent(
                    model_override=turn_route["model"],
                    runtime_override=turn_route["runtime"],
                    request_overrides=turn_route.get("request_overrides"),
                ):
                    _configure_quiet_agent(cli.agent)
                    if emitter is not None:
                        emitter.attach(cli.agent)
                    _run_quiet_single_query(cli, effective_query, emitter=emitter)

            if emitter is not None:
                emitter.emit_result({"failed": True, "error": "credentials or agent init failed"},
                                    session_id=cli.session_id or "", exit_code=1)
            sys.exit(1)  # credentials or agent init failed
        # No welcome banner (~420 ms cold); session id / resume hint come from _print_exit_summary().
        _query_label = query or ("[image attached]" if single_query_images else "")
        if _query_label:
            cli.console.print(f"[bold blue]Query:[/] {_query_label}")
        cli._show_security_advisories()
        cli.chat(query, images=single_query_images or None)
        cli._print_exit_summary(clear_screen=False)
    finally:
        _finalize_single_query(cli)


def main(
    query: str = None,
    q: str = None,
    oneshot: bool = False,
    image: str = None,
    toolsets: str = None,
    skills: str | list[str] | tuple[str, ...] = None,
    model: str = None,
    provider: str = None,
    reasoning: str = None,
    api_key: str = None,
    base_url: str = None,
    max_turns: int = None,
    run_budget: float = None,
    verbose: Optional[bool] = None,
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
    output_format: str = "text",
    ignore_user_config: bool = False,
    ignore_rules: bool = False,
):
    """
    Hermes Agent CLI - Interactive AI Assistant
    
    Args:
        query: Query to run. On a real TTY this seeds an interactive session
            (submitted literally as the first turn); with --oneshot/-Q or a
            non-TTY it answers and exits. Alias: -q
        q: Shorthand for --query
        oneshot: With -q: force the legacy answer-and-exit single-query mode
            even on a TTY.
        image: Optional local image path to attach to a single query
        toolsets: Comma-separated list of toolsets to enable (e.g., "web,terminal")
        skills: Comma-separated or repeated list of skills to preload for the session
        model: Model to use (default: anthropic/claude-opus-4-20250514)
        provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
        reasoning: Reasoning effort for this run (none|minimal|low|medium|high|xhigh|max|ultra). Overrides agent.reasoning_effort.
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
        python cli.py -q "Describe this" --image ~/storage/shared/Pictures/cat.png
        python cli.py --list-tools               # List tools and exit
        python cli.py --resume 20260225_143052_a1b2c3  # Resume session
        python cli.py -w                         # Start in isolated git worktree
        python cli.py -w -q "Fix issue #123"     # Single query in worktree
    """
    # UTF-8 stdio on Windows before any print (Rich box-drawing would UnicodeEncodeError on cp1252).
    with suppress(Exception):
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()

    os.environ["HERMES_INTERACTIVE"] = "1"  # terminal_tool: interactive sudo prompts with timeout
    # The banner names affected plugins; the raw per-name compat warnings would only duplicate it on stderr.
    with suppress(Exception):
        from hermes_cli.plugin_compat import quiet_for_interactive
        quiet_for_interactive()

    if gateway:
        _run_legacy_gateway()
        return

    _join_worktree = _start_worktree_setup(list_tools, list_toolsets, worktree, w)
    query = query or q
    # ``hermes chat`` already validated this; the direct Fire entry point gets the same contract.
    if output_format == "stream-json":
        if not query:
            raise ValueError("--format stream-json requires -q/--query")
        quiet = True
    cli = _build_cli_from_args(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget,
                               verbose, compact, resume, checkpoints, pass_session_id, ignore_rules, skills)

    # Create CLI instance
    cli = HermesCLI(
        model=model,
        toolsets=toolsets_list,
        provider=provider,
        reasoning=reasoning,
        api_key=api_key,
        base_url=base_url,
        max_turns=max_turns,
        run_budget=run_budget,
        verbose=verbose,
        compact=compact,
        resume=resume,
        checkpoints=checkpoints,
        pass_session_id=pass_session_id,
        ignore_rules=ignore_rules,
    )

    if parsed_skills:
        # Load the skill payloads in the background: skill_view walks the
        # full skills tree per skill (~0.5s for a large library) and the
        # result is only consumed at agent init (first message / first
        # agent-touching command), not by the banner. cmd_chat joins the
        # thread via cli.finalize_preloaded_skills() before any consumer
        # reads cli.system_prompt — HermesCLI._create_agent calls it too,
        # so no agent can be built with the skills missing.
        def _load_preloaded_skills() -> None:
            try:
                cli._preload_skills_result = build_preloaded_skills_prompt(
                    parsed_skills,
                    task_id=cli.session_id,
                )
            except Exception as exc:  # surfaced by finalize below
                cli._preload_skills_error = exc

        cli._preload_skills_requested = parsed_skills
        cli._preload_skills_thread = threading.Thread(
            target=_load_preloaded_skills, name="skills-preload", daemon=True
        )
        cli._preload_skills_thread.start()

    # Join the background worktree creation (started above) before anything
    # consumes TERMINAL_CWD / wt_info — the HermesCLI construction it
    # overlapped with is done. Setup failure keeps the old abort semantics.
    if _join_worktree is not None:
        wt_info = _join_worktree()
        if not wt_info:
            # Worktree was explicitly requested but setup failed —
            # don't silently run without isolation.
            return

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

    if list_tools or list_toolsets:
        cli.show_banner()
        (cli.show_tools if list_tools else cli.show_toolsets)()
        sys.exit(0)

    atexit.register(_run_cleanup)  # interactive mode registers again in run() (idempotent)
    _install_single_query_signal_handlers(cli)

    if query or image:
        # One-shot mode: no between-turns MCP late-binding refresh, so the
        # agent must wait the full MCP cold-start bound before its first
        # (and only) tool snapshot. See #51316.
        cli._single_query_mode = True
        # Mark single-query for the approval gate. cli.py sets
        # HERMES_INTERACTIVE earlier for interactive sudo prompts, but a -q
        # run has NO user waiting to answer approval prompts. The gate reads
        # this marker (via gateway.session_context.get_session_env, which falls
        # back to os.environ when the session-context layer isn't engaged) and
        # takes the deterministic approvals.single_query_mode path instead of
        # waiting the full timeout. See #86878.
        os.environ["HERMES_SINGLE_QUERY_SESSION"] = "1"
        if not cli._claim_active_session("cli", stderr=bool(quiet)):
            sys.exit(1)
        try:
            query, single_query_images = _collect_query_images(query, image)
            # Kanban workers spawn with ``hermes chat -q "work kanban task <id>"``;
            # the actual task description lives in the task body. Mirror the
            # gateway/CLI behaviour for inbound images by scanning the body for
            # local image paths and http(s) image URLs and attaching them to the
            # worker's first turn. Without this, users who paste a screenshot
            # path or URL into a kanban task body never get it routed to the
            # model's vision input.
            single_query_image_urls: list[str] = []
            _kanban_task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
            if _kanban_task_id:
                try:
                    from hermes_cli import kanban_db as _kb
                    from agent.image_routing import extract_image_refs as _extract_refs

                    _conn = _kb.connect()
                    try:
                        _task = _kb.get_task(_conn, _kanban_task_id)
                    finally:
                        try:
                            _conn.close()
                        except Exception:
                            pass
                    _body = getattr(_task, "body", "") if _task is not None else ""
                    if _body:
                        _kb_paths, _kb_urls = _extract_refs(_body)
                        if _kb_paths:
                            # Dedupe against any --image the user already passed.
                            _seen = {str(p) for p in single_query_images}
                            for _p in _kb_paths:
                                if _p not in _seen:
                                    _seen.add(_p)
                                    single_query_images.append(Path(_p))
                        if _kb_urls:
                            single_query_image_urls.extend(_kb_urls)
                except Exception as _exc:
                    # Best-effort enrichment; never block worker startup on it.
                    logger.debug("kanban image-ref extraction failed: %s", _exc)
            if quiet:
                # Quiet mode: suppress banner, spinner, tool previews.
                # Only print the final response and parseable session info.
                cli.tool_progress_mode = "off"
                if cli._ensure_runtime_credentials():
                    effective_query: Any = query
                    if single_query_images or single_query_image_urls:
                        # Honour the same image-routing decision used by the
                        # interactive path. With a vision-capable model (incl.
                        # custom-provider models declared via
                        # `model.supports_vision: true`), attach images natively
                        # as image_url content parts. Otherwise fall back to the
                        # text-pipeline (vision_analyze pre-description).
                        _img_mode = "text"
                        _build_parts = None
                        try:
                            from agent.image_routing import (
                                build_native_content_parts as _build_parts,  # noqa: F811
                            )
                            from agent.image_routing import decide_image_input_mode
                            from hermes_cli.config import load_config

                            _img_mode = decide_image_input_mode(
                                (cli.provider or "").strip(),
                                (cli.model or "").strip(),
                                load_config(),
                                requested_provider=(
                                    cli.requested_provider or ""
                                ).strip(),
                            )
                        except Exception:
                            _img_mode = "text"

                        if _img_mode == "native" and _build_parts is not None:
                            try:
                                _parts, _skipped = _build_parts(
                                    query if isinstance(query, str) else "",
                                    [str(p) for p in single_query_images],
                                    image_urls=list(single_query_image_urls) or None,
                                )
                                if any(p.get("type") == "image_url" for p in _parts):
                                    effective_query = _parts
                                else:
                                    # All images unreadable — text fallback.
                                    # ``_preprocess_images_with_vision`` only knows
                                    # about local files; URLs would be lost there,
                                    # so keep the original query text intact when
                                    # only URLs were supplied.
                                    if single_query_images:
                                        effective_query = cli._preprocess_images_with_vision(
                                            query, single_query_images, announce=False,
                                        )
                            except Exception:
                                if single_query_images:
                                    effective_query = cli._preprocess_images_with_vision(
                                        query, single_query_images, announce=False,
                                    )
                        elif single_query_images:
                            effective_query = cli._preprocess_images_with_vision(
                                query,
                                single_query_images,
                                announce=False,
                            )
                    turn_route = cli._resolve_turn_agent_config(effective_query)
                    if turn_route["signature"] != cli._active_agent_route_signature:
                        cli.agent = None
                    if cli._init_agent(
                        model_override=turn_route["model"],
                        runtime_override=turn_route["runtime"],
                        request_overrides=turn_route.get("request_overrides"),
                    ):
                        cli.agent.quiet_mode = True
                        cli.agent.suppress_status_output = True
                        # Suppress streaming display callbacks so stdout stays
                        # machine-readable (no styled "Hermes" box, no tool-gen
                        # status lines, no reasoning box).  The response is
                        # printed once below.
                        cli.agent.stream_delta_callback = None
                        cli.agent.tool_gen_callback = None
                        cli.agent.reasoning_callback = None
                        # Inline-diff and progress callbacks print directly to
                        # stdout and are gated by NEITHER quiet_mode nor
                        # tool_progress_mode: _on_tool_complete renders full
                        # file diffs via render_edit_diff_with_delta, and
                        # _on_tool_progress prints MoA reference blocks before
                        # its mode check. Neutralize them too so -Q stdout
                        # carries only the final response (#93220).
                        cli.agent.tool_progress_callback = None
                        cli.agent.tool_start_callback = None
                        cli.agent.tool_complete_callback = None
                        # Belt-and-braces for the executor's direct prints
                        # (they check agent.tool_progress_mode, initialized
                        # from display.tool_progress at construction).
                        cli.agent.tool_progress_mode = "off"
                        try:
                            result = cli.agent.run_conversation(
                                user_message=effective_query,
                                conversation_history=cli.conversation_history,
                            )
                        except KeyboardInterrupt:
                            _emit_interrupted_session_end(cli, reason="keyboard_interrupt")
                            print(f"\nsession_id: {cli.session_id}", file=sys.stderr)
                            sys.exit(130)
                        # Sync session_id if mid-run compression created a
                        # continuation session. The exit line below reports
                        # session_id to stderr for automation wrappers; without
                        # this sync it would point at the ended parent.
                        if (
                            getattr(cli.agent, "session_id", None)
                            and cli.agent.session_id != cli.session_id
                        ):
                            cli.session_id = cli.agent.session_id
                        response = result.get("final_response", "") if isinstance(result, dict) else str(result)
                        # Surface backend errors that produced no visible output
                        # (e.g. invalid model slug → provider 4xx). Mirrors the
                        # interactive CLI path. Write to stderr so piped stdout
                        # stays clean for automation wrappers.
                        if (
                            not response
                            and isinstance(result, dict)
                            and result.get("error")
                            and (result.get("failed") or result.get("partial"))
                        ):
                            print(f"Error: {result['error']}", file=sys.stderr)
                        elif response:
                            print(response)

                        # Kanban goal-loop mode: a worker spawned for a
                        # goal_mode card keeps working in THIS session until an
                        # auxiliary judge agrees the card is done, the worker
                        # terminates the task itself, or the turn budget runs
                        # out (→ sticky block). Gated on the env vars the
                        # dispatcher sets in `_default_spawn`; a no-op for every
                        # normal worker and every non-kanban `-q` run.
                        if os.environ.get("HERMES_KANBAN_GOAL_MODE") == "1":
                            try:
                                _run_kanban_goal_loop_q(cli, response)
                            except Exception as _goal_exc:
                                logger.debug("kanban goal loop failed: %s", _goal_exc)

                        # Session ID goes to stderr so piped stdout is clean.
                        print(f"\nsession_id: {cli.session_id}", file=sys.stderr)

                        # Ensure proper exit code for automation wrappers.
                        #
                        # Kanban workers get a special case: when the run failed
                        # purely because the provider rate-limited / exhausted
                        # quota (not because the task itself is broken), exit with
                        # the EX_TEMPFAIL sentinel instead of the generic 1. The
                        # dispatcher's reap classifier maps that code to a
                        # ``rate_limited`` exit and releases the task back to
                        # ``ready`` WITHOUT incrementing the failure counter, so a
                        # 5-hour quota window can't trip the circuit breaker and
                        # permanently block the card. Non-kanban runs keep the
                        # plain 0/1 contract automation wrappers expect.
                        _exit_code = 0
                        if isinstance(result, dict) and result.get("failed"):
                            _exit_code = 1
                            if os.environ.get("HERMES_KANBAN_TASK") and result.get(
                                "failure_reason"
                            ) in ("rate_limit", "billing"):
                                try:
                                    from hermes_cli.kanban_db import (
                                        KANBAN_RATE_LIMIT_EXIT_CODE as _RL_CODE,
                                    )
                                    _exit_code = _RL_CODE
                                except Exception:
                                    _exit_code = 1
                        sys.exit(_exit_code)

                # Exit with error code if credentials or agent init fails
                sys.exit(1)
            else:
                # Single-query mode (`hermes chat -q "…"`): skip the welcome
                # banner. Building the banner takes ~420 ms on cold start —
                # ~200 ms of that is the version-update check, the rest is
                # toolset / skill enumeration and Rich panel rendering. None
                # of that is useful for a one-shot query: the user already
                # picked the prompt, doesn't need a toolset reference, and
                # gets the session ID + resume hint from
                # ``_print_exit_summary()`` after the response prints.
                #
                # The fully-quiet ``-Q`` / ``--quiet`` machine-readable path
                # above was already banner-free; this brings the human-
                # facing single-query path in line so all non-interactive
                # invocations are fast.
                _query_label = query or ("[image attached]" if single_query_images else "")
                if _query_label:
                    cli.console.print(f"[bold blue]Query:[/] {_query_label}")
                # Surface security advisories before the agent runs — short
                # banner, doesn't depend on the welcome banner being shown.
                cli._show_security_advisories()
                cli.chat(query, images=single_query_images or None)
                cli._print_exit_summary(clear_screen=False)
        finally:
            _finalize_single_query(cli)
        return
    cli.run()


if __name__ == "__main__":
    import fire

    fire.Fire(main)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from prompt_toolkit.layout.menus import CompletionsMenu  # noqa: F401,E402
from prompt_toolkit.filters import Condition  # noqa: F401,E402
from prompt_toolkit.layout import ConditionalContainer  # noqa: F401,E402
from prompt_toolkit.layout.processors import ConditionalProcessor  # noqa: F401,E402
from prompt_toolkit.layout.dimension import Dimension  # noqa: F401,E402
from prompt_toolkit.history import FileHistory  # noqa: F401,E402
from prompt_toolkit.layout import FormattedTextControl  # noqa: F401,E402
from prompt_toolkit.layout import HSplit  # noqa: F401,E402
from prompt_toolkit.key_binding import KeyBindings  # noqa: F401,E402
from prompt_toolkit.layout import Layout  # noqa: F401,E402
from prompt_toolkit.styles import Style as PTStyle  # noqa: F401,E402
from rich.panel import Panel  # noqa: F401,E402
from prompt_toolkit.layout.processors import PasswordProcessor  # noqa: F401,E402
from prompt_toolkit.layout.processors import Processor  # noqa: F401,E402
from prompt_toolkit.widgets import TextArea  # noqa: F401,E402
from prompt_toolkit.layout.processors import Transformation  # noqa: F401,E402
from prompt_toolkit.layout import Window  # noqa: F401,E402
from prompt_toolkit.layout import WindowAlign  # noqa: F401,E402
import base64  # noqa: F401,E402
import concurrent.futures  # noqa: F401,E402
import copy  # noqa: F401,E402
from rich import box as rich_box  # noqa: F401,E402
import tempfile  # noqa: F401,E402

def AIAgent(*args, **kwargs):
    from run_agent import AIAgent as _AIAgent

    return _AIAgent(*args, **kwargs)

def CanonicalUsage(*args, **kwargs):
    from agent.usage_pricing import CanonicalUsage as _CanonicalUsage

    return _CanonicalUsage(*args, **kwargs)


_PLUGIN_COMPAT_LAZY = {
    'DEFAULT_BROWSER_CDP_URL': ('hermes_cli.browser_connect', 'DEFAULT_BROWSER_CDP_URL'),
    'HERMES_AGENT_LOGO': ('hermes_cli.banner', 'HERMES_AGENT_LOGO'),
    'HERMES_CADUCEUS': ('hermes_cli.banner', 'HERMES_CADUCEUS'),
    'SlashCommandAutoSuggest': ('hermes_cli.commands_completion', 'SlashCommandAutoSuggest'),
    'SlashCommandCompleter': ('hermes_cli.commands_completion', 'SlashCommandCompleter'),
    'build_welcome_banner': ('hermes_cli.banner', 'build_welcome_banner'),
    'display_hermes_home': ('hermes_constants', 'display_hermes_home'),
    'estimate_usage_cost': ('agent.usage_pricing', 'estimate_usage_cost'),
    'get_all_toolsets': ('toolsets', 'get_all_toolsets'),
    'get_job': ('cron.jobs', 'get_job'),
    'get_toolset_for_tool': ('model_tools', 'get_toolset_for_tool'),
    'get_toolset_info': ('toolsets', 'get_toolset_info'),
    'init_skin_from_config': ('hermes_cli.skin_engine', 'init_skin_from_config'),
    'is_browser_debug_ready': ('hermes_cli.browser_connect', 'is_browser_debug_ready'),
    'is_table_divider': ('agent.markdown_tables', 'is_table_divider'),
    'looks_like_table_row': ('agent.markdown_tables', 'looks_like_table_row'),
    'manual_chrome_debug_command': ('hermes_cli.browser_connect', 'manual_chrome_debug_command'),
    'print_config_warnings': ('hermes_cli.config', 'print_config_warnings'),
    'prompt_for_secret': ('hermes_cli.callbacks', 'prompt_for_secret'),
    'set_friendly_tool_labels': ('agent.display', 'set_friendly_tool_labels'),
    'set_tool_preview_max_len': ('agent.display', 'set_tool_preview_max_len'),
    'setup_logging': ('hermes_logging', 'setup_logging'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
