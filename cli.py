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
import uuid
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
from agent.interrupt_compat import request_hard_interrupt
from agent.pet import render as pet_render

from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import Application
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
from hermes_cli.env_loader import load_hermes_dotenv
from utils import base_url_host_matches, base_url_hostname, fast_safe_load

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


def _mirror_config_to_env(defaults, _file_has_terminal_config):
    """Project config.yaml values into the env vars the tool modules read (terminal/browser/auxiliary/security/sessions). Env always wins when already set."""
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

    # TERMINAL_CWD is force-exported (beats stale .env) except inside a gateway process,
    # whose config bridge already set it.
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
    for task_key, env_map in _AUXILIARY_TASK_ENV.items():
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
    """Bounded linger for notify_on_complete background processes (children write to our pipes).

    Waits on the whole registry: a one-shot process hosts one agent, and task_id
    filtering would skip processes registered before the session id settled.

    See #90879.
    """
    from tools.process_registry import process_registry

    _agent, task_id = _oneshot_agent_and_session(cli)
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
        # Order matters: linger for spawned background work BEFORE any teardown (the
        # parent owns those children's stdout pipes); then the durable flush, since
        # memory-provider shutdown inside _run_cleanup can issue aux-LLM calls and
        # nothing after it may fail in a way that loses the turn.
        for step, what in (
            (_wait_for_oneshot_background_completions, "background completion wait"),
            (_flush_one_shot_session_store, "session store flush"),
        ):
            try:
                step(cli)
            except Exception:
                logger.debug("one-shot %s failed", what, exc_info=True)
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
# (#652) ======================================================================_active_worktree: Optional[Dict[str, str]] = None


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
            # Shallow boundary makes the unpushed verdict unreliable; the startup pruner reaps later.
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


_ACCENT_ANSI_DEFAULT = "\033[1;38;2;255;215;0m"  # #FFD700 bold fallback
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
    """Whether the terminal is Ghostty.

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

    Pushes BOTH kitty keyboard protocol and xterm modifyOtherKeys (kitty dropped the
    latter; tmux/VS Code only accept it). Both re-encode modified keys as sequences
    stock prompt_toolkit barely maps (Ctrl+C once arrived as ``ESC[99;5u``), so
    ``install_modify_other_keys_aliases()`` must have run first. Ghostty gets only
    modifyOtherKeys. The exit reset pops both modes.

    Under either protocol the terminal re-encodes modified keys as escape sequences — Kitty disambiguate
    mode as ``ESC[<codepoint>;<mod>u`` (plus the Esc key as ``ESC[27u``), modifyOtherKeys=2 as
    ``ESC[27;<mod>;<codepoint>~``. Stock prompt_toolkit 3.x maps almost none of these, which is why the CSI
    >1u push was temporarily removed in 87074 (Ctrl+C arrived as ``ESC[99;5u`` and died, #56684).
    ``install_modify_other_keys_aliases()`` (called at CLI startup from ``hermes_cli.pt_input_extras``) now
    populates ``ANSI_SEQUENCES`` with the full Ctrl/Alt/Shift/multi-modifier and functional-key tables under
    BOTH formats, so every existing key binding continues to fire — including Ctrl+C, which is handled by
    prompt_toolkit's ``c-c`` binding (raw mode clears ISIG, so the kernel INTR path was never in play for
    the CLI).
    See #87630.
    """
    if not _terminal_supports_extended_enter_keys(env):
        return False
    seq = _MODIFY_OTHER_KEYS_SEQ if _is_ghostty_terminal(env) else _EXTENDED_ENTER_KEYS_SEQ
    try:
        if output is not None and hasattr(output, "write_raw"):
            output.write_raw(seq)
            output.flush()
            return True
        if sys.stdout is not None and sys.stdout.isatty():
            sys.stdout.write(seq)
            sys.stdout.flush()
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
    under heavy subagent / status-line load — see ``tests/cli/test_cpr_local_leak.py``.
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
        tiny_line = "⚕ NOUS HERMES"
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
        # Same fail-closed cron drift warning as `hermes config set` for every model switch.
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


class HermesCLI(CLIAgentSetupMixin, CLICommandsMixin, CLIBillingMixin, CLITuiMixin, CLIStatusBarMixin, CLIVoiceMixin, CLIModelSwitchMixin, CLISessionMixin, CLIStreamMixin, CLIModalMixin, CLITerminalMixin, CLIInfoMixin, CLILoopsMixin, CLIChatTurnMixin):
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
        self._stream_started = self._stream_box_opened = False
        # Possible markdown-table lines held until the block ends for wcwidth-aware re-padding.
        self._stream_table_buf: list[str] = []
        self._in_stream_table = False
        self._pending_edit_snapshots = {}
        self._last_input_mode_recovery = self._last_termios_drift_check = 0.0
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
        _env_mt = os.environ.get("HERMES_MAX_TOKENS")
        _mt = _model_config.get("max_tokens")
        self.max_tokens = _int_or(_env_mt, None) if _env_mt else (_mt if isinstance(_mt, int) else None)
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
        # Key matches the resolved base_url; re-resolved by _ensure_runtime_credentials().
        _keys = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
        if not (self.base_url and base_url_host_matches(self.base_url, "openrouter.ai")):
            _keys = _keys[::-1]
        self.api_key = api_key or os.getenv(_keys[0]) or os.getenv(_keys[1])

    def _init_turn_limits(self, max_turns, run_budget):
        """max_turns: CLI arg > config > env var > default; run budget: CLI flag > config."""
        # resolve_turn_limit() accepts "none"/"unlimited" (-> sys.maxsize) alongside ints.
        # KEEP the root-level CLI_CONFIG["max_turns"] fallback: it is never migrated on disk
        # and other config paths may bypass the load-time fold.
        from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
        self.max_turns = _resolve_turn_limit(next(
            (v for v in (max_turns, CLI_CONFIG["agent"].get("max_turns"), CLI_CONFIG.get("max_turns")) if v is not None),
            os.getenv("HERMES_MAX_ITERATIONS"),
        ))
        self.run_budget_seconds = run_budget if run_budget is not None else CLI_CONFIG["agent"].get("run_budget_seconds")

    def _init_toolsets(self, toolsets):
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
        self.ignore_rules = ignore_rules or os.environ.get("HERMES_IGNORE_RULES") == "1"

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
        self.session_id = resume or f"{self.session_start.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        getattr(self, "_write_terminal_breadcrumb", lambda: None)()

        self._history_file = _hermes_home / ".hermes_history"
        self._last_invalidate: float = 0.0  # throttles UI repaints
        self._init_ui_state()

    def _init_session_store(self):
        """Open the session store early (so /title works before the first message) + opportunistic maintenance."""
        self._session_db = None
        self._session_db_unavailable = False
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            # Without a store the transcript is NOT persisted while the chat looks healthy,
            # so surface it prominently rather than only logging.
            # #41386: a failed session store means the transcript is NOT persisted to state.db — the live
            # chat looks healthy but resume later shows a truncated/empty session. A buried log line is not
            # enough; surface it prominently so the user knows persistence is off for this run and can fix
            # the store before relying on resume.
            self._session_db_unavailable = True
            logger.warning("Failed to initialize SessionDB — session will NOT be indexed for search: %s", e)
            try:
                Console(stderr=True).print(
                    "[bold yellow]⚠ Session store unavailable[/bold yellow] — "
                    "this conversation will [bold]NOT be saved[/bold] to disk and "
                    "cannot be resumed later. Searching past sessions is also disabled.\n"
                    f"  Reason: {e}\n"
                    "  Fix the state.db store (e.g. `hermes update` to rebuild the venv) to restore persistence."
                )
            except Exception:
                print(
                    "WARNING: Session store unavailable — this conversation will NOT be "
                    f"saved to disk and cannot be resumed later. Reason: {e}"
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
        self._clarify_prefill = ""
        self._sudo_state = self._modal_input_snapshot = self._approval_state = None
        self._slash_confirm_state = self._model_picker_state = None
        self._clarify_deadline = self._sudo_deadline = self._approval_deadline = self._slash_confirm_deadline = 0
        self._approval_lock = threading.Lock()
        try:  # composer placeholder chosen once so it stays stable on screen
            from hermes_cli.tips import get_random_composer_placeholder
            self._composer_placeholder = get_random_composer_placeholder()
        except Exception:
            self._composer_placeholder = ""
        self._command_palette_state = self._secret_state = None
        self._pending_resume_sessions = None  # armed by a bare `/resume`; the next bare number selects
        self._pending_agent_seed = None  # one-shot seed from a slash handler
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
            from hermes_cli.active_sessions import try_acquire_active_session

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
            print(message, file=sys.stderr) if stderr else self._console_print(f"[bold red]{message}[/]")
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

    def _install_tool_callbacks(self) -> None:
        """Install tool callbacks that need the live prompt UI."""
        if self._tool_callbacks_installed:
            return
        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)
        set_secret_capture_callback(self._secret_capture_callback)
        try:
            from tools.computer_use_tool import set_approval_callback as _set_cu_cb

            _set_cu_cb(self._computer_use_approval_callback)
        except ImportError:
            pass
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
            self.preloaded_skills = loaded_skills

    def _show_tool_availability_warnings(self):
        """Warn about tools disabled by missing API keys (not system deps)."""
        try:
            from model_tools import check_tool_availability

            available, unavailable = check_tool_availability()
            api_key_missing = [u for u in unavailable if u["missing_vars"]]

            if api_key_missing:
                self._console_print()
                self._console_print("[yellow]⚠️  Some tools disabled (missing API keys):[/]")
                for item in api_key_missing:
                    self._console_print(f"   [dim]• {item['name']}[/] [dim italic]({', '.join(item['missing_vars'])})[/]")
                self._console_print("[dim]   Run 'hermes setup' to configure[/]")
        except Exception:
            pass

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

        display_key = self.api_key
        if self.agent is not None and getattr(self.agent, "api_key", None):
            display_key = self.agent.api_key
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
        for name, rows in sections:
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

        entry = self._slash_handler(canonical)
        if entry is None:
            return self._process_unregistered_slash(cmd_original, cmd_lower)
        method_name, pass_arg = entry
        handler = getattr(self, method_name)
        result = handler(cmd_original) if pass_arg else handler()
        return result is not False

    def _process_unregistered_slash(self, cmd_original: str, cmd_lower: str) -> bool:
        """Slash input with no built-in handler; precedence: quick_commands -> plugins -> bundles -> skills -> prefix expansion."""
        base_cmd = cmd_lower.split()[0]
        bare = base_cmd.lstrip("/")
        skill_commands = _ensure_skill_commands()
        skill_bundles = get_skill_bundles()
        quick_commands = self.config.get("quick_commands", {})
        user_args = cmd_original[len(base_cmd):].strip()
        if bare in quick_commands:
            return self._run_quick_command(base_cmd, quick_commands[bare], user_args)
        if bare in _get_plugin_cmd_handler_names():
            self._run_plugin_slash_command(base_cmd, user_args)
        elif base_cmd in skill_bundles:
            self._run_skill_bundle_command(base_cmd, skill_bundles[base_cmd], user_args)
        elif base_cmd in skill_commands:
            self._run_skill_slash_command(base_cmd, skill_commands[base_cmd], user_args)
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
            _cprint(f"\033[1;31mUnknown command: {cmd_lower}{_RST}")
            _cprint(f"{_DIM}{_ACCENT}Type /help for available commands{_RST}")
        return True

    def _owns_process_notification(self, event: dict) -> bool:
        """Whether this session owns a delegation event (pre-compression keys resolve to their continuation; fail closed)."""
        event_key = str(event.get("session_key") or "")
        current_key = str(getattr(self, "session_id", "") or "")
        if not event_key or not current_key:
            return False
        if event_key == current_key:
            return True
        try:
            session_db = getattr(self, "_session_db", None)
            resolved_key = (
                session_db.resolve_resume_session_id(event_key) if session_db is not None else event_key
            ) or event_key
        except Exception:
            resolved_key = event_key
        return str(resolved_key) == current_key

    def _drain_process_notifications(self, consumer: str) -> None:
        """Queue background notifications owned by this session (drained with our stable identity so another window can't claim them)."""
        from tools.process_registry import process_registry
        from tools.async_delegation import claim_event_delivery, complete_event_delivery

        for event, synthetic_message in process_registry.drain_notifications(
            session_key=getattr(self, "session_id", "") or "", owns_event=self._owns_process_notification,
        ):
            claim = claim_event_delivery(event, consumer)
            if claim is None:
                continue
            self._pending_input.put(synthetic_message)
            complete_event_delivery(event, claim)

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

    def _on_reasoning(self, reasoning_text: str):
        """Callback for intermediate reasoning display during tool-call loops."""
        if not reasoning_text:
            return
        self._reasoning_preview_buf = getattr(self, "_reasoning_preview_buf", "") + reasoning_text
        self._flush_reasoning_preview(force=False)

    # Inline tokens that bypass the destructive-slash confirmation modal (scripting, or
    # when the modal can't be marshaled onto the app loop).
    # A general escape hatch for non-interactive use (scripting/automation) and for the degraded path where
    # the modal can't be marshaled onto the app loop — lets users self-serve without flipping
    # approvals.destructive_slash_confirm in config. (Native Windows now drives the modal normally — see
    # #33961.)
    _DESTRUCTIVE_SKIP_TOKENS = frozenset({"now", "--yes", "-y"})

    def _tui_process_loop(self):
        """REPL worker thread: drain ``_pending_input``, run idle housekeeping, dispatch each input."""
        while not self._should_exit:
            try:
                try:
                    user_input = self._pending_input.get(timeout=0.1)
                except queue.Empty:
                    if not self._agent_running:
                        self._tui_idle_tick()
                    continue
                self._tui_process_one_input(user_input)
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
        ):
            with suppress(Exception):
                step()

    def _tui_unwrap_input(self, user_input):
        """Unwrap ``_VoiceInputMessage`` / ``_SeededQueryMessage`` -> ``(text_or_tuple, is_voice_input, is_seeded_query)``."""
        # Voice-transcribed messages arrive wrapped in a sentinel so only genuine STT output gets the voice
        # prefix (#65827).
        is_voice_input = isinstance(user_input, _VoiceInputMessage)
        if is_voice_input:
            user_input = user_input.text
        is_seeded_query = isinstance(user_input, _SeededQueryMessage)
        if is_seeded_query:
            user_input = (user_input.text, user_input.images) if user_input.images else user_input.text
        return user_input, is_voice_input, is_seeded_query

    def _tui_process_one_input(self, user_input):
        """Route one submitted input: file drop, /resume pick, ! shell, slash command, or a chat turn."""
        user_input, is_voice_input, is_seeded_query = self._tui_unwrap_input(user_input)
        if not user_input:
            return
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
        self._print_user_message_preview(user_input)

        if submit_images:
            n = len(submit_images)
            _cprint(f"  {_DIM}📎 {n} image{'s' if n > 1 else ''} attached{_RST}")

        self._agent_running = self._interactive_turn = True
        self._pet_turn_error = self._pet_reasoning = False
        self._turn_summary_begin()
        self._app.invalidate()
        try:
            self.chat(user_input, images=submit_images or None, voice_input=is_voice_input)
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

    def _tui_print_startup(self):
        """Startup output: light-mode probe, banner, advisories, resume/welcome lines, tips."""
        with suppress(Exception):  # light-mode probe before pt grabs the tty (cached)
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

    def _tui_build_application(self, layout, kb, style):
        """Construct the prompt_toolkit Application for the REPL."""
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
        for _unset in (set_sudo_password_callback, set_approval_callback, set_secret_capture_callback):
            _unset(None)
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


def _run_quiet_single_query(cli, effective_query):
    """Quiet (-Q) one-shot turn: run, print the response (stderr for errors/session_id), then sys.exit with the automation exit code."""
    try:
        result = cli.agent.run_conversation(user_message=effective_query, conversation_history=cli.conversation_history)
    except KeyboardInterrupt:
        _emit_interrupted_session_end(cli, reason="keyboard_interrupt")
        print(f"\nsession_id: {cli.session_id}", file=sys.stderr)
        sys.exit(130)
    # The exit line below reports session_id to stderr for automation wrappers;
    # without this sync it would point at the ended parent after compression.
    _sync_cli_session_id_from_agent(cli)
    response = result.get("final_response", "") if isinstance(result, dict) else str(result)
    # Surface backend errors that produced no visible output (e.g. invalid model slug
    # -> provider 4xx) on stderr so piped stdout stays clean.
    if (
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

    if parsed_skills:
        # Load the skill payloads in the background: skill_view walks the full skills
        # tree per skill (~0.5s for a large library) and the result is only consumed
        # at agent init, not by the banner. finalize_preloaded_skills() joins the
        # thread before any consumer reads cli.system_prompt.
        def _load_preloaded_skills() -> None:
            try:
                cli._preload_skills_result = build_preloaded_skills_prompt(parsed_skills, task_id=cli.session_id)
            except Exception as exc:  # surfaced by finalize
                cli._preload_skills_error = exc

        cli._preload_skills_requested = parsed_skills
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


def _run_single_query_mode(cli, query, image, quiet, oneshot):
    """``-q``/``--image`` entry: seed an interactive session on a TTY, else run the one-shot turn and exit."""
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
                    _run_quiet_single_query(cli, effective_query)

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
    cli = _build_cli_from_args(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget,
                               verbose, compact, resume, checkpoints, pass_session_id, ignore_rules, skills)

    # Join the background worktree creation before anything consumes TERMINAL_CWD.
    # A requested worktree whose setup failed aborts: never silently run without isolation.
    wt_info = _join_worktree() if _join_worktree is not None else None
    if _join_worktree is not None and not wt_info:
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
        _run_single_query_mode(cli, query, image, quiet, oneshot)
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
