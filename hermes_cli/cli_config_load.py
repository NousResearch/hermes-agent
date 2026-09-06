"""Classic-CLI config loading: prefill messages, reasoning/service-tier parsing, terminal env mirroring, CLI defaults + user YAML merge and the logging/display bootstrap.

Split out of ``cli.py``; ``cli`` re-exports every public name and moved bodies late-bind
cli-level names through ``from cli import ...`` at call time so facade monkeypatch seams hold.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from utils import fast_safe_load

# Log-record parity with the origin module.
logger = logging.getLogger("cli")


def _cli():
    """Late import of the ``cli`` facade: mutable CLI module state (and its test seams) lives there."""
    import cli

    return cli


def _load_prefill_messages(file_path: str) -> List[Dict[str, Any]]:
    """Load prefill messages (JSON array) from *file_path*; relative to ~/.hermes/; missing/empty -> []."""
    from cli import _hermes_home
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
        "docker_snap_compat", "docker_zero_cap",
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
    from cli import _AUXILIARY_TASK_ENV, _CWD_PLACEHOLDERS, _TERMINAL_ENV_MAPPINGS
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
    from cli import _cli_config_defaults, _hermes_home, _merge_file_config, _mirror_config_to_env
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


def _init_logging_and_display_from_config() -> None:
    """Best-effort startup side effects: logging, config warnings, skin, display knobs."""
    from importlib import import_module as _im

    def _display(key, default):
        return _cli().CLI_CONFIG.get("display", {}).get(key, default)

    for step in (
        lambda: _im("hermes_logging").setup_logging(mode="cli"),
        lambda: _im("hermes_cli.config").print_config_warnings(),
        lambda: _im("hermes_cli.skin_engine").init_skin_from_config(_cli().CLI_CONFIG),
        lambda: _im("agent.display").set_tool_preview_max_len(int(_display("tool_preview_length", 0) or 0)),
        lambda: _im("agent.display").set_friendly_tool_labels(bool(_display("friendly_tool_labels", True))),
    ):
        try:
            step()
        except Exception:
            pass
