"""
Configuration management for Hermes Agent.

Config files are stored in ~/.hermes/ for easy access:
- ~/.hermes/config.yaml  - All settings (model, toolsets, terminal, etc.)
- ~/.hermes/.env         - API keys and secrets

This module provides:
- hermes config          - Show current configuration
- hermes config edit     - Open config in editor
- hermes config set      - Set a specific value
- hermes config wizard   - Re-run setup wizard
"""

import copy
import difflib
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


_IS_WINDOWS = platform.system() == "Windows"
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LAST_EXPANDED_CONFIG_BY_PATH: Dict[str, Any] = {}
# Env var names written to .env that aren't in OPTIONAL_ENV_VARS
# (managed by setup/provider flows directly).
_EXTRA_ENV_KEYS = frozenset({
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    "DISCORD_HOME_CHANNEL", "TELEGRAM_HOME_CHANNEL",
    "SIGNAL_ACCOUNT", "SIGNAL_HTTP_URL",
    "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
    "DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET",
    "FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_ENCRYPT_KEY", "FEISHU_VERIFICATION_TOKEN",
    "WECOM_BOT_ID", "WECOM_SECRET",
    "WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET", "WECOM_CALLBACK_AGENT_ID",
    "WECOM_CALLBACK_TOKEN", "WECOM_CALLBACK_ENCODING_AES_KEY",
    "WECOM_CALLBACK_HOST", "WECOM_CALLBACK_PORT",
    "WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN", "WEIXIN_BASE_URL", "WEIXIN_CDN_BASE_URL",
    "WEIXIN_HOME_CHANNEL", "WEIXIN_HOME_CHANNEL_NAME", "WEIXIN_DM_POLICY", "WEIXIN_GROUP_POLICY",
    "WEIXIN_ALLOWED_USERS", "WEIXIN_GROUP_ALLOWED_USERS", "WEIXIN_ALLOW_ALL_USERS",
    "BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD",
    "QQ_APP_ID", "QQ_CLIENT_SECRET", "QQBOT_HOME_CHANNEL", "QQBOT_HOME_CHANNEL_NAME",
    "QQ_HOME_CHANNEL", "QQ_HOME_CHANNEL_NAME",  # legacy aliases (pre-rename, still read for back-compat)
    "QQ_ALLOWED_USERS", "QQ_GROUP_ALLOWED_USERS", "QQ_ALLOW_ALL_USERS", "QQ_MARKDOWN_SUPPORT",
    "QQ_STT_API_KEY", "QQ_STT_BASE_URL", "QQ_STT_MODEL",
    "TERMINAL_ENV", "TERMINAL_SSH_KEY", "TERMINAL_SSH_PORT",
    "WHATSAPP_MODE", "WHATSAPP_ENABLED",
    "MATTERMOST_HOME_CHANNEL", "MATTERMOST_REPLY_MODE",
    "MATRIX_PASSWORD", "MATRIX_ENCRYPTION", "MATRIX_DEVICE_ID", "MATRIX_HOME_ROOM",
    "MATRIX_REQUIRE_MENTION", "MATRIX_FREE_RESPONSE_ROOMS", "MATRIX_AUTO_THREAD",
    "MATRIX_RECOVERY_KEY",
})
import yaml

from agent.archetypes import (
    DEFAULT_ARCHETYPE_NAME,
    REQUIRED_ARCHETYPE_FIELDS,
    get_tool_restrictions,
    resolve_archetype,
    resolve_archetype_defaults,
    resolve_specialist_mapping,
)
from agent.route_categories import (
    BUILTIN_LITERAL_CATEGORIES,
    BUILTIN_ROUTE_CATEGORIES,
    DEFAULT_LITERAL_CATEGORY,
    DEFAULT_ROUTE_CATEGORY,
    resolve_literal_category,
)
from agent.runtime_modes import (
    BUILTIN_RUNTIME_MODES,
    DEFAULT_RUNTIME_MODE_NAME,
    resolve_runtime_mode,
)
from agent.task_contracts import REQUIRED_TASK_CONTRACT_FIELDS, validate_task_contract
from hermes_cli.colors import Colors, color
from hermes_cli.default_soul import DEFAULT_SOUL_MD


DEFAULT_OMO_AGENTS = {
    "sisyphus": {
        "archetype": "generalist",
        "specialist": "builder",
        "mode": "primary",
        "color": "blue",
        "aliases": ["orchestrator"],
    },
    "hephaestus": {
        "archetype": "implementer",
        "route_category": "deep",
        "mode": "subagent",
        "color": "red",
        "aliases": ["deep_worker"],
    },
    "oracle": {
        "archetype": "researcher",
        "blocked_tools": ["write_file", "patch", "terminal", "execute_code", "delegate_task", "task"],
        "mode": "subagent",
    },
    "librarian": {
        "archetype": "researcher",
        "mode": "subagent",
        "blocked_tools": ["write_file", "patch", "terminal", "execute_code", "delegate_task", "task"],
    },
    "explore": {
        "archetype": "researcher",
        "specialist": "investigator",
        "mode": "subagent",
        "blocked_tools": ["write_file", "patch", "execute_code", "delegate_task"],
        "aliases": ["explorer"],
    },
    "multimodal-looker": {
        "archetype": "generalist",
        "specialist": "multimodal_specialist",
        "allowed_tools": [
            "read_file",
            "search_files",
            "vision_analyze",
            "browser_vision",
            "browser_snapshot",
            "browser_get_images",
        ],
        "mode": "subagent",
        "aliases": ["looker"],
    },
    "prometheus": {
        "archetype": "generalist",
        "specialist": "planner",
        "mode": "subagent",
        "color": "green",
        "aliases": ["planner"],
    },
    "metis": {
        "archetype": "verifier",
        "mode": "subagent",
        "aliases": ["gap_analyzer"],
    },
    "momus": {
        "archetype": "verifier",
        "mode": "subagent",
        "blocked_tools": ["write_file", "patch", "terminal", "execute_code", "delegate_task", "task"],
        "aliases": ["critic"],
    },
    "atlas": {
        "archetype": "generalist",
        "specialist": "planner",
        "mode": "subagent",
        "blocked_tools": ["delegate_task", "task"],
    },
    "sisyphus-junior": {
        "archetype": "implementer",
        "mode": "subagent",
        "aliases": ["executor"],
    },
}


# =============================================================================
# Managed mode (NixOS declarative config)
# =============================================================================

_MANAGED_TRUE_VALUES = ("true", "1", "yes")
_MANAGED_SYSTEM_NAMES = {
    "brew": "Homebrew",
    "homebrew": "Homebrew",
    "nix": "NixOS",
    "nixos": "NixOS",
}


def get_managed_system() -> Optional[str]:
    """Return the package manager owning this install, if any."""
    raw = os.getenv("HERMES_MANAGED", "").strip()
    if raw:
        normalized = raw.lower()
        if normalized in _MANAGED_TRUE_VALUES:
            return "NixOS"
        return _MANAGED_SYSTEM_NAMES.get(normalized, raw)

    managed_marker = get_hermes_home() / ".managed"
    if managed_marker.exists():
        return "NixOS"
    return None


def is_managed() -> bool:
    """Check if Hermes is running in package-manager-managed mode.

    Two signals: the HERMES_MANAGED env var (set by the systemd service),
    or a .managed marker file in HERMES_HOME (set by the NixOS activation
    script, so interactive shells also see it).
    """
    return get_managed_system() is not None


def get_managed_update_command() -> Optional[str]:
    """Return the preferred upgrade command for a managed install."""
    managed_system = get_managed_system()
    if managed_system == "Homebrew":
        return "brew upgrade hermes-agent"
    if managed_system == "NixOS":
        return "sudo nixos-rebuild switch"
    return None


def recommended_update_command() -> str:
    """Return the best update command for the current installation."""
    return get_managed_update_command() or "hermes update"


def format_managed_message(action: str = "modify this Hermes installation") -> str:
    """Build a user-facing error for managed installs."""
    managed_system = get_managed_system() or "a package manager"
    raw = os.getenv("HERMES_MANAGED", "").strip().lower()

    if managed_system == "NixOS":
        env_hint = "true" if raw in _MANAGED_TRUE_VALUES else raw or "true"
        return (
            f"Cannot {action}: this Hermes installation is managed by NixOS "
            f"(HERMES_MANAGED={env_hint}).\n"
            "Edit services.hermes-agent.settings in your configuration.nix and run:\n"
            "  sudo nixos-rebuild switch"
        )

    if managed_system == "Homebrew":
        env_hint = raw or "homebrew"
        return (
            f"Cannot {action}: this Hermes installation is managed by Homebrew "
            f"(HERMES_MANAGED={env_hint}).\n"
            "Use:\n"
            "  brew upgrade hermes-agent"
        )

    return (
        f"Cannot {action}: this Hermes installation is managed by {managed_system}.\n"
        "Use your package manager to upgrade or reinstall Hermes."
    )

def managed_error(action: str = "modify configuration"):
    """Print user-friendly error for managed mode."""
    print(format_managed_message(action), file=sys.stderr)


# =============================================================================
# Container-aware CLI (NixOS container mode)
# =============================================================================

def get_container_exec_info() -> Optional[dict]:
    """Read container mode metadata from HERMES_HOME/.container-mode.

    Returns a dict with keys: backend, container_name, exec_user, hermes_bin
    or None if container mode is not active, we're already inside the
    container, or HERMES_DEV=1 is set.

    The .container-mode file is written by the NixOS activation script when
    container.enable = true. It tells the host CLI to exec into the container
    instead of running locally.
    """
    if os.environ.get("HERMES_DEV") == "1":
        return None

    from hermes_constants import is_container
    if is_container():
        return None

    container_mode_file = get_hermes_home() / ".container-mode"

    try:
        info = {}
        with open(container_mode_file, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    info[key.strip()] = value.strip()
    except FileNotFoundError:
        return None
    # All other exceptions (PermissionError, malformed data, etc.) propagate

    backend = info.get("backend", "docker")
    container_name = info.get("container_name", "hermes-agent")
    exec_user = info.get("exec_user", "hermes")
    hermes_bin = info.get("hermes_bin", "/data/current-package/bin/hermes")

    return {
        "backend": backend,
        "container_name": container_name,
        "exec_user": exec_user,
        "hermes_bin": hermes_bin,
    }


# =============================================================================
# Config paths
# =============================================================================

# Re-export from hermes_constants — canonical definition lives there.
from hermes_constants import get_default_hermes_root, get_hermes_home  # noqa: F811,E402

def get_config_path() -> Path:
    """Get the main config file path."""
    return get_hermes_home() / "config.yaml"

def get_env_path() -> Path:
    """Get the .env file path (for API keys)."""
    return get_hermes_home() / ".env"

def get_project_root() -> Path:
    """Get the project installation directory."""
    return Path(__file__).parent.parent.resolve()


def _normalize_profile_local_path(value: Any) -> Any:
    """Map legacy in-container profile paths onto the local Hermes root when safe.

    Older profile files sometimes persisted paths like
    ``/root/.hermes/profiles/<name>/workspace``. On a host install, the same
    profile lives under the local Hermes root (for example
    ``/Users/alice/.hermes/profiles/<name>/workspace``). Only rewrite when the
    equivalent local target already exists, so we don't silently point users at
    a guessed path.
    """
    if not isinstance(value, str):
        return value

    raw = value.strip()
    if not raw.startswith("/root/.hermes/"):
        return raw

    try:
        relative = Path(raw).relative_to(Path("/root/.hermes"))
    except ValueError:
        return raw

    candidate = get_default_hermes_root() / relative
    if candidate.exists() or candidate.parent.exists():
        return str(candidate)
    return raw


def _normalize_profile_path_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize known profile-local path settings without touching unrelated keys."""
    normalized = dict(config)

    terminal_cfg = normalized.get("terminal")
    if isinstance(terminal_cfg, dict) and "cwd" in terminal_cfg:
        updated_terminal = dict(terminal_cfg)
        updated_terminal["cwd"] = _normalize_profile_local_path(updated_terminal.get("cwd"))
        normalized["terminal"] = updated_terminal

    skills_cfg = normalized.get("skills")
    if isinstance(skills_cfg, dict):
        external_dirs = skills_cfg.get("external_dirs")
        if isinstance(external_dirs, list):
            updated_skills = dict(skills_cfg)
            updated_skills["external_dirs"] = [
                _normalize_profile_local_path(item) for item in external_dirs
            ]
            normalized["skills"] = updated_skills

    return normalized

def _secure_dir(path):
    """Set directory to owner-only access (0700 by default). No-op on Windows.

    Skipped in managed mode — the NixOS module sets group-readable
    permissions (0750) so interactive users in the hermes group can
    share state with the gateway service.

    The mode can be overridden via the HERMES_HOME_MODE environment variable
    (e.g. HERMES_HOME_MODE=0701) for deployments where a web server (nginx,
    caddy, etc.) needs to traverse HERMES_HOME to reach a served subdirectory.
    The execute-only bit on a directory permits cd-through without exposing
    directory listings.
    """
    if is_managed():
        return
    try:
        mode_str = os.environ.get("HERMES_HOME_MODE", "").strip()
        mode = int(mode_str, 8) if mode_str else 0o700
    except ValueError:
        mode = 0o700
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass


def _is_container() -> bool:
    """Detect if we're running inside a Docker/Podman/LXC container.

    When Hermes runs in a container with volume-mounted config files, forcing
    0o600 permissions breaks multi-process setups where the gateway and
    dashboard run as different UIDs or the volume mount requires broader
    permissions.
    """
    # Explicit opt-out
    if os.environ.get("HERMES_CONTAINER") or os.environ.get("HERMES_SKIP_CHMOD"):
        return True
    # Docker / Podman marker file
    if os.path.exists("/.dockerenv"):
        return True
    # LXC / cgroup-based detection
    try:
        with open("/proc/1/cgroup", "r") as f:
            cgroup_content = f.read()
        if "docker" in cgroup_content or "lxc" in cgroup_content or "kubepods" in cgroup_content:
            return True
    except (OSError, IOError):
        pass
    return False


def _secure_file(path):
    """Set file to owner-only read/write (0600). No-op on Windows.

    Skipped in managed mode — the NixOS activation script sets
    group-readable permissions (0640) on config files.

    Skipped in containers — Docker/Podman volume mounts often need broader
    permissions.  Set HERMES_SKIP_CHMOD=1 to force-skip on other systems.
    """
    if is_managed() or _is_container():
        return
    try:
        if os.path.exists(str(path)):
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _ensure_default_soul_md(home: Path) -> None:
    """Seed a default SOUL.md into HERMES_HOME if the user doesn't have one yet."""
    soul_path = home / "SOUL.md"
    if soul_path.exists():
        return
    soul_path.write_text(DEFAULT_SOUL_MD, encoding="utf-8")
    _secure_file(soul_path)


def ensure_hermes_home():
    """Ensure ~/.hermes directory structure exists with secure permissions.

    In managed mode (NixOS), dirs are created by the activation script with
    setgid + group-writable (2770). We skip mkdir and set umask(0o007) so
    any files created (e.g. SOUL.md) are group-writable (0660).
    """
    home = get_hermes_home()
    if is_managed():
        old_umask = os.umask(0o007)
        try:
            _ensure_hermes_home_managed(home)
        finally:
            os.umask(old_umask)
    else:
        home.mkdir(parents=True, exist_ok=True)
        _secure_dir(home)
        for subdir in ("cron", "sessions", "logs", "memories"):
            d = home / subdir
            d.mkdir(parents=True, exist_ok=True)
            _secure_dir(d)
        _ensure_default_soul_md(home)


def _ensure_hermes_home_managed(home: Path):
    """Managed-mode variant: verify dirs exist (activation creates them), seed SOUL.md."""
    if not home.is_dir():
        raise RuntimeError(
            f"HERMES_HOME {home} does not exist. "
            "Run 'sudo nixos-rebuild switch' first."
        )
    for subdir in ("cron", "sessions", "logs", "memories"):
        d = home / subdir
        if not d.is_dir():
            raise RuntimeError(
                f"{d} does not exist. "
                "Run 'sudo nixos-rebuild switch' first."
            )
    # Inside umask(0o007) scope — SOUL.md will be created as 0660
    _ensure_default_soul_md(home)


# =============================================================================
# Config loading/saving
# =============================================================================

DEFAULT_CONFIG = {
    "model": "",
    "providers": {},
    "fallback_providers": [],
    "credential_pool_strategies": {},
    "agents": copy.deepcopy(DEFAULT_OMO_AGENTS),
    "categories": {},
    "runtime_fallback": {"enabled": False},
    "model_capabilities": {},
    "toolsets": ["hermes-cli"],
    "agent": {
        "max_turns": 90,
        # Inactivity timeout for gateway agent execution (seconds).
        # The agent can run indefinitely as long as it's actively calling
        # tools or receiving API responses.  Only fires when the agent has
        # been completely idle for this duration.  0 = unlimited.
        "gateway_timeout": 1800,
        # Graceful drain timeout for gateway stop/restart (seconds).
        # The gateway stops accepting new work, waits for running agents
        # to finish, then interrupts any remaining runs after the timeout.
        # 0 = no drain, interrupt immediately.
        "restart_drain_timeout": 60,
        "service_tier": "",
        # Tool-use enforcement: injects system prompt guidance that tells the
        # model to actually call tools instead of describing intended actions.
        # Values: "auto" (default — applies to gpt/codex models), true/false
        # (force on/off for all models), or a list of model-name substrings
        # to match (e.g. ["gpt", "codex", "gemini", "qwen"]).
        "tool_use_enforcement": "auto",
        # Staged inactivity warning: send a warning to the user at this
        # threshold before escalating to a full timeout.  The warning fires
        # once per run and does not interrupt the agent.  0 = disable warning.
        "gateway_timeout_warning": 900,
        # Periodic "still working" notification interval (seconds).
        # Sends a status message every N seconds so the user knows the
        # agent hasn't died during long tasks.  0 = disable notifications.
        "gateway_notify_interval": 600,
    },
    
    "terminal": {
        "backend": "local",
        "modal_mode": "auto",
        "cwd": ".",  # Use current directory
        "timeout": 180,
        # Environment variables to pass through to sandboxed execution
        # (terminal and execute_code).  Skill-declared required_environment_variables
        # are passed through automatically; this list is for non-skill use cases.
        "env_passthrough": [],
        "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "docker_forward_env": [],
        # Explicit environment variables to set inside Docker containers.
        # Unlike docker_forward_env (which reads values from the host process),
        # docker_env lets you specify exact key-value pairs — useful when Hermes
        # runs as a systemd service without access to the user's shell environment.
        # Example: {"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock"}
        "docker_env": {},
        "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
        "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        # Container resource limits (docker, singularity, modal, daytona — ignored for local/ssh)
        "container_cpu": 1,
        "container_memory": 5120,       # MB (default 5GB)
        "container_disk": 51200,        # MB (default 50GB)
        "container_persistent": True,   # Persist filesystem across sessions
        # Docker volume mounts — share host directories with the container.
        # Each entry is "host_path:container_path" (standard Docker -v syntax).
        # Example: ["/home/user/projects:/workspace/projects", "/data:/data"]
        "docker_volumes": [],
        # Explicit opt-in: mount the host cwd into /workspace for Docker sessions.
        # Default off because passing host directories into a sandbox weakens isolation.
        "docker_mount_cwd_to_workspace": False,
        # Persistent shell — keep a long-lived bash shell across execute() calls
        # so cwd/env vars/shell variables survive between commands.
        # Enabled by default for non-local backends (SSH); local is always opt-in
        # via TERMINAL_LOCAL_PERSISTENT env var.
        "persistent_shell": True,
    },
    
    "browser": {
        "inactivity_timeout": 120,
        "command_timeout": 30,  # Timeout for browser commands in seconds (screenshot, navigate, etc.)
        "record_sessions": False,  # Auto-record browser sessions as WebM videos
        "allow_private_urls": False,  # Allow navigating to private/internal IPs (localhost, 192.168.x.x, etc.)
        "cdp_url": "",  # Optional persistent CDP endpoint for attaching to an existing Chromium/Chrome
        "camofox": {
            # When true, Hermes sends a stable profile-scoped userId to Camofox
            # so the server maps it to a persistent Firefox profile automatically.
            # When false (default), each session gets a random userId (ephemeral).
            "managed_persistence": False,
        },
    },

    # Filesystem checkpoints — automatic snapshots before destructive file ops.
    # When enabled, the agent takes a snapshot of the working directory once per
    # conversation turn (on first write_file/patch call).  Use /rollback to restore.
    "checkpoints": {
        "enabled": True,
        "max_snapshots": 50,  # Max checkpoints to keep per directory
    },

    # Maximum characters returned by a single read_file call.  Reads that
    # exceed this are rejected with guidance to use offset+limit.
    # 100K chars ≈ 25–35K tokens across typical tokenisers.
    "file_read_max_chars": 100_000,
    
    "compression": {
        "enabled": True,
        "threshold": 0.50,            # compress when context usage exceeds this ratio
        "target_ratio": 0.20,         # fraction of threshold to preserve as recent tail
        "protect_last_n": 20,         # minimum recent messages to keep uncompressed
        "protected_tools": [
            "task",
            "todoread",
            "todowrite",
            "lsp_rename",
            "session_search",
            "session_read",
            "session_write",
        ],

    },

    # AWS Bedrock provider configuration.
    # Only used when model.provider is "bedrock".
    "bedrock": {
        "region": "",  # AWS region for Bedrock API calls (empty = AWS_REGION env var → us-east-1)
        "discovery": {
            "enabled": True,           # Auto-discover models via ListFoundationModels
            "provider_filter": [],     # Only show models from these providers (e.g. ["anthropic", "amazon"])
            "refresh_interval": 3600,  # Cache discovery results for this many seconds
        },
        "guardrail": {
            # Amazon Bedrock Guardrails — content filtering and safety policies.
            # Create a guardrail in the Bedrock console, then set the ID and version here.
            # See: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
            "guardrail_identifier": "",  # e.g. "abc123def456"
            "guardrail_version": "",     # e.g. "1" or "DRAFT"
            "stream_processing_mode": "async",  # "sync" or "async"
            "trace": "disabled",         # "enabled", "disabled", or "enabled_full"
        },
    },

    "smart_model_routing": {
        "enabled": False,
        "max_simple_chars": 160,
        "max_simple_words": 28,
        "cheap_model": {},
    },
    
    # Auxiliary model config — provider:model for each side task.
    # Format: provider is the provider name, model is the model slug.
    # "auto" for provider = auto-detect best available provider.
    # Empty model = use provider's default auxiliary model.
    # All tasks fall back to openrouter:google/gemini-3-flash-preview if
    # the configured provider is unavailable.
    "auxiliary": {
        "vision": {
            "provider": "auto",    # auto | openrouter | nous | codex | custom
            "model": "",           # e.g. "google/gemini-2.5-flash", "gpt-4o"
            "base_url": "",        # direct OpenAI-compatible endpoint (takes precedence over provider)
            "api_key": "",         # API key for base_url (falls back to OPENAI_API_KEY)
            "timeout": 120,        # seconds — LLM API call timeout; vision payloads need generous timeout
            "download_timeout": 30,  # seconds — image HTTP download timeout; increase for slow connections
        },
        "web_extract": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 360,        # seconds (6min) — per-attempt LLM summarization timeout; increase for slow local models
        },
        "compression": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,        # seconds — compression summarises large contexts; increase for local models
        },
        "session_search": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
        },
        "skills_hub": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
        },
        "approval": {
            "provider": "auto",
            "model": "",           # fast/cheap model recommended (e.g. gemini-flash, haiku)
            "base_url": "",
            "api_key": "",
            "timeout": 30,
        },
        "mcp": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
        },
        "flush_memories": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
        },
        "title_generation": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
        },
    },
    
    "display": {
        "compact": False,
        "personality": "kawaii",
        "resume_display": "full",
        "busy_input_mode": "interrupt",
        "bell_on_complete": False,
        "show_reasoning": False,
        "streaming": False,
        "inline_diffs": True,     # Show inline diff previews for write actions (write_file, patch, skill_manage)
        "show_cost": False,       # Show $ cost in the status bar (off by default)
        "skin": "default",
        "interim_assistant_messages": True,  # Gateway: show natural mid-turn assistant status messages
        "tool_progress_command": False,  # Enable /verbose command in messaging gateway
        "tool_progress_overrides": {},  # DEPRECATED — use display.platforms instead
        "tool_preview_length": 0,  # Max chars for tool call previews (0 = no limit, show full paths/commands)
        "platforms": {},  # Per-platform display overrides: {"telegram": {"tool_progress": "all"}, "slack": {"tool_progress": "off"}}
    },

    # Web dashboard settings
    "dashboard": {
        "theme": "default",  # Dashboard visual theme: "default", "midnight", "ember", "mono", "cyberpunk", "rose"
    },

    # Privacy settings
    "privacy": {
        "redact_pii": False,  # When True, hash user IDs and strip phone numbers from LLM context
    },
    
    # Text-to-speech configuration
    "tts": {
        "provider": "edge",  # "edge" (free) | "elevenlabs" (premium) | "openai" | "xai" | "minimax" | "mistral" | "neutts" (local)
        "edge": {
            "voice": "en-US-AriaNeural",
            # Popular: AriaNeural, JennyNeural, AndrewNeural, BrianNeural, SoniaNeural
        },
        "elevenlabs": {
            "voice_id": "pNInz6obpgDQGcFmaJgB",  # Adam
            "model_id": "eleven_multilingual_v2",
        },
        "openai": {
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
            # Voices: alloy, echo, fable, onyx, nova, shimmer
        },
        "xai": {
            "voice_id": "eve",
            "language": "en",
            "sample_rate": 24000,
            "bit_rate": 128000,
        },
        "mistral": {
            "model": "voxtral-mini-tts-2603",
            "voice_id": "c69964a6-ab8b-4f8a-9465-ec0925096ec8",  # Paul - Neutral
        },
        "neutts": {
            "ref_audio": "",  # Path to reference voice audio (empty = bundled default)
            "ref_text": "",   # Path to reference voice transcript (empty = bundled default)
            "model": "neuphonic/neutts-air-q4-gguf",  # HuggingFace model repo
            "device": "cpu",  # cpu, cuda, or mps
        },
    },
    
    "stt": {
        "enabled": True,
        "provider": "local",  # "local" (free, faster-whisper) | "groq" | "openai" (Whisper API) | "mistral" (Voxtral Transcribe)
        "local": {
            "model": "base",  # tiny, base, small, medium, large-v3
            "language": "",  # auto-detect by default; set to "en", "es", "fr", etc. to force
        },
        "openai": {
            "model": "whisper-1",  # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe
        },
        "mistral": {
            "model": "voxtral-mini-latest",  # voxtral-mini-latest, voxtral-mini-2602
        },
    },

    "voice": {
        "record_key": "ctrl+b",
        "max_recording_seconds": 120,
        "auto_tts": False,
        "silence_threshold": 200,     # RMS below this = silence (0-32767)
        "silence_duration": 3.0,      # Seconds of silence before auto-stop
    },
    
    "human_delay": {
        "mode": "off",
        "min_ms": 800,
        "max_ms": 2500,
    },
    
    # Context engine -- controls how the context window is managed when
    # approaching the model's token limit.
    # "compressor" = built-in lossy summarization (default).
    # Set to a plugin name to activate an alternative engine (e.g. "lcm"
    # for Lossless Context Management).  The engine must be installed as
    # a plugin in plugins/context_engine/<name>/ or ~/.hermes/plugins/.
    "context": {
        "engine": "compressor",
    },

    # Persistent memory -- bounded curated memory injected into system prompt
    "memory": {
        "memory_enabled": True,
        "user_profile_enabled": True,
        "memory_char_limit": 2200,   # ~800 tokens at 2.75 chars/token
        "user_char_limit": 1375,     # ~500 tokens at 2.75 chars/token
        # External memory provider plugin (empty = built-in only).
        # Set to a provider name to activate: "openviking", "mem0",
        # "hindsight", "holographic", "retaindb", "byterover".
        # Only ONE external provider is allowed at a time.
        "provider": "",
    },

    # Subagent delegation — override the provider:model used by delegate_task
    # so child agents can run on a different (cheaper/faster) provider and model.
    # Uses the same runtime provider resolution as CLI/gateway startup, so all
    # configured providers (OpenRouter, Nous, Z.ai, Kimi, etc.) are supported.
    # delegation_profiles is the primary Wave 1 config surface for delegation
    # policy. DG2 adds category as a bounded literal upstream product surface.
    # Route categories and runtime modes stay separate internal registries:
    # their metadata is still rendered into prompts and runtime state, while
    # literal category names map onto that substrate truthfully.
    "delegation": {
        "model": "",       # e.g. "google/gemini-3-flash-preview" (empty = inherit parent model)
        "provider": "",    # e.g. "openrouter" (empty = inherit parent provider + credentials)
        "base_url": "",    # direct OpenAI-compatible endpoint for subagents
        "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        "max_iterations": 50,  # per-subagent iteration cap (each subagent gets its own budget,
                               # independent of the parent's max_iterations)
        "max_concurrent_children": 3,
        "reasoning_effort": "",  # reasoning effort for subagents: "xhigh", "high", "medium",
                                 # "low", "minimal", "none" (empty = inherit parent's level)
        "auto_decompose_large_tasks": True,  # prompt the primary agent to make a todo list on large requests
        "auto_fanout_batch_mode": True,      # prefer one batched delegate_task call over many single-task calls
        "auto_fanout_max_tasks": 3,          # soft cap for one delegated batch after routing/truncation logic
        "default_category": DEFAULT_LITERAL_CATEGORY,
        # Wave 1 schema/default surfaces. These remain structurally distinct:
        # category is the literal upstream-facing product surface, route_category
        # is the mapped internal routing label, delegation_profile selects a child
        # policy profile, runtime_mode is an operating-mode concept, and
        # task_contract stays canonical structured data.
        "category": DEFAULT_LITERAL_CATEGORY,
        "archetype": DEFAULT_ARCHETYPE_NAME,
        "runtime_mode": DEFAULT_RUNTIME_MODE_NAME,
        "route_category": DEFAULT_ROUTE_CATEGORY,
        "default_route_category": DEFAULT_ROUTE_CATEGORY,
        "default_delegation_profile": "general",
        "default_skills": ["general_reasoning", "task_execution"],
        "default_required_tools": ["read_file", "search_files"],
        "permission_preset": "inherit",
        "fallback_policy": "legacy_default_mapping",
        "task_contract": None,
        "literal_categories": {
            name: {
                "summary": category.summary,
                "route_category": category.route_category,
                "default_runtime_mode": category.default_runtime_mode,
            }
            for name, category in BUILTIN_LITERAL_CATEGORIES.items()
        },  # built-in literal-category mapping metadata
        "route_categories": {
            name: {
                "summary": category.summary,
                "intensity": category.intensity,
            }
            for name, category in BUILTIN_ROUTE_CATEGORIES.items()
        },  # built-in route-category metadata overrides only
        "runtime_modes": {
            mode.name: {
                "description": mode.description,
                "operating_posture": mode.operating_posture,
                "kind": mode.kind,
            }
            for mode in BUILTIN_RUNTIME_MODES
        },  # built-in runtime-mode metadata overrides only
        "delegation_profiles": {
            "general": {
                "routing_description": "Default fallback for uncategorized delegated work.",
                "max_concurrent_children": 3,
            },
            "research": {
                "routing_description": "Audits, comparisons, investigation, and read-heavy research.",
                "max_concurrent_children": 3,
                "max_iterations": 25,
                "enabled_tools": [
                    "read_file",
                    "search_files",
                    "session_search",
                    "skills_list",
                    "skill_view",
                    "web_search",
                    "web_extract",
                    "browser_navigate",
                    "browser_snapshot",
                    "browser_console",
                    "browser_scroll",
                    "browser_get_images",
                    "vision_analyze",
                    "browser_vision",
                ],
            },
            "implementation": {
                "routing_description": "Code changes, patches, refactors, and other write-heavy implementation work.",
                "max_concurrent_children": 2,
                "max_iterations": 35,
                "toolsets": ["terminal", "file"],
            },
            "verification": {
                "routing_description": "Tests, validation, regression checks, and review passes.",
                "max_concurrent_children": 3,
                "max_iterations": 20,
                "toolsets": ["terminal", "file", "web"],
            },
        },
    },

    # Ephemeral prefill messages file — JSON list of {role, content} dicts
    # injected at the start of every API call for few-shot priming.
    # Never saved to sessions, logs, or trajectories.
    "prefill_messages_file": "",
    
    # Skills — external skill directories for sharing skills across tools/agents.
    # Each path is expanded (~, ${VAR}) and resolved.  Read-only — skill creation
    # always goes to ~/.hermes/skills/.
    "skills": {
        "external_dirs": [],   # e.g. ["~/.agents/skills", "/shared/team-skills"]
    },

    # Honcho AI-native memory -- reads ~/.honcho/config.json as single source of truth.
    # This section is only needed for hermes-specific overrides; everything else
    # (apiKey, workspace, peerName, sessions, enabled) comes from the global config.
    "honcho": {},

    # IANA timezone (e.g. "Asia/Kolkata", "America/New_York").
    # Empty string means use server-local time.
    "timezone": "",

    # Discord platform settings (gateway mode)
    "discord": {
        "require_mention": True,       # Require @mention to respond in server channels
        "free_response_channels": "",  # Comma-separated channel IDs where bot responds without mention
        "allowed_channels": "",        # If set, bot ONLY responds in these channel IDs (whitelist)
        "auto_thread": True,           # Auto-create threads on @mention in channels (like Slack)
        "reactions": True,             # Add 👀/✅/❌ reactions to messages during processing
        "channel_prompts": {},         # Per-channel ephemeral system prompts (forum parents apply to child threads)
    },

    # WhatsApp platform settings (gateway mode)
    "whatsapp": {
        # Reply prefix prepended to every outgoing WhatsApp message.
        # Default (None) uses the built-in "⚕ *Hermes Agent*" header.
        # Set to "" (empty string) to disable the header entirely.
        # Supports \n for newlines, e.g. "🤖 *My Bot*\n──────\n"
    },

    # Telegram platform settings (gateway mode)
    "telegram": {
        "channel_prompts": {},         # Per-chat/topic ephemeral system prompts (topics inherit from parent group)
    },

    # Slack platform settings (gateway mode)
    "slack": {
        "channel_prompts": {},         # Per-channel ephemeral system prompts
    },

    # Mattermost platform settings (gateway mode)
    "mattermost": {
        "channel_prompts": {},         # Per-channel ephemeral system prompts
    },

    # Approval mode for dangerous commands:
    #   manual — always prompt the user (default)
    #   smart  — use auxiliary LLM to auto-approve low-risk commands, prompt for high-risk
    #   off    — skip all approval prompts (equivalent to --yolo)
    "approvals": {
        "mode": "manual",
        "timeout": 60,
    },

    # Permanently allowed dangerous command patterns (added via "always" approval)
    "command_allowlist": [],
    # User-defined quick commands that bypass the agent loop (type: exec only)
    "quick_commands": {},
    # Custom personalities — add your own entries here
    # Supports string format: {"name": "system prompt"}
    # Or dict format: {"name": {"description": "...", "system_prompt": "...", "tone": "...", "style": "..."}}
    "personalities": {},

    # Pre-exec security scanning via tirith
    "security": {
        "redact_secrets": True,
        "tirith_enabled": True,
        "tirith_path": "tirith",
        "tirith_timeout": 5,
        "tirith_fail_open": True,
        "website_blocklist": {
            "enabled": False,
            "domains": [],
            "shared_files": [],
        },
    },

    "cron": {
        # Wrap delivered cron responses with a header (task name) and footer
        # ("The agent cannot see this message").  Set to false for clean output.
        "wrap_response": True,
    },

    # Logging — controls file logging to ~/.hermes/logs/.
    # agent.log captures INFO+ (all agent activity); errors.log captures WARNING+.
    "logging": {
        "level": "INFO",       # Minimum level for agent.log: DEBUG, INFO, WARNING
        "max_size_mb": 5,      # Max size per log file before rotation
        "backup_count": 3,     # Number of rotated backup files to keep
    },

    # Network settings — workarounds for connectivity issues.
    "network": {
        # Force IPv4 connections.  On servers with broken or unreachable IPv6,
        # Python tries AAAA records first and hangs for the full TCP timeout
        # before falling back to IPv4.  Set to true to skip IPv6 entirely.
        "force_ipv4": False,
    },

    # Config schema version - bump this when adding new required fields
    "_config_version": 20,
}

# =============================================================================
# Config Migration System
# =============================================================================

# Track which env vars were introduced in each config version.
# Migration only mentions vars new since the user's previous version.
ENV_VARS_BY_VERSION: Dict[int, List[str]] = {
    3: ["FIRECRAWL_API_KEY", "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID", "FAL_KEY"],
    4: ["VOICE_TOOLS_OPENAI_KEY", "ELEVENLABS_API_KEY"],
    5: ["WHATSAPP_ENABLED", "WHATSAPP_MODE", "WHATSAPP_ALLOWED_USERS",
        "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS"],
    10: ["TAVILY_API_KEY"],
    11: ["TERMINAL_MODAL_MODE"],
}

# Required environment variables with metadata for migration prompts.
# LLM provider is required but handled in the setup wizard's provider
# selection step (Nous Portal / OpenRouter / Custom endpoint), so this
# dict is intentionally empty — no single env var is universally required.
REQUIRED_ENV_VARS = {}

# Optional environment variables that enhance functionality
OPTIONAL_ENV_VARS = {
    # ── Provider (handled in provider selection, not shown in checklists) ──
    "NOUS_BASE_URL": {
        "description": "Nous Portal base URL override",
        "prompt": "Nous Portal base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENROUTER_API_KEY": {
        "description": "OpenRouter API key (for vision, web scraping helpers, and MoA)",
        "prompt": "OpenRouter API key",
        "url": "https://openrouter.ai/keys",
        "password": True,
        "tools": ["vision_analyze", "mixture_of_agents"],
        "category": "provider",
        "advanced": True,
    },
    "GOOGLE_API_KEY": {
        "description": "Google AI Studio API key (also recognized as GEMINI_API_KEY)",
        "prompt": "Google AI Studio API key",
        "url": "https://aistudio.google.com/app/apikey",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GEMINI_API_KEY": {
        "description": "Google AI Studio API key (alias for GOOGLE_API_KEY)",
        "prompt": "Gemini API key",
        "url": "https://aistudio.google.com/app/apikey",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GEMINI_BASE_URL": {
        "description": "Google AI Studio base URL override",
        "prompt": "Gemini base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "XAI_API_KEY": {
        "description": "xAI API key",
        "prompt": "xAI API key",
        "url": "https://console.x.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "XAI_BASE_URL": {
        "description": "xAI base URL override",
        "prompt": "xAI base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "NVIDIA_API_KEY": {
        "description": "NVIDIA NIM API key (build.nvidia.com or local NIM endpoint)",
        "prompt": "NVIDIA NIM API key",
        "url": "https://build.nvidia.com/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "NVIDIA_BASE_URL": {
        "description": "NVIDIA NIM base URL override (e.g. http://localhost:8000/v1 for local NIM)",
        "prompt": "NVIDIA NIM base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "GLM_API_KEY": {
        "description": "Z.AI / GLM API key (also recognized as ZAI_API_KEY / Z_AI_API_KEY)",
        "prompt": "Z.AI / GLM API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "ZAI_API_KEY": {
        "description": "Z.AI API key (alias for GLM_API_KEY)",
        "prompt": "Z.AI API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "Z_AI_API_KEY": {
        "description": "Z.AI API key (alias for GLM_API_KEY)",
        "prompt": "Z.AI API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GLM_BASE_URL": {
        "description": "Z.AI / GLM base URL override",
        "prompt": "Z.AI / GLM base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_API_KEY": {
        "description": "Kimi / Moonshot API key",
        "prompt": "Kimi API key",
        "url": "https://platform.moonshot.cn/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_BASE_URL": {
        "description": "Kimi / Moonshot base URL override",
        "prompt": "Kimi base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_CN_API_KEY": {
        "description": "Kimi / Moonshot China API key",
        "prompt": "Kimi (China) API key",
        "url": "https://platform.moonshot.cn/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "ARCEEAI_API_KEY": {
        "description": "Arcee AI API key",
        "prompt": "Arcee AI API key",
        "url": "https://chat.arcee.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "ARCEE_BASE_URL": {
        "description": "Arcee AI base URL override",
        "prompt": "Arcee base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_API_KEY": {
        "description": "MiniMax API key (international)",
        "prompt": "MiniMax API key",
        "url": "https://www.minimax.io/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_BASE_URL": {
        "description": "MiniMax base URL override",
        "prompt": "MiniMax base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_CN_API_KEY": {
        "description": "MiniMax API key (China endpoint)",
        "prompt": "MiniMax (China) API key",
        "url": "https://www.minimaxi.com/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_CN_BASE_URL": {
        "description": "MiniMax (China) base URL override",
        "prompt": "MiniMax (China) base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "DEEPSEEK_API_KEY": {
        "description": "DeepSeek API key for direct DeepSeek access",
        "prompt": "DeepSeek API Key",
        "url": "https://platform.deepseek.com/api_keys",
        "password": True,
        "category": "provider",
    },
    "DEEPSEEK_BASE_URL": {
        "description": "Custom DeepSeek API base URL (advanced)",
        "prompt": "DeepSeek Base URL",
        "url": "",
        "password": False,
        "category": "provider",
    },
    "DASHSCOPE_API_KEY": {
        "description": "Alibaba Cloud DashScope API key (Qwen + multi-provider models)",
        "prompt": "DashScope API Key",
        "url": "https://modelstudio.console.alibabacloud.com/",
        "password": True,
        "category": "provider",
    },
    "DASHSCOPE_BASE_URL": {
        "description": "Custom DashScope base URL (default: coding-intl OpenAI-compat endpoint)",
        "prompt": "DashScope Base URL",
        "url": "",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "HERMES_QWEN_BASE_URL": {
        "description": "Qwen Portal base URL override (default: https://portal.qwen.ai/v1)",
        "prompt": "Qwen Portal base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "HERMES_GEMINI_CLIENT_ID": {
        "description": "Google OAuth client ID for google-gemini-cli (optional; defaults to Google's public gemini-cli client)",
        "prompt": "Google OAuth client ID (optional — leave empty to use the public default)",
        "url": "https://console.cloud.google.com/apis/credentials",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "HERMES_GEMINI_CLIENT_SECRET": {
        "description": "Google OAuth client secret for google-gemini-cli (optional)",
        "prompt": "Google OAuth client secret (optional)",
        "url": "https://console.cloud.google.com/apis/credentials",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "HERMES_GEMINI_PROJECT_ID": {
        "description": "GCP project ID for paid Gemini tiers (free tier auto-provisions)",
        "prompt": "GCP project ID for Gemini OAuth (leave empty for free tier)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_ZEN_API_KEY": {
        "description": "OpenCode Zen API key (pay-as-you-go access to curated models)",
        "prompt": "OpenCode Zen API key",
        "url": "https://opencode.ai/auth",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_ZEN_BASE_URL": {
        "description": "OpenCode Zen base URL override",
        "prompt": "OpenCode Zen base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_GO_API_KEY": {
        "description": "OpenCode Go API key ($10/month subscription for open models)",
        "prompt": "OpenCode Go API key",
        "url": "https://opencode.ai/auth",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_GO_BASE_URL": {
        "description": "OpenCode Go base URL override",
        "prompt": "OpenCode Go base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "HF_TOKEN": {
        "description": "Hugging Face token for Inference Providers (20+ open models via router.huggingface.co)",
        "prompt": "Hugging Face Token",
        "url": "https://huggingface.co/settings/tokens",
        "password": True,
        "category": "provider",
    },
    "HF_BASE_URL": {
        "description": "Hugging Face Inference Providers base URL override",
        "prompt": "HF base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OLLAMA_API_KEY": {
        "description": "Ollama Cloud API key (ollama.com — cloud-hosted open models)",
        "prompt": "Ollama Cloud API key",
        "url": "https://ollama.com/settings",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OLLAMA_BASE_URL": {
        "description": "Ollama Cloud base URL override (default: https://ollama.com/v1)",
        "prompt": "Ollama base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "XIAOMI_API_KEY": {
        "description": "Xiaomi MiMo API key for MiMo models (mimo-v2-pro, mimo-v2-omni, mimo-v2-flash)",
        "prompt": "Xiaomi MiMo API Key",
        "url": "https://platform.xiaomimimo.com",
        "password": True,
        "category": "provider",
    },
    "XIAOMI_BASE_URL": {
        "description": "Xiaomi MiMo base URL override (default: https://api.xiaomimimo.com/v1)",
        "prompt": "Xiaomi base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "AWS_REGION": {
        "description": "AWS region for Bedrock API calls (e.g. us-east-1, eu-central-1)",
        "prompt": "AWS Region",
        "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "AWS_PROFILE": {
        "description": "AWS named profile for Bedrock authentication (from ~/.aws/credentials)",
        "prompt": "AWS Profile",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },

    # ── Tool API keys ──
    "EXA_API_KEY": {
        "description": "Exa API key for AI-native web search and contents",
        "prompt": "Exa API key",
        "url": "https://exa.ai/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "PARALLEL_API_KEY": {
        "description": "Parallel API key for AI-native web search and extract",
        "prompt": "Parallel API key",
        "url": "https://parallel.ai/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_API_KEY": {
        "description": "Firecrawl API key for web search and scraping",
        "prompt": "Firecrawl API key",
        "url": "https://firecrawl.dev/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_API_URL": {
        "description": "Firecrawl API URL for self-hosted instances (optional)",
        "prompt": "Firecrawl API URL (leave empty for cloud)",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "FIRECRAWL_GATEWAY_URL": {
        "description": "Exact Firecrawl tool-gateway origin override for Nous Subscribers only (optional)",
        "prompt": "Firecrawl gateway URL (leave empty to derive from domain)",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_DOMAIN": {
        "description": "Shared tool-gateway domain suffix for Nous Subscribers only, used to derive vendor hosts, e.g. nousresearch.com -> firecrawl-gateway.nousresearch.com",
        "prompt": "Tool-gateway domain suffix",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_SCHEME": {
        "description": "Shared tool-gateway URL scheme for Nous Subscribers only, used to derive vendor hosts (`https` by default, set `http` for local gateway testing)",
        "prompt": "Tool-gateway URL scheme",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_USER_TOKEN": {
        "description": "Explicit Nous Subscriber access token for tool-gateway requests (optional; otherwise read from the Hermes auth store)",
        "prompt": "Tool-gateway user token",
        "url": None,
        "password": True,
        "category": "tool",
        "advanced": True,
    },
    "TAVILY_API_KEY": {
        "description": "Tavily API key for AI-native web search, extract, and crawl",
        "prompt": "Tavily API key",
        "url": "https://app.tavily.com/home",
        "tools": ["web_search", "web_extract", "web_crawl"],
        "password": True,
        "category": "tool",
    },
    "BROWSERBASE_API_KEY": {
        "description": "Browserbase API key for cloud browser (optional — local browser works without this)",
        "prompt": "Browserbase API key",
        "url": "https://browserbase.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": True,
        "category": "tool",
    },
    "BROWSERBASE_PROJECT_ID": {
        "description": "Browserbase project ID (optional — only needed for cloud browser)",
        "prompt": "Browserbase project ID",
        "url": "https://browserbase.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "BROWSER_USE_API_KEY": {
        "description": "Browser Use API key for cloud browser (optional — local browser works without this)",
        "prompt": "Browser Use API key",
        "url": "https://browser-use.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_BROWSER_TTL": {
        "description": "Firecrawl browser session TTL in seconds (optional, default 300)",
        "prompt": "Browser session TTL (seconds)",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "CAMOFOX_URL": {
        "description": "Camofox browser server URL for local anti-detection browsing (e.g. http://localhost:9377)",
        "prompt": "Camofox server URL",
        "url": "https://github.com/jo-inc/camofox-browser",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "FAL_KEY": {
        "description": "FAL API key for image generation",
        "prompt": "FAL API key",
        "url": "https://fal.ai/",
        "tools": ["image_generate"],
        "password": True,
        "category": "tool",
    },
    "TINKER_API_KEY": {
        "description": "Tinker API key for RL training",
        "prompt": "Tinker API key",
        "url": "https://tinker-console.thinkingmachines.ai/keys",
        "tools": ["rl_start_training", "rl_check_status", "rl_stop_training"],
        "password": True,
        "category": "tool",
    },
    "WANDB_API_KEY": {
        "description": "Weights & Biases API key for experiment tracking",
        "prompt": "WandB API key",
        "url": "https://wandb.ai/authorize",
        "tools": ["rl_get_results", "rl_check_status"],
        "password": True,
        "category": "tool",
    },
    "VOICE_TOOLS_OPENAI_KEY": {
        "description": "OpenAI API key for voice transcription (Whisper) and OpenAI TTS",
        "prompt": "OpenAI API Key (for Whisper STT + TTS)",
        "url": "https://platform.openai.com/api-keys",
        "tools": ["voice_transcription", "openai_tts"],
        "password": True,
        "category": "tool",
    },
    "ELEVENLABS_API_KEY": {
        "description": "ElevenLabs API key for premium text-to-speech voices",
        "prompt": "ElevenLabs API key",
        "url": "https://elevenlabs.io/",
        "password": True,
        "category": "tool",
    },
    "MISTRAL_API_KEY": {
        "description": "Mistral API key for Voxtral TTS and transcription (STT)",
        "prompt": "Mistral API key",
        "url": "https://console.mistral.ai/",
        "password": True,
        "category": "tool",
    },
    "GITHUB_TOKEN": {
        "description": "GitHub token for Skills Hub (higher API rate limits, skill publish)",
        "prompt": "GitHub Token",
        "url": "https://github.com/settings/tokens",
        "password": True,
        "category": "tool",
    },

    # ── Honcho ──
    "HONCHO_API_KEY": {
        "description": "Honcho API key for AI-native persistent memory",
        "prompt": "Honcho API key",
        "url": "https://app.honcho.dev",
        "tools": ["honcho_context"],
        "password": True,
        "category": "tool",
    },
    "HONCHO_BASE_URL": {
        "description": "Base URL for self-hosted Honcho instances (no API key needed)",
        "prompt": "Honcho base URL (e.g. http://localhost:8000)",
        "category": "tool",
    },

    # ── Messaging platforms ──
    "TELEGRAM_BOT_TOKEN": {
        "description": "Telegram bot token from @BotFather",
        "prompt": "Telegram bot token",
        "url": "https://t.me/BotFather",
        "password": True,
        "category": "messaging",
    },
    "TELEGRAM_ALLOWED_USERS": {
        "description": "Comma-separated Telegram user IDs allowed to use the bot (get ID from @userinfobot)",
        "prompt": "Allowed Telegram user IDs (comma-separated)",
        "url": "https://t.me/userinfobot",
        "password": False,
        "category": "messaging",
    },
    "TELEGRAM_PROXY": {
        "description": "Proxy URL for Telegram connections (overrides HTTPS_PROXY). Supports http://, https://, socks5://",
        "prompt": "Telegram proxy URL (optional)",
        "password": False,
        "category": "messaging",
    },
    "DISCORD_BOT_TOKEN": {
        "description": "Discord bot token from Developer Portal",
        "prompt": "Discord bot token",
        "url": "https://discord.com/developers/applications",
        "password": True,
        "category": "messaging",
    },
    "DISCORD_ALLOWED_USERS": {
        "description": "Comma-separated Discord user IDs allowed to use the bot",
        "prompt": "Allowed Discord user IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "DISCORD_REPLY_TO_MODE": {
        "description": "Discord reply threading mode: 'off' (no reply references), 'first' (reply on first message only, default), 'all' (reply on every chunk)",
        "prompt": "Discord reply mode (off/first/all)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "SLACK_BOT_TOKEN": {
        "description": "Slack bot token (xoxb-). Get from OAuth & Permissions after installing your app. "
                       "Required scopes: chat:write, app_mentions:read, channels:history, groups:history, "
                       "im:history, im:read, im:write, users:read, files:read, files:write",
        "prompt": "Slack Bot Token (xoxb-...)",
        "url": "https://api.slack.com/apps",
        "password": True,
        "category": "messaging",
    },
    "SLACK_APP_TOKEN": {
        "description": "Slack app-level token (xapp-) for Socket Mode. Get from Basic Information → "
                       "App-Level Tokens. Also ensure Event Subscriptions include: message.im, "
                       "message.channels, message.groups, app_mention",
        "prompt": "Slack App Token (xapp-...)",
        "url": "https://api.slack.com/apps",
        "password": True,
        "category": "messaging",
    },
    "MATTERMOST_URL": {
        "description": "Mattermost server URL (e.g. https://mm.example.com)",
        "prompt": "Mattermost server URL",
        "url": "https://mattermost.com/deploy/",
        "password": False,
        "category": "messaging",
    },
    "MATTERMOST_TOKEN": {
        "description": "Mattermost bot token or personal access token",
        "prompt": "Mattermost bot token",
        "url": None,
        "password": True,
        "category": "messaging",
    },
    "MATTERMOST_ALLOWED_USERS": {
        "description": "Comma-separated Mattermost user IDs allowed to use the bot",
        "prompt": "Allowed Mattermost user IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATTERMOST_REQUIRE_MENTION": {
        "description": "Require @mention in Mattermost channels (default: true). Set to false to respond to all messages.",
        "prompt": "Require @mention in channels",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATTERMOST_FREE_RESPONSE_CHANNELS": {
        "description": "Comma-separated Mattermost channel IDs where bot responds without @mention",
        "prompt": "Free-response channel IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATRIX_HOMESERVER": {
        "description": "Matrix homeserver URL (e.g. https://matrix.example.org)",
        "prompt": "Matrix homeserver URL",
        "url": "https://matrix.org/ecosystem/servers/",
        "password": False,
        "category": "messaging",
    },
    "MATRIX_ACCESS_TOKEN": {
        "description": "Matrix access token (preferred over password login)",
        "prompt": "Matrix access token",
        "url": None,
        "password": True,
        "category": "messaging",
    },
    "MATRIX_USER_ID": {
        "description": "Matrix user ID (e.g. @hermes:example.org)",
        "prompt": "Matrix user ID (@user:server)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATRIX_ALLOWED_USERS": {
        "description": "Comma-separated Matrix user IDs allowed to use the bot (@user:server format)",
        "prompt": "Allowed Matrix user IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATRIX_REQUIRE_MENTION": {
        "description": "Require @mention in Matrix rooms (default: true). Set to false to respond to all messages.",
        "prompt": "Require @mention in rooms (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_FREE_RESPONSE_ROOMS": {
        "description": "Comma-separated Matrix room IDs where bot responds without @mention",
        "prompt": "Free-response room IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_AUTO_THREAD": {
        "description": "Auto-create threads for messages in Matrix rooms (default: true)",
        "prompt": "Auto-create threads in rooms (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_DEVICE_ID": {
        "description": "Stable Matrix device ID for E2EE persistence across restarts (e.g. HERMES_BOT)",
        "prompt": "Matrix device ID (stable across restarts)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_RECOVERY_KEY": {
        "description": "Matrix recovery key for cross-signing verification after device key rotation (from Element: Settings → Security → Recovery Key)",
        "prompt": "Matrix recovery key",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "BLUEBUBBLES_SERVER_URL": {
        "description": "BlueBubbles server URL for iMessage integration (e.g. http://192.168.1.10:1234)",
        "prompt": "BlueBubbles server URL",
        "url": "https://bluebubbles.app/",
        "password": False,
        "category": "messaging",
    },
    "BLUEBUBBLES_PASSWORD": {
        "description": "BlueBubbles server password (from BlueBubbles Server → Settings → API)",
        "prompt": "BlueBubbles server password",
        "url": None,
        "password": True,
        "category": "messaging",
    },
    "BLUEBUBBLES_ALLOWED_USERS": {
        "description": "Comma-separated iMessage addresses (email or phone) allowed to use the bot",
        "prompt": "Allowed iMessage addresses (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "BLUEBUBBLES_ALLOW_ALL_USERS": {
        "description": "Allow all BlueBubbles users without allowlist",
        "prompt": "Allow All BlueBubbles Users",
        "category": "messaging",
    },
    "QQ_APP_ID": {
        "description": "QQ Bot App ID from QQ Open Platform (q.qq.com)",
        "prompt": "QQ App ID",
        "url": "https://q.qq.com",
        "category": "messaging",
    },
    "QQ_CLIENT_SECRET": {
        "description": "QQ Bot Client Secret from QQ Open Platform",
        "prompt": "QQ Client Secret",
        "password": True,
        "category": "messaging",
    },
    "QQ_ALLOWED_USERS": {
        "description": "Comma-separated QQ user IDs allowed to use the bot",
        "prompt": "QQ Allowed Users",
        "category": "messaging",
    },
    "QQ_GROUP_ALLOWED_USERS": {
        "description": "Comma-separated QQ group IDs allowed to interact with the bot",
        "prompt": "QQ Group Allowed Users",
        "category": "messaging",
    },
    "QQ_ALLOW_ALL_USERS": {
        "description": "Allow all QQ users without an allowlist (true/false)",
        "prompt": "Allow All QQ Users",
        "category": "messaging",
    },
    "QQBOT_HOME_CHANNEL": {
        "description": "Default QQ channel/group for cron delivery and notifications",
        "prompt": "QQ Home Channel",
        "category": "messaging",
    },
    "QQBOT_HOME_CHANNEL_NAME": {
        "description": "Display name for the QQ home channel",
        "prompt": "QQ Home Channel Name",
        "category": "messaging",
    },
    "QQ_SANDBOX": {
        "description": "Enable QQ sandbox mode for development testing (true/false)",
        "prompt": "QQ Sandbox Mode",
        "category": "messaging",
    },
    "GATEWAY_ALLOW_ALL_USERS": {
        "description": "Allow all users to interact with messaging bots (true/false). Default: false.",
        "prompt": "Allow all users (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_ENABLED": {
        "description": "Enable the OpenAI-compatible API server (true/false). Allows frontends like Open WebUI, LobeChat, etc. to connect.",
        "prompt": "Enable API server (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_KEY": {
        "description": "Bearer token for API server authentication. Required for non-loopback binding; server refuses to start without it. On loopback (127.0.0.1), all requests are allowed if empty.",
        "prompt": "API server auth key (required for network access)",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_PORT": {
        "description": "Port for the API server (default: 8642).",
        "prompt": "API server port",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_HOST": {
        "description": "Host/bind address for the API server (default: 127.0.0.1). Use 0.0.0.0 for network access — server refuses to start without API_SERVER_KEY.",
        "prompt": "API server host",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_MODEL_NAME": {
        "description": "Model name advertised on /v1/models. Defaults to the profile name (or 'hermes-agent' for the default profile). Useful for multi-user setups with OpenWebUI.",
        "prompt": "API server model name",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "GATEWAY_PROXY_URL": {
        "description": "URL of a remote Hermes API server to forward messages to (proxy mode). When set, the gateway handles platform I/O only — all agent work is delegated to the remote server. Use for Docker E2EE containers that relay to a host agent. Also configurable via gateway.proxy_url in config.yaml.",
        "prompt": "Remote Hermes API server URL (e.g. http://192.168.1.100:8642)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "GATEWAY_PROXY_KEY": {
        "description": "Bearer token for authenticating with the remote Hermes API server (proxy mode). Must match the API_SERVER_KEY on the remote host.",
        "prompt": "Remote API server auth key",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "WEBHOOK_ENABLED": {
        "description": "Enable the webhook platform adapter for receiving events from GitHub, GitLab, etc.",
        "prompt": "Enable webhooks (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "WEBHOOK_PORT": {
        "description": "Port for the webhook HTTP server (default: 8644).",
        "prompt": "Webhook port",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "WEBHOOK_SECRET": {
        "description": "Global HMAC secret for webhook signature validation (overridable per route in config.yaml).",
        "prompt": "Webhook secret",
        "url": None,
        "password": True,
        "category": "messaging",
    },

    # ── Agent settings ──
    # NOTE: MESSAGING_CWD was removed here — use terminal.cwd in config.yaml
    # instead.  The gateway reads TERMINAL_CWD (bridged from terminal.cwd).
    "SUDO_PASSWORD": {
        "description": "Sudo password for terminal commands requiring root access; set to an explicit empty string to try empty without prompting",
        "prompt": "Sudo password",
        "url": None,
        "password": True,
        "category": "setting",
    },
    "HERMES_MAX_ITERATIONS": {
        "description": "Maximum tool-calling iterations per conversation (default: 90)",
        "prompt": "Max iterations",
        "url": None,
        "password": False,
        "category": "setting",
    },
    # HERMES_TOOL_PROGRESS and HERMES_TOOL_PROGRESS_MODE are deprecated —
    # now configured via display.tool_progress in config.yaml (off|new|all|verbose).
    # Gateway falls back to these env vars for backward compatibility.
    "HERMES_TOOL_PROGRESS": {
        "description": "(deprecated) Use display.tool_progress in config.yaml instead",
        "prompt": "Tool progress (deprecated — use config.yaml)",
        "url": None,
        "password": False,
        "category": "setting",
    },
    "HERMES_TOOL_PROGRESS_MODE": {
        "description": "(deprecated) Use display.tool_progress in config.yaml instead",
        "prompt": "Progress mode (deprecated — use config.yaml)",
        "url": None,
        "password": False,
        "category": "setting",
    },
    "HERMES_PREFILL_MESSAGES_FILE": {
        "description": "Path to JSON file with ephemeral prefill messages for few-shot priming",
        "prompt": "Prefill messages file path",
        "url": None,
        "password": False,
        "category": "setting",
    },
    "HERMES_EPHEMERAL_SYSTEM_PROMPT": {
        "description": "Ephemeral system prompt injected at API-call time (never persisted to sessions)",
        "prompt": "Ephemeral system prompt",
        "url": None,
        "password": False,
        "category": "setting",
    },
}

# Tool Gateway env vars are always visible — they're useful for
# self-hosted / custom gateway setups regardless of subscription state.


def get_missing_env_vars(required_only: bool = False) -> List[Dict[str, Any]]:
    """
    Check which environment variables are missing.
    
    Returns list of dicts with var info for missing variables.
    """
    missing = []
    
    # Check required vars
    for var_name, info in REQUIRED_ENV_VARS.items():
        if not get_env_value(var_name):
            missing.append({"name": var_name, **info, "is_required": True})
    
    # Check optional vars (if not required_only)
    if not required_only:
        for var_name, info in OPTIONAL_ENV_VARS.items():
            if not get_env_value(var_name):
                missing.append({"name": var_name, **info, "is_required": False})
    
    return missing


def _set_nested(config: dict, dotted_key: str, value):
    """Set a value at an arbitrarily nested dotted key path.

    Creates intermediate dicts as needed, e.g. ``_set_nested(c, "a.b.c", 1)``
    ensures ``c["a"]["b"]["c"] == 1``.
    """
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def get_missing_config_fields() -> List[Dict[str, Any]]:
    """
    Check which config fields are missing or outdated (recursive).
    
    Walks the DEFAULT_CONFIG tree at arbitrary depth and reports any keys
    present in defaults but absent from the user's loaded config.
    """
    config = load_config()
    missing = []

    def _check(defaults: dict, current: dict, prefix: str = ""):
        for key, default_value in defaults.items():
            if key.startswith('_'):
                continue
            full_key = key if not prefix else f"{prefix}.{key}"
            if key not in current:
                missing.append({
                    "key": full_key,
                    "default": default_value,
                    "description": f"New config option: {full_key}",
                })
            elif isinstance(default_value, dict) and isinstance(current.get(key), dict):
                _check(default_value, current[key], full_key)

    _check(DEFAULT_CONFIG, config)
    return missing


def get_missing_skill_config_vars() -> List[Dict[str, Any]]:
    """Return skill-declared config vars that are missing or empty in config.yaml.

    Scans all enabled skills for ``metadata.hermes.config`` entries, then checks
    which ones are absent or empty under ``skills.config.<key>`` in the user's
    config.yaml.  Returns a list of dicts suitable for prompting.
    """
    try:
        from agent.skill_utils import discover_all_skill_config_vars, SKILL_CONFIG_PREFIX
    except Exception:
        return []

    all_vars = discover_all_skill_config_vars()
    if not all_vars:
        return []

    config = load_config()
    missing: List[Dict[str, Any]] = []
    for var in all_vars:
        # Skill config is stored under skills.config.<logical_key>
        storage_key = f"{SKILL_CONFIG_PREFIX}.{var['key']}"
        parts = storage_key.split(".")
        current = config
        value = None
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
                value = current
            else:
                value = None
                break
        # Missing = key doesn't exist or is empty string
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(var)
    return missing


def _normalize_custom_provider_entry(
    entry: Any,
    *,
    provider_key: str = "",
) -> Optional[Dict[str, Any]]:
    """Return a runtime-compatible custom provider entry or ``None``."""
    if not isinstance(entry, dict):
        return None

    base_url = ""
    for url_key in ("api", "url", "base_url"):
        raw_url = entry.get(url_key)
        if isinstance(raw_url, str) and raw_url.strip():
            base_url = raw_url.strip()
            break
    if not base_url:
        return None

    name = ""
    raw_name = entry.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        name = raw_name.strip()
    elif provider_key.strip():
        name = provider_key.strip()
    if not name:
        return None

    normalized: Dict[str, Any] = {
        "name": name,
        "base_url": base_url,
    }

    provider_key = provider_key.strip()
    if provider_key:
        normalized["provider_key"] = provider_key

    api_key = entry.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        normalized["api_key"] = api_key.strip()

    key_env = entry.get("key_env")
    if isinstance(key_env, str) and key_env.strip():
        normalized["key_env"] = key_env.strip()

    api_mode = entry.get("api_mode") or entry.get("transport")
    if isinstance(api_mode, str) and api_mode.strip():
        normalized["api_mode"] = api_mode.strip()

    model_name = entry.get("model") or entry.get("default_model")
    if isinstance(model_name, str) and model_name.strip():
        normalized["model"] = model_name.strip()

    models = entry.get("models")
    if isinstance(models, dict) and models:
        normalized["models"] = models

    context_length = entry.get("context_length")
    if isinstance(context_length, int) and context_length > 0:
        normalized["context_length"] = context_length

    rate_limit_delay = entry.get("rate_limit_delay")
    if isinstance(rate_limit_delay, (int, float)) and rate_limit_delay >= 0:
        normalized["rate_limit_delay"] = rate_limit_delay

    return normalized


def providers_dict_to_custom_providers(providers_dict: Any) -> List[Dict[str, Any]]:
    """Normalize ``providers`` config entries into the legacy custom-provider shape."""
    if not isinstance(providers_dict, dict):
        return []

    custom_providers: List[Dict[str, Any]] = []
    for key, entry in providers_dict.items():
        normalized = _normalize_custom_provider_entry(entry, provider_key=str(key))
        if normalized is not None:
            custom_providers.append(normalized)

    return custom_providers


def get_compatible_custom_providers(
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return a deduplicated custom-provider view across legacy and v12+ config.

    ``custom_providers`` remains the on-disk legacy format, while ``providers``
    is the newer keyed schema.  Runtime and picker flows still need a single
    list-shaped view, but we should not materialise that compatibility layer
    back into config.yaml because it duplicates entries in UIs.
    """
    if config is None:
        config = load_config()

    compatible: List[Dict[str, Any]] = []
    seen_provider_keys: set = set()
    seen_name_url_pairs: set = set()

    def _append_if_new(entry: Optional[Dict[str, Any]]) -> None:
        if entry is None:
            return
        provider_key = str(entry.get("provider_key", "") or "").strip().lower()
        name = str(entry.get("name", "") or "").strip().lower()
        base_url = str(entry.get("base_url", "") or "").strip().rstrip("/").lower()
        model = str(entry.get("model", "") or "").strip().lower()
        pair = (name, base_url, model)

        if provider_key and provider_key in seen_provider_keys:
            return
        if name and base_url and pair in seen_name_url_pairs:
            return

        compatible.append(entry)
        if provider_key:
            seen_provider_keys.add(provider_key)
        if name and base_url:
            seen_name_url_pairs.add(pair)

    custom_providers = config.get("custom_providers")
    if custom_providers is not None:
        if not isinstance(custom_providers, list):
            return []
        for entry in custom_providers:
            _append_if_new(_normalize_custom_provider_entry(entry))

    for entry in providers_dict_to_custom_providers(config.get("providers")):
        _append_if_new(entry)

    return compatible


def check_config_version() -> Tuple[int, int]:
    """
    Check config version.
    
    Returns (current_version, latest_version).
    """
    config = load_config()
    current = config.get("_config_version", 0)
    latest = DEFAULT_CONFIG.get("_config_version", 1)
    return current, latest


# =============================================================================
# Config structure validation
# =============================================================================

# Fields that are valid at root level of config.yaml
_KNOWN_ROOT_KEYS = {
    "_config_version", "model", "providers", "fallback_model",
    "fallback_providers", "credential_pool_strategies", "agents",
    "categories", "runtime_fallback", "model_capabilities", "toolsets",
    "agent", "terminal", "display", "compression", "delegation",
    "auxiliary", "custom_providers", "context", "memory", "gateway",
}

# Valid fields inside a custom_providers list entry
_VALID_CUSTOM_PROVIDER_FIELDS = {
    "name", "base_url", "api_key", "api_mode", "model", "models",
    "context_length", "rate_limit_delay",
}

# Fields that look like they should be inside custom_providers, not at root
_CUSTOM_PROVIDER_LIKE_FIELDS = {"base_url", "api_key", "rate_limit_delay", "api_mode"}


@dataclass
class ConfigIssue:
    """A detected config structure problem."""

    severity: str  # "error", "warning"
    message: str
    hint: str


def validate_config_structure(config: Optional[Dict[str, Any]] = None) -> List["ConfigIssue"]:
    """Validate config.yaml structure and return a list of detected issues.

    Catches common YAML formatting mistakes that produce confusing runtime
    errors (like "Unknown provider") instead of clear diagnostics.

    Can be called with a pre-loaded config dict, or will load the raw config
    from disk so warnings can still see ignored/unknown keys before
    normalization drops them.
    """
    raw_config = config
    if config is None:
        try:
            config = read_raw_config()
            raw_config = config
        except Exception:
            return [ConfigIssue("error", "Could not load config.yaml", "Run 'hermes setup' to create a valid config")]

    issues: List[ConfigIssue] = []

    agents_cfg = config.get("agents")
    if agents_cfg is not None and not isinstance(agents_cfg, dict):
        issues.append(ConfigIssue(
            "warning",
            f"agents should be a mapping of named agent configs, got {type(agents_cfg).__name__}",
            "Change to:\n  agents:\n    oracle:\n      model: openai/gpt-5.4",
        ))

    categories_cfg = config.get("categories")
    if categories_cfg is not None and not isinstance(categories_cfg, dict):
        issues.append(ConfigIssue(
            "warning",
            f"categories should be a mapping of named category configs, got {type(categories_cfg).__name__}",
            "Change to:\n  categories:\n    research:\n      model: anthropic/claude-haiku-4-5",
        ))

    runtime_fallback_cfg = config.get("runtime_fallback")
    if runtime_fallback_cfg is not None and not isinstance(runtime_fallback_cfg, (bool, dict)):
        issues.append(ConfigIssue(
            "warning",
            f"runtime_fallback should be either a boolean or a config dict, got {type(runtime_fallback_cfg).__name__}",
            "Use either 'runtime_fallback: true' or a mapping like runtime_fallback:\n  enabled: true",
        ))

    model_capabilities_cfg = config.get("model_capabilities")
    if model_capabilities_cfg is not None and not isinstance(model_capabilities_cfg, dict):
        issues.append(ConfigIssue(
            "warning",
            f"model_capabilities should be a config dict, got {type(model_capabilities_cfg).__name__}",
            "Change to:\n  model_capabilities:\n    enabled: true",
        ))

    # ── custom_providers must be a list, not a dict ──────────────────────
    cp = config.get("custom_providers")
    if cp is not None:
        if isinstance(cp, dict):
            issues.append(ConfigIssue(
                "error",
                "custom_providers is a dict — it must be a YAML list (items prefixed with '-')",
                "Change to:\n"
                "  custom_providers:\n"
                "    - name: my-provider\n"
                "      base_url: https://...\n"
                "      api_key: ...",
            ))
            # Check if dict keys look like they should be list-entry fields
            cp_keys = set(cp.keys()) if isinstance(cp, dict) else set()
            suspicious = cp_keys & _CUSTOM_PROVIDER_LIKE_FIELDS
            if suspicious:
                issues.append(ConfigIssue(
                    "warning",
                    f"Root-level keys {sorted(suspicious)} look like custom_providers entry fields",
                    "These should be indented under a '- name: ...' list entry, not at root level",
                ))
        elif isinstance(cp, list):
            # Validate each entry in the list
            for i, entry in enumerate(cp):
                if not isinstance(entry, dict):
                    issues.append(ConfigIssue(
                        "warning",
                        f"custom_providers[{i}] is not a dict (got {type(entry).__name__})",
                        "Each entry should have at minimum: name, base_url",
                    ))
                    continue
                if not entry.get("name"):
                    issues.append(ConfigIssue(
                        "warning",
                        f"custom_providers[{i}] is missing 'name' field",
                        "Add a name, e.g.: name: my-provider",
                    ))
                if not entry.get("base_url"):
                    issues.append(ConfigIssue(
                        "warning",
                        f"custom_providers[{i}] is missing 'base_url' field",
                        "Add the API endpoint URL, e.g.: base_url: https://api.example.com/v1",
                    ))

    # ── fallback_model must be a top-level dict with provider + model ────
    fb = config.get("fallback_model")
    if fb is not None:
        if not isinstance(fb, dict):
            issues.append(ConfigIssue(
                "error",
                f"fallback_model should be a dict with 'provider' and 'model', got {type(fb).__name__}",
                "Change to:\n"
                "  fallback_model:\n"
                "    provider: openrouter\n"
                "    model: anthropic/claude-sonnet-4",
            ))
        elif fb:
            if not fb.get("provider"):
                issues.append(ConfigIssue(
                    "warning",
                    "fallback_model is missing 'provider' field — fallback will be disabled",
                    "Add: provider: openrouter (or another provider)",
                ))
            if not fb.get("model"):
                issues.append(ConfigIssue(
                    "warning",
                    "fallback_model is missing 'model' field — fallback will be disabled",
                    "Add: model: anthropic/claude-sonnet-4 (or another model)",
                ))

    fallback_providers = config.get("fallback_providers")
    if fallback_providers is not None:
        if not isinstance(fallback_providers, list):
            issues.append(ConfigIssue(
                "error",
                f"fallback_providers should be a list of provider/model dicts, got {type(fallback_providers).__name__}",
                "Change to:\n  fallback_providers:\n    - provider: openrouter\n      model: anthropic/claude-sonnet-4",
            ))
        else:
            for i, entry in enumerate(fallback_providers):
                if not isinstance(entry, dict):
                    issues.append(ConfigIssue(
                        "warning",
                        f"fallback_providers[{i}] should be a dict, got {type(entry).__name__}",
                        "Each fallback_providers entry should include provider and model fields.",
                    ))
                    continue
                if not entry.get("provider"):
                    issues.append(ConfigIssue(
                        "warning",
                        f"fallback_providers[{i}] is missing 'provider' field",
                        "Add: provider: openrouter (or another provider)",
                    ))
                if not entry.get("model"):
                    issues.append(ConfigIssue(
                        "warning",
                        f"fallback_providers[{i}] is missing 'model' field",
                        "Add: model: anthropic/claude-sonnet-4 (or another model)",
                    ))

    # ── Check for fallback_model accidentally nested inside custom_providers ──
    if isinstance(cp, dict) and "fallback_model" not in config and "fallback_model" in (cp or {}):
        issues.append(ConfigIssue(
            "error",
            "fallback_model appears inside custom_providers instead of at root level",
            "Move fallback_model to the top level of config.yaml (no indentation)",
        ))

    # ── model section: should exist when custom_providers is configured ──
    model_cfg = config.get("model")
    if cp and not model_cfg:
        issues.append(ConfigIssue(
            "warning",
            "custom_providers defined but no 'model' section — Hermes won't know which provider to use",
            "Add a model section:\n"
            "  model:\n"
            "    provider: custom\n"
            "    default: your-model-name\n"
            "    base_url: https://...",
        ))

    delegation_cfg = config.get("delegation")
    if delegation_cfg is not None:
        if not isinstance(delegation_cfg, dict):
            issues.append(ConfigIssue(
                "error",
                f"delegation should be a dict, got {type(delegation_cfg).__name__}",
                "Change to:\n  delegation:\n    max_iterations: 50\n    max_concurrent_children: 3",
            ))
        else:
            profile_buckets = []
            delegation_profiles = delegation_cfg.get("delegation_profiles")
            if delegation_profiles is not None:
                profile_buckets.append(("delegation_profiles", delegation_profiles, "primary Wave 1 delegation-profile config"))
            categories = delegation_cfg.get("categories")
            if categories is not None:
                profile_buckets.append(("categories", categories, "legacy alias for delegation_profiles"))

            for bucket_name, bucket_value, bucket_note in profile_buckets:
                if not isinstance(bucket_value, dict):
                    issues.append(ConfigIssue(
                        "error",
                        f"delegation.{bucket_name} should be a dict of delegation profiles, got {type(bucket_value).__name__}",
                        f"Change to:\n  delegation:\n    {bucket_name}:\n      research:\n        enabled_tools:\n          - read_file",
                    ))
                    continue

                for name, entry in bucket_value.items():
                    if not isinstance(entry, dict):
                        issues.append(ConfigIssue(
                            "warning",
                            f"delegation.{bucket_name}.{name} should be a dict, got {type(entry).__name__}",
                            f"Each {bucket_note} entry should define keys like toolsets, enabled_tools, runtime_mode, max_iterations, or max_concurrent_children",
                        ))
                        continue

                    routing_description = entry.get("routing_description")
                    if routing_description is not None and not isinstance(routing_description, str):
                        issues.append(ConfigIssue(
                            "warning",
                            f"delegation.{bucket_name}.{name}.routing_description should be a string, got {type(routing_description).__name__}",
                            "Use a short sentence describing when the primary agent should route work into this delegation profile.",
                        ))

            route_categories = delegation_cfg.get("route_categories")
            if route_categories is not None and not isinstance(route_categories, dict):
                issues.append(ConfigIssue(
                    "error",
                    f"delegation.route_categories should be a dict of built-in route-category metadata overrides, got {type(route_categories).__name__}",
                    "Change to:\n  delegation:\n    route_categories:\n      quick:\n        summary: Fast-path routing lane\n        intensity: low",
                ))

            runtime_modes = delegation_cfg.get("runtime_modes")
            if runtime_modes is not None and not isinstance(runtime_modes, dict):
                issues.append(ConfigIssue(
                    "error",
                    f"delegation.runtime_modes should be a dict of built-in runtime-mode metadata overrides, got {type(runtime_modes).__name__}",
                    "Change to:\n  delegation:\n    runtime_modes:\n      default:\n        description: Standard Hermes operating posture\n        operating_posture: balanced_general_operation",
                ))

            delegation_metadata = inspect_wave1_delegation_metadata(
                _normalize_wave1_delegation_config({"delegation": delegation_cfg}, include_defaults=False),
                raw_config if isinstance(raw_config, dict) else {"delegation": delegation_cfg},
            )
            ignored_route_category_names = delegation_metadata["ignored_route_category_names"]
            if ignored_route_category_names:
                issues.append(ConfigIssue(
                    "warning",
                    "delegation.route_categories contains unknown names that stay inactive metadata only: "
                    + ", ".join(ignored_route_category_names),
                    "Wave 1 only applies built-in route-category metadata overrides. Unknown names are ignored for routing semantics; use delegation_profiles/categories for active delegation profiles.",
                ))

            ignored_runtime_mode_names = delegation_metadata["ignored_runtime_mode_names"]
            if ignored_runtime_mode_names:
                issues.append(ConfigIssue(
                    "warning",
                    "delegation.runtime_modes contains unknown names that stay inactive metadata only: "
                    + ", ".join(ignored_runtime_mode_names),
                    "Wave 1 only applies built-in runtime-mode metadata overrides. Unknown names are ignored for runtime semantics.",
                ))

            task_contract = delegation_cfg.get("task_contract")
            if task_contract not in (None, "", {}) and not isinstance(task_contract, dict):
                issues.append(ConfigIssue(
                    "error",
                    f"delegation.task_contract should be a dict matching the Wave 1 task-contract schema, got {type(task_contract).__name__}",
                    "Remove delegation.task_contract to preserve legacy delegation behavior, or add all required fields under delegation.task_contract.",
                ))

            permission_preset = delegation_cfg.get("permission_preset")
            if permission_preset is not None and not isinstance(permission_preset, str):
                issues.append(ConfigIssue(
                    "warning",
                    f"delegation.permission_preset should be a string, got {type(permission_preset).__name__}",
                    "Use values like 'inherit' or 'workspace_write' to preserve Wave 1 permission semantics.",
                ))

            fallback_policy = delegation_cfg.get("fallback_policy")
            if fallback_policy is not None and not isinstance(fallback_policy, str):
                issues.append(ConfigIssue(
                    "warning",
                    f"delegation.fallback_policy should be a string, got {type(fallback_policy).__name__}",
                    "Use values like 'legacy_default_mapping' or 'degrade_to_generalist' to preserve Wave 1 fallback semantics.",
                ))

    # ── Root-level keys that look misplaced ──────────────────────────────
    for key in config:
        if key.startswith("_"):
            continue
        if key not in _KNOWN_ROOT_KEYS and key in _CUSTOM_PROVIDER_LIKE_FIELDS:
            issues.append(ConfigIssue(
                "warning",
                f"Root-level key '{key}' looks misplaced — should it be under 'model:' or inside a 'custom_providers' entry?",
                f"Move '{key}' under the appropriate section",
            ))

    return issues


def print_config_warnings(config: Optional[Dict[str, Any]] = None) -> None:
    """Print config structure warnings to stderr at startup.

    Called early in CLI and gateway init so users see problems before
    they hit cryptic "Unknown provider" errors.  Prints nothing if
    config is healthy.
    """
    try:
        issues = validate_config_structure(config)
    except Exception:
        return
    if not issues:
        return

    import sys
    lines = ["\033[33m⚠ Config issues detected in config.yaml:\033[0m"]
    for ci in issues:
        marker = "\033[31m✗\033[0m" if ci.severity == "error" else "\033[33m⚠\033[0m"
        lines.append(f"  {marker} {ci.message}")
    lines.append("  \033[2mRun 'hermes doctor' for fix suggestions.\033[0m")
    sys.stderr.write("\n".join(lines) + "\n\n")


def warn_deprecated_cwd_env_vars(config: Optional[Dict[str, Any]] = None) -> None:
    """Warn if MESSAGING_CWD or TERMINAL_CWD is set in .env instead of config.yaml.

    These env vars are deprecated — the canonical setting is terminal.cwd
    in config.yaml.  Prints a migration hint to stderr.
    """
    import os, sys
    messaging_cwd = os.environ.get("MESSAGING_CWD")
    terminal_cwd_env = os.environ.get("TERMINAL_CWD")

    if config is None:
        try:
            config = load_config()
        except Exception:
            return

    terminal_cfg = config.get("terminal", {})
    config_cwd = terminal_cfg.get("cwd", ".") if isinstance(terminal_cfg, dict) else "."
    # Only warn if config.yaml doesn't have an explicit path
    config_has_explicit_cwd = config_cwd not in (".", "auto", "cwd", "")

    lines: list[str] = []
    if messaging_cwd:
        lines.append(
            f"  \033[33m⚠\033[0m MESSAGING_CWD={messaging_cwd} found in .env — "
            f"this is deprecated."
        )
    if terminal_cwd_env and not config_has_explicit_cwd:
        # TERMINAL_CWD in env but not from config bridge — likely from .env
        lines.append(
            f"  \033[33m⚠\033[0m TERMINAL_CWD={terminal_cwd_env} found in .env — "
            f"this is deprecated."
        )
    if lines:
        hint_path = os.environ.get("HERMES_HOME", "~/.hermes")
        lines.insert(0, "\033[33m⚠ Deprecated .env settings detected:\033[0m")
        lines.append(
            f"  \033[2mMove to config.yaml instead:  "
            f"terminal:\\n    cwd: /your/project/path\033[0m"
        )
        lines.append(
            f"  \033[2mThen remove the old entries from {hint_path}/.env\033[0m"
        )
        sys.stderr.write("\n".join(lines) + "\n\n")


def migrate_config(interactive: bool = True, quiet: bool = False) -> Dict[str, Any]:
    """
    Migrate config to latest version, prompting for new required fields.
    
    Args:
        interactive: If True, prompt user for missing values
        quiet: If True, suppress output
        
    Returns:
        Dict with migration results: {"env_added": [...], "config_added": [...], "warnings": [...]}
    """
    results = {"env_added": [], "config_added": [], "warnings": []}

    # ── Always: sanitize .env (split concatenated keys) ──
    try:
        fixes = sanitize_env_file()
        if fixes and not quiet:
            print(f"  ✓ Repaired .env file ({fixes} corrupted entries fixed)")
    except Exception:
        pass  # best-effort; don't block migration on sanitize failure

    # Check config version
    current_ver, latest_ver = check_config_version()
    
    # ── Version 3 → 4: migrate tool progress from .env to config.yaml ──
    if current_ver < 4:
        config = load_config()
        display = config.get("display", {})
        if not isinstance(display, dict):
            display = {}
        if "tool_progress" not in display:
            old_enabled = get_env_value("HERMES_TOOL_PROGRESS")
            old_mode = get_env_value("HERMES_TOOL_PROGRESS_MODE")
            if old_enabled and old_enabled.lower() in ("false", "0", "no"):
                display["tool_progress"] = "off"
                results["config_added"].append("display.tool_progress=off (from HERMES_TOOL_PROGRESS=false)")
            elif old_mode and old_mode.lower() in ("new", "all"):
                display["tool_progress"] = old_mode.lower()
                results["config_added"].append(f"display.tool_progress={old_mode.lower()} (from HERMES_TOOL_PROGRESS_MODE)")
            else:
                display["tool_progress"] = "all"
                results["config_added"].append("display.tool_progress=all (default)")
            config["display"] = display
            save_config(config)
            if not quiet:
                print(f"  ✓ Migrated tool progress to config.yaml: {display['tool_progress']}")
    
    # ── Version 4 → 5: add timezone field ──
    if current_ver < 5:
        config = load_config()
        if "timezone" not in config:
            old_tz = os.getenv("HERMES_TIMEZONE", "")
            if old_tz and old_tz.strip():
                config["timezone"] = old_tz.strip()
                results["config_added"].append(f"timezone={old_tz.strip()} (from HERMES_TIMEZONE)")
            else:
                config["timezone"] = ""
                results["config_added"].append("timezone= (empty, uses server-local)")
            save_config(config)
            if not quiet:
                tz_display = config["timezone"] or "(server-local)"
                print(f"  ✓ Added timezone to config.yaml: {tz_display}")

    # ── Version 8 → 9: clear ANTHROPIC_TOKEN from .env ──
    # The new Anthropic auth flow no longer uses this env var.
    if current_ver < 9:
        try:
            old_token = get_env_value("ANTHROPIC_TOKEN")
            if old_token:
                save_env_value("ANTHROPIC_TOKEN", "")
                if not quiet:
                    print("  ✓ Cleared ANTHROPIC_TOKEN from .env (no longer used)")
        except Exception:
            pass

    # ── Version 11 → 12: migrate custom_providers list → providers dict ──
    if current_ver < 12:
        config = load_config()
        custom_list = config.get("custom_providers")
        if isinstance(custom_list, list) and custom_list:
            providers_dict = config.get("providers", {})
            if not isinstance(providers_dict, dict):
                providers_dict = {}
            migrated_count = 0
            for entry in custom_list:
                if not isinstance(entry, dict):
                    continue
                old_name = entry.get("name", "")
                old_url = entry.get("base_url", "") or entry.get("url", "") or ""
                old_key = entry.get("api_key", "")
                if not old_url:
                    continue  # skip entries with no URL

                # Generate a kebab-case key from the display name
                key = old_name.strip().lower().replace(" ", "-").replace("(", "").replace(")", "")
                # Remove consecutive hyphens and trailing hyphens
                while "--" in key:
                    key = key.replace("--", "-")
                key = key.strip("-")
                if not key:
                    # Fallback: derive from URL hostname
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(old_url)
                        key = (parsed.hostname or "endpoint").replace(".", "-")
                    except Exception:
                        key = f"endpoint-{migrated_count}"

                # Don't overwrite existing entries
                if key in providers_dict:
                    key = f"{key}-{migrated_count}"

                new_entry = {"api": old_url}
                if old_name:
                    new_entry["name"] = old_name
                if old_key and old_key not in ("no-key", "no-key-required", ""):
                    new_entry["api_key"] = old_key

                # Carry over model and api_mode if present
                if entry.get("model"):
                    new_entry["default_model"] = entry["model"]
                if entry.get("api_mode"):
                    new_entry["transport"] = entry["api_mode"]

                providers_dict[key] = new_entry
                migrated_count += 1

            if migrated_count > 0:
                config["providers"] = providers_dict
                # Remove the old list — runtime reads via get_compatible_custom_providers()
                config.pop("custom_providers", None)
                save_config(config)
                if not quiet:
                    print(f"  ✓ Migrated {migrated_count} custom provider(s) to providers: section")
                    for key in list(providers_dict.keys())[-migrated_count:]:
                        ep = providers_dict[key]
                        print(f"    → {key}: {ep.get('api', '')}")

    # ── Version 12 → 13: clear dead LLM_MODEL / OPENAI_MODEL from .env ──
    # These env vars were written by the old setup wizard but nothing reads
    # them anymore (config.yaml is the sole source of truth since March 2026).
    # Stale entries cause user confusion — see issue report.
    if current_ver < 13:
        for dead_var in ("LLM_MODEL", "OPENAI_MODEL"):
            try:
                old_val = get_env_value(dead_var)
                if old_val:
                    save_env_value(dead_var, "")
                    if not quiet:
                        print(f"  ✓ Cleared {dead_var} from .env (no longer used — config.yaml is source of truth)")
            except Exception:
                pass

    # ── Version 13 → 14: migrate legacy flat stt.model to provider section ──
    # Old configs (and cli-config.yaml.example) had a flat `stt.model` key
    # that was provider-agnostic.  When the provider was "local" this caused
    # OpenAI model names (e.g. "whisper-1") to be fed to faster-whisper,
    # crashing with "Invalid model size".  Move the value into the correct
    # provider-specific section and remove the flat key.
    if current_ver < 14:
        # Read raw config (no defaults merged) to check what the user actually
        # wrote, then apply changes to the merged config for saving.
        raw = read_raw_config()
        raw_stt = raw.get("stt", {})
        if isinstance(raw_stt, dict) and "model" in raw_stt:
            legacy_model = raw_stt["model"]
            provider = raw_stt.get("provider", "local")
            config = load_config()
            stt = config.get("stt", {})
            # Remove the legacy flat key
            stt.pop("model", None)
            # Place it in the appropriate provider section only if the
            # user didn't already set a model there
            if provider in ("local", "local_command"):
                # Don't migrate an OpenAI model name into the local section
                _local_models = {
                    "tiny.en", "tiny", "base.en", "base", "small.en", "small",
                    "medium.en", "medium", "large-v1", "large-v2", "large-v3",
                    "large", "distil-large-v2", "distil-medium.en",
                    "distil-small.en", "distil-large-v3", "distil-large-v3.5",
                    "large-v3-turbo", "turbo",
                }
                if legacy_model in _local_models:
                    # Check raw config — only set if user didn't already
                    # have a nested local.model
                    raw_local = raw_stt.get("local", {})
                    if not isinstance(raw_local, dict) or "model" not in raw_local:
                        local_cfg = stt.setdefault("local", {})
                        local_cfg["model"] = legacy_model
                # else: drop it — it was an OpenAI model name, local section
                # already defaults to "base" via DEFAULT_CONFIG
            else:
                # Cloud provider — put it in that provider's section only
                # if user didn't already set a nested model
                raw_provider = raw_stt.get(provider, {})
                if not isinstance(raw_provider, dict) or "model" not in raw_provider:
                    provider_cfg = stt.setdefault(provider, {})
                    provider_cfg["model"] = legacy_model
            config["stt"] = stt
            save_config(config)
            if not quiet:
                print(f"  ✓ Migrated legacy stt.model to provider-specific config")

    # ── Version 14 → 15: add explicit gateway interim-message gate ──
    if current_ver < 15:
        config = read_raw_config()
        display = config.get("display", {})
        if not isinstance(display, dict):
            display = {}
        if "interim_assistant_messages" not in display:
            display["interim_assistant_messages"] = True
            config["display"] = display
            results["config_added"].append("display.interim_assistant_messages=true (default)")
            save_config(config)
            if not quiet:
                print("  ✓ Added display.interim_assistant_messages=true")

    # ── Version 15 → 16: migrate tool_progress_overrides into display.platforms ──
    if current_ver < 16:
        config = read_raw_config()
        display = config.get("display", {})
        if not isinstance(display, dict):
            display = {}
        old_overrides = display.get("tool_progress_overrides")
        if isinstance(old_overrides, dict) and old_overrides:
            platforms = display.get("platforms", {})
            if not isinstance(platforms, dict):
                platforms = {}
            for plat, mode in old_overrides.items():
                if plat not in platforms:
                    platforms[plat] = {}
                if "tool_progress" not in platforms[plat]:
                    platforms[plat]["tool_progress"] = mode
            display["platforms"] = platforms
            config["display"] = display
            save_config(config)
            if not quiet:
                migrated = ", ".join(f"{p}={m}" for p, m in old_overrides.items())
                print(f"  ✓ Migrated tool_progress_overrides → display.platforms: {migrated}")
            results["config_added"].append("display.platforms (migrated from tool_progress_overrides)")

    # ── Version 16 → 17: remove legacy compression.summary_* keys ──
    if current_ver < 17:
        config = read_raw_config()
        comp = config.get("compression", {})
        if isinstance(comp, dict):
            s_model = comp.pop("summary_model", None)
            s_provider = comp.pop("summary_provider", None)
            s_base_url = comp.pop("summary_base_url", None)
            migrated_keys = []
            # Migrate non-empty, non-default values to auxiliary.compression
            if s_model and str(s_model).strip():
                aux = config.setdefault("auxiliary", {})
                aux_comp = aux.setdefault("compression", {})
                if not aux_comp.get("model"):
                    aux_comp["model"] = str(s_model).strip()
                    migrated_keys.append(f"model={s_model}")
            if s_provider and str(s_provider).strip() not in ("", "auto"):
                aux = config.setdefault("auxiliary", {})
                aux_comp = aux.setdefault("compression", {})
                if not aux_comp.get("provider") or aux_comp.get("provider") == "auto":
                    aux_comp["provider"] = str(s_provider).strip()
                    migrated_keys.append(f"provider={s_provider}")
            if s_base_url and str(s_base_url).strip():
                aux = config.setdefault("auxiliary", {})
                aux_comp = aux.setdefault("compression", {})
                if not aux_comp.get("base_url"):
                    aux_comp["base_url"] = str(s_base_url).strip()
                    migrated_keys.append(f"base_url={s_base_url}")
            if migrated_keys or s_model is not None or s_provider is not None or s_base_url is not None:
                config["compression"] = comp
                save_config(config)
                if not quiet:
                    if migrated_keys:
                        print(f"  ✓ Migrated compression.summary_* → auxiliary.compression: {', '.join(migrated_keys)}")
                    else:
                        print("  ✓ Removed unused compression.summary_* keys")

    if current_ver < latest_ver and not quiet:
        print(f"Config version: {current_ver} → {latest_ver}")
    
    # Check for missing required env vars
    missing_env = get_missing_env_vars(required_only=True)
    
    if missing_env and not quiet:
        print("\n⚠️  Missing required environment variables:")
        for var in missing_env:
            print(f"   • {var['name']}: {var['description']}")
    
    if interactive and missing_env:
        print("\nLet's configure them now:\n")
        for var in missing_env:
            if var.get("url"):
                print(f"  Get your key at: {var['url']}")
            
            if var.get("password"):
                import getpass
                value = getpass.getpass(f"  {var['prompt']}: ")
            else:
                value = input(f"  {var['prompt']}: ").strip()
            
            if value:
                save_env_value(var["name"], value)
                results["env_added"].append(var["name"])
                print(f"  ✓ Saved {var['name']}")
            else:
                results["warnings"].append(f"Skipped {var['name']} - some features may not work")
            print()
    
    # Check for missing optional env vars and offer to configure interactively
    # Skip "advanced" vars (like OPENAI_BASE_URL) -- those are for power users
    missing_optional = get_missing_env_vars(required_only=False)
    required_names = {v["name"] for v in missing_env} if missing_env else set()
    missing_optional = [
        v for v in missing_optional
        if v["name"] not in required_names and not v.get("advanced")
    ]
    
    # Only offer to configure env vars that are NEW since the user's previous version
    new_var_names = set()
    for ver in range(current_ver + 1, latest_ver + 1):
        new_var_names.update(ENV_VARS_BY_VERSION.get(ver, []))

    if new_var_names and interactive and not quiet:
        new_and_unset = [
            (name, OPTIONAL_ENV_VARS[name])
            for name in sorted(new_var_names)
            if not get_env_value(name) and name in OPTIONAL_ENV_VARS
        ]
        if new_and_unset:
            print(f"\n  {len(new_and_unset)} new optional key(s) in this update:")
            for name, info in new_and_unset:
                print(f"    • {name} — {info.get('description', '')}")
            print()
            try:
                answer = input("  Configure new keys? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"

            if answer in ("y", "yes"):
                print()
                for name, info in new_and_unset:
                    if info.get("url"):
                        print(f"  {info.get('description', name)}")
                        print(f"  Get your key at: {info['url']}")
                    else:
                        print(f"  {info.get('description', name)}")
                    if info.get("password"):
                        import getpass
                        value = getpass.getpass(f"  {info.get('prompt', name)} (Enter to skip): ")
                    else:
                        value = input(f"  {info.get('prompt', name)} (Enter to skip): ").strip()
                    if value:
                        save_env_value(name, value)
                        results["env_added"].append(name)
                        print(f"  ✓ Saved {name}")
                    print()
            else:
                print("  Set later with: hermes config set <key> <value>")
    
    # Check for missing config fields
    missing_config = get_missing_config_fields()
    
    if missing_config:
        config = load_config()
        
        for field in missing_config:
            key = field["key"]
            default = field["default"]
            
            _set_nested(config, key, default)
            results["config_added"].append(key)
            if not quiet:
                print(f"  ✓ Added {key} = {default}")
        
        # Update version and save
        config["_config_version"] = latest_ver
        save_config(config)
    elif current_ver < latest_ver:
        # Just update version
        config = load_config()
        config["_config_version"] = latest_ver
        save_config(config)

    # ── Skill-declared config vars ──────────────────────────────────────
    # Skills can declare config.yaml settings they need via
    # metadata.hermes.config in their SKILL.md frontmatter.
    # Prompt for any that are missing/empty.
    missing_skill_config = get_missing_skill_config_vars()
    if missing_skill_config and interactive and not quiet:
        print(f"\n  {len(missing_skill_config)} skill setting(s) not configured:")
        for var in missing_skill_config:
            skill_name = var.get("skill", "unknown")
            print(f"    • {var['key']} — {var['description']} (from skill: {skill_name})")
        print()
        try:
            answer = input("  Configure skill settings? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer in ("y", "yes"):
            print()
            config = load_config()
            try:
                from agent.skill_utils import SKILL_CONFIG_PREFIX
            except Exception:
                SKILL_CONFIG_PREFIX = "skills.config"
            for var in missing_skill_config:
                default = var.get("default", "")
                default_hint = f" (default: {default})" if default else ""
                value = input(f"  {var['prompt']}{default_hint}: ").strip()
                if not value and default:
                    value = str(default)
                if value:
                    storage_key = f"{SKILL_CONFIG_PREFIX}.{var['key']}"
                    _set_nested(config, storage_key, value)
                    results["config_added"].append(var["key"])
                    print(f"  ✓ Saved {var['key']} = {value}")
                else:
                    results["warnings"].append(
                        f"Skipped {var['key']} — skill '{var.get('skill', '?')}' may ask for it later"
                    )
                print()
            save_config(config)
        else:
            print("  Set later with: hermes config set <key> <value>")

    return results


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, preserving nested defaults.

    Keys in *override* take precedence. If both values are dicts the merge
    recurses, so a user who overrides only ``tts.elevenlabs.voice_id`` will
    keep the default ``tts.elevenlabs.model_id`` intact.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_env_vars(obj):
    """Recursively expand ``${VAR}`` references in config values.

    Only string values are processed; dict keys, numbers, booleans, and
    None are left untouched.  Unresolved references (variable not in
    ``os.environ``) are kept verbatim so callers can detect them.
    """
    if isinstance(obj, str):
        return re.sub(
            r"\${([^}]+)}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def _items_by_unique_name(items):
    """Return a name-indexed dict only when all items have unique string names."""
    if not isinstance(items, list):
        return None
    indexed = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        name = item["name"]
        if name in indexed:
            return None
        indexed[name] = item
    return indexed


def _preserve_env_ref_templates(current, raw, loaded_expanded=None):
    """Restore raw ``${VAR}`` templates when a value is otherwise unchanged.

    ``load_config()`` expands env refs for runtime use. When a caller later
    persists that config after modifying some unrelated setting, keep the
    original on-disk template instead of writing the expanded plaintext
    secret back to ``config.yaml``.

    Prefer preserving the raw template when ``current`` still matches either
    the value previously returned by ``load_config()`` for this config path or
    the current environment expansion of ``raw``. This handles env-var
    rotation between load and save while still treating mixed literal/template
    string edits as caller-owned once their rendered value diverges.
    """
    if isinstance(current, str) and isinstance(raw, str) and re.search(r"\${[^}]+}", raw):
        if current == raw:
            return raw
        if isinstance(loaded_expanded, str) and current == loaded_expanded:
            return raw
        if _expand_env_vars(raw) == current:
            return raw
        return current

    if isinstance(current, dict) and isinstance(raw, dict):
        return {
            key: _preserve_env_ref_templates(
                value,
                raw.get(key),
                loaded_expanded.get(key) if isinstance(loaded_expanded, dict) else None,
            )
            for key, value in current.items()
        }

    if isinstance(current, list) and isinstance(raw, list):
        # Prefer matching named config objects (e.g. custom_providers) by name
        # so harmless reordering doesn't drop the original template. If names
        # are duplicated, fall back to positional matching instead of silently
        # shadowing one entry.
        current_by_name = _items_by_unique_name(current)
        raw_by_name = _items_by_unique_name(raw)
        loaded_by_name = _items_by_unique_name(loaded_expanded)
        if current_by_name is not None and raw_by_name is not None:
            return [
                _preserve_env_ref_templates(
                    item,
                    raw_by_name.get(item.get("name")),
                    loaded_by_name.get(item.get("name")) if loaded_by_name is not None else None,
                )
                for item in current
            ]
        return [
            _preserve_env_ref_templates(
                item,
                raw[index] if index < len(raw) else None,
                loaded_expanded[index]
                if isinstance(loaded_expanded, list) and index < len(loaded_expanded)
                else None,
            )
            for index, item in enumerate(current)
        ]

    return current


def _normalize_root_model_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Move stale root-level provider/base_url into model section.

    Some users (or older code) placed ``provider:`` and ``base_url:`` at the
    config root instead of inside ``model:``.  These root-level keys are only
    used as a fallback when the corresponding ``model.*`` key is empty — they
    never override an existing ``model.provider`` or ``model.base_url``.
    After migration the root-level keys are removed so they can't cause
    confusion on subsequent loads.
    """
    # Only act if there are root-level keys to migrate
    has_root = any(config.get(k) for k in ("provider", "base_url"))
    if not has_root:
        return config

    config = dict(config)
    model = config.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        config["model"] = model

    for key in ("provider", "base_url"):
        root_val = config.get(key)
        if root_val and not model.get(key):
            model[key] = root_val
        config.pop(key, None)

    return config


def _normalize_max_turns_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy root-level max_turns into agent.max_turns."""
    config = dict(config)
    agent_config = dict(config.get("agent") or {})

    if "max_turns" in config and "max_turns" not in agent_config:
        agent_config["max_turns"] = config["max_turns"]

    if "max_turns" not in agent_config:
        agent_config["max_turns"] = DEFAULT_CONFIG["agent"]["max_turns"]

    config["agent"] = agent_config
    config.pop("max_turns", None)
    return config


def _normalize_openagent_named_bucket(bucket: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize OpenAgent-style named agent/category mappings."""
    return _normalize_openagent_named_bucket_with_defaults(bucket)


def _normalize_openagent_named_bucket_with_defaults(
    bucket: Any,
    defaults: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(bucket, dict):
        bucket = {}

    normalized: Dict[str, Dict[str, Any]] = {}
    if isinstance(defaults, dict):
        for raw_name, entry in defaults.items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            normalized[name] = _normalize_openagent_named_entry(entry)

    alias_map = _build_named_agent_alias_map(normalized)
    for raw_name, entry in bucket.items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        name = alias_map.get(name, name)
        base = dict(normalized.get(name, {}))
        normalized[name] = _normalize_openagent_named_entry(entry, defaults=base)
    return normalized


def _build_named_agent_alias_map(registry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Map every explicit secondary alias to its canonical agent name."""
    alias_map: Dict[str, str] = {}
    for canonical_name, entry in registry.items():
        for alias in _normalize_named_string_list(entry.get("aliases")):
            alias_map.setdefault(alias, canonical_name)
    return alias_map


def _canonicalize_named_agent_lookup_key(value: Any) -> str:
    """Normalize named-agent invocation tokens for stable lookup."""
    token = str(value or "").strip().lower()
    if not token:
        return ""
    token = re.sub(r"[\s_]+", "-", token)
    token = re.sub(r"-+", "-", token)
    return token.strip("-")


def _build_named_agent_lookup_map(registry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Map canonical/alias invocation variants onto canonical names."""
    lookup: Dict[str, str] = {}
    for canonical_name, entry in registry.items():
        lookup.setdefault(_canonicalize_named_agent_lookup_key(canonical_name), canonical_name)
        for alias in _normalize_named_string_list(entry.get("aliases")):
            lookup.setdefault(_canonicalize_named_agent_lookup_key(alias), canonical_name)
    return lookup


def _resolve_named_agent_disabled_state(
    canonical_name: str,
    *,
    registry: Dict[str, Dict[str, Any]],
    config: Optional[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Return whether an agent is disabled and which canonical surface disabled it."""
    source = config if isinstance(config, dict) else {}
    lookup_map = _build_named_agent_lookup_map(registry)

    for entry in source.get("disabled_agents") or []:
        disabled_name = lookup_map.get(_canonicalize_named_agent_lookup_key(entry))
        if disabled_name == canonical_name:
            return True, "disabled_agents"

    raw_agents = source.get("agents")
    if isinstance(raw_agents, dict):
        for raw_name, raw_entry in raw_agents.items():
            if not isinstance(raw_entry, dict) or not raw_entry.get("disable"):
                continue
            disabled_name = lookup_map.get(_canonicalize_named_agent_lookup_key(raw_name))
            if disabled_name == canonical_name:
                return True, f"agents.{canonical_name}.disable"

    return False, None


_NAMED_AGENT_PERMISSION_KEYS = (
    "edit",
    "bash",
    "webfetch",
    "doom_loop",
    "external_directory",
)
_NAMED_AGENT_PERMISSION_VALUES = {"ask", "allow", "deny"}
_NAMED_AGENT_FALLBACK_MODEL_KEYS = (
    "model",
    "variant",
    "reasoningEffort",
    "temperature",
    "top_p",
    "maxTokens",
    "thinking",
)


def _normalize_named_agent_permission_surface(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize the bounded upstream-style named-agent permission surface."""
    if not isinstance(value, dict):
        return None

    normalized: Dict[str, Any] = {}
    for key in _NAMED_AGENT_PERMISSION_KEYS:
        if key not in value:
            continue
        raw = value.get(key)
        if isinstance(raw, str):
            permission_value = raw.strip().lower()
            if permission_value in _NAMED_AGENT_PERMISSION_VALUES:
                normalized[key] = permission_value
            continue
        if key == "bash" and isinstance(raw, dict):
            per_command: Dict[str, str] = {}
            for raw_command, raw_permission in raw.items():
                command = str(raw_command or "").strip()
                permission_value = str(raw_permission or "").strip().lower()
                if command and permission_value in _NAMED_AGENT_PERMISSION_VALUES:
                    per_command[command] = permission_value
            if per_command:
                normalized[key] = per_command
    return normalized or None


def _normalize_named_agent_fallback_models(value: Any) -> List[Dict[str, Any]]:
    """Normalize fallback_models into a bounded list of model entries."""
    entries = value if isinstance(value, list) else [value]
    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            model = entry.strip()
            if model:
                normalized.append({"model": model})
            continue
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("model") or "").strip()
        if not model:
            continue
        normalized_entry: Dict[str, Any] = {"model": model}
        for key in _NAMED_AGENT_FALLBACK_MODEL_KEYS:
            if key == "model" or key not in entry:
                continue
            raw = entry.get(key)
            if raw in (None, ""):
                continue
            if key == "thinking":
                if isinstance(raw, dict) and raw:
                    normalized_entry[key] = copy.deepcopy(raw)
                continue
            if isinstance(raw, str):
                stripped = raw.strip()
                if stripped:
                    normalized_entry[key] = stripped
                continue
            if isinstance(raw, (int, float)):
                normalized_entry[key] = raw
        normalized.append(normalized_entry)
    return normalized


def _normalize_named_agent_disable_flag(value: Any) -> bool:
    """Normalize a named-agent disable flag to a strict boolean."""
    return value is True



def _resolve_disabled_named_agents(
    registry: Dict[str, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[set, Dict[str, str]]:
    """Resolve disabled named agents from per-agent and global config surfaces."""
    lookup_map = _build_named_agent_lookup_map(registry)
    disabled_names = set()
    disabled_via: Dict[str, str] = {}

    for canonical_name, entry in registry.items():
        if _normalize_named_agent_disable_flag(entry.get("disable")):
            disabled_names.add(canonical_name)
            disabled_via.setdefault(canonical_name, f"agents.{canonical_name}.disable")

    raw_disabled_agents = []
    if isinstance(config, dict):
        raw_disabled_agents = _normalize_named_string_list(config.get("disabled_agents"))
    for raw_name in raw_disabled_agents:
        canonical_name = lookup_map.get(_canonicalize_named_agent_lookup_key(raw_name), raw_name)
        if canonical_name in registry:
            disabled_names.add(canonical_name)
            disabled_via[canonical_name] = "disabled_agents"

    return disabled_names, disabled_via


def _format_named_agent_permission_surface(permission_surface: Optional[Dict[str, Any]]) -> str:
    """Render the bounded named-agent permission surface for detail views."""
    if not permission_surface:
        return "-"

    parts: List[str] = []
    for key in _NAMED_AGENT_PERMISSION_KEYS:
        if key not in permission_surface:
            continue
        value = permission_surface[key]
        if isinstance(value, dict):
            command_summary = ",".join(
                f"{command}:{value[command]}"
                for command in sorted(value)
            )
            parts.append(f"{key}={command_summary}")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else "-"


def _format_named_agent_fallback_models(fallback_models: List[Dict[str, Any]]) -> str:
    """Render bounded fallback-model truth for detail views."""
    if not fallback_models:
        return "-"

    rendered_models: List[str] = []
    for entry in fallback_models:
        label = entry.get("model", "")
        extras: List[str] = []
        for key in _NAMED_AGENT_FALLBACK_MODEL_KEYS:
            if key == "model" or key not in entry:
                continue
            value = entry[key]
            if isinstance(value, dict):
                extras.append(f"{key}={value}")
            else:
                extras.append(f"{key}={value}")
        if extras:
            label = f"{label} [{' '.join(extras)}]"
        rendered_models.append(label)
    return " -> ".join(rendered_models)


def _normalize_openagent_named_entry(
    entry: Any,
    *,
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = copy.deepcopy(defaults or {})

    if isinstance(entry, str):
        model = entry.strip()
        if model:
            normalized["model"] = model
        return normalized

    if not isinstance(entry, dict):
        return normalized

    for key, value in entry.items():
        normalized[key] = copy.deepcopy(value)

    for key in (
        "model",
        "provider",
        "archetype",
        "specialist",
        "route_category",
        "runtime_mode",
        "color",
        "description",
        "base_url",
        "api_key",
        "reasoning_effort",
        "routing_description",
        "acp_command",
    ):
        if key in entry:
            value = str(entry.get(key) or "").strip()
            if value:
                normalized[key] = value
            else:
                normalized.pop(key, None)

    for key in ("blocked_tools", "allowed_tools", "toolsets", "enabled_tools", "acp_args", "aliases"):
        if key in entry:
            value = _normalize_named_string_list(entry.get(key))
            if value:
                normalized[key] = value
            else:
                normalized.pop(key, None)

    if "provider_options" in entry:
        provider_options = entry.get("provider_options")
        if isinstance(provider_options, dict):
            normalized["provider_options"] = copy.deepcopy(provider_options)
        else:
            normalized.pop("provider_options", None)

    if "permission" in entry:
        permission_surface = _normalize_named_agent_permission_surface(entry.get("permission"))
        if permission_surface:
            normalized["permission"] = permission_surface
        else:
            normalized.pop("permission", None)

    if "fallback_models" in entry:
        fallback_models = _normalize_named_agent_fallback_models(entry.get("fallback_models"))
        if fallback_models:
            normalized["fallback_models"] = fallback_models
        else:
            normalized.pop("fallback_models", None)

    if "disable" in entry:
        if _normalize_named_agent_disable_flag(entry.get("disable")):
            normalized["disable"] = True
        else:
            normalized.pop("disable", None)

    if "ultrawork" in entry:
        ultrawork = entry.get("ultrawork")
        if isinstance(ultrawork, dict):
            normalized_ultrawork = {
                key: str(ultrawork.get(key) or "").strip()
                for key in ("model", "variant")
                if str(ultrawork.get(key) or "").strip()
            }
            if normalized_ultrawork:
                normalized["ultrawork"] = normalized_ultrawork
            else:
                normalized.pop("ultrawork", None)
        else:
            normalized.pop("ultrawork", None)

    if "mode" in entry:
        mode = str(entry.get("mode") or "").strip().lower()
        if mode in {"primary", "subagent", "disabled"}:
            normalized["mode"] = mode
        else:
            normalized.pop("mode", None)

    return normalized


def _normalize_named_agent_registry(bucket: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize named agents, merging user config onto built-in OMO defaults."""
    return _normalize_openagent_named_bucket_with_defaults(bucket, DEFAULT_OMO_AGENTS)


def get_named_agent_registry(config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Return the normalized named-agent registry from config."""
    source = load_config() if config is None else config
    return _normalize_named_agent_registry(source.get("agents"))


def describe_named_agent(name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a resolved, user-facing description of a named agent."""
    source = config if isinstance(config, dict) else load_config()
    registry = get_named_agent_registry(source)
    requested_name = str(name or "").strip()
    alias_map = _build_named_agent_alias_map(registry)
    disabled_names, disabled_via = _resolve_disabled_named_agents(registry, source)
    canonical_name = alias_map.get(requested_name, requested_name)
    if canonical_name == requested_name:
        lookup_map = _build_named_agent_lookup_map(registry)
        canonical_name = lookup_map.get(_canonicalize_named_agent_lookup_key(requested_name), requested_name)
    entry = registry.get(canonical_name)
    if not entry:
        available = sorted({*registry, *alias_map})
        hint = ""
        matches = difflib.get_close_matches(requested_name, available, n=3, cutoff=0.5)
        if matches:
            hint = f" Did you mean: {', '.join(matches)}?"
        raise KeyError(f"Unknown named agent '{requested_name}'.{hint}")

    specialist = str(entry.get("specialist") or "").strip() or None
    specialist_mapping = resolve_specialist_mapping(specialist)
    resolved_archetype = str(entry.get("archetype") or "").strip() or (
        specialist_mapping.archetype_name if specialist_mapping is not None else DEFAULT_ARCHETYPE_NAME
    )
    archetype_defaults = resolve_archetype_defaults(resolved_archetype)
    blocked_tools, allowed_tools = get_tool_restrictions(resolved_archetype, specialist)

    named_blocked = set(_normalize_named_string_list(entry.get("blocked_tools")))
    named_allowed = set(_normalize_named_string_list(entry.get("allowed_tools")))
    effective_blocked = set(blocked_tools) | named_blocked
    effective_allowed = set(allowed_tools)
    if named_allowed:
        effective_allowed = effective_allowed.intersection(named_allowed) if effective_allowed else set(named_allowed)
    effective_allowed.difference_update(effective_blocked)
    reviewer_like = bool((specialist or "").strip().lower() in {"code_reviewer", "qa_guard"} or resolved_archetype == "verifier")
    configured_permission_surface = _normalize_named_agent_permission_surface(entry.get("permission"))
    configured_fallback_models = _normalize_named_agent_fallback_models(entry.get("fallback_models"))

    return {
        "name": canonical_name,
        "requested_name": requested_name or canonical_name,
        "aliases": _normalize_named_string_list(entry.get("aliases")),
        "description": str(entry.get("description") or "").strip(),
        "mode": str(entry.get("mode") or "subagent").strip() or "subagent",
        "provider": str(entry.get("provider") or "").strip(),
        "model": str(entry.get("model") or "").strip(),
        "resolved_archetype": resolved_archetype,
        "resolved_specialist": specialist,
        "resolved_route_category": str(entry.get("route_category") or archetype_defaults.get("default_route_category") or DEFAULT_ROUTE_CATEGORY).strip(),
        "resolved_runtime_mode": str(entry.get("runtime_mode") or DEFAULT_RUNTIME_MODE_NAME).strip(),
        "toolsets": _normalize_named_string_list(entry.get("toolsets")),
        "enabled_tools": _normalize_named_string_list(entry.get("enabled_tools")),
        "named_allowed_tools": sorted(named_allowed),
        "named_blocked_tools": sorted(named_blocked),
        "effective_allowed_tools": sorted(effective_allowed),
        "effective_blocked_tools": sorted(effective_blocked),
        "configured_permission_surface": configured_permission_surface,
        "configured_fallback_models": configured_fallback_models,
        "permission_truth_surface": "configured_named_agent_surface" if configured_permission_surface else None,
        "fallback_truth_surface": "configured_named_agent_surface" if configured_fallback_models else None,
        "completion_gate": "verification_evidence_required" if reviewer_like else None,
        "behavior_boundary": "reviewer_read_only" if reviewer_like else None,
        "is_disabled": canonical_name in disabled_names,
        "disabled_via": disabled_via.get(canonical_name),
    }


def render_named_agents_text(name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    """Render a CLI/gateway-friendly named-agent overview or detail view."""
    source = config if isinstance(config, dict) else load_config()
    registry = get_named_agent_registry(source)
    requested_name = str(name or "").strip()
    if requested_name:
        desc = describe_named_agent(requested_name, config=source)
        lines = [f"Named agent: {desc['name']}"]
        if desc["requested_name"] != desc["name"]:
            lines.append(f"Requested as: {desc['requested_name']}")
        if desc["aliases"]:
            lines.append(f"Aliases: {', '.join(desc['aliases'])}")
        if desc["description"]:
            lines.append(f"Description: {desc['description']}")
        lines.append(f"Status: {'disabled' if desc['is_disabled'] else 'enabled'}")
        if desc["disabled_via"]:
            lines.append(f"Disabled via: {desc['disabled_via']}")
        lines.extend(
            [
                f"Mode: {desc['mode']}",
                f"Invocation: @{desc['name']} <prompt> (leading mention only)",
                f"Archetype: {desc['resolved_archetype']}",
                f"Specialist: {desc['resolved_specialist'] or '-'}",
                f"Route category: {desc['resolved_route_category']}",
                f"Runtime mode: {desc['resolved_runtime_mode']}",
                f"Provider: {desc['provider'] or '-'}",
                f"Model: {desc['model'] or '-'}",
                f"Toolsets: {', '.join(desc['toolsets']) if desc['toolsets'] else '-'}",
                f"Enabled tools: {', '.join(desc['enabled_tools']) if desc['enabled_tools'] else '-'}",
                f"Named allowed tools: {', '.join(desc['named_allowed_tools']) if desc['named_allowed_tools'] else '-'}",
                f"Named blocked tools: {', '.join(desc['named_blocked_tools']) if desc['named_blocked_tools'] else '-'}",
                f"Effective allowed tools: {', '.join(desc['effective_allowed_tools']) if desc['effective_allowed_tools'] else '-'}",
                f"Effective blocked tools: {', '.join(desc['effective_blocked_tools']) if desc['effective_blocked_tools'] else '-'}",
                f"Configured permission surface: {_format_named_agent_permission_surface(desc['configured_permission_surface'])}",
                f"Permission truth surface: {desc['permission_truth_surface'] or '-'}",
                f"Configured fallback models: {_format_named_agent_fallback_models(desc['configured_fallback_models'])}",
                f"Fallback truth surface: {desc['fallback_truth_surface'] or '-'}",
                f"Behavior boundary: {desc['behavior_boundary'] or '-'}",
                f"Completion gate: {desc['completion_gate'] or '-'}",
            ]
        )
        return "\n".join(lines)

    visible_agent_names = [
        agent_name for agent_name in registry
        if not describe_named_agent(agent_name, config=source)["is_disabled"]
    ]
    lines = [f"Named agents ({len(visible_agent_names)}):"]
    for agent_name in visible_agent_names:
        desc = describe_named_agent(agent_name, config=source)
        label = agent_name
        if desc["aliases"]:
            label = f"{label} (aliases: {', '.join(desc['aliases'])})"
        summary = [desc["resolved_archetype"]]
        if desc["resolved_specialist"]:
            summary.append(desc["resolved_specialist"])
        summary.append(desc["mode"])
        if desc["model"]:
            summary.append(desc["model"])
        restriction = ""
        if desc["effective_allowed_tools"]:
            restriction = f" · allow={len(desc['effective_allowed_tools'])}"
        elif desc["effective_blocked_tools"]:
            restriction = f" · block={len(desc['effective_blocked_tools'])}"
        lines.append(f"- {label} · {' · '.join(summary)}{restriction}")
    lines.append("Use /named-agents <name> to inspect one agent.")
    return "\n".join(lines)


def _normalize_runtime_fallback_config(config: Any) -> Dict[str, Any]:
    """Normalize runtime_fallback from bool-or-dict into dict form."""
    if isinstance(config, bool):
        return {"enabled": config}
    if isinstance(config, dict):
        return dict(config)
    return {"enabled": False}


def _normalize_model_capabilities_config(config: Any) -> Dict[str, Any]:
    """Normalize model_capabilities to a mapping or empty dict."""
    return dict(config) if isinstance(config, dict) else {}


def _normalize_named_string_list(value: Any) -> List[str]:
    """Normalize a list/tuple of names into trimmed unique strings."""
    if not isinstance(value, (list, tuple)):
        return []
    normalized: List[str] = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _normalize_wave1_route_categories(route_categories: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize built-in route-category metadata overrides while preserving built-ins."""
    normalized: Dict[str, Dict[str, Any]] = {
        name: {
            "summary": category.summary,
            "intensity": category.intensity,
        }
        for name, category in BUILTIN_ROUTE_CATEGORIES.items()
    }
    if not isinstance(route_categories, dict):
        return normalized

    for raw_name, entry in route_categories.items():
        name = str(raw_name or "").strip()
        if not name or name not in normalized:
            continue
        if isinstance(entry, str):
            normalized[name] = {
                "summary": entry.strip(),
                "intensity": normalized.get(name, {}).get("intensity", ""),
            }
            continue
        if not isinstance(entry, dict):
            continue
        merged = dict(normalized.get(name, {}))
        if "summary" in entry and entry["summary"] is not None:
            merged["summary"] = str(entry["summary"]).strip()
        if "intensity" in entry and entry["intensity"] is not None:
            merged["intensity"] = str(entry["intensity"]).strip()
        normalized[name] = merged
    return normalized


def _normalize_wave1_runtime_modes(runtime_modes: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize built-in runtime-mode metadata overrides while preserving built-ins."""
    normalized: Dict[str, Dict[str, Any]] = {
        mode.name: {
            "description": mode.description,
            "operating_posture": mode.operating_posture,
            "kind": mode.kind,
        }
        for mode in BUILTIN_RUNTIME_MODES
    }
    if not isinstance(runtime_modes, dict):
        return normalized

    for raw_name, entry in runtime_modes.items():
        name = str(raw_name or "").strip()
        if not name or name not in normalized:
            continue
        if not isinstance(entry, dict):
            continue
        merged = dict(normalized.get(name, {}))
        for field_name in ("description", "operating_posture", "kind"):
            if field_name in entry and entry[field_name] is not None:
                merged[field_name] = str(entry[field_name]).strip()
        if not merged.get("kind"):
            merged["kind"] = "runtime_mode"
        normalized[name] = merged
    return normalized


def _collect_unknown_wave1_metadata_names(entries: Any, known_names: List[str]) -> List[str]:
    """Return trimmed unknown metadata-override names in stable order."""
    if not isinstance(entries, dict):
        return []

    known_name_set = {name.strip() for name in known_names if str(name).strip()}
    unknown_names: List[str] = []
    seen_names = set()
    for raw_name in entries:
        name = str(raw_name or "").strip()
        if not name or name in known_name_set or name in seen_names:
            continue
        seen_names.add(name)
        unknown_names.append(name)
    return unknown_names


def _wave1_effective_metadata_override_names(
    defaults: Dict[str, Dict[str, Any]],
    normalized: Any,
) -> List[str]:
    """Return built-in metadata names whose effective values differ from defaults."""
    if not isinstance(normalized, dict):
        return []

    override_names: List[str] = []
    for name, default_entry in defaults.items():
        if normalized.get(name) != default_entry:
            override_names.append(name)
    return override_names


def inspect_wave1_delegation_metadata(
    config: Optional[Dict[str, Any]] = None,
    raw_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """Inspect effective and ignored Wave 1 delegation metadata overrides."""
    if config is None:
        config = load_config()
    if raw_config is None:
        raw_config = read_raw_config()

    normalized_delegation = config.get("delegation") if isinstance(config, dict) else {}
    if not isinstance(normalized_delegation, dict):
        normalized_delegation = {}

    raw_delegation = raw_config.get("delegation") if isinstance(raw_config, dict) else {}
    if not isinstance(raw_delegation, dict):
        raw_delegation = {}

    default_route_categories = {
        name: {
            "summary": category.summary,
            "intensity": category.intensity,
        }
        for name, category in BUILTIN_ROUTE_CATEGORIES.items()
    }
    default_runtime_modes = {
        mode.name: {
            "description": mode.description,
            "operating_posture": mode.operating_posture,
            "kind": mode.kind,
        }
        for mode in BUILTIN_RUNTIME_MODES
    }

    return {
        "effective_route_category_override_names": _wave1_effective_metadata_override_names(
            default_route_categories,
            normalized_delegation.get("route_categories"),
        ),
        "ignored_route_category_names": _collect_unknown_wave1_metadata_names(
            raw_delegation.get("route_categories"),
            list(default_route_categories.keys()),
        ),
        "effective_runtime_mode_override_names": _wave1_effective_metadata_override_names(
            default_runtime_modes,
            normalized_delegation.get("runtime_modes"),
        ),
        "ignored_runtime_mode_names": _collect_unknown_wave1_metadata_names(
            raw_delegation.get("runtime_modes"),
            list(default_runtime_modes.keys()),
        ),
    }


def _normalize_wave1_task_contract(task_contract: Any) -> Optional[Dict[str, Any]]:
    """Normalize a configured task contract or raise with migration guidance."""
    if task_contract in (None, "", {}):
        return None
    if not isinstance(task_contract, dict):
        raise ValueError(
            "delegation.task_contract must be a dict matching the Wave 1 task-contract schema. "
            "Migration guidance: either remove delegation.task_contract to keep the legacy default-mapping path, "
            f"or provide all required fields: {', '.join(REQUIRED_TASK_CONTRACT_FIELDS)}"
        )
    try:
        return validate_task_contract(task_contract).model_dump()
    except Exception as exc:
        raise ValueError(
            "Invalid delegation.task_contract in config. Migration guidance: remove delegation.task_contract "
            "to preserve legacy delegation behavior, or update it to include the required structured fields "
            f"({', '.join(REQUIRED_TASK_CONTRACT_FIELDS)}). Original error: {exc}"
        ) from exc


def _normalize_wave1_delegation_config(
    config: Dict[str, Any], *, include_defaults: bool = True,
) -> Dict[str, Any]:
    """Resolve Wave 1 delegation surfaces into compatibility-safe defaults."""
    config = dict(config)
    if not include_defaults and "delegation" not in config:
        return config

    delegation = config.get("delegation")
    if not isinstance(delegation, dict):
        delegation = {}
    else:
        delegation = dict(delegation)

    requested_archetype = delegation.get("archetype")
    resolved_archetype = resolve_archetype(requested_archetype)
    delegation["archetype"] = resolved_archetype.name

    archetype_overrides = {}
    for field_name in REQUIRED_ARCHETYPE_FIELDS:
        if field_name not in delegation or delegation[field_name] is None:
            continue
        value = delegation[field_name]
        if field_name in {"default_skills", "default_required_tools"}:
            archetype_overrides[field_name] = _normalize_named_string_list(value)
        else:
            archetype_overrides[field_name] = value
    archetype_defaults = resolve_archetype_defaults(
        resolved_archetype.name,
        overrides=archetype_overrides,
    )

    explicit_category_input = str(
        delegation.get("category")
        or delegation.get("default_category")
        or ""
    ).strip()
    resolved_literal_category = resolve_literal_category(explicit_category_input)

    explicit_route_category = str(
        delegation.get("route_category")
        or delegation.get("default_route_category")
        or ""
    ).strip()
    resolved_route_category = (
        explicit_route_category
        if explicit_route_category in BUILTIN_ROUTE_CATEGORIES
        else resolved_literal_category.route_category
        or str(archetype_defaults["default_route_category"]).strip()
        or DEFAULT_ROUTE_CATEGORY
    )

    runtime_mode = resolve_runtime_mode(delegation.get("runtime_mode"))
    delegation["category"] = resolved_literal_category.name
    delegation["runtime_mode"] = runtime_mode.name
    delegation["route_category"] = resolved_route_category
    delegation["default_route_category"] = resolved_route_category
    delegation["default_category"] = resolved_literal_category.name
    resolved_default_profile = str(
        delegation.get("default_delegation_profile")
        or archetype_defaults["default_delegation_profile"]
    ).strip()
    delegation["default_delegation_profile"] = resolved_default_profile
    delegation["default_skills"] = _normalize_named_string_list(
        delegation.get("default_skills") or archetype_defaults["default_skills"]
    )
    delegation["default_required_tools"] = _normalize_named_string_list(
        delegation.get("default_required_tools") or archetype_defaults["default_required_tools"]
    )
    delegation["permission_preset"] = str(
        delegation.get("permission_preset") or archetype_defaults["permission_preset"]
    ).strip()
    delegation["fallback_policy"] = str(
        delegation.get("fallback_policy") or archetype_defaults["fallback_policy"]
    ).strip()
    delegation["task_contract"] = _normalize_wave1_task_contract(delegation.get("task_contract"))
    delegation["route_categories"] = _normalize_wave1_route_categories(delegation.get("route_categories"))
    delegation["runtime_modes"] = _normalize_wave1_runtime_modes(delegation.get("runtime_modes"))
    normalized_profiles: Dict[str, Dict[str, Any]] = {}
    for bucket_name in ("delegation_profiles", "categories"):
        bucket = delegation.get(bucket_name)
        if not isinstance(bucket, dict):
            continue
        for raw_name, entry in _normalize_openagent_named_bucket(bucket).items():
            merged = dict(normalized_profiles.get(raw_name, {}))
            if isinstance(entry, dict):
                merged.update(entry)
            normalized_profiles[raw_name] = merged
    delegation["delegation_profiles"] = normalized_profiles
    delegation["categories"] = copy.deepcopy(normalized_profiles)
    delegation["archetype_defaults"] = dict(archetype_defaults)

    config["delegation"] = delegation
    return config


def _normalize_openagent_config_buckets(
    config: Dict[str, Any], *, include_defaults: bool = True,
) -> Dict[str, Any]:
    """Normalize OpenAgent-style config buckets while preserving shorthand."""
    config = dict(config)

    if include_defaults or "agents" in config:
        config["agents"] = _normalize_named_agent_registry(config.get("agents"))
    if include_defaults or "categories" in config:
        config["categories"] = _normalize_openagent_named_bucket(config.get("categories"))
    if include_defaults or "runtime_fallback" in config:
        config["runtime_fallback"] = _normalize_runtime_fallback_config(config.get("runtime_fallback"))
    if include_defaults or "model_capabilities" in config:
        config["model_capabilities"] = _normalize_model_capabilities_config(config.get("model_capabilities"))

    return config


def read_raw_config() -> Dict[str, Any]:
    """Read ~/.hermes/config.yaml as-is, without merging defaults or migrating.

    Returns the raw YAML dict, or ``{}`` if the file doesn't exist or can't
    be parsed.  Use this for lightweight config reads where you just need a
    single value and don't want the overhead of ``load_config()``'s deep-merge
    + migration pipeline.
    """
    try:
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def load_config() -> Dict[str, Any]:
    """Load configuration from ~/.hermes/config.yaml."""
    ensure_hermes_home()
    config_path = get_config_path()
    
    config = copy.deepcopy(DEFAULT_CONFIG)
    
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}

            if "max_turns" in user_config:
                agent_user_config = dict(user_config.get("agent") or {})
                if agent_user_config.get("max_turns") is None:
                    agent_user_config["max_turns"] = user_config["max_turns"]
                user_config["agent"] = agent_user_config
                user_config.pop("max_turns", None)

            config = _deep_merge(config, user_config)
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")

    normalized = _normalize_profile_path_settings(
        _expand_env_vars(
            _normalize_wave1_delegation_config(
                _normalize_openagent_config_buckets(
                    _normalize_root_model_keys(_normalize_max_turns_config(config))
                )
            )
        )
    )
    _LAST_EXPANDED_CONFIG_BY_PATH[str(config_path)] = copy.deepcopy(normalized)
    return normalized


_SECURITY_COMMENT = """
# ── Security ──────────────────────────────────────────────────────────
# API keys, tokens, and passwords are redacted from tool output by default.
# Set to false to see full values (useful for debugging auth issues).
# tirith pre-exec scanning is enabled by default when the tirith binary
# is available. Configure via security.tirith_* keys or env vars
# (TIRITH_ENABLED, TIRITH_BIN, TIRITH_TIMEOUT, TIRITH_FAIL_OPEN).
#
# security:
#   redact_secrets: false
#   tirith_enabled: true
#   tirith_path: "tirith"
#   tirith_timeout: 5
#   tirith_fail_open: true
"""

_FALLBACK_COMMENT = """
# ── Fallback Model ────────────────────────────────────────────────────
# Automatic provider failover when primary is unavailable.
# Uncomment and configure to enable. Triggers on rate limits (429),
# overload (529), service errors (503), or connection failures.
#
# Supported providers:
#   openrouter   (OPENROUTER_API_KEY)  — routes to any model
#   openai-codex (OAuth — hermes auth) — OpenAI Codex
#   nous         (OAuth — hermes auth) — Nous Portal
#   zai          (ZAI_API_KEY)         — Z.AI / GLM
#   kimi-coding  (KIMI_API_KEY)        — Kimi / Moonshot
#   kimi-coding-cn (KIMI_CN_API_KEY)   — Kimi / Moonshot (China)
#   minimax      (MINIMAX_API_KEY)     — MiniMax
#   minimax-cn   (MINIMAX_CN_API_KEY)  — MiniMax (China)
#
# For custom OpenAI-compatible endpoints, add base_url and api_key_env.
#
# fallback_model:
#   provider: openrouter
#   model: anthropic/claude-sonnet-4
#
# ── Smart Model Routing ────────────────────────────────────────────────
# Optional cheap-vs-strong routing for simple turns.
# Keeps the primary model for complex work, but can route short/simple
# messages to a cheaper model across providers.
#
# smart_model_routing:
#   enabled: true
#   max_simple_chars: 160
#   max_simple_words: 28
#   cheap_model:
#     provider: openrouter
#     model: google/gemini-2.5-flash
"""


_COMMENTED_SECTIONS = """
# ── Security ──────────────────────────────────────────────────────────
# API keys, tokens, and passwords are redacted from tool output by default.
# Set to false to see full values (useful for debugging auth issues).
#
# security:
#   redact_secrets: false

# ── Fallback Model ────────────────────────────────────────────────────
# Automatic provider failover when primary is unavailable.
# Uncomment and configure to enable. Triggers on rate limits (429),
# overload (529), service errors (503), or connection failures.
#
# Supported providers:
#   openrouter   (OPENROUTER_API_KEY)  — routes to any model
#   openai-codex (OAuth — hermes auth) — OpenAI Codex
#   nous         (OAuth — hermes auth) — Nous Portal
#   zai          (ZAI_API_KEY)         — Z.AI / GLM
#   kimi-coding  (KIMI_API_KEY)        — Kimi / Moonshot
#   kimi-coding-cn (KIMI_CN_API_KEY)   — Kimi / Moonshot (China)
#   minimax      (MINIMAX_API_KEY)     — MiniMax
#   minimax-cn   (MINIMAX_CN_API_KEY)  — MiniMax (China)
#
# For custom OpenAI-compatible endpoints, add base_url and api_key_env.
#
# fallback_model:
#   provider: openrouter
#   model: anthropic/claude-sonnet-4
#
# ── Smart Model Routing ────────────────────────────────────────────────
# Optional cheap-vs-strong routing for simple turns.
# Keeps the primary model for complex work, but can route short/simple
# messages to a cheaper model across providers.
#
# smart_model_routing:
#   enabled: true
#   max_simple_chars: 160
#   max_simple_words: 28
#   cheap_model:
#     provider: openrouter
#     model: google/gemini-2.5-flash
"""


def save_config(config: Dict[str, Any]):
    """Save configuration to ~/.hermes/config.yaml."""
    if is_managed():
        managed_error("save configuration")
        return
    from utils import atomic_yaml_write

    ensure_hermes_home()
    config_path = get_config_path()
    current_normalized = _normalize_profile_path_settings(
        _normalize_wave1_delegation_config(
            _normalize_openagent_config_buckets(
                _normalize_root_model_keys(_normalize_max_turns_config(config)),
                include_defaults=False,
            ),
            include_defaults=False,
        )
    )
    normalized = current_normalized
    raw_existing = _normalize_profile_path_settings(
        _normalize_wave1_delegation_config(
            _normalize_openagent_config_buckets(
                _normalize_root_model_keys(_normalize_max_turns_config(read_raw_config())),
                include_defaults=False,
            ),
            include_defaults=False,
        )
    )
    if raw_existing:
        normalized = _preserve_env_ref_templates(
            normalized,
            raw_existing,
            _LAST_EXPANDED_CONFIG_BY_PATH.get(str(config_path)),
        )

    # Build optional commented-out sections for features that are off by
    # default or only relevant when explicitly configured.
    parts = []
    sec = normalized.get("security", {})
    if not sec or sec.get("redact_secrets") is None:
        parts.append(_SECURITY_COMMENT)
    fb = normalized.get("fallback_model", {})
    if not fb or not (fb.get("provider") and fb.get("model")):
        parts.append(_FALLBACK_COMMENT)

    atomic_yaml_write(
        config_path,
        normalized,
        extra_content="".join(parts) if parts else None,
    )
    _secure_file(config_path)
    _LAST_EXPANDED_CONFIG_BY_PATH[str(config_path)] = copy.deepcopy(current_normalized)


def load_env() -> Dict[str, str]:
    """Load environment variables from ~/.hermes/.env.

    Sanitizes lines before parsing so that corrupted files (e.g.
    concatenated KEY=VALUE pairs on a single line) are handled
    gracefully instead of producing mangled values such as duplicated
    bot tokens.  See #8908.
    """
    env_path = get_env_path()
    env_vars = {}
    
    if env_path.exists():
        # On Windows, open() defaults to the system locale (cp1252) which can
        # fail on UTF-8 .env files. Use explicit UTF-8 only on Windows.
        open_kw = {"encoding": "utf-8", "errors": "replace"} if _IS_WINDOWS else {}
        with open(env_path, **open_kw) as f:
            raw_lines = f.readlines()
        # Sanitize before parsing: split concatenated lines & drop stale
        # placeholders so corrupted .env files don't produce invalid tokens.
        lines = _sanitize_env_lines(raw_lines)
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                parsed_key = key.strip()
                parsed_value = value.strip().strip('"\'')
                if parsed_key in {"MESSAGING_CWD", "TERMINAL_CWD"}:
                    parsed_value = _normalize_profile_local_path(parsed_value)
                env_vars[parsed_key] = parsed_value
    
    return env_vars


def _sanitize_env_lines(lines: list) -> list:
    """Fix corrupted .env lines before reading or writing.

    Handles two known corruption patterns:
    1. Concatenated KEY=VALUE pairs on a single line (missing newline between
       entries, e.g. ``ANTHROPIC_API_KEY=sk-...OPENAI_BASE_URL=https://...``).
    2. Stale ``KEY=***`` placeholder entries left by incomplete setup runs.

    Uses a known-keys set (OPTIONAL_ENV_VARS + _EXTRA_ENV_KEYS) so we only
    split on real Hermes env var names, avoiding false positives from values
    that happen to contain uppercase text with ``=``.
    """
    # Build the known keys set lazily from OPTIONAL_ENV_VARS + extras.
    # Done inside the function so OPTIONAL_ENV_VARS is guaranteed to be defined.
    known_keys = set(OPTIONAL_ENV_VARS.keys()) | _EXTRA_ENV_KEYS

    sanitized: list[str] = []
    for line in lines:
        raw = line.rstrip("\r\n")
        stripped = raw.strip()

        # Preserve blank lines and comments
        if not stripped or stripped.startswith("#"):
            sanitized.append(raw + "\n")
            continue

        # Detect concatenated KEY=VALUE pairs on one line.
        # Search for known KEY= patterns at any position in the line.
        split_positions = []
        for key_name in known_keys:
            needle = key_name + "="
            idx = stripped.find(needle)
            while idx >= 0:
                split_positions.append(idx)
                idx = stripped.find(needle, idx + len(needle))

        if len(split_positions) > 1:
            split_positions.sort()
            # Deduplicate (shouldn't happen, but be safe)
            split_positions = sorted(set(split_positions))
            for i, pos in enumerate(split_positions):
                end = split_positions[i + 1] if i + 1 < len(split_positions) else len(stripped)
                part = stripped[pos:end].strip()
                if part:
                    sanitized.append(part + "\n")
        else:
            sanitized.append(stripped + "\n")

    return sanitized


def sanitize_env_file() -> int:
    """Read, sanitize, and rewrite ~/.hermes/.env in place.

    Returns the number of lines that were fixed (concatenation splits +
    placeholder removals).  Returns 0 when no changes are needed.
    """
    env_path = get_env_path()
    if not env_path.exists():
        return 0

    read_kw = {"encoding": "utf-8", "errors": "replace"} if _IS_WINDOWS else {}
    write_kw = {"encoding": "utf-8"} if _IS_WINDOWS else {}

    with open(env_path, **read_kw) as f:
        original_lines = f.readlines()

    sanitized = _sanitize_env_lines(original_lines)

    if sanitized == original_lines:
        return 0

    # Count fixes: difference in line count (from splits) + removed lines
    fixes = abs(len(sanitized) - len(original_lines))
    if fixes == 0:
        # Lines changed content (e.g. *** removal) even if count is same
        fixes = sum(1 for a, b in zip(original_lines, sanitized) if a != b)
        fixes += abs(len(sanitized) - len(original_lines))

    fd, tmp_path = tempfile.mkstemp(dir=str(env_path.parent), suffix=".tmp", prefix=".env_")
    try:
        with os.fdopen(fd, "w", **write_kw) as f:
            f.writelines(sanitized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, env_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    _secure_file(env_path)
    return fixes


def _check_non_ascii_credential(key: str, value: str) -> str:
    """Warn and strip non-ASCII characters from credential values.

    API keys and tokens must be pure ASCII — they are sent as HTTP header
    values which httpx/httpcore encode as ASCII.  Non-ASCII characters
    (commonly introduced by copy-pasting from rich-text editors or PDFs
    that substitute lookalike Unicode glyphs for ASCII letters) cause
    ``UnicodeEncodeError: 'ascii' codec can't encode character`` at
    request time.

    Returns the sanitized (ASCII-only) value.  Prints a warning if any
    non-ASCII characters were found and removed.
    """
    try:
        value.encode("ascii")
        return value  # all ASCII — nothing to do
    except UnicodeEncodeError:
        pass

    # Build a readable list of the offending characters
    bad_chars: list[str] = []
    for i, ch in enumerate(value):
        if ord(ch) > 127:
            bad_chars.append(f"  position {i}: {ch!r} (U+{ord(ch):04X})")
    sanitized = value.encode("ascii", errors="ignore").decode("ascii")

    import sys
    print(
        f"\n  Warning: {key} contains non-ASCII characters that will break API requests.\n"
        f"  This usually happens when copy-pasting from a PDF, rich-text editor,\n"
        f"  or web page that substitutes lookalike Unicode glyphs for ASCII letters.\n"
        f"\n"
        + "\n".join(f"  {line}" for line in bad_chars[:5])
        + ("\n  ... and more" if len(bad_chars) > 5 else "")
        + f"\n\n  The non-ASCII characters have been stripped automatically.\n"
        f"  If authentication fails, re-copy the key from the provider's dashboard.\n",
        file=sys.stderr,
    )
    return sanitized


def save_env_value(key: str, value: str):
    """Save or update a value in ~/.hermes/.env."""
    if is_managed():
        managed_error(f"set {key}")
        return
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    value = value.replace("\n", "").replace("\r", "")
    if key in {"MESSAGING_CWD", "TERMINAL_CWD"}:
        value = _normalize_profile_local_path(value)
    # API keys / tokens must be ASCII — strip non-ASCII with a warning.
    value = _check_non_ascii_credential(key, value)
    ensure_hermes_home()
    env_path = get_env_path()
    
    # On Windows, open() defaults to the system locale (cp1252) which can
    # cause OSError errno 22 on UTF-8 .env files.
    read_kw = {"encoding": "utf-8", "errors": "replace"} if _IS_WINDOWS else {}
    write_kw = {"encoding": "utf-8"} if _IS_WINDOWS else {}

    lines = []
    if env_path.exists():
        with open(env_path, **read_kw) as f:
            lines = f.readlines()
        # Sanitize on every read: split concatenated keys, drop stale placeholders
        lines = _sanitize_env_lines(lines)
    
    # Find and update or append
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    
    if not found:
        # Ensure there's a newline at the end of the file before appending
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={value}\n")
    
    fd, tmp_path = tempfile.mkstemp(dir=str(env_path.parent), suffix='.tmp', prefix='.env_')
    # Preserve original permissions so Docker volume mounts aren't clobbered.
    original_mode = None
    if env_path.exists():
        try:
            original_mode = stat.S_IMODE(env_path.stat().st_mode)
        except OSError:
            pass
    try:
        with os.fdopen(fd, 'w', **write_kw) as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, env_path)
        # Restore original permissions before _secure_file may tighten them.
        if original_mode is not None:
            try:
                os.chmod(env_path, original_mode)
            except OSError:
                pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    _secure_file(env_path)

    os.environ[key] = value


def remove_env_value(key: str) -> bool:
    """Remove a key from ~/.hermes/.env and os.environ.

    Returns True if the key was found and removed, False otherwise.
    """
    if is_managed():
        managed_error(f"remove {key}")
        return False
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    env_path = get_env_path()
    if not env_path.exists():
        os.environ.pop(key, None)
        return False

    read_kw = {"encoding": "utf-8", "errors": "replace"} if _IS_WINDOWS else {}
    write_kw = {"encoding": "utf-8"} if _IS_WINDOWS else {}

    with open(env_path, **read_kw) as f:
        lines = f.readlines()
    lines = _sanitize_env_lines(lines)

    new_lines = [line for line in lines if not line.strip().startswith(f"{key}=")]
    found = len(new_lines) < len(lines)

    if found:
        fd, tmp_path = tempfile.mkstemp(dir=str(env_path.parent), suffix='.tmp', prefix='.env_')
        # Preserve original permissions so Docker volume mounts aren't clobbered.
        original_mode = None
        try:
            original_mode = stat.S_IMODE(env_path.stat().st_mode)
        except OSError:
            pass
        try:
            with os.fdopen(fd, 'w', **write_kw) as f:
                f.writelines(new_lines)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, env_path)
            if original_mode is not None:
                try:
                    os.chmod(env_path, original_mode)
                except OSError:
                    pass
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        _secure_file(env_path)

    os.environ.pop(key, None)
    return found


def save_anthropic_oauth_token(value: str, save_fn=None):
    """Persist an Anthropic OAuth/setup token and clear the API-key slot."""
    writer = save_fn or save_env_value
    writer("ANTHROPIC_TOKEN", value)
    writer("ANTHROPIC_API_KEY", "")


def use_anthropic_claude_code_credentials(save_fn=None):
    """Use Claude Code's own credential files instead of persisting env tokens."""
    writer = save_fn or save_env_value
    writer("ANTHROPIC_TOKEN", "")
    writer("ANTHROPIC_API_KEY", "")


def save_anthropic_api_key(value: str, save_fn=None):
    """Persist an Anthropic API key and clear the OAuth/setup-token slot."""
    writer = save_fn or save_env_value
    writer("ANTHROPIC_API_KEY", value)
    writer("ANTHROPIC_TOKEN", "")


def save_env_value_secure(key: str, value: str) -> Dict[str, Any]:
    save_env_value(key, value)
    return {
        "success": True,
        "stored_as": key,
        "validated": False,
    }



def reload_env() -> int:
    """Re-read ~/.hermes/.env into os.environ. Returns count of vars updated.

    Adds/updates vars that changed and removes vars that were deleted from
    the .env file (but only vars known to Hermes — OPTIONAL_ENV_VARS and
    _EXTRA_ENV_KEYS — to avoid clobbering unrelated environment).
    """
    env_vars = load_env()
    known_keys = set(OPTIONAL_ENV_VARS.keys()) | _EXTRA_ENV_KEYS
    count = 0
    for key, value in env_vars.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            count += 1
    # Remove known Hermes vars that are no longer in .env
    for key in known_keys:
        if key not in env_vars and key in os.environ:
            del os.environ[key]
            count += 1
    return count


def get_env_value(key: str) -> Optional[str]:
    """Get a value from ~/.hermes/.env or environment."""
    # Check environment first
    if key in os.environ:
        return os.environ[key]
    
    # Then check .env file
    env_vars = load_env()
    return env_vars.get(key)


# =============================================================================
# Config display
# =============================================================================

def redact_key(key: str) -> str:
    """Redact an API key for display."""
    if not key:
        return color("(not set)", Colors.DIM)
    if len(key) < 12:
        return "***"
    return key[:4] + "..." + key[-4:]


def show_config():
    """Display current configuration."""
    config = load_config()
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│              ⚕ Hermes Configuration                    │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    
    # Paths
    print()
    print(color("◆ Paths", Colors.CYAN, Colors.BOLD))
    print(f"  Config:       {get_config_path()}")
    print(f"  Secrets:      {get_env_path()}")
    print(f"  Install:      {get_project_root()}")
    
    # API Keys
    print()
    print(color("◆ API Keys", Colors.CYAN, Colors.BOLD))
    
    keys = [
        ("OPENROUTER_API_KEY", "OpenRouter"),
        ("VOICE_TOOLS_OPENAI_KEY", "OpenAI (STT/TTS)"),
        ("EXA_API_KEY", "Exa"),
        ("PARALLEL_API_KEY", "Parallel"),
        ("FIRECRAWL_API_KEY", "Firecrawl"),
        ("TAVILY_API_KEY", "Tavily"),
        ("BROWSERBASE_API_KEY", "Browserbase"),
        ("BROWSER_USE_API_KEY", "Browser Use"),
        ("FAL_KEY", "FAL"),
    ]
    
    for env_key, name in keys:
        value = get_env_value(env_key)
        print(f"  {name:<14} {redact_key(value)}")
    from hermes_cli.auth import get_anthropic_key
    anthropic_value = get_anthropic_key()
    print(f"  {'Anthropic':<14} {redact_key(anthropic_value)}")
    
    # Model settings
    print()
    print(color("◆ Model", Colors.CYAN, Colors.BOLD))
    print(f"  Model:        {config.get('model', 'not set')}")
    print(f"  Max turns:    {config.get('agent', {}).get('max_turns', DEFAULT_CONFIG['agent']['max_turns'])}")

    # Delegation
    print()
    print(color("◆ Delegation", Colors.CYAN, Colors.BOLD))
    delegation = config.get('delegation', {}) if isinstance(config.get('delegation'), dict) else {}
    raw_config = read_raw_config()
    delegation_metadata = inspect_wave1_delegation_metadata(config, raw_config)
    print(f"  Default profile: {delegation.get('default_delegation_profile', 'general')}")
    print(f"  Category:        {delegation.get('category', DEFAULT_LITERAL_CATEGORY)}")
    print(f"  Route category:  {delegation.get('route_category', DEFAULT_ROUTE_CATEGORY)}")
    print(f"  Runtime mode:    {delegation.get('runtime_mode', DEFAULT_RUNTIME_MODE_NAME)}")

    effective_route_overrides = delegation_metadata['effective_route_category_override_names']
    effective_runtime_overrides = delegation_metadata['effective_runtime_mode_override_names']
    ignored_route_names = delegation_metadata['ignored_route_category_names']
    ignored_runtime_names = delegation_metadata['ignored_runtime_mode_names']
    print(f"  Route metadata overrides:   {', '.join(effective_route_overrides) if effective_route_overrides else color('(none)', Colors.DIM)}")
    print(f"  Runtime metadata overrides: {', '.join(effective_runtime_overrides) if effective_runtime_overrides else color('(none)', Colors.DIM)}")
    if ignored_route_names:
        print(f"  Ignored route metadata:     {', '.join(ignored_route_names)}")
    if ignored_runtime_names:
        print(f"  Ignored runtime metadata:   {', '.join(ignored_runtime_names)}")

    # Display
    print()
    print(color("◆ Display", Colors.CYAN, Colors.BOLD))
    display = config.get('display', {})
    print(f"  Personality:  {display.get('personality', 'kawaii')}")
    print(f"  Reasoning:    {'on' if display.get('show_reasoning', False) else 'off'}")
    print(f"  Bell:         {'on' if display.get('bell_on_complete', False) else 'off'}")

    # Terminal
    print()
    print(color("◆ Terminal", Colors.CYAN, Colors.BOLD))
    terminal = config.get('terminal', {})
    print(f"  Backend:      {terminal.get('backend', 'local')}")
    print(f"  Working dir:  {terminal.get('cwd', '.')}")
    print(f"  Timeout:      {terminal.get('timeout', 60)}s")
    
    if terminal.get('backend') == 'docker':
        print(f"  Docker image: {terminal.get('docker_image', 'nikolaik/python-nodejs:python3.11-nodejs20')}")
    elif terminal.get('backend') == 'singularity':
        print(f"  Image:        {terminal.get('singularity_image', 'docker://nikolaik/python-nodejs:python3.11-nodejs20')}")
    elif terminal.get('backend') == 'modal':
        print(f"  Modal image:  {terminal.get('modal_image', 'nikolaik/python-nodejs:python3.11-nodejs20')}")
        modal_token = get_env_value('MODAL_TOKEN_ID')
        print(f"  Modal token:  {'configured' if modal_token else '(not set)'}")
    elif terminal.get('backend') == 'daytona':
        print(f"  Daytona image: {terminal.get('daytona_image', 'nikolaik/python-nodejs:python3.11-nodejs20')}")
        daytona_key = get_env_value('DAYTONA_API_KEY')
        print(f"  API key:      {'configured' if daytona_key else '(not set)'}")
    elif terminal.get('backend') == 'ssh':
        ssh_host = get_env_value('TERMINAL_SSH_HOST')
        ssh_user = get_env_value('TERMINAL_SSH_USER')
        print(f"  SSH host:     {ssh_host or '(not set)'}")
        print(f"  SSH user:     {ssh_user or '(not set)'}")
    
    # Timezone
    print()
    print(color("◆ Timezone", Colors.CYAN, Colors.BOLD))
    tz = config.get('timezone', '')
    if tz:
        print(f"  Timezone:     {tz}")
    else:
        print(f"  Timezone:     {color('(server-local)', Colors.DIM)}")

    # Compression
    print()
    print(color("◆ Context Compression", Colors.CYAN, Colors.BOLD))
    compression = config.get('compression', {})
    enabled = compression.get('enabled', True)
    print(f"  Enabled:      {'yes' if enabled else 'no'}")
    if enabled:
        print(f"  Threshold:    {compression.get('threshold', 0.50) * 100:.0f}%")
        print(f"  Target ratio: {compression.get('target_ratio', 0.20) * 100:.0f}% of threshold preserved")
        print(f"  Protect last: {compression.get('protect_last_n', 20)} messages")
        _aux_comp = config.get('auxiliary', {}).get('compression', {})
        _sm = _aux_comp.get('model', '') or '(auto)'
        print(f"  Model:        {_sm}")
        comp_provider = _aux_comp.get('provider', 'auto')
        if comp_provider and comp_provider != 'auto':
            print(f"  Provider:     {comp_provider}")
    
    # Auxiliary models
    auxiliary = config.get('auxiliary', {})
    aux_tasks = {
        "Vision":      auxiliary.get('vision', {}),
        "Web extract": auxiliary.get('web_extract', {}),
    }
    has_overrides = any(
        t.get('provider', 'auto') != 'auto' or t.get('model', '')
        for t in aux_tasks.values()
    )
    if has_overrides:
        print()
        print(color("◆ Auxiliary Models (overrides)", Colors.CYAN, Colors.BOLD))
        for label, task_cfg in aux_tasks.items():
            prov = task_cfg.get('provider', 'auto')
            mdl = task_cfg.get('model', '')
            if prov != 'auto' or mdl:
                parts = [f"provider={prov}"]
                if mdl:
                    parts.append(f"model={mdl}")
                print(f"  {label:12s}  {', '.join(parts)}")
    
    # Messaging
    print()
    print(color("◆ Messaging Platforms", Colors.CYAN, Colors.BOLD))
    
    telegram_token = get_env_value('TELEGRAM_BOT_TOKEN')
    discord_token = get_env_value('DISCORD_BOT_TOKEN')
    
    print(f"  Telegram:     {'configured' if telegram_token else color('not configured', Colors.DIM)}")
    print(f"  Discord:      {'configured' if discord_token else color('not configured', Colors.DIM)}")
    
    # Skill config
    try:
        from agent.skill_utils import discover_all_skill_config_vars, resolve_skill_config_values
        skill_vars = discover_all_skill_config_vars()
        if skill_vars:
            resolved = resolve_skill_config_values(skill_vars)
            print()
            print(color("◆ Skill Settings", Colors.CYAN, Colors.BOLD))
            for var in skill_vars:
                key = var["key"]
                value = resolved.get(key, "")
                skill_name = var.get("skill", "")
                display_val = str(value) if value else color("(not set)", Colors.DIM)
                print(f"  {key:<20s} {display_val}  {color(f'[{skill_name}]', Colors.DIM)}")
    except Exception:
        pass

    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  hermes config edit     # Edit config file", Colors.DIM))
    print(color("  hermes config set <key> <value>", Colors.DIM))
    print(color("  hermes setup           # Run setup wizard", Colors.DIM))
    print()


def edit_config():
    """Open config file in user's editor."""
    if is_managed():
        managed_error("edit configuration")
        return
    config_path = get_config_path()
    
    # Ensure config exists
    if not config_path.exists():
        save_config(DEFAULT_CONFIG)
        print(f"Created {config_path}")
    
    # Find editor
    editor = os.getenv('EDITOR') or os.getenv('VISUAL')
    
    if not editor:
        # Try common editors
        for cmd in ['nano', 'vim', 'vi', 'code', 'notepad']:
            import shutil
            if shutil.which(cmd):
                editor = cmd
                break
    
    if not editor:
        print("No editor found. Config file is at:")
        print(f"  {config_path}")
        return
    
    print(f"Opening {config_path} in {editor}...")
    subprocess.run([editor, str(config_path)])


def set_config_value(key: str, value: str):
    """Set a configuration value."""
    if is_managed():
        managed_error("set configuration values")
        return
    # Check if it's an API key (goes to .env)
    api_keys = [
        'OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'VOICE_TOOLS_OPENAI_KEY',
        'EXA_API_KEY', 'PARALLEL_API_KEY', 'FIRECRAWL_API_KEY', 'FIRECRAWL_API_URL',
        'FIRECRAWL_GATEWAY_URL', 'TOOL_GATEWAY_DOMAIN', 'TOOL_GATEWAY_SCHEME',
        'TOOL_GATEWAY_USER_TOKEN', 'TAVILY_API_KEY',
        'BROWSERBASE_API_KEY', 'BROWSERBASE_PROJECT_ID', 'BROWSER_USE_API_KEY',
        'FAL_KEY', 'TELEGRAM_BOT_TOKEN', 'DISCORD_BOT_TOKEN',
        'TERMINAL_SSH_HOST', 'TERMINAL_SSH_USER', 'TERMINAL_SSH_KEY',
        'SUDO_PASSWORD', 'SLACK_BOT_TOKEN', 'SLACK_APP_TOKEN',
        'GITHUB_TOKEN', 'HONCHO_API_KEY', 'WANDB_API_KEY',
        'TINKER_API_KEY',
    ]
    
    if key.upper() in api_keys or key.upper().endswith(('_API_KEY', '_TOKEN')) or key.upper().startswith('TERMINAL_SSH'):
        save_env_value(key.upper(), value)
        print(f"✓ Set {key} in {get_env_path()}")
        return
    
    # Otherwise it goes to config.yaml
    # Read the raw user config (not merged with defaults) to avoid
    # dumping all default values back to the file
    config_path = get_config_path()
    user_config = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
        except Exception:
            user_config = {}
    
    # Handle nested keys (e.g., "tts.provider")
    parts = key.split('.')
    current = user_config
    
    for part in parts[:-1]:
        if part not in current or not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    
    # Convert value to appropriate type
    if value.lower() in ('true', 'yes', 'on'):
        value = True
    elif value.lower() in ('false', 'no', 'off'):
        value = False
    elif value.isdigit():
        value = int(value)
    elif value.replace('.', '', 1).isdigit():
        value = float(value)
    
    current[parts[-1]] = value
    
    # Write only user config back (not the full merged defaults)
    ensure_hermes_home()
    from utils import atomic_yaml_write
    atomic_yaml_write(config_path, user_config, sort_keys=False)
    
    # Keep .env in sync for keys that terminal_tool reads directly from env vars.
    # config.yaml is authoritative, but terminal_tool only reads TERMINAL_ENV etc.
    _config_to_env_sync = {
        "terminal.backend": "TERMINAL_ENV",
        "terminal.modal_mode": "TERMINAL_MODAL_MODE",
        "terminal.docker_image": "TERMINAL_DOCKER_IMAGE",
        "terminal.singularity_image": "TERMINAL_SINGULARITY_IMAGE",
        "terminal.modal_image": "TERMINAL_MODAL_IMAGE",
        "terminal.daytona_image": "TERMINAL_DAYTONA_IMAGE",
        "terminal.docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
        "terminal.cwd": "TERMINAL_CWD",
        "terminal.timeout": "TERMINAL_TIMEOUT",
        "terminal.sandbox_dir": "TERMINAL_SANDBOX_DIR",
        "terminal.persistent_shell": "TERMINAL_PERSISTENT_SHELL",
        "terminal.container_cpu": "TERMINAL_CONTAINER_CPU",
        "terminal.container_memory": "TERMINAL_CONTAINER_MEMORY",
        "terminal.container_disk": "TERMINAL_CONTAINER_DISK",
        "terminal.container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
    }
    if key in _config_to_env_sync:
        save_env_value(_config_to_env_sync[key], str(value))

    print(f"✓ Set {key} = {value} in {config_path}")


# =============================================================================
# Command handler
# =============================================================================

def config_command(args):
    """Handle config subcommands."""
    subcmd = getattr(args, 'config_command', None)
    
    if subcmd is None or subcmd == "show":
        show_config()
    
    elif subcmd == "edit":
        edit_config()
    
    elif subcmd == "set":
        key = getattr(args, 'key', None)
        value = getattr(args, 'value', None)
        if not key or value is None:
            print("Usage: hermes config set <key> <value>")
            print()
            print("Examples:")
            print("  hermes config set model anthropic/claude-sonnet-4")
            print("  hermes config set terminal.backend docker")
            print("  hermes config set OPENROUTER_API_KEY sk-or-...")
            sys.exit(1)
        set_config_value(key, value)
    
    elif subcmd == "path":
        print(get_config_path())
    
    elif subcmd == "env-path":
        print(get_env_path())
    
    elif subcmd == "migrate":
        print()
        print(color("🔄 Checking configuration for updates...", Colors.CYAN, Colors.BOLD))
        print()
        
        # Check what's missing
        missing_env = get_missing_env_vars(required_only=False)
        missing_config = get_missing_config_fields()
        current_ver, latest_ver = check_config_version()
        
        if not missing_env and not missing_config and current_ver >= latest_ver:
            print(color("✓ Configuration is up to date!", Colors.GREEN))
            print()
            return
        
        # Show what needs to be updated
        if current_ver < latest_ver:
            print(f"  Config version: {current_ver} → {latest_ver}")
        
        if missing_config:
            print(f"\n  {len(missing_config)} new config option(s) will be added with defaults")
        
        required_missing = [v for v in missing_env if v.get("is_required")]
        optional_missing = [
            v for v in missing_env
            if not v.get("is_required") and not v.get("advanced")
        ]
        
        if required_missing:
            print(f"\n  ⚠️  {len(required_missing)} required API key(s) missing:")
            for var in required_missing:
                print(f"     • {var['name']}")
        
        if optional_missing:
            print(f"\n  ℹ️  {len(optional_missing)} optional API key(s) not configured:")
            for var in optional_missing:
                tools = var.get("tools", [])
                tools_str = f" (enables: {', '.join(tools[:2])})" if tools else ""
                print(f"     • {var['name']}{tools_str}")
        
        print()
        
        # Run migration
        results = migrate_config(interactive=True, quiet=False)
        
        print()
        if results["env_added"] or results["config_added"]:
            print(color("✓ Configuration updated!", Colors.GREEN))
        
        if results["warnings"]:
            print()
            for warning in results["warnings"]:
                print(color(f"  ⚠️  {warning}", Colors.YELLOW))
        
        print()
    
    elif subcmd == "check":
        # Non-interactive check for what's missing
        print()
        print(color("📋 Configuration Status", Colors.CYAN, Colors.BOLD))
        print()
        
        current_ver, latest_ver = check_config_version()
        if current_ver >= latest_ver:
            print(f"  Config version: {current_ver} ✓")
        else:
            print(color(f"  Config version: {current_ver} → {latest_ver} (update available)", Colors.YELLOW))
        
        print()
        print(color("  Required:", Colors.BOLD))
        for var_name in REQUIRED_ENV_VARS:
            if get_env_value(var_name):
                print(f"    ✓ {var_name}")
            else:
                print(color(f"    ✗ {var_name} (missing)", Colors.RED))
        
        print()
        print(color("  Optional:", Colors.BOLD))
        for var_name, info in OPTIONAL_ENV_VARS.items():
            if get_env_value(var_name):
                print(f"    ✓ {var_name}")
            else:
                tools = info.get("tools", [])
                tools_str = f" → {', '.join(tools[:2])}" if tools else ""
                print(color(f"    ○ {var_name}{tools_str}", Colors.DIM))
        
        missing_config = get_missing_config_fields()
        if missing_config:
            print()
            print(color(f"  {len(missing_config)} new config option(s) available", Colors.YELLOW))
            print("    Run 'hermes config migrate' to add them")
        
        print()
    
    else:
        print(f"Unknown config command: {subcmd}")
        print()
        print("Available commands:")
        print("  hermes config           Show current configuration")
        print("  hermes config edit      Open config in editor")
        print("  hermes config set <key> <value>   Set a config value")
        print("  hermes config check     Check for missing/outdated config")
        print("  hermes config migrate   Update config with new options")
        print("  hermes config path      Show config file path")
        print("  hermes config env-path  Show .env file path")
        sys.exit(1)
