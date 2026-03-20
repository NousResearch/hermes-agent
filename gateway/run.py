"""
Gateway runner - entry point for messaging platform integrations.

This module provides:
- start_gateway(): Start all configured platform adapters
- GatewayRunner: Main class managing the gateway lifecycle

Usage:
    # Start the gateway
    python -m gateway.run
    
    # Or from CLI
    python cli.py --gateway
"""

import asyncio
import json
import logging
import os
import re
import shlex
import sys
import signal
import tempfile
import threading
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

# ---------------------------------------------------------------------------
# SSL certificate auto-detection for NixOS and other non-standard systems.
# Must run BEFORE any HTTP library (discord, aiohttp, etc.) is imported.
# ---------------------------------------------------------------------------
def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE if the system doesn't expose CA certs to Python."""
    if "SSL_CERT_FILE" in os.environ:
        return  # user already configured it

    import ssl

    # 1. Python's compiled-in defaults
    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    # 2. certifi (ships its own Mozilla bundle)
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        return
    except ImportError:
        pass

    # 3. Common distro / macOS locations
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",               # Debian/Ubuntu/Gentoo
        "/etc/pki/tls/certs/ca-bundle.crt",                 # RHEL/CentOS 7
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # RHEL/CentOS 8+
        "/etc/ssl/ca-bundle.pem",                            # SUSE/OpenSUSE
        "/etc/ssl/cert.pem",                                 # Alpine / macOS
        "/etc/pki/tls/cert.pem",                             # Fedora
        "/usr/local/etc/openssl@1.1/cert.pem",               # macOS Homebrew Intel
        "/opt/homebrew/etc/openssl@1.1/cert.pem",            # macOS Homebrew ARM
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

_ensure_ssl_certs()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

# Load environment variables from ~/.hermes/.env first.
# User-managed env files should override stale shell exports on restart.
from dotenv import load_dotenv  # backward-compat for tests that monkeypatch this symbol
from hermes_cli.env_loader import load_hermes_dotenv
_env_path = _hermes_home / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')

# Bridge config.yaml values into the environment so os.getenv() picks them up.
# config.yaml is authoritative for terminal settings — overrides .env.
_config_path = _hermes_home / 'config.yaml'
if _config_path.exists():
    try:
        import yaml as _yaml
        with open(_config_path, encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f) or {}
        # Top-level simple values (fallback only — don't override .env)
        for _key, _val in _cfg.items():
            if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
                os.environ[_key] = str(_val)
        # Terminal config is nested — bridge to TERMINAL_* env vars.
        # config.yaml overrides .env for these since it's the documented config path.
        _terminal_cfg = _cfg.get("terminal", {})
        if _terminal_cfg and isinstance(_terminal_cfg, dict):
            _terminal_env_map = {
                "backend": "TERMINAL_ENV",
                "cwd": "TERMINAL_CWD",
                "timeout": "TERMINAL_TIMEOUT",
                "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
                "docker_image": "TERMINAL_DOCKER_IMAGE",
                "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
                "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
                "modal_image": "TERMINAL_MODAL_IMAGE",
                "daytona_image": "TERMINAL_DAYTONA_IMAGE",
                "ssh_host": "TERMINAL_SSH_HOST",
                "ssh_user": "TERMINAL_SSH_USER",
                "ssh_port": "TERMINAL_SSH_PORT",
                "ssh_key": "TERMINAL_SSH_KEY",
                "container_cpu": "TERMINAL_CONTAINER_CPU",
                "container_memory": "TERMINAL_CONTAINER_MEMORY",
                "container_disk": "TERMINAL_CONTAINER_DISK",
                "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
                "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
                "sandbox_dir": "TERMINAL_SANDBOX_DIR",
                "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
            }
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    _val = _terminal_cfg[_cfg_key]
                    if isinstance(_val, list):
                        os.environ[_env_var] = json.dumps(_val)
                    else:
                        os.environ[_env_var] = str(_val)
        # Compression config is read directly from config.yaml by run_agent.py
        # and auxiliary_client.py — no env var bridging needed.
        # Auxiliary model/direct-endpoint overrides (vision, web_extract).
        # Each task has provider/model/base_url/api_key; bridge non-default values to env vars.
        _auxiliary_cfg = _cfg.get("auxiliary", {})
        if _auxiliary_cfg and isinstance(_auxiliary_cfg, dict):
            _aux_task_env = {
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
            for _task_key, _env_map in _aux_task_env.items():
                _task_cfg = _auxiliary_cfg.get(_task_key, {})
                if not isinstance(_task_cfg, dict):
                    continue
                _prov = str(_task_cfg.get("provider", "")).strip()
                _model = str(_task_cfg.get("model", "")).strip()
                _base_url = str(_task_cfg.get("base_url", "")).strip()
                _api_key = str(_task_cfg.get("api_key", "")).strip()
                if _prov and _prov != "auto":
                    os.environ[_env_map["provider"]] = _prov
                if _model:
                    os.environ[_env_map["model"]] = _model
                if _base_url:
                    os.environ[_env_map["base_url"]] = _base_url
                if _api_key:
                    os.environ[_env_map["api_key"]] = _api_key
        _agent_cfg = _cfg.get("agent", {})
        if _agent_cfg and isinstance(_agent_cfg, dict):
            if "max_turns" in _agent_cfg:
                os.environ["HERMES_MAX_ITERATIONS"] = str(_agent_cfg["max_turns"])
        # Timezone: bridge config.yaml → HERMES_TIMEZONE env var.
        # HERMES_TIMEZONE from .env takes precedence (already in os.environ).
        _tz_cfg = _cfg.get("timezone", "")
        if _tz_cfg and isinstance(_tz_cfg, str) and "HERMES_TIMEZONE" not in os.environ:
            os.environ["HERMES_TIMEZONE"] = _tz_cfg.strip()
        # Security settings
        _security_cfg = _cfg.get("security", {})
        if isinstance(_security_cfg, dict):
            _redact = _security_cfg.get("redact_secrets")
            if _redact is not None:
                os.environ["HERMES_REDACT_SECRETS"] = str(_redact).lower()
    except Exception:
        pass  # Non-fatal; gateway can still run with .env values

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

# Enable interactive exec approval for dangerous commands on messaging platforms
os.environ["HERMES_EXEC_ASK"] = "1"

# Set terminal working directory for messaging platforms.
# If the user set an explicit path in config.yaml (not "." or "auto"),
# respect it. Otherwise use MESSAGING_CWD or default to home directory.
_configured_cwd = os.environ.get("TERMINAL_CWD", "")
if not _configured_cwd or _configured_cwd in (".", "auto", "cwd"):
    messaging_cwd = os.getenv("MESSAGING_CWD") or str(Path.home())
    os.environ["TERMINAL_CWD"] = messaging_cwd

from gateway.config import (
    Platform,
    GatewayConfig,
    load_gateway_config,
)
from gateway.session import (
    SessionStore,
    SessionSource,
    SessionContext,
    build_session_context,
    build_session_context_prompt,
    build_session_key,
)
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.command_catalog import (
    build_session_export_html,
    format_yaml_block,
    load_raw_user_config,
    read_config_or_env_value,
    resolve_export_path,
    unset_config_or_env_value,
    write_config_or_env_value,
)

logger = logging.getLogger(__name__)

# Sentinel placed into _running_agents immediately when a session starts
# processing, *before* any await.  Prevents a second message for the same
# session from bypassing the "already running" guard during the async gap
# between the guard check and actual agent creation.
_AGENT_PENDING_SENTINEL = object()


def _resolve_runtime_agent_kwargs() -> dict:
    """Resolve provider credentials for gateway-created AIAgent instances."""
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
    )

    try:
        runtime = resolve_runtime_provider(
            requested=os.getenv("HERMES_INFERENCE_PROVIDER"),
        )
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


def _resolve_gateway_model() -> str:
    """Read model from env/config — mirrors the resolution in _run_agent_sync.

    Without this, temporary AIAgent instances (memory flush, /compress) fall
    back to the hardcoded default ("anthropic/claude-opus-4.6") which fails
    when the active provider is openai-codex.
    """
    model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL") or "anthropic/claude-opus-4.6"
    try:
        import yaml as _y
        _cfg_path = _hermes_home / "config.yaml"
        if _cfg_path.exists():
            with open(_cfg_path, encoding="utf-8") as _f:
                _cfg = _y.safe_load(_f) or {}
            _model_cfg = _cfg.get("model", {})
            if isinstance(_model_cfg, str):
                model = _model_cfg
            elif isinstance(_model_cfg, dict):
                model = _model_cfg.get("default", model)
    except Exception:
        pass
    return model


def _resolve_hermes_bin() -> Optional[list[str]]:
    """Resolve the Hermes update command as argv parts.

    Tries in order:
    1. ``shutil.which("hermes")`` — standard PATH lookup
    2. ``sys.executable -m hermes_cli.main`` — fallback when Hermes is running
       from a venv/module invocation and the ``hermes`` shim is not on PATH

    Returns argv parts ready for quoting/joining, or ``None`` if neither works.
    """
    import shutil

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]

    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass

    return None


class GatewayRunner:
    """
    Main gateway controller.
    
    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or load_gateway_config()
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        self._runtime_debug_overrides: Dict[str, Any] = {}

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._ephemeral_system_prompt = self._load_ephemeral_system_prompt()
        self._reasoning_config = self._load_reasoning_config()
        self._show_reasoning = self._load_show_reasoning()
        self._provider_routing = self._load_provider_routing()
        self._fallback_model = self._load_fallback_model()
        self._smart_model_routing = self._load_smart_model_routing()

        # Wire process registry into session store for reset protection
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(key),
        )
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._exit_cleanly = False
        self._exit_reason: Optional[str] = None
        
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._pending_messages: Dict[str, str] = {}  # Queued messages during interrupt
        self._bash_jobs: Dict[str, str] = {}
        self._subagent_runtime: Dict[str, Dict[str, Any]] = {}
        self._acp_session_bindings: Dict[str, str] = {}
        self._acp_session_meta: Dict[str, Dict[str, Any]] = {}
        self._acp_running_tasks: Dict[str, asyncio.Task] = {}
        self._acp_session_manager = None

        # Track active fallback model/provider when primary is rate-limited.
        # Set after an agent run where fallback was activated; cleared when
        # the primary model succeeds again or the user switches via /model.
        self._effective_model: Optional[str] = None
        self._effective_provider: Optional[str] = None

        # Track pending exec approvals per session
        # Key: session_key, Value: {"command": str, "pattern_key": str, ...}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Persistent Honcho managers keyed by gateway session key.
        # This preserves write_frequency="session" semantics across short-lived
        # per-message AIAgent instances.
        self._honcho_managers: Dict[str, Any] = {}
        self._honcho_configs: Dict[str, Any] = {}

        # Ensure tirith security scanner is available (downloads if needed)
        try:
            from tools.tirith_security import ensure_installed
            ensure_installed(log_failures=False)
        except Exception:
            pass  # Non-fatal — fail-open at scan time if unavailable
        
        # Initialize session database for session_search tool support
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            logger.debug("SQLite session store not available: %s", e)
        
        # DM pairing store for code-based user authorization
        from gateway.pairing import PairingStore
        self.pairing_store = PairingStore()
        
        # Event hook system
        from gateway.hooks import HookRegistry
        self.hooks = HookRegistry()

        # Per-chat voice reply mode: "off" | "voice_only" | "all"
        self._voice_mode: Dict[str, str] = self._load_voice_modes()
        runtime_controls = self._load_runtime_controls()
        self._session_send_policies: Dict[str, str] = runtime_controls["send_policies"]
        self._session_docks: Dict[str, Dict[str, Any]] = runtime_controls["dock_targets"]

    _DEBUG_REASONING_LEVELS = {"none", "minimal", "low", "medium", "high", "xhigh"}
    _DEBUG_BACKGROUND_NOTIFICATION_MODES = {"all", "result", "error", "off"}
    _DEBUG_PROVIDER_DATA_COLLECTION_MODES = {"allow", "deny"}
    _DEBUG_DISCORD_BOT_FILTER_POLICIES = {"none", "mentions", "all"}
    _DEBUG_SUPPORTED_KEYS: Dict[str, str] = {
        "agent.system_prompt": "Ephemeral system prompt for new turns in the running gateway process.",
        "agent.reasoning_effort": "Runtime reasoning effort (none|minimal|low|medium|high|xhigh).",
        "display.show_reasoning": "Show or hide model reasoning in responses.",
        "display.background_process_notifications": "Background process watcher notifications (all|result|error|off).",
        "provider_routing.only": "Preferred provider allowlist for OpenRouter routing.",
        "provider_routing.ignore": "Providers to skip for OpenRouter routing.",
        "provider_routing.order": "Explicit provider preference order for OpenRouter routing.",
        "provider_routing.sort": "Provider routing sort strategy.",
        "provider_routing.require_parameters": "Require providers that support all request parameters.",
        "provider_routing.data_collection": "Provider data collection policy (allow|deny).",
        "fallback_model.provider": "Fallback provider used when the primary route fails.",
        "fallback_model.model": "Fallback model used when the primary route fails.",
        "smart_model_routing.enabled": "Enable or disable cheap-vs-strong smart routing.",
        "smart_model_routing.max_simple_chars": "Maximum message characters for cheap-route eligibility.",
        "smart_model_routing.max_simple_words": "Maximum word count for cheap-route eligibility.",
        "smart_model_routing.cheap_model.provider": "Cheap-route provider.",
        "smart_model_routing.cheap_model.model": "Cheap-route model.",
        "discord.allow_bots": "Discord bot-message policy (none|mentions|all).",
        "discord.free_response_channels": "Discord channel IDs that bypass mention gating.",
        "discord.require_mention": "Require @mention in Discord server channels.",
        "discord.auto_thread": "Auto-create Discord threads on @mention.",
    }

    def _get_runtime_debug_overrides(self) -> Dict[str, Any]:
        overrides = getattr(self, "_runtime_debug_overrides", None)
        if overrides is None:
            overrides = {}
            self._runtime_debug_overrides = overrides
        return overrides

    @staticmethod
    def _parse_debug_scalar(raw_value: str) -> Any:
        import yaml

        text = (raw_value or "").strip()
        if not text:
            return ""
        try:
            return yaml.safe_load(text)
        except Exception:
            return text

    @classmethod
    def _parse_debug_bool(cls, raw_value: str) -> bool:
        parsed = cls._parse_debug_scalar(raw_value)
        if isinstance(parsed, bool):
            return parsed
        if isinstance(parsed, str):
            normalized = parsed.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise ValueError("expected a boolean value")

    @classmethod
    def _parse_debug_string_list(cls, raw_value: str) -> List[str]:
        parsed = cls._parse_debug_scalar(raw_value)
        if parsed is None:
            return []
        if isinstance(parsed, (list, tuple, set)):
            items = list(parsed)
        elif isinstance(parsed, str):
            if "," in parsed:
                items = [item.strip() for item in parsed.split(",")]
            else:
                items = [parsed.strip()]
        else:
            items = [str(parsed).strip()]
        return [str(item).strip() for item in items if str(item).strip()]

    @classmethod
    def _parse_debug_optional_string(cls, raw_value: str) -> Optional[str]:
        parsed = cls._parse_debug_scalar(raw_value)
        if parsed is None:
            return None
        text = str(parsed).strip()
        if not text:
            return None
        return text

    @classmethod
    def _parse_debug_int(cls, raw_value: str) -> int:
        parsed = cls._parse_debug_scalar(raw_value)
        try:
            return int(parsed)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected an integer value") from exc

    @classmethod
    def _parse_debug_value(cls, key: str, raw_value: str) -> Any:
        if key == "agent.system_prompt":
            return str(raw_value)

        if key == "agent.reasoning_effort":
            normalized = str(raw_value).strip().lower()
            if normalized == "off":
                normalized = "none"
            if normalized not in cls._DEBUG_REASONING_LEVELS:
                raise ValueError(
                    "expected one of: none, minimal, low, medium, high, xhigh"
                )
            return normalized

        if key == "display.show_reasoning":
            return cls._parse_debug_bool(raw_value)

        if key == "display.background_process_notifications":
            normalized = str(raw_value).strip().lower()
            if normalized not in cls._DEBUG_BACKGROUND_NOTIFICATION_MODES:
                raise ValueError("expected one of: all, result, error, off")
            return normalized

        if key in {
            "provider_routing.only",
            "provider_routing.ignore",
            "provider_routing.order",
            "discord.free_response_channels",
        }:
            return cls._parse_debug_string_list(raw_value)

        if key == "provider_routing.sort":
            return cls._parse_debug_optional_string(raw_value)

        if key == "provider_routing.require_parameters":
            return cls._parse_debug_bool(raw_value)

        if key == "provider_routing.data_collection":
            value = cls._parse_debug_optional_string(raw_value)
            if value is None:
                return None
            normalized = value.lower()
            if normalized not in cls._DEBUG_PROVIDER_DATA_COLLECTION_MODES:
                raise ValueError("expected one of: allow, deny")
            return normalized

        if key in {"fallback_model.provider", "fallback_model.model"}:
            return cls._parse_debug_optional_string(raw_value)

        if key == "smart_model_routing.enabled":
            return cls._parse_debug_bool(raw_value)

        if key in {"smart_model_routing.max_simple_chars", "smart_model_routing.max_simple_words"}:
            return cls._parse_debug_int(raw_value)

        if key in {"smart_model_routing.cheap_model.provider", "smart_model_routing.cheap_model.model"}:
            return cls._parse_debug_optional_string(raw_value)

        if key == "discord.allow_bots":
            normalized = str(raw_value).strip().lower()
            if normalized not in cls._DEBUG_DISCORD_BOT_FILTER_POLICIES:
                raise ValueError("expected one of: none, mentions, all")
            return normalized

        if key in {"discord.require_mention", "discord.auto_thread"}:
            return cls._parse_debug_bool(raw_value)

        raise KeyError(key)

    @staticmethod
    def _format_runtime_debug_value(value: Any) -> str:
        return format_yaml_block(value)

    @staticmethod
    def _reasoning_config_to_level(reasoning_config: dict | None) -> str:
        if reasoning_config is None:
            return "medium"
        if reasoning_config.get("enabled") is False:
            return "none"
        return str(reasoning_config.get("effort") or "medium").strip().lower()

    @classmethod
    def _reasoning_level_to_config(cls, level: str) -> dict | None:
        normalized = (level or "").strip().lower()
        if not normalized or normalized == "medium":
            return {"enabled": True, "effort": "medium"}
        if normalized == "none":
            return {"enabled": False}
        if normalized in {"minimal", "low", "high", "xhigh"}:
            return {"enabled": True, "effort": normalized}
        return None

    def _get_effective_reasoning_config(self) -> dict | None:
        overrides = self._get_runtime_debug_overrides()
        if "agent.reasoning_effort" not in overrides:
            return self._load_reasoning_config()
        return self._reasoning_level_to_config(str(overrides["agent.reasoning_effort"]))

    def _get_effective_show_reasoning(self) -> bool:
        overrides = self._get_runtime_debug_overrides()
        if "display.show_reasoning" in overrides:
            return bool(overrides["display.show_reasoning"])
        return self._load_show_reasoning()

    def _get_effective_background_notifications_mode(self) -> str:
        overrides = self._get_runtime_debug_overrides()
        if "display.background_process_notifications" in overrides:
            return str(overrides["display.background_process_notifications"])
        return self._load_background_notifications_mode()

    def _build_effective_provider_routing(self) -> dict:
        merged = dict(self._load_provider_routing() or {})
        overrides = self._get_runtime_debug_overrides()
        for suffix in ("only", "ignore", "order", "sort", "require_parameters", "data_collection"):
            key = f"provider_routing.{suffix}"
            if key in overrides:
                merged[suffix] = overrides[key]
        return merged

    def _build_effective_fallback_model(self) -> dict | None:
        merged = dict(self._load_fallback_model() or {})
        overrides = self._get_runtime_debug_overrides()
        for suffix in ("provider", "model"):
            key = f"fallback_model.{suffix}"
            if key in overrides:
                merged[suffix] = overrides[key]
        provider = str(merged.get("provider") or "").strip()
        model = str(merged.get("model") or "").strip()
        if not provider or not model:
            return None
        return {"provider": provider, "model": model}

    def _build_effective_smart_model_routing(self) -> dict:
        merged = dict(self._load_smart_model_routing() or {})
        cheap_model = dict(merged.get("cheap_model") or {})
        merged["cheap_model"] = cheap_model
        overrides = self._get_runtime_debug_overrides()

        for suffix in ("enabled", "max_simple_chars", "max_simple_words"):
            key = f"smart_model_routing.{suffix}"
            if key in overrides:
                merged[suffix] = overrides[key]

        for suffix in ("provider", "model"):
            key = f"smart_model_routing.cheap_model.{suffix}"
            if key in overrides:
                cheap_model[suffix] = overrides[key]

        return merged

    def _build_effective_discord_policy_overrides(self) -> dict:
        overrides = self._get_runtime_debug_overrides()
        policy_overrides: Dict[str, Any] = {}
        for source_key, target_key in (
            ("discord.allow_bots", "allow_bots"),
            ("discord.free_response_channels", "free_response_channels"),
            ("discord.require_mention", "require_mention"),
            ("discord.auto_thread", "auto_thread"),
        ):
            if source_key in overrides:
                policy_overrides[target_key] = overrides[source_key]
        return policy_overrides

    def _get_effective_runtime_debug_value(self, key: str) -> Any:
        overrides = self._get_runtime_debug_overrides()
        if key == "agent.system_prompt":
            return overrides[key] if key in overrides else self._load_ephemeral_system_prompt()
        if key == "agent.reasoning_effort":
            if key in overrides:
                return overrides[key]
            return self._reasoning_config_to_level(self._load_reasoning_config())
        if key == "display.show_reasoning":
            return self._get_effective_show_reasoning()
        if key == "display.background_process_notifications":
            return self._get_effective_background_notifications_mode()
        if key.startswith("provider_routing."):
            return self._build_effective_provider_routing().get(key.split(".", 1)[1])
        if key.startswith("fallback_model."):
            current = self._build_effective_fallback_model() or {}
            return current.get(key.split(".", 1)[1])
        if key.startswith("smart_model_routing.cheap_model."):
            current = self._build_effective_smart_model_routing().get("cheap_model") or {}
            return current.get(key.rsplit(".", 1)[1])
        if key.startswith("smart_model_routing."):
            current = self._build_effective_smart_model_routing()
            return current.get(key.split(".", 1)[1])
        if key.startswith("discord."):
            from gateway.platforms.discord_impl import config as discord_config

            policy = discord_config.load_policy_config(
                getattr(self, "config", None).platforms.get(Platform.DISCORD)
                if getattr(self, "config", None) and getattr(self.config, "platforms", None)
                else None,
                overrides=self._build_effective_discord_policy_overrides(),
            )
            mapping = {
                "discord.allow_bots": policy.bot_filter_policy,
                "discord.free_response_channels": sorted(policy.free_response_channels),
                "discord.require_mention": policy.require_mention,
                "discord.auto_thread": policy.auto_thread,
            }
            return mapping[key]
        raise KeyError(key)

    def _apply_runtime_debug_overrides(self) -> None:
        self._ephemeral_system_prompt = str(self._get_effective_runtime_debug_value("agent.system_prompt") or "")
        self._reasoning_config = self._get_effective_reasoning_config()
        self._show_reasoning = self._get_effective_show_reasoning()
        self._provider_routing = self._build_effective_provider_routing()
        self._fallback_model = self._build_effective_fallback_model()
        self._smart_model_routing = self._build_effective_smart_model_routing()

        discord_adapter = getattr(self, "adapters", {}).get(Platform.DISCORD)
        if discord_adapter and hasattr(discord_adapter, "apply_runtime_policy_overrides"):
            discord_adapter.apply_runtime_policy_overrides(
                self._build_effective_discord_policy_overrides()
            )

    def _get_or_create_gateway_honcho(self, session_key: str):
        """Return a persistent Honcho manager/config pair for this gateway session."""
        if not hasattr(self, "_honcho_managers"):
            self._honcho_managers = {}
        if not hasattr(self, "_honcho_configs"):
            self._honcho_configs = {}

        if session_key in self._honcho_managers:
            return self._honcho_managers[session_key], self._honcho_configs.get(session_key)

        try:
            from honcho_integration.client import HonchoClientConfig, get_honcho_client
            from honcho_integration.session import HonchoSessionManager

            hcfg = HonchoClientConfig.from_global_config()
            if not hcfg.enabled or not hcfg.api_key:
                return None, hcfg

            client = get_honcho_client(hcfg)
            manager = HonchoSessionManager(
                honcho=client,
                config=hcfg,
                context_tokens=hcfg.context_tokens,
            )
            self._honcho_managers[session_key] = manager
            self._honcho_configs[session_key] = hcfg
            return manager, hcfg
        except Exception as e:
            logger.debug("Gateway Honcho init failed for %s: %s", session_key, e)
            return None, None

    def _shutdown_gateway_honcho(self, session_key: str) -> None:
        """Flush and close the persistent Honcho manager for a gateway session."""
        managers = getattr(self, "_honcho_managers", None)
        configs = getattr(self, "_honcho_configs", None)
        if managers is None or configs is None:
            return

        manager = managers.pop(session_key, None)
        configs.pop(session_key, None)
        if not manager:
            return
        try:
            manager.shutdown()
        except Exception as e:
            logger.debug("Gateway Honcho shutdown failed for %s: %s", session_key, e)

    def _shutdown_all_gateway_honcho(self) -> None:
        """Flush and close all persistent Honcho managers."""
        managers = getattr(self, "_honcho_managers", None)
        if not managers:
            return
        for session_key in list(managers.keys()):
            self._shutdown_gateway_honcho(session_key)
    
    # -- Setup skill availability ----------------------------------------

    def _has_setup_skill(self) -> bool:
        """Check if the hermes-agent-setup skill is installed."""
        try:
            from tools.skill_manager_tool import _find_skill
            return _find_skill("hermes-agent-setup") is not None
        except Exception:
            return False

    # -- Voice mode persistence ------------------------------------------

    _VOICE_MODE_PATH = _hermes_home / "gateway_voice_mode.json"
    _RUNTIME_CONTROLS_PATH = _hermes_home / "gateway_runtime_controls.json"

    def _load_voice_modes(self) -> Dict[str, str]:
        try:
            data = json.loads(self._VOICE_MODE_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}

        valid_modes = {"off", "voice_only", "all"}
        return {
            str(chat_id): mode
            for chat_id, mode in data.items()
            if mode in valid_modes
        }

    def _save_voice_modes(self) -> None:
        try:
            self._VOICE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._VOICE_MODE_PATH.write_text(
                json.dumps(self._voice_mode, indent=2)
            )
        except OSError as e:
            logger.warning("Failed to save voice modes: %s", e)

    def _set_adapter_auto_tts_disabled(self, adapter, chat_id: str, disabled: bool) -> None:
        """Update an adapter's in-memory auto-TTS suppression set if present."""
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if not isinstance(disabled_chats, set):
            return
        if disabled:
            disabled_chats.add(chat_id)
        else:
            disabled_chats.discard(chat_id)

    def _sync_voice_mode_state_to_adapter(self, adapter) -> None:
        """Restore persisted /voice off state into a live platform adapter."""
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if not isinstance(disabled_chats, set):
            return
        disabled_chats.clear()
        disabled_chats.update(
            chat_id for chat_id, mode in self._voice_mode.items() if mode == "off"
        )

    def _load_runtime_controls(self) -> Dict[str, Dict[str, Any]]:
        """Load persisted per-session send-policy and dock-routing controls."""
        try:
            payload = json.loads(self._RUNTIME_CONTROLS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"send_policies": {}, "dock_targets": {}}

        if not isinstance(payload, dict):
            return {"send_policies": {}, "dock_targets": {}}

        raw_send_policies = payload.get("send_policies") or {}
        send_policies = {
            str(session_key): str(mode)
            for session_key, mode in raw_send_policies.items()
            if str(mode) in {"on", "off", "inherit"}
        }

        raw_dock_targets = payload.get("dock_targets") or {}
        dock_targets: Dict[str, Dict[str, Any]] = {}
        for session_key, target in raw_dock_targets.items():
            if not isinstance(target, dict):
                continue
            platform = str(target.get("platform") or "").strip().lower()
            chat_id = str(target.get("chat_id") or "").strip()
            if not platform or not chat_id:
                continue
            dock_targets[str(session_key)] = {
                "platform": platform,
                "chat_id": chat_id,
                "thread_id": str(target.get("thread_id") or "").strip() or None,
                "name": str(target.get("name") or "").strip(),
            }

        return {
            "send_policies": send_policies,
            "dock_targets": dock_targets,
        }

    def _save_runtime_controls(self) -> None:
        """Persist per-session send-policy and dock-routing controls."""
        payload = {
            "send_policies": getattr(self, "_session_send_policies", {}) or {},
            "dock_targets": getattr(self, "_session_docks", {}) or {},
        }
        try:
            self._RUNTIME_CONTROLS_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._RUNTIME_CONTROLS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except OSError as e:
            logger.warning("Failed to save gateway runtime controls: %s", e)

    # -----------------------------------------------------------------

    def _flush_memories_for_session(
        self,
        old_session_id: str,
        honcho_session_key: Optional[str] = None,
    ):
        """Prompt the agent to save memories/skills before context is lost.

        Synchronous worker — meant to be called via run_in_executor from
        an async context so it doesn't block the event loop.
        """
        try:
            history = self.session_store.load_transcript(old_session_id)
            if not history or len(history) < 4:
                return

            from run_agent import AIAgent
            runtime_kwargs = _resolve_runtime_agent_kwargs()
            if not runtime_kwargs.get("api_key"):
                return

            # Resolve model from config — AIAgent's default is OpenRouter-
            # formatted ("anthropic/claude-opus-4.6") which fails when the
            # active provider is openai-codex.
            model = _resolve_gateway_model()

            tmp_agent = AIAgent(
                **runtime_kwargs,
                model=model,
                max_iterations=8,
                quiet_mode=True,
                enabled_toolsets=["memory", "skills"],
                session_id=old_session_id,
                honcho_session_key=honcho_session_key,
            )

            # Build conversation history from transcript
            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]

            # Give the agent a real turn to think about what to save
            flush_prompt = (
                "[System: This session is about to be automatically reset due to "
                "inactivity or a scheduled daily reset. The conversation context "
                "will be cleared after this turn.\n\n"
                "Review the conversation above and:\n"
                "1. Save any important facts, preferences, or decisions to memory "
                "(user profile or your notes) that would be useful in future sessions.\n"
                "2. If you discovered a reusable workflow or solved a non-trivial "
                "problem, consider saving it as a skill.\n"
                "3. If nothing is worth saving, that's fine — just skip.\n\n"
                "Do NOT respond to the user. Just use the memory and skill_manage "
                "tools if needed, then stop.]"
            )

            tmp_agent.run_conversation(
                user_message=flush_prompt,
                conversation_history=msgs,
                sync_honcho=False,
            )
            logger.info("Pre-reset memory flush completed for session %s", old_session_id)
            # Flush any queued Honcho writes before the session is dropped
            if getattr(tmp_agent, '_honcho', None):
                try:
                    tmp_agent._honcho.shutdown()
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Pre-reset memory flush failed for session %s: %s", old_session_id, e)

    async def _async_flush_memories(
        self,
        old_session_id: str,
        honcho_session_key: Optional[str] = None,
    ):
        """Run the sync memory flush in a thread pool so it won't block the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._flush_memories_for_session,
            old_session_id,
            honcho_session_key,
        )

    @property
    def should_exit_cleanly(self) -> bool:
        return self._exit_cleanly

    @property
    def exit_reason(self) -> Optional[str]:
        return self._exit_reason

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        if hasattr(self, "session_store") and self.session_store is not None:
            try:
                session_key = self.session_store._generate_session_key(source)
                if isinstance(session_key, str) and session_key:
                    return session_key
            except Exception:
                pass
        config = getattr(self, "config", None)
        return build_session_key(
            source,
            group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
        )

    def _session_source_for_event(self, event: MessageEvent) -> SessionSource:
        """Resolve the session source for an event, honoring command-session overrides."""
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict):
            session_source = metadata.get("session_source")
            if isinstance(session_source, SessionSource):
                return session_source
        return event.source

    def _command_target_source_for_event(self, event: MessageEvent) -> SessionSource:
        """Resolve the command target source, falling back to the event source."""
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict):
            target_source = metadata.get("command_target_source")
            if isinstance(target_source, SessionSource):
                return target_source
        return event.source

    @staticmethod
    def _new_approval_id() -> str:
        """Generate a short human-readable approval identifier."""
        return f"appr-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _normalize_approval_decision(raw: str) -> Optional[str]:
        value = str(raw or "").strip().lower()
        mapping = {
            "allow-once": "allow-once",
            "once": "allow-once",
            "allow_once": "allow-once",
            "yes": "allow-once",
            "approve": "allow-once",
            "allow-session": "allow-session",
            "allow_session": "allow-session",
            "session": "allow-session",
            "ses": "allow-session",
            "allow-always": "allow-always",
            "always": "allow-always",
            "allow_always": "allow-always",
            "deny": "deny",
            "no": "deny",
            "reject": "deny",
            "cancel": "deny",
            "show": "show",
            "view": "show",
            "full": "show",
        }
        return mapping.get(value)

    def _find_pending_approval(
        self,
        *,
        approval_id: Optional[str] = None,
        source: Optional[SessionSource] = None,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Find a pending approval by explicit approval ID or source session."""
        if approval_id:
            for session_key, pending in self._pending_approvals.items():
                if str(pending.get("approval_id") or "") == approval_id:
                    return session_key, pending
            return None, None

        if source is None:
            return None, None

        session_key = self._session_key_for_source(source)
        return session_key, self._pending_approvals.get(session_key)

    def _approval_usage_text(self) -> str:
        return "`/approve`, `/approve session`, `/approve always`, or `/approve [id] <allow-once|allow-session|allow-always|deny|show>`"

    def _resolve_pending_approval(
        self,
        *,
        decision: str,
        approval_id: Optional[str] = None,
        source: Optional[SessionSource] = None,
    ) -> str:
        """Resolve a pending approval by ID or source session."""
        normalized = self._normalize_approval_decision(decision)
        if normalized is None:
            return (
                f"⚠️ Unknown approval decision: `{decision}`\n\n"
                f"Use {self._approval_usage_text()}"
            )

        session_key, pending = self._find_pending_approval(approval_id=approval_id, source=source)
        if not session_key or not pending:
            if normalized == "deny":
                return "No pending command to deny."
            return "No pending command to approve."

        pending_id = str(pending.get("approval_id") or "")
        command = str(pending.get("command") or "").strip()
        pattern_keys = list(pending.get("pattern_keys") or [])
        if not pattern_keys:
            pattern_key = str(pending.get("pattern_key") or "").strip()
            if pattern_key:
                pattern_keys = [pattern_key]

        timestamp = pending.get("timestamp")
        if isinstance(timestamp, (int, float)) and time.time() - float(timestamp) > 300:
            self._pending_approvals.pop(session_key, None)
            return "⚠️ Approval expired (timed out after 5 minutes). Ask the agent to try again."

        if normalized == "show":
            return (
                f"⚠️ Pending approval `{pending_id}`\n\n"
                f"```\n{command}\n```\n\n"
                f"Use {self._approval_usage_text()}"
            )

        if normalized == "deny":
            self._pending_approvals.pop(session_key, None)
            return "❌ Command denied."

        import tools.approval as approval_mod
        from tools.terminal_tool import terminal_tool

        if normalized == "allow-session":
            for pattern_key in pattern_keys:
                approval_mod.approve_session(session_key, pattern_key)
        elif normalized == "allow-always":
            for pattern_key in pattern_keys:
                approval_mod.approve_session(session_key, pattern_key)
                approval_mod.approve_permanent(pattern_key)
            approval_mod.save_permanent_allowlist(approval_mod._permanent_approved)
        else:
            for pattern_key in pattern_keys:
                approval_mod.approve_session(session_key, pattern_key)

        self._pending_approvals.pop(session_key, None)
        on_approve = pending.get("on_approve")
        if callable(on_approve):
            try:
                return on_approve(normalized)
            except Exception as e:
                logger.exception("Pending approval callback failed")
                return f"❌ Approved command failed: {e}"

        result = terminal_tool(command=command, force=True)
        if normalized == "allow-session":
            prefix = (
                "✅ Command approved and executed "
                "(pattern approved for this session). Decision: approved (allow-session)."
            )
        elif normalized == "allow-always":
            prefix = (
                "✅ Command approved and executed "
                "(pattern approved permanently). Decision: approved (allow-always)."
            )
        else:
            prefix = "✅ Command approved and executed. Decision: approved (allow-once)."
        if pending_id:
            prefix = f"{prefix} (`{pending_id}`)"
        return f"{prefix}\n\n```\n{result[:3500]}\n```"

    @staticmethod
    def _format_minutes_label(minutes: int) -> str:
        if minutes <= 0:
            return "off"
        if minutes % 1440 == 0:
            days = minutes // 1440
            return f"{days}d"
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"{hours}h"
        return f"{minutes}m"

    @staticmethod
    def _parse_duration_minutes(raw: str) -> Optional[int]:
        value = str(raw or "").strip().lower()
        if not value:
            return None
        if value in {"off", "none", "disable", "disabled"}:
            return 0

        from cron.jobs import parse_duration

        return parse_duration(value)

    def _get_discord_adapter(self):
        adapter = self.adapters.get(Platform.DISCORD)
        if adapter is None:
            return None
        return adapter

    def _session_send_policy(self, session_key: str) -> str:
        policies = getattr(self, "_session_send_policies", None)
        if not isinstance(policies, dict):
            return "inherit"
        return policies.get(session_key, "inherit")

    def _dock_target_for_session(self, session_key: str) -> Optional[Dict[str, Any]]:
        dock_targets = getattr(self, "_session_docks", None)
        if not isinstance(dock_targets, dict):
            return None
        target = dock_targets.get(session_key)
        if not isinstance(target, dict):
            return None
        return target

    def _format_dock_target(self, target: Dict[str, Any]) -> str:
        platform = str(target.get("platform") or "unknown")
        name = str(target.get("name") or "").strip()
        chat_id = str(target.get("chat_id") or "")
        if name:
            return f"{platform}:{name} (`{chat_id}`)"
        return f"{platform} home channel (`{chat_id}`)"

    def _get_bash_jobs(self) -> Dict[str, str]:
        jobs = getattr(self, "_bash_jobs", None)
        if jobs is None:
            jobs = {}
            self._bash_jobs = jobs
        return jobs

    @staticmethod
    def _rewrite_bash_shortcut(raw_text: str) -> Optional[str]:
        text = str(raw_text or "")
        stripped = text.lstrip()
        if not stripped.startswith("!"):
            return None
        payload = stripped[1:].strip()
        if not payload:
            return None
        if payload.startswith("poll"):
            args = payload[4:].strip()
            return f"/bash poll {args}".strip()
        if payload.startswith("stop"):
            args = payload[4:].strip()
            return f"/bash stop {args}".strip()
        return f"/bash {payload}"

    @staticmethod
    def _split_command_args(raw: str, *, max_parts: int | None = None) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            parts = shlex.split(text)
        except ValueError:
            parts = text.split()
        if max_parts is None or len(parts) <= max_parts:
            return parts
        head = parts[: max_parts - 1]
        tail = " ".join(parts[max_parts - 1 :]).strip()
        return [*head, tail]

    @staticmethod
    def _normalize_skill_name(raw: str) -> str:
        return str(raw or "").strip().lower().replace(" ", "-").replace("_", "-").lstrip("/")

    @staticmethod
    def _parity_blocker_text(command: str, plan_id: str, detail: str) -> str:
        return (
            f"⚠️ `/{command}` is still blocked in Hermes parity work.\n\n"
            f"**Blocked by:** `{plan_id}`\n"
            f"**Reason:** {detail}"
        )

    def _get_bash_foreground_seconds(self) -> int:
        raw_ms = os.getenv("HERMES_BASH_FOREGROUND_MS", "").strip()
        if not raw_ms:
            try:
                cfg = load_raw_user_config() or {}
                commands_cfg = cfg.get("commands", {}) if isinstance(cfg, dict) else {}
                raw_ms = str(commands_cfg.get("bashForegroundMs", "") or "").strip()
            except Exception:
                raw_ms = ""
        try:
            ms = int(raw_ms or "2000")
        except ValueError:
            ms = 2000
        if ms <= 0:
            return 0
        return max(1, (ms + 999) // 1000)

    @staticmethod
    def _format_bash_output(output: str, *, limit: int = 3500) -> str:
        text = str(output or "").rstrip()
        if not text:
            return "_(no output)_"
        if len(text) > limit:
            text = text[-limit:]
            text = f"...\n{text}"
        return f"```\n{text}\n```"

    def _resolve_bash_target_session_id(self, session_key: str, raw_target: str) -> Optional[str]:
        from tools.process_registry import process_registry

        target = str(raw_target or "").strip()
        jobs = self._get_bash_jobs()
        if target:
            return target
        session_id = jobs.get(session_key)
        if not session_id:
            return None
        current = process_registry.get(session_id)
        if current is None or current.exited:
            jobs.pop(session_key, None)
            return None
        return session_id

    def _store_pending_bash_approval(
        self,
        *,
        session_key: str,
        command: str,
        approval_payload: dict[str, Any],
        on_approve,
    ) -> str:
        approval_id = self._new_approval_id()
        pending = {
            "approval_id": approval_id,
            "session_key": session_key,
            "created_at": datetime.now().isoformat(),
            "command": command,
            "description": approval_payload.get("description", "command flagged"),
            "pattern_key": approval_payload.get("pattern_key", ""),
            "on_approve": on_approve,
        }
        self._pending_approvals[session_key] = pending
        return approval_id

    def _run_bash_process(
        self,
        *,
        source: SessionSource,
        session_key: str,
        command: str,
        force: bool = False,
    ) -> dict[str, Any]:
        from tools.terminal_tool import terminal_tool

        context = build_session_context(source, self.config)
        tracked_vars = [
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_SESSION_THREAD_ID",
            "HERMES_SESSION_KEY",
            "HERMES_SESSION_SEND_POLICY",
        ]
        previous_env = {name: os.environ.get(name) for name in tracked_vars}
        try:
            self._set_session_env(context)
            os.environ["HERMES_SESSION_KEY"] = session_key
            os.environ["HERMES_SESSION_SEND_POLICY"] = self._session_send_policy(session_key)
            raw = terminal_tool(command=command, background=True, force=force)
        finally:
            self._clear_session_env()
            for name in ("HERMES_SESSION_KEY", "HERMES_SESSION_SEND_POLICY"):
                previous = previous_env.get(name)
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
            for name in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_CHAT_NAME", "HERMES_SESSION_THREAD_ID"):
                previous = previous_env.get(name)
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous

        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                return {"status": "error", "error": raw}
            if isinstance(parsed, dict):
                return parsed
        return {"status": "error", "error": "Unexpected terminal result"}

    def _resume_bash_after_approval(
        self,
        *,
        source: SessionSource,
        session_key: str,
        command: str,
    ) -> str:
        result = self._run_bash_process(
            source=source,
            session_key=session_key,
            command=command,
            force=True,
        )
        return self._format_bash_result(
            session_key=session_key,
            command=command,
            result=result,
        )

    def _format_bash_result(
        self,
        *,
        session_key: str,
        command: str,
        result: dict[str, Any],
    ) -> str:
        from tools.process_registry import process_registry

        status = str(result.get("status") or "").strip()
        if status == "approval_required":
            return str(result.get("error") or "Approval required.")
        if status == "blocked":
            return f"❌ {result.get('error') or 'Command blocked.'}"
        if result.get("error") and not result.get("session_id"):
            return f"❌ Bash command failed: {result.get('error')}"

        proc_id = str(result.get("session_id") or "").strip()
        if not proc_id:
            return f"❌ Bash command failed: {result.get('error') or 'No process session ID returned.'}"

        self._get_bash_jobs()[session_key] = proc_id
        wait_seconds = self._get_bash_foreground_seconds()
        wait_result = process_registry.wait(proc_id, timeout=wait_seconds) if wait_seconds > 0 else {"status": "timeout"}

        if wait_result.get("status") == "exited":
            self._get_bash_jobs().pop(session_key, None)
            exit_code = wait_result.get("exit_code")
            output = wait_result.get("output", "")
            return "\n".join(
                [
                    "💻 **Bash Complete**",
                    "",
                    f"**Command:** `{command}`",
                    f"**Exit Code:** `{exit_code}`",
                    "",
                    self._format_bash_output(output),
                ]
            )

        if wait_result.get("status") == "interrupted":
            self._get_bash_jobs().pop(session_key, None)
            return "\n".join(
                [
                    "⚡ **Bash Interrupted**",
                    "",
                    f"**Command:** `{command}`",
                    "",
                    self._format_bash_output(wait_result.get("output", "")),
                ]
            )

        process_info = process_registry.poll(proc_id)
        pid = process_info.get("pid")
        preview = process_info.get("output_preview", "")
        return "\n".join(
            [
                "💻 **Bash Running**",
                "",
                f"**Command:** `{command}`",
                f"**Session ID:** `{proc_id}`",
                f"**PID:** `{pid}`" if pid else "**PID:** `unknown`",
                "",
                "Use `/bash poll` or `!poll` to check progress.",
                "Use `/bash stop` or `!stop` to terminate it.",
                "",
                self._format_bash_output(preview, limit=1200),
            ]
        )

    @staticmethod
    def _control_commands_during_run() -> set[str]:
        return {"status", "stop", "approve", "bash", "subagents", "kill", "steer", "tell", "acp"}

    def _get_subagent_runtime(self) -> dict[str, dict[str, Any]]:
        runtime = getattr(self, "_subagent_runtime", None)
        if not isinstance(runtime, dict):
            runtime = {}
            self._subagent_runtime = runtime
        return runtime

    def _get_subagent_session_state(self, session_key: str) -> dict[str, Any]:
        runtime = self._get_subagent_runtime()
        state = runtime.get(session_key)
        if not isinstance(state, dict):
            state = {"next_ordinal": 1, "entries": {}}
            runtime[session_key] = state
        state.setdefault("next_ordinal", 1)
        state.setdefault("entries", {})
        return state

    @staticmethod
    def _subagent_now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _truncate_line(text: str, *, limit: int = 120) -> str:
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    def _append_subagent_log(self, entry: dict[str, Any], message: str) -> None:
        logs = entry.setdefault("logs", [])
        logs.append(f"[{self._subagent_now()}] {message}")
        if len(logs) > 60:
            del logs[:-60]
        entry["updated_at"] = self._subagent_now()

    def _create_subagent_entry(
        self,
        *,
        session_key: str,
        child: Any,
        task_index: int,
        goal: str,
        context: Optional[str],
        toolsets: Optional[list[str]],
        model: Optional[str],
        source_kind: str,
    ) -> dict[str, Any]:
        state = self._get_subagent_session_state(session_key)
        ordinal = int(state.get("next_ordinal", 1))
        state["next_ordinal"] = ordinal + 1
        subagent_id = f"sa-{ordinal}"
        entry = {
            "id": subagent_id,
            "ordinal": ordinal,
            "task_index": task_index,
            "goal": str(goal or "").strip(),
            "context": str(context or "").strip(),
            "toolsets": list(toolsets or []),
            "model": model or "",
            "status": "pending",
            "source_kind": source_kind,
            "created_at": self._subagent_now(),
            "updated_at": self._subagent_now(),
            "started_at": None,
            "ended_at": None,
            "summary": "",
            "error": "",
            "logs": [],
            "restarts": 0,
            "child": child,
            "pending_steer": None,
        }
        state["entries"][subagent_id] = entry
        self._append_subagent_log(entry, f"Registered subagent for goal: {self._truncate_line(goal)}")
        return entry

    def _find_subagent_entry(self, session_key: str, child: Any) -> Optional[dict[str, Any]]:
        if child is None:
            return None
        state = self._get_subagent_session_state(session_key)
        for entry in state["entries"].values():
            if entry.get("child") is child:
                return entry
        child_id = str(getattr(child, "_gateway_subagent_id", "") or "").strip()
        if child_id:
            return state["entries"].get(child_id)
        return None

    def _resolve_subagent_targets(
        self,
        session_key: str,
        raw_target: str,
        *,
        active_only: bool = False,
        allow_all: bool = False,
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        state = self._get_subagent_session_state(session_key)
        entries = list(state["entries"].values())
        if active_only:
            entries = [e for e in entries if e.get("status") in {"pending", "running"}]
        target = str(raw_target or "").strip()
        if not target:
            if len(entries) == 1:
                return [entries[0]], None
            if not entries:
                return [], "No matching subagents found for this session."
            return [], "Multiple subagents match. Use an explicit `sa-#` or `#<n>` target."
        if allow_all and target.lower() == "all":
            return entries, None
        if target.startswith("#"):
            target = target[1:].strip()
        if target.isdigit():
            ordinal = int(target)
            for entry in entries:
                if int(entry.get("ordinal", 0)) == ordinal:
                    return [entry], None
            return [], f"No subagent found for `#{ordinal}`."
        match = state["entries"].get(target)
        if match and (not active_only or match.get("status") in {"pending", "running"}):
            return [match], None
        return [], f"No subagent found for `{target}`."

    def _format_subagent_list(self, session_key: str) -> str:
        state = self._get_subagent_session_state(session_key)
        entries = list(state["entries"].values())
        if not entries:
            return "No subagents recorded for this session."
        lines = ["🤖 **Subagents**", ""]
        for entry in sorted(entries, key=lambda item: int(item.get("ordinal", 0))):
            status = str(entry.get("status") or "unknown")
            goal = self._truncate_line(entry.get("goal") or "(no goal)")
            lines.append(f"- `#{entry['ordinal']}` / `{entry['id']}` — {status} — {goal}")
            summary = str(entry.get("summary") or "").strip()
            if summary:
                lines.append(f"  summary: {self._truncate_line(summary, limit=140)}")
        return "\n".join(lines)

    def _format_subagent_info(self, entry: dict[str, Any]) -> str:
        lines = [
            "🤖 **Subagent Info**",
            "",
            f"**ID:** `{entry['id']}`",
            f"**Ordinal:** `#{entry['ordinal']}`",
            f"**Status:** `{entry.get('status')}`",
            f"**Source:** `{entry.get('source_kind')}`",
            f"**Model:** `{entry.get('model') or 'default'}`",
            f"**Toolsets:** `{', '.join(entry.get('toolsets') or []) or 'default'}`",
            f"**Goal:** {entry.get('goal') or '_(empty)_'}",
        ]
        if entry.get("error"):
            lines.append(f"**Error:** `{entry['error']}`")
        if entry.get("summary"):
            lines.append(f"**Summary:** {entry['summary']}")
        if entry.get("started_at"):
            lines.append(f"**Started:** `{entry['started_at']}`")
        if entry.get("ended_at"):
            lines.append(f"**Ended:** `{entry['ended_at']}`")
        return "\n".join(lines)

    def _format_subagent_log(self, entry: dict[str, Any]) -> str:
        lines = ["🤖 **Subagent Log**", "", f"**ID:** `{entry['id']}`", ""]
        logs = entry.get("logs") or []
        if not logs:
            lines.append("_(no log events yet)_")
        else:
            lines.append("```")
            lines.extend(logs[-25:])
            lines.append("```")
        return "\n".join(lines)

    def _attach_subagent_runtime_hooks(self, agent: Any, session_key: str) -> None:
        if agent is None:
            return
        agent._register_delegate_child = (
            lambda **payload: self._register_delegate_child(session_key=session_key, **payload)
        )
        agent._start_delegate_child = (
            lambda **payload: self._start_delegate_child(session_key=session_key, **payload)
        )
        agent._record_delegate_child_progress = (
            lambda **payload: self._record_delegate_child_progress(session_key=session_key, **payload)
        )
        agent._complete_delegate_child = (
            lambda **payload: self._complete_delegate_child(session_key=session_key, **payload)
        )

    def _register_delegate_child(
        self,
        *,
        session_key: str,
        child: Any,
        task_index: int,
        goal: str,
        context: Optional[str],
        toolsets: Optional[list[str]],
        model: Optional[str],
    ) -> str:
        entry = self._create_subagent_entry(
            session_key=session_key,
            child=child,
            task_index=task_index,
            goal=goal,
            context=context,
            toolsets=toolsets,
            model=model,
            source_kind="delegate",
        )
        return entry["id"]

    def _start_delegate_child(
        self,
        *,
        session_key: str,
        child: Any,
        goal: str,
        restarted: bool = False,
    ) -> None:
        entry = self._find_subagent_entry(session_key, child)
        if not entry:
            return
        entry["status"] = "running"
        entry["started_at"] = self._subagent_now()
        entry["updated_at"] = self._subagent_now()
        entry["goal"] = str(goal or "").strip()
        if restarted:
            entry["restarts"] = int(entry.get("restarts") or 0) + 1
            self._append_subagent_log(entry, f"Restarted on steer: {self._truncate_line(goal)}")
        else:
            self._append_subagent_log(entry, f"Started: {self._truncate_line(goal)}")

    def _record_delegate_child_progress(
        self,
        *,
        session_key: str,
        child: Any,
        task_index: int,
        tool_name: str,
        preview: Optional[str],
        args: Optional[dict[str, Any]],
        summary: Optional[str],
    ) -> None:
        entry = self._find_subagent_entry(session_key, child)
        if not entry:
            return
        text = summary or preview or tool_name
        entry["last_progress"] = str(text or "").strip()
        self._append_subagent_log(entry, f"{tool_name}: {self._truncate_line(text, limit=160)}")

    def _complete_delegate_child(
        self,
        *,
        session_key: str,
        child: Any,
        entry: dict[str, Any],
        goal: str,
    ) -> None:
        stored = self._find_subagent_entry(session_key, child)
        if not stored:
            return
        stored["status"] = entry.get("status", "completed")
        stored["summary"] = str(entry.get("summary") or "").strip()
        stored["error"] = str(entry.get("error") or "").strip()
        stored["goal"] = str(goal or stored.get("goal") or "").strip()
        stored["ended_at"] = self._subagent_now()
        stored["updated_at"] = self._subagent_now()
        stored["child"] = None
        if stored["status"] == "completed":
            self._append_subagent_log(stored, f"Completed: {self._truncate_line(stored.get('summary') or 'done', limit=160)}")
        elif stored["status"] == "interrupted":
            self._append_subagent_log(stored, "Interrupted.")
        else:
            detail = stored.get("error") or stored.get("summary") or "failed"
            self._append_subagent_log(stored, f"Finished with {stored['status']}: {self._truncate_line(detail, limit=160)}")

    def _build_subagent_progress_callback(self, session_key: str, child: Any):
        def _callback(tool_name: str, preview: str = None, args: dict = None):
            self._record_delegate_child_progress(
                session_key=session_key,
                child=child,
                task_index=int(getattr(child, "_gateway_subagent_task_index", 0) or 0),
                tool_name=tool_name,
                preview=preview,
                args=args,
                summary=preview or tool_name,
            )

        return _callback

    def _get_acp_session_manager(self):
        manager = getattr(self, "_acp_session_manager", None)
        if manager is None:
            from acp_adapter.session import SessionManager

            manager = SessionManager()
            self._acp_session_manager = manager
        return manager

    def _get_acp_bindings(self) -> dict[str, str]:
        bindings = getattr(self, "_acp_session_bindings", None)
        if not isinstance(bindings, dict):
            bindings = {}
            self._acp_session_bindings = bindings
        return bindings

    def _get_acp_meta(self) -> dict[str, dict[str, Any]]:
        meta = getattr(self, "_acp_session_meta", None)
        if not isinstance(meta, dict):
            meta = {}
            self._acp_session_meta = meta
        return meta

    def _get_acp_tasks(self) -> dict[str, asyncio.Task]:
        tasks = getattr(self, "_acp_running_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._acp_running_tasks = tasks
        return tasks

    def _get_acp_session_meta_entry(self, session_id: str) -> dict[str, Any]:
        meta = self._get_acp_meta()
        entry = meta.get(session_id)
        if not isinstance(entry, dict):
            entry = {"options": {}, "last_result": None, "last_prompt": "", "updated_at": self._subagent_now()}
            meta[session_id] = entry
        entry.setdefault("options", {})
        entry.setdefault("updated_at", self._subagent_now())
        return entry

    def _resolve_acp_target(self, session_key: str, payload: str) -> tuple[Optional[str], str]:
        manager = self._get_acp_session_manager()
        text = str(payload or "").strip()
        parts = self._split_command_args(text, max_parts=2)
        if parts and manager.get_session(parts[0]):
            return parts[0], parts[1] if len(parts) > 1 else ""
        bound = self._get_acp_bindings().get(session_key)
        return bound, text

    def _format_acp_sessions(self, session_key: str) -> str:
        manager = self._get_acp_session_manager()
        listing = manager.list_sessions()
        if not listing:
            return "No ACP sessions are active."
        bound = self._get_acp_bindings().get(session_key)
        tasks = self._get_acp_tasks()
        lines = ["🧩 **ACP Sessions**", ""]
        for item in listing:
            session_id = item.get("session_id", "")
            busy = session_id in tasks and not tasks[session_id].done()
            label = " (bound)" if session_id == bound else ""
            lines.append(
                f"- `{session_id}`{label} — cwd `{item.get('cwd')}` — model `{item.get('model') or 'default'}` — busy `{str(busy).lower()}`"
            )
        return "\n".join(lines)

    def _format_acp_status(self, session_id: str) -> str:
        manager = self._get_acp_session_manager()
        state = manager.get_session(session_id)
        if state is None:
            return f"No ACP session found for `{session_id}`."
        meta = self._get_acp_session_meta_entry(session_id)
        task = self._get_acp_tasks().get(session_id)
        busy = bool(task and not task.done())
        lines = [
            "🧩 **ACP Status**",
            "",
            f"**Session:** `{session_id}`",
            f"**Busy:** `{str(busy).lower()}`",
            f"**CWD:** `{state.cwd}`",
            f"**Model:** `{state.model or getattr(state.agent, 'model', '') or 'default'}`",
            f"**History Messages:** `{len(state.history)}`",
        ]
        options = meta.get("options") or {}
        if options:
            lines.append(f"**Options:** `{json.dumps(options, default=str, sort_keys=True)}`")
        if meta.get("last_prompt"):
            lines.append(f"**Last Prompt:** {self._truncate_line(meta['last_prompt'], limit=180)}")
        last_result = meta.get("last_result") or {}
        if isinstance(last_result, dict) and last_result.get("final_response"):
            lines.append(f"**Last Result:** {self._truncate_line(last_result['final_response'], limit=180)}")
        return "\n".join(lines)

    def _prepare_skill_command(
        self,
        event: MessageEvent,
        *,
        task_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Rewrite `/skill` into the underlying skill invocation message."""
        args_text = event.get_command_args().strip()
        parts = self._split_command_args(args_text, max_parts=2)
        if not parts:
            from agent.skill_commands import get_skill_commands

            skill_cmds = get_skill_commands()
            if not skill_cmds:
                return None, "No skill commands are installed."
            lines = ["⚡ **Installed Skills**", ""]
            for name, payload in sorted(skill_cmds.items()):
                lines.append(f"`{name}` — {payload.get('description', 'Skill command')}")
            lines.append("")
            lines.append("Usage: `/skill <name> [input]`")
            return None, "\n".join(lines)

        from agent.skill_commands import build_skill_invocation_message

        skill_name = self._normalize_skill_name(parts[0])
        user_instruction = parts[1] if len(parts) > 1 else ""
        cmd_key = f"/{skill_name}"
        prepared = build_skill_invocation_message(
            cmd_key,
            user_instruction,
            task_id=task_id,
        )
        if not prepared:
            return None, f"Unknown skill `{skill_name}`. Use `/skill` to list installed skills."
        event.text = prepared
        return prepared, None

    def _schedule_gateway_restart(self, delay_seconds: float = 1.5) -> None:
        async def _restart_later():
            await asyncio.sleep(delay_seconds)
            hermes_cmd = _resolve_hermes_bin()
            if not hermes_cmd:
                logger.error("Could not locate hermes executable for gateway restart")
                return
            cmd = hermes_cmd + ["gateway", "restart"]
            try:
                import shutil
                import subprocess

                systemd_run = shutil.which("systemd-run")
                if systemd_run:
                    subprocess.Popen(
                        [systemd_run, "--user", "--scope", "--unit=hermes-discord-restart", "--", *cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                else:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
            except Exception:
                logger.exception("Gateway restart scheduling failed")

        asyncio.create_task(_restart_later())

    async def _deliver_docked_response(
        self,
        *,
        session_key: str,
        response: str,
        source: SessionSource,
    ) -> Optional[str]:
        target = self._dock_target_for_session(session_key)
        if not target or not response.strip():
            return response

        target_platform = str(target.get("platform") or "").strip().lower()
        if not target_platform:
            return response

        try:
            platform = Platform(target_platform)
        except ValueError:
            return response

        same_target = (
            platform == source.platform
            and str(target.get("chat_id") or "") == str(source.chat_id)
            and str(target.get("thread_id") or "") == str(source.thread_id or "")
        )
        if same_target:
            return response

        dock_target = DeliveryTarget(
            platform=platform,
            chat_id=str(target.get("chat_id") or ""),
            thread_id=str(target.get("thread_id") or "").strip() or None,
        )
        results = await self.delivery_router.deliver(
            response,
            [dock_target],
            metadata={
                "source_platform": source.platform.value if source.platform else "unknown",
                "source_chat_id": source.chat_id,
            },
        )
        result = results.get(dock_target.to_string()) or {}
        if result.get("success"):
            return None
        error = result.get("error") or "unknown dock delivery failure"
        return f"⚠️ Reply docking failed: {error}\n\n{response}"

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        from agent.smart_model_routing import resolve_turn_route

        primary = {
            "model": model,
            "api_key": runtime_kwargs.get("api_key"),
            "base_url": runtime_kwargs.get("base_url"),
            "provider": runtime_kwargs.get("provider"),
            "api_mode": runtime_kwargs.get("api_mode"),
            "command": runtime_kwargs.get("command"),
            "args": list(runtime_kwargs.get("args") or []),
        }
        return resolve_turn_route(user_message, getattr(self, "_smart_model_routing", {}), primary)

    async def _handle_adapter_fatal_error(self, adapter: BasePlatformAdapter) -> None:
        """React to a non-retryable adapter failure after startup."""
        logger.error(
            "Fatal %s adapter error (%s): %s",
            adapter.platform.value,
            adapter.fatal_error_code or "unknown",
            adapter.fatal_error_message or "unknown error",
        )

        existing = self.adapters.get(adapter.platform)
        if existing is adapter:
            try:
                await adapter.disconnect()
            finally:
                self.adapters.pop(adapter.platform, None)
                self.delivery_router.adapters = self.adapters

        if not self.adapters:
            self._exit_reason = adapter.fatal_error_message or "All messaging adapters disconnected"
            logger.error("No connected messaging platforms remain. Shutting down gateway cleanly.")
            await self.stop()

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()
    
    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.
        
        Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
        the prefill_messages_file key in ~/.hermes/config.yaml.
        Relative paths are resolved from ~/.hermes/.
        """
        import json as _json
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    file_path = cfg.get("prefill_messages_file", "")
            except Exception:
                pass
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
                data = _json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt from config or env var.
        
        Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
        agent.system_prompt in ~/.hermes/config.yaml.
        """
        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return (cfg.get("agent", {}).get("system_prompt", "") or "").strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _load_reasoning_config() -> dict | None:
        """Load reasoning effort from config with env fallback.

        Checks agent.reasoning_effort in config.yaml first, then
        HERMES_REASONING_EFFORT as a fallback. Valid: "xhigh", "high",
        "medium", "low", "minimal", "none". Returns None to use default
        (medium).
        """
        effort = ""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                effort = str(cfg.get("agent", {}).get("reasoning_effort", "") or "").strip()
        except Exception:
            pass
        if not effort:
            effort = os.getenv("HERMES_REASONING_EFFORT", "")
        if not effort:
            return None
        effort = effort.lower().strip()
        if effort == "none":
            return {"enabled": False}
        valid = ("xhigh", "high", "medium", "low", "minimal")
        if effort in valid:
            return {"enabled": True, "effort": effort}
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
        return None

    @staticmethod
    def _load_show_reasoning() -> bool:
        """Load show_reasoning toggle from config.yaml display section."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return bool(cfg.get("display", {}).get("show_reasoning", False))
        except Exception:
            pass
        return False

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Load background process notification mode from config or env var.

        Modes:
          - ``all``    — push running-output updates *and* the final message (default)
          - ``result`` — only the final completion message (regardless of exit code)
          - ``error``  — only the final message when exit code is non-zero
          - ``off``    — no watcher messages at all
        """
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    raw = cfg.get("display", {}).get("background_process_notifications")
                    if raw is False:
                        mode = "off"
                    elif raw not in (None, ""):
                        mode = str(raw)
            except Exception:
                pass
        mode = (mode or "all").strip().lower()
        valid = {"all", "result", "error", "off"}
        if mode not in valid:
            logger.warning(
                "Unknown background_process_notifications '%s', defaulting to 'all'",
                mode,
            )
            return "all"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """Load OpenRouter provider routing preferences from config.yaml."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return cfg.get("provider_routing", {}) or {}
        except Exception:
            pass
        return {}

    @staticmethod
    def _load_fallback_model() -> dict | None:
        """Load fallback model config from config.yaml.

        Returns a dict with 'provider' and 'model' keys, or None if
        not configured / both fields empty.
        """
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                fb = cfg.get("fallback_model", {}) or {}
                if fb.get("provider") and fb.get("model"):
                    return fb
        except Exception:
            pass
        return None

    @staticmethod
    def _load_smart_model_routing() -> dict:
        """Load optional smart cheap-vs-strong model routing config."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return cfg.get("smart_model_routing", {}) or {}
        except Exception:
            pass
        return {}

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        logger.info("Session storage: %s", self.config.sessions_dir)
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(gateway_state="starting", exit_reason=None)
        except Exception:
            pass
        
        # Warn if no user allowlists are configured and open access is not opted in
        _any_allowlist = any(
            os.getenv(v)
            for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                       "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
                       "SMS_ALLOWED_USERS",
                       "GATEWAY_ALLOWED_USERS")
        )
        _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")
        if not _any_allowlist and not _allow_all:
            logger.warning(
                "No user allowlists configured. All unauthorized users will be denied. "
                "Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access, "
                "or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id)."
            )
        
        # Discover and load event hooks
        self.hooks.discover_and_load()
        
        # Recover background processes from checkpoint (crash recovery)
        try:
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        except Exception as e:
            logger.warning("Process checkpoint recovery: %s", e)
        
        connected_count = 0
        enabled_platform_count = 0
        startup_nonretryable_errors: list[str] = []
        startup_retryable_errors: list[str] = []
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
            enabled_platform_count += 1
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                logger.warning("No adapter available for %s", platform.value)
                continue
            
            # Set up message + fatal error handlers
            adapter.set_message_handler(self._handle_message)
            adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
            if platform == Platform.DISCORD and hasattr(adapter, "_resolve_exec_approval"):
                adapter._resolve_exec_approval = self._resolve_pending_approval
            
            # Try to connect
            logger.info("Connecting to %s...", platform.value)
            try:
                success = await adapter.connect()
                if success:
                    self.adapters[platform] = adapter
                    self._sync_voice_mode_state_to_adapter(adapter)
                    connected_count += 1
                    logger.info("✓ %s connected", platform.value)
                else:
                    logger.warning("✗ %s failed to connect", platform.value)
                    if adapter.has_fatal_error:
                        target = (
                            startup_retryable_errors
                            if adapter.fatal_error_retryable
                            else startup_nonretryable_errors
                        )
                        target.append(
                            f"{platform.value}: {adapter.fatal_error_message}"
                        )
                    else:
                        startup_retryable_errors.append(
                            f"{platform.value}: failed to connect"
                        )
            except Exception as e:
                logger.error("✗ %s error: %s", platform.value, e)
                startup_retryable_errors.append(f"{platform.value}: {e}")
        
        if connected_count == 0:
            if startup_nonretryable_errors:
                reason = "; ".join(startup_nonretryable_errors)
                logger.error("Gateway hit a non-retryable startup conflict: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
                except Exception:
                    pass
                self._request_clean_exit(reason)
                return True
            if enabled_platform_count > 0:
                reason = "; ".join(startup_retryable_errors) or "all configured messaging platforms failed to connect"
                logger.error("Gateway failed to connect any configured messaging platform: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
                except Exception:
                    pass
                return False
            logger.warning("No messaging platforms enabled.")
            logger.info("Gateway will continue running for cron job execution.")
        
        # Update delivery router with adapters
        self.delivery_router.adapters = self.adapters
        
        self._running = True
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(gateway_state="running", exit_reason=None)
        except Exception:
            pass
        
        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters.keys()],
        })
        
        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)
        
        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)
        
        # Check if we're restarting after a /update command. If the update is
        # still running, keep watching so we notify once it actually finishes.
        notified = await self._send_update_notification()
        if not notified and any(
            path.exists()
            for path in (
                _hermes_home / ".update_pending.json",
                _hermes_home / ".update_pending.claimed.json",
            )
        ):
            self._schedule_update_notification_watch()

        # Drain any recovered process watchers (from crash recovery checkpoint)
        try:
            from tools.process_registry import process_registry
            while process_registry.pending_watchers:
                watcher = process_registry.pending_watchers.pop(0)
                asyncio.create_task(self._run_process_watcher(watcher))
                logger.info("Resumed watcher for recovered process %s", watcher.get("session_id"))
        except Exception as e:
            logger.error("Recovered watcher setup error: %s", e)

        # Start background session expiry watcher for proactive memory flushing
        asyncio.create_task(self._session_expiry_watcher())

        logger.info("Press Ctrl+C to stop")
        
        return True
    
    async def _session_expiry_watcher(self, interval: int = 300):
        """Background task that proactively flushes memories for expired sessions.
        
        Runs every `interval` seconds (default 5 min).  For each session that
        has expired according to its reset policy, flushes memories in a thread
        pool and marks the session so it won't be flushed again.

        This means memories are already saved by the time the user sends their
        next message, so there's no blocking delay.
        """
        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        while self._running:
            try:
                self.session_store._ensure_loaded()
                for key, entry in list(self.session_store._entries.items()):
                    if entry.session_id in self.session_store._pre_flushed_sessions:
                        continue  # already flushed this session
                    if not self.session_store._is_session_expired(entry):
                        continue  # session still active
                    # Session has expired — flush memories in the background
                    logger.info(
                        "Session %s expired (key=%s), flushing memories proactively",
                        entry.session_id, key,
                    )
                    try:
                        await self._async_flush_memories(entry.session_id, key)
                        self._shutdown_gateway_honcho(key)
                        self.session_store._pre_flushed_sessions.add(entry.session_id)
                    except Exception as e:
                        logger.debug("Proactive memory flush failed for %s: %s", entry.session_id, e)
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the gateway and disconnect all adapters."""
        logger.info("Stopping gateway...")
        self._running = False

        for session_key, agent in list(self._running_agents.items()):
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                agent.interrupt("Gateway shutting down")
                logger.debug("Interrupted running agent for session %s during shutdown", session_key[:20])
            except Exception as e:
                logger.debug("Failed interrupting agent during shutdown: %s", e)

        for runtime in self._get_subagent_runtime().values():
            for entry in runtime.get("entries", {}).values():
                child = entry.get("child")
                if child is not None and hasattr(child, "interrupt"):
                    try:
                        child.interrupt("Gateway shutting down")
                    except Exception as e:
                        logger.debug("Failed interrupting subagent during shutdown: %s", e)

        for session_id, task in list(self._get_acp_tasks().items()):
            state = self._get_acp_session_manager().get_session(session_id)
            if state and state.cancel_event:
                state.cancel_event.set()
                try:
                    state.agent.interrupt("Gateway shutting down")
                except Exception:
                    logger.debug("Failed interrupting ACP session during shutdown", exc_info=True)
            if task and not task.done():
                task.cancel()

        for platform, adapter in list(self.adapters.items()):
            try:
                await adapter.cancel_background_tasks()
            except Exception as e:
                logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
            try:
                await adapter.disconnect()
                logger.info("✓ %s disconnected", platform.value)
            except Exception as e:
                logger.error("✗ %s disconnect error: %s", platform.value, e)

        self.adapters.clear()
        self._running_agents.clear()
        self._pending_messages.clear()
        self._pending_approvals.clear()
        self._shutdown_all_gateway_honcho()
        self._shutdown_event.set()
        
        from gateway.status import remove_pid_file, write_runtime_status
        remove_pid_file()
        try:
            write_runtime_status(gateway_state="stopped", exit_reason=self._exit_reason)
        except Exception:
            pass
        
        logger.info("Gateway stopped")
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()
    
    def _create_adapter(
        self, 
        platform: Platform, 
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        """Create the appropriate adapter for a platform."""
        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault(
                "group_sessions_per_user",
                self.config.group_sessions_per_user,
            )

        if platform == Platform.TELEGRAM:
            from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements
            if not check_telegram_requirements():
                logger.warning("Telegram: python-telegram-bot not installed")
                return None
            return TelegramAdapter(config)
        
        elif platform == Platform.DISCORD:
            from gateway.platforms.discord import DiscordAdapter, check_discord_requirements
            if not check_discord_requirements():
                logger.warning("Discord: discord.py not installed")
                return None
            return DiscordAdapter(config)
        
        elif platform == Platform.WHATSAPP:
            from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                logger.warning("WhatsApp: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)
        
        elif platform == Platform.SLACK:
            from gateway.platforms.slack import SlackAdapter, check_slack_requirements
            if not check_slack_requirements():
                logger.warning("Slack: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'")
                return None
            return SlackAdapter(config)

        elif platform == Platform.SIGNAL:
            from gateway.platforms.signal import SignalAdapter, check_signal_requirements
            if not check_signal_requirements():
                logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
                return None
            return SignalAdapter(config)

        elif platform == Platform.HOMEASSISTANT:
            from gateway.platforms.homeassistant import HomeAssistantAdapter, check_ha_requirements
            if not check_ha_requirements():
                logger.warning("HomeAssistant: aiohttp not installed or HASS_TOKEN not set")
                return None
            return HomeAssistantAdapter(config)

        elif platform == Platform.EMAIL:
            from gateway.platforms.email import EmailAdapter, check_email_requirements
            if not check_email_requirements():
                logger.warning("Email: EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set")
                return None
            return EmailAdapter(config)

        elif platform == Platform.SMS:
            from gateway.platforms.sms import SmsAdapter, check_sms_requirements
            if not check_sms_requirements():
                logger.warning("SMS: aiohttp not installed or TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set")
                return None
            return SmsAdapter(config)

        elif platform == Platform.DINGTALK:
            from gateway.platforms.dingtalk import DingTalkAdapter, check_dingtalk_requirements
            if not check_dingtalk_requirements():
                logger.warning("DingTalk: dingtalk-stream not installed or DINGTALK_CLIENT_ID/SECRET not set")
                return None
            return DingTalkAdapter(config)

        elif platform == Platform.MATTERMOST:
            from gateway.platforms.mattermost import MattermostAdapter, check_mattermost_requirements
            if not check_mattermost_requirements():
                logger.warning("Mattermost: MATTERMOST_TOKEN or MATTERMOST_URL not set, or aiohttp missing")
                return None
            return MattermostAdapter(config)

        elif platform == Platform.MATRIX:
            from gateway.platforms.matrix import MatrixAdapter, check_matrix_requirements
            if not check_matrix_requirements():
                logger.warning("Matrix: matrix-nio not installed or credentials not set. Run: pip install 'matrix-nio[e2e]'")
                return None
            return MatrixAdapter(config)

        elif platform == Platform.API_SERVER:
            from gateway.platforms.api_server import APIServerAdapter, check_api_server_requirements
            if not check_api_server_requirements():
                logger.warning("API Server: aiohttp not installed")
                return None
            return APIServerAdapter(config)

        elif platform == Platform.WEBHOOK:
            from gateway.platforms.webhook import WebhookAdapter, check_webhook_requirements
            if not check_webhook_requirements():
                logger.warning("Webhook: aiohttp not installed")
                return None
            adapter = WebhookAdapter(config)
            adapter.gateway_runner = self  # For cross-platform delivery
            return adapter

        return None
    
    def _is_user_authorized(self, source: SessionSource) -> bool:
        """
        Check if a user is authorized to use the bot.
        
        Checks in order:
        1. Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        2. Environment variable allowlists (TELEGRAM_ALLOWED_USERS, etc.)
        3. DM pairing approved list
        4. Global allow-all (GATEWAY_ALLOW_ALL_USERS=true)
        5. Default: deny
        """
        # Home Assistant events are system-generated (state changes), not
        # user-initiated messages.  The HASS_TOKEN already authenticates the
        # connection, so HA events are always authorized.
        # Webhook events are authenticated via HMAC signature validation in
        # the adapter itself — no user allowlist applies.
        if source.platform in (Platform.HOMEASSISTANT, Platform.WEBHOOK):
            return True

        user_id = source.user_id
        if not user_id:
            return False

        platform_env_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
            Platform.DISCORD: "DISCORD_ALLOWED_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
            Platform.SLACK: "SLACK_ALLOWED_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOWED_USERS",
            Platform.EMAIL: "EMAIL_ALLOWED_USERS",
            Platform.SMS: "SMS_ALLOWED_USERS",
            Platform.MATTERMOST: "MATTERMOST_ALLOWED_USERS",
            Platform.MATRIX: "MATRIX_ALLOWED_USERS",
            Platform.DINGTALK: "DINGTALK_ALLOWED_USERS",
        }
        platform_allow_all_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
            Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
            Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOW_ALL_USERS",
            Platform.EMAIL: "EMAIL_ALLOW_ALL_USERS",
            Platform.SMS: "SMS_ALLOW_ALL_USERS",
            Platform.MATTERMOST: "MATTERMOST_ALLOW_ALL_USERS",
            Platform.MATRIX: "MATRIX_ALLOW_ALL_USERS",
            Platform.DINGTALK: "DINGTALK_ALLOW_ALL_USERS",
        }

        # Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        platform_allow_all_var = platform_allow_all_map.get(source.platform, "")
        if platform_allow_all_var and os.getenv(platform_allow_all_var, "").lower() in ("true", "1", "yes"):
            return True

        # Check pairing store (always checked, regardless of allowlists)
        platform_name = source.platform.value if source.platform else ""
        if self.pairing_store.is_approved(platform_name, user_id):
            return True

        # Check platform-specific and global allowlists
        platform_allowlist = os.getenv(platform_env_map.get(source.platform, ""), "").strip()
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "").strip()

        if not platform_allowlist and not global_allowlist:
            # No allowlists configured -- check global allow-all flag
            return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")

        # Check if user is in any allowlist
        allowed_ids = set()
        if platform_allowlist:
            allowed_ids.update(uid.strip() for uid in platform_allowlist.split(",") if uid.strip())
        if global_allowlist:
            allowed_ids.update(uid.strip() for uid in global_allowlist.split(",") if uid.strip())

        # WhatsApp JIDs have @s.whatsapp.net suffix — strip it for comparison
        check_ids = {user_id}
        if "@" in user_id:
            check_ids.add(user_id.split("@")[0])
        return bool(check_ids & allowed_ids)

    def _get_unauthorized_dm_behavior(self, platform: Optional[Platform]) -> str:
        """Return how unauthorized DMs should be handled for a platform."""
        config = getattr(self, "config", None)
        if config and hasattr(config, "get_unauthorized_dm_behavior"):
            return config.get_unauthorized_dm_behavior(platform)
        return "pair"
    
    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """
        Handle an incoming message from any platform.
        
        This is the core message processing pipeline:
        1. Check user authorization
        2. Check for commands (/new, /reset, etc.)
        3. Check for running agent and interrupt if needed
        4. Get or create session
        5. Build context for agent
        6. Run agent conversation
        7. Return response
        """
        source = event.source
        session_source = self._session_source_for_event(event)

        # Check if user is authorized
        if not self._is_user_authorized(source):
            logger.warning("Unauthorized user: %s (%s) on %s", source.user_id, source.user_name, source.platform.value)
            # In DMs: offer pairing code. In groups: silently ignore.
            if source.chat_type == "dm" and self._get_unauthorized_dm_behavior(source.platform) == "pair":
                platform_name = source.platform.value if source.platform else "unknown"
                code = self.pairing_store.generate_code(
                    platform_name, source.user_id, source.user_name or ""
                )
                if code:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            f"Hi~ I don't recognize you yet!\n\n"
                            f"Here's your pairing code: `{code}`\n\n"
                            f"Ask the bot owner to run:\n"
                            f"`hermes pairing approve {platform_name} {code}`"
                        )
                else:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            "Too many pairing requests right now~ "
                            "Please try again later!"
                        )
            return None
        
        # User-defined quick commands (bypass agent loop, no LLM call)
        command = event.get_command()
        if command:
            if isinstance(self.config, dict):
                quick_commands = self.config.get("quick_commands", {}) or {}
            else:
                quick_commands = getattr(self.config, "quick_commands", {}) or {}
            if not isinstance(quick_commands, dict):
                quick_commands = {}
            if command in quick_commands:
                qcmd = quick_commands[command]
                if qcmd.get("type") == "exec":
                    exec_cmd = qcmd.get("command", "")
                    if exec_cmd:
                        try:
                            proc = await asyncio.create_subprocess_shell(
                                exec_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                            output = (stdout or stderr).decode().strip()
                            return output if output else "Command returned no output."
                        except asyncio.TimeoutError:
                            return "Quick command timed out (30s)."
                        except Exception as e:
                            return f"Quick command error: {e}"
                    else:
                        return f"Quick command '/{command}' has no command defined."
                elif qcmd.get("type") == "alias":
                    target = qcmd.get("target", "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        user_args = event.get_command_args().strip()
                        event.text = f"{target} {user_args}".strip()
                    else:
                        return f"Quick command '/{command}' has no target defined."
                else:
                    return f"Quick command '/{command}' has unsupported type (supported: 'exec', 'alias')."

        # PRIORITY handling when an agent is already running for this session.
        # Default behavior is to interrupt immediately so user text/stop messages
        # are handled with minimal latency.
        #
        # Special case: Telegram/photo bursts often arrive as multiple near-
        # simultaneous updates. Do NOT interrupt for photo-only follow-ups here;
        # let the adapter-level batching/queueing logic absorb them.
        _quick_key = self._session_key_for_source(session_source)
        if _quick_key in self._running_agents:
            command_name = (event.get_command() or "").strip().lower()
            running_agent = self._running_agents.get(_quick_key)
            if command_name == "status":
                return await self._handle_status_command(event)
            if command_name == "stop" and running_agent is _AGENT_PENDING_SENTINEL:
                return "⏳ The agent is still starting up — nothing to stop yet."
            # /reset and /new must bypass the running-agent guard so they
            # actually dispatch as commands instead of being queued as user
            # text (which would be fed back to the agent with the same
            # broken history — #2170). Interrupt the agent first, then
            # clear the adapter's pending queue so the stale "/reset" text
            # doesn't get re-processed as a user message after the
            # interrupt completes.
            from hermes_cli.commands import resolve_command as _resolve_cmd_inner

            _evt_cmd = event.get_command()
            _cmd_def_inner = _resolve_cmd_inner(_evt_cmd) if _evt_cmd else None
            if _cmd_def_inner and _cmd_def_inner.name == "new":
                if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
                    running_agent.interrupt("Session reset requested")
                adapter = self.adapters.get(source.platform)
                if adapter and hasattr(adapter, "get_pending_message"):
                    adapter.get_pending_message(_quick_key)
                self._pending_messages.pop(_quick_key, None)
                if _quick_key in self._running_agents:
                    del self._running_agents[_quick_key]
                return await self._handle_reset_command(event)

            if _cmd_def_inner and _cmd_def_inner.name == "queue":
                queued_text = event.get_command_args().strip()
                if not queued_text:
                    return "Usage: /queue <prompt>"
                adapter = self.adapters.get(source.platform)
                if adapter:
                    from gateway.platforms.base import MessageEvent as _ME, MessageType as _MT

                    queued_event = _ME(
                        text=queued_text,
                        message_type=_MT.TEXT,
                        source=event.source,
                        message_id=event.message_id,
                    )
                    adapter._pending_messages[_quick_key] = queued_event
                return "Queued for the next turn."

            if command_name in self._control_commands_during_run():
                pass
            elif event.message_type == MessageType.PHOTO:
                logger.debug("PRIORITY photo follow-up for session %s — queueing without interrupt", _quick_key[:20])
                adapter = self.adapters.get(source.platform)
                if adapter:
                    # Reuse adapter queue semantics so photo bursts merge cleanly.
                    if _quick_key in adapter._pending_messages:
                        existing = adapter._pending_messages[_quick_key]
                        if getattr(existing, "message_type", None) == MessageType.PHOTO:
                            existing.media_urls.extend(event.media_urls)
                            existing.media_types.extend(event.media_types)
                            if event.text:
                                if not existing.text:
                                    existing.text = event.text
                                elif event.text not in existing.text:
                                    existing.text = f"{existing.text}\n\n{event.text}".strip()
                        else:
                            adapter._pending_messages[_quick_key] = event
                    else:
                        adapter._pending_messages[_quick_key] = event
                return None
            else:
                if running_agent is _AGENT_PENDING_SENTINEL:
                    if command_name == "stop":
                        return "⏳ The agent is still starting up — nothing to stop yet."
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        adapter._pending_messages[_quick_key] = event
                    return None
                logger.debug("PRIORITY interrupt for session %s", _quick_key[:20])
                running_agent.interrupt(event.text)
                if _quick_key in self._pending_messages:
                    self._pending_messages[_quick_key] += "\n" + event.text
                else:
                    self._pending_messages[_quick_key] = event.text
                return None

        rewritten_bash = self._rewrite_bash_shortcut(event.text)
        if rewritten_bash:
            event.text = rewritten_bash

        # Check for commands
        command = event.get_command()
        
        # Emit command:* hook for any recognized slash command.
        # GATEWAY_KNOWN_COMMANDS is derived from the central COMMAND_REGISTRY
        # in hermes_cli/commands.py — no hardcoded set to maintain here.
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command as _resolve_cmd
        hooks = getattr(self, "hooks", None)
        if command and command in GATEWAY_KNOWN_COMMANDS and hooks is not None:
            await hooks.emit(f"command:{command}", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "command": command,
                "args": event.get_command_args().strip(),
            })

        # Resolve aliases to canonical name so dispatch only checks canonicals.
        _cmd_def = _resolve_cmd(command) if command else None
        canonical = _cmd_def.name if _cmd_def else command

        if canonical == "new":
            return await self._handle_reset_command(event)
        
        if canonical == "help":
            return await self._handle_help_command(event)

        if canonical == "commands":
            return await self._handle_commands_command(event)

        if canonical == "context":
            return await self._handle_context_command(event)

        if canonical == "export-session":
            return await self._handle_export_session_command(event)

        if canonical == "whoami":
            return await self._handle_whoami_command(event)

        if canonical == "focus":
            return await self._handle_focus_command(event)

        if canonical == "unfocus":
            return await self._handle_unfocus_command(event)

        if canonical == "agents":
            return await self._handle_agents_command(event)

        if canonical == "session":
            return await self._handle_session_command(event)
        
        if canonical == "status":
            return await self._handle_status_command(event)

        if canonical == "approve":
            return await self._handle_approve_command(event)

        if canonical == "allowlist":
            return await self._handle_allowlist_command(event)

        if canonical == "config":
            return await self._handle_config_command(event)

        if canonical == "debug":
            return await self._handle_debug_command(event)

        if canonical == "stop":
            return await self._handle_stop_command(event)

        if canonical in {"model", "models"}:
            return await self._handle_model_command(event)

        if canonical == "reasoning":
            return await self._handle_reasoning_command(event)

        if canonical == "think":
            return await self._handle_think_command(event)

        if canonical == "send":
            return await self._handle_send_command(event)

        if canonical == "activation":
            return await self._handle_activation_command(event)

        if canonical == "provider":
            return await self._handle_provider_command(event)
        
        if canonical == "personality":
            return await self._handle_personality_command(event)

        if canonical == "plan":
            try:
                from agent.skill_commands import build_plan_path, build_skill_invocation_message

                user_instruction = event.get_command_args().strip()
                plan_path = build_plan_path(user_instruction)
                event.text = build_skill_invocation_message(
                    "/plan",
                    user_instruction,
                    task_id=_quick_key,
                    runtime_note=(
                        "Save the markdown plan with write_file to this exact relative path "
                        f"inside the active workspace/backend cwd: {plan_path}"
                    ),
                )
                if not event.text:
                    return "Failed to load the bundled /plan skill."
                canonical = None
            except Exception as e:
                logger.exception("Failed to prepare /plan command")
                return f"Failed to enter plan mode: {e}"
        
        if canonical == "retry":
            return await self._handle_retry_command(event)
        
        if canonical == "undo":
            return await self._handle_undo_command(event)
        
        if canonical == "sethome":
            return await self._handle_set_home_command(event)

        if canonical == "compact":
            return await self._handle_compress_command(event)

        if canonical == "compress":
            return await self._handle_compress_command(event)

        if canonical == "usage":
            return await self._handle_usage_command(event)

        if canonical == "insights":
            return await self._handle_insights_command(event)

        if canonical == "reload-mcp":
            return await self._handle_reload_mcp_command(event)

        if canonical == "approve":
            return await self._handle_approve_command(event)

        if canonical == "deny":
            return await self._handle_deny_command(event)

        if canonical == "skill":
            try:
                _prepared, error = self._prepare_skill_command(event, task_id=_quick_key)
                if error:
                    return error
                canonical = None
            except Exception as e:
                logger.exception("Failed to prepare /skill command")
                return f"Failed to invoke skill: {e}"

        if canonical == "update":
            return await self._handle_update_command(event)

        if canonical == "restart":
            return await self._handle_restart_command(event)

        if canonical == "dock-telegram":
            return await self._handle_dock_command(event, Platform.TELEGRAM)

        if canonical == "dock-discord":
            return await self._handle_dock_command(event, Platform.DISCORD)

        if canonical == "dock-slack":
            return await self._handle_dock_command(event, Platform.SLACK)

        if canonical == "title":
            return await self._handle_title_command(event)

        if canonical == "resume":
            return await self._handle_resume_command(event)

        if canonical == "rollback":
            return await self._handle_rollback_command(event)

        if canonical == "background":
            return await self._handle_background_command(event)

        if canonical == "voice":
            return await self._handle_voice_command(event)

        if canonical == "subagents":
            return await self._handle_subagents_command(event)

        if canonical == "kill":
            return await self._handle_kill_command(event)

        if canonical == "steer":
            return await self._handle_steer_command(event)

        if canonical == "acp":
            return await self._handle_acp_command(event)

        if canonical == "bash":
            return await self._handle_bash_command(event)

        # Skill slash commands: /skill-name loads the skill and sends to agent
        if command:
            try:
                from agent.skill_commands import get_skill_commands, build_skill_invocation_message
                skill_cmds = get_skill_commands()
                cmd_key = f"/{command}"
                if cmd_key in skill_cmds:
                    user_instruction = event.get_command_args().strip()
                    msg = build_skill_invocation_message(
                        cmd_key, user_instruction, task_id=_quick_key
                    )
                    if msg:
                        event.text = msg
                        # Fall through to normal message processing with skill content
            except Exception as e:
                logger.debug("Skill command check failed (non-fatal): %s", e)
        
        # Pending exec approvals are handled by /approve and /deny commands above.
        # No bare text matching — "yes" in normal conversation must not trigger
        # execution of a dangerous command.

        # ── Claim this session before any await ───────────────────────
        # Between here and _run_agent registering the real AIAgent, there
        # are numerous await points (hooks, vision enrichment, STT,
        # session hygiene compression).  Without this sentinel a second
        # message arriving during any of those yields would pass the
        # "already running" guard and spin up a duplicate agent for the
        # same session — corrupting the transcript.
        self._running_agents[_quick_key] = _AGENT_PENDING_SENTINEL

        try:
            return await self._handle_message_with_agent(event, source, _quick_key)
        finally:
            # If _run_agent replaced the sentinel with a real agent and
            # then cleaned it up, this is a no-op.  If we exited early
            # (exception, command fallthrough, etc.) the sentinel must
            # not linger or the session would be permanently locked out.
            if self._running_agents.get(_quick_key) is _AGENT_PENDING_SENTINEL:
                del self._running_agents[_quick_key]

    async def _handle_message_with_agent(self, event, source, _quick_key: str):
        """Inner handler that runs under the _running_agents sentinel guard."""

        # Get or create session
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        # Emit session:start for new or auto-reset sessions
        _is_new_session = (
            session_entry.created_at == session_entry.updated_at
            or getattr(session_entry, "was_auto_reset", False)
        )
        hooks = getattr(self, "hooks", None)
        if _is_new_session and hooks is not None:
            await hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })
        
        # Build session context
        context = build_session_context(source, self.config, session_entry)
        
        # Set environment variables for tools
        self._set_session_env(context)
        
        # Read privacy.redact_pii from config (re-read per message)
        _redact_pii = False
        try:
            import yaml as _pii_yaml
            with open(_config_path, encoding="utf-8") as _pf:
                _pcfg = _pii_yaml.safe_load(_pf) or {}
            _redact_pii = bool((_pcfg.get("privacy") or {}).get("redact_pii", False))
        except Exception:
            pass

        # Build the context prompt to inject
        context_prompt = build_session_context_prompt(context, redact_pii=_redact_pii)
        
        # If the previous session expired and was auto-reset, prepend a notice
        # so the agent knows this is a fresh conversation (not an intentional /reset).
        if getattr(session_entry, 'was_auto_reset', False):
            context_prompt = (
                "[System note: The user's previous session expired due to inactivity. "
                "This is a fresh conversation with no prior context.]\n\n"
                + context_prompt
            )
            session_entry.was_auto_reset = False
        
        # Load conversation history from transcript
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # -----------------------------------------------------------------
        # Session hygiene: auto-compress pathologically large transcripts
        #
        # Long-lived gateway sessions can accumulate enough history that
        # every new message rehydrates an oversized transcript, causing
        # repeated truncation/context failures.  Detect this early and
        # compress proactively — before the agent even starts.  (#628)
        #
        # Token source priority:
        # 1. Actual API-reported prompt_tokens from the last turn
        #    (stored in session_entry.last_prompt_tokens)
        # 2. Rough char-based estimate (str(msg)//4) with a 1.4x
        #    safety factor to account for overestimation on tool-heavy
        #    conversations (code/JSON tokenizes at 5-7+ chars/token).
        # -----------------------------------------------------------------
        if history and len(history) >= 4:
            from agent.model_metadata import (
                estimate_messages_tokens_rough,
                get_model_context_length,
            )

            # Read model + compression config from config.yaml.
            # NOTE: hygiene threshold is intentionally HIGHER than the agent's
            # own compressor (0.85 vs 0.50).  Hygiene is a safety net for
            # sessions that grew too large between turns — it fires pre-agent
            # to prevent API failures.  The agent's own compressor handles
            # normal context management during its tool loop with accurate
            # real token counts.  Having hygiene at 0.50 caused premature
            # compression on every turn in long gateway sessions.
            _hyg_model = "anthropic/claude-sonnet-4.6"
            _hyg_threshold_pct = 0.85
            _hyg_compression_enabled = True
            try:
                _hyg_cfg_path = _hermes_home / "config.yaml"
                if _hyg_cfg_path.exists():
                    import yaml as _hyg_yaml
                    with open(_hyg_cfg_path, encoding="utf-8") as _hyg_f:
                        _hyg_data = _hyg_yaml.safe_load(_hyg_f) or {}

                    # Resolve model name (same logic as run_sync)
                    _model_cfg = _hyg_data.get("model", {})
                    if isinstance(_model_cfg, str):
                        _hyg_model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        _hyg_model = _model_cfg.get("default", _hyg_model)

                    # Read compression settings — only use enabled flag.
                    # The threshold is intentionally separate from the agent's
                    # compression.threshold (hygiene runs higher).
                    _comp_cfg = _hyg_data.get("compression", {})
                    if isinstance(_comp_cfg, dict):
                        _hyg_compression_enabled = str(
                            _comp_cfg.get("enabled", True)
                        ).lower() in ("true", "1", "yes")
            except Exception:
                pass

            if _hyg_compression_enabled:
                _hyg_context_length = get_model_context_length(_hyg_model)
                _compress_token_threshold = int(
                    _hyg_context_length * _hyg_threshold_pct
                )
                _warn_token_threshold = int(_hyg_context_length * 0.95)

                _msg_count = len(history)

                # Prefer actual API-reported tokens from the last turn
                # (stored in session entry) over the rough char-based estimate.
                # The rough estimate (str(msg)//4) overestimates by 30-50% on
                # tool-heavy/code-heavy conversations, causing premature compression.
                _stored_tokens = session_entry.last_prompt_tokens
                if _stored_tokens > 0:
                    _approx_tokens = _stored_tokens
                    _token_source = "actual"
                else:
                    _approx_tokens = estimate_messages_tokens_rough(history)
                    # Apply safety factor only for rough estimates
                    _compress_token_threshold = int(
                        _compress_token_threshold * 1.4
                    )
                    _warn_token_threshold = int(_warn_token_threshold * 1.4)
                    _token_source = "estimated"

                _needs_compress = _approx_tokens >= _compress_token_threshold

                if _needs_compress:
                    logger.info(
                        "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                        "(threshold: %s%% of %s = %s tokens)",
                        _msg_count, f"{_approx_tokens:,}", _token_source,
                        int(_hyg_threshold_pct * 100),
                        f"{_hyg_context_length:,}",
                        f"{_compress_token_threshold:,}",
                    )

                    _hyg_adapter = self.adapters.get(source.platform)
                    _hyg_meta = {"thread_id": source.thread_id} if source.thread_id else None
                    if _hyg_adapter:
                        try:
                            await _hyg_adapter.send(
                                source.chat_id,
                                f"🗜️ Session is large ({_msg_count} messages, "
                                f"~{_approx_tokens:,} tokens). Auto-compressing...",
                                metadata=_hyg_meta,
                            )
                        except Exception:
                            pass

                    try:
                        from run_agent import AIAgent

                        _hyg_runtime = _resolve_runtime_agent_kwargs()
                        if _hyg_runtime.get("api_key"):
                            _hyg_msgs = [
                                {"role": m.get("role"), "content": m.get("content")}
                                for m in history
                                if m.get("role") in ("user", "assistant")
                                and m.get("content")
                            ]

                            if len(_hyg_msgs) >= 4:
                                _hyg_agent = AIAgent(
                                    **_hyg_runtime,
                                    model=_hyg_model,
                                    max_iterations=4,
                                    quiet_mode=True,
                                    enabled_toolsets=["memory"],
                                    session_id=session_entry.session_id,
                                )

                                loop = asyncio.get_event_loop()
                                _compressed, _ = await loop.run_in_executor(
                                    None,
                                    lambda: _hyg_agent._compress_context(
                                        _hyg_msgs, "",
                                        approx_tokens=_approx_tokens,
                                    ),
                                )

                                self.session_store.rewrite_transcript(
                                    session_entry.session_id, _compressed
                                )
                                # Reset stored token count — transcript was rewritten
                                session_entry.last_prompt_tokens = 0
                                history = _compressed
                                _new_count = len(_compressed)
                                _new_tokens = estimate_messages_tokens_rough(
                                    _compressed
                                )

                                logger.info(
                                    "Session hygiene: compressed %s → %s msgs, "
                                    "~%s → ~%s tokens",
                                    _msg_count, _new_count,
                                    f"{_approx_tokens:,}", f"{_new_tokens:,}",
                                )

                                if _hyg_adapter:
                                    try:
                                        await _hyg_adapter.send(
                                            source.chat_id,
                                            f"🗜️ Compressed: {_msg_count} → "
                                            f"{_new_count} messages, "
                                            f"~{_approx_tokens:,} → "
                                            f"~{_new_tokens:,} tokens",
                                            metadata=_hyg_meta,
                                        )
                                    except Exception:
                                        pass

                                # Still too large after compression — warn user
                                if _new_tokens >= _warn_token_threshold:
                                    logger.warning(
                                        "Session hygiene: still ~%s tokens after "
                                        "compression — suggesting /reset",
                                        f"{_new_tokens:,}",
                                    )
                                    if _hyg_adapter:
                                        try:
                                            await _hyg_adapter.send(
                                                source.chat_id,
                                                "⚠️ Session is still very large "
                                                "after compression "
                                                f"(~{_new_tokens:,} tokens). "
                                                "Consider using /reset to start "
                                                "fresh if you experience issues.",
                                                metadata=_hyg_meta,
                                            )
                                        except Exception:
                                            pass

                    except Exception as e:
                        logger.warning(
                            "Session hygiene auto-compress failed: %s", e
                        )
                        # Compression failed and session is dangerously large
                        if _approx_tokens >= _warn_token_threshold:
                            _hyg_adapter = self.adapters.get(source.platform)
                            _hyg_meta = {"thread_id": source.thread_id} if source.thread_id else None
                            if _hyg_adapter:
                                try:
                                    await _hyg_adapter.send(
                                        source.chat_id,
                                        f"⚠️ Session is very large "
                                        f"({_msg_count} messages, "
                                        f"~{_approx_tokens:,} tokens) and "
                                        "auto-compression failed. Consider "
                                        "using /compress or /reset to avoid "
                                        "issues.",
                                        metadata=_hyg_meta,
                                    )
                                except Exception:
                                    pass

        # First-message onboarding -- only on the very first interaction ever
        if not history and not self.session_store.has_any_sessions():
            context_prompt += (
                "\n\n[System note: This is the user's very first message ever. "
                "Briefly introduce yourself and mention that /help shows available commands. "
                "Keep the introduction concise -- one or two sentences max.]"
            )
        
        # One-time prompt if no home channel is set for this platform
        if not history and source.platform and source.platform != Platform.LOCAL:
            platform_name = source.platform.value
            env_key = f"{platform_name.upper()}_HOME_CHANNEL"
            if not os.getenv(env_key):
                adapter = self.adapters.get(source.platform)
                if adapter:
                    await adapter.send(
                        source.chat_id,
                        f"📬 No home channel is set for {platform_name.title()}. "
                        f"A home channel is where Hermes delivers cron job results "
                        f"and cross-platform messages.\n\n"
                        f"Type /sethome to make this chat your home channel, "
                        f"or ignore to skip."
                    )
        
        # -----------------------------------------------------------------
        # Voice channel awareness — inject current voice channel state
        # into context so the agent knows who is in the channel and who
        # is speaking, without needing a separate tool call.
        # -----------------------------------------------------------------
        if source.platform == Platform.DISCORD:
            adapter = self.adapters.get(Platform.DISCORD)
            guild_id = self._get_guild_id(event)
            if guild_id and adapter and hasattr(adapter, "get_voice_channel_context"):
                vc_context = adapter.get_voice_channel_context(guild_id)
                if vc_context:
                    context_prompt += f"\n\n{vc_context}"

        # -----------------------------------------------------------------
        # Auto-analyze images sent by the user
        #
        # If the user attached image(s), we run the vision tool eagerly so
        # the conversation model always receives a text description.  The
        # local file path is also included so the model can re-examine the
        # image later with a more targeted question via vision_analyze.
        #
        # We filter to image paths only (by media_type) so that non-image
        # attachments (documents, audio, etc.) are not sent to the vision
        # tool even when they appear in the same message.
        # -----------------------------------------------------------------
        message_text = event.text or ""
        if event.media_urls:
            image_paths = []
            for i, path in enumerate(event.media_urls):
                # Check media_types if available; otherwise infer from message type
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                is_image = (
                    mtype.startswith("image/")
                    or event.message_type == MessageType.PHOTO
                )
                if is_image:
                    image_paths.append(path)
            if image_paths:
                message_text = await self._enrich_message_with_vision(
                    message_text, image_paths
                )
        
        # -----------------------------------------------------------------
        # Auto-transcribe voice/audio messages sent by the user
        # -----------------------------------------------------------------
        if event.media_urls:
            audio_paths = []
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                is_audio = (
                    mtype.startswith("audio/")
                    or event.message_type in (MessageType.VOICE, MessageType.AUDIO)
                )
                if is_audio:
                    audio_paths.append(path)
            if audio_paths:
                message_text = await self._enrich_message_with_transcription(
                    message_text, audio_paths
                )
                # If STT failed, send a direct message to the user so they
                # know voice isn't configured — don't rely on the agent to
                # relay the error clearly.
                _stt_fail_markers = (
                    "No STT provider",
                    "STT is disabled",
                    "can't listen",
                    "VOICE_TOOLS_OPENAI_KEY",
                )
                if any(m in message_text for m in _stt_fail_markers):
                    _stt_adapter = self.adapters.get(source.platform)
                    _stt_meta = {"thread_id": source.thread_id} if source.thread_id else None
                    if _stt_adapter:
                        try:
                            _stt_msg = (
                                "🎤 I received your voice message but can't transcribe it — "
                                "no speech-to-text provider is configured.\n\n"
                                "To enable voice: install faster-whisper "
                                "(`pip install faster-whisper` in the Hermes venv) "
                                "and set `stt.enabled: true` in config.yaml, "
                                "then /restart the gateway."
                            )
                            # Point to setup skill if it's installed
                            if self._has_setup_skill():
                                _stt_msg += "\n\nFor full setup instructions, type: `/skill hermes-agent-setup`"
                            await _stt_adapter.send(
                                source.chat_id, _stt_msg,
                                metadata=_stt_meta,
                            )
                        except Exception:
                            pass

        # -----------------------------------------------------------------
        # Enrich document messages with context notes for the agent
        # -----------------------------------------------------------------
        if event.media_urls and event.message_type == MessageType.DOCUMENT:
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                if not (mtype.startswith("application/") or mtype.startswith("text/")):
                    continue
                # Extract display filename by stripping the doc_{uuid12}_ prefix
                import os as _os
                basename = _os.path.basename(path)
                # Format: doc_<12hex>_<original_filename>
                parts = basename.split("_", 2)
                display_name = parts[2] if len(parts) >= 3 else basename
                # Sanitize to prevent prompt injection via filenames
                import re as _re
                display_name = _re.sub(r'[^\w.\- ]', '_', display_name)

                if mtype.startswith("text/"):
                    context_note = (
                        f"[The user sent a text document: '{display_name}'. "
                        f"Its content has been included below. "
                        f"The file is also saved at: {path}]"
                    )
                else:
                    context_note = (
                        f"[The user sent a document: '{display_name}'. "
                        f"The file is saved at: {path}. "
                        f"Ask the user what they'd like you to do with it.]"
                    )
                message_text = f"{context_note}\n\n{message_text}"

        # -----------------------------------------------------------------
        # Inject reply context when user replies to a message not in history.
        # Telegram (and other platforms) let users reply to specific messages,
        # but if the quoted message is from a previous session, cron delivery,
        # or background task, the agent has no context about what's being
        # referenced. Prepend the quoted text so the agent understands. (#1594)
        # -----------------------------------------------------------------
        if getattr(event, 'reply_to_text', None) and event.reply_to_message_id:
            reply_snippet = event.reply_to_text[:500]
            found_in_history = any(
                reply_snippet[:200] in (msg.get("content") or "")
                for msg in history
                if msg.get("role") in ("assistant", "user", "tool")
            )
            if not found_in_history:
                message_text = f'[Replying to: "{reply_snippet}"]\n\n{message_text}'

        try:
            # Emit agent:start hook
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)
            
            # Run the agent
            agent_result = await self._run_agent(
                message=message_text,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_entry.session_id,
                session_key=session_key
            )
            
            response = agent_result.get("final_response") or ""
            agent_messages = agent_result.get("messages", [])

            # Surface error details when the agent failed silently (final_response=None)
            if not response and agent_result.get("failed"):
                error_detail = agent_result.get("error", "unknown error")
                error_str = str(error_detail).lower()

                # Detect context-overflow failures and give specific guidance.
                # Generic 400 "Error" from Anthropic with large sessions is the
                # most common cause of this (#1630).
                _is_ctx_fail = any(p in error_str for p in (
                    "context", "token", "too large", "too long",
                    "exceed", "payload",
                )) or (
                    "400" in error_str
                    and len(history) > 50
                )

                if _is_ctx_fail:
                    response = (
                        "⚠️ Session too large for the model's context window.\n"
                        "Use /compact to compress the conversation, or "
                        "/reset to start fresh."
                    )
                else:
                    response = (
                        f"The request failed: {str(error_detail)[:300]}\n"
                        "Try again or use /reset to start a fresh session."
                    )

            # If the agent's session_id changed during compression, update
            # session_entry so transcript writes below go to the right session.
            if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
                session_entry.session_id = agent_result["session_id"]

            # Prepend reasoning/thinking if display is enabled
            if getattr(self, "_show_reasoning", False) and response:
                last_reasoning = agent_result.get("last_reasoning")
                if last_reasoning:
                    # Collapse long reasoning to keep messages readable
                    lines = last_reasoning.strip().splitlines()
                    if len(lines) > 15:
                        display_reasoning = "\n".join(lines[:15])
                        display_reasoning += f"\n_... ({len(lines) - 15} more lines)_"
                    else:
                        display_reasoning = last_reasoning.strip()
                    response = f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

            # Emit agent:end hook
            await self.hooks.emit("agent:end", {
                **hook_ctx,
                "response": (response or "")[:500],
            })
            
            # Check for pending process watchers (check_interval on background processes)
            try:
                from tools.process_registry import process_registry
                while process_registry.pending_watchers:
                    watcher = process_registry.pending_watchers.pop(0)
                    asyncio.create_task(self._run_process_watcher(watcher))
            except Exception as e:
                logger.error("Process watcher setup error: %s", e)

            # Check if the agent encountered a dangerous command needing approval
            try:
                from tools.approval import pop_pending
                import time as _time
                pending = pop_pending(session_key)
                if pending:
                    approval_id = str(pending.get("approval_id") or self._new_approval_id())
                    pending["approval_id"] = approval_id
                    pending["session_key"] = session_key
                    pending.setdefault("created_at", datetime.now().isoformat())
                    self._pending_approvals[session_key] = pending
                    approval_note = (
                        f"\n\nApproval ID: `{approval_id}`\n"
                        f"Use {self._approval_usage_text()} or the Discord approval buttons."
                    )
                    response = f"{response}{approval_note}" if response else approval_note.strip()

                    if source.platform == Platform.DISCORD:
                        approval_adapter = self.adapters.get(Platform.DISCORD)
                        if approval_adapter and hasattr(approval_adapter, "send_exec_approval"):
                            try:
                                await approval_adapter.send_exec_approval(
                                    source.chat_id,
                                    pending.get("command", ""),
                                    approval_id,
                                )
                            except Exception as approval_error:
                                logger.debug("Discord exec approval UI failed: %s", approval_error)
            except Exception as e:
                logger.debug("Failed to check pending approvals: %s", e)
            
            # Save the full conversation to the transcript, including tool calls.
            # This preserves the complete agent loop (tool_calls, tool results,
            # intermediate reasoning) so sessions can be resumed with full context
            # and transcripts are useful for debugging and training data.
            #
            # IMPORTANT: When the agent failed before producing any response
            # (e.g. context-overflow 400), do NOT persist the user's message.
            # Persisting it would make the session even larger, causing the
            # same failure on the next attempt — an infinite loop. (#1630)
            agent_failed_early = (
                agent_result.get("failed")
                and not agent_result.get("final_response")
            )
            if agent_failed_early:
                logger.info(
                    "Skipping transcript persistence for failed request in "
                    "session %s to prevent session growth loop.",
                    session_entry.session_id,
                )

            ts = datetime.now().isoformat()
            
            # If this is a fresh session (no history), write the full tool
            # definitions as the first entry so the transcript is self-describing
            # -- the same list of dicts sent as tools=[...] in the API request.
            if agent_failed_early:
                pass  # Skip all transcript writes — don't grow a broken session
            elif not history:
                tool_defs = agent_result.get("tools", [])
                self.session_store.append_to_transcript(
                    session_entry.session_id,
                    {
                        "role": "session_meta",
                        "tools": tool_defs or [],
                        "model": os.getenv("HERMES_MODEL", ""),
                        "platform": source.platform.value if source.platform else "",
                        "timestamp": ts,
                    }
                )
            
            # Find only the NEW messages from this turn (skip history we loaded).
            # Use the filtered history length (history_offset) that was actually
            # passed to the agent, not len(history) which includes session_meta
            # entries that were stripped before the agent saw them.
            if not agent_failed_early:
                history_len = agent_result.get("history_offset", len(history))
                new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []
                
                # If no new messages found (edge case), fall back to simple user/assistant
                if not new_messages:
                    self.session_store.append_to_transcript(
                        session_entry.session_id,
                        {"role": "user", "content": message_text, "timestamp": ts}
                    )
                    if response:
                        self.session_store.append_to_transcript(
                            session_entry.session_id,
                            {"role": "assistant", "content": response, "timestamp": ts}
                        )
                else:
                    # The agent already persisted these messages to SQLite via
                    # _flush_messages_to_session_db(), so skip the DB write here
                    # to prevent the duplicate-write bug (#860).  We still write
                    # to JSONL for backward compatibility and as a backup.
                    agent_persisted = self._session_db is not None
                    for msg in new_messages:
                        # Skip system messages (they're rebuilt each run)
                        if msg.get("role") == "system":
                            continue
                        # Add timestamp to each message for debugging
                        entry = {**msg, "timestamp": ts}
                        self.session_store.append_to_transcript(
                            session_entry.session_id, entry,
                            skip_db=agent_persisted,
                        )
            
            # Update session with actual prompt token count and model from the agent
            self.session_store.update_session(
                session_entry.session_key,
                input_tokens=agent_result.get("input_tokens", 0),
                output_tokens=agent_result.get("output_tokens", 0),
                cache_read_tokens=agent_result.get("cache_read_tokens", 0),
                cache_write_tokens=agent_result.get("cache_write_tokens", 0),
                last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
                model=agent_result.get("model"),
                estimated_cost_usd=agent_result.get("estimated_cost_usd"),
                cost_status=agent_result.get("cost_status"),
                cost_source=agent_result.get("cost_source"),
                provider=agent_result.get("provider"),
                base_url=agent_result.get("base_url"),
            )

            dock_target = self._dock_target_for_session(session_key)

            # Auto voice reply: send TTS audio before the text response.
            # Skip when replies for this session are docked elsewhere.
            if dock_target is None and self._should_send_voice_reply(event, response, agent_messages):
                await self._send_voice_reply(event, response)

            # If streaming already delivered the response, return None so
            # _process_message_background doesn't send it again.
            if agent_result.get("already_sent"):
                return None

            if response:
                response = await self._deliver_docked_response(
                    session_key=session_key,
                    response=response,
                    source=source,
                )

            return response
            
        except Exception as e:
            logger.exception("Agent error in session %s", session_key)
            error_type = type(e).__name__
            error_detail = str(e)[:300] if str(e) else "no details available"
            status_hint = ""
            status_code = getattr(e, "status_code", None)
            _hist_len = len(history) if 'history' in locals() else 0
            if status_code == 401:
                status_hint = " Check your API key or run `claude /login` to refresh OAuth credentials."
            elif status_code == 429:
                # Check if this is a plan usage limit (resets on a schedule) vs a transient rate limit
                _err_body = getattr(e, "response", None)
                _err_json = {}
                try:
                    if _err_body is not None:
                        _err_json = _err_body.json().get("error", {})
                except Exception:
                    pass
                if _err_json.get("type") == "usage_limit_reached":
                    _resets_in = _err_json.get("resets_in_seconds")
                    if _resets_in and _resets_in > 0:
                        import math
                        _hours = math.ceil(_resets_in / 3600)
                        status_hint = f" Your plan's usage limit has been reached. It resets in ~{_hours}h."
                    else:
                        status_hint = " Your plan's usage limit has been reached. Please wait until it resets."
                else:
                    status_hint = " You are being rate-limited. Please wait a moment and try again."
            elif status_code == 529:
                status_hint = " The API is temporarily overloaded. Please try again shortly."
            elif status_code in (400, 500):
                # 400 with a large session is context overflow.
                # 500 with a large session often means the payload is too large
                # for the API to process — treat it the same way.
                if _hist_len > 50:
                    return (
                        "⚠️ Session too large for the model's context window.\n"
                        "Use /compact to compress the conversation, or "
                        "/reset to start fresh."
                    )
                elif status_code == 400:
                    status_hint = " The request was rejected by the API."
            return (
                f"Sorry, I encountered an error ({error_type}).\n"
                f"{error_detail}\n"
                f"{status_hint}"
                "Try again or use /reset to start a fresh session."
            )
        finally:
            # Clear session env
            self._clear_session_env()
    
    async def _handle_reset_command(self, event: MessageEvent) -> str:
        """Handle /new or /reset command."""
        source = event.source
        
        # Get existing session key
        session_key = self._session_key_for_source(source)
        
        # Flush memories in the background (fire-and-forget) so the user
        # gets the "Session reset!" response immediately.
        try:
            old_entry = self.session_store._entries.get(session_key)
            if old_entry:
                asyncio.create_task(
                    self._async_flush_memories(old_entry.session_id, session_key)
                )
        except Exception as e:
            logger.debug("Gateway memory flush on reset failed: %s", e)

        self._shutdown_gateway_honcho(session_key)
        
        # Reset the session
        new_entry = self.session_store.reset_session(session_key)

        # Emit session:end hook (session is ending)
        await self.hooks.emit("session:end", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })

        # Emit session:reset hook
        await self.hooks.emit("session:reset", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })
        
        if new_entry:
            return "✨ Session reset! I've started fresh with no memory of our previous conversation."
        else:
            # No existing session, just create one
            self.session_store.get_or_create_session(source, force_new=True)
            return "✨ New session started!"
    
    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        if self._session_source_for_event(event).platform == Platform.DISCORD:
            from gateway.platforms.discord_impl import runtime_views

            snapshot = runtime_views.collect_discord_status_snapshot(self, event)
            return runtime_views.render_discord_status(snapshot)

        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        
        connected_platforms = [p.value for p in self.adapters.keys()]
        
        # Check if there's an active agent
        session_key = session_entry.session_key
        is_running = session_key in self._running_agents
        
        lines = [
            "📊 **Hermes Gateway Status**",
            "",
            f"**Session ID:** `{session_entry.session_id[:12]}...`",
            f"**Created:** {session_entry.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Last Activity:** {session_entry.updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Tokens:** {session_entry.total_tokens:,}",
            f"**Agent Running:** {'Yes ⚡' if is_running else 'No'}",
            "",
            f"**Connected Platforms:** {', '.join(connected_platforms)}",
        ]
        
        return "\n".join(lines)

    async def _handle_approve_command(self, event: MessageEvent) -> str:
        """Handle /approve command for pending exec approvals."""
        raw_args = event.get_command_args().strip()
        target_source = self._command_target_source_for_event(event)
        if not raw_args:
            return self._resolve_pending_approval(decision="allow-once", source=target_source)

        tokens = raw_args.split()
        approval_id: Optional[str] = None
        decision: Optional[str] = None

        if len(tokens) == 1:
            maybe_decision = self._normalize_approval_decision(tokens[0])
            if maybe_decision:
                decision = maybe_decision
            else:
                approval_id = tokens[0]
        else:
            first_decision = self._normalize_approval_decision(tokens[0])
            second_decision = self._normalize_approval_decision(tokens[1])
            if first_decision and not second_decision:
                decision = first_decision
                approval_id = tokens[1]
            else:
                approval_id = tokens[0]
                decision = second_decision or self._normalize_approval_decision(tokens[1])

        if approval_id and decision is None:
            return self._resolve_pending_approval(decision="show", approval_id=approval_id)
        if decision is None:
            return f"Use {self._approval_usage_text()}"

        return self._resolve_pending_approval(
            decision=decision,
            approval_id=approval_id,
            source=target_source,
        )

    async def _handle_allowlist_command(self, event: MessageEvent) -> str:
        """Handle /allowlist command - inspect or edit command approval patterns."""
        import tools.approval as approval_mod

        raw_args = event.get_command_args().strip()
        parts = self._split_command_args(raw_args, max_parts=2)
        mode = (parts[0] if parts else "list").strip().lower()
        entry = parts[1].strip() if len(parts) > 1 else ""

        if mode not in {"list", "add", "remove"}:
            return (
                f"⚠️ Unknown allowlist action: `{mode}`\n\n"
                "Use `/allowlist [list|add|remove] [entry]`"
            )

        patterns = set(approval_mod.load_permanent_allowlist())
        if mode == "list":
            if not patterns:
                return "No permanently approved command patterns are stored."
            lines = ["🛡️ **Command Allowlist**", ""]
            for pattern in sorted(patterns):
                lines.append(f"• `{pattern}`")
            return "\n".join(lines)

        if not entry:
            return f"Usage: `/allowlist {mode} <entry>`"

        if mode == "add":
            patterns.add(entry)
            approval_mod.load_permanent(patterns)
            approval_mod.save_permanent_allowlist(patterns)
            return f"✅ Added `{entry}` to the command allowlist."

        if entry not in patterns:
            return f"`{entry}` is not in the command allowlist."
        patterns.remove(entry)
        approval_mod._permanent_approved = set(patterns)
        approval_mod.save_permanent_allowlist(patterns)
        return f"🗑️ Removed `{entry}` from the command allowlist."

    async def _handle_config_command(self, event: MessageEvent) -> str:
        """Handle /config command - inspect or update on-disk Hermes config."""
        raw_args = event.get_command_args().strip()
        parts = self._split_command_args(raw_args, max_parts=3)
        mode = (parts[0] if parts else "show").strip().lower()
        key = parts[1].strip() if len(parts) > 1 else ""
        value = parts[2].strip() if len(parts) > 2 else ""

        if mode == "show":
            if key:
                try:
                    kind, resolved = read_config_or_env_value(key)
                except KeyError:
                    return f"`{key}` is not set."
                return f"📄 **Config Show** (`{kind}`)\n\n`{key}`\n{format_yaml_block(resolved)}"
            raw_config = load_raw_user_config()
            return (
                "📄 **Hermes Config (raw user file)**\n\n"
                f"{format_yaml_block(raw_config or {})}"
            )

        if mode == "get":
            if not key:
                return "Usage: `/config get <key>`"
            try:
                kind, resolved = read_config_or_env_value(key)
            except KeyError:
                return f"`{key}` is not set."
            return f"📄 **Config Value** (`{kind}`)\n\n`{key}`\n{format_yaml_block(resolved)}"

        if mode == "set":
            if not key or not value:
                return "Usage: `/config set <key> <value>`"
            kind, stored_value, path = write_config_or_env_value(key, value)
            return (
                f"✅ Updated `{key}` in `{path}` (`{kind}`)\n\n"
                f"{format_yaml_block(stored_value)}"
            )

        if mode == "unset":
            if not key:
                return "Usage: `/config unset <key>`"
            kind, removed, path = unset_config_or_env_value(key)
            if not removed:
                return f"`{key}` is not set."
            return f"🗑️ Removed `{key}` from `{path}` (`{kind}`)."

        return (
            f"⚠️ Unknown config action: `{mode}`\n\n"
            "Use `/config [show|get|set|unset] [key] [value]`"
        )

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug command - live runtime overrides for the current gateway process."""
        args = event.get_command_args().strip()
        try:
            parts = shlex.split(args) if args else []
        except ValueError as exc:
            return f"⚠️ Invalid debug arguments: {exc}"

        if not parts or parts[0] == "show":
            if len(parts) > 2:
                return "Usage: `/debug show [key]`"
            if len(parts) == 2:
                key = parts[1].strip()
                if key not in self._DEBUG_SUPPORTED_KEYS:
                    supported = "\n".join(f"- `{name}`" for name in sorted(self._DEBUG_SUPPORTED_KEYS))
                    return (
                        f"⚠️ Unsupported debug key: `{key}`\n\n"
                        f"**Supported keys**\n{supported}"
                    )
                overrides = self._get_runtime_debug_overrides()
                value = self._get_effective_runtime_debug_value(key)
                override_state = "yes" if key in overrides else "no"
                return (
                    "🛠️ **Runtime Debug Key**\n\n"
                    f"**Key:** `{key}`\n"
                    f"**Override active:** {override_state}\n\n"
                    f"{self._format_runtime_debug_value(value)}"
                )

            overrides = self._get_runtime_debug_overrides()
            active = {
                key: self._get_effective_runtime_debug_value(key)
                for key in sorted(overrides)
            }
            supported = "\n".join(
                f"- `{name}` — {description}"
                for name, description in self._DEBUG_SUPPORTED_KEYS.items()
            )
            active_block = "None" if not active else self._format_runtime_debug_value(active)
            return (
                "🛠️ **Runtime Debug Overrides**\n\n"
                "Applies only to the running Hermes gateway process and the current Discord connection.\n"
                "Config files are unchanged.\n\n"
                "**Active overrides**\n"
                f"{active_block}\n\n"
                "**Usage**\n"
                "`/debug show [key]`\n"
                "`/debug set <key> <value>`\n"
                "`/debug unset <key>`\n"
                "`/debug reset`\n\n"
                f"**Supported keys**\n{supported}"
            )

        action = parts[0].lower()
        if action == "reset":
            if len(parts) != 1:
                return "Usage: `/debug reset`"
            cleared = len(self._get_runtime_debug_overrides())
            self._get_runtime_debug_overrides().clear()
            self._apply_runtime_debug_overrides()
            return (
                "🛠️ ✓ Runtime debug overrides cleared.\n"
                f"Removed `{cleared}` override(s) from the running gateway process."
            )

        if action == "unset":
            if len(parts) != 2:
                return "Usage: `/debug unset <key>`"
            key = parts[1].strip()
            if key not in self._DEBUG_SUPPORTED_KEYS:
                return f"⚠️ Unsupported debug key: `{key}`"
            removed = self._get_runtime_debug_overrides().pop(key, None)
            self._apply_runtime_debug_overrides()
            if removed is None:
                return f"`{key}` did not have a runtime override."
            return (
                f"🛠️ ✓ Removed runtime override for `{key}`.\n\n"
                f"{self._format_runtime_debug_value(self._get_effective_runtime_debug_value(key))}"
            )

        if action == "set":
            if len(parts) < 3:
                return "Usage: `/debug set <key> <value>`"
            key = parts[1].strip()
            if key not in self._DEBUG_SUPPORTED_KEYS:
                return f"⚠️ Unsupported debug key: `{key}`"
            value_text = args.split(None, 2)[2]
            try:
                parsed = self._parse_debug_value(key, value_text)
            except ValueError as exc:
                return f"⚠️ Invalid value for `{key}`: {exc}"
            self._get_runtime_debug_overrides()[key] = parsed
            self._apply_runtime_debug_overrides()
            return (
                f"🛠️ ✓ Runtime override set for `{key}`.\n"
                "Applies immediately to the running gateway process only.\n\n"
                f"{self._format_runtime_debug_value(self._get_effective_runtime_debug_value(key))}"
            )

        return (
            f"⚠️ Unknown debug action: `{action}`\n\n"
            "Use `/debug show [key]`, `/debug set <key> <value>`, `/debug unset <key>`, or `/debug reset`."
        )

    def _build_manual_subagent_agent(
        self,
        *,
        source: SessionSource,
        session_key: str,
        entry: dict[str, Any],
    ):
        from run_agent import AIAgent
        from tools.delegate_tool import DEFAULT_TOOLSETS

        model = _resolve_gateway_model()
        runtime_kwargs = _resolve_runtime_agent_kwargs()
        self._apply_runtime_debug_overrides()
        turn_route = self._resolve_turn_agent_config(entry.get("goal", ""), model, runtime_kwargs)
        platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value
        agent = AIAgent(
            model=turn_route["model"],
            **turn_route["runtime"],
            max_iterations=int(os.getenv("HERMES_MAX_ITERATIONS", "90")),
            quiet_mode=True,
            verbose_logging=False,
            enabled_toolsets=DEFAULT_TOOLSETS,
            reasoning_config=self._get_effective_reasoning_config(),
            skip_context_files=True,
            skip_memory=True,
            session_id=f"{session_key}:{entry['id']}",
            platform=platform_key,
        )
        agent._gateway_subagent_id = entry["id"]
        agent._gateway_subagent_task_index = entry["ordinal"]
        agent.tool_progress_callback = self._build_subagent_progress_callback(session_key, agent)
        self._attach_subagent_runtime_hooks(agent, session_key)
        entry["child"] = agent
        entry["toolsets"] = list(DEFAULT_TOOLSETS)
        entry["model"] = turn_route["model"]
        return agent

    async def _notify_manual_subagent_completion(
        self,
        *,
        source: SessionSource,
        entry: dict[str, Any],
    ) -> None:
        adapter = self.adapters.get(source.platform)
        if not adapter:
            return
        metadata = {"thread_id": source.thread_id} if source.thread_id else None
        status = str(entry.get("status") or "")
        if status == "completed":
            content = (
                f"🤖 **Subagent Complete**\n\n"
                f"**ID:** `{entry['id']}`\n"
                f"**Goal:** {self._truncate_line(entry.get('goal') or '', limit=160)}\n\n"
                f"{entry.get('summary') or '_(no summary)_'}"
            )
        else:
            detail = entry.get("error") or entry.get("summary") or "Subagent stopped."
            content = (
                f"🤖 **Subagent {status.title() or 'Update'}**\n\n"
                f"**ID:** `{entry['id']}`\n"
                f"{self._truncate_line(detail, limit=500)}"
            )
        try:
            await adapter.send(source.chat_id, content, metadata=metadata)
        except Exception as e:
            logger.debug("Manual subagent completion send failed: %s", e)

    async def _start_manual_subagent(
        self,
        *,
        source: SessionSource,
        session_key: str,
        goal: str,
    ) -> dict[str, Any]:
        entry = self._create_subagent_entry(
            session_key=session_key,
            child=None,
            task_index=0,
            goal=goal,
            context=None,
            toolsets=[],
            model=None,
            source_kind="manual",
        )
        try:
            agent = self._build_manual_subagent_agent(source=source, session_key=session_key, entry=entry)
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            self._append_subagent_log(entry, f"Failed to start: {exc}")
            return entry

        self._start_delegate_child(session_key=session_key, child=agent, goal=goal)
        loop = asyncio.get_running_loop()

        async def _runner():
            current_goal = goal

            def _run_sync():
                nonlocal current_goal
                while True:
                    result = agent.run_conversation(
                        user_message=current_goal,
                        task_id=f"{session_key}:{entry['id']}",
                    )
                    pending_steer = str(entry.get("pending_steer") or "").strip()
                    if result.get("interrupted") and pending_steer:
                        entry["pending_steer"] = None
                        current_goal = pending_steer
                        self._start_delegate_child(
                            session_key=session_key,
                            child=agent,
                            goal=current_goal,
                            restarted=True,
                        )
                        continue
                    return result, current_goal

            result, final_goal = await loop.run_in_executor(None, _run_sync)
            summary = str(result.get("final_response") or "").strip()
            completed = bool(result.get("completed")) and bool(summary)
            status = "interrupted" if result.get("interrupted") else "completed" if completed else "failed"
            self._complete_delegate_child(
                session_key=session_key,
                child=agent,
                entry={
                    "status": status,
                    "summary": summary,
                    "error": result.get("error") or "",
                },
                goal=final_goal,
            )
            await self._notify_manual_subagent_completion(source=source, entry=entry)

        entry["async_task"] = asyncio.create_task(_runner())
        return entry

    async def _handle_subagents_command(self, event: MessageEvent) -> str:
        target_source = self._command_target_source_for_event(event)
        session_entry = self.session_store.get_or_create_session(target_source)
        session_key = session_entry.session_key

        raw_args = event.get_command_args().strip()
        parts = self._split_command_args(raw_args, max_parts=2)
        action = (parts[0].strip().lower() if parts else "list") or "list"
        payload = parts[1] if len(parts) > 1 else ""

        if action == "list":
            return self._format_subagent_list(session_key)

        if action == "info":
            targets, error = self._resolve_subagent_targets(session_key, payload, active_only=False)
            if error:
                return error
            return self._format_subagent_info(targets[0])

        if action == "log":
            targets, error = self._resolve_subagent_targets(session_key, payload, active_only=False)
            if error:
                return error
            return self._format_subagent_log(targets[0])

        if action == "spawn":
            goal = str(payload or "").strip()
            if not goal:
                return "Usage: `/subagents spawn <goal>`"
            entry = await self._start_manual_subagent(source=target_source, session_key=session_key, goal=goal)
            if entry.get("status") == "error":
                return f"❌ Failed to spawn subagent: {entry.get('error')}"
            return (
                f"🤖 Spawned subagent `#{entry['ordinal']}` / `{entry['id']}`.\n\n"
                f"**Goal:** {goal}\n"
                "Use `/subagents list`, `/subagents info <id>`, or `/subagents log <id>` to inspect it."
            )

        if action == "kill":
            target = payload.strip()
            return await self._handle_kill_command(
                MessageEvent(
                    text=f"/kill {target}".strip(),
                    message_type=event.message_type,
                    source=event.source,
                    raw_message=event.raw_message,
                    message_id=event.message_id,
                    media_urls=event.media_urls,
                    media_types=event.media_types,
                    reply_to_message_id=event.reply_to_message_id,
                    reply_to_text=event.reply_to_text,
                    timestamp=event.timestamp,
                    metadata=dict(event.metadata or {}),
                )
            )

        if action in {"steer", "send"}:
            if not payload.strip():
                return f"Usage: `/subagents {action} <id|#> <message>`"
            steer_parts = self._split_command_args(payload, max_parts=2)
            if len(steer_parts) < 2:
                return f"Usage: `/subagents {action} <id|#> <message>`"
            target, message = steer_parts
            steer_text = f"/steer {target} {message}"
            return await self._handle_steer_command(
                MessageEvent(
                    text=steer_text,
                    message_type=event.message_type,
                    source=event.source,
                    raw_message=event.raw_message,
                    message_id=event.message_id,
                    media_urls=event.media_urls,
                    media_types=event.media_types,
                    reply_to_message_id=event.reply_to_message_id,
                    reply_to_text=event.reply_to_text,
                    timestamp=event.timestamp,
                    metadata=dict(event.metadata or {}),
                )
            )

        return (
            "Usage: `/subagents list|info|log|spawn|kill|steer|send`\n"
            "Examples:\n"
            "`/subagents list`\n"
            "`/subagents spawn research the failing tests`\n"
            "`/subagents steer #1 focus on the Discord adapter only`"
        )

    async def _handle_kill_command(self, event: MessageEvent) -> str:
        target_source = self._command_target_source_for_event(event)
        session_entry = self.session_store.get_or_create_session(target_source)
        session_key = session_entry.session_key
        raw_target = event.get_command_args().strip()
        targets, error = self._resolve_subagent_targets(
            session_key,
            raw_target,
            active_only=True,
            allow_all=True,
        )
        if error:
            return error
        if not targets:
            return "No active subagents found for this session."
        for entry in targets:
            entry["pending_steer"] = None
            child = entry.get("child")
            if child is not None and hasattr(child, "interrupt"):
                try:
                    child.interrupt("Killed from /kill")
                except Exception as e:
                    logger.debug("Subagent kill failed for %s: %s", entry.get("id"), e)
            self._append_subagent_log(entry, "Kill requested.")
        if raw_target.strip().lower() == "all":
            return f"🛑 Requested stop for `{len(targets)}` active subagent(s)."
        return f"🛑 Requested stop for `{targets[0]['id']}`."

    async def _handle_steer_command(self, event: MessageEvent) -> str:
        target_source = self._command_target_source_for_event(event)
        session_entry = self.session_store.get_or_create_session(target_source)
        session_key = session_entry.session_key
        parts = self._split_command_args(event.get_command_args().strip(), max_parts=2)
        if len(parts) < 2:
            return "Usage: `/steer <id|#> <message>`"
        target, message = parts
        targets, error = self._resolve_subagent_targets(session_key, target, active_only=True)
        if error:
            return error
        entry = targets[0]
        child = entry.get("child")
        if child is None or not hasattr(child, "interrupt"):
            return f"Subagent `{entry['id']}` is not running."
        entry["pending_steer"] = message
        self._append_subagent_log(entry, f"Steer requested: {self._truncate_line(message, limit=160)}")
        child.interrupt(message)
        return (
            f"↪ Steering subagent `{entry['id']}`.\n\n"
            f"**Next goal:** {message}\n"
            "The current run will stop and restart on the steer message."
        )

    async def _run_acp_prompt(
        self,
        *,
        source: SessionSource,
        session_id: str,
        prompt: str,
    ) -> str:
        manager = self._get_acp_session_manager()
        state = manager.get_session(session_id)
        if state is None:
            return f"No ACP session found for `{session_id}`."
        tasks = self._get_acp_tasks()
        meta = self._get_acp_session_meta_entry(session_id)
        if session_id in tasks and not tasks[session_id].done():
            meta["pending_steer"] = prompt
            meta["updated_at"] = self._subagent_now()
            try:
                state.cancel_event.set()
                state.agent.interrupt(prompt)
            except Exception:
                logger.debug("ACP steer interrupt failed", exc_info=True)
            return f"↪ Steering ACP session `{session_id}`."

        loop = asyncio.get_running_loop()
        meta["last_prompt"] = prompt
        meta["status"] = "running"
        meta["updated_at"] = self._subagent_now()

        async def _runner():
            current_prompt = prompt
            while True:
                if state.cancel_event:
                    state.cancel_event.clear()

                def _sync_run():
                    return state.agent.run_conversation(
                        user_message=current_prompt,
                        conversation_history=state.history,
                        task_id=session_id,
                    )

                result = await loop.run_in_executor(None, _sync_run)
                state.history = result.get("messages") or state.history
                pending_steer = str(meta.pop("pending_steer", "") or "").strip()
                if result.get("interrupted") and pending_steer:
                    current_prompt = pending_steer
                    meta["last_prompt"] = current_prompt
                    meta["updated_at"] = self._subagent_now()
                    continue
                meta["last_result"] = result
                meta["status"] = "interrupted" if result.get("interrupted") else "completed" if result.get("completed") else "failed"
                meta["updated_at"] = self._subagent_now()
                adapter = self.adapters.get(source.platform)
                if adapter and result.get("final_response"):
                    metadata = {"thread_id": source.thread_id} if source.thread_id else None
                    try:
                        await adapter.send(
                            source.chat_id,
                            f"🧩 **ACP Session `{session_id}`**\n\n{result.get('final_response')}",
                            metadata=metadata,
                        )
                    except Exception as e:
                        logger.debug("ACP completion send failed: %s", e)
                break

        tasks[session_id] = asyncio.create_task(_runner())
        return f"🧩 ACP session `{session_id}` is running."

    async def _handle_acp_command(self, event: MessageEvent) -> str:
        target_source = self._command_target_source_for_event(event)
        session_entry = self.session_store.get_or_create_session(target_source)
        session_key = session_entry.session_key
        args = event.get_command_args().strip()
        parts = self._split_command_args(args, max_parts=2)
        action = (parts[0].strip().lower() if parts else "sessions") or "sessions"
        payload = parts[1] if len(parts) > 1 else ""
        manager = self._get_acp_session_manager()
        bindings = self._get_acp_bindings()

        if action == "sessions":
            return self._format_acp_sessions(session_key)

        if action == "spawn":
            cwd = str(payload or ".").strip() or "."
            state = manager.create_session(cwd=cwd)
            bindings[session_key] = state.session_id
            meta = self._get_acp_session_meta_entry(state.session_id)
            meta["source"] = target_source
            meta["updated_at"] = self._subagent_now()
            return (
                f"🧩 Created ACP session `{state.session_id}`.\n\n"
                f"**CWD:** `{cwd}`\n"
                "This session is now the default ACP target for this chat."
            )

        if action == "doctor":
            try:
                import acp as _acp  # noqa: F401
                available = "yes"
            except Exception:
                available = "no"
            bound = bindings.get(session_key) or "none"
            running = sum(1 for task in self._get_acp_tasks().values() if not task.done())
            return (
                "🧩 **ACP Doctor**\n\n"
                f"**Python ACP package:** `{available}`\n"
                f"**Sessions:** `{len(manager.list_sessions())}`\n"
                f"**Running:** `{running}`\n"
                f"**Bound session for this chat:** `{bound}`"
            )

        if action == "install":
            try:
                import acp as _acp  # noqa: F401
                return "ACP support is already installed in this Hermes environment."
            except Exception as exc:
                return f"ACP tooling is not importable in this environment: {exc}"

        if action == "status":
            target_id, remainder = self._resolve_acp_target(session_key, payload)
            if target_id:
                return self._format_acp_status(target_id)
            return self._format_acp_sessions(session_key)

        target_id, remainder = self._resolve_acp_target(session_key, payload)
        if not target_id:
            return "No ACP session is bound for this chat. Use `/acp spawn [cwd]` first or pass an explicit session ID."
        state = manager.get_session(target_id)
        if state is None:
            return f"No ACP session found for `{target_id}`."
        meta = self._get_acp_session_meta_entry(target_id)

        if action == "close":
            task = self._get_acp_tasks().get(target_id)
            if task and not task.done():
                return f"ACP session `{target_id}` is still running. Use `/acp cancel {target_id}` first."
            removed = manager.remove_session(target_id)
            if removed:
                for key, value in list(bindings.items()):
                    if value == target_id:
                        bindings.pop(key, None)
                self._get_acp_meta().pop(target_id, None)
                return f"🧩 Closed ACP session `{target_id}`."
            return f"No ACP session found for `{target_id}`."

        if action == "cancel":
            task = self._get_acp_tasks().get(target_id)
            if state.cancel_event:
                state.cancel_event.set()
            try:
                state.agent.interrupt()
            except Exception:
                logger.debug("ACP cancel interrupt failed", exc_info=True)
            if task and not task.done():
                return f"🛑 Cancel requested for ACP session `{target_id}`."
            return f"ACP session `{target_id}` was idle; cancel flag set."

        if action == "cwd":
            new_cwd = str(remainder or "").strip()
            if not new_cwd:
                return f"🧩 ACP session `{target_id}` cwd: `{state.cwd}`"
            manager.update_cwd(target_id, new_cwd)
            return f"🧩 ACP session `{target_id}` cwd set to `{new_cwd}`."

        if action == "model":
            new_model = str(remainder or "").strip()
            if not new_model:
                return f"🧩 ACP session `{target_id}` model: `{state.model or getattr(state.agent, 'model', '') or 'default'}`"
            state.model = new_model
            try:
                state.agent.model = new_model
            except Exception:
                logger.debug("Failed to update ACP agent model", exc_info=True)
            meta["updated_at"] = self._subagent_now()
            return f"🧩 ACP session `{target_id}` model set to `{new_model}`."

        if action == "set-mode":
            mode = str(remainder or "").strip()
            if not mode:
                return "Usage: `/acp set-mode [session_id] <mode>`"
            meta["options"]["mode"] = mode
            meta["updated_at"] = self._subagent_now()
            return f"🧩 ACP session `{target_id}` mode set to `{mode}`."

        if action == "permissions":
            perms = str(remainder or "").strip()
            if not perms:
                current = meta["options"].get("permissions", "default")
                return f"🧩 ACP session `{target_id}` permissions: `{current}`"
            meta["options"]["permissions"] = perms
            meta["updated_at"] = self._subagent_now()
            return f"🧩 ACP session `{target_id}` permissions set to `{perms}`."

        if action == "timeout":
            raw_timeout = str(remainder or "").strip()
            if not raw_timeout:
                current = meta["options"].get("timeout", "default")
                return f"🧩 ACP session `{target_id}` timeout: `{current}`"
            try:
                timeout_value = int(raw_timeout)
            except ValueError:
                return f"Invalid timeout `{raw_timeout}`. Expected an integer number of seconds."
            meta["options"]["timeout"] = timeout_value
            meta["updated_at"] = self._subagent_now()
            return f"🧩 ACP session `{target_id}` timeout set to `{timeout_value}`s."

        if action == "set":
            kv = self._split_command_args(remainder, max_parts=2)
            if len(kv) < 2:
                return "Usage: `/acp set [session_id] <key> <value>`"
            key, value = kv
            meta["options"][key] = value
            meta["updated_at"] = self._subagent_now()
            return f"🧩 ACP session `{target_id}` option `{key}` set to `{value}`."

        if action == "reset-options":
            cleared = len(meta["options"])
            meta["options"] = {}
            meta["updated_at"] = self._subagent_now()
            return f"🧩 ACP session `{target_id}` reset `{cleared}` option(s)."

        if action == "steer":
            prompt = str(remainder or "").strip()
            if not prompt:
                return "Usage: `/acp steer [session_id] <message>`"
            return await self._run_acp_prompt(source=target_source, session_id=target_id, prompt=prompt)

        return (
            "Usage: `/acp sessions|spawn|status|cancel|close|steer|cwd|model|set-mode|set|permissions|timeout|reset-options|doctor|install`\n"
            "Examples:\n"
            "`/acp spawn /home/alan/projects/hermes-agent`\n"
            "`/acp steer write a quick status report`\n"
            "`/acp status`"
        )

    async def _handle_bash_command(self, event: MessageEvent) -> str:
        """Handle /bash command and !-style shortcuts for host shell execution."""
        from tools.process_registry import process_registry

        target_source = self._command_target_source_for_event(event)
        session_entry = self.session_store.get_or_create_session(target_source)
        session_key = session_entry.session_key

        raw_args = event.get_command_args().strip()
        parts = self._split_command_args(raw_args, max_parts=2)
        if not parts:
            return (
                "Usage: `/bash <command>`\n"
                "       `/bash poll [session_id]`\n"
                "       `/bash stop [session_id]`\n\n"
                "Shortcuts: `! <command>`, `!poll`, `!stop`"
            )

        action = parts[0].strip().lower()
        if action == "poll":
            target_id = self._resolve_bash_target_session_id(session_key, parts[1] if len(parts) > 1 else "")
            if not target_id:
                return "No active bash job found for this session."
            poll = process_registry.poll(target_id)
            if poll.get("status") == "not_found":
                self._get_bash_jobs().pop(session_key, None)
                return f"No bash job found for `{target_id}`."
            log_payload = process_registry.read_log(target_id, limit=40)
            if poll.get("status") == "exited":
                self._get_bash_jobs().pop(session_key, None)
                return "\n".join(
                    [
                        "💻 **Bash Complete**",
                        "",
                        f"**Session ID:** `{target_id}`",
                        f"**Exit Code:** `{poll.get('exit_code')}`",
                        "",
                        self._format_bash_output(log_payload.get("output", "")),
                    ]
                )
            return "\n".join(
                [
                    "💻 **Bash Status**",
                    "",
                    f"**Session ID:** `{target_id}`",
                    f"**PID:** `{poll.get('pid')}`" if poll.get("pid") else "**PID:** `unknown`",
                    f"**Uptime:** `{poll.get('uptime_seconds', 0)}s`",
                    "",
                    self._format_bash_output(log_payload.get("output", "")),
                ]
            )

        if action == "stop":
            target_id = self._resolve_bash_target_session_id(session_key, parts[1] if len(parts) > 1 else "")
            if not target_id:
                return "No active bash job found for this session."
            result = process_registry.kill_process(target_id)
            if self._get_bash_jobs().get(session_key) == target_id:
                self._get_bash_jobs().pop(session_key, None)
            if result.get("status") in {"killed", "already_exited"}:
                return f"🛑 Bash job `{target_id}` stopped."
            return f"❌ Failed to stop bash job `{target_id}`: {result.get('error') or result.get('status')}"

        current_job_id = self._resolve_bash_target_session_id(session_key, "")
        if current_job_id:
            return (
                "A bash job is already running for this session.\n\n"
                f"**Session ID:** `{current_job_id}`\n"
                "Use `/bash poll` to inspect it or `/bash stop` to terminate it first."
            )

        command = raw_args
        result = self._run_bash_process(
            source=target_source,
            session_key=session_key,
            command=command,
            force=False,
        )

        if str(result.get("status") or "") == "approval_required":
            approval_id = self._store_pending_bash_approval(
                session_key=session_key,
                command=command,
                approval_payload=result,
                on_approve=lambda _decision: self._resume_bash_after_approval(
                    source=target_source,
                    session_key=session_key,
                    command=command,
                ),
            )
            if target_source.platform == Platform.DISCORD:
                approval_adapter = self.adapters.get(Platform.DISCORD)
                if approval_adapter and hasattr(approval_adapter, "send_exec_approval"):
                    try:
                        await approval_adapter.send_exec_approval(
                            target_source.chat_id,
                            command,
                            approval_id,
                        )
                    except Exception as approval_error:
                        logger.debug("Discord exec approval UI failed for /bash: %s", approval_error)
            return (
                f"⚠️ Bash command requires approval (`{approval_id}`).\n\n"
                f"```\n{command}\n```\n\n"
                f"Use {self._approval_usage_text()} or the Discord approval buttons."
            )

        return self._format_bash_result(
            session_key=session_key,
            command=command,
            result=result,
        )
    
    async def _handle_stop_command(self, event: MessageEvent) -> str:
        """Handle /stop command - interrupt a running agent."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        agent = self._running_agents.get(session_key)
        if agent is _AGENT_PENDING_SENTINEL:
            return "⏳ The agent is still starting up — nothing to stop yet."
        if agent:
            agent.interrupt()
            return "⚡ Stopping the current task... The agent will finish its current step and respond."
        else:
            return "No active task to stop."
    
    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        if self._session_source_for_event(event).platform == Platform.DISCORD:
            from gateway.platforms.discord_impl import runtime_views

            return runtime_views.render_discord_help()

        from hermes_cli.commands import gateway_help_lines
        lines = [
            "📖 **Hermes Commands**\n",
            *gateway_help_lines(),
        ]
        try:
            from agent.skill_commands import get_skill_commands
            skill_cmds = get_skill_commands()
            if skill_cmds:
                lines.append(f"\n⚡ **Skill Commands** ({len(skill_cmds)} installed):")
                for cmd in sorted(skill_cmds):
                    lines.append(f"`{cmd}` — {skill_cmds[cmd]['description']}")
        except Exception:
            pass
        return "\n".join(lines)

    async def _handle_commands_command(self, event: MessageEvent) -> str:
        """Handle /commands command - show the full gateway command catalog."""
        if self._session_source_for_event(event).platform == Platform.DISCORD:
            from gateway.platforms.discord_impl import runtime_views

            return runtime_views.render_discord_commands()

        from hermes_cli.commands import gateway_help_lines

        return "\n".join([
            "🧭 **Hermes Command Catalog**",
            "",
            *gateway_help_lines(),
        ])

    async def _handle_context_command(self, event: MessageEvent) -> str:
        """Handle /context command - inspect the current transcript and prompt context."""
        mode = (event.get_command_args().strip() or "list").lower()
        if mode not in {"list", "detail", "json"}:
            return (
                f"⚠️ Unknown context mode: `{mode}`\n\n"
                "Use `/context [list|detail|json]`"
            )

        target_source = self._command_target_source_for_event(event)
        session_entry = self.session_store.get_or_create_session(target_source)
        history = self.session_store.load_transcript(session_entry.session_id)

        if self._session_source_for_event(event).platform == Platform.DISCORD:
            from gateway.platforms.discord_impl import runtime_views

            snapshot = runtime_views.collect_discord_context_snapshot(self, event)
            return runtime_views.render_discord_context(snapshot, mode)

        role_counts: dict[str, int] = {}
        for item in history:
            role = str(item.get("role") or "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        lines = [
            "🧠 **Hermes Context**",
            "",
            f"• Session ID: `{session_entry.session_id}`",
            f"• Session Key: `{session_entry.session_key}`",
            f"• Messages: {len(history)}",
            f"• Roles: {', '.join(f'{k}={v}' for k, v in sorted(role_counts.items())) or 'none'}",
        ]
        if mode == "detail":
            preview = history[-5:]
            lines.extend(["", "```json", json.dumps(preview, indent=2, ensure_ascii=False), "```"])
        elif mode == "json":
            payload = {
                "session_id": session_entry.session_id,
                "session_key": session_entry.session_key,
                "message_count": len(history),
                "role_counts": role_counts,
            }
            return f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"
        return "\n".join(lines)

    async def _handle_export_session_command(self, event: MessageEvent) -> str:
        """Handle /export-session command - write a session export to disk."""
        target_source = self._command_target_source_for_event(event)
        session_entry = self.session_store.get_or_create_session(target_source)
        history = self.session_store.load_transcript(session_entry.session_id)
        from gateway.platforms.discord_impl import runtime_views

        snapshot = runtime_views.collect_discord_context_snapshot(self, event)
        export_payload = {
            "session": {
                "session_id": session_entry.session_id,
                "session_key": session_entry.session_key,
                "source": target_source.to_dict(),
                "current_model": snapshot.current_model,
                "current_provider": snapshot.current_provider,
                "connected_platforms": list(snapshot.connected_platforms),
            },
            "context_prompt": snapshot.context_prompt,
            "messages": history,
        }
        cwd = os.environ.get("TERMINAL_CWD") or os.environ.get("MESSAGING_CWD") or os.getcwd()
        raw_path = event.get_command_args().strip()
        export_path = resolve_export_path(raw_path, cwd, session_id=session_entry.session_id)
        if export_path.suffix.lower() == ".json":
            export_path.write_text(json.dumps(export_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            mode = "JSON"
        else:
            export_path.write_text(build_session_export_html(export_payload), encoding="utf-8")
            mode = "HTML"
        return (
            f"📦 Exported the current session to `{export_path}`\n\n"
            f"• Format: {mode}\n"
            f"• Messages: {len(history)}\n"
            f"• Session ID: `{session_entry.session_id}`"
        )

    async def _handle_whoami_command(self, event: MessageEvent) -> str:
        """Handle /whoami command - show the sender and routing identity Hermes sees."""
        if self._session_source_for_event(event).platform == Platform.DISCORD:
            from gateway.platforms.discord_impl import runtime_views

            snapshot = runtime_views.collect_discord_whoami_snapshot(self, event)
            return runtime_views.render_discord_whoami(snapshot)

        source = event.source
        lines = [
            "👤 **Hermes Sees You As**",
            "",
            f"**Platform:** {source.platform.value if source.platform else 'unknown'}",
            f"**User ID:** `{source.user_id or 'unknown'}`",
            f"**User Name:** {source.user_name or 'unknown'}",
            f"**Chat ID:** `{source.chat_id}`",
            f"**Chat Type:** {source.chat_type}",
        ]
        if source.chat_name:
            lines.append(f"**Chat Name:** {source.chat_name}")
        if source.thread_id:
            lines.append(f"**Thread ID:** `{source.thread_id}`")
        if source.chat_topic:
            lines.append(f"**Chat Topic:** {source.chat_topic}")
        return "\n".join(lines)

    def _resolve_discord_thread_target(
        self,
        event: MessageEvent,
    ) -> tuple[Optional[Any], Optional[SessionSource], Optional[str], Optional[str]]:
        """Return Discord adapter, target source, thread id, and parent channel id."""
        target_source = self._command_target_source_for_event(event)
        if target_source.platform != Platform.DISCORD:
            return None, None, None, None

        adapter = self._get_discord_adapter()
        if adapter is None:
            return None, None, None, None

        thread_id = str(target_source.thread_id or "").strip()
        if not thread_id:
            return adapter, target_source, None, None

        raw = getattr(event, "raw_message", None)
        channel = getattr(raw, "channel", None)
        parent_chat_id = None
        if channel is not None and hasattr(adapter, "_get_parent_channel_id"):
            parent_chat_id = adapter._get_parent_channel_id(channel)
        return adapter, target_source, thread_id, parent_chat_id

    def _render_thread_binding_status(self, binding: Any) -> str:
        idle_label = self._format_minutes_label(int(binding.idle_timeout_minutes or 0))
        max_age_label = self._format_minutes_label(int(binding.max_age_minutes or 0))
        lines = [
            "🧵 **Focused Thread**",
            "",
            f"• Thread: `{binding.thread_id}`",
            f"• Session: `{binding.session_key}`",
            f"• Label: {binding.chat_name or binding.thread_id}",
            f"• Bound By: {binding.bound_by or 'unknown'}",
            f"• Idle Auto-Unfocus: `{idle_label}`",
            f"• Max Age: `{max_age_label}`",
        ]
        if binding.parent_chat_id:
            lines.append(f"• Parent Channel: `{binding.parent_chat_id}`")
        return "\n".join(lines)

    async def _handle_focus_command(self, event: MessageEvent) -> str:
        """Handle /focus for Discord thread binding."""
        adapter, target_source, thread_id, parent_chat_id = self._resolve_discord_thread_target(event)
        if adapter is None or target_source is None:
            return "This command is only available on Discord."
        if not thread_id:
            return "Use `/focus` inside a Discord thread."

        session_key = self._session_key_for_source(target_source)
        label = event.get_command_args().strip() or (target_source.chat_name or thread_id)
        binding = adapter.focus_thread_binding(
            thread_id=thread_id,
            session_key=session_key,
            chat_name=label,
            parent_chat_id=parent_chat_id,
            bound_by=target_source.user_name or target_source.user_id or "unknown",
        )
        return (
            "🧵 Thread focused.\n\n"
            f"{self._render_thread_binding_status(binding)}"
        )

    async def _handle_unfocus_command(self, event: MessageEvent) -> str:
        """Handle /unfocus for Discord thread binding removal."""
        adapter, _target_source, thread_id, _parent_chat_id = self._resolve_discord_thread_target(event)
        if adapter is None:
            return "This command is only available on Discord."
        if not thread_id:
            return "Use `/unfocus` inside a focused Discord thread."

        binding = adapter.unfocus_thread_binding(thread_id)
        if binding is None:
            return "No focused thread binding exists here."
        return "🧵 Thread unfocused. Messages here now require normal Discord activation again."

    async def _handle_agents_command(self, event: MessageEvent) -> str:
        """Handle /agents for Discord thread-binding inspection."""
        adapter, target_source, thread_id, parent_chat_id = self._resolve_discord_thread_target(event)
        if adapter is None or target_source is None:
            return "This command is only available on Discord."

        if thread_id:
            binding = adapter.get_thread_binding(thread_id)
            if binding is None:
                return "No focused thread binding exists for this thread."
            is_running = binding.session_key in getattr(self, "_running_agents", {})
            return "\n".join(
                [
                    "🧵 **Thread-Bound Hermes Sessions**",
                    "",
                    f"• Target: `{binding.session_key}`",
                    f"• Label: {binding.chat_name or binding.thread_id}",
                    f"• Running: {'yes' if is_running else 'no'}",
                    f"• Idle Auto-Unfocus: `{self._format_minutes_label(int(binding.idle_timeout_minutes or 0))}`",
                    f"• Max Age: `{self._format_minutes_label(int(binding.max_age_minutes or 0))}`",
                ]
            )

        bindings = adapter.list_thread_bindings(parent_chat_id=parent_chat_id or target_source.chat_id)
        if not bindings:
            return "No focused Discord thread bindings were found for this chat."
        lines = ["🧵 **Focused Discord Threads**", ""]
        for binding in bindings:
            running = "yes" if binding.session_key in getattr(self, "_running_agents", {}) else "no"
            lines.append(
                f"• `{binding.thread_id}` — {binding.chat_name or binding.thread_id} "
                f"(running: {running}, idle: {self._format_minutes_label(int(binding.idle_timeout_minutes or 0))}, "
                f"max-age: {self._format_minutes_label(int(binding.max_age_minutes or 0))})"
            )
        return "\n".join(lines)

    async def _handle_session_command(self, event: MessageEvent) -> str:
        """Handle /session idle|max-age controls for focused Discord threads."""
        adapter, _target_source, thread_id, _parent_chat_id = self._resolve_discord_thread_target(event)
        if adapter is None:
            return "This command is only available on Discord."
        if not thread_id:
            return "Use `/session` inside a focused Discord thread."

        binding = adapter.get_thread_binding(thread_id)
        if binding is None:
            return "No focused thread binding exists here. Use `/focus` first."

        tokens = [token for token in event.get_command_args().strip().split() if token]
        if not tokens or tokens[0].lower() == "status":
            return self._render_thread_binding_status(binding)
        if len(tokens) < 2:
            return "Use `/session idle <duration|off>` or `/session max-age <duration|off>`."

        mode = tokens[0].strip().lower()
        try:
            minutes = self._parse_duration_minutes(tokens[1])
        except ValueError as exc:
            return f"⚠️ {exc}"
        if minutes is None:
            return "Use `/session idle <duration|off>` or `/session max-age <duration|off>`."

        if mode == "idle":
            binding = adapter.update_thread_binding_limits(thread_id, idle_timeout_minutes=minutes)
        elif mode in {"max-age", "max_age", "maxage"}:
            binding = adapter.update_thread_binding_limits(thread_id, max_age_minutes=minutes)
            mode = "max-age"
        else:
            return "Use `/session idle <duration|off>` or `/session max-age <duration|off>`."

        if binding is None:
            return "No focused thread binding exists here. Use `/focus` first."
        return (
            f"🧵 Updated thread session `{mode}` to `{self._format_minutes_label(minutes)}`.\n\n"
            f"{self._render_thread_binding_status(binding)}"
        )

    async def _handle_send_command(self, event: MessageEvent) -> str:
        """Handle /send on|off|inherit for the current session."""
        target_source = self._command_target_source_for_event(event)
        session_key = self._session_key_for_source(target_source)
        args = event.get_command_args().strip().lower()
        current = self._session_send_policy(session_key)
        if not args:
            return f"📮 Current send policy for this session: `{current}`"
        if args not in {"on", "off", "inherit"}:
            return "Use `/send on`, `/send off`, or `/send inherit`."

        if not isinstance(getattr(self, "_session_send_policies", None), dict):
            self._session_send_policies = {}
        self._session_send_policies[session_key] = args
        if args == "inherit":
            self._session_send_policies.pop(session_key, None)
        self._save_runtime_controls()
        return f"📮 Send policy for this session is now `{args}`."

    async def _handle_activation_command(self, event: MessageEvent) -> str:
        """Handle /activation mention|always for Discord chats."""
        target_source = self._command_target_source_for_event(event)
        if target_source.platform != Platform.DISCORD or target_source.chat_type == "dm":
            return "This command only works in Discord servers and threads."

        adapter = self._get_discord_adapter()
        if adapter is None:
            return "Discord is not connected."

        args = event.get_command_args().strip().lower()
        current = adapter.get_activation_mode(target_source.chat_id) or (
            "mention" if adapter._get_discord_policy().require_mention else "always"
        )
        if not args:
            return f"🎛️ Current activation mode for this chat: `{current}`"
        if args not in {"mention", "always"}:
            return "Use `/activation mention` or `/activation always`."

        adapter.set_activation_mode(target_source.chat_id, args)
        return f"🎛️ Activation mode for this chat is now `{args}`."

    async def _handle_restart_command(self, event: MessageEvent) -> str:
        """Handle /restart by scheduling a delayed gateway restart."""
        del event
        self._schedule_gateway_restart()
        return "♻️ Gateway restart scheduled. Hermes will reconnect shortly."

    async def _handle_dock_command(self, event: MessageEvent, platform: Platform) -> str:
        """Handle /dock-* by routing future replies for this session to a home channel."""
        target_source = self._command_target_source_for_event(event)
        session_key = self._session_key_for_source(target_source)
        home = self.config.get_home_channel(platform)
        if home is None:
            return f"No home channel is configured for `{platform.value}`."

        if not isinstance(getattr(self, "_session_docks", None), dict):
            self._session_docks = {}
        self._session_docks[session_key] = {
            "platform": platform.value,
            "chat_id": home.chat_id,
            "thread_id": None,
            "name": home.name,
        }
        self._save_runtime_controls()
        return f"📮 Replies for this session are now docked to {platform.value} home channel **{home.name}** (`{home.chat_id}`)."

    def _load_current_model_selection(self) -> dict[str, Any]:
        """Resolve the configured model/provider from config plus runtime env."""
        import yaml
        from hermes_cli.models import normalize_provider

        config_path = _hermes_home / "config.yaml"
        current_model = os.getenv("HERMES_MODEL") or "anthropic/claude-opus-4.6"
        current_provider = "openrouter"

        try:
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, str):
                    current_model = model_cfg
                elif isinstance(model_cfg, dict):
                    current_model = model_cfg.get("default", current_model)
                    current_provider = model_cfg.get("provider", current_provider)
        except Exception:
            pass

        current_provider = normalize_provider(current_provider)
        if current_provider == "auto":
            try:
                from hermes_cli.auth import resolve_provider as _resolve_provider

                current_provider = _resolve_provider(current_provider)
            except Exception:
                current_provider = "openrouter"

        if current_provider == "openrouter" and os.getenv("OPENAI_BASE_URL", "").strip():
            current_provider = "custom"

        return {
            "config_path": config_path,
            "current_model": current_model,
            "current_provider": current_provider,
        }

    def _resolve_model_runtime_details(self, requested_provider: str) -> tuple[dict[str, Any], Optional[str]]:
        """Return runtime provider details for the active model selection."""
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime = resolve_runtime_provider(requested=requested_provider)
            return runtime, None
        except Exception as exc:
            return {}, str(exc)

    async def _respond_to_native_slash(
        self,
        event: MessageEvent,
        content: str,
    ) -> bool:
        """Send an ephemeral response for Discord native slash commands."""
        interaction = getattr(event, "raw_message", None)
        if interaction is None:
            return False

        response = getattr(interaction, "response", None)
        if response is not None and hasattr(response, "send_message"):
            is_done = getattr(response, "is_done", None)
            if not callable(is_done) or not is_done():
                await response.send_message(content, ephemeral=True)
                return True

        followup = getattr(interaction, "followup", None)
        if followup is not None and hasattr(followup, "send"):
            await followup.send(content, ephemeral=True)
            return True
        return False

    def _is_native_discord_slash(self, event: MessageEvent) -> bool:
        metadata = getattr(event, "metadata", None) or {}
        return bool(
            event.source.platform == Platform.DISCORD
            and isinstance(metadata, dict)
            and metadata.get("is_native_slash")
        )

    def _render_model_catalog_response(self) -> str:
        """Render the current model plus numbered catalog for the active provider."""
        from hermes_cli.models import curated_models_for_provider, provider_label

        state = self._load_current_model_selection()
        current_model = state["current_model"]
        current_provider = state["current_provider"]
        provider_name = provider_label(current_provider)
        lines = [
            "🤖 **Model Catalog**",
            "",
            f"**Current model:** `{current_model}`",
            f"**Provider:** {provider_name} (`{current_provider}`)",
        ]
        if self._effective_model:
            lines.extend(
                [
                    f"**Active fallback:** `{self._effective_model}`",
                    f"**Fallback provider:** {provider_label(self._effective_provider or 'unknown')}",
                ]
            )
        lines.append("")

        curated = curated_models_for_provider(current_provider)
        if curated:
            lines.append(f"**Available models ({provider_name}):**")
            for index, (model_id, desc) in enumerate(curated, start=1):
                marker = " ← current" if model_id == current_model else ""
                detail = f" — {desc}" if desc else ""
                lines.append(f"{index}. `{model_id}`{detail}{marker}")
            lines.append("")
        else:
            lines.extend(
                [
                    f"No curated model list is available for `{current_provider}` right now.",
                    "",
                ]
            )

        lines.extend(
            [
                "Use `/model <number>` to choose by index.",
                "Use `/models` on Discord to open the interactive picker.",
                "Use `/model provider:model-name` to switch providers.",
                "Use `/model status` for runtime endpoint and auth details.",
            ]
        )
        return "\n".join(lines)

    def _render_model_status_response(self) -> str:
        """Render detailed model/provider/runtime status."""
        from hermes_cli.models import provider_label

        state = self._load_current_model_selection()
        current_model = state["current_model"]
        current_provider = state["current_provider"]
        runtime, runtime_error = self._resolve_model_runtime_details(current_provider)
        runtime_provider = str(runtime.get("provider") or current_provider)
        runtime_label = provider_label(runtime_provider)
        api_key = str(runtime.get("api_key") or "").strip()
        command = runtime.get("command")
        lines = [
            "🤖 **Model Status**",
            "",
            f"**Configured model:** `{current_model}`",
            f"**Configured provider:** {provider_label(current_provider)} (`{current_provider}`)",
        ]
        if self._effective_model:
            effective_provider = self._effective_provider or current_provider
            lines.extend(
                [
                    f"**Active model:** `{self._effective_model}` (fallback)",
                    f"**Active provider:** {provider_label(effective_provider)} (`{effective_provider}`)",
                    f"**Primary model:** `{current_model}`",
                ]
            )
        else:
            lines.append(f"**Active model:** `{current_model}`")

        if runtime_error:
            lines.extend(
                [
                    "",
                    f"⚠️ Runtime provider resolution failed: {runtime_error}",
                ]
            )
            return "\n".join(lines)

        lines.extend(
            [
                f"**Runtime provider:** {runtime_label} (`{runtime_provider}`)",
                f"**API mode:** `{runtime.get('api_mode') or 'unknown'}`",
                f"**Base URL:** `{runtime.get('base_url') or 'unknown'}`",
                f"**Credentials:** {'configured ✓' if api_key or command else 'missing ⚠️'}",
            ]
        )
        if command:
            lines.append(f"**Transport command:** `{command}`")
        source = runtime.get("source")
        if source:
            lines.append(f"**Credential source:** `{source}`")
        return "\n".join(lines)

    def _resolve_numbered_model(self, raw_index: str, current_provider: str) -> Optional[str]:
        """Resolve `/model <number>` against the current provider catalog."""
        from hermes_cli.models import curated_models_for_provider

        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return None
        if index < 1:
            return None
        curated = curated_models_for_provider(current_provider)
        if index > len(curated):
            return None
        return curated[index - 1][0]

    async def _apply_model_selection(
        self,
        target_provider: str,
        new_model: str,
        user_id: Optional[str] = None,
    ) -> str:
        """Validate, persist, and activate a model/provider choice."""
        import yaml
        from hermes_cli.models import validate_requested_model, provider_label

        state = self._load_current_model_selection()
        current_provider = state["current_provider"]
        config_path = state["config_path"]
        provider_changed = target_provider != current_provider
        runtime, runtime_error = self._resolve_model_runtime_details(
            target_provider if provider_changed else current_provider
        )
        if provider_changed and runtime_error:
            return f"⚠️ Could not resolve credentials for provider '{provider_label(target_provider)}': {runtime_error}"

        api_key = str(runtime.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
        base_url = str(runtime.get("base_url") or "https://openrouter.ai/api/v1")

        try:
            validation = validate_requested_model(
                new_model,
                target_provider,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception:
            validation = {"accepted": True, "persist": True, "recognized": False, "message": None}

        if not validation.get("accepted"):
            message = validation.get("message", "Invalid model")
            tip = "\n\nUse `/model list` to see numbered models, `/provider` to see providers." if "Did you mean" not in message else ""
            return f"⚠️ {message}{tip}"

        if validation.get("persist"):
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                if "model" not in user_config or not isinstance(user_config["model"], dict):
                    user_config["model"] = {}
                user_config["model"]["default"] = new_model
                user_config["model"]["provider"] = target_provider
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(user_config, f, default_flow_style=False, sort_keys=False)
            except Exception as exc:
                return f"⚠️ Failed to save model change: {exc}"

        os.environ["HERMES_MODEL"] = new_model
        os.environ["HERMES_INFERENCE_PROVIDER"] = target_provider

        try:
            from gateway.platforms.discord_impl import model_picker as discord_model_picker

            discord_model_picker.record_recent_model(user_id, target_provider, new_model)
        except Exception:
            pass

        self._effective_model = None
        self._effective_provider = None

        warning = f"\n⚠️ {validation['message']}" if validation.get("message") else ""
        persist_note = "saved to config" if validation.get("persist") else "this session only — will revert on restart"
        return (
            f"🤖 Model changed to `{new_model}` ({persist_note})\n"
            f"**Provider:** {provider_label(target_provider)} (`{target_provider}`){warning}\n"
            "_(takes effect on next message)_"
        )

    async def _open_discord_model_picker(self, event: MessageEvent, command_name: str) -> str | None:
        """Open the Discord-native interactive model picker."""
        if not self._is_native_discord_slash(event):
            return self._render_model_catalog_response()

        adapter = self.adapters.get(Platform.DISCORD)
        interaction = getattr(event, "raw_message", None)
        if adapter is None or interaction is None:
            return self._render_model_catalog_response()

        state = self._load_current_model_selection()
        try:
            from gateway.platforms.discord_impl import model_picker as discord_model_picker

            await discord_model_picker.open_model_picker(
                adapter=adapter,
                interaction=interaction,
                command_name=command_name,
                user_id=str(event.source.user_id or ""),
                current_provider=state["current_provider"],
                current_model=state["current_model"],
                apply_selection=self._apply_model_selection,
            )
        except Exception as exc:
            logger.warning("Discord model picker failed; falling back to text response: %s", exc, exc_info=True)
            return self._render_model_catalog_response()
        return None

    async def _handle_model_command(self, event: MessageEvent) -> str | None:
        """Handle /model and /models commands."""
        from hermes_cli.models import parse_model_input, detect_provider_for_model

        command_name = (event.get_command() or "model").lower()
        args = event.get_command_args().strip()
        is_native_slash = self._is_native_discord_slash(event)
        state = self._load_current_model_selection()
        current_provider = state["current_provider"]

        response: Optional[str]

        lowered = args.lower()
        if command_name == "models":
            if is_native_slash:
                return await self._open_discord_model_picker(event, command_name="models")
            response = self._render_model_catalog_response()
        elif not args:
            if is_native_slash:
                return await self._open_discord_model_picker(event, command_name="model")
            response = self._render_model_catalog_response()
        elif lowered == "status":
            response = self._render_model_status_response()
        elif lowered == "list":
            response = self._render_model_catalog_response()
        else:
            numbered_model = self._resolve_numbered_model(args, current_provider)
            if numbered_model:
                response = await self._apply_model_selection(
                    current_provider,
                    numbered_model,
                    user_id=event.source.user_id,
                )
            else:
                target_provider, new_model = parse_model_input(args, current_provider)
                if target_provider == current_provider:
                    detected = detect_provider_for_model(new_model, current_provider)
                    if detected:
                        target_provider, new_model = detected
                response = await self._apply_model_selection(
                    target_provider,
                    new_model,
                    user_id=event.source.user_id,
                )

        if is_native_slash and response:
            await self._respond_to_native_slash(event, response)
            return None
        return response

    async def _handle_provider_command(self, event: MessageEvent) -> str:
        """Handle /provider command - show available providers."""
        import yaml
        from hermes_cli.models import (
            list_available_providers,
            normalize_provider,
            _PROVIDER_LABELS,
        )

        # Resolve current provider from config
        current_provider = "openrouter"
        config_path = _hermes_home / 'config.yaml'
        try:
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    current_provider = model_cfg.get("provider", current_provider)
        except Exception:
            pass

        current_provider = normalize_provider(current_provider)
        if current_provider == "auto":
            try:
                from hermes_cli.auth import resolve_provider as _resolve_provider
                current_provider = _resolve_provider(current_provider)
            except Exception:
                current_provider = "openrouter"

        # Detect custom endpoint
        if current_provider == "openrouter" and os.getenv("OPENAI_BASE_URL", "").strip():
            current_provider = "custom"

        current_label = _PROVIDER_LABELS.get(current_provider, current_provider)

        lines = [
            f"🔌 **Current provider:** {current_label} (`{current_provider}`)",
            "",
            "**Available providers:**",
        ]

        providers = list_available_providers()
        for p in providers:
            marker = " ← active" if p["id"] == current_provider else ""
            auth = "✅" if p["authenticated"] else "❌"
            aliases = f"  _(also: {', '.join(p['aliases'])})_" if p["aliases"] else ""
            lines.append(f"{auth} `{p['id']}` — {p['label']}{aliases}{marker}")

        lines.append("")
        lines.append("Switch: `/model provider:model-name`")
        lines.append("Setup: `hermes setup`")
        return "\n".join(lines)
    
    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command - list or set a personality."""
        import yaml

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / 'config.yaml'

        try:
            if config_path.exists():
                with open(config_path, 'r', encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                personalities = config.get("agent", {}).get("personalities", {})
            else:
                config = {}
                personalities = {}
        except Exception:
            config = {}
            personalities = {}

        if not personalities:
            return "No personalities configured in `~/.hermes/config.yaml`"

        if not args:
            lines = ["🎭 **Available Personalities**\n"]
            lines.append("• `none` — (no personality overlay)")
            for name, prompt in personalities.items():
                if isinstance(prompt, dict):
                    preview = prompt.get("description") or prompt.get("system_prompt", "")[:50]
                else:
                    preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                lines.append(f"• `{name}` — {preview}")
            lines.append(f"\nUsage: `/personality <name>`")
            return "\n".join(lines)

        def _resolve_prompt(value):
            if isinstance(value, dict):
                parts = [value.get("system_prompt", "")]
                if value.get("tone"):
                    parts.append(f'Tone: {value["tone"]}')
                if value.get("style"):
                    parts.append(f'Style: {value["style"]}')
                return "\n".join(p for p in parts if p)
            return str(value)

        if args in ("none", "default", "neutral"):
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = ""
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                return f"⚠️ Failed to save personality change: {e}"
            self._ephemeral_system_prompt = ""
            return "🎭 Personality cleared — using base agent behavior.\n_(takes effect on next message)_"
        elif args in personalities:
            new_prompt = _resolve_prompt(personalities[args])

            # Write to config.yaml, same pattern as CLI save_config_value.
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = new_prompt
                with open(config_path, 'w', encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                return f"⚠️ Failed to save personality change: {e}"

            # Update in-memory so it takes effect on the very next message.
            self._ephemeral_system_prompt = new_prompt

            return f"🎭 Personality set to **{args}**\n_(takes effect on next message)_"

        available = "`none`, " + ", ".join(f"`{n}`" for n in personalities.keys())
        return f"Unknown personality: `{args}`\n\nAvailable: {available}"
    
    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message
        last_user_msg = None
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_msg = history[i].get("content", "")
                last_user_idx = i
                break
        
        if not last_user_msg:
            return "No previous message to retry."
        
        # Truncate history to before the last user message and persist
        truncated = history[:last_user_idx]
        self.session_store.rewrite_transcript(session_entry.session_id, truncated)
        # Reset stored token count — transcript was truncated
        session_entry.last_prompt_tokens = 0
        
        # Re-send by creating a fake text event with the old message
        retry_event = MessageEvent(
            text=last_user_msg,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event.raw_message,
        )
        
        # Let the normal message handler process it
        return await self._handle_message(retry_event)
    
    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo command - remove the last user/assistant exchange."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message and remove everything from it onward
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            return "Nothing to undo."
        
        removed_msg = history[last_user_idx].get("content", "")
        removed_count = len(history) - last_user_idx
        self.session_store.rewrite_transcript(session_entry.session_id, history[:last_user_idx])
        # Reset stored token count — transcript was truncated
        session_entry.last_prompt_tokens = 0
        
        preview = removed_msg[:40] + "..." if len(removed_msg) > 40 else removed_msg
        return f"↩️ Undid {removed_count} message(s).\nRemoved: \"{preview}\""
    
    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id
        
        env_key = f"{platform_name.upper()}_HOME_CHANNEL"
        
        # Save to config.yaml
        try:
            import yaml
            config_path = _hermes_home / 'config.yaml'
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            user_config[env_key] = chat_id
            with open(config_path, 'w', encoding="utf-8") as f:
                yaml.dump(user_config, f, default_flow_style=False)
            # Also set in the current environment so it takes effect immediately
            os.environ[env_key] = str(chat_id)
        except Exception as e:
            return f"Failed to save home channel: {e}"
        
        return (
            f"✅ Home channel set to **{chat_name}** (ID: {chat_id}).\n"
            f"Cron jobs and cross-platform messages will be delivered here."
        )
    
    @staticmethod
    def _get_guild_id(event: MessageEvent) -> Optional[int]:
        """Extract Discord guild_id from the raw message object."""
        raw = getattr(event, "raw_message", None)
        if raw is None:
            return None
        # Slash command interaction
        if hasattr(raw, "guild_id") and raw.guild_id:
            return int(raw.guild_id)
        # Regular message
        if hasattr(raw, "guild") and raw.guild:
            return raw.guild.id
        return None

    async def _handle_voice_command(self, event: MessageEvent) -> str:
        """Handle /voice [on|off|tts|channel|leave|status] command."""
        args = event.get_command_args().strip().lower()
        chat_id = event.source.chat_id

        adapter = self.adapters.get(event.source.platform)

        if args in ("on", "enable"):
            self._voice_mode[chat_id] = "voice_only"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=False)
            return (
                "Voice mode enabled.\n"
                "I'll reply with voice when you send voice messages.\n"
                "Use /voice tts to get voice replies for all messages."
            )
        elif args in ("off", "disable"):
            self._voice_mode[chat_id] = "off"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
            return "Voice mode disabled. Text-only replies."
        elif args == "tts":
            self._voice_mode[chat_id] = "all"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=False)
            return (
                "Auto-TTS enabled.\n"
                "All replies will include a voice message."
            )
        elif args in ("channel", "join"):
            return await self._handle_voice_channel_join(event)
        elif args == "leave":
            return await self._handle_voice_channel_leave(event)
        elif args == "status":
            mode = self._voice_mode.get(chat_id, "off")
            labels = {
                "off": "Off (text only)",
                "voice_only": "On (voice reply to voice messages)",
                "all": "TTS (voice reply to all messages)",
            }
            # Append voice channel info if connected
            adapter = self.adapters.get(event.source.platform)
            guild_id = self._get_guild_id(event)
            if guild_id and hasattr(adapter, "get_voice_channel_info"):
                info = adapter.get_voice_channel_info(guild_id)
                if info:
                    lines = [
                        f"Voice mode: {labels.get(mode, mode)}",
                        f"Voice channel: #{info['channel_name']}",
                        f"Participants: {info['member_count']}",
                    ]
                    for m in info["members"]:
                        status = " (speaking)" if m.get("is_speaking") else ""
                        lines.append(f"  - {m['display_name']}{status}")
                    return "\n".join(lines)
            return f"Voice mode: {labels.get(mode, mode)}"
        else:
            # Toggle: off → on, on/all → off
            current = self._voice_mode.get(chat_id, "off")
            if current == "off":
                self._voice_mode[chat_id] = "voice_only"
                self._save_voice_modes()
                if adapter:
                    self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=False)
                return "Voice mode enabled."
            else:
                self._voice_mode[chat_id] = "off"
                self._save_voice_modes()
                if adapter:
                    self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
                return "Voice mode disabled."

    async def _handle_voice_channel_join(self, event: MessageEvent) -> str:
        """Join the user's current Discord voice channel."""
        adapter = self.adapters.get(event.source.platform)
        if not hasattr(adapter, "join_voice_channel"):
            return "Voice channels are not supported on this platform."

        guild_id = self._get_guild_id(event)
        if not guild_id:
            return "This command only works in a Discord server."

        voice_channel = await adapter.get_user_voice_channel(
            guild_id, event.source.user_id
        )
        if not voice_channel:
            return "You need to be in a voice channel first."

        # Wire callbacks BEFORE join so voice input arriving immediately
        # after connection is not lost.
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = self._handle_voice_channel_input
        if hasattr(adapter, "_on_voice_disconnect"):
            adapter._on_voice_disconnect = self._handle_voice_timeout_cleanup

        try:
            success = await adapter.join_voice_channel(voice_channel)
        except Exception as e:
            logger.warning("Failed to join voice channel: %s", e)
            adapter._voice_input_callback = None
            err_lower = str(e).lower()
            if "pynacl" in err_lower or "nacl" in err_lower or "davey" in err_lower:
                return (
                    "Voice dependencies are missing (PyNaCl / davey). "
                    "Install or reinstall Hermes with the messaging extra, e.g. "
                    "`pip install hermes-agent[messaging]`."
                )
            return f"Failed to join voice channel: {e}"

        if success:
            adapter._voice_text_channels[guild_id] = int(event.source.chat_id)
            self._voice_mode[event.source.chat_id] = "all"
            self._save_voice_modes()
            self._set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=False)
            return (
                f"Joined voice channel **{voice_channel.name}**.\n"
                f"I'll speak my replies and listen to you. Use /voice leave to disconnect."
            )
        # Join failed — clear callback
        adapter._voice_input_callback = None
        return "Failed to join voice channel. Check bot permissions (Connect + Speak)."

    async def _handle_voice_channel_leave(self, event: MessageEvent) -> str:
        """Leave the Discord voice channel."""
        adapter = self.adapters.get(event.source.platform)
        guild_id = self._get_guild_id(event)

        if not guild_id or not hasattr(adapter, "leave_voice_channel"):
            return "Not in a voice channel."

        if not hasattr(adapter, "is_in_voice_channel") or not adapter.is_in_voice_channel(guild_id):
            return "Not in a voice channel."

        try:
            await adapter.leave_voice_channel(guild_id)
        except Exception as e:
            logger.warning("Error leaving voice channel: %s", e)
        # Always clean up state even if leave raised an exception
        self._voice_mode[event.source.chat_id] = "off"
        self._save_voice_modes()
        self._set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=True)
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = None
        return "Left voice channel."

    def _handle_voice_timeout_cleanup(self, chat_id: str) -> None:
        """Called by the adapter when a voice channel times out.

        Cleans up runner-side voice_mode state that the adapter cannot reach.
        """
        self._voice_mode[chat_id] = "off"
        self._save_voice_modes()
        adapter = self.adapters.get(Platform.DISCORD)
        self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)

    async def _handle_voice_channel_input(
        self, guild_id: int, user_id: int, transcript: str
    ):
        """Handle transcribed voice from a user in a voice channel.

        Creates a synthetic MessageEvent and processes it through the
        adapter's full message pipeline (session, typing, agent, TTS reply).
        """
        adapter = self.adapters.get(Platform.DISCORD)
        if not adapter:
            return

        text_ch_id = adapter._voice_text_channels.get(guild_id)
        if not text_ch_id:
            return

        # Check authorization before processing voice input
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id=str(text_ch_id),
            user_id=str(user_id),
            user_name=str(user_id),
            chat_type="channel",
        )
        if not self._is_user_authorized(source):
            logger.debug("Unauthorized voice input from user %d, ignoring", user_id)
            return

        # Show transcript in text channel (after auth, with mention sanitization)
        try:
            channel = adapter._client.get_channel(text_ch_id)
            if channel:
                safe_text = transcript[:2000].replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
                await channel.send(f"**[Voice]** <@{user_id}>: {safe_text}")
        except Exception:
            pass

        # Build a synthetic MessageEvent and feed through the normal pipeline
        # Use SimpleNamespace as raw_message so _get_guild_id() can extract
        # guild_id and _send_voice_reply() plays audio in the voice channel.
        from types import SimpleNamespace
        event = MessageEvent(
            source=source,
            text=transcript,
            message_type=MessageType.VOICE,
            raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
        )

        await adapter.handle_message(event)

    def _should_send_voice_reply(
        self,
        event: MessageEvent,
        response: str,
        agent_messages: list,
    ) -> bool:
        """Decide whether the runner should send a TTS voice reply.

        Returns False when:
        - voice_mode is off for this chat
        - response is empty or an error
        - agent already called text_to_speech tool (dedup)
        - voice input and base adapter auto-TTS already handled it (skip_double)
          Exception: Discord voice channel — base play_tts is a no-op there,
          so the runner must handle VC playback.
        """
        if not response or response.startswith("Error:"):
            return False

        chat_id = event.source.chat_id
        voice_mode = self._voice_mode.get(chat_id, "off")
        is_voice_input = (event.message_type == MessageType.VOICE)

        should = (
            (voice_mode == "all")
            or (voice_mode == "voice_only" and is_voice_input)
        )
        if not should:
            return False

        # Dedup: agent already called TTS tool
        has_agent_tts = any(
            msg.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "text_to_speech"
                for tc in (msg.get("tool_calls") or [])
            )
            for msg in agent_messages
        )
        if has_agent_tts:
            return False

        # Dedup: base adapter auto-TTS already handles voice input
        # (play_tts plays in VC when connected, so runner can skip).
        if is_voice_input:
            return False

        return True

    async def _send_voice_reply(self, event: MessageEvent, text: str) -> None:
        """Generate TTS audio and send as a voice message before the text reply."""
        import uuid as _uuid
        audio_path = None
        actual_path = None
        try:
            from tools.tts_tool import text_to_speech_tool, _strip_markdown_for_tts

            tts_text = _strip_markdown_for_tts(text[:4000])
            if not tts_text:
                return

            # Use .mp3 extension so edge-tts conversion to opus works correctly.
            # The TTS tool may convert to .ogg — use file_path from result.
            audio_path = os.path.join(
                tempfile.gettempdir(), "hermes_voice",
                f"tts_reply_{_uuid.uuid4().hex[:12]}.mp3",
            )
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            result_json = await asyncio.to_thread(
                text_to_speech_tool, text=tts_text, output_path=audio_path
            )
            result = json.loads(result_json)

            # Use the actual file path from result (may differ after opus conversion)
            actual_path = result.get("file_path", audio_path)
            if not result.get("success") or not os.path.isfile(actual_path):
                logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
                return

            adapter = self.adapters.get(event.source.platform)

            # If connected to a voice channel, play there instead of sending a file
            guild_id = self._get_guild_id(event)
            if (guild_id
                    and hasattr(adapter, "play_in_voice_channel")
                    and hasattr(adapter, "is_in_voice_channel")
                    and adapter.is_in_voice_channel(guild_id)):
                await adapter.play_in_voice_channel(guild_id, actual_path)
            elif adapter and hasattr(adapter, "send_voice"):
                send_kwargs: Dict[str, Any] = {
                    "chat_id": event.source.chat_id,
                    "audio_path": actual_path,
                    "reply_to": event.message_id,
                }
                if event.source.thread_id:
                    send_kwargs["metadata"] = {"thread_id": event.source.thread_id}
                await adapter.send_voice(**send_kwargs)
        except Exception as e:
            logger.warning("Auto voice reply failed: %s", e, exc_info=True)
        finally:
            for p in {audio_path, actual_path} - {None}:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from tools.checkpoint_manager import CheckpointManager, format_checkpoint_list

        # Read checkpoint config from config.yaml
        cp_cfg = {}
        try:
            import yaml as _y
            _cfg_path = _hermes_home / "config.yaml"
            if _cfg_path.exists():
                with open(_cfg_path, encoding="utf-8") as _f:
                    _data = _y.safe_load(_f) or {}
                cp_cfg = _data.get("checkpoints", {})
                if isinstance(cp_cfg, bool):
                    cp_cfg = {"enabled": cp_cfg}
        except Exception:
            pass

        if not cp_cfg.get("enabled", False):
            return (
                "Checkpoints are not enabled.\n"
                "Enable in config.yaml:\n```\ncheckpoints:\n  enabled: true\n```"
            )

        mgr = CheckpointManager(
            enabled=True,
            max_snapshots=cp_cfg.get("max_snapshots", 50),
        )

        cwd = os.getenv("MESSAGING_CWD", str(Path.home()))
        arg = event.get_command_args().strip()

        if not arg:
            checkpoints = mgr.list_checkpoints(cwd)
            return format_checkpoint_list(checkpoints, cwd)

        # Restore by number or hash
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            return f"No checkpoints found for {cwd}"

        target_hash = None
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(checkpoints):
                target_hash = checkpoints[idx]["hash"]
            else:
                return f"Invalid checkpoint number. Use 1-{len(checkpoints)}."
        except ValueError:
            target_hash = arg

        result = mgr.restore(cwd, target_hash)
        if result["success"]:
            return (
                f"✅ Restored to checkpoint {result['restored_to']}: {result['reason']}\n"
                f"A pre-rollback snapshot was saved automatically."
            )
        return f"❌ {result['error']}"

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /background <prompt> — run a prompt in a separate background session.

        Spawns a new AIAgent in a background thread with its own session.
        When it completes, sends the result back to the same chat without
        modifying the active session's conversation history.
        """
        prompt = event.get_command_args().strip()
        if not prompt:
            return (
                "Usage: /background <prompt>\n"
                "Example: /background Summarize the top HN stories today\n\n"
                "Runs the prompt in a separate session. "
                "You can keep chatting — the result will appear here when done."
            )

        source = event.source
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"

        # Fire-and-forget the background task
        asyncio.create_task(
            self._run_background_task(prompt, source, task_id)
        )

        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
        return f'🔄 Background task started: "{preview}"\nTask ID: {task_id}\nYou can keep chatting — results will appear when done.'

    async def _run_background_task(
        self, prompt: str, source: "SessionSource", task_id: str
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        from run_agent import AIAgent

        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return

        _thread_metadata = {"thread_id": source.thread_id} if source.thread_id else None

        try:
            runtime_kwargs = _resolve_runtime_agent_kwargs()
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            # Read model from config via shared helper
            model = _resolve_gateway_model()
            self._apply_runtime_debug_overrides()

            # Determine toolset (same logic as _run_agent)
            default_toolset_map = {
                Platform.LOCAL: "hermes-cli",
                Platform.TELEGRAM: "hermes-telegram",
                Platform.DISCORD: "hermes-discord",
                Platform.WHATSAPP: "hermes-whatsapp",
                Platform.SLACK: "hermes-slack",
                Platform.SIGNAL: "hermes-signal",
                Platform.HOMEASSISTANT: "hermes-homeassistant",
                Platform.EMAIL: "hermes-email",
                Platform.DINGTALK: "hermes-dingtalk",
            }
            platform_toolsets_config = {}
            try:
                config_path = _hermes_home / 'config.yaml'
                if config_path.exists():
                    import yaml
                    with open(config_path, 'r', encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                    platform_toolsets_config = user_config.get("platform_toolsets", {})
            except Exception:
                pass

            platform_config_key = {
                Platform.LOCAL: "cli",
                Platform.TELEGRAM: "telegram",
                Platform.DISCORD: "discord",
                Platform.WHATSAPP: "whatsapp",
                Platform.SLACK: "slack",
                Platform.SIGNAL: "signal",
                Platform.HOMEASSISTANT: "homeassistant",
                Platform.EMAIL: "email",
                Platform.DINGTALK: "dingtalk",
            }.get(source.platform, "telegram")

            config_toolsets = platform_toolsets_config.get(platform_config_key)
            if config_toolsets and isinstance(config_toolsets, list):
                enabled_toolsets = config_toolsets
            else:
                default_toolset = default_toolset_map.get(source.platform, "hermes-telegram")
                enabled_toolsets = [default_toolset]

            platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value

            pr = self._provider_routing
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
            reasoning_config = self._get_effective_reasoning_config()
            self._reasoning_config = reasoning_config
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    reasoning_config=reasoning_config,
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    session_db=self._session_db,
                    fallback_model=self._fallback_model,
                )

                return agent.run_conversation(
                    user_message=prompt,
                    task_id=task_id,
                )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"

            # Extract media files from the response
            if response:
                media_files, response = adapter.extract_media(response)
                images, text_content = adapter.extract_images(response)

                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'

                if text_content:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + text_content,
                        metadata=_thread_metadata,
                    )
                elif not images and not media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + "(No response generated)",
                        metadata=_thread_metadata,
                    )

                # Send extracted images
                for image_url, alt_text in (images or []):
                    try:
                        await adapter.send_image(
                            chat_id=source.chat_id,
                            image_url=image_url,
                            caption=alt_text,
                        )
                    except Exception:
                        pass

                # Send media files
                for media_path in (media_files or []):
                    try:
                        await adapter.send_file(
                            chat_id=source.chat_id,
                            file_path=media_path,
                        )
                    except Exception:
                        pass
            else:
                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f'✅ Background task complete\nPrompt: "{preview}"\n\n(No response generated)',
                    metadata=_thread_metadata,
                )

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )
            except Exception:
                pass

    async def _handle_reasoning_command(self, event: MessageEvent) -> str:
        """Handle /reasoning command — manage reasoning effort and display toggle.

        Usage:
            /reasoning              Show current effort level and display state
            /reasoning <level>      Set reasoning effort (none, low, medium, high, xhigh)
            /reasoning show|on      Show model reasoning in responses
            /reasoning hide|off     Hide model reasoning from responses
        """
        import yaml

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / "config.yaml"
        self._apply_runtime_debug_overrides()

        def _save_config_key(key_path: str, value):
            """Save a dot-separated key to config.yaml."""
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                keys = key_path.split(".")
                current = user_config
                for k in keys[:-1]:
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(user_config, f, default_flow_style=False, sort_keys=False)
                return True
            except Exception as e:
                logger.error("Failed to save config key %s: %s", key_path, e)
                return False

        if not args:
            # Show current state
            rc = self._reasoning_config
            if rc is None:
                level = "medium (default)"
            elif rc.get("enabled") is False:
                level = "none (disabled)"
            else:
                level = rc.get("effort", "medium")
            display_state = "on ✓" if self._show_reasoning else "off"
            return (
                "🧠 **Reasoning Settings**\n\n"
                f"**Effort:** `{level}`\n"
                f"**Display:** {display_state}\n\n"
                "_Usage:_ `/reasoning <none|low|medium|high|xhigh|show|hide>`"
            )

        # Display toggle
        if args in ("show", "on"):
            self._show_reasoning = True
            _save_config_key("display.show_reasoning", True)
            self._apply_runtime_debug_overrides()
            return "🧠 ✓ Reasoning display: **ON**\nModel thinking will be shown before each response."

        if args in ("hide", "off"):
            self._show_reasoning = False
            _save_config_key("display.show_reasoning", False)
            self._apply_runtime_debug_overrides()
            return "🧠 ✓ Reasoning display: **OFF**"

        # Effort level change
        effort = args.strip()
        if effort == "none":
            parsed = {"enabled": False}
        elif effort in ("xhigh", "high", "medium", "low", "minimal"):
            parsed = {"enabled": True, "effort": effort}
        else:
            return (
                f"⚠️ Unknown argument: `{effort}`\n\n"
                "**Valid levels:** none, low, minimal, medium, high, xhigh\n"
                "**Display:** show, hide"
            )

        self._reasoning_config = parsed
        if _save_config_key("agent.reasoning_effort", effort):
            self._apply_runtime_debug_overrides()
            return f"🧠 ✓ Reasoning effort set to `{effort}` (saved to config)\n_(takes effect on next message)_"
        else:
            self._apply_runtime_debug_overrides()
            return f"🧠 ✓ Reasoning effort set to `{effort}` (this session only)"

    async def _handle_think_command(self, event: MessageEvent) -> str:
        """Handle /think command — reasoning effort only, without display toggles."""
        args = event.get_command_args().strip().lower()
        self._apply_runtime_debug_overrides()

        if not args:
            rc = self._reasoning_config
            if rc is None:
                level = "medium (default)"
            elif rc.get("enabled") is False:
                level = "off"
            else:
                level = rc.get("effort", "medium")
            return (
                "🧠 **Thinking Effort**\n\n"
                f"**Current:** `{level}`\n\n"
                "_Usage:_ `/think <off|minimal|low|medium|high|xhigh>`"
            )

        effort = "none" if args == "off" else args
        if effort not in ("none", "minimal", "low", "medium", "high", "xhigh"):
            return (
                f"⚠️ Unknown thinking level: `{args}`\n\n"
                "**Valid levels:** off, minimal, low, medium, high, xhigh"
            )

        think_event = MessageEvent(
            text=f"/reasoning {effort}",
            source=event.source,
            message_type=event.message_type,
            raw_message=event.raw_message,
            message_id=event.message_id,
            media_urls=list(event.media_urls),
            media_types=list(event.media_types),
            reply_to_message_id=event.reply_to_message_id,
            reply_to_text=event.reply_to_text,
            metadata=event.metadata,
        )
        return await self._handle_reasoning_command(think_event)

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context."""
        source = self._command_target_source_for_event(event)
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        instructions = event.get_command_args().strip()

        if not history or len(history) < 4:
            return "Not enough conversation to compress (need at least 4 messages)."

        try:
            from run_agent import AIAgent
            from agent.model_metadata import estimate_messages_tokens_rough

            runtime_kwargs = _resolve_runtime_agent_kwargs()
            if not runtime_kwargs.get("api_key"):
                return "No provider configured -- cannot compress."

            # Resolve model from config (same reason as memory flush above).
            model = _resolve_gateway_model()

            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            original_count = len(msgs)
            approx_tokens = estimate_messages_tokens_rough(msgs)

            tmp_agent = AIAgent(
                **runtime_kwargs,
                model=model,
                max_iterations=4,
                quiet_mode=True,
                enabled_toolsets=["memory"],
                session_id=session_entry.session_id,
            )

            loop = asyncio.get_event_loop()
            compressed, _ = await loop.run_in_executor(
                None,
                lambda: tmp_agent._compress_context(msgs, instructions, approx_tokens=approx_tokens),
            )

            self.session_store.rewrite_transcript(session_entry.session_id, compressed)
            # Reset stored token count — transcript changed, old value is stale
            self.session_store.update_session(
                session_entry.session_key, last_prompt_tokens=0,
            )
            new_count = len(compressed)
            new_tokens = estimate_messages_tokens_rough(compressed)

            result = (
                f"🗜️ Compressed: {original_count} → {new_count} messages\n"
                f"~{approx_tokens:,} → ~{new_tokens:,} tokens"
            )
            if instructions:
                result += f"\nPreserved guidance: `{instructions[:160]}`"
            return result
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return f"Compression failed: {e}"

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_id = session_entry.session_id

        if not self._session_db:
            return "Session database not available."

        title_arg = event.get_command_args().strip()
        if title_arg:
            # Sanitize the title before setting
            try:
                sanitized = self._session_db.sanitize_title(title_arg)
            except ValueError as e:
                return f"⚠️ {e}"
            if not sanitized:
                return "⚠️ Title is empty after cleanup. Please use printable characters."
            # Set the title
            try:
                if self._session_db.set_session_title(session_id, sanitized):
                    return f"✏️ Session title set: **{sanitized}**"
                else:
                    return "Session not found in database."
            except ValueError as e:
                return f"⚠️ {e}"
        else:
            # Show the current title and session ID
            title = self._session_db.get_session_title(session_id)
            if title:
                return f"📌 Session: `{session_id}`\nTitle: **{title}**"
            else:
                return f"📌 Session: `{session_id}`\nNo title set. Usage: `/title My Session Name`"

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — switch to a previously-named session."""
        if not self._session_db:
            return "Session database not available."

        source = event.source
        session_key = self._session_key_for_source(source)
        name = event.get_command_args().strip()

        if not name:
            # List recent titled sessions for this user/platform
            try:
                user_source = source.platform.value if source.platform else None
                sessions = self._session_db.list_sessions_rich(
                    source=user_source, limit=10
                )
                titled = [s for s in sessions if s.get("title")]
                if not titled:
                    return (
                        "No named sessions found.\n"
                        "Use `/title My Session` to name your current session, "
                        "then `/resume My Session` to return to it later."
                    )
                lines = ["📋 **Named Sessions**\n"]
                for s in titled[:10]:
                    title = s["title"]
                    preview = s.get("preview", "")[:40]
                    preview_part = f" — _{preview}_" if preview else ""
                    lines.append(f"• **{title}**{preview_part}")
                lines.append("\nUsage: `/resume <session name>`")
                return "\n".join(lines)
            except Exception as e:
                logger.debug("Failed to list titled sessions: %s", e)
                return f"Could not list sessions: {e}"

        # Resolve the name to a session ID
        target_id = self._session_db.resolve_session_by_title(name)
        if not target_id:
            return (
                f"No session found matching '**{name}**'.\n"
                "Use `/resume` with no arguments to see available sessions."
            )

        # Check if already on that session
        current_entry = self.session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return f"📌 Already on session **{name}**."

        # Flush memories for current session before switching
        try:
            asyncio.create_task(
                self._async_flush_memories(current_entry.session_id, session_key)
            )
        except Exception as e:
            logger.debug("Memory flush on resume failed: %s", e)

        self._shutdown_gateway_honcho(session_key)

        # Clear any running agent for this session key
        if session_key in self._running_agents:
            del self._running_agents[session_key]

        # Switch the session entry to point at the old session
        new_entry = self.session_store.switch_session(session_key, target_id)
        if not new_entry:
            return "Failed to switch session."

        # Get the title for confirmation
        title = self._session_db.get_session_title(target_id) or name

        # Count messages for context
        history = self.session_store.load_transcript(target_id)
        msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
        msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""

        return f"↻ Resumed session **{title}**{msg_part}. Conversation restored."

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the session's last agent run."""
        source = event.source
        session_key = self._session_key_for_source(source)

        agent = self._running_agents.get(session_key)
        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = [
                "📊 **Session Token Usage**",
                f"Prompt (input): {agent.session_prompt_tokens:,}",
                f"Completion (output): {agent.session_completion_tokens:,}",
                f"Total: {agent.session_total_tokens:,}",
                f"API calls: {agent.session_api_calls}",
            ]
            ctx = agent.context_compressor
            if ctx.last_prompt_tokens:
                pct = ctx.last_prompt_tokens / ctx.context_length * 100 if ctx.context_length else 0
                lines.append(f"Context: {ctx.last_prompt_tokens:,} / {ctx.context_length:,} ({pct:.0f}%)")
            if ctx.compression_count:
                lines.append(f"Compressions: {ctx.compression_count}")
            return "\n".join(lines)

        # No running agent -- check session history for a rough count
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough
            msgs = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
            approx = estimate_messages_tokens_rough(msgs)
            return (
                f"📊 **Session Info**\n"
                f"Messages: {len(msgs)}\n"
                f"Estimated context: ~{approx:,} tokens\n"
                f"_(Detailed usage available during active conversations)_"
            )
        return "No usage data available for this session."

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        import asyncio as _asyncio

        args = event.get_command_args().strip()
        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return f"Invalid --days value: {parts[i + 1]}"
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

            loop = _asyncio.get_event_loop()

            def _run_insights():
                db = SessionDB()
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                result = engine.format_gateway(report)
                db.close()
                return result

            return await loop.run_in_executor(None, _run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return f"Error generating insights: {e}"

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> str:
        """Handle /reload-mcp command -- disconnect and reconnect all MCP servers."""
        loop = asyncio.get_event_loop()
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _load_mcp_config, _servers, _lock

            # Capture old server names before shutdown
            with _lock:
                old_servers = set(_servers.keys())

            # Read new config before shutting down, so we know what will be added/removed
            new_config = _load_mcp_config()
            new_server_names = set(new_config.keys())

            # Shutdown existing connections
            await loop.run_in_executor(None, shutdown_mcp_servers)

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await loop.run_in_executor(None, discover_mcp_tools)

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = ["🔄 **MCP Servers Reloaded**\n"]
            if reconnected:
                lines.append(f"♻️ Reconnected: {', '.join(sorted(reconnected))}")
            if added:
                lines.append(f"➕ Added: {', '.join(sorted(added))}")
            if removed:
                lines.append(f"➖ Removed: {', '.join(sorted(removed))}")
            if not connected_servers:
                lines.append("No MCP servers connected.")
            else:
                lines.append(f"\n🔧 {len(new_tools)} tool(s) available from {len(connected_servers)} server(s)")

            # Inject a message at the END of the session history so the
            # model knows tools changed on its next turn.  Appended after
            # all existing messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[SYSTEM: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = self.session_store.get_or_create_session(event.source)
                self.session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return f"❌ MCP reload failed: {e}"

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command for pending exec approvals."""
        approval_id = event.get_command_args().strip() or None
        return self._resolve_pending_approval(
            decision="deny",
            approval_id=approval_id,
            source=self._command_target_source_for_event(event),
        )

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update Hermes Agent to the latest version.

        Spawns ``hermes update`` in a separate systemd scope so it survives the
        gateway restart that ``hermes update`` may trigger at the end. Marker
        files are written so either the current gateway process or the next one
        can notify the user when the update finishes.
        """
        import json
        import shutil
        import subprocess
        from datetime import datetime

        project_root = Path(__file__).parent.parent.resolve()
        git_dir = project_root / '.git'

        if not git_dir.exists():
            return "✗ Not a git repository — cannot update."

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            return (
                "✗ Could not locate the `hermes` command. "
                "Hermes is running, but the update command could not find the "
                "executable on PATH or via the current Python interpreter. "
                "Try running `hermes update` manually in your terminal."
            )

        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        pending = {
            "platform": event.source.platform.value,
            "chat_id": event.source.chat_id,
            "user_id": event.source.user_id,
            "timestamp": datetime.now().isoformat(),
        }
        pending_path.write_text(json.dumps(pending))
        exit_code_path.unlink(missing_ok=True)

        # Spawn `hermes update` in a separate cgroup so it survives gateway
        # restart. systemd-run --user --scope creates a transient scope unit.
        hermes_cmd_str = " ".join(shlex.quote(part) for part in hermes_cmd)
        update_cmd = (
            f"{hermes_cmd_str} update > {shlex.quote(str(output_path))} 2>&1; "
            f"status=$?; printf '%s' \"$status\" > {shlex.quote(str(exit_code_path))}"
        )
        try:
            systemd_run = shutil.which("systemd-run")
            if systemd_run:
                subprocess.Popen(
                    [systemd_run, "--user", "--scope",
                     "--unit=hermes-update", "--",
                     "bash", "-c", update_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                # Fallback: best-effort detach with start_new_session
                subprocess.Popen(
                    ["bash", "-c", f"nohup {update_cmd} &"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except Exception as e:
            pending_path.unlink(missing_ok=True)
            exit_code_path.unlink(missing_ok=True)
            return f"✗ Failed to start update: {e}"

        self._schedule_update_notification_watch()
        return "⚕ Starting Hermes update… I'll notify you when it's done."

    def _schedule_update_notification_watch(self) -> None:
        """Ensure a background task is watching for update completion."""
        existing_task = getattr(self, "_update_notification_task", None)
        if existing_task and not existing_task.done():
            return

        try:
            self._update_notification_task = asyncio.create_task(
                self._watch_for_update_completion()
            )
        except RuntimeError:
            logger.debug("Skipping update notification watcher: no running event loop")

    async def _watch_for_update_completion(
        self,
        poll_interval: float = 2.0,
        timeout: float = 1800.0,
    ) -> None:
        """Wait for ``hermes update`` to finish, then send its notification."""
        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        exit_code_path = _hermes_home / ".update_exit_code"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while (pending_path.exists() or claimed_path.exists()) and loop.time() < deadline:
            if exit_code_path.exists():
                await self._send_update_notification()
                return
            await asyncio.sleep(poll_interval)

        if (pending_path.exists() or claimed_path.exists()) and not exit_code_path.exists():
            logger.warning("Update watcher timed out waiting for completion marker")
            exit_code_path.write_text("124")
            await self._send_update_notification()

    async def _send_update_notification(self) -> bool:
        """If an update finished, notify the user.

        Returns False when the update is still running so a caller can retry
        later. Returns True after a definitive send/skip decision.
        """
        import json
        import re as _re

        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"

        if not pending_path.exists() and not claimed_path.exists():
            return False

        cleanup = True
        active_pending_path = claimed_path
        try:
            if pending_path.exists():
                try:
                    pending_path.replace(claimed_path)
                except FileNotFoundError:
                    if not claimed_path.exists():
                        return True
            elif not claimed_path.exists():
                return True

            pending = json.loads(claimed_path.read_text())
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")

            if not exit_code_path.exists():
                logger.info("Update notification deferred: update still running")
                cleanup = False
                active_pending_path = pending_path
                claimed_path.replace(pending_path)
                return False

            exit_code_raw = exit_code_path.read_text().strip() or "1"
            exit_code = int(exit_code_raw)

            # Read the captured update output
            output = ""
            if output_path.exists():
                output = output_path.read_text()

            # Resolve adapter
            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)

            if adapter and chat_id:
                # Strip ANSI escape codes for clean display
                output = _re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
                if output:
                    if len(output) > 3500:
                        output = "…" + output[-3500:]
                    if exit_code == 0:
                        msg = f"✅ Hermes update finished.\n\n```\n{output}\n```"
                    else:
                        msg = f"❌ Hermes update failed.\n\n```\n{output}\n```"
                else:
                    if exit_code == 0:
                        msg = "✅ Hermes update finished successfully."
                    else:
                        msg = "❌ Hermes update failed. Check the gateway logs or run `hermes update` manually for details."
                await adapter.send(chat_id, msg)
                logger.info(
                    "Sent post-update notification to %s:%s (exit=%s)",
                    platform_str,
                    chat_id,
                    exit_code,
                )
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            if cleanup:
                active_pending_path.unlink(missing_ok=True)
                claimed_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                exit_code_path.unlink(missing_ok=True)

        return True

    def _set_session_env(self, context: SessionContext) -> None:
        """Set environment variables for the current session."""
        os.environ["HERMES_SESSION_PLATFORM"] = context.source.platform.value
        os.environ["HERMES_SESSION_CHAT_ID"] = context.source.chat_id
        if context.source.chat_name:
            os.environ["HERMES_SESSION_CHAT_NAME"] = context.source.chat_name
        if context.source.thread_id:
            os.environ["HERMES_SESSION_THREAD_ID"] = str(context.source.thread_id)
    
    def _clear_session_env(self) -> None:
        """Clear session environment variables."""
        for var in [
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_SESSION_THREAD_ID",
            "HERMES_SESSION_SEND_POLICY",
        ]:
            if var in os.environ:
                del os.environ[var]
    
    async def _enrich_message_with_vision(
        self,
        user_text: str,
        image_paths: List[str],
    ) -> str:
        """
        Auto-analyze user-attached images with the vision tool and prepend
        the descriptions to the message text.

        Each image is analyzed with a general-purpose prompt.  The resulting
        description *and* the local cache path are injected so the model can:
          1. Immediately understand what the user sent (no extra tool call).
          2. Re-examine the image with vision_analyze if it needs more detail.

        Args:
            user_text:   The user's original caption / message text.
            image_paths: List of local file paths to cached images.

        Returns:
            The enriched message string with vision descriptions prepended.
        """
        from tools.vision_tools import vision_analyze_tool
        import json as _json

        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        enriched_parts = []
        for path in image_paths:
            try:
                logger.debug("Auto-analyzing user image: %s", path)
                result_json = await vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                )
                result = _json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    enriched_parts.append(
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as e:
                logger.error("Vision auto-analysis error: %s", e)
                enriched_parts.append(
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )

        # Combine: vision descriptions first, then the user's original text
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> str:
        """
        Auto-transcribe user voice/audio messages using the configured STT provider
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            The enriched message string with transcriptions prepended.
        """
        if not getattr(self.config, "stt_enabled", True):
            disabled_note = "[The user sent voice message(s), but transcription is disabled in config."
            if self._has_setup_skill():
                disabled_note += (
                    " You have a skill called hermes-agent-setup that can help "
                    "users configure Hermes features including voice, tools, and more."
                )
            disabled_note += "]"
            if user_text:
                return f"{disabled_note}\n\n{user_text}"
            return disabled_note

        from tools.transcription_tools import transcribe_audio, get_stt_model_from_config
        import asyncio

        stt_model = get_stt_model_from_config()

        enriched_parts = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(transcribe_audio, path, model=stt_model)
                if result["success"]:
                    transcript = result["transcript"]
                    enriched_parts.append(
                        f'[The user sent a voice message~ '
                        f'Here\'s what they said: "{transcript}"]'
                    )
                else:
                    error = result.get("error", "unknown error")
                    if (
                        "No STT provider" in error
                        or error.startswith("Neither VOICE_TOOLS_OPENAI_KEY nor OPENAI_API_KEY is set")
                    ):
                        _no_stt_note = (
                            "[The user sent a voice message but I can't listen "
                            "to it right now — no STT provider is configured. "
                            "A direct message has already been sent to the user "
                            "with setup instructions."
                        )
                        if self._has_setup_skill():
                            _no_stt_note += (
                                " You have a skill called hermes-agent-setup "
                                "that can help users configure Hermes features "
                                "including voice, tools, and more."
                            )
                        _no_stt_note += "]"
                        enriched_parts.append(_no_stt_note)
                    else:
                        enriched_parts.append(
                            "[The user sent a voice message but I had trouble "
                            f"transcribing it~ ({error})]"
                        )
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append(
                    "[The user sent a voice message but something went wrong "
                    "when I tried to listen to it~ Let them know!]"
                )

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.

        Notification mode (from ``display.background_process_notifications``):
          - ``all``    — running-output updates + final message
          - ``result`` — final completion message only
          - ``error``  — final message only when exit code != 0
          - ``off``    — no messages at all
        """
        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")
        thread_id = watcher.get("thread_id", "")
        notify_mode = self._get_effective_background_notifications_mode()

        logger.debug("Process watcher started: %s (every %ss, notify=%s)",
                      session_id, interval, notify_mode)

        if notify_mode == "off":
            # Still wait for the process to exit so we can log it, but don't
            # push any messages to the user.
            while True:
                await asyncio.sleep(interval)
                session = process_registry.get(session_id)
                if session is None or session.exited:
                    break
            logger.debug("Process watcher ended (silent): %s", session_id)
            return

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # Decide whether to notify based on mode
                should_notify = (
                    notify_mode in ("all", "result")
                    or (notify_mode == "error" and session.exit_code not in (0, None))
                )
                if should_notify:
                    new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                    message_text = (
                        f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                        f"Here's the final output:\n{new_output}]"
                    )
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter and chat_id:
                        try:
                            send_meta = {"thread_id": thread_id} if thread_id else None
                            await adapter.send(chat_id, message_text, metadata=send_meta)
                        except Exception as e:
                            logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output and notify_mode == "all":
                # New output available -- deliver status update (only in "all" mode)
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        send_meta = {"thread_id": thread_id} if thread_id else None
                        await adapter.send(chat_id, message_text, metadata=send_meta)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)

    _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling (#816)

    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        _interrupt_depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the agent with the given message and context.
        
        Returns the full result dict from run_conversation, including:
          - "final_response": str (the text to send back)
          - "messages": list (full conversation including tool calls)
          - "api_calls": int
          - "completed": bool
        
        This is run in a thread pool to not block the event loop.
        Supports interruption via new messages.
        """
        from run_agent import AIAgent
        import queue
        
        # Determine toolset based on platform.
        # Check config.yaml for per-platform overrides, fallback to hardcoded defaults.
        default_toolset_map = {
            Platform.LOCAL: "hermes-cli",
            Platform.TELEGRAM: "hermes-telegram",
            Platform.DISCORD: "hermes-discord",
            Platform.WHATSAPP: "hermes-whatsapp",
            Platform.SLACK: "hermes-slack",
            Platform.SIGNAL: "hermes-signal",
            Platform.HOMEASSISTANT: "hermes-homeassistant",
            Platform.EMAIL: "hermes-email",
            Platform.DINGTALK: "hermes-dingtalk",
        }

        # Try to load platform_toolsets from config
        platform_toolsets_config = {}
        try:
            config_path = _hermes_home / 'config.yaml'
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                platform_toolsets_config = user_config.get("platform_toolsets", {})
        except Exception as e:
            logger.debug("Could not load platform_toolsets config: %s", e)

        # Map platform enum to config key
        platform_config_key = {
            Platform.LOCAL: "cli",
            Platform.TELEGRAM: "telegram",
            Platform.DISCORD: "discord",
            Platform.WHATSAPP: "whatsapp",
            Platform.SLACK: "slack",
            Platform.SIGNAL: "signal",
            Platform.HOMEASSISTANT: "homeassistant",
            Platform.EMAIL: "email",
            Platform.DINGTALK: "dingtalk",
        }.get(source.platform, "telegram")
        
        # Use config override if present (list of toolsets), otherwise hardcoded default
        config_toolsets = platform_toolsets_config.get(platform_config_key)
        if config_toolsets and isinstance(config_toolsets, list):
            enabled_toolsets = config_toolsets
        else:
            default_toolset = default_toolset_map.get(source.platform, "hermes-telegram")
            enabled_toolsets = [default_toolset]
        
        # Tool progress mode from config.yaml: "all", "new", "verbose", "off"
        # Falls back to env vars for backward compatibility
        _progress_cfg = {}
        try:
            _tp_cfg_path = _hermes_home / "config.yaml"
            if _tp_cfg_path.exists():
                import yaml as _tp_yaml
                with open(_tp_cfg_path, encoding="utf-8") as _tp_f:
                    _tp_data = _tp_yaml.safe_load(_tp_f) or {}
                _progress_cfg = _tp_data.get("display", {})
        except Exception:
            pass
        progress_mode = (
            _progress_cfg.get("tool_progress")
            or os.getenv("HERMES_TOOL_PROGRESS_MODE")
            or "all"
        )
        tool_progress_enabled = progress_mode != "off"
        
        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if tool_progress_enabled else None
        last_tool = [None]  # Mutable container for tracking in closure
        last_progress_msg = [None]  # Track last message for dedup
        repeat_count = [0]  # How many times the same message repeated
        
        def progress_callback(tool_name: str, preview: str = None, args: dict = None):
            """Callback invoked by agent when a tool is called."""
            if not progress_queue:
                return
            
            # "new" mode: only report when tool changes
            if progress_mode == "new" and tool_name == last_tool[0]:
                return
            last_tool[0] = tool_name
            
            # Build progress message with primary argument preview
            from agent.display import get_tool_emoji
            emoji = get_tool_emoji(tool_name, default="⚙️")
            
            # Verbose mode: show detailed arguments
            if progress_mode == "verbose" and args:
                import json as _json
                args_str = _json.dumps(args, ensure_ascii=False, default=str)
                if len(args_str) > 200:
                    args_str = args_str[:197] + "..."
                msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
                progress_queue.put(msg)
                return
            
            if preview:
                # Truncate preview to keep messages clean
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                msg = f"{emoji} {tool_name}: \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            
            # Dedup: collapse consecutive identical progress messages.
            # Common with execute_code where models iterate with the same
            # code (same boilerplate imports → identical previews).
            if msg == last_progress_msg[0]:
                repeat_count[0] += 1
                # Update the last line in progress_lines with a counter
                # via a special "dedup" queue message.
                progress_queue.put(("__dedup__", msg, repeat_count[0]))
                return
            last_progress_msg[0] = msg
            repeat_count[0] = 0
            
            progress_queue.put(msg)
        
        # Background task to send progress messages
        # Accumulates tool lines into a single message that gets edited
        _progress_metadata = {"thread_id": source.thread_id} if source.thread_id else None

        async def send_progress_messages():
            if not progress_queue:
                return

            adapter = self.adapters.get(source.platform)
            if not adapter:
                return

            progress_lines = []      # Accumulated tool lines
            progress_msg_id = None   # ID of the progress message to edit
            can_edit = True          # False once an edit fails (platform doesn't support it)

            while True:
                try:
                    raw = progress_queue.get_nowait()
                    
                    # Handle dedup messages: update last line with repeat counter
                    if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                        _, base_msg, count = raw
                        if progress_lines:
                            progress_lines[-1] = f"{base_msg} (×{count + 1})"
                        msg = progress_lines[-1] if progress_lines else base_msg
                    else:
                        msg = raw
                        progress_lines.append(msg)

                    if can_edit and progress_msg_id is not None:
                        # Try to edit the existing progress message
                        full_text = "\n".join(progress_lines)
                        result = await adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=progress_msg_id,
                            content=full_text,
                        )
                        if not result.success:
                            # Platform doesn't support editing — stop trying,
                            # send just this new line as a separate message
                            can_edit = False
                            await adapter.send(chat_id=source.chat_id, content=msg, metadata=_progress_metadata)
                    else:
                        if can_edit:
                            # First tool: send all accumulated text as new message
                            full_text = "\n".join(progress_lines)
                            result = await adapter.send(chat_id=source.chat_id, content=full_text, metadata=_progress_metadata)
                        else:
                            # Editing unsupported: send just this line
                            result = await adapter.send(chat_id=source.chat_id, content=msg, metadata=_progress_metadata)
                        if result.success and result.message_id:
                            progress_msg_id = result.message_id

                    # Restore typing indicator
                    await asyncio.sleep(0.3)
                    await adapter.send_typing(source.chat_id, metadata=_progress_metadata)

                except queue.Empty:
                    await asyncio.sleep(0.3)
                except asyncio.CancelledError:
                    # Drain remaining queued messages
                    while not progress_queue.empty():
                        try:
                            raw = progress_queue.get_nowait()
                            if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                                _, base_msg, count = raw
                                if progress_lines:
                                    progress_lines[-1] = f"{base_msg} (×{count + 1})"
                            else:
                                progress_lines.append(raw)
                        except Exception:
                            break
                    # Final edit with all remaining tools (only if editing works)
                    if can_edit and progress_lines and progress_msg_id:
                        full_text = "\n".join(progress_lines)
                        try:
                            await adapter.edit_message(
                                chat_id=source.chat_id,
                                message_id=progress_msg_id,
                                content=full_text,
                            )
                        except Exception:
                            pass
                    return
                except Exception as e:
                    logger.error("Progress message error: %s", e)
                    await asyncio.sleep(1)
        
        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        stream_consumer_holder = [None]  # Mutable container for stream consumer
        
        # Bridge sync step_callback → async hooks.emit for agent:step events
        _loop_for_step = asyncio.get_event_loop()
        _hooks_ref = self.hooks

        def _step_callback_sync(iteration: int, tool_names: list) -> None:
            try:
                asyncio.run_coroutine_threadsafe(
                    _hooks_ref.emit("agent:step", {
                        "platform": source.platform.value if source.platform else "",
                        "user_id": source.user_id,
                        "session_id": session_id,
                        "iteration": iteration,
                        "tool_names": tool_names,
                    }),
                    _loop_for_step,
                )
            except Exception as _e:
                logger.debug("agent:step hook error: %s", _e)

        # Bridge sync status_callback → async adapter.send for context pressure
        _status_adapter = self.adapters.get(source.platform)
        _status_chat_id = source.chat_id
        _status_thread_metadata = {"thread_id": source.thread_id} if source.thread_id else None

        def _status_callback_sync(event_type: str, message: str) -> None:
            if not _status_adapter:
                return
            try:
                asyncio.run_coroutine_threadsafe(
                    _status_adapter.send(
                        _status_chat_id,
                        message,
                        metadata=_status_thread_metadata,
                    ),
                    _loop_for_step,
                )
            except Exception as _e:
                logger.debug("status_callback error (%s): %s", event_type, _e)

        def run_sync():
            # Pass session_key to process registry via env var so background
            # processes can be mapped back to this gateway session
            os.environ["HERMES_SESSION_KEY"] = session_key or ""
            os.environ["HERMES_SESSION_SEND_POLICY"] = self._session_send_policy(session_key)

            # Read from env var or use default (same as CLI)
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
            
            # Map platform enum to the platform hint key the agent understands.
            # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
            platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value
            
            # Combine platform context with user-configured ephemeral system prompt
            combined_ephemeral = context_prompt or ""
            if self._ephemeral_system_prompt:
                combined_ephemeral = (combined_ephemeral + "\n\n" + self._ephemeral_system_prompt).strip()

            # Re-read .env and config for fresh credentials (gateway is long-lived,
            # keys may change without restart).
            try:
                load_dotenv(_env_path, override=True, encoding="utf-8")
            except UnicodeDecodeError:
                load_dotenv(_env_path, override=True, encoding="latin-1")
            except Exception:
                pass

            model = _resolve_gateway_model()

            try:
                runtime_kwargs = _resolve_runtime_agent_kwargs()
            except Exception as exc:
                return {
                    "final_response": f"⚠️ Provider authentication failed: {exc}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }

            self._apply_runtime_debug_overrides()
            pr = self._provider_routing
            honcho_manager, honcho_config = self._get_or_create_gateway_honcho(session_key)
            reasoning_config = self._get_effective_reasoning_config()
            self._reasoning_config = reasoning_config
            # Set up streaming consumer if enabled
            _stream_consumer = None
            _stream_delta_cb = None
            _scfg = getattr(getattr(self, 'config', None), 'streaming', None)
            if _scfg is None:
                from gateway.config import StreamingConfig
                _scfg = StreamingConfig()

            if _scfg.enabled and _scfg.transport != "off":
                try:
                    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
                    _adapter = self.adapters.get(source.platform)
                    if _adapter:
                        _consumer_cfg = StreamConsumerConfig(
                            edit_interval=_scfg.edit_interval,
                            buffer_threshold=_scfg.buffer_threshold,
                            cursor=_scfg.cursor,
                        )
                        _stream_consumer = GatewayStreamConsumer(
                            adapter=_adapter,
                            chat_id=source.chat_id,
                            config=_consumer_cfg,
                            metadata={"thread_id": source.thread_id} if source.thread_id else None,
                        )
                        _stream_delta_cb = _stream_consumer.on_delta
                        stream_consumer_holder[0] = _stream_consumer
                except Exception as _sc_err:
                    logger.debug("Could not set up stream consumer: %s", _sc_err)

            turn_route = self._resolve_turn_agent_config(message, model, runtime_kwargs)
            agent = AIAgent(
                model=turn_route["model"],
                **turn_route["runtime"],
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=enabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=self._prefill_messages or None,
                reasoning_config=reasoning_config,
                providers_allowed=pr.get("only"),
                providers_ignored=pr.get("ignore"),
                providers_order=pr.get("order"),
                provider_sort=pr.get("sort"),
                provider_require_parameters=pr.get("require_parameters", False),
                provider_data_collection=pr.get("data_collection"),
                session_id=session_id,
                tool_progress_callback=progress_callback if tool_progress_enabled else None,
                step_callback=_step_callback_sync if _hooks_ref.loaded_hooks else None,
                stream_delta_callback=_stream_delta_cb,
                status_callback=_status_callback_sync,
                platform=platform_key,
                honcho_session_key=session_key,
                honcho_manager=honcho_manager,
                honcho_config=honcho_config,
                session_db=self._session_db,
                fallback_model=self._fallback_model,
            )
            self._attach_subagent_runtime_hooks(agent, session_key)
            
            # Store agent reference for interrupt support
            agent_holder[0] = agent
            # Capture the full tool definitions for transcript logging
            tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None
            
            # Convert history to agent format.
            # Two cases:
            #   1. Normal path (from transcript): simple {role, content, timestamp} dicts
            #      - Strip timestamps, keep role+content
            #   2. Interrupt path (from agent result["messages"]): full agent messages
            #      that may include tool_calls, tool_call_id, reasoning, etc.
            #      - These must be passed through intact so the API sees valid
            #        assistant→tool sequences (dropping tool_calls causes 500 errors)
            agent_history = []
            for msg in history:
                role = msg.get("role")
                if not role:
                    continue
                
                # Skip metadata entries (tool definitions, session info)
                # -- these are for transcript logging, not for the LLM
                if role in ("session_meta",):
                    continue
                
                # Skip system messages -- the agent rebuilds its own system prompt
                if role == "system":
                    continue
                
                # Rich agent messages (tool_calls, tool results) must be passed
                # through intact so the API sees valid assistant→tool sequences
                has_tool_calls = "tool_calls" in msg
                has_tool_call_id = "tool_call_id" in msg
                is_tool_message = role == "tool"
                
                if has_tool_calls or has_tool_call_id or is_tool_message:
                    clean_msg = {k: v for k, v in msg.items() if k != "timestamp"}
                    agent_history.append(clean_msg)
                else:
                    # Simple text message - just need role and content
                    content = msg.get("content")
                    if content:
                        # Tag cross-platform mirror messages so the agent knows their origin
                        if msg.get("mirror"):
                            mirror_src = msg.get("mirror_source", "another session")
                            content = f"[Delivered from {mirror_src}] {content}"
                        agent_history.append({"role": role, "content": content})
            
            # Collect MEDIA paths already in history so we can exclude them
            # from the current turn's extraction. This is compression-safe:
            # even if the message list shrinks, we know which paths are old.
            _history_media_paths: set = set()
            for _hm in agent_history:
                if _hm.get("role") in ("tool", "function"):
                    _hc = _hm.get("content", "")
                    if "MEDIA:" in _hc:
                        for _match in re.finditer(r'MEDIA:(\S+)', _hc):
                            _p = _match.group(1).strip().rstrip('",}')
                            if _p:
                                _history_media_paths.add(_p)
            
            result = agent.run_conversation(message, conversation_history=agent_history, task_id=session_id)
            result_holder[0] = result

            # Signal the stream consumer that the agent is done
            if _stream_consumer is not None:
                _stream_consumer.finish()
            
            # Return final response, or a message if something went wrong
            final_response = result.get("final_response")

            # Extract actual token counts from the agent instance used for this run
            _last_prompt_toks = 0
            _input_toks = 0
            _output_toks = 0
            _agent = agent_holder[0]
            if _agent and hasattr(_agent, "context_compressor"):
                _last_prompt_toks = getattr(_agent.context_compressor, "last_prompt_tokens", 0)
                _input_toks = getattr(_agent, "session_prompt_tokens", 0)
                _output_toks = getattr(_agent, "session_completion_tokens", 0)
            _resolved_model = getattr(_agent, "model", None) if _agent else None

            if not final_response:
                error_msg = f"⚠️ {result['error']}" if result.get("error") else "(No response generated)"
                return {
                    "final_response": error_msg,
                    "messages": result.get("messages", []),
                    "api_calls": result.get("api_calls", 0),
                    "tools": tools_holder[0] or [],
                    "history_offset": len(agent_history),
                    "last_prompt_tokens": _last_prompt_toks,
                    "input_tokens": _input_toks,
                    "output_tokens": _output_toks,
                    "model": _resolved_model,
                }
            
            # Scan tool results for MEDIA:<path> tags that need to be delivered
            # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
            # in its JSON response, but the model's final text reply usually
            # doesn't include them.  We collect unique tags from tool results and
            # append any that aren't already present in the final response, so the
            # adapter's extract_media() can find and deliver the files exactly once.
            #
            # Uses path-based deduplication against _history_media_paths (collected
            # before run_conversation) instead of index slicing. This is safe even
            # when context compression shrinks the message list. (Fixes #160)
            if "MEDIA:" not in final_response:
                media_tags = []
                has_voice_directive = False
                for msg in result.get("messages", []):
                    if msg.get("role") in ("tool", "function"):
                        content = msg.get("content", "")
                        if "MEDIA:" in content:
                            for match in re.finditer(r'MEDIA:(\S+)', content):
                                path = match.group(1).strip().rstrip('",}')
                                if path and path not in _history_media_paths:
                                    media_tags.append(f"MEDIA:{path}")
                            if "[[audio_as_voice]]" in content:
                                has_voice_directive = True
                
                if media_tags:
                    seen = set()
                    unique_tags = []
                    for tag in media_tags:
                        if tag not in seen:
                            seen.add(tag)
                            unique_tags.append(tag)
                    if has_voice_directive:
                        unique_tags.insert(0, "[[audio_as_voice]]")
                    final_response = final_response + "\n" + "\n".join(unique_tags)
            
            # Sync session_id: the agent may have created a new session during
            # mid-run context compression (_compress_context splits sessions).
            # If so, update the session store entry so the NEXT message loads
            # the compressed transcript, not the stale pre-compression one.
            agent = agent_holder[0]
            if agent and session_key and hasattr(agent, 'session_id') and agent.session_id != session_id:
                logger.info(
                    "Session split detected: %s → %s (compression)",
                    session_id, agent.session_id,
                )
                entry = self.session_store._entries.get(session_key)
                if entry:
                    entry.session_id = agent.session_id
                    self.session_store._save()

            effective_session_id = getattr(agent, 'session_id', session_id) if agent else session_id

            # Auto-generate session title after first exchange (non-blocking)
            if final_response and self._session_db:
                try:
                    from agent.title_generator import maybe_auto_title
                    all_msgs = result_holder[0].get("messages", []) if result_holder[0] else []
                    maybe_auto_title(
                        self._session_db,
                        effective_session_id,
                        message,
                        final_response,
                        all_msgs,
                    )
                except Exception:
                    pass

            return {
                "final_response": final_response,
                "last_reasoning": result.get("last_reasoning"),
                "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                "api_calls": result_holder[0].get("api_calls", 0) if result_holder[0] else 0,
                "tools": tools_holder[0] or [],
                "history_offset": len(agent_history),
                "last_prompt_tokens": _last_prompt_toks,
                "input_tokens": _input_toks,
                "output_tokens": _output_toks,
                "model": _resolved_model,
                "session_id": effective_session_id,
            }
        
        # Start progress message sender if enabled
        progress_task = None
        if tool_progress_enabled:
            progress_task = asyncio.create_task(send_progress_messages())

        # Start stream consumer task — polls for consumer creation since it
        # happens inside run_sync (thread pool) after the agent is constructed.
        stream_task = None

        async def _start_stream_consumer():
            """Wait for the stream consumer to be created, then run it."""
            for _ in range(200):  # Up to 10s wait
                if stream_consumer_holder[0] is not None:
                    await stream_consumer_holder[0].run()
                    return
                await asyncio.sleep(0.05)

        stream_task = asyncio.create_task(_start_stream_consumer())
        
        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        async def track_agent():
            # Wait for agent to be created
            while agent_holder[0] is None:
                await asyncio.sleep(0.05)
            if session_key:
                self._running_agents[session_key] = agent_holder[0]
        
        tracking_task = asyncio.create_task(track_agent())
        
        # Monitor for interrupts from the adapter (new messages arriving)
        async def monitor_for_interrupt():
            adapter = self.adapters.get(source.platform)
            if not adapter or not session_key:
                return
            
            while True:
                await asyncio.sleep(0.2)  # Check every 200ms
                # Check if adapter has a pending interrupt for this session.
                # Must use session_key (build_session_key output) — NOT
                # source.chat_id — because the adapter stores interrupt events
                # under the full session key.
                if hasattr(adapter, 'has_pending_interrupt') and adapter.has_pending_interrupt(session_key):
                    agent = agent_holder[0]
                    if agent:
                        pending_event = adapter.get_pending_message(session_key)
                        pending_text = pending_event.text if pending_event else None
                        logger.debug("Interrupt detected from adapter, signaling agent...")
                        agent.interrupt(pending_text)
                        break
        
        interrupt_monitor = asyncio.create_task(monitor_for_interrupt())
        
        try:
            # Run in thread pool to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, run_sync)

            # Track fallback model state: if the agent switched to a
            # fallback model during this run, persist it so /model shows
            # the actually-active model instead of the config default.
            _agent = agent_holder[0]
            if _agent is not None and hasattr(_agent, 'model'):
                _cfg_model = _resolve_gateway_model()
                if _agent.model != _cfg_model:
                    self._effective_model = _agent.model
                    self._effective_provider = getattr(_agent, 'provider', None)
                else:
                    # Primary model worked — clear any stale fallback state
                    self._effective_model = None
                    self._effective_provider = None

            # Check if we were interrupted and have a pending message
            result = result_holder[0]
            adapter = self.adapters.get(source.platform)
            
            # Get pending message from adapter if interrupted.
            # Use session_key (not source.chat_id) to match adapter's storage keys.
            pending = None
            if result and result.get("interrupted") and adapter:
                pending_event = adapter.get_pending_message(session_key) if session_key else None
                if pending_event:
                    pending = pending_event.text
                elif result.get("interrupt_message"):
                    pending = result.get("interrupt_message")
            
            if pending:
                logger.debug("Processing interrupted message: '%s...'", pending[:40])
                
                # Clear the adapter's interrupt event so the next _run_agent call
                # doesn't immediately re-trigger the interrupt before the new agent
                # even makes its first API call (this was causing an infinite loop).
                if adapter and hasattr(adapter, '_active_sessions') and session_key and session_key in adapter._active_sessions:
                    adapter._active_sessions[session_key].clear()
                
                # Cap recursion depth to prevent resource exhaustion when the
                # user sends multiple messages while the agent keeps failing. (#816)
                if _interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
                    logger.warning(
                        "Interrupt recursion depth %d reached for session %s — "
                        "queueing message instead of recursing.",
                        _interrupt_depth, session_key,
                    )
                    # Queue the pending message for normal processing on next turn
                    adapter = self.adapters.get(source.platform)
                    if adapter and hasattr(adapter, 'queue_message'):
                        adapter.queue_message(session_key, pending)
                    return result_holder[0] or {"final_response": response, "messages": history}

                # Don't send the interrupted response to the user — it's just noise
                # like "Operation interrupted." They already know they sent a new
                # message, so go straight to processing it.
                
                # Now process the pending message with updated history
                updated_history = result.get("messages", history)
                return await self._run_agent(
                    message=pending,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=source,
                    session_id=session_id,
                    session_key=session_key,
                    _interrupt_depth=_interrupt_depth + 1,
                )
        finally:
            # Stop progress sender and interrupt monitor
            if progress_task:
                progress_task.cancel()
            interrupt_monitor.cancel()

            # Wait for stream consumer to finish its final edit
            if stream_task:
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
            
            # Clean up tracking
            tracking_task.cancel()
            if session_key and session_key in self._running_agents:
                del self._running_agents[session_key]
            
            # Wait for cancelled tasks
            for task in [progress_task, interrupt_monitor, tracking_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # If streaming already delivered the response, mark it so the
        # caller's send() is skipped (avoiding duplicate messages).
        _sc = stream_consumer_holder[0]
        if _sc and _sc.already_sent and isinstance(response, dict):
            response["already_sent"] = True
        
        return response


def _start_cron_ticker(stop_event: threading.Event, adapters=None, interval: int = 60):
    """
    Background thread that ticks the cron scheduler at a regular interval.
    
    Runs inside the gateway process so cronjobs fire automatically without
    needing a separate `hermes cron daemon` or system cron entry.

    Also refreshes the channel directory every 5 minutes and prunes the
    image/audio/document cache once per hour.
    """
    from cron.scheduler import tick as cron_tick
    from gateway.platforms.base import cleanup_image_cache, cleanup_document_cache

    IMAGE_CACHE_EVERY = 60   # ticks — once per hour at default 60s interval
    CHANNEL_DIR_EVERY = 5    # ticks — every 5 minutes

    logger.info("Cron ticker started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        try:
            cron_tick(verbose=False)
        except Exception as e:
            logger.debug("Cron tick error: %s", e)

        tick_count += 1

        if tick_count % CHANNEL_DIR_EVERY == 0 and adapters:
            try:
                from gateway.channel_directory import build_channel_directory
                build_channel_directory(adapters)
            except Exception as e:
                logger.debug("Channel directory refresh error: %s", e)

        if tick_count % IMAGE_CACHE_EVERY == 0:
            try:
                removed = cleanup_image_cache(max_age_hours=24)
                if removed:
                    logger.info("Image cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Image cache cleanup error: %s", e)
            try:
                removed = cleanup_document_cache(max_age_hours=24)
                if removed:
                    logger.info("Document cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Document cache cleanup error: %s", e)

        stop_event.wait(timeout=interval)
    logger.info("Cron ticker stopped")


async def start_gateway(config: Optional[GatewayConfig] = None, replace: bool = False) -> bool:
    """
    Start the gateway and run until interrupted.
    
    This is the main entry point for running the gateway.
    Returns True if the gateway ran successfully, False if it failed to start.
    A False return causes a non-zero exit code so systemd can auto-restart.
    
    Args:
        config: Optional gateway configuration override.
        replace: If True, kill any existing gateway instance before starting.
                 Useful for systemd services to avoid restart-loop deadlocks
                 when the previous process hasn't fully exited yet.
    """
    # ── Duplicate-instance guard ──────────────────────────────────────
    # Prevent two gateways from running under the same HERMES_HOME.
    # The PID file is scoped to HERMES_HOME, so future multi-profile
    # setups (each profile using a distinct HERMES_HOME) will naturally
    # allow concurrent instances without tripping this guard.
    import time as _time
    from gateway.status import get_running_pid, remove_pid_file
    existing_pid = get_running_pid()
    if existing_pid is not None and existing_pid != os.getpid():
        if replace:
            logger.info(
                "Replacing existing gateway instance (PID %d) with --replace.",
                existing_pid,
            )
            try:
                os.kill(existing_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # Already gone
            except PermissionError:
                logger.error(
                    "Permission denied killing PID %d. Cannot replace.",
                    existing_pid,
                )
                return False
            # Wait up to 10 seconds for the old process to exit
            for _ in range(20):
                try:
                    os.kill(existing_pid, 0)
                    _time.sleep(0.5)
                except (ProcessLookupError, PermissionError):
                    break  # Process is gone
            else:
                # Still alive after 10s — force kill
                logger.warning(
                    "Old gateway (PID %d) did not exit after SIGTERM, sending SIGKILL.",
                    existing_pid,
                )
                try:
                    os.kill(existing_pid, signal.SIGKILL)
                    _time.sleep(0.5)
                except (ProcessLookupError, PermissionError):
                    pass
            remove_pid_file()
        else:
            hermes_home = os.getenv("HERMES_HOME", "~/.hermes")
            logger.error(
                "Another gateway instance is already running (PID %d, HERMES_HOME=%s). "
                "Use 'hermes gateway restart' to replace it, or 'hermes gateway stop' first.",
                existing_pid, hermes_home,
            )
            print(
                f"\n❌ Gateway already running (PID {existing_pid}).\n"
                f"   Use 'hermes gateway restart' to replace it,\n"
                f"   or 'hermes gateway stop' to kill it first.\n"
                f"   Or use 'hermes gateway run --replace' to auto-replace.\n"
            )
            return False

    # Sync bundled skills on gateway start (fast -- skips unchanged)
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
    except Exception:
        pass

    # Configure rotating file log so gateway output is persisted for debugging
    log_dir = _hermes_home / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / 'gateway.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    from agent.redact import RedactingFormatter
    file_handler.setFormatter(RedactingFormatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    # Separate errors-only log for easy debugging
    error_handler = RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=2 * 1024 * 1024,
        backupCount=2,
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(RedactingFormatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logging.getLogger().addHandler(error_handler)

    runner = GatewayRunner(config)
    
    # Set up signal handlers
    def signal_handler():
        asyncio.create_task(runner.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    # Start the gateway
    success = await runner.start()
    if not success:
        return False
    if runner.should_exit_cleanly:
        if runner.exit_reason:
            logger.error("Gateway exiting cleanly: %s", runner.exit_reason)
        return True
    
    # Write PID file so CLI can detect gateway is running
    import atexit
    from gateway.status import write_pid_file, remove_pid_file
    write_pid_file()
    atexit.register(remove_pid_file)
    
    # Start background cron ticker so scheduled jobs fire automatically
    cron_stop = threading.Event()
    cron_thread = threading.Thread(
        target=_start_cron_ticker,
        args=(cron_stop,),
        kwargs={"adapters": runner.adapters},
        daemon=True,
        name="cron-ticker",
    )
    cron_thread.start()
    
    # Wait for shutdown
    await runner.wait_for_shutdown()
    
    # Stop cron ticker cleanly
    cron_stop.set()
    cron_thread.join(timeout=5)

    # Close MCP server connections
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
    except Exception:
        pass

    return True


def main():
    """CLI entry point for the gateway."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        import json
        with open(args.config, encoding="utf-8") as f:
            data = json.load(f)
            config = GatewayConfig.from_dict(data)
    
    # Run the gateway - exit with code 1 if no platforms connected,
    # so systemd Restart=on-failure will retry on transient errors (e.g. DNS)
    success = asyncio.run(start_gateway(config))
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
