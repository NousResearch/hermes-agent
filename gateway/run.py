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
import sys
import signal
import threading
import time
import shlex
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

# Load environment variables from ~/.hermes/.env first
from dotenv import load_dotenv
_env_path = _hermes_home / '.env'
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="latin-1")
# Also try project .env as fallback
load_dotenv()

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
            }
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    _val = _terminal_cfg[_cfg_key]
                    if isinstance(_val, list):
                        os.environ[_env_var] = json.dumps(_val)
                    else:
                        os.environ[_env_var] = str(_val)
        _compression_cfg = _cfg.get("compression", {})
        if _compression_cfg and isinstance(_compression_cfg, dict):
            _compression_env_map = {
                "enabled": "CONTEXT_COMPRESSION_ENABLED",
                "threshold": "CONTEXT_COMPRESSION_THRESHOLD",
                "summary_model": "CONTEXT_COMPRESSION_MODEL",
                "summary_provider": "CONTEXT_COMPRESSION_PROVIDER",
            }
            for _cfg_key, _env_var in _compression_env_map.items():
                if _cfg_key in _compression_cfg:
                    os.environ[_env_var] = str(_compression_cfg[_cfg_key])
        # Auxiliary model overrides (vision, web_extract).
        # Each task has provider + model; bridge non-default values to env vars.
        _auxiliary_cfg = _cfg.get("auxiliary", {})
        if _auxiliary_cfg and isinstance(_auxiliary_cfg, dict):
            _aux_task_env = {
                "vision":      ("AUXILIARY_VISION_PROVIDER",      "AUXILIARY_VISION_MODEL"),
                "web_extract": ("AUXILIARY_WEB_EXTRACT_PROVIDER",  "AUXILIARY_WEB_EXTRACT_MODEL"),
            }
            for _task_key, (_prov_env, _model_env) in _aux_task_env.items():
                _task_cfg = _auxiliary_cfg.get(_task_key, {})
                if not isinstance(_task_cfg, dict):
                    continue
                _prov = str(_task_cfg.get("provider", "")).strip()
                _model = str(_task_cfg.get("model", "")).strip()
                if _prov and _prov != "auto":
                    os.environ[_prov_env] = _prov
                if _model:
                    os.environ[_model_env] = _model
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
    build_session_key,
    build_session_context,
    build_session_context_prompt,
)
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType

logger = logging.getLogger(__name__)


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


class GatewayRunner:
    """
    Main gateway controller.
    
    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or load_gateway_config()
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._ephemeral_system_prompt = self._load_ephemeral_system_prompt()
        self._reasoning_config = self._load_reasoning_config()
        self._show_reasoning = self._load_show_reasoning()
        self._provider_routing = self._load_provider_routing()
        self._task_routing_policy = self._load_task_routing_policy()
        self._fallback_model = self._load_fallback_model()

        # Wire process registry into session store for reset protection
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(key),
        )
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._pending_messages: Dict[str, str] = {}  # Queued messages during interrupt
        self._session_runtime_overrides: Dict[str, Dict[str, Any]] = {}
        
        # Track pending exec approvals per session
        # Key: session_key, Value: {"command": str, "pattern_key": str}
        self._pending_approvals: Dict[str, Dict[str, str]] = {}

        # Runtime ops metrics (/ops) -- in-memory rolling window
        self._ops_window_seconds = int(os.getenv("HERMES_OPS_WINDOW_SECONDS", str(24 * 3600)))
        self._ops_stall_threshold_seconds = int(os.getenv("HERMES_OPS_STALL_SECONDS", "180"))
        self._ops_runs = deque(maxlen=int(os.getenv("HERMES_OPS_MAX_RUNS", "5000")))
        self._ops_active_runs: Dict[str, Dict[str, Any]] = {}
        self._ops_run_seq = 0
        self._ops_recent_debug_limit = int(os.getenv("HERMES_OPS_RECENT_DEBUG_LIMIT", "5"))
        self._ops_alerted_run_ids: set[str] = set()
        self._ops_stall_alerts_enabled = os.getenv("HERMES_OPS_STALL_ALERTS", "true").lower() in {"1", "true", "yes", "on"}
        self._ops_stall_recovery_notices_enabled = os.getenv("HERMES_OPS_STALL_RECOVERY_NOTICES", "true").lower() in {"1", "true", "yes", "on"}
        self._ops_stall_alert_home_fanout_main_enabled = os.getenv("HERMES_OPS_STALL_ALERT_HOME_FANOUT_MAIN", "false").lower() in {"1", "true", "yes", "on"}

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
    
    def _flush_memories_for_session(self, old_session_id: str):
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
            )
            logger.info("Pre-reset memory flush completed for session %s", old_session_id)
        except Exception as e:
            logger.debug("Pre-reset memory flush failed for session %s: %s", old_session_id, e)

    async def _async_flush_memories(self, old_session_id: str):
        """Run the sync memory flush in a thread pool so it won't block the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._flush_memories_for_session, old_session_id)

    def _clear_session_runtime_state(self, session_key: str) -> None:
        """Clear in-memory runtime state for a session key.

        Used by lifecycle transitions (reset/resume) so stale interrupt queues or
        pending approvals do not leak into a newly-created/rebound session.
        """
        self._running_agents.pop(session_key, None)
        self._pending_messages.pop(session_key, None)
        self._pending_approvals.pop(session_key, None)
        if hasattr(self, "_session_runtime_overrides"):
            self._session_runtime_overrides.pop(session_key, None)
        if hasattr(self, "_ops_active_runs"):
            self._ops_active_runs.pop(session_key, None)

    @staticmethod
    def _classify_task_type(event: MessageEvent, message_text: str) -> str:
        """Best-effort task classification for telemetry/routing signals."""
        if event.is_command():
            return "command"
        if event.message_type in (MessageType.PHOTO, MessageType.VIDEO):
            return "vision"
        if event.message_type in (MessageType.VOICE, MessageType.AUDIO):
            return "audio"
        if event.message_type == MessageType.DOCUMENT:
            return "document"

        lowered = (message_text or "").lower()
        if any(tok in lowered for tok in ("```", "traceback", "stack", "exception", "pytest")):
            return "code"
        if any(tok in lowered for tok in ("summarize", "summary", "explain", "research")):
            return "analysis"
        return "chat"

    @staticmethod
    def _token_cost_class(total_tokens: int) -> str:
        if total_tokens <= 2_000:
            return "tiny"
        if total_tokens <= 8_000:
            return "small"
        if total_tokens <= 24_000:
            return "medium"
        return "large"

    async def _emit_message_telemetry(self, payload: Dict[str, Any]) -> None:
        """Emit non-invasive gateway telemetry event (best effort)."""
        try:
            await self.hooks.emit("telemetry:message", payload)
        except Exception as e:
            logger.debug("Telemetry hook emission failed: %s", e)

    @staticmethod
    def _is_thread_source(source: SessionSource) -> bool:
        """Return True when a message source is scoped to a thread/topic."""
        if not source:
            return False
        if getattr(source, "thread_id", None):
            return True
        return str(getattr(source, "chat_type", "")).lower() == "thread"

    def _next_ops_run_id(self) -> str:
        """Return a compact monotonic run id for /ops correlation."""
        self._ops_run_seq = int(getattr(self, "_ops_run_seq", 0)) + 1
        return f"run-{self._ops_run_seq:06d}"

    def _record_run_started(self, session_key: str, source: SessionSource) -> Optional[str]:
        """Track an active run for live /ops reporting and return run_id."""
        if not session_key:
            return None

        active_runs = getattr(self, "_ops_active_runs", None)
        if active_runs is None:
            active_runs = {}
            self._ops_active_runs = active_runs

        run_id = self._next_ops_run_id()
        active_runs[session_key] = {
            "run_id": run_id,
            "session_key": session_key,
            "started_at": time.time(),
            "thread": self._is_thread_source(source),
            "platform": source.platform.value if source and source.platform else "",
            "chat_id": getattr(source, "chat_id", ""),
            "thread_id": getattr(source, "thread_id", None),
        }
        return run_id

    def _record_run_outcome(self, session_key: str, source: SessionSource, outcome: str, latency_ms: int) -> Dict[str, Any]:
        """Record a completed run outcome in the rolling /ops window and return run metadata."""
        now_ts = time.time()

        active_runs = getattr(self, "_ops_active_runs", None)
        if active_runs is None:
            active_runs = {}
            self._ops_active_runs = active_runs

        alerted_run_ids = getattr(self, "_ops_alerted_run_ids", None)
        if alerted_run_ids is None:
            alerted_run_ids = set()
            self._ops_alerted_run_ids = alerted_run_ids

        runs = getattr(self, "_ops_runs", None)
        if runs is None:
            runs = deque()
            self._ops_runs = runs

        run_meta = active_runs.pop(session_key, None)
        started_at = float(run_meta.get("started_at", now_ts)) if isinstance(run_meta, dict) else now_ts
        run_id = (run_meta or {}).get("run_id") if isinstance(run_meta, dict) else None
        stall_alerted = bool(run_id and run_id in alerted_run_ids)
        run_record: Dict[str, Any] = {
            "ts": now_ts,
            "started_at": started_at,
            "finished_at": now_ts,
            "outcome": outcome,
            "latency_ms": max(0, int(latency_ms or 0)),
            "thread": bool(run_meta.get("thread")) if isinstance(run_meta, dict) else self._is_thread_source(source),
            "run_id": run_id,
            "session_key": session_key,
            "platform": (run_meta or {}).get("platform") if isinstance(run_meta, dict) else (source.platform.value if source and source.platform else ""),
            "chat_id": (run_meta or {}).get("chat_id") if isinstance(run_meta, dict) else getattr(source, "chat_id", ""),
            "thread_id": (run_meta or {}).get("thread_id") if isinstance(run_meta, dict) else getattr(source, "thread_id", None),
            "stall_alerted": stall_alerted,
        }
        runs.append(run_record)

        # Prune stale entries outside the rolling window.
        cutoff = now_ts - max(60, int(getattr(self, "_ops_window_seconds", 3600)))
        while runs and runs[0].get("ts", 0) < cutoff:
            runs.popleft()

        return run_record

    @staticmethod
    def _p95(values: List[int]) -> int:
        if not values:
            return 0
        ordered = sorted(int(v) for v in values)
        idx = int(round(0.95 * (len(ordered) - 1)))
        idx = max(0, min(idx, len(ordered) - 1))
        return ordered[idx]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds or 0))
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m"
        return f"{seconds // 3600}h"

    @staticmethod
    def _format_timestamp_iso(ts: Optional[float]) -> str:
        if not ts:
            return "never"
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _find_ops_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Locate a run by id in active or recent history."""
        run_id = (run_id or "").strip()
        if not run_id:
            return None

        for item in self._ops_active_runs.values():
            if str(item.get("run_id", "")).strip() == run_id:
                now_ts = time.time()
                started_at = float(item.get("started_at", now_ts))
                return {
                    **item,
                    "status": "active",
                    "elapsed_seconds": max(0, int(now_ts - started_at)),
                }

        for item in reversed(self._ops_runs):
            if str(item.get("run_id", "")).strip() == run_id:
                started_at = float(item.get("started_at", item.get("ts", 0) or 0) or 0)
                finished_at = float(item.get("finished_at", item.get("ts", 0) or 0) or 0)
                return {
                    **item,
                    "status": "finished",
                    "elapsed_seconds": max(0, int(finished_at - started_at)) if started_at and finished_at else 0,
                }
        return None

    def _resolve_ops_debug_run_id(self, selector: str) -> str:
        """Resolve `/ops debug` selector tokens into a concrete run id."""
        token = (selector or "").strip()
        if not token:
            return ""
        if token.lower() != "latest":
            return token

        # Prefer most recent active run, otherwise most recent finished run.
        active_runs = [
            item for item in self._ops_active_runs.values()
            if str(item.get("run_id") or "").strip()
        ]
        if active_runs:
            newest_active = max(active_runs, key=lambda item: float(item.get("started_at", 0) or 0))
            return str(newest_active.get("run_id") or "").strip()

        for item in reversed(self._ops_runs):
            run_id = str(item.get("run_id") or "").strip()
            if run_id:
                return run_id
        return ""

    def _resolve_ops_debug_selector(self, selector: str) -> str:
        """Resolve selector aliases for `/ops debug` across main/thread run classes."""
        token = (selector or "").strip()
        mode = token.lower()
        if mode not in {"latest-main", "latest-thread"}:
            return self._resolve_ops_debug_run_id(token)

        want_thread = (mode == "latest-thread")

        active_runs = [
            item
            for item in self._ops_active_runs.values()
            if str(item.get("run_id") or "").strip() and bool(item.get("thread")) == want_thread
        ]
        if active_runs:
            newest_active = max(active_runs, key=lambda item: float(item.get("started_at", 0) or 0))
            return str(newest_active.get("run_id") or "").strip()

        for item in reversed(self._ops_runs):
            run_id = str(item.get("run_id") or "").strip()
            if run_id and bool(item.get("thread")) == want_thread:
                return run_id
        return ""

    async def _send_ops_stall_alert(self, run_id: str, session_key: str, source: SessionSource, elapsed_seconds: int) -> None:
        """Emit one-shot stall alert to the originating chat."""
        if not run_id or not getattr(self, "_ops_stall_alerts_enabled", True):
            return

        alerted = getattr(self, "_ops_alerted_run_ids", set())
        if run_id in alerted:
            return

        adapter = self.adapters.get(source.platform) if source else None
        if not adapter:
            return

        message = (
            "⚠️ **Ops stall alert**\n"
            f"- Run: `{run_id}`\n"
            f"- Session: `{session_key}`\n"
            f"- Elapsed: {self._format_duration(elapsed_seconds)}\n"
            f"- Threshold: {self._format_duration(getattr(self, '_ops_stall_threshold_seconds', 0))}\n"
            "Use `/ops` for board view or `/ops debug <run_id|latest|latest-main|latest-thread>` for details."
        )
        metadata = {"thread_id": source.thread_id} if getattr(source, "thread_id", None) else None

        try:
            result = await adapter.send(chat_id=source.chat_id, content=message, metadata=metadata)
            if getattr(result, "success", True):
                alerted.add(run_id)
                self._ops_alerted_run_ids = alerted

                should_fanout_home = (
                    getattr(self, "_ops_stall_alert_home_fanout_main_enabled", False)
                    and source
                    and source.platform == Platform.DISCORD
                    and not self._is_thread_source(source)
                )
                if should_fanout_home:
                    home_channel_id = str(os.getenv("DISCORD_HOME_CHANNEL", "")).strip()
                    if home_channel_id and home_channel_id != str(getattr(source, "chat_id", "")):
                        fanout_message = (
                            "📣 **Home mirror: ops stall alert (main channel run)**\n"
                            + message
                        )
                        try:
                            await adapter.send(chat_id=home_channel_id, content=fanout_message)
                        except Exception as e:
                            logger.debug("Failed to fanout ops stall alert for %s to home channel: %s", run_id, e)
        except Exception as e:
            logger.debug("Failed to send ops stall alert for %s: %s", run_id, e)

    async def _send_ops_stall_recovered_notice(
        self,
        run_id: str,
        session_key: str,
        source: SessionSource,
        elapsed_seconds: int,
        outcome: str,
    ) -> None:
        """Notify when a previously alerted stall run completes."""
        if not run_id or not getattr(self, "_ops_stall_recovery_notices_enabled", True):
            return

        alerted = getattr(self, "_ops_alerted_run_ids", set())
        if run_id not in alerted:
            return

        adapter = self.adapters.get(source.platform) if source else None
        if not adapter:
            return

        message = (
            "✅ **Ops stall recovered**\n"
            f"- Run: `{run_id}`\n"
            f"- Session: `{session_key}`\n"
            f"- Outcome: {outcome}\n"
            f"- Total elapsed: {self._format_duration(elapsed_seconds)}"
        )
        metadata = {"thread_id": source.thread_id} if getattr(source, "thread_id", None) else None

        try:
            result = await adapter.send(chat_id=source.chat_id, content=message, metadata=metadata)
            if getattr(result, "success", True):
                should_fanout_home = (
                    getattr(self, "_ops_stall_alert_home_fanout_main_enabled", False)
                    and source
                    and source.platform == Platform.DISCORD
                    and not self._is_thread_source(source)
                )
                if should_fanout_home:
                    home_channel_id = str(os.getenv("DISCORD_HOME_CHANNEL", "")).strip()
                    if home_channel_id and home_channel_id != str(getattr(source, "chat_id", "")):
                        fanout_message = (
                            "📣 **Home mirror: ops stall recovered (main channel run)**\n"
                            + message
                        )
                        try:
                            await adapter.send(chat_id=home_channel_id, content=fanout_message)
                        except Exception as e:
                            logger.debug(
                                "Failed to fanout ops stall recovery notice for %s to home channel: %s",
                                run_id,
                                e,
                            )

                alerted.discard(run_id)
                self._ops_alerted_run_ids = alerted
        except Exception as e:
            logger.debug("Failed to send ops stall recovery notice for %s: %s", run_id, e)

    def _build_ops_snapshot(self) -> Dict[str, Any]:
        """Build live gateway ops metrics for /ops."""
        now_ts = time.time()
        cutoff = now_ts - max(60, int(self._ops_window_seconds))
        runs_window = [r for r in self._ops_runs if r.get("ts", 0) >= cutoff]

        success_runs = [r for r in runs_window if r.get("outcome") == "success"]
        error_runs = [r for r in runs_window if r.get("outcome") == "error"]

        main_success_lat = [r.get("latency_ms", 0) for r in success_runs if not r.get("thread")]
        thread_success_lat = [r.get("latency_ms", 0) for r in success_runs if r.get("thread")]

        active_items = list(self._ops_active_runs.values())
        active_main = sum(1 for item in active_items if not item.get("thread"))
        active_thread = sum(1 for item in active_items if item.get("thread"))
        stalled = [
            item for item in active_items
            if (now_ts - float(item.get("started_at", now_ts))) >= self._ops_stall_threshold_seconds
        ]

        started = len(runs_window)
        completed = len(success_runs)
        failed = len(error_runs)
        success_rate = (completed / started * 100.0) if started else 0.0
        last_success_ts = max((r.get("ts", 0) for r in success_runs), default=0)

        recent_limit = max(1, int(getattr(self, "_ops_recent_debug_limit", 5) or 5))
        recent_runs = []
        for item in list(reversed(runs_window)):
            run_id = str(item.get("run_id") or "").strip()
            if not run_id:
                continue
            recent_runs.append(
                {
                    "run_id": run_id,
                    "outcome": str(item.get("outcome", "unknown")),
                    "thread": bool(item.get("thread")),
                    "latency_ms": int(item.get("latency_ms", 0) or 0),
                }
            )
            if len(recent_runs) >= recent_limit:
                break

        return {
            "window_hours": round(self._ops_window_seconds / 3600, 2),
            "started": started,
            "completed": completed,
            "failed": failed,
            "success_rate": success_rate,
            "p95_latency_main_ms": self._p95(main_success_lat),
            "p95_latency_thread_ms": self._p95(thread_success_lat),
            "active_total": len(active_items),
            "active_main": active_main,
            "active_thread": active_thread,
            "queue_depth": len(self._pending_messages),
            "stalled_active": len(stalled),
            "stall_threshold_seconds": self._ops_stall_threshold_seconds,
            "stalled_longest_seconds": max(
                (int(now_ts - float(item.get("started_at", now_ts))) for item in stalled),
                default=0,
            ),
            "last_success_ts": last_success_ts or None,
            "recent_runs": recent_runs,
        }

    @staticmethod
    def _get_write_command_set() -> set[str]:
        """Resolve slash commands that mutate gateway/runtime state."""
        configured = os.getenv("GATEWAY_WRITE_COMMANDS", "").strip()
        if not configured:
            configured = "sethome,set-home,model,update,reload-mcp,reload_mcp"
        return {cmd.strip().lower() for cmd in configured.split(",") if cmd.strip()}

    @classmethod
    def _is_write_command(cls, command: str) -> bool:
        return bool(command and command.lower() in cls._get_write_command_set())

    @staticmethod
    def _parse_allowlist(value: str) -> set[str]:
        return {token.strip() for token in (value or "").split(",") if token.strip()}

    @classmethod
    def _is_write_command_authorized(cls, source: SessionSource, command: str) -> tuple[bool, str]:
        """Check command-level RBAC for write commands.

        Returns (authorized, reason).
        """
        if not cls._is_write_command(command):
            return True, "read_command"

        if os.getenv("GATEWAY_WRITE_ALLOW_ALL", "false").lower() == "true":
            return True, "allow_all"

        platform_name = source.platform.value.upper() if source.platform else ""
        global_allow = cls._parse_allowlist(os.getenv("GATEWAY_WRITE_ALLOWLIST", ""))
        platform_allow = cls._parse_allowlist(os.getenv(f"{platform_name}_WRITE_ALLOWLIST", "")) if platform_name else set()
        allowed = global_allow | platform_allow

        # Backward-compatible default: if no explicit write allowlist exists, do not block.
        if not allowed:
            return True, "no_allowlist_configured"

        user_id = source.user_id
        check_ids = {user_id}
        if "@" in user_id:
            check_ids.add(user_id.split("@")[0])

        if check_ids & allowed:
            return True, "allowlist"
        return False, "missing_write_permission"

    async def _audit_write_command(
        self,
        *,
        event: MessageEvent,
        command: str,
        args: str,
        authorized: bool,
        reason: str,
    ) -> None:
        """Best-effort write-command audit log + hook emission."""
        source = event.source
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": source.platform.value if source.platform else "",
            "chat_id": source.chat_id,
            "chat_type": source.chat_type,
            "user_id": source.user_id,
            "user_name": source.user_name,
            "command": command,
            "args": args,
            "authorized": authorized,
            "reason": reason,
        }

        try:
            log_dir = _hermes_home / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            audit_path = log_dir / "gateway_command_audit.jsonl"
            with audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Write-command audit file write failed: %s", e)

        try:
            await self.hooks.emit("audit:command", payload)
        except Exception as e:
            logger.debug("Write-command audit hook emission failed: %s", e)

    @staticmethod
    def _parse_runtime_message_control(text: str) -> tuple[str, str]:
        """Parse runtime control syntax for in-flight messages.

        Returns a tuple of ``(mode, payload)`` where mode is one of:
        - ``steer``: interrupt the active run and steer immediately (default)
        - ``queue``: queue follow-up intent to run deterministically after current run

        Supported explicit prefixes:
        - /queue <message>
        - queue: <message>
        - q: <message>
        - /steer <message>
        - steer: <message>
        """
        raw = (text or "").strip()
        if not raw:
            return ("steer", "")

        queue_match = re.match(r"^(?:/queue\b|queue:|q:)\s*(.*)$", raw, flags=re.IGNORECASE)
        if queue_match:
            payload = queue_match.group(1).strip()
            return ("queue", payload)

        steer_match = re.match(r"^(?:/steer\b|steer:)\s*(.*)$", raw, flags=re.IGNORECASE)
        if steer_match:
            payload = steer_match.group(1).strip()
            return ("steer", payload)

        return ("steer", raw)

    @staticmethod
    def _derive_parent_thread_source(source: SessionSource, event: MessageEvent) -> Optional[SessionSource]:
        """Derive parent-channel source for a thread/topic message.

        Returns ``None`` when parent source cannot be resolved.
        """
        if not source.thread_id:
            return None

        parent_chat_id = source.chat_id
        parent_chat_type = source.chat_type

        if source.chat_type == "thread":
            raw_channel = getattr(event.raw_message, "channel", None)
            parent_candidate = getattr(raw_channel, "parent_id", None)
            if not parent_candidate:
                parent_obj = getattr(raw_channel, "parent", None)
                parent_candidate = getattr(parent_obj, "id", None)
            if not parent_candidate:
                return None
            parent_chat_id = str(parent_candidate)
            parent_chat_type = "group"

        return SessionSource(
            platform=source.platform,
            chat_id=parent_chat_id,
            chat_name=source.chat_name,
            chat_type=parent_chat_type,
            user_id=source.user_id,
            user_name=source.user_name,
            thread_id=None,
            chat_topic=source.chat_topic,
            user_id_alt=source.user_id_alt,
            chat_id_alt=source.chat_id_alt,
        )

    def _build_thread_bootstrap_context(
        self,
        source: SessionSource,
        event: MessageEvent,
        history: list[dict],
    ) -> str:
        """Build context note for first turn in a thread from parent session history."""
        if history or not source.thread_id:
            return ""

        parent_source = self._derive_parent_thread_source(source, event)
        if not parent_source:
            return ""

        parent_key = build_session_key(
            parent_source,
            isolate_threads=self.config.session_lifecycle.isolate_threads,
        )
        parent_entry = self.session_store._entries.get(parent_key)
        if not parent_entry:
            return ""

        try:
            parent_history = self.session_store.load_transcript(parent_entry.session_id)
        except Exception:
            return ""

        if not parent_history:
            return ""

        recent_turns: list[str] = []
        for msg in parent_history:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            content = " ".join(content.split())
            if len(content) > 240:
                content = content[:237] + "..."
            label = "User" if role == "user" else "Assistant"
            recent_turns.append(f"- {label}: {content}")

        if not recent_turns:
            return ""

        recent_turns = recent_turns[-8:]
        return (
            "[System note: New thread detected. Bootstrap with recent parent-channel context "
            "for continuity, then prioritize this thread independently.]\n"
            + "\n".join(recent_turns)
        )

    @staticmethod
    def _format_command_confirmation(command: str, body: str) -> str:
        """Format slash-command responses with consistent, tool-style confirmation framing."""
        cmd = (command or "").strip().lstrip("/") or "command"
        text = (body or "").strip() or "(no details)"

        emoji = "✅"
        lowered = text.lower()
        if text.startswith(("❌", "⛔")) or " not authorized" in lowered:
            emoji = "❌"
        elif text.startswith("⚠️") or "invalid" in lowered or "missing" in lowered:
            emoji = "⚠️"
        elif text.startswith(("ℹ️", "🔎", "📊", "📋")):
            emoji = "ℹ️"

        details = "\n".join((f"┊ {line}" if line else "┊") for line in text.splitlines())
        return f"{emoji} **/{cmd}**\n{details}"

    @staticmethod
    def _format_ask_runtime_confirmation(runtime_policy: dict) -> str:
        """Build a visible runtime receipt for /ask one-turn overrides."""

        def _fmt_reasoning(cfg: dict | None) -> str:
            if not cfg:
                return "default"
            if cfg.get("enabled") is False:
                return "none"
            return str(cfg.get("effort") or "default")

        model = runtime_policy.get("model") or "unset"
        provider = runtime_policy.get("provider") or "default"
        task_type = runtime_policy.get("effective_task_type") or "chat"
        reasoning = _fmt_reasoning(runtime_policy.get("reasoning_config"))
        return (
            "🚀 **/ask** runtime applied\n"
            f"┊ model: `{model}`\n"
            f"┊ provider: `{provider}`\n"
            f"┊ reasoning: `{reasoning}`\n"
            f"┊ task: `{task_type}`"
        )
    
    @staticmethod
    def _parse_ask_runtime_overrides(text: str) -> tuple[dict, str, str | None]:
        """Parse one-turn runtime overrides from /ask arguments.

        Supported forms (prefix options only, then prompt):
          - /ask model=<provider:model> reasoning=<high|...> <prompt>
          - /ask --model <provider:model> --reasoning <level> <prompt>
          - /ask provider=<provider> model=<model> <prompt>

        Returns: (overrides, prompt, error)
        """
        raw = (text or "").strip()
        if not raw:
            return {}, "", "Usage: /ask [model=<provider:model>|provider=<provider>|reasoning=<level>] <prompt>"

        try:
            tokens = shlex.split(raw)
        except ValueError as exc:
            return {}, "", f"⚠️ Invalid /ask arguments: {exc}"

        if not tokens:
            return {}, "", "Usage: /ask [model=<provider:model>|provider=<provider>|reasoning=<level>] <prompt>"

        overrides: dict[str, str] = {}
        i = 0
        valid_reasoning = {"xhigh", "high", "medium", "low", "minimal", "none", "default", "auto", "clear"}

        while i < len(tokens):
            token = tokens[i]
            key = ""
            value = ""

            if token == "--":
                i += 1
                break

            if token.startswith("--"):
                key = token[2:].strip().lower()
                if key not in {"model", "provider", "reasoning", "effort"}:
                    break
                if i + 1 >= len(tokens):
                    return {}, "", f"⚠️ Missing value for --{key}."
                value = tokens[i + 1].strip()
                i += 2
            elif "=" in token:
                maybe_key, maybe_value = token.split("=", 1)
                key = maybe_key.strip().lower()
                value = maybe_value.strip()
                if key not in {"model", "provider", "reasoning", "effort"}:
                    break
                if not value:
                    return {}, "", f"⚠️ Missing value for {key}=..."
                i += 1
            else:
                break

            if key in {"reasoning", "effort"}:
                norm = value.lower()
                if norm not in valid_reasoning:
                    opts = "xhigh, high, medium, low, minimal, none, default"
                    return {}, "", f"⚠️ Invalid reasoning value `{value}`. Use one of: {opts}."
                if norm in {"default", "auto", "clear"}:
                    continue
                overrides["reasoning_effort"] = norm
            elif key == "provider":
                overrides["provider"] = value
            elif key == "model":
                from hermes_cli.models import parse_model_input

                default_provider = overrides.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER", "openrouter")
                try:
                    provider, model = parse_model_input(value, default_provider)
                except Exception:
                    provider, model = default_provider, value
                overrides["provider"] = provider
                overrides["model"] = model

        prompt = " ".join(tokens[i:]).strip()
        if not prompt:
            return {}, "", "⚠️ Missing prompt. Usage: /ask [model=<...>] [reasoning=<...>] <prompt>"

        return overrides, prompt, None

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
        """Load reasoning effort from config or env var.
        
        Checks HERMES_REASONING_EFFORT env var first, then agent.reasoning_effort
        in config.yaml. Valid: "xhigh", "high", "medium", "low", "minimal", "none".
        Returns None to use default (medium).
        """
        effort = os.getenv("HERMES_REASONING_EFFORT", "")
        if not effort:
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
    def _load_task_routing_policy() -> dict:
        """Load task routing policy from config.yaml.

        Shape (all keys optional):
          task_routing:
            defaults: {model, provider, reasoning_effort, fallback_ladder}
            by_task:
              code: {model, provider, reasoning_effort, fallback_ladder}
        """
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                policy = cfg.get("task_routing", {}) or {}
                return policy if isinstance(policy, dict) else {}
        except Exception:
            pass
        return {}

    @staticmethod
    def _normalize_reasoning_config(effort: str | None) -> dict | None:
        """Normalize reasoning effort string to agent reasoning_config payload."""
        if not effort:
            return None
        normalized = str(effort).strip().lower()
        if normalized in ("default", "auto", ""):
            return None
        if normalized == "none":
            return {"enabled": False}
        valid = {"xhigh", "high", "medium", "low", "minimal"}
        if normalized in valid:
            return {"enabled": True, "effort": normalized}
        return None

    @staticmethod
    def _valid_task_types() -> set[str]:
        return {"command", "vision", "audio", "document", "code", "analysis", "chat"}

    @staticmethod
    def _normalize_fallback_ladder(value: Any) -> list[dict]:
        """Normalize fallback ladder entries to [{provider, model}, ...]."""
        if not isinstance(value, list):
            return []
        normalized: list[dict] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            provider = str(entry.get("provider", "")).strip()
            model = str(entry.get("model", "")).strip()
            if provider and model:
                normalized.append({"provider": provider, "model": model})
        return normalized

    def _resolve_runtime_policy(self, session_key: str, task_type: str) -> dict:
        """Resolve per-turn runtime policy with thread-local overrides."""
        runtime_overrides = getattr(self, "_session_runtime_overrides", {}) or {}
        overrides = runtime_overrides.get(session_key, {}) if session_key else {}
        forced_task = str(overrides.get("forced_task_type", "")).strip().lower()
        effective_task = forced_task if forced_task in self._valid_task_types() else task_type

        task_routing_policy = getattr(self, "_task_routing_policy", {})
        policy = task_routing_policy if isinstance(task_routing_policy, dict) else {}
        defaults = policy.get("defaults", {}) if isinstance(policy.get("defaults"), dict) else {}
        by_task = policy.get("by_task", {}) if isinstance(policy.get("by_task"), dict) else {}
        task_cfg = by_task.get(effective_task, {}) if isinstance(by_task.get(effective_task), dict) else {}

        provider = str(
            overrides.get("provider")
            or task_cfg.get("provider")
            or defaults.get("provider")
            or ""
        ).strip()
        model = str(
            overrides.get("model")
            or task_cfg.get("model")
            or defaults.get("model")
            or os.getenv("HERMES_MODEL")
            or os.getenv("LLM_MODEL")
            or ""
        ).strip()

        reasoning_effort = str(
            overrides.get("reasoning_effort")
            or task_cfg.get("reasoning_effort")
            or defaults.get("reasoning_effort")
            or ""
        ).strip().lower()
        default_reasoning = getattr(self, "_reasoning_config", None)
        reasoning_config = self._normalize_reasoning_config(reasoning_effort) or default_reasoning

        ladder = self._normalize_fallback_ladder(task_cfg.get("fallback_ladder"))
        if not ladder:
            ladder = self._normalize_fallback_ladder(defaults.get("fallback_ladder"))
        fallback_model = ladder[0] if ladder else getattr(self, "_fallback_model", None)

        return {
            "task_type": task_type,
            "effective_task_type": effective_task,
            "provider": provider,
            "model": model,
            "reasoning_config": reasoning_config,
            "fallback_model": fallback_model,
        }

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

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        logger.info("Session storage: %s", self.config.sessions_dir)
        
        # Warn if no user allowlists are configured and open access is not opted in
        _any_allowlist = any(
            os.getenv(v)
            for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                       "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
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
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                logger.warning("No adapter available for %s", platform.value)
                continue
            
            # Set up message handler
            adapter.set_message_handler(self._handle_message)
            
            # Try to connect
            logger.info("Connecting to %s...", platform.value)
            try:
                success = await adapter.connect()
                if success:
                    self.adapters[platform] = adapter
                    connected_count += 1
                    logger.info("✓ %s connected", platform.value)
                else:
                    logger.warning("✗ %s failed to connect", platform.value)
            except Exception as e:
                logger.error("✗ %s error: %s", platform.value, e)
        
        if connected_count == 0:
            logger.warning("No messaging platforms connected.")
            logger.info("Gateway will continue running for cron job execution.")
        
        # Update delivery router with adapters
        self.delivery_router.adapters = self.adapters
        
        self._running = True
        
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
        
        # Check if we're restarting after a /update command
        await self._send_update_notification()

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
                        await self._async_flush_memories(entry.session_id)
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
        
        for platform, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                logger.info("✓ %s disconnected", platform.value)
            except Exception as e:
                logger.error("✗ %s disconnect error: %s", platform.value, e)
        
        self.adapters.clear()
        self._shutdown_event.set()
        
        from gateway.status import remove_pid_file
        remove_pid_file()
        
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
        if platform == Platform.TELEGRAM:
            from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements
            if not check_telegram_requirements():
                logger.info("Telegram unavailable: python-telegram-bot not installed")
                return None
            return TelegramAdapter(config)
        
        elif platform == Platform.DISCORD:
            from gateway.platforms.discord import DiscordAdapter, check_discord_requirements
            if not check_discord_requirements():
                logger.info("Discord unavailable: discord.py not installed")
                return None
            return DiscordAdapter(config)
        
        elif platform == Platform.WHATSAPP:
            from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                logger.info("WhatsApp unavailable: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)
        
        elif platform == Platform.SLACK:
            from gateway.platforms.slack import SlackAdapter, check_slack_requirements
            if not check_slack_requirements():
                logger.info("Slack unavailable: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'")
                return None
            return SlackAdapter(config)

        elif platform == Platform.SIGNAL:
            from gateway.platforms.signal import SignalAdapter, check_signal_requirements
            if not check_signal_requirements():
                logger.info("Signal unavailable: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
                return None
            return SignalAdapter(config)

        elif platform == Platform.HOMEASSISTANT:
            from gateway.platforms.homeassistant import HomeAssistantAdapter, check_ha_requirements
            if not check_ha_requirements():
                logger.info("HomeAssistant unavailable: aiohttp not installed or HASS_TOKEN not set")
                return None
            return HomeAssistantAdapter(config)

        elif platform == Platform.EMAIL:
            from gateway.platforms.email import EmailAdapter, check_email_requirements
            if not check_email_requirements():
                logger.warning("Email: EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set")
                return None
            return EmailAdapter(config)

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
        if source.platform == Platform.HOMEASSISTANT:
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
        }
        platform_allow_all_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
            Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
            Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOW_ALL_USERS",
            Platform.EMAIL: "EMAIL_ALLOW_ALL_USERS",
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
        
        # Check if user is authorized
        if not self._is_user_authorized(source):
            logger.warning("Unauthorized user: %s (%s) on %s", source.user_id, source.user_name, source.platform.value)
            # In DMs: offer pairing code. In groups: silently ignore.
            if source.chat_type == "dm":
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
        
        # PRIORITY: If an agent is already running for this session, interrupt it
        # immediately. This is before command parsing to minimize latency -- the
        # user's "stop" message reaches the agent as fast as possible.
        session_store = getattr(self, "session_store", None)
        if session_store is not None:
            _quick_key = session_store._generate_session_key(source)
        else:
            # __new__-constructed test instances may not initialize session_store.
            _quick_key = build_session_key(source)

        if _quick_key in self._running_agents:
            running_agent = self._running_agents[_quick_key]
            mode, payload = self._parse_runtime_message_control(event.text)
            payload = payload.strip() if payload else ""
            if not payload:
                payload = event.text.strip()

            if mode == "queue":
                logger.debug("Queued follow-up for active session %s", _quick_key[:20])
            else:
                logger.debug("PRIORITY interrupt for session %s", _quick_key[:20])
                running_agent.interrupt(payload)

            if _quick_key in self._pending_messages:
                self._pending_messages[_quick_key] += "\n" + payload
            else:
                self._pending_messages[_quick_key] = payload
            return None
        
        # Check for commands
        command = event.get_command()
        command_args = event.get_command_args().strip()
        one_turn_runtime_overrides: dict[str, str] = {}

        if command and self._is_write_command(command):
            authorized, reason = self._is_write_command_authorized(source, command)
            await self._audit_write_command(
                event=event,
                command=command,
                args=command_args,
                authorized=authorized,
                reason=reason,
            )
            if not authorized:
                return (
                    "⛔ You are not authorized to run this write command. "
                    "Ask the operator to add your user ID to GATEWAY_WRITE_ALLOWLIST "
                    "or DISCORD_WRITE_ALLOWLIST."
                )

        # Emit command:* hook for any recognized slash command
        _known_commands = {"new", "reset", "help", "status", "ops", "stop", "model", "ask",
                          "modelpin", "reasoning", "route", "runtime", "personality", "retry", "undo", "sethome", "set-home",
                          "compress", "usage", "insights", "reload-mcp", "reload_mcp",
                          "update", "title", "resume", "provider", "rollback", "background", "now", "blocked", "next"}
        if command and command in _known_commands:
            await self.hooks.emit(f"command:{command}", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "command": command,
                "args": command_args,
            })
        
        def _confirm(cmd_name: str, output: str) -> str:
            return self._format_command_confirmation(cmd_name, output)

        if command in ["new", "reset"]:
            return _confirm(command, await self._handle_reset_command(event))
        
        if command == "help":
            return _confirm(command, await self._handle_help_command(event))
        
        if command == "status":
            return _confirm(command, await self._handle_status_command(event))

        if command == "ops":
            return _confirm(command, await self._handle_ops_command(event))

        if command == "now":
            return _confirm(command, await self._handle_now_command(event))

        if command == "blocked":
            return _confirm(command, await self._handle_blocked_command(event))

        if command == "next":
            return _confirm(command, await self._handle_next_command(event))
        
        if command == "stop":
            return _confirm(command, await self._handle_stop_command(event))
        
        if command == "model":
            return _confirm(command, await self._handle_model_command(event))

        if command == "ask":
            overrides, prompt, parse_error = self._parse_ask_runtime_overrides(command_args)
            if parse_error:
                return _confirm("ask", parse_error)
            one_turn_runtime_overrides = overrides
            event.text = prompt
            command = None

        if command == "modelpin":
            return _confirm(command, await self._handle_modelpin_command(event))

        if command == "reasoning":
            return _confirm(command, await self._handle_reasoning_command(event))

        if command == "route":
            return _confirm(command, await self._handle_route_command(event))

        if command == "runtime":
            return _confirm(command, await self._handle_runtime_command(event))
        
        if command == "provider":
            return _confirm(command, await self._handle_provider_command(event))
        
        if command == "personality":
            return _confirm(command, await self._handle_personality_command(event))
        
        if command == "retry":
            return _confirm(command, await self._handle_retry_command(event))
        
        if command == "undo":
            return _confirm(command, await self._handle_undo_command(event))
        
        if command in ["sethome", "set-home"]:
            return _confirm(command, await self._handle_set_home_command(event))

        if command == "compress":
            return _confirm(command, await self._handle_compress_command(event))

        if command == "usage":
            return _confirm(command, await self._handle_usage_command(event))

        if command == "insights":
            return _confirm(command, await self._handle_insights_command(event))

        if command in ("reload-mcp", "reload_mcp"):
            return _confirm(command, await self._handle_reload_mcp_command(event))

        if command == "update":
            return _confirm(command, await self._handle_update_command(event))

        if command == "title":
            return _confirm(command, await self._handle_title_command(event))

        if command == "resume":
            return _confirm(command, await self._handle_resume_command(event))

        if command == "rollback":
            return _confirm(command, await self._handle_rollback_command(event))

        if command == "background":
            return _confirm(command, await self._handle_background_command(event))
        
        # User-defined quick commands (bypass agent loop, no LLM call)
        if command:
            quick_commands = self.config.get("quick_commands", {})
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
                else:
                    return f"Quick command '/{command}' has unsupported type (only 'exec' is supported)."

        # Skill slash commands: /skill-name loads the skill and sends to agent
        if command:
            try:
                from agent.skill_commands import get_skill_commands, build_skill_invocation_message
                skill_cmds = get_skill_commands()
                cmd_key = f"/{command}"
                if cmd_key in skill_cmds:
                    user_instruction = event.get_command_args().strip()
                    msg = build_skill_invocation_message(cmd_key, user_instruction)
                    if msg:
                        event.text = msg
                        # Fall through to normal message processing with skill content
            except Exception as e:
                logger.debug("Skill command check failed (non-fatal): %s", e)
        
        # Check for pending exec approval responses
        session_key_preview = self.session_store._generate_session_key(source)
        if session_key_preview in self._pending_approvals:
            user_text = event.text.strip().lower()
            if user_text in ("yes", "y", "approve", "ok", "go", "do it"):
                approval = self._pending_approvals.pop(session_key_preview)
                cmd = approval["command"]
                pattern_key = approval.get("pattern_key", "")
                logger.info("User approved dangerous command: %s...", cmd[:60])
                from tools.terminal_tool import terminal_tool
                from tools.approval import approve_session
                approve_session(session_key_preview, pattern_key)
                result = terminal_tool(command=cmd, force=True)
                return f"✅ Command approved and executed.\n\n```\n{result[:3500]}\n```"
            elif user_text in ("no", "n", "deny", "cancel", "nope"):
                self._pending_approvals.pop(session_key_preview)
                return "❌ Command denied."
            elif user_text in ("full", "show", "view", "show full", "view full"):
                # Show full command without consuming the approval
                cmd = self._pending_approvals[session_key_preview]["command"]
                return f"Full command:\n\n```\n{cmd}\n```\n\nReply yes/no to approve or deny."
            # If it's not clearly an approval/denial, fall through to normal processing
        
        # Get or create session
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        # Emit session:start for new or auto-reset sessions
        _is_new_session = (
            session_entry.created_at == session_entry.updated_at
            or getattr(session_entry, "was_auto_reset", False)
        )
        if _is_new_session:
            await self.hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })
        
        # Build session context
        context = build_session_context(source, self.config, session_entry)
        
        # Set environment variables for tools
        self._set_session_env(context)
        
        # Build the context prompt to inject
        context_prompt = build_session_context_prompt(context)
        
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

        # For brand-new threads/topics, seed context from recent parent-channel
        # turns so thread work starts with relevant continuity while remaining
        # independently runnable.
        thread_bootstrap = self._build_thread_bootstrap_context(source, event, history)
        if thread_bootstrap:
            context_prompt = f"{thread_bootstrap}\n\n{context_prompt}"

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

            # Read model + compression config from config.yaml — same
            # source of truth the agent itself uses.
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

                    # Read compression settings
                    _comp_cfg = _hyg_data.get("compression", {})
                    if isinstance(_comp_cfg, dict):
                        _hyg_threshold_pct = float(
                            _comp_cfg.get("threshold", _hyg_threshold_pct)
                        )
                        _hyg_compression_enabled = str(
                            _comp_cfg.get("enabled", True)
                        ).lower() in ("true", "1", "yes")
            except Exception:
                pass

            # Also check env overrides (same as run_agent.py)
            _hyg_threshold_pct = float(
                os.getenv("CONTEXT_COMPRESSION_THRESHOLD", str(_hyg_threshold_pct))
            )
            if os.getenv("CONTEXT_COMPRESSION_ENABLED", "").lower() in ("false", "0", "no"):
                _hyg_compression_enabled = False

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

        task_type = self._classify_task_type(event, message_text)
        runtime_policy = self._resolve_runtime_policy(session_key=session_key, task_type=task_type)

        if one_turn_runtime_overrides:
            merged_policy = dict(runtime_policy)
            provider_override = one_turn_runtime_overrides.get("provider")
            model_override = one_turn_runtime_overrides.get("model")
            reasoning_override = one_turn_runtime_overrides.get("reasoning_effort")
            if provider_override:
                merged_policy["provider"] = provider_override
            if model_override:
                merged_policy["model"] = model_override
            if reasoning_override:
                merged_policy["reasoning_config"] = self._normalize_reasoning_config(reasoning_override) or merged_policy.get("reasoning_config")
            runtime_policy = merged_policy

        telemetry_started = time.perf_counter()
        telemetry_base = {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_id": session_entry.session_id,
            "session_key": session_key,
            "task_type": runtime_policy.get("effective_task_type", task_type),
            "task_type_original": task_type,
            "model": runtime_policy.get("model") or os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL") or "",
            "provider": runtime_policy.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER") or "",
        }

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
                session_key=session_key,
                runtime_policy=runtime_policy,
            )
            
            response = agent_result.get("final_response", "")
            agent_messages = agent_result.get("messages", [])

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

            if one_turn_runtime_overrides:
                ask_runtime_receipt = self._format_ask_runtime_confirmation(runtime_policy)
                if response:
                    response = f"{ask_runtime_receipt}\n\n{response}"
                else:
                    response = ask_runtime_receipt
            
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
                pending = pop_pending(session_key)
                if pending:
                    self._pending_approvals[session_key] = pending
            except Exception as e:
                logger.debug("Failed to check pending approvals: %s", e)
            
            # Save the full conversation to the transcript, including tool calls.
            # This preserves the complete agent loop (tool_calls, tool results,
            # intermediate reasoning) so sessions can be resumed with full context
            # and transcripts are useful for debugging and training data.
            ts = datetime.now().isoformat()
            
            # If this is a fresh session (no history), write the full tool
            # definitions as the first entry so the transcript is self-describing
            # -- the same list of dicts sent as tools=[...] in the API request.
            if not history:
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
            
            # Update session with actual prompt token count from the agent
            self.session_store.update_session(
                session_entry.session_key,
                last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
            )

            total_tokens = 0
            try:
                usage = agent_result.get("usage") or {}
                if isinstance(usage, dict):
                    total_tokens = int(usage.get("total_tokens", 0) or 0)
            except Exception:
                total_tokens = 0

            latency_ms = int((time.perf_counter() - telemetry_started) * 1000)
            await self._emit_message_telemetry({
                **telemetry_base,
                "outcome": "success",
                "latency_ms": latency_ms,
                "token_total": total_tokens,
                "cost_class": self._token_cost_class(total_tokens),
            })
            run_record = self._record_run_outcome(session_key, source, "success", latency_ms)
            run_id = str(run_record.get("run_id") or "").strip()
            if run_record.get("stall_alerted") and run_id:
                elapsed_seconds = max(0, int(float(run_record.get("finished_at", 0) or 0) - float(run_record.get("started_at", 0) or 0)))
                await self._send_ops_stall_recovered_notice(
                    run_id=run_id,
                    session_key=session_key,
                    source=source,
                    elapsed_seconds=elapsed_seconds,
                    outcome="success",
                )

            return response
            
        except Exception as e:
            logger.exception("Agent error in session %s", session_key)
            latency_ms = int((time.perf_counter() - telemetry_started) * 1000)
            await self._emit_message_telemetry({
                **telemetry_base,
                "outcome": "error",
                "latency_ms": latency_ms,
                "error": str(e)[:240],
            })
            run_record = self._record_run_outcome(session_key, source, "error", latency_ms)
            run_id = str(run_record.get("run_id") or "").strip()
            if run_record.get("stall_alerted") and run_id:
                elapsed_seconds = max(0, int(float(run_record.get("finished_at", 0) or 0) - float(run_record.get("started_at", 0) or 0)))
                await self._send_ops_stall_recovered_notice(
                    run_id=run_id,
                    session_key=session_key,
                    source=source,
                    elapsed_seconds=elapsed_seconds,
                    outcome="error",
                )
            return (
                "Sorry, I encountered an unexpected error. "
                "The details have been logged for debugging. "
                "Try again or use /reset to start a fresh session."
            )
        finally:
            # Clear session env
            self._clear_session_env()
    
    async def _handle_reset_command(self, event: MessageEvent) -> str:
        """Handle /new or /reset command."""
        source = event.source
        
        # Get existing session key
        session_key = self.session_store._generate_session_key(source)
        
        # Flush memories in the background (fire-and-forget) so the user
        # gets the "Session reset!" response immediately.
        try:
            old_entry = self.session_store._entries.get(session_key)
            if old_entry:
                asyncio.create_task(self._async_flush_memories(old_entry.session_id))
        except Exception as e:
            logger.debug("Gateway memory flush on reset failed: %s", e)
        
        # Clear runtime state (configurable) and reset the session
        if self.config.session_lifecycle.clear_runtime_on_reset:
            self._clear_session_runtime_state(session_key)
        new_entry = self.session_store.reset_session(session_key)
        
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

    async def _handle_ops_command(self, event: MessageEvent) -> str:
        """Handle /ops command -- live reliability/throughput snapshot."""
        args = (event.get_command_args() if event else "") or ""
        tokens = args.strip().split()

        if tokens and tokens[0].lower() == "debug":
            selector = tokens[1].strip() if len(tokens) > 1 else ""
            if not selector:
                return "Usage: `/ops debug <run_id|latest|latest-main|latest-thread>`"

            run_id = self._resolve_ops_debug_selector(selector)
            if not run_id:
                selector_mode = selector.lower()
                if selector_mode == "latest-main":
                    return "No main-channel runs available yet. Use `/ops` to inspect recent run IDs."
                if selector_mode == "latest-thread":
                    return "No thread runs available yet. Use `/ops` to inspect recent run IDs."
                return "No runs available yet. Use `/ops` after at least one run completes."

            run = self._find_ops_run(run_id)
            if not run:
                return f"Run `{selector}` not found. Use `/ops` to see recent run IDs."

            started_at = run.get("started_at")
            finished_at = run.get("finished_at")
            lines = [
                "🧪 **Ops Run Debug**",
                f"- Run: `{run_id}`",
                f"- Status: {run.get('status', 'unknown')}",
                f"- Outcome: {run.get('outcome', 'in_progress')}",
                f"- Thread: {'yes' if run.get('thread') else 'no'}",
                f"- Session: `{run.get('session_key', 'unknown')}`",
                f"- Platform: `{run.get('platform', 'unknown')}`",
                f"- Chat: `{run.get('chat_id', '')}`",
                f"- Thread ID: `{run.get('thread_id', '')}`",
                f"- Started: {self._format_timestamp_iso(float(started_at) if started_at else None)}",
                f"- Finished: {self._format_timestamp_iso(float(finished_at) if finished_at else None)}",
                f"- Elapsed: {self._format_duration(run.get('elapsed_seconds', 0))}",
                f"- Latency: {int(run.get('latency_ms', 0) or 0)} ms",
            ]
            return "\n".join(lines)

        ops = self._build_ops_snapshot()
        portfolio = self._load_exec_portfolio_snapshot()

        blocked = portfolio.get("blocked", [])
        health = portfolio.get("health", "unknown")
        status_line = portfolio.get("status_line", "")

        lines = [
            "🛟 **Hermes Ops Board (live)**",
            f"`window={ops.get('window_hours')}h | success={ops.get('success_rate', 0):.1f}% | ingest={health}`",
            "",
            "**Execution**",
            f"- Started: {ops.get('started', 0)}",
            f"- Completed: {ops.get('completed', 0)}",
            f"- Failed: {ops.get('failed', 0)}",
            f"- Queue depth: {ops.get('queue_depth', 0)}",
            "",
            "**Concurrency (active now)**",
            f"- Total: {ops.get('active_total', 0)}",
            f"- Main: {ops.get('active_main', 0)}",
            f"- Thread: {ops.get('active_thread', 0)}",
            "",
            "**Latency (P95, successful runs)**",
            f"- Main: {ops.get('p95_latency_main_ms', 0)} ms",
            f"- Thread: {ops.get('p95_latency_thread_ms', 0)} ms",
            "",
            "**Stalls**",
            (
                f"- Active stalled (>{ops.get('stall_threshold_seconds', 0)}s): "
                f"{ops.get('stalled_active', 0)}"
            ),
            f"- Longest active stall: {self._format_duration(ops.get('stalled_longest_seconds', 0))}",
            "",
            "**Freshness / Sync**",
            f"- Last success: {self._format_timestamp_iso(ops.get('last_success_ts'))}",
            f"- Snapshot: `{status_line}`",
        ]

        recent_runs = ops.get("recent_runs") or []
        if recent_runs:
            lines.extend(["", "**Recent runs**"])
            for item in recent_runs:
                lines.append(
                    f"- `{item.get('run_id')}` · {item.get('outcome')} · "
                    f"{'thread' if item.get('thread') else 'main'} · {int(item.get('latency_ms', 0) or 0)} ms"
                )

        if blocked:
            lines.append("- Top blocker: " + str(blocked[0]))

        return "\n".join(lines)

    def _load_exec_portfolio_snapshot(self) -> dict[str, Any]:
        """Load execution-owner snapshot from local report artifacts with freshness metadata."""
        import json

        reports_dir = _hermes_home / "reports"
        kb_dir = _hermes_home / "kb"
        now_ts = time.time()

        def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
            exists = path.exists()
            meta: dict[str, Any] = {
                "path": str(path),
                "exists": exists,
                "read_ok": False,
                "age_seconds": None,
            }
            if not exists:
                return {}, meta

            try:
                stat = path.stat()
                meta["age_seconds"] = max(0, int(now_ts - stat.st_mtime))
            except Exception:
                meta["age_seconds"] = None

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    meta["read_ok"] = True
                    return payload, meta
            except Exception:
                pass
            return {}, meta

        github, github_meta = _read_json(reports_dir / "github_sync_latest.json")
        bookmarks, bookmarks_meta = _read_json(kb_dir / "twitter_bookmarks_state.json")
        vector, vector_meta = _read_json(kb_dir / "twitter_vector_state.json")

        counts = github.get("counts", {}) if isinstance(github, dict) else {}
        freshness = bookmarks.get("freshness", {}) if isinstance(bookmarks, dict) else {}
        ingest = github.get("ingest", {}) if isinstance(github.get("ingest", {}), dict) else {}
        projects = github.get("projects_v2", {}) if isinstance(github.get("projects_v2", {}), dict) else {}
        event_source = str(ingest.get("event_source", "unknown"))

        def _to_int(value: Any, default: int = 0) -> int:
            try:
                if value is None or value == "":
                    return default
                return int(value)
            except Exception:
                return default

        ages = [
            m.get("age_seconds")
            for m in (github_meta, bookmarks_meta, vector_meta)
            if m.get("age_seconds") is not None
        ]
        snapshot_age_seconds = max(ages) if ages else None

        def _format_age(age_seconds: Optional[int]) -> str:
            if age_seconds is None:
                return "unknown"
            if age_seconds < 60:
                return f"{age_seconds}s"
            if age_seconds < 3600:
                return f"{age_seconds // 60}m"
            return f"{age_seconds // 3600}h"

        # Ingest reliability contract
        checkpoint_id = str(ingest.get("processor_checkpoint", "")).strip()
        checkpoint_age_minutes = _to_int(ingest.get("checkpoint_age_minutes"), default=-1)
        last_event_id = str(ingest.get("last_event_id", "")).strip()
        replay_backlog = _to_int(ingest.get("replay_backlog"), default=0)
        dlq_depth = _to_int(ingest.get("dlq_depth"), default=0)
        duplicates_dropped = _to_int(ingest.get("duplicate_events_dropped"), default=0)
        idempotency = ingest.get("idempotency", {}) if isinstance(ingest.get("idempotency", {}), dict) else {}
        idempotency_enabled = bool(idempotency.get("enabled", False))

        # GitHub Projects v2 field-model signals
        field_drift_count = _to_int(projects.get("field_drift_count"), default=0)
        unmapped_labels = _to_int(projects.get("unmapped_labels"), default=0)
        projects_health = str(projects.get("health", "unknown"))
        missing_fields = projects.get("missing_fields", []) if isinstance(projects.get("missing_fields", []), list) else []
        canonical_fields = projects.get("canonical_fields", []) if isinstance(projects.get("canonical_fields", []), list) else []

        warnings: list[str] = []
        missing = [
            name
            for name, meta in (
                ("github_sync_latest.json", github_meta),
                ("twitter_bookmarks_state.json", bookmarks_meta),
                ("twitter_vector_state.json", vector_meta),
            )
            if not meta.get("exists") or not meta.get("read_ok")
        ]
        if missing:
            warnings.append(f"Missing/unreadable artifacts: {', '.join(missing)}")

        if event_source == "polling":
            warnings.append("GitHub webhook stream inactive; ingest is on polling fallback.")

        if snapshot_age_seconds is not None and snapshot_age_seconds > 15 * 60:
            warnings.append(
                f"Snapshot is stale ({_format_age(snapshot_age_seconds)} old; target <= 15m)."
            )

        if not checkpoint_id:
            warnings.append("Ingest checkpoint missing: no processor checkpoint ID in github_sync_latest.json")
        if not last_event_id:
            warnings.append("Ingest event cursor missing: no last_event_id recorded for replay safety")
        if checkpoint_age_minutes >= 0 and checkpoint_age_minutes > 15:
            warnings.append(f"Ingest checkpoint is stale ({checkpoint_age_minutes}m since advance; target <= 15m).")
        if not idempotency_enabled:
            warnings.append("Idempotency guard not enabled for ingest pipeline (duplicate event risk).")
        if replay_backlog > 0:
            warnings.append(f"Replay backlog pending: {replay_backlog} events.")
        if dlq_depth > 0:
            warnings.append(f"DLQ has pending failed events: {dlq_depth}.")

        if missing_fields:
            warnings.append(f"Projects v2 field drift detected: missing {', '.join(str(x) for x in missing_fields)}")
        if field_drift_count > 0:
            warnings.append(f"Projects v2 drift count: {field_drift_count}")
        if unmapped_labels > 0:
            warnings.append(f"Label taxonomy drift: {unmapped_labels} unmapped labels still active.")

        owner_open_prs = _to_int(counts.get("owner_open_prs"), default=0)
        bookmarks_updated = _to_int(bookmarks.get("updated"), default=0)
        bookmarks_inserted = _to_int(bookmarks.get("inserted"), default=0)

        bridge_suggestions: list[str] = []
        if owner_open_prs > 0:
            bridge_suggestions.append("Link open owner PRs to matching bookmarked context before merge.")
        if bookmarks_updated > 0 or bookmarks_inserted > 0:
            bridge_suggestions.append("Run semantic bookmark query for active GitHub items to surface relevant prior decisions.")
        if vector.get("errors", 0):
            bridge_suggestions.append("Resolve vector indexing errors so cross-app retrieval stays reliable.")

        now_items = [
            f"Owner open PRs: {counts.get('owner_open_prs', 0)}",
            f"Review queue: {counts.get('review_requested_prs', 0)}",
            f"Assigned issues: {counts.get('assigned_open_issues', 0)}",
            f"Bookmarks fetched: {bookmarks.get('fetched_items', 0)}",
            f"Bookmarks freshness stale_minutes: {freshness.get('stale_minutes')}",
            f"Ingest: checkpoint={'set' if checkpoint_id else 'missing'}, replay_backlog={replay_backlog}, dlq={dlq_depth}, idempotency={'on' if idempotency_enabled else 'off'}",
            f"Projects v2: health={projects_health}, field_drift={field_drift_count}, unmapped_labels={unmapped_labels}",
        ]

        if canonical_fields:
            now_items.append(f"Projects canonical fields: {', '.join(str(x) for x in canonical_fields)}")

        blocked_items: list[str] = []
        if bookmarks.get("error"):
            blocked_items.append(f"Bookmarks fetch error: {bookmarks.get('error')}")
        if freshness.get("alert_triggered"):
            blocked_items.append("Bookmarks freshness alert triggered.")
        if vector.get("errors", 0):
            blocked_items.append(f"Vector index errors: {vector.get('errors')}")
        blocked_items.extend(warnings)

        next_items: list[str] = []
        if owner_open_prs > 0:
            next_items.append("Triage owner PR queue to zero (merge or comment unblockers).")
        if event_source == "polling":
            next_items.append("Restore webhook flow and keep polling as fallback only.")
        if freshness.get("sla_met") is False:
            next_items.append("Fix bookmark freshness SLA miss (target <= 15m).")
        if snapshot_age_seconds is not None and snapshot_age_seconds > 15 * 60:
            next_items.append("Fix delayed sync job before using this snapshot for decision-making.")
        if missing:
            next_items.append("Repair missing artifact generation to avoid blind spots in /now, /blocked, and /next.")
        if not idempotency_enabled:
            next_items.append("Enable ingest idempotency keys before trusting replay/retry processing.")
        if replay_backlog > 0:
            next_items.append("Drain replay backlog and verify checkpoint progression is monotonic.")
        if dlq_depth > 0:
            next_items.append("Process DLQ items and add replay-safe recovery for failed events.")
        if not checkpoint_id or not last_event_id:
            next_items.append("Persist processor checkpoints + last_event_id for deterministic replay recovery.")
        if checkpoint_age_minutes >= 0 and checkpoint_age_minutes > 15:
            next_items.append("Advance ingest checkpoint cadence (target checkpoint_age_minutes <= 15).")
        if missing_fields or field_drift_count > 0:
            next_items.append("Apply canonical Projects v2 field schema and backfill missing fields.")
        if unmapped_labels > 0:
            next_items.append("Map legacy labels into Projects v2 fields and retire duplicate label taxonomy.")
        if not next_items:
            next_items.append("No urgent blockers detected. Continue planned roadmap execution.")

        health = "degraded" if blocked_items else "ok"
        status_line = (
            f"source={event_source} | age={_format_age(snapshot_age_seconds)} | health={health}"
        )

        return {
            "now": now_items,
            "blocked": blocked_items,
            "next": next_items,
            "bridge": bridge_suggestions,
            "status_line": status_line,
            "health": health,
            "event_source": event_source,
            "snapshot_age_seconds": snapshot_age_seconds,
            "ingest": ingest,
            "projects_v2": projects,
            "github": github,
            "bookmarks": bookmarks,
            "vector": vector,
            "duplicates_dropped": duplicates_dropped,
        }

    async def _handle_now_command(self, event: MessageEvent) -> str:
        """Handle /now command - show current priorities and bridge opportunities."""
        snap = self._load_exec_portfolio_snapshot()
        lines = ["📍 **Execution Owner — NOW**", f"`{snap.get('status_line', '')}`", ""]
        lines.extend([f"- {x}" for x in snap.get("now", [])])

        bridge = snap.get("bridge", [])
        if bridge:
            lines.extend(["", "🔗 **Cross-app bridge suggestions**"])
            lines.extend([f"- {x}" for x in bridge])

        return "\n".join(lines)

    async def _handle_blocked_command(self, event: MessageEvent) -> str:
        """Handle /blocked command - show blocked items and reasons."""
        snap = self._load_exec_portfolio_snapshot()
        blocked = snap.get("blocked", [])
        if not blocked:
            return "✅ **Blocked**\n\n- none"

        lines = ["🚧 **Blocked**", f"`{snap.get('status_line', '')}`", ""]
        lines.extend([f"- {x}" for x in blocked])
        return "\n".join(lines)

    async def _handle_next_command(self, event: MessageEvent) -> str:
        """Handle /next command - show recommended next actions."""
        snap = self._load_exec_portfolio_snapshot()
        lines = ["⏭️ **Next Actions**", f"`{snap.get('status_line', '')}`", ""]
        lines.extend([f"- {x}" for x in snap.get("next", [])])
        return "\n".join(lines)

    async def _handle_stop_command(self, event: MessageEvent) -> str:
        """Handle /stop command - interrupt a running agent."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        if session_key in self._running_agents:
            agent = self._running_agents[session_key]
            agent.interrupt()
            return "⚡ Stopping the current task... The agent will finish its current step and respond."
        else:
            return "No active task to stop."
    
    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        lines = [
            "📖 **Hermes Commands**\n",
            "`/new` — Start a new conversation",
            "`/reset` — Reset conversation history",
            "`/status` — Show session info",
            "`/ops` — Show live reliability board (queue, active runs, latency, stalls; `/ops debug <run_id|latest|latest-main|latest-thread>` for details)",
            "`/now` — Show active priorities + cross-app bridge suggestions",
            "`/blocked` — Show currently blocked items and reasons",
            "`/next` — Show recommended next actions",
            "`/stop` — Interrupt the running agent",
            "`/model [provider:model]` — Show/change default model (global)",
            "`/ask [model=<provider:model>] [reasoning=<level>] <prompt>` — Run one message with per-query runtime overrides",
            "`/modelpin [provider:model|clear]` — Pin model for this thread/session only",
            "`/reasoning [xhigh|high|medium|low|minimal|none|default]` — Thread-local reasoning override",
            "`/route [auto|command|vision|audio|document|code|analysis|chat]` — Force routing class for this thread",
            "`/runtime` — Show active thread runtime overrides and resolved routing",
            "`/provider` — Show available providers and auth status",
            "`/personality [name]` — Set a personality",
            "`/retry` — Retry your last message",
            "`/undo` — Remove the last exchange",
            "`/sethome` — Set this chat as the home channel",
            "`/compress` — Compress conversation context",
            "`/title [name]` — Set or show the session title",
            "`/resume [name]` — Resume a previously-named session",
            "`/usage` — Show token usage for this session",
            "`/insights [days]` — Show usage insights and analytics",
            "`/reasoning [level|show|hide]` — Set reasoning effort or toggle display",
            "`/rollback [number]` — List or restore filesystem checkpoints",
            "`/background <prompt>` — Run a prompt in a separate background session",
            "`/reload-mcp` — Reload MCP servers from config",
            "`/update` — Update Hermes Agent to the latest version",
            "`/help` — Show this message",
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
    
    async def _handle_model_command(self, event: MessageEvent) -> str:
        """Handle /model command - show or change the current model."""
        import yaml
        from hermes_cli.models import (
            parse_model_input,
            validate_requested_model,
            curated_models_for_provider,
            normalize_provider,
            _PROVIDER_LABELS,
        )

        args = event.get_command_args().strip()
        config_path = _hermes_home / 'config.yaml'

        # Resolve current model and provider from config
        current = os.getenv("HERMES_MODEL") or "anthropic/claude-opus-4.6"
        current_provider = "openrouter"
        try:
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, str):
                    current = model_cfg
                elif isinstance(model_cfg, dict):
                    current = model_cfg.get("default", current)
                    current_provider = model_cfg.get("provider", current_provider)
        except Exception:
            pass

        # Resolve "auto" to the actual provider using credential detection
        current_provider = normalize_provider(current_provider)
        if current_provider == "auto":
            try:
                from hermes_cli.auth import resolve_provider as _resolve_provider
                current_provider = _resolve_provider(current_provider)
            except Exception:
                current_provider = "openrouter"

        # Detect custom endpoint: provider resolved to openrouter but a custom
        # base URL is configured — the user set up a custom endpoint.
        if current_provider == "openrouter" and os.getenv("OPENAI_BASE_URL", "").strip():
            current_provider = "custom"

        if not args:
            provider_label = _PROVIDER_LABELS.get(current_provider, current_provider)
            lines = [
                f"🤖 **Current model:** `{current}`",
                f"**Provider:** {provider_label}",
                "",
            ]
            curated = curated_models_for_provider(current_provider)
            if curated:
                lines.append(f"**Available models ({provider_label}):**")
                for mid, desc in curated:
                    marker = " ←" if mid == current else ""
                    label = f"  _{desc}_" if desc else ""
                    lines.append(f"• `{mid}`{label}{marker}")
                lines.append("")
            lines.append("To change: `/model model-name`")
            lines.append("Switch provider: `/model provider:model-name`")
            return "\n".join(lines)

        # Parse provider:model syntax
        target_provider, new_model = parse_model_input(args, current_provider)
        provider_changed = target_provider != current_provider

        # Resolve credentials for the target provider (for API probe)
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        base_url = "https://openrouter.ai/api/v1"
        if provider_changed:
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                runtime = resolve_runtime_provider(requested=target_provider)
                api_key = runtime.get("api_key", "")
                base_url = runtime.get("base_url", "")
            except Exception as e:
                provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
                return f"⚠️ Could not resolve credentials for provider '{provider_label}': {e}"
        else:
            # Use current provider's base_url from config or registry
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                runtime = resolve_runtime_provider(requested=current_provider)
                api_key = runtime.get("api_key", "")
                base_url = runtime.get("base_url", "")
            except Exception:
                pass

        # Validate the model against the live API
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
            msg = validation.get("message", "Invalid model")
            tip = "\n\nUse `/model` to see available models, `/provider` to see providers" if "Did you mean" not in msg else ""
            return f"⚠️ {msg}{tip}"

        # Persist to config only if validation approves
        if validation.get("persist"):
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                if "model" not in user_config or not isinstance(user_config["model"], dict):
                    user_config["model"] = {}
                user_config["model"]["default"] = new_model
                if provider_changed:
                    user_config["model"]["provider"] = target_provider
                with open(config_path, 'w', encoding="utf-8") as f:
                    yaml.dump(user_config, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                return f"⚠️ Failed to save model change: {e}"

        # Set env vars so the next agent run picks up the change
        os.environ["HERMES_MODEL"] = new_model
        if provider_changed:
            os.environ["HERMES_INFERENCE_PROVIDER"] = target_provider

        provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
        provider_note = f"\n**Provider:** {provider_label}" if provider_changed else ""

        warning = ""
        if validation.get("message"):
            warning = f"\n⚠️ {validation['message']}"

        if validation.get("persist"):
            persist_note = "saved to config"
        else:
            persist_note = "this session only — will revert on restart"
        return f"🤖 Model changed to `{new_model}` ({persist_note}){provider_note}{warning}\n_(takes effect on next message)_"

    async def _handle_modelpin_command(self, event: MessageEvent) -> str:
        """Handle /modelpin command - set thread-local model/provider override."""
        from hermes_cli.models import parse_model_input

        source = event.source
        session_key = self.session_store._generate_session_key(source)
        args = event.get_command_args().strip()
        state = self._session_runtime_overrides.setdefault(session_key, {})

        if not args:
            model = state.get("model")
            provider = state.get("provider")
            if model:
                provider_label = f" ({provider})" if provider else ""
                return f"📌 Thread model pin: `{model}`{provider_label}"
            return "No thread model pin set. Use `/modelpin provider:model` or `/modelpin model-name`."

        if args.lower() in {"clear", "off", "none", "reset", "default"}:
            state.pop("model", None)
            state.pop("provider", None)
            return "✅ Cleared thread model pin. Routing will use policy/default model again."

        default_provider = os.getenv("HERMES_INFERENCE_PROVIDER", "openrouter")
        try:
            provider, model = parse_model_input(args, default_provider)
        except Exception:
            provider, model = default_provider, args

        state["provider"] = provider
        state["model"] = model
        return f"✅ Thread model pinned to `{model}` ({provider})."

    async def _handle_reasoning_command(self, event: MessageEvent) -> str:
        """Handle /reasoning command - set thread-local reasoning override."""
        source = event.source
        session_key = self.session_store._generate_session_key(source)
        args = event.get_command_args().strip().lower()
        state = self._session_runtime_overrides.setdefault(session_key, {})

        if not args:
            current = state.get("reasoning_effort")
            if current:
                return f"🧠 Thread reasoning override: `{current}`"
            if self._reasoning_config:
                if self._reasoning_config.get("enabled") is False:
                    return "🧠 Thread reasoning override: none (default config disables reasoning)."
                return f"🧠 Thread reasoning override: default (`{self._reasoning_config.get('effort', 'medium')}`)"
            return "🧠 Thread reasoning override: default (`medium`)."

        valid = {"xhigh", "high", "medium", "low", "minimal", "none", "default", "auto", "clear"}
        if args not in valid:
            return "⚠️ Invalid reasoning value. Use one of: xhigh, high, medium, low, minimal, none, default."

        if args in {"default", "auto", "clear"}:
            state.pop("reasoning_effort", None)
            return "✅ Cleared thread reasoning override (back to default policy)."

        state["reasoning_effort"] = args
        return f"✅ Thread reasoning set to `{args}`."

    async def _handle_route_command(self, event: MessageEvent) -> str:
        """Handle /route command - force task-class routing for this thread."""
        source = event.source
        session_key = self.session_store._generate_session_key(source)
        args = event.get_command_args().strip().lower()
        state = self._session_runtime_overrides.setdefault(session_key, {})

        if not args:
            current = state.get("forced_task_type", "auto")
            return (
                f"🧭 Thread route mode: `{current}`\n"
                "Use `/route auto` to clear, or `/route code|analysis|chat|vision|audio|document|command`."
            )

        if args in {"auto", "default", "clear", "off"}:
            state.pop("forced_task_type", None)
            return "✅ Cleared thread route override (auto classification restored)."

        if args not in self._valid_task_types():
            options = ", ".join(sorted(self._valid_task_types()))
            return f"⚠️ Invalid route class `{args}`. Valid: {options}."

        state["forced_task_type"] = args
        return f"✅ Thread route class forced to `{args}`."

    async def _handle_runtime_command(self, event: MessageEvent) -> str:
        """Handle /runtime command - show resolved runtime policy for this thread."""
        source = event.source
        session_key = self.session_store._generate_session_key(source)
        overrides = (self._session_runtime_overrides.get(session_key, {}) or {}).copy()

        forced_task_type = str(overrides.get("forced_task_type") or "").strip().lower()
        probe_task_type = forced_task_type if forced_task_type in self._valid_task_types() else "chat"
        resolved = self._resolve_runtime_policy(session_key=session_key, task_type=probe_task_type)

        def _fmt_reasoning(cfg: dict | None) -> str:
            if not cfg:
                return "default"
            if cfg.get("enabled") is False:
                return "none"
            return str(cfg.get("effort") or "default")

        lines = [
            "⚙️ **Thread Runtime Policy**",
            f"**Session Key:** `{session_key}`",
            "",
            "**Overrides (thread-local):**",
            f"- model pin: `{overrides.get('model') or 'none'}`",
            f"- provider pin: `{overrides.get('provider') or 'none'}`",
            f"- reasoning override: `{overrides.get('reasoning_effort') or 'none'}`",
            f"- route override: `{overrides.get('forced_task_type') or 'auto'}`",
            "",
            "**Resolved (next turn):**",
            f"- effective task type: `{resolved.get('effective_task_type') or 'chat'}`",
            f"- model: `{resolved.get('model') or 'unset'}`",
            f"- provider: `{resolved.get('provider') or 'default'}`",
            f"- reasoning: `{_fmt_reasoning(resolved.get('reasoning_config'))}`",
        ]

        fallback = resolved.get("fallback_model")
        if isinstance(fallback, dict) and fallback.get("provider") and fallback.get("model"):
            lines.append(f"- fallback: `{fallback['provider']}:{fallback['model']}`")

        return "\n".join(lines)

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

            def run_sync():
                agent = AIAgent(
                    model=model,
                    **runtime_kwargs,
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    reasoning_config=self._reasoning_config,
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

    async def _handle_reasoning_display_command(self, event: MessageEvent) -> str:
        """Legacy reasoning settings handler (unused).

        Kept for backward compatibility with older call sites; primary `/reasoning`
        behavior is defined earlier as thread-local runtime override.

        Usage:
            /reasoning              Show current effort level and display state
            /reasoning <level>      Set reasoning effort (none, low, medium, high, xhigh)
            /reasoning show|on      Show model reasoning in responses
            /reasoning hide|off     Hide model reasoning from responses
        """
        import yaml

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / "config.yaml"

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
            return "🧠 ✓ Reasoning display: **ON**\nModel thinking will be shown before each response."

        if args in ("hide", "off"):
            self._show_reasoning = False
            _save_config_key("display.show_reasoning", False)
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
            return f"🧠 ✓ Reasoning effort set to `{effort}` (saved to config)\n_(takes effect on next message)_"
        else:
            return f"🧠 ✓ Reasoning effort set to `{effort}` (this session only)"

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)

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
                lambda: tmp_agent._compress_context(msgs, "", approx_tokens=approx_tokens),
            )

            self.session_store.rewrite_transcript(session_entry.session_id, compressed)
            # Reset stored token count — transcript changed, old value is stale
            self.session_store.update_session(
                session_entry.session_key, last_prompt_tokens=0,
            )
            new_count = len(compressed)
            new_tokens = estimate_messages_tokens_rough(compressed)

            return (
                f"🗜️ Compressed: {original_count} → {new_count} messages\n"
                f"~{approx_tokens:,} → ~{new_tokens:,} tokens"
            )
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
            # Show the current title
            title = self._session_db.get_session_title(session_id)
            if title:
                return f"📌 Session title: **{title}**"
            else:
                return "No title set. Usage: `/title My Session Name`"

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — switch to a previously-named session."""
        if not self._session_db:
            return "Session database not available."

        source = event.source
        session_key = self.session_store._generate_session_key(source)
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
            asyncio.create_task(self._async_flush_memories(current_entry.session_id))
        except Exception as e:
            logger.debug("Memory flush on resume failed: %s", e)

        # Clear runtime state before switching this key to a different session ID
        if self.config.session_lifecycle.clear_runtime_on_resume:
            self._clear_session_runtime_state(session_key)

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
        session_key = self.session_store._generate_session_key(source)

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

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update Hermes Agent to the latest version.

        Spawns ``hermes update`` in a separate systemd scope so it survives the
        gateway restart that ``hermes update`` triggers at the end.  A marker
        file is written so the *new* gateway process can notify the user of the
        result on startup.
        """
        import json
        import shutil
        import subprocess
        from datetime import datetime

        project_root = Path(__file__).parent.parent.resolve()
        git_dir = project_root / '.git'

        if not git_dir.exists():
            return "✗ Not a git repository — cannot update."

        hermes_bin = shutil.which("hermes")
        if not hermes_bin:
            return "✗ `hermes` command not found on PATH."

        # Write marker so the restarted gateway can notify this chat
        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"
        pending = {
            "platform": event.source.platform.value,
            "chat_id": event.source.chat_id,
            "user_id": event.source.user_id,
            "timestamp": datetime.now().isoformat(),
        }
        pending_path.write_text(json.dumps(pending))

        # Spawn `hermes update` in a separate cgroup so it survives gateway
        # restart.  systemd-run --user --scope creates a transient scope unit.
        update_cmd = f"{hermes_bin} update > {output_path} 2>&1"
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
            return f"✗ Failed to start update: {e}"

        return "⚕ Starting Hermes update… I'll notify you when it's done."

    async def _send_update_notification(self) -> None:
        """If the gateway is starting after a ``/update``, notify the user."""
        import json
        import re as _re

        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"

        if not pending_path.exists():
            return

        try:
            pending = json.loads(pending_path.read_text())
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")

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
                    # Truncate if too long for a single message
                    if len(output) > 3500:
                        output = "…" + output[-3500:]
                    msg = f"✅ Hermes update finished — gateway restarted.\n\n```\n{output}\n```"
                else:
                    msg = "✅ Hermes update finished — gateway restarted successfully."
                await adapter.send(chat_id, msg)
                logger.info("Sent post-update notification to %s:%s", platform_str, chat_id)
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            pending_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def _set_session_env(self, context: SessionContext) -> None:
        """Set environment variables for the current session."""
        os.environ["HERMES_SESSION_PLATFORM"] = context.source.platform.value
        os.environ["HERMES_SESSION_CHAT_ID"] = context.source.chat_id
        if context.source.chat_name:
            os.environ["HERMES_SESSION_CHAT_NAME"] = context.source.chat_name
    
    def _clear_session_env(self) -> None:
        """Clear session environment variables."""
        for var in ["HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_CHAT_NAME"]:
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
        Auto-transcribe user voice/audio messages using OpenAI Whisper API
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            The enriched message string with transcriptions prepended.
        """
        from tools.transcription_tools import transcribe_audio
        import asyncio

        enriched_parts = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(transcribe_audio, path)
                if result["success"]:
                    transcript = result["transcript"]
                    enriched_parts.append(
                        f'[The user sent a voice message~ '
                        f'Here\'s what they said: "{transcript}"]'
                    )
                else:
                    error = result.get("error", "unknown error")
                    if "OPENAI_API_KEY" in error or "VOICE_TOOLS_OPENAI_KEY" in error:
                        enriched_parts.append(
                            "[The user sent a voice message but I can't listen "
                            "to it right now~ VOICE_TOOLS_OPENAI_KEY isn't set up yet "
                            "(';w;') Let them know!]"
                        )
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
        notify_mode = self._load_background_notifications_mode()

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
                            await adapter.send(chat_id, message_text)
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
                        await adapter.send(chat_id, message_text)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)

    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        runtime_policy: Dict[str, Any] | None = None,
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
        
        def progress_callback(tool_name: str, preview: str = None, args: dict = None):
            """Callback invoked by agent when a tool is called."""
            if not progress_queue:
                return
            
            # "new" mode: only report when tool changes
            if progress_mode == "new" and tool_name == last_tool[0]:
                return
            last_tool[0] = tool_name
            
            # Build progress message with primary argument preview
            tool_emojis = {
                "terminal": "💻",
                "process": "⚙️",
                "web_search": "🔍",
                "web_extract": "📄",
                "read_file": "📖",
                "write_file": "✍️",
                "patch": "🔧",
                "search": "🔎",
                "search_files": "🔎",
                "list_directory": "📂",
                "image_generate": "🎨",
                "text_to_speech": "🔊",
                "browser_navigate": "🌐",
                "browser_click": "👆",
                "browser_type": "⌨️",
                "browser_snapshot": "📸",
                "browser_scroll": "📜",
                "browser_back": "◀️",
                "browser_press": "⌨️",
                "browser_close": "🚪",
                "browser_get_images": "🖼️",
                "browser_vision": "👁️",
                "moa_query": "🧠",
                "mixture_of_agents": "🧠",
                "vision_analyze": "👁️",
                "skill_view": "📚",
                "skills_list": "📋",
                "todo": "📋",
                "memory": "🧠",
                "session_search": "🔍",
                "send_message": "📨",
                "schedule_cronjob": "⏰",
                "list_cronjobs": "⏰",
                "remove_cronjob": "⏰",
                "execute_code": "🐍",
                "delegate_task": "🔀",
                "clarify": "❓",
                "skill_manage": "📝",
            }
            emoji = tool_emojis.get(tool_name, "⚙️")
            
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
                # Truncate preview to keep messages clean while preserving more context
                # in Discord progress lines (80 -> 240 chars).
                if len(preview) > 240:
                    preview = preview[:237] + "..."
                msg = f'{emoji} {tool_name}: "{preview}"'
            else:
                msg = f"{emoji} {tool_name}..."
            
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
            can_edit = True          # False once an edit fails permanently
            pending_since_last_flush = 0
            last_flush = 0.0
            min_flush_interval = 1.2  # avoid Discord PATCH 429 storms on rapid tool bursts

            while True:
                try:
                    # Drain all queued lines in one pass (batch updates)
                    drained_any = False
                    while True:
                        try:
                            msg = progress_queue.get_nowait()
                            progress_lines.append(msg)
                            pending_since_last_flush += 1
                            drained_any = True
                        except queue.Empty:
                            break

                    # Retry previously buffered lines even when no new events arrived
                    # (e.g. after a transient 429 on edit/send). Without this, progress
                    # output can appear "stuck" until another tool event is emitted.
                    if not drained_any and pending_since_last_flush == 0:
                        await asyncio.sleep(0.25)
                        continue

                    now = asyncio.get_event_loop().time()
                    # Throttle edits/sends to reduce rate-limit risk.
                    if now - last_flush < min_flush_interval:
                        await asyncio.sleep(min_flush_interval - (now - last_flush))

                    if can_edit and progress_msg_id is not None:
                        # Edit existing progress message with batched lines.
                        full_text = "\n".join(progress_lines)
                        result = await adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=progress_msg_id,
                            content=full_text,
                        )
                        if not result.success:
                            # If we're being rate-limited, keep edit mode and retry later.
                            err = (result.error or "").lower()
                            if "429" in err or "rate limit" in err:
                                await asyncio.sleep(1.5)
                                continue

                            # Editing unsupported/hard failure — degrade to append-only sends.
                            can_edit = False
                            tail = progress_lines[-pending_since_last_flush:]
                            for line in tail:
                                await adapter.send(chat_id=source.chat_id, content=line, metadata=_progress_metadata)
                    else:
                        if can_edit:
                            # First tool: send aggregated text as one message, then edit it.
                            full_text = "\n".join(progress_lines)
                            result = await adapter.send(chat_id=source.chat_id, content=full_text, metadata=_progress_metadata)
                            if result.success and result.message_id:
                                progress_msg_id = result.message_id
                            elif not result.success:
                                can_edit = False
                                tail = progress_lines[-pending_since_last_flush:]
                                for line in tail:
                                    await adapter.send(chat_id=source.chat_id, content=line, metadata=_progress_metadata)
                        else:
                            # Editing unavailable: emit only newly queued lines.
                            tail = progress_lines[-pending_since_last_flush:]
                            for line in tail:
                                await adapter.send(chat_id=source.chat_id, content=line, metadata=_progress_metadata)

                    pending_since_last_flush = 0
                    last_flush = asyncio.get_event_loop().time()

                    # Restore typing indicator
                    await asyncio.sleep(0.3)
                    await adapter.send_typing(source.chat_id, metadata=_progress_metadata)

                except asyncio.CancelledError:
                    # Drain remaining queued messages
                    while not progress_queue.empty():
                        try:
                            msg = progress_queue.get_nowait()
                            progress_lines.append(msg)
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

        def run_sync():
            # Pass session_key to process registry via env var so background
            # processes can be mapped back to this gateway session
            os.environ["HERMES_SESSION_KEY"] = session_key or ""

            # Use gateway-configured cap, with env override support.
            max_iterations = 90
            cfg = getattr(self, "config", None)
            if cfg is not None:
                max_iterations = getattr(cfg, "agent_max_iterations", max_iterations)
            
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

            if runtime_policy and runtime_policy.get("model"):
                model = runtime_policy.get("model")

            try:
                provider_override = ""
                if runtime_policy:
                    provider_override = str(runtime_policy.get("provider", "")).strip()
                if provider_override:
                    from hermes_cli.runtime_provider import (
                        resolve_runtime_provider,
                        format_runtime_provider_error,
                    )
                    try:
                        runtime = resolve_runtime_provider(requested=provider_override)
                    except RuntimeError as exc:
                        raise RuntimeError(format_runtime_provider_error(exc)) from exc
                    runtime_kwargs = {
                        "api_key": runtime.get("api_key"),
                        "base_url": runtime.get("base_url"),
                        "provider": runtime.get("provider"),
                        "api_mode": runtime.get("api_mode"),
                    }
                else:
                    runtime_kwargs = _resolve_runtime_agent_kwargs()
            except Exception as exc:
                return {
                    "final_response": f"⚠️ Provider authentication failed: {exc}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }

            pr = self._provider_routing
            agent = AIAgent(
                model=model,
                **runtime_kwargs,
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=enabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=self._prefill_messages or None,
                reasoning_config=(runtime_policy.get("reasoning_config") if runtime_policy else self._reasoning_config),
                providers_allowed=pr.get("only"),
                providers_ignored=pr.get("ignore"),
                providers_order=pr.get("order"),
                provider_sort=pr.get("sort"),
                provider_require_parameters=pr.get("require_parameters", False),
                provider_data_collection=pr.get("data_collection"),
                session_id=session_id,
                tool_progress_callback=progress_callback if tool_progress_enabled else None,
                step_callback=_step_callback_sync if _hooks_ref.loaded_hooks else None,
                platform=platform_key,
                honcho_session_key=session_key,
                session_db=self._session_db,
                fallback_model=(runtime_policy.get("fallback_model") if runtime_policy else self._fallback_model),
            )
            
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
            
            # Return final response, or a message if something went wrong
            final_response = result.get("final_response")

            # Extract last actual prompt token count from the agent's compressor
            _last_prompt_toks = 0
            _agent = agent_holder[0]
            if _agent and hasattr(_agent, "context_compressor"):
                _last_prompt_toks = getattr(_agent.context_compressor, "last_prompt_tokens", 0)

            if not final_response:
                error_msg = f"⚠️ {result['error']}" if result.get("error") else "(No response generated)"
                return {
                    "final_response": error_msg,
                    "messages": result.get("messages", []),
                    "api_calls": result.get("api_calls", 0),
                    "tools": tools_holder[0] or [],
                    "history_offset": len(agent_history),
                    "last_prompt_tokens": _last_prompt_toks,
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
            
            return {
                "final_response": final_response,
                "last_reasoning": result.get("last_reasoning"),
                "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                "api_calls": result_holder[0].get("api_calls", 0) if result_holder[0] else 0,
                "tools": tools_holder[0] or [],
                "history_offset": len(agent_history),
                "last_prompt_tokens": _last_prompt_toks,
            }
        
        # Start progress message sender if enabled
        progress_task = None
        if tool_progress_enabled:
            progress_task = asyncio.create_task(send_progress_messages())
        
        run_ref: Dict[str, Optional[str]] = {"run_id": None}

        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        async def track_agent():
            # Wait for agent to be created
            while agent_holder[0] is None:
                await asyncio.sleep(0.05)
            if session_key:
                self._running_agents[session_key] = agent_holder[0]
                run_ref["run_id"] = self._record_run_started(session_key, source)

        tracking_task = asyncio.create_task(track_agent())

        async def stall_watchdog():
            threshold = max(1, int(getattr(self, "_ops_stall_threshold_seconds", 180) or 180))
            await asyncio.sleep(threshold)
            if not session_key:
                return

            active_runs = getattr(self, "_ops_active_runs", {})
            active = active_runs.get(session_key)
            if not isinstance(active, dict):
                return

            run_id = str(active.get("run_id") or run_ref.get("run_id") or "").strip()
            started_at = float(active.get("started_at", time.time()))
            elapsed_seconds = max(0, int(time.time() - started_at))
            if elapsed_seconds >= threshold:
                await self._send_ops_stall_alert(
                    run_id=run_id,
                    session_key=session_key,
                    source=source,
                    elapsed_seconds=elapsed_seconds,
                )

        stall_watchdog_task = asyncio.create_task(stall_watchdog())

        # Monitor for interrupts from the adapter (new messages arriving)
        async def monitor_for_interrupt():
            adapter = self.adapters.get(source.platform)
            if not adapter or not session_key:
                return
            interrupt_key = session_key or source.chat_id
            if hasattr(adapter, "get_interrupt_session_key"):
                interrupt_key = adapter.get_interrupt_session_key(source)
            while True:
                await asyncio.sleep(0.2)  # Check every 200ms
                # Check if adapter has a pending interrupt for this session.
                # Prefer adapter-provided key mapping; fall back to session_key.
                if hasattr(adapter, 'has_pending_interrupt') and adapter.has_pending_interrupt(interrupt_key):
                    agent = agent_holder[0]
                    if agent:
                        pending_event = adapter.get_pending_message(interrupt_key)
                        pending_text = pending_event.text if pending_event else None
                        logger.debug("Interrupt detected from adapter, signaling agent...")
                        agent.interrupt(pending_text)
                        break
        
        interrupt_monitor = asyncio.create_task(monitor_for_interrupt())
        
        try:
            # Run in thread pool to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, run_sync)
            
            # Check if we were interrupted and have a pending message
            result = result_holder[0]
            adapter = self.adapters.get(source.platform)
            
            # Get pending message from adapter if interrupted.
            # Use session_key (not source.chat_id) to match adapter's storage keys.
            pending = None
            interrupt_key = source.chat_id
            if adapter and hasattr(adapter, "get_interrupt_session_key"):
                interrupt_key = adapter.get_interrupt_session_key(source)
            if result and result.get("interrupted") and adapter:
                pending_event = adapter.get_pending_message(interrupt_key)
                if pending_event:
                    pending = pending_event.text
                elif result.get("interrupt_message"):
                    pending = result.get("interrupt_message")
            
            if pending:
                logger.debug("Processing interrupted message: '%s...'", pending[:40])
                
                # Clear the adapter's interrupt event so the next _run_agent call
                # doesn't immediately re-trigger the interrupt before the new agent
                # even makes its first API call (this was causing an infinite loop).
                if adapter and hasattr(adapter, '_active_sessions') and interrupt_key in adapter._active_sessions:
                    adapter._active_sessions[interrupt_key].clear()
                
                # Don't send the interrupted response to the user — it's just noise
                # like "Operation interrupted." They already know they sent a new
                # message, so go straight to processing it.
                
                # Now process the pending message with updated history
                updated_history = result.get("messages", history)
                next_policy = self._resolve_runtime_policy(
                    session_key=session_key,
                    task_type=self._classify_task_type(MessageEvent(text=pending, source=source), pending),
                )
                return await self._run_agent(
                    message=pending,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=source,
                    session_id=session_id,
                    session_key=session_key,
                    runtime_policy=next_policy,
                )

            # Explicit queue mode (/queue, queue:, q:) does not interrupt the
            # current run, but it should execute deterministically afterward.
            pending_map = getattr(self, "_pending_messages", {})
            queued = pending_map.pop(session_key, None) if session_key else None
            if queued and result and not result.get("interrupted"):
                logger.debug("Processing queued follow-up for session %s", (session_key or "")[:20])
                updated_history = result.get("messages", history)
                queued_policy = self._resolve_runtime_policy(
                    session_key=session_key,
                    task_type=self._classify_task_type(MessageEvent(text=queued, source=source), queued),
                )
                queued_response = await self._run_agent(
                    message=queued,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=source,
                    session_id=session_id,
                    session_key=session_key,
                    runtime_policy=queued_policy,
                )
                if response and queued_response:
                    return f"{response}\n\n{queued_response}"
                return queued_response or response
        finally:
            # Stop progress sender and interrupt monitor
            if progress_task:
                progress_task.cancel()
            interrupt_monitor.cancel()
            stall_watchdog_task.cancel()

            # Clean up tracking
            tracking_task.cancel()
            if session_key and session_key in self._running_agents:
                del self._running_agents[session_key]
            if session_key:
                getattr(self, "_ops_active_runs", {}).pop(session_key, None)

            # Wait for cancelled tasks
            for task in [progress_task, interrupt_monitor, tracking_task, stall_watchdog_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
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
