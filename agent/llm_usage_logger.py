"""
LLM Usage Logger — captures request/response pairs for debugging and analysis.

Records every LLM API call (prompt, response, token usage, latency) to a
JSONL file when enabled via config.yaml::

    llm_usage_logger:
        enabled: true
        log_path: ~/.hermes/logs/llm_usage.jsonl   # optional, default below

Useful for prompt engineering, cost tracking, and debugging model behavior.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = "~/.hermes/logs/llm_usage.jsonl"

# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------


@dataclass
class LLMUsageRecord:
    """Single LLM API call record written as a JSONL line."""

    timestamp: str
    provider: str
    model: str
    api_mode: str
    duration_seconds: float

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Request/response — may be truncated or omitted for privacy
    request_messages: Optional[List[Dict[str, Any]]] = None
    response_text: str = ""

    # Error tracking
    error: Optional[str] = None

    # Metadata
    session_id: str = ""
    turn_id: str = ""
    api_request_id: str = ""
    api_call_count: int = 0
    finish_reason: str = ""
    cost_usd: Optional[float] = None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class LLMUsageLogger:
    """Logs LLM API calls to a JSONL file for later analysis.

    Safe to call from any thread — writes are serialised via a lock.
    The config toggle is checked on every ``log()`` call (not cached) so
    editing ``config.yaml`` at runtime takes effect immediately.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._file: Optional[Path] = None  # never None after _ensure_file
        self._handle: Any = None

    # -- public API ---------------------------------------------------------

    def log(
        self,
        *,
        provider: str,
        model: str,
        api_mode: str,
        duration: float,
        request_messages: Optional[List[Dict[str, Any]]] = None,
        response_text: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        error: Optional[str] = None,
        session_id: str = "",
        turn_id: str = "",
        api_request_id: str = "",
        api_call_count: int = 0,
        finish_reason: str = "",
        cost_usd: Optional[float] = None,
    ) -> None:
        """Write one usage record if logging is enabled in config."""
        if not self._is_enabled():
            return

        record = LLMUsageRecord(
            timestamp=_now_iso(),
            provider=provider,
            model=model,
            api_mode=api_mode,
            duration_seconds=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            request_messages=request_messages,
            response_text=response_text,
            error=error,
            session_id=session_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            api_call_count=api_call_count,
            finish_reason=finish_reason,
            cost_usd=cost_usd,
        )
        line = json.dumps(asdict(record), ensure_ascii=False, default=str)

        with self._lock:
            try:
                self._ensure_file()
                self._handle.write(line + "\n")
                self._handle.flush()
            except Exception as exc:
                logger.warning("Failed to write LLM usage log: %s", exc)

    def close(self) -> None:
        """Close the log file (e.g. on agent shutdown)."""
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                except Exception:
                    pass
                self._handle = None

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _is_enabled() -> bool:
        """Read ``llm_usage_logger.enabled`` from config every call.

        Not cached so config edits take effect without restarting the agent.
        """
        try:
            # Lazy import to avoid circular deps at module level.
            from hermes_cli.config import load_config_readonly  # fmt: skip

            config = load_config_readonly()
            if not isinstance(config, dict):
                return False
            section = config.get("llm_usage_logger") or {}
            if not isinstance(section, dict):
                return False
            return bool(section.get("enabled", False))
        except Exception:
            return False

    def _resolve_log_path(self) -> Optional[Path]:
        """Resolve the log file path from config or default."""
        try:
            from hermes_cli.config import load_config_readonly  # fmt: skip

            config = load_config_readonly()
            if isinstance(config, dict):
                section = config.get("llm_usage_logger") or {}
                if isinstance(section, dict):
                    raw = section.get("log_path")
                    if raw and isinstance(raw, str):
                        return Path(os.path.expanduser(raw)).resolve()
        except Exception:
            pass
        return Path(os.path.expanduser(DEFAULT_LOG_PATH)).resolve()

    def _ensure_file(self) -> None:
        if self._handle is not None:
            return
        path = self._resolve_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Append-mode so restarts don't clobber history.
        self._handle = open(path, "a", encoding="utf-8")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_logger: Optional[LLMUsageLogger] = None
_logger_lock = threading.Lock()


def _get_logger() -> LLMUsageLogger:
    """Return the module-level singleton."""
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = LLMUsageLogger()
    return _logger


def log_llm_call(**kwargs: Any) -> None:
    """Convenience: log one LLM call via the singleton."""
    _get_logger().log(**kwargs)


def close_llm_usage_logger() -> None:
    """Close the singleton logger (agent shutdown hook)."""
    inst = _get_logger()
    inst.close()


def _now_iso() -> str:
    """ISO-8601 UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
