"""
Mission Control Token Reporter for Hermes Agent.

Reports LLM token usage to Mission Control's heartbeat endpoint after each API call.
Also maintains a background heartbeat to keep the agent "online" in Mission Control.

Configuration (from environment):
  MISSION_CONTROL_URL    - Base URL for Mission Control (default: http://localhost:3000)
  MISSION_CONTROL_API_KEY - API key for authentication
  MC_AGENT_NAME          - Agent identifier in Mission Control (default: hermes)

Usage:
  from agent.mc_token_reporter import report_token_usage_async, report_token_usage_sync
  
  # Fire-and-forget (recommended for non-blocking)
  report_token_usage_async(model="gpt-4o", input_tokens=1500, output_tokens=320, task_id=12)
  
  # Synchronous with result
  result = report_token_usage_sync(model="gpt-4o", input_tokens=1500, output_tokens=320)
  
  # Start background heartbeat (call once at agent startup)
  start_heartbeat()
  
  # Stop background heartbeat (call at shutdown)
  stop_heartbeat()
"""

import os
import logging
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Retry configuration
RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff in seconds
MAX_RETRIES = len(RETRY_DELAYS)

# Heartbeat configuration
HEARTBEAT_INTERVAL = 60.0  # seconds between heartbeats (keep agent "online")


@dataclass
class TokenUsageReport:
    """Token usage data for reporting."""
    model: str
    input_tokens: int
    output_tokens: int
    task_id: Optional[int] = None


class MCTokenReporter:
    """Mission Control token usage reporter."""
    
    def __init__(self):
        self._base_url: Optional[str] = None
        self._api_key: Optional[str] = None
        self._agent_id: Optional[str] = None
        self._enabled: bool = False
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()
        self._load_config()
        # Auto-start heartbeat if enabled
        if self._enabled:
            self._start_heartbeat()
    
    def _start_heartbeat(self) -> None:
        """Start the background heartbeat thread."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return  # Already running
        
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.debug("[TokenReporter] Heartbeat thread started")
    
    def _stop_heartbeat(self) -> None:
        """Stop the background heartbeat thread."""
        if self._heartbeat_thread is None:
            return
        
        self._heartbeat_stop_event.set()
        self._heartbeat_thread.join(timeout=5.0)
        self._heartbeat_thread = None
        logger.debug("[TokenReporter] Heartbeat thread stopped")
    
    def _heartbeat_loop(self) -> None:
        """Background thread that sends periodic heartbeats to keep agent online."""
        # Initial delay before first heartbeat
        time.sleep(5.0)
        
        while not self._heartbeat_stop_event.is_set():
            try:
                self._send_heartbeat()
            except Exception as exc:
                logger.debug("[TokenReporter] Heartbeat error: %s", exc)
            
            # Wait for next interval or until stopped
            self._heartbeat_stop_event.wait(timeout=HEARTBEAT_INTERVAL)
    
    def _send_heartbeat(self) -> Dict[str, Any]:
        """
        Send a keep-alive heartbeat to Mission Control.
        
        Returns:
            Dict with "success" and optional "error" keys
        """
        if not self._enabled:
            return {"success": False, "error": "Token reporting disabled"}
        
        url = self._build_heartbeat_url()
        headers = self._build_headers()
        
        try:
            import requests
            response = requests.post(
                url,
                json={},  # Empty body for keep-alive heartbeat
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                logger.debug("[TokenReporter] Heartbeat sent successfully")
                return {"success": True}
            elif response.status_code == 404:
                logger.warning("[TokenReporter] Agent not found: %s", self._agent_id)
                return {"success": False, "error": "Agent not found"}
            elif response.status_code == 403:
                logger.error("[TokenReporter] Invalid API key")
                return {"success": False, "error": "Invalid API key"}
            else:
                logger.debug("[TokenReporter] Heartbeat failed: HTTP %s", response.status_code)
                return {"success": False, "error": f"HTTP {response.status_code}"}
        
        except Exception as exc:
            logger.debug("[TokenReporter] Heartbeat network error: %s", exc)
            return {"success": False, "error": str(exc)}
    
    def _load_config(self) -> None:
        """Load configuration from environment variables."""
        self._base_url = (
            os.environ.get("MISSION_CONTROL_URL") or
            os.environ.get("MC_BASE_URL") or
            ""
        )
        self._api_key = (
            os.environ.get("MISSION_CONTROL_API_KEY") or
            os.environ.get("MC_API_KEY") or
            ""
        )
        self._agent_id = (
            os.environ.get("MC_AGENT_NAME") or
            os.environ.get("HERMES_AGENT_NAME") or
            "hermes"
        )
        
        # Enable only if both URL and API key are present
        self._enabled = bool(self._base_url and self._api_key)
        
        if not self._enabled:
            if self._base_url or self._api_key:
                logger.debug(
                    "[TokenReporter] Partial config: both MISSION_CONTROL_URL "
                    "and MISSION_CONTROL_API_KEY required"
                )
            else:
                logger.debug("[TokenReporter] Token reporting disabled (no config)")
        else:
            logger.debug(
                f"[TokenReporter] Enabled for agent '{self._agent_id}' "
                f"at {self._base_url}"
            )
    
    @property
    def enabled(self) -> bool:
        """Return True if token reporting is enabled."""
        return self._enabled
    
    @property
    def agent_id(self) -> str:
        """Return the configured agent ID."""
        return self._agent_id or "hermes"
    
    def _build_heartbeat_url(self) -> str:
        """Build the heartbeat endpoint URL."""
        base = self._base_url.rstrip("/")
        return f"{base}/api/agents/{self._agent_id}/heartbeat"
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["X-API-Key"] = self._api_key
        return headers
    
    def _send_report(
        self,
        report: TokenUsageReport,
        attempt: int = 0
    ) -> Dict[str, Any]:
        """
        Send a token usage report to Mission Control.
        
        Args:
            report: Token usage data
            attempt: Current retry attempt (0 = first try)
            
        Returns:
            Dict with "success", "recorded", and optional "error" keys
        """
        if not self._enabled:
            return {"success": False, "recorded": False, "error": "Token reporting disabled"}
        
        # Validate required fields
        if not report.model or report.input_tokens < 0 or report.output_tokens < 0:
            error = "Invalid token usage report: model, input_tokens, and output_tokens required"
            logger.error("[TokenReporter] %s", error)
            return {"success": False, "recorded": False, "error": error}
        
        url = self._build_heartbeat_url()
        headers = self._build_headers()
        
        body = {
            "token_usage": {
                "model": report.model,
                "inputTokens": report.input_tokens,
                "outputTokens": report.output_tokens,
                **({"taskId": report.task_id} if report.task_id is not None else {})
            }
        }
        
        try:
            import requests
            response = requests.post(
                url,
                json=body,
                headers=headers,
                timeout=10.0
            )
            
            # Handle specific status codes per contract
            if response.status_code == 200:
                data = response.json() if response.text else {}
                recorded = data.get("token_recorded") is True
                
                if not recorded:
                    logger.debug(
                        "[TokenReporter] Usage sent but token_recorded is not true: %s",
                        data
                    )
                
                return {"success": True, "recorded": recorded}
            
            if response.status_code == 404:
                error = f"Agent not found in Mission Control: {self._agent_id}"
                logger.error("[TokenReporter] %s", error)
                return {"success": False, "recorded": False, "error": error}
            
            if response.status_code == 403:
                error = "Invalid or insufficient API key for token reporting"
                logger.error("[TokenReporter] %s", error)
                return {"success": False, "recorded": False, "error": error}
            
            if response.status_code == 429:
                error = "Rate limit exceeded (30/min) for token reporting"
                logger.warning("[TokenReporter] %s", error)
                
                # Retry with backoff for rate limit
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.debug(
                        "[TokenReporter] Retrying after %.1fs (attempt %d/%d)",
                        delay, attempt + 1, MAX_RETRIES
                    )
                    time.sleep(delay)
                    return self._send_report(report, attempt + 1)
                
                return {"success": False, "recorded": False, "error": error}
            
            # 5xx errors - retry with backoff
            if response.status_code >= 500:
                error = f"Server error {response.status_code}: {response.reason}"
                logger.warning("[TokenReporter] %s", error)
                
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.debug(
                        "[TokenReporter] Retrying after %.1fs (attempt %d/%d)",
                        delay, attempt + 1, MAX_RETRIES
                    )
                    time.sleep(delay)
                    return self._send_report(report, attempt + 1)
                
                return {
                    "success": False,
                    "recorded": False,
                    "error": f"{error} (max retries exceeded)"
                }
            
            # Other 4xx errors - don't retry
            error = f"HTTP {response.status_code}: {response.reason}"
            logger.error("[TokenReporter] %s", error)
            return {"success": False, "recorded": False, "error": error}
        
        except Exception as exc:
            # Network errors - retry with backoff
            error = str(exc)
            logger.warning("[TokenReporter] Network error: %s", error)
            
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                logger.debug(
                    "[TokenReporter] Retrying after %.1fs (attempt %d/%d)",
                    delay, attempt + 1, MAX_RETRIES
                )
                time.sleep(delay)
                return self._send_report(report, attempt + 1)
            
            return {
                "success": False,
                "recorded": False,
                "error": f"Network error: {error} (max retries exceeded)"
            }


# Global reporter instance
_reporter = MCTokenReporter()


def report_token_usage_sync(
    model: str,
    input_tokens: int,
    output_tokens: int,
    task_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Report token usage synchronously.
    
    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-6")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        task_id: Optional Mission Control task ID
        
    Returns:
        Dict with "success", "recorded", and optional "error" keys
    """
    report = TokenUsageReport(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        task_id=task_id
    )
    return _reporter._send_report(report)


def report_token_usage_async(
    model: str,
    input_tokens: int,
    output_tokens: int,
    task_id: Optional[int] = None
) -> None:
    """
    Report token usage asynchronously (fire-and-forget).
    
    This is the recommended method for non-blocking token reporting.
    Errors are logged but not returned.
    
    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-6")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        task_id: Optional Mission Control task ID
    """
    if not _reporter.enabled:
        return
    
    def _report():
        try:
            result = report_token_usage_sync(model, input_tokens, output_tokens, task_id)
            # Log failures but don't raise
            if not result.get("success"):
                logger.debug(
                    "[TokenReporter] Async report failed: %s",
                    result.get("error", "unknown error")
                )
        except Exception as exc:
            logger.debug("[TokenReporter] Async report exception: %s", exc)
    
    # Run in background thread so we don't block
    thread = threading.Thread(target=_report, daemon=True)
    thread.start()


def get_reporter_status() -> Dict[str, Any]:
    """
    Get current reporter configuration status.
    
    Returns:
        Dict with enabled, base_url, agent_id, has_api_key
    """
    return {
        "enabled": _reporter.enabled,
        "base_url": _reporter._base_url,
        "agent_id": _reporter.agent_id,
        "has_api_key": bool(_reporter._api_key),
    }


# Convenience function for extracting usage from OpenAI-style responses
def extract_usage_from_response(response: Any) -> Optional[Dict[str, int]]:
    """
    Extract token usage from an OpenAI-style response object.
    
    Args:
        response: Response object from chat.completions.create()
        
    Returns:
        Dict with "prompt_tokens" and "completion_tokens" or None
    """
    if not response:
        return None
    
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    
    # OpenAI format
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    
    if prompt_tokens is not None and completion_tokens is not None:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    
    # Anthropic format (input_tokens / output_tokens)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    
    if input_tokens is not None and output_tokens is not None:
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }
    
    return None


def report_from_response(response: Any, task_id: Optional[int] = None) -> None:
    """
    Extract usage from a response and report it asynchronously.
    
    This is a convenience function that extracts usage from an OpenAI-style
    response and reports it in one call.
    
    Args:
        response: Response object from chat.completions.create()
        task_id: Optional Mission Control task ID
    """
    if not _reporter.enabled:
        return
    
    usage = extract_usage_from_response(response)
    if not usage:
        logger.debug("[TokenReporter] No usage data in response")
        return
    
    model = getattr(response, "model", "unknown")
    
    report_token_usage_async(
        model=model,
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        task_id=task_id
    )


def start_heartbeat() -> None:
    """
    Start the background heartbeat thread to keep the agent online in Mission Control.
    
    This is called automatically when the module is imported if configuration is valid.
    Call this manually if you need to restart the heartbeat after configuration changes.
    """
    global _reporter
    if _reporter.enabled:
        _reporter._start_heartbeat()
        logger.debug("[TokenReporter] Heartbeat started")
    else:
        logger.debug("[TokenReporter] Cannot start heartbeat - reporting disabled")


def stop_heartbeat() -> None:
    """
    Stop the background heartbeat thread.
    
    Call this at application shutdown for clean exit.
    """
    global _reporter
    _reporter._stop_heartbeat()
    logger.debug("[TokenReporter] Heartbeat stopped")


def cleanup() -> None:
    """
    Clean up resources - stop heartbeat thread.
    
    Call this at application shutdown.
    """
    stop_heartbeat()


# Register cleanup on exit
import atexit
atexit.register(cleanup)
