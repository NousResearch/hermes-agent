"""LLM 調用標準化服務 — 層級：Core Logic Layer (Layer 2)

對齊 CODEX.md 重試策略：
- 429 Rate Limit: max_retries=3, backoff_base=1.0 (1s→2s→4s)
- 5xx Server Error: max_retries=3, backoff_base=2.0
- 4xx Client Error: max_retries=0 (fail immediately)
- timeout: max_retries=3, backoff_base=2.0
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .base import LLMServiceInterface

logger = logging.getLogger(__name__)


# ── Custom Exceptions ──────────────────────────────────────────────────────────

class LLMServiceError(Exception):
    """Base exception for LLM service errors."""
    pass


class RateLimitError(LLMServiceError):
    """Raised when provider returns 429."""
    pass


class ServerError(LLMServiceError):
    """Raised when provider returns 5xx."""
    pass


class ClientError(LLMServiceError):
    """Raised when provider returns 4xx (non-rate-limit)."""
    pass


class TimeoutError(LLMServiceError):
    """Raised on request timeout."""
    pass


class MaxRetriesExceeded(LLMServiceError):
    """Raised when all retry attempts are exhausted."""
    pass


# ── LLM Service ────────────────────────────────────────────────────────────────

class LLMService(LLMServiceInterface):
    """LLM 調用標準化服務"""

    # 重試策略：對齊 CODEX.md
    RETRY_STRATEGY = {
        "429": {"max_retries": 3, "backoff_base": 1.0},   # 1s→2s→4s
        "5xx": {"max_retries": 3, "backoff_base": 2.0},
        "4xx": {"max_retries": 0},
        "timeout": {"max_retries": 3, "backoff_base": 2.0},
    }

    def __init__(self, provider: str, base_url: str, api_key: str):
        """Initialize LLM service.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            base_url: Base API URL (trailing slash stripped)
            api_key: API key for authentication
        """
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def health_check(self) -> bool:
        """Provider 連線測試.

        Calls provider /models or lightweight endpoint to verify connectivity.
        Returns True if successful, False otherwise.
        """
        import urllib.request
        import json

        url = f"{self.base_url}/models"
        try:
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self.api_key}")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                logger.debug("health_check(%s): OK - %s", self.provider, data)
                return True
        except urllib.error.HTTPError as e:
            logger.warning("health_check(%s): HTTP %s", self.provider, e.code)
            return False
        except Exception as e:
            logger.warning("health_check(%s): %s", self.provider, e)
            return False

    def get_config(self) -> Dict[str, Any]:
        """取得服務配置."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
        }

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """標準 chat completion（含自動 retry）.

        Implements retry logic per CODEX.md strategy:
        - 429: retries with exponential backoff (1s→2s→4s)
        - 5xx: retries with exponential backoff
        - 4xx: fail immediately
        - timeout: retries with exponential backoff
        """
        error_types = ["429", "5xx", "timeout"]

        for attempt in range(self.RETRY_STRATEGY["429"]["max_retries"] + 1):
            try:
                return self._call_provider(messages, model, temperature=temperature,
                                           max_tokens=max_tokens, stream=stream, **kwargs)
            except RateLimitError as e:
                if "429" in error_types:
                    delay = self.get_retry_delay(attempt, "429")
                    logger.warning("Rate limited (attempt %d/%d), retrying in %.1fs: %s",
                                   attempt + 1, self.RETRY_STRATEGY["429"]["max_retries"] + 1,
                                   delay, e)
                    time.sleep(delay)
                    error_types.remove("429")
                else:
                    raise
            except ServerError as e:
                if "5xx" in error_types:
                    delay = self.get_retry_delay(attempt, "5xx")
                    logger.warning("Server error (attempt %d/%d), retrying in %.1fs: %s",
                                   attempt + 1, self.RETRY_STRATEGY["5xx"]["max_retries"] + 1,
                                   delay, e)
                    time.sleep(delay)
                    error_types.remove("5xx")
                else:
                    raise
            except TimeoutError as e:
                if "timeout" in error_types:
                    delay = self.get_retry_delay(attempt, "timeout")
                    logger.warning("Request timeout (attempt %d/%d), retrying in %.1fs: %s",
                                   attempt + 1, self.RETRY_STRATEGY["timeout"]["max_retries"] + 1,
                                   delay, e)
                    time.sleep(delay)
                    error_types.remove("timeout")
                else:
                    raise
            except ClientError:
                raise  # 4xx 直接失敗，不重試

        raise MaxRetriesExceeded(f"All retry attempts exhausted for chat_completion")

    def _call_provider(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """Call the LLM provider API.

        Raises:
            RateLimitError: On 429 response
            ServerError: On 5xx response
            ClientError: On 4xx response
            TimeoutError: On request timeout
        """
        import urllib.request
        import urllib.error
        import json

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429:
                raise RateLimitError(f"Rate limited (429): {body}")
            elif 500 <= e.code < 600:
                raise ServerError(f"Server error ({e.code}): {body}")
            else:
                raise ClientError(f"Client error ({e.code}): {body}")
        except urllib.error.URLError as e:
            if "timed out" in str(e.reason).lower():
                raise TimeoutError(f"Request timeout: {e}")
            raise ClientError(f"URL error: {e}")

    def classify_error(self, error: Exception) -> str:
        """錯誤分類.

        Args:
            error: The exception to classify

        Returns:
            Error category: "retryable" | "fatal" | "rate_limit"
        """
        if isinstance(error, RateLimitError):
            return "rate_limit"
        elif isinstance(error, (ServerError, TimeoutError)):
            return "retryable"
        elif isinstance(error, ClientError):
            return "fatal"
        elif isinstance(error, MaxRetriesExceeded):
            return "fatal"
        else:
            # Unknown error type - treat as fatal
            logger.warning("Unknown error type %s: %s", type(error).__name__, error)
            return "fatal"

    def get_retry_delay(self, attempt: int, error_type: str) -> float:
        """指數退避計算.

        Args:
            attempt: Current attempt number (0-indexed)
            error_type: Error category ("429", "5xx", "timeout")

        Returns:
            Delay in seconds before next retry
        """
        strategy = self.RETRY_STRATEGY.get(error_type, {})
        base = strategy.get("backoff_base", 1.0)
        return base * (2 ** attempt)