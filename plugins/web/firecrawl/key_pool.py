from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable


def configured_key_pool() -> tuple[str, ...]:
    from hermes_cli.config import get_env_value
    raw = (get_env_value("FIRECRAWL_API_KEYS") or "").strip()
    if not raw:
        return ()
    message = "FIRECRAWL_API_KEYS must be a JSON array of non-empty strings."
    try:
        keys = json.loads(raw)
    except (ValueError, TypeError):
        raise ValueError(message) from None
    if not isinstance(keys, list) or any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError(message)
    return tuple(dict.fromkeys(k.strip() for k in keys))


def key_pool_configured() -> bool:
    from hermes_cli.config import get_env_value
    return bool((get_env_value("FIRECRAWL_API_KEYS") or "").strip())


def _http_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def credential_cache_key() -> str:
    from hermes_cli.config import get_env_value
    from tools.tool_backend_helpers import read_selection
    values = [get_env_value(name) for name in ("FIRECRAWL_API_KEYS", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL")]
    values.append(read_selection("web"))
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


class FirecrawlKeyPool:
    def __init__(self, keys: tuple[str, ...], api_url: str, factory: Callable[..., Any]):
        self._keys = keys
        self._api_url = api_url
        self._factory = factory
        self._cursor = 0
        self._client = None
        self._lock = threading.Lock()

    def _call(self, method: str, **kwargs: Any) -> Any:
        with self._lock:
            while self._cursor < len(self._keys):
                try:
                    if self._client is None:
                        config = {"api_key": self._keys[self._cursor], "max_retries": 0}
                        if self._api_url:
                            config["api_url"] = self._api_url
                        self._client = self._factory(**config)
                    return getattr(self._client, method)(**kwargs)
                except Exception as exc:
                    status = _http_status(exc)
                    if status != 402:
                        detail = f"HTTP {status}" if status is not None else type(exc).__name__
                        raise RuntimeError(f"Firecrawl {method} failed ({detail}); API key was not rotated.") from None
                    self._cursor += 1
                    self._client = None
            raise RuntimeError(
                "Firecrawl API key pool exhausted (HTTP 402: payment/credits required). "
                "Check billing, then restart Hermes or change FIRECRAWL_API_KEYS to reset the pool."
            )

    def search(self, **kwargs: Any) -> Any:
        return self._call("search", **kwargs)

    def scrape(self, **kwargs: Any) -> Any:
        return self._call("scrape", **kwargs)
