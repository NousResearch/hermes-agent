"""Thin Rasputin HTTP client for the Hermes memory provider scaffold.

V1 intentionally stays small:
- use only the public /health, /search, and /commit endpoints
- fail open on transport or payload issues
- avoid third-party dependencies so the plugin remains import-safe

TODO:
- add richer response schema validation once the provider graduates past scaffold
- add retry/backoff and queue instrumentation if commit volume increases
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from urllib import error, request

_MAX_COMMIT_TEXT_CHARS = 8000

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RasputinClientConfig:
    """Runtime config for talking to Rasputin over HTTP."""

    base_url: str = "http://127.0.0.1:7777"
    timeout_seconds: float = 8.0
    commit_timeout_seconds: float = 20.0
    fail_open: bool = True


class RasputinClient:
    """Best-effort HTTP wrapper around Rasputin's public API."""

    def __init__(self, config: RasputinClientConfig):
        self._config = config
        self._base_url = config.base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        return self._base_url

    def healthcheck(self) -> bool:
        """Return True when Rasputin's health endpoint answers successfully."""
        payload = self._request_json("/health", timeout=3.0, method="GET")
        if payload is None:
            return False
        if isinstance(payload, dict):
            status = str(payload.get("status", "")).strip().lower()
            if status:
                return status in {"ok", "healthy", "pass"}
        return True

    def search(self, query: str, *, limit: int = 8) -> List[Dict[str, Any]]:
        """Search Rasputin and normalize the response into a list of hits.

        Use POST /search so long prompts stay in the request body instead of the
        query string. Rasputin supports both GET and POST search, and POST avoids
        noisy HTTP 414 errors on large recall queries.
        """
        query = (query or "").strip()
        if not query:
            return []

        started = time.monotonic()
        payload = self._request_json(
            "/search",
            timeout=self._config.timeout_seconds,
            method="POST",
            payload={
                "query": query,
                "limit": max(int(limit or 1), 1),
            },
        )
        hits = self._normalize_search_results(payload)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.debug(
            "rasputin_search_ms=%s rasputin_search_hits=%s base_url=%s",
            elapsed_ms,
            len(hits),
            self._base_url,
        )
        return hits

    def commit(self, payload: Mapping[str, Any]) -> bool:
        """POST a commit payload to Rasputin.

        Returns False instead of raising on expected fail-open paths.
        Oversized text is clamped client-side so derived-memory mirroring stays
        quiet and fail-open instead of producing predictable HTTP 400 warnings.
        """
        if not payload:
            return False

        started = time.monotonic()
        response = self._request_json(
            "/commit",
            timeout=self._config.commit_timeout_seconds,
            method="POST",
            payload=self._prepare_commit_payload(payload),
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        success = response is not None
        if success:
            logger.debug("rasputin_commit_ms=%s", elapsed_ms)
        else:
            logger.debug("rasputin_commit_ms=%s rasputin_commit_failures=1", elapsed_ms)
        return success

    def _request_json(
        self,
        path: str,
        *,
        timeout: float,
        method: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Any]:
        """Issue an HTTP request and decode JSON.

        Returns None on failure when fail_open is enabled.
        """
        url = f"{self._base_url}{path}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace").strip()
        except error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            logger.warning(
                "Rasputin %s %s failed with HTTP %s%s",
                method,
                url,
                exc.code,
                f": {detail[:200]}" if detail else "",
            )
            if self._config.fail_open:
                return None
            raise
        except (error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Rasputin %s %s failed: %s", method, url, exc)
            if self._config.fail_open:
                return None
            raise

        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Rasputin returned non-JSON payload for %s %s", method, url)
            if self._config.fail_open:
                return None
            raise

    @staticmethod
    def _prepare_commit_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = dict(payload)
        text = clean.get("text")
        if not isinstance(text, str) or len(text) <= _MAX_COMMIT_TEXT_CHARS:
            return clean

        metadata = clean.get("metadata")
        if isinstance(metadata, Mapping):
            metadata = dict(metadata)
        else:
            metadata = {}
        metadata["rasputin_truncated"] = True
        metadata["rasputin_original_text_length"] = len(text)

        clean["metadata"] = metadata
        clean["text"] = text[:_MAX_COMMIT_TEXT_CHARS]
        return clean

    @staticmethod
    def _normalize_search_results(payload: Any) -> List[Dict[str, Any]]:
        """Normalize several plausible search response envelopes.

        The scaffold tolerates light schema drift because Rasputin is derived,
        not canonical. Unknown shapes safely collapse to an empty result set.
        """
        items: Any = []
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            for key in ("results", "hits", "items", "memories", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    items = value
                    break

        normalized: List[Dict[str, Any]] = []
        for index, item in enumerate(items or []):
            if isinstance(item, dict):
                normalized.append(item)
                continue
            if isinstance(item, str):
                normalized.append(
                    {
                        "id": f"rasputin-hit-{index + 1}",
                        "text": item,
                        "metadata": {},
                    }
                )
        return normalized
