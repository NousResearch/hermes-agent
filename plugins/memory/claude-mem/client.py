"""HTTP client for the claude-mem local worker.

Wraps the nine endpoints exposed by the claude-mem worker on
``localhost:37777`` (see plan §0.C). Transport only — no caching,
retries, or backoff. Callers in ``__init__.py`` are responsible for
threading and fallback logic (e.g. using ``/api/search`` when a
semantic query is too short).

Error model:
    * :class:`ClaudeMemError`       — HTTP / worker-side errors.
    * :class:`ClaudeMemUnavailable` — worker unreachable (subclass of
      :class:`ClaudeMemError`).

Every method (except ``health()``) catches ``requests.RequestException``
and re-raises as :class:`ClaudeMemUnavailable`. ``health()`` swallows
all exceptions and returns ``False`` so callers can use it as a cheap
probe.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClaudeMemError(Exception):
    """Raised for HTTP or worker-side errors."""


class ClaudeMemUnavailable(ClaudeMemError):
    """Raised when the worker is unreachable or unhealthy."""


# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------

_HEALTH_TIMEOUT = 2.0
_READ_TIMEOUT = 5.0
_WRITE_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ClaudeMemClient:
    """Typed, timeout-safe HTTP client for the claude-mem worker."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:37777",
        timeout: float = 5.0,
    ) -> None:
        # Strip trailing slash so we can always concatenate with a leading-slash path.
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _get(self, path: str, *, params: dict[str, Any] | None = None, timeout: float | None = None) -> dict:
        """GET a worker endpoint and return the decoded JSON body.

        Filters ``None`` values out of ``params`` so optional query args
        don't end up as the literal string "None".
        """
        import requests  # lazy import — is_available() must not trigger HTTP

        clean_params = None
        if params is not None:
            clean_params = {k: v for k, v in params.items() if v is not None}

        try:
            resp = requests.get(
                self._url(path),
                params=clean_params,
                timeout=timeout if timeout is not None else self._timeout,
            )
        except requests.RequestException as e:
            raise ClaudeMemUnavailable(f"claude-mem worker unreachable: {e}") from e

        return self._decode(resp, path)

    def _post(self, path: str, *, json_body: dict[str, Any] | None = None, timeout: float | None = None) -> dict:
        """POST a JSON body to a worker endpoint and return the decoded JSON body."""
        import requests  # lazy import

        clean_body = None
        if json_body is not None:
            clean_body = {k: v for k, v in json_body.items() if v is not None}

        try:
            resp = requests.post(
                self._url(path),
                json=clean_body,
                timeout=timeout if timeout is not None else self._timeout,
            )
        except requests.RequestException as e:
            raise ClaudeMemUnavailable(f"claude-mem worker unreachable: {e}") from e

        return self._decode(resp, path)

    @staticmethod
    def _decode(resp, path: str) -> dict:
        """Validate an HTTP response and return a decoded dict body."""
        if resp.status_code >= 500:
            raise ClaudeMemError(
                f"claude-mem worker error on {path}: "
                f"HTTP {resp.status_code} {resp.text[:200]!r}"
            )
        if resp.status_code >= 400:
            raise ClaudeMemError(
                f"claude-mem request to {path} failed: "
                f"HTTP {resp.status_code} {resp.text[:200]!r}"
            )
        try:
            data = resp.json()
        except ValueError as e:
            raise ClaudeMemError(
                f"claude-mem {path} returned non-JSON body: {resp.text[:200]!r}"
            ) from e

        if not isinstance(data, dict):
            raise ClaudeMemError(
                f"claude-mem {path} returned non-object JSON: {type(data).__name__}"
            )
        return data

    # ------------------------------------------------------------------
    # The nine endpoints
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """GET ``/health``. Never raises — returns ``False`` on any error."""
        try:
            import requests  # lazy import
            resp = requests.get(self._url("/health"), timeout=_HEALTH_TIMEOUT)
            if resp.status_code != 200:
                return False
            try:
                data = resp.json()
            except ValueError:
                return False
            return isinstance(data, dict) and data.get("status") == "ok"
        except Exception:
            # Swallow everything — health() is a probe, not an assertion.
            return False

    def init_session(
        self,
        content_session_id: str,
        *,
        project: str | None = None,
        prompt: str | None = None,
        platform_source: str = "hermes",
        custom_title: str | None = None,
    ) -> dict:
        """POST ``/api/sessions/init``.

        Returns ``{sessionDbId, promptNumber, skipped, reason?, contextInjected}``.
        """
        body: dict[str, Any] = {
            "contentSessionId": content_session_id,
            "project": project,
            "prompt": prompt,
            "platformSource": platform_source,
            "customTitle": custom_title,
        }
        return self._post("/api/sessions/init", json_body=body, timeout=_WRITE_TIMEOUT)

    def post_observation(
        self,
        content_session_id: str,
        *,
        tool_name: str,
        tool_input: dict,
        tool_response: dict,
        cwd: str,
        platform_source: str = "hermes",
    ) -> dict:
        """POST ``/api/sessions/observations`` — queue a tool-use observation."""
        body: dict[str, Any] = {
            "contentSessionId": content_session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "cwd": cwd,
            "platformSource": platform_source,
        }
        return self._post(
            "/api/sessions/observations", json_body=body, timeout=_WRITE_TIMEOUT
        )

    def post_summarize(
        self,
        content_session_id: str,
        *,
        last_assistant_message: str,
        platform_source: str = "hermes",
    ) -> dict:
        """POST ``/api/sessions/summarize`` — queue a session summary."""
        body: dict[str, Any] = {
            "contentSessionId": content_session_id,
            "last_assistant_message": last_assistant_message,
            "platformSource": platform_source,
        }
        return self._post(
            "/api/sessions/summarize", json_body=body, timeout=_WRITE_TIMEOUT
        )

    def complete_session(
        self,
        content_session_id: str,
        *,
        platform_source: str = "hermes",
    ) -> dict:
        """POST ``/api/sessions/complete`` — mark a session complete."""
        body: dict[str, Any] = {
            "contentSessionId": content_session_id,
            "platformSource": platform_source,
        }
        return self._post(
            "/api/sessions/complete", json_body=body, timeout=_WRITE_TIMEOUT
        )

    def search(
        self,
        query: str,
        *,
        project: str | None = None,
        type: str | None = None,  # one of: observations|sessions|prompts|all  # noqa: A002
        limit: int = 20,
        obs_type: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        offset: int = 0,
        order_by: str = "date_desc",
    ) -> dict:
        """GET ``/api/search``.

        Returns a dict like ``{"observations": [...], "total": N, ...}``.
        Callers should NOT assume a flat list.
        """
        params: dict[str, Any] = {
            "query": query,
            "project": project,
            "type": type,
            "limit": limit,
            "obs_type": obs_type,
            "dateStart": date_start,
            "dateEnd": date_end,
            "offset": offset,
            "orderBy": order_by,
        }
        return self._get("/api/search", params=params, timeout=_READ_TIMEOUT)

    def timeline(
        self,
        *,
        anchor: int | None = None,
        query: str | None = None,
        depth_before: int = 3,
        depth_after: int = 3,
        project: str | None = None,
    ) -> dict:
        """GET ``/api/timeline`` — observations around an anchor."""
        params: dict[str, Any] = {
            "anchor": anchor,
            "query": query,
            "depth_before": depth_before,
            "depth_after": depth_after,
            "project": project,
        }
        return self._get("/api/timeline", params=params, timeout=_READ_TIMEOUT)

    def context_semantic(
        self,
        q: str,
        *,
        project: str | None = None,
        limit: int = 5,  # server default is 5, clamped to [1, 20]
    ) -> dict:
        """POST ``/api/context/semantic``.

        Returns ``{context: str, count: int}``.

        Note: the worker silently returns an empty result for queries
        shorter than 20 characters. This client does NOT enforce that
        constraint — transport only. The provider layer is responsible
        for falling back to :meth:`search` when appropriate.
        """
        body: dict[str, Any] = {
            "q": q,
            "project": project,
            "limit": limit,
        }
        return self._post(
            "/api/context/semantic", json_body=body, timeout=_WRITE_TIMEOUT
        )

    def memory_save(
        self,
        text: str,
        *,
        title: str | None = None,
        project: str | None = None,
    ) -> dict:
        """POST ``/api/memory/save`` — manually save a durable fact."""
        body: dict[str, Any] = {
            "text": text,
            "title": title,
            "project": project,
        }
        return self._post(
            "/api/memory/save", json_body=body, timeout=_WRITE_TIMEOUT
        )
