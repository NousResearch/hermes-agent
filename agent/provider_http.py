"""HttpTransport abstraction for Hermes provider adapters.

Adapters depend on :class:`HttpTransport` (Protocol), not on any concrete
HTTP library. The V1 concrete implementation is :class:`HttpxTransport`,
which lazy-imports ``httpx`` so the import graph stays light and the
adapter remains usable in environments where ``httpx`` is not installed.

The transport:

* NEVER logs headers, payloads, or responses.
* NEVER mutates the headers or payload it receives.
* NEVER holds secrets beyond the lifetime of a single ``post()`` call
  (Authorization header is passed through, not stored on the instance).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HttpResponse(Protocol):
    """Minimal HTTP response shape used by provider adapters."""

    status_code: int
    text: str

    def json(self) -> Mapping[str, Any]: ...


@runtime_checkable
class HttpTransport(Protocol):
    """Minimal HTTP transport used by provider adapters."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> HttpResponse: ...


# ---------------------------------------------------------------------------
# HttpxTransport
# ---------------------------------------------------------------------------


class HttpxTransport:
    """V1 real HTTP transport. Lazy-imports ``httpx``.

    Tests should NEVER instantiate this directly; they use
    :class:`FakeHttpTransport` (defined in tests).
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        self._timeout_s = float(timeout_s)
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import httpx  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    "httpx is not installed; install httpx or inject a different HttpTransport"
                ) from exc
            self._client = httpx.Client(timeout=self._timeout_s)
        return self._client

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> HttpResponse:
        client = self._ensure_client()
        # Do not mutate caller-provided mappings; httpx accepts Mapping.
        response = client.post(url, headers=dict(headers), json=dict(json), timeout=float(timeout))
        return _HttpxResponseAdapter(response)  # type: ignore[return-value]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class _HttpxResponseAdapter:
    """Adapts an httpx.Response to the :class:`HttpResponse` Protocol."""

    def __init__(self, response: Any) -> None:
        self._response = response

    @property
    def status_code(self) -> int:
        return int(self._response.status_code)

    @property
    def text(self) -> str:
        return str(self._response.text)

    def json(self) -> Mapping[str, Any]:
        data = self._response.json()
        if not isinstance(data, Mapping):
            # httpx.json() can return list/None/etc.; we always want Mapping.
            return {}
        return data