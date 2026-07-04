"""HTTP client and error handling for the OpenViking memory plugin."""

from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from typing import Optional

from .constants import _DEFAULT_AGENT, _TIMEOUT

class _OpenVikingHTTPError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _sanitize_openviking_error_message(message: str, status_code: Optional[int] = None) -> str:
    text = (message or "").strip()
    status = f"HTTP {status_code}" if status_code else "HTTP error"
    looks_like_html = bool(re.search(r"^\s*<(!doctype|html|head|body)\b", text, flags=re.IGNORECASE))
    if looks_like_html:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1)).strip()
            if "|" in title:
                title = title.split("|", 1)[1].strip()
            if status_code and title.startswith(f"{status_code}:"):
                title = title.split(":", 1)[1].strip()
            if title:
                return f"{status}: {title}"
        return f"{status}: OpenViking endpoint returned an HTML error page."

    if len(text) > 300:
        return text[:297].rstrip() + "..."
    return text or status


def _format_openviking_exception(error: Exception) -> str:
    status_code = None
    if isinstance(error, _OpenVikingHTTPError):
        status_code = error.status_code
    else:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    return _sanitize_openviking_error_message(str(error), status_code)




def _get_httpx():
    """Lazy import httpx."""
    try:
        import httpx
        return httpx
    except ImportError:
        return None


class _VikingClient:
    """Thin HTTP client for the OpenViking REST API."""

    def __init__(self, endpoint: str, api_key: str = "",
                 account: Optional[str] = None, user: Optional[str] = None,
                 agent: Optional[str] = None):
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        # Account/user are local/trusted-mode tenant identity. API-key requests
        # omit these headers by default; trusted-mode retry may send them only
        # after OpenViking explicitly asks for asserted tenant identity.
        self._account = account or os.environ.get("OPENVIKING_ACCOUNT", "default")
        self._user = user or os.environ.get("OPENVIKING_USER", "default")
        self._agent = agent if agent is not None else os.environ.get("OPENVIKING_AGENT", _DEFAULT_AGENT)
        self._httpx = _get_httpx()
        if self._httpx is None:
            raise ImportError("httpx is required for OpenViking: pip install httpx")

    def _headers(self, *, include_tenant: bool | None = None) -> dict:
        if include_tenant is None:
            include_tenant = not bool(self._api_key)

        h = {"Content-Type": "application/json"}
        if self._agent:
            h["X-OpenViking-Actor-Peer"] = self._agent
        if include_tenant:
            if self._account:
                h["X-OpenViking-Account"] = self._account
            if self._user:
                h["X-OpenViking-User"] = self._user
        if self._api_key:
            h["X-API-Key"] = self._api_key
            h["Authorization"] = "Bearer " + self._api_key
        return h

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    def _multipart_headers(self, *, include_tenant: bool | None = None) -> dict:
        headers = self._headers(include_tenant=include_tenant)
        headers.pop("Content-Type", None)
        return headers

    @staticmethod
    def _needs_trusted_identity_retry(exc: Exception) -> bool:
        message = str(exc)
        return (
            "Trusted mode requests must include X-OpenViking-Account" in message
            or "Trusted mode requests must include X-OpenViking-User" in message
            or "Trusted mode requests must include X-OpenViking-Account or explicit account_id" in message
        )

    def _send_with_trusted_identity_retry(self, send, *, multipart: bool = False) -> dict:
        try:
            headers = self._multipart_headers() if multipart else self._headers()
            return self._parse_response(send(headers))
        except Exception as exc:
            if not self._api_key or not self._needs_trusted_identity_retry(exc):
                raise
            headers = (
                self._multipart_headers(include_tenant=True)
                if multipart else self._headers(include_tenant=True)
            )
            return self._parse_response(send(headers))

    def _parse_response(self, resp) -> dict:
        try:
            data = resp.json()
        except Exception:
            data = None

        if resp.status_code >= 400:
            message = _sanitize_openviking_error_message(
                getattr(resp, "text", ""),
                resp.status_code,
            )
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    code = error.get("code", "HTTP_ERROR")
                    message = f"{code}: {error.get('message', message)}"
                    raise _OpenVikingHTTPError(message, resp.status_code)
                if data.get("status") == "error":
                    raise _OpenVikingHTTPError(str(data), resp.status_code)
            raise _OpenVikingHTTPError(message or f"HTTP {resp.status_code}", resp.status_code)

        if isinstance(data, dict) and data.get("status") == "error":
            error = data.get("error")
            if isinstance(error, dict):
                code = error.get("code", "OPENVIKING_ERROR")
                message = error.get("message", "")
                raise RuntimeError(f"{code}: {message}")
            raise RuntimeError(str(data))

        if data is None:
            return {}
        return data

    def get(self, path: str, **kwargs) -> dict:
        timeout = kwargs.pop("timeout", _TIMEOUT)
        return self._send_with_trusted_identity_retry(
            lambda headers: self._httpx.get(
                self._url(path), headers=headers, timeout=timeout, **kwargs
            )
        )

    def post(self, path: str, payload: dict = None, **kwargs) -> dict:
        timeout = kwargs.pop("timeout", _TIMEOUT)
        return self._send_with_trusted_identity_retry(
            lambda headers: self._httpx.post(
                self._url(path), json=payload or {}, headers=headers,
                timeout=timeout, **kwargs
            )
        )

    def delete(self, path: str, **kwargs) -> dict:
        timeout = kwargs.pop("timeout", _TIMEOUT)
        return self._send_with_trusted_identity_retry(
            lambda headers: self._httpx.delete(
                self._url(path), headers=headers, timeout=timeout, **kwargs
            )
        )

    def upload_temp_file(self, file_path: Path) -> str:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

        def _send(headers):
            with file_path.open("rb") as f:
                return self._httpx.post(
                    self._url("/api/v1/resources/temp_upload"),
                    files={"file": (file_path.name, f, mime_type)},
                    headers=headers,
                    timeout=_TIMEOUT,
                )

        data = self._send_with_trusted_identity_retry(_send, multipart=True)
        result = data.get("result", {})
        temp_file_id = result.get("temp_file_id", "")
        if not temp_file_id:
            raise RuntimeError("OpenViking temp upload did not return temp_file_id")
        return temp_file_id

    def health(self) -> bool:
        try:
            resp = self._httpx.get(
                self._url("/health"), headers=self._headers(), timeout=3.0
            )
            return resp.status_code == 200
        except Exception:
            return False

    def health_payload(self) -> dict:
        resp = self._httpx.get(
            self._url("/health"), headers=self._headers(), timeout=3.0
        )
        return self._parse_response(resp)

    def validate_auth(self) -> dict:
        """Validate authenticated OpenViking access without mutating state."""
        return self.get("/api/v1/system/status")

    def validate_root_access(self) -> dict:
        """Validate ROOT access against a read-only admin endpoint."""
        return self.get("/api/v1/admin/accounts")
