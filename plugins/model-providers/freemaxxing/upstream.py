"""Bounded upstream completion I/O and typed failure classification."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .policy import (
    _MAX_RESPONSE_BODY_BYTES,
    AuthError,
    ClientRequestError,
    ModelNotFoundError,
    RateLimitError,
    TransientError,
    _hermes_user_agent,
    _is_router_model,
    _open_credentialed,
    _parse_retry_after,
)
from .pool import Backend, _refresh_backend_credentials, _resolve_auto_model, pool

def _exhausted_message(last_error: Optional[str]) -> str:
    detail = last_error or pool.exhaustion_detail()
    return f"All backends exhausted. Last error: {detail}"


def _forward(backend: Backend, body: Dict[str, Any]) -> Dict[str, Any]:
    response = _open_response(backend, body)
    try:
        try:
            raw = response.read(_MAX_RESPONSE_BODY_BYTES + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise TransientError(
                f"backend {backend.name} response body was interrupted"
            ) from exc
        if len(raw) > _MAX_RESPONSE_BODY_BYTES:
            raise TransientError(
                f"backend {backend.name} response exceeded "
                f"{_MAX_RESPONSE_BODY_BYTES} bytes"
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransientError(
                f"backend {backend.name} returned invalid completion JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise TransientError(
                f"backend {backend.name} completion response was not an object"
            )
        return payload
    finally:
        try:
            response.close()
        except Exception:
            pass


def _open_stream(backend: Backend, body: Dict[str, Any]):
    return _open_response(backend, body)


def _http_error_text(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read(16_385)
    except Exception:
        return ""
    if len(raw) > 16_384:
        raw = raw[:16_384]
    try:
        return raw.decode("utf-8", errors="replace").lower()
    except Exception:
        return ""


def _looks_like_model_error(code: int, body: str) -> bool:
    if code == 404:
        return True
    if code != 400 or "model" not in body:
        return False
    return any(
        marker in body
        for marker in ("not found", "invalid model", "not available", "unknown")
    )


def _open_response(backend: Backend, body: Dict[str, Any]):
    def attempt(base_url: str, api_key: str):
        if not api_key:
            raise AuthError(f"backend {backend.name} has no credential")
        outgoing = body
        if _is_router_model(str(body.get("model", ""))):
            real_model = _resolve_auto_model(backend)
            if not real_model:
                raise ModelNotFoundError(
                    f"backend {backend.name} has no proven-free model"
                )
            outgoing = dict(body)
            outgoing["model"] = real_model

        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(outgoing).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": _hermes_user_agent(),
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            return _open_credentialed(request, timeout=120.0)
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            try:
                if code == 429:
                    raise RateLimitError(
                        f"backend {backend.name} rate-limited",
                        _parse_retry_after(exc.headers),
                    ) from exc
                body_text = _http_error_text(exc)
                if _looks_like_model_error(code, body_text):
                    raise ModelNotFoundError(
                        f"backend {backend.name} does not serve the selected model"
                    ) from exc
                if code in {401, 403}:
                    raise AuthError(
                        f"backend {backend.name} auth rejected (HTTP {code})"
                    ) from exc
                if 400 <= code < 500:
                    raise ClientRequestError(
                        f"backend {backend.name} rejected request (HTTP {code})"
                    ) from exc
                raise TransientError(
                    f"backend {backend.name} returned HTTP {code}"
                ) from exc
            finally:
                try:
                    exc.close()
                except Exception:
                    pass
        except urllib.error.URLError as exc:
            raise TransientError(
                f"backend {backend.name} unreachable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise TransientError(
                f"backend {backend.name} timed out"
            ) from exc
        except (http.client.HTTPException, ConnectionError, OSError) as exc:
            raise TransientError(
                f"backend {backend.name} connection interrupted"
            ) from exc

    base_url, api_key = backend.credential_snapshot()
    if not api_key and backend.refresh is not None:
        if not _refresh_backend_credentials(backend, require_new=False):
            raise AuthError(f"backend {backend.name} has no credential")
        base_url, api_key = backend.credential_snapshot()

    try:
        return attempt(base_url, api_key)
    except AuthError:
        if backend.refresh is None:
            raise
        before_base, before_key = backend.credential_snapshot()
        with backend.refresh_lock:
            # Another request may already have repaired the credential.
            if (
                backend.api_key != before_key
                or backend.base_url != before_base
            ):
                return attempt(backend.base_url, backend.api_key)
            new_base, new_key = backend.refresh()
            new_base = str(new_base or before_base).rstrip("/")
            new_key = str(new_key or "").strip()
            if (
                new_key
                and (new_key != before_key or new_base != before_base)
            ):
                backend.base_url = new_base
                backend.api_key = new_key
                return attempt(new_base, new_key)
        raise
