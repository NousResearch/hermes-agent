"""Thin Spotify Web API helper used by Hermes native tools.

Owns auth (token refresh/401 retry), error mapping and id/URI normalization;
endpoint paths live with their tool handlers in ``tools.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import httpx

from hermes_cli.auth import AuthError, resolve_spotify_runtime_credentials


class SpotifyError(RuntimeError): """Base Spotify tool error."""
class SpotifyAuthRequiredError(SpotifyError): """Raised when the user needs to authenticate with Spotify first."""


class SpotifyAPIError(SpotifyError):
    """Structured Spotify API failure."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, response_body: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code, self.response_body, self.path = status_code, response_body, None


_empty_204 = lambda message: {"status_code": 204, "empty": True, "message": message}  # noqa: E731  explanatory stand-in for a bare 204


class SpotifyClient:
    def __init__(self) -> None:
        self._runtime = self._resolve_runtime(refresh_if_expiring=True)

    def _resolve_runtime(self, *, force_refresh: bool = False, refresh_if_expiring: bool = True) -> Dict[str, Any]:
        try:
            return resolve_spotify_runtime_credentials(force_refresh=force_refresh, refresh_if_expiring=refresh_if_expiring)
        except AuthError as exc:
            raise SpotifyAuthRequiredError(str(exc)) from exc

    @property
    def base_url(self) -> str:
        return str(self._runtime.get("base_url") or "").rstrip("/")

    def request(
        self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, json_body: Optional[Dict[str, Any]] = None,
        allow_retry_on_401: bool = True, empty_response: Optional[Dict[str, Any]] = None,
    ) -> Any:
        response = httpx.request(
            method, f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self._runtime['access_token']}", "Content-Type": "application/json"},
            params=_strip_none(params), json=_strip_none(json_body) if json_body is not None else None, timeout=30.0,
        )
        if response.status_code == 401 and allow_retry_on_401:
            # One forced token refresh, then retry exactly once.
            self._runtime = self._resolve_runtime(force_refresh=True, refresh_if_expiring=True)
            return self.request(method, path, params=params, json_body=json_body, allow_retry_on_401=False)
        if response.status_code >= 400:
            detail = response.text.strip()
            message = _friendly_spotify_error_message(
                status_code=response.status_code, detail=_extract_spotify_error_detail(response, fallback=detail),
                path=path, retry_after=response.headers.get("Retry-After"),
            )
            error = SpotifyAPIError(message, status_code=response.status_code, response_body=detail)
            error.path = path
            raise error
        if response.status_code == 204 or not response.content:
            return empty_response or {"success": True, "status_code": response.status_code, "empty": True}
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return {"success": True, "text": response.text}

    # -- player: reads return an explanatory payload instead of a bare 204 --------

    def get_playback_state(self, *, market: Optional[str] = None) -> Any:
        return self.request("GET", "/me/player", params={"market": market}, empty_response=_empty_204(
            "No active Spotify playback session was found. Open Spotify on a device and start playback, or transfer playback to an available device."
        ))

    def get_currently_playing(self, *, market: Optional[str] = None) -> Any:
        return self.request("GET", "/me/player/currently-playing", params={"market": market}, empty_response=_empty_204(
            "Spotify is not currently playing anything. Start playback in Spotify and try again."
        ))


def _extract_spotify_error_detail(response: httpx.Response, *, fallback: str) -> str:
    """Prefer Spotify's ``{"error": {"message": ...}}`` (or ``{"error": "..."}``) body over raw text."""
    detail = fallback
    try:
        error_obj = response.json().get("error")
        if isinstance(error_obj, dict):
            detail = str(error_obj.get("message") or detail)
        elif isinstance(error_obj, str):
            detail = error_obj
    except Exception:
        pass
    return detail.strip()


def _friendly_spotify_error_message(*, status_code: int, detail: str, path: str, retry_after: Optional[str]) -> str:
    is_playback_path = path.startswith("/me/player")
    if status_code == 401:
        return "Spotify authentication failed or expired. Run `hermes auth spotify` again."
    if status_code == 403:
        if is_playback_path:
            return ("Spotify rejected this playback request. Playback control usually requires a Spotify Premium account "
                    "and an active Spotify Connect device.")
        if "scope" in detail.lower() or "permission" in detail.lower():
            return "Spotify rejected the request because the current auth scope is insufficient. Re-run `hermes auth spotify` to refresh permissions."
        return "Spotify rejected the request. The account may not have permission for this action."
    if status_code == 404:
        return "Spotify could not find an active playback device or player session for this request." if is_playback_path else "Spotify resource not found."
    if status_code == 429:
        return "Spotify rate limit exceeded." + (f" Retry after {retry_after} seconds." if retry_after else "")
    return detail or f"Spotify API request failed with status {status_code}."


_strip_none = lambda payload: {key: value for key, value in (payload or {}).items() if value is not None}  # noqa: E731


def _check_type(item_type: str, expected_type: Optional[str]) -> None:
    if expected_type and item_type != expected_type:
        raise SpotifyError(f"Expected a Spotify {expected_type}, got {item_type}.")


def normalize_spotify_id(value: str, expected_type: Optional[str] = None) -> str:
    """Accept a bare id, ``spotify:<type>:<id>`` URI, or open.spotify.com URL; return the id."""
    cleaned = (value or "").strip()
    if not cleaned:
        raise SpotifyError("Spotify id/uri/url is required.")
    # (type, id) segments of a URI or URL; a bare id (or malformed ref) has no segments and passes through.
    parts = cleaned.split(":")[1:] if cleaned.startswith("spotify:") else []
    if len(parts) < 2 and "open.spotify.com" in cleaned:
        parts = [part for part in urlparse(cleaned).path.split("/") if part]
    if len(parts) >= 2:
        _check_type(parts[0], expected_type)
        return parts[1]
    return cleaned


def normalize_spotify_uri(value: str, expected_type: Optional[str] = None) -> str:
    """Like normalize_spotify_id but returns a URI; bare ids need *expected_type* to become one."""
    cleaned = (value or "").strip()
    if not cleaned:
        raise SpotifyError("Spotify URI/url/id is required.")
    if cleaned.startswith("spotify:"):
        parts = cleaned.split(":")
        if expected_type and len(parts) >= 3:
            _check_type(parts[1], expected_type)
        return cleaned
    item_id = normalize_spotify_id(cleaned, expected_type)
    return f"spotify:{expected_type}:{item_id}" if expected_type else cleaned


def normalize_spotify_uris(values: Iterable[str], expected_type: Optional[str] = None) -> list[str]:
    """Normalize each value, dropping duplicates while keeping first-seen order."""
    uris = list(dict.fromkeys(normalize_spotify_uri(str(value), expected_type) for value in values))
    if not uris:
        raise SpotifyError("At least one Spotify item is required.")
    return uris


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import json  # noqa: F401,E402

def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)
# ---- END PLUGIN-COMPAT ----
