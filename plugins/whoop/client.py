"""Thin WHOOP API helper used by Hermes native tools."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx

from hermes_cli.auth import AuthError, resolve_whoop_runtime_credentials


class WHOOPError(RuntimeError):
    """Base WHOOP tool error."""


class WHOOPAuthRequiredError(WHOOPError):
    """Raised when the user needs to authenticate with WHOOP first."""


class WHOOPAPIError(WHOOPError):
    """Structured WHOOP API failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.path: Optional[str] = None


class WHOOPClient:
    def __init__(self) -> None:
        self._runtime = self._resolve_runtime(refresh_if_expiring=True)

    def _resolve_runtime(self, *, force_refresh: bool = False, refresh_if_expiring: bool = True) -> Dict[str, Any]:
        try:
            return resolve_whoop_runtime_credentials(
                force_refresh=force_refresh,
                refresh_if_expiring=refresh_if_expiring,
            )
        except AuthError as exc:
            raise WHOOPAuthRequiredError(str(exc)) from exc

    @property
    def base_url(self) -> str:
        return str(self._runtime.get("base_url") or "").rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._runtime['access_token']}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        allow_retry_on_401: bool = True,
        empty_response: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        response = httpx.request(
            method,
            url,
            headers=self._headers(),
            params=_strip_none(params),
            timeout=30.0,
        )
        if response.status_code == 401 and allow_retry_on_401:
            self._runtime = self._resolve_runtime(force_refresh=True, refresh_if_expiring=True)
            return self.request(
                method,
                path,
                params=params,
                allow_retry_on_401=False,
                empty_response=empty_response,
            )
        if response.status_code >= 400:
            self._raise_api_error(response, method=method, path=path)
        if response.status_code == 204 or not response.content:
            return empty_response or {"success": True, "status_code": response.status_code, "empty": True}
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return {"success": True, "text": response.text}

    def _raise_api_error(self, response: httpx.Response, *, method: str, path: str) -> None:
        detail = response.text.strip()
        message = _friendly_whoop_error_message(
            status_code=response.status_code,
            detail=_extract_whoop_error_detail(response, fallback=detail),
            method=method,
            path=path,
            retry_after=response.headers.get("Retry-After"),
        )
        error = WHOOPAPIError(message, status_code=response.status_code, response_body=detail)
        error.path = path
        raise error

    def get_profile(self) -> Any:
        return self.request("GET", "/user/profile/basic")

    def list_cycles(self, **params: Any) -> Any:
        return self._list_collection("/cycle", **params)

    def get_cycle(self, cycle_id: str) -> Any:
        return self.request("GET", f"/cycle/{cycle_id}")

    def list_recovery(self, **params: Any) -> Any:
        return self._list_collection("/recovery", **params)

    def get_recovery(self, cycle_id: str) -> Any:
        return self.request("GET", f"/cycle/{cycle_id}/recovery")

    def list_sleep(self, **params: Any) -> Any:
        return self._list_collection("/activity/sleep", **params)

    def get_sleep(self, sleep_id: str) -> Any:
        return self.request("GET", f"/activity/sleep/{sleep_id}")

    def list_workouts(self, **params: Any) -> Any:
        return self._list_collection("/activity/workout", **params)

    def get_workout(self, workout_id: str) -> Any:
        return self.request("GET", f"/activity/workout/{workout_id}")

    def _list_collection(
        self,
        path: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        next_token: Optional[str] = None,
        max_pages: int = 1,
    ) -> Any:
        pages = max(1, int(max_pages or 1))
        params: Dict[str, Any] = {
            "start": start,
            "end": end,
            "limit": limit,
        }
        token = next_token
        records: list[Any] = []
        last_payload: Any = None
        for _ in range(pages):
            if token:
                params["nextToken"] = token
            payload = self.request("GET", path, params=params)
            last_payload = payload
            if not isinstance(payload, dict):
                return payload
            page_records = payload.get("records")
            if isinstance(page_records, list):
                records.extend(page_records)
            token = str(payload.get("next_token") or "").strip() or None
            if not token:
                break
        if isinstance(last_payload, dict):
            combined = dict(last_payload)
            combined["records"] = records
            combined["next_token"] = token
            return combined
        return {"records": records, "next_token": token}


def _strip_none(values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not values:
        return {}
    return {key: value for key, value in values.items() if value is not None}


def _extract_whoop_error_detail(response: httpx.Response, *, fallback: str = "") -> str:
    try:
        payload = response.json()
    except Exception:
        return fallback
    if isinstance(payload, dict):
        for key in ("error_description", "message", "detail", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return fallback
    return fallback


def _friendly_whoop_error_message(
    *,
    status_code: int,
    detail: str,
    method: str,
    path: str,
    retry_after: Optional[str] = None,
) -> str:
    prefix = f"WHOOP API {method.upper()} {path} failed with HTTP {status_code}"
    if status_code == 401:
        prefix += "; authentication failed. Run `hermes auth whoop` again if this persists"
    elif status_code == 403:
        prefix += "; requested WHOOP scope may not be granted"
    elif status_code == 429:
        prefix += "; rate limited"
        if retry_after:
            prefix += f" (retry after {retry_after}s)"
    if detail:
        prefix += f": {detail}"
    return prefix
