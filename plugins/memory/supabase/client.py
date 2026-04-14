from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx


_PREFER_RETURN_REPRESENTATION = "return=representation"


class SupabaseClientError(RuntimeError):
    """Base exception raised for Supabase client failures."""


class SupabaseAuthError(SupabaseClientError):
    """Raised when Supabase rejects the request for auth reasons."""


class SupabaseConstraintError(SupabaseClientError):
    """Raised when Supabase reports a constraint violation."""


class SupabaseClient:
    """Thin PostgREST client for the Supabase-backed memory schema."""

    def __init__(self, url: str, api_key: str, *, timeout: float = 30.0) -> None:
        normalized_url = url.rstrip("/")
        if normalized_url.endswith("/rest/v1"):
            self._base_url = normalized_url
        else:
            self._base_url = f"{normalized_url}/rest/v1"

        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._closed = False

    def select(
        self,
        table: str,
        columns: str = "*",
        filters: Mapping[str, Any] | None = None,
        order: str | Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params = self._build_params(filters)
        params["select"] = columns
        if order:
            params["order"] = ",".join(order) if isinstance(order, Sequence) and not isinstance(order, str) else str(order)
        if limit is not None:
            params["limit"] = str(limit)
        return self._request("GET", table, params=params)

    def insert(self, table: str, data: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._request(
            "POST",
            table,
            json=data,
            headers={"Prefer": _PREFER_RETURN_REPRESENTATION},
        )

    def update(
        self,
        table: str,
        data: dict[str, Any],
        filters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        self._require_filters("update", filters)
        return self._request(
            "PATCH",
            table,
            params=self._build_params(filters),
            json=data,
            headers={"Prefer": _PREFER_RETURN_REPRESENTATION},
        )

    def delete(self, table: str, filters: Mapping[str, Any]) -> list[dict[str, Any]]:
        self._require_filters("delete", filters)
        return self._request(
            "DELETE",
            table,
            params=self._build_params(filters),
            headers={"Prefer": _PREFER_RETURN_REPRESENTATION},
        )

    def upsert(
        self,
        table: str,
        data: dict[str, Any] | list[dict[str, Any]],
        on_conflict: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict
        return self._request(
            "POST",
            table,
            params=params,
            json=data,
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._client.close()
        self._closed = True

    def _request(
        self,
        method: str,
        table: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        url = f"{self._base_url}/{table.lstrip('/')}"
        try:
            response = self._client.request(
                method,
                url,
                params=params,
                json=json,
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise SupabaseClientError(
                f"Network error during {method.upper()} {url}: {exc}"
            ) from exc

        return self._handle_response(response, method=method, url=url)

    def _handle_response(
        self,
        response: httpx.Response,
        *,
        method: str,
        url: str,
    ) -> list[dict[str, Any]]:
        if response.is_success:
            return self._decode_success_payload(response, method=method, url=url)

        raise self._build_error(response, method=method, url=url)

    def _decode_success_payload(
        self,
        response: httpx.Response,
        *,
        method: str,
        url: str,
    ) -> list[dict[str, Any]]:
        if not response.content or not response.content.strip():
            return []

        try:
            payload = response.json()
        except ValueError as exc:
            raise SupabaseClientError(
                f"Supabase returned a non-JSON response for {method.upper()} {url} "
                f"(HTTP {response.status_code})"
            ) from exc

        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]

        raise SupabaseClientError(
            f"Supabase returned an unexpected payload for {method.upper()} {url}: "
            f"{type(payload).__name__}"
        )

    def _build_error(
        self,
        response: httpx.Response,
        *,
        method: str,
        url: str,
    ) -> SupabaseClientError:
        prefix = f"Supabase request failed for {method.upper()} {url} (HTTP {response.status_code})"

        try:
            payload = response.json()
        except ValueError:
            detail = response.text.strip()
            if detail:
                return SupabaseClientError(f"{prefix}: {detail[:200]}")
            return SupabaseClientError(f"{prefix}: non-JSON error response")

        if not isinstance(payload, dict):
            return SupabaseClientError(f"{prefix}: unexpected JSON error payload")

        code = str(payload.get("code") or "")
        message_parts = [
            str(payload.get(field)).strip()
            for field in ("message", "details", "hint")
            if payload.get(field)
        ]
        suffix = ": " + " | ".join(message_parts) if message_parts else ""
        message = f"{prefix}{suffix}"

        if response.status_code in {401, 403}:
            return SupabaseAuthError(message)
        if response.status_code == 409 or code.startswith("23"):
            return SupabaseConstraintError(message)
        return SupabaseClientError(message)

    def _build_params(self, params: Mapping[str, Any] | None) -> dict[str, str]:
        built: dict[str, str] = {}
        if not params:
            return built

        for key, value in params.items():
            if value is None:
                continue
            built[str(key)] = str(value)
        return built

    def _ensure_open(self) -> None:
        if self._closed:
            raise SupabaseClientError("SupabaseClient is closed")

    def _require_filters(self, operation: str, filters: Mapping[str, Any] | None) -> None:
        if filters:
            return
        raise SupabaseClientError(
            f"{operation.capitalize()} requires at least one PostgREST filter"
        )
