from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from ..config import BrokerSettings
from ..core import ToolExecutionError, result_payload


class ZoomProvider:
    def __init__(self, settings: BrokerSettings) -> None:
        self.settings = settings
        self._token = ""
        self._token_expires_at = 0.0

    def _ensure_credentials(self) -> None:
        if not (self.settings.zoom_account_id and self.settings.zoom_client_id and self.settings.zoom_client_secret):
            raise ToolExecutionError(
                "Zoom server-to-server OAuth is not configured on Hermes Spark.",
                category="auth",
            )

    def _access_token(self) -> str:
        if not self.settings.zoom_account_id:
            raise ToolExecutionError(
                "Zoom Account ID is required to mint server-to-server OAuth tokens. "
                "ZOOM_SECRET_TOKEN is for webhooks and is not the OAuth account_id.",
                category="auth",
            )
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        headers = {"Authorization": "Basic placeholder"}
        if self.settings.zoom_client_id and self.settings.zoom_client_secret:
            auth = base64.b64encode(f"{self.settings.zoom_client_id}:{self.settings.zoom_client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {auth}"
        response = httpx.post(
            "https://zoom.us/oauth/token",
            data={"grant_type": "account_credentials", "account_id": self.settings.zoom_account_id},
            headers=headers,
            timeout=20.0,
            trust_env=True,
        )
        if response.status_code >= 400:
            raise ToolExecutionError(
                f"Zoom OAuth {response.status_code}: {response.text[:800]}. "
                "If Zoom credentials live in OneCLI, configure a generic OneCLI secret for host zoom.us "
                "that injects Authorization: Basic <base64(client_id:client_secret)>.",
                category="auth",
            )
        payload = response.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.time() + int(payload.get("expires_in", 3600))
        return self._token

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, json_body: Any | None = None) -> Any:
        response = httpx.request(
            method,
            f"https://api.zoom.us/v2{path}",
            headers={"Authorization": f"Bearer {self._access_token()}"},
            params=params,
            json=json_body,
            timeout=40.0,
            trust_env=True,
        )
        if response.status_code >= 400:
            raise ToolExecutionError(f"Zoom API {response.status_code}: {response.text[:800]}", category="provider")
        return response.json() if response.text else {}

    def meetings_list(self, *, user_id: str = "me", meeting_type: str = "scheduled", page_size: int = 30) -> dict[str, Any]:
        return result_payload(data=self._request("GET", f"/users/{user_id}/meetings", params={"type": meeting_type, "page_size": page_size}))

    def meeting_get(self, *, meeting_id: str) -> dict[str, Any]:
        return result_payload(data=self._request("GET", f"/meetings/{meeting_id}"))

    def meeting_create(self, *, user_id: str = "me", topic: str, start_time: str | None = None, duration_minutes: int | None = None, agenda: str | None = None, dry_run: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"topic": topic, "type": 2}
        if start_time:
            payload["start_time"] = start_time
        if duration_minutes:
            payload["duration"] = duration_minutes
        if agenda:
            payload["agenda"] = agenda
        if dry_run:
            return result_payload(data={"dry_run": True, "user_id": user_id, "payload": payload})
        return result_payload(data=self._request("POST", f"/users/{user_id}/meetings", json_body=payload))

    def recordings_list(self, *, user_id: str = "me", from_date: str, to_date: str, page_size: int = 30) -> dict[str, Any]:
        return result_payload(data=self._request("GET", f"/users/{user_id}/recordings", params={"from": from_date, "to": to_date, "page_size": page_size}))

    def recording_get(self, *, meeting_id: str) -> dict[str, Any]:
        return result_payload(data=self._request("GET", f"/meetings/{meeting_id}/recordings"))

    def recording_transcript_get(self, *, meeting_id: str) -> dict[str, Any]:
        data = self._request("GET", f"/meetings/{meeting_id}/recordings")
        transcripts = [
            item
            for item in data.get("recording_files", [])
            if item.get("file_type") in {"TRANSCRIPT", "CC"}
        ]
        return result_payload(data={"meeting_id": meeting_id, "transcripts": transcripts})

    def users_get(self, *, user_id: str) -> dict[str, Any]:
        return result_payload(data=self._request("GET", f"/users/{user_id}"))
