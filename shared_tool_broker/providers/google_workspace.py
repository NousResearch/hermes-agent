from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx

from ..config import BrokerSettings
from ..core import ToolExecutionError, result_payload


class GoogleWorkspaceProvider:
    def __init__(self, settings: BrokerSettings) -> None:
        self.settings = settings
        self.script = settings.hermes_home / "skills" / "productivity" / "google-workspace" / "scripts" / "google_api.py"
        self.bridge = settings.hermes_home / "skills" / "productivity" / "google-workspace" / "scripts" / "gws_bridge.py"
        self.token_path = settings.hermes_home / "google_token.json"

    def _google_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Google Workspace uses broker-local OAuth token files. Do not route these
        # calls through OneCLI's HTTPS proxy, which can break gws TLS validation.
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
            "NODE_EXTRA_CA_CERTS",
        ):
            env.pop(key, None)
        env["HERMES_HOME"] = str(self.settings.hermes_home)
        return env

    def _require_token(self) -> None:
        if not self.token_path.exists():
            raise ToolExecutionError(
                f"Google Workspace auth is not configured on Hermes Spark. Expected token at {self.token_path}",
                category="auth",
            )

    def _run_script(self, *args: str) -> Any:
        self._require_token()
        result = subprocess.run(
            [str(Path("/home/rj/.hermes/hermes-agent/venv/bin/python")), str(self.script), *args],
            capture_output=True,
            text=True,
            env=self._google_env(),
        )
        if result.returncode != 0:
            raise ToolExecutionError(result.stderr.strip() or result.stdout.strip() or "Google Workspace command failed", category="provider")
        text = result.stdout.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    def _access_token(self) -> str:
        self._require_token()
        result = subprocess.run(
            [str(Path("/home/rj/.hermes/hermes-agent/venv/bin/python")), str(self.bridge), "gmail", "labels"],
            capture_output=True,
            text=True,
            env=self._google_env(),
        )
        token_data = json.loads(self.token_path.read_text(encoding="utf-8"))
        token = token_data.get("token")
        if not token:
            raise ToolExecutionError("Unable to read Google access token from token store", category="auth")
        return token

    def _request(self, method: str, url: str, *, json_body: Any | None = None, params: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        response = httpx.request(method, url, headers=headers, json=json_body, params=params, timeout=40.0, trust_env=False)
        if response.status_code >= 400:
            raise ToolExecutionError(f"Google API {response.status_code}: {response.text[:800]}", category="provider")
        return response.json() if response.text else {}

    def gmail_search(self, *, query: str, max_results: int = 10) -> dict[str, Any]:
        return result_payload(data=self._run_script("gmail", "search", query, "--max", str(max_results)))

    def gmail_get(self, *, message_id: str) -> dict[str, Any]:
        return result_payload(data=self._run_script("gmail", "get", message_id))

    def gmail_send(self, *, to: str, subject: str, body: str, html: bool = False) -> dict[str, Any]:
        args = ["gmail", "send", "--to", to, "--subject", subject, "--body", body]
        if html:
            args.append("--html")
        return result_payload(data=self._run_script(*args))

    def calendar_events_search(self, *, start: str | None = None, end: str | None = None, calendar: str = "primary") -> dict[str, Any]:
        args = ["calendar", "list"]
        if start:
            args += ["--start", start]
        if end:
            args += ["--end", end]
        args += ["--calendar", calendar]
        return result_payload(data=self._run_script(*args))

    def calendar_freebusy(self, *, time_min: str, time_max: str, calendars: list[str]) -> dict[str, Any]:
        payload = {"timeMin": time_min, "timeMax": time_max, "items": [{"id": cal} for cal in calendars]}
        return result_payload(data=self._request("POST", "https://www.googleapis.com/calendar/v3/freeBusy", json_body=payload))

    def calendar_event_create(self, *, summary: str, start: str, end: str, location: str | None = None, attendees: list[str] | None = None) -> dict[str, Any]:
        args = ["calendar", "create", "--summary", summary, "--start", start, "--end", end]
        if location:
            args += ["--location", location]
        if attendees:
            args += ["--attendees", ",".join(attendees)]
        return result_payload(data=self._run_script(*args))

    def drive_search(self, *, query: str, max_results: int = 10, raw_query: bool = False) -> dict[str, Any]:
        args = ["drive", "search", query, "--max", str(max_results)]
        if raw_query:
            args.append("--raw-query")
        return result_payload(data=self._run_script(*args))

    def drive_file_get(self, *, file_id: str) -> dict[str, Any]:
        params = {"fields": "id,name,mimeType,webViewLink,webContentLink,owners,modifiedTime,size"}
        return result_payload(data=self._request("GET", f"https://www.googleapis.com/drive/v3/files/{file_id}", params=params))

    def docs_get(self, *, doc_id: str) -> dict[str, Any]:
        return result_payload(data=self._run_script("docs", "get", doc_id))

    def docs_patch(self, *, doc_id: str, requests_payload: list[dict[str, Any]]) -> dict[str, Any]:
        return result_payload(
            data=self._request(
                "POST",
                f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate",
                json_body={"requests": requests_payload},
            )
        )

    def sheets_read(self, *, spreadsheet_id: str, cell_range: str) -> dict[str, Any]:
        return result_payload(data=self._run_script("sheets", "get", spreadsheet_id, cell_range))

    def sheets_update(self, *, spreadsheet_id: str, cell_range: str, values: list[list[Any]]) -> dict[str, Any]:
        return result_payload(
            data=self._run_script("sheets", "update", spreadsheet_id, cell_range, "--values", json.dumps(values))
        )
