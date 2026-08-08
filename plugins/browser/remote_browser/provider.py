"""Remote Browser cloud browser provider.

This provider follows the same lifecycle contract as the bundled cloud browser
plugins: create a remote session, return its CDP URL to Hermes, and stop the
session when Hermes is done.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import requests

from agent.browser_provider import BrowserProvider
from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://brapi.remote-browser.dev"
_DEFAULT_CREATE_PATH = "/dashboard/remote-browsers"
_DEFAULT_STATUS_PATH_TEMPLATE = "/dashboard/remote-browsers/{session_id}/status"
_DEFAULT_TERMINATE_PATH_TEMPLATE = "/dashboard/remote-browsers/{session_id}/terminate"
_DEFAULT_TIMEOUT_MINUTES = 5
_DEFAULT_POLL_TIMEOUT_SECONDS = 120
_DEFAULT_POLL_INTERVAL_SECONDS = 2
_DEFAULT_READY_GRACE_SECONDS = 10
_DEFAULT_RESOLUTION = "2560x1440"
_DEFAULT_REGION = "auto"
_DEFAULT_RECORDING_ENABLED = True
_DEFAULT_RECORDING_RETENTION_DAYS = 7
_READY_STATUSES = {"running"}
_FAILED_STATUSES = {"failed", "stopped", "timed_out"}


class RemoteBrowserProvider(BrowserProvider):
    """Remote Browser cloud browser backend."""

    @property
    def name(self) -> str:
        return "remote_browser"

    @property
    def display_name(self) -> str:
        return "Remote Browser"

    def is_available(self) -> bool:
        return self._get_config_or_none() is not None

    def _get_config_or_none(self) -> Optional[Dict[str, Any]]:
        api_key = get_secret("REMOTE_BROWSER_API_KEY")
        if not api_key:
            return None

        user_cfg = self._read_provider_config()
        return {
            "api_key": api_key,
            "base_url": self._read_str(
                user_cfg,
                "base_url",
                "REMOTE_BROWSER_BASE_URL",
                _DEFAULT_BASE_URL,
            ).rstrip("/"),
            "create_path": self._read_str(
                user_cfg,
                "create_path",
                "REMOTE_BROWSER_CREATE_PATH",
                _DEFAULT_CREATE_PATH,
            ),
            "terminate_path_template": self._read_str(
                user_cfg,
                "terminate_path_template",
                "REMOTE_BROWSER_TERMINATE_PATH_TEMPLATE",
                _DEFAULT_TERMINATE_PATH_TEMPLATE,
            ),
            "status_path_template": self._read_str(
                user_cfg,
                "status_path_template",
                "REMOTE_BROWSER_STATUS_PATH_TEMPLATE",
                _DEFAULT_STATUS_PATH_TEMPLATE,
            ),
            "timeout_minutes": self._read_int(
                user_cfg,
                "timeout_minutes",
                "REMOTE_BROWSER_TIMEOUT_MINUTES",
                _DEFAULT_TIMEOUT_MINUTES,
            ),
            "poll_timeout_seconds": self._read_int(
                user_cfg,
                "poll_timeout_seconds",
                "REMOTE_BROWSER_POLL_TIMEOUT_SECONDS",
                _DEFAULT_POLL_TIMEOUT_SECONDS,
            ),
            "poll_interval_seconds": self._read_int(
                user_cfg,
                "poll_interval_seconds",
                "REMOTE_BROWSER_POLL_INTERVAL_SECONDS",
                _DEFAULT_POLL_INTERVAL_SECONDS,
            ),
            "ready_grace_seconds": self._read_int(
                user_cfg,
                "ready_grace_seconds",
                "REMOTE_BROWSER_READY_GRACE_SECONDS",
                _DEFAULT_READY_GRACE_SECONDS,
            ),
            "resolution": self._read_str(
                user_cfg,
                "resolution",
                "REMOTE_BROWSER_RESOLUTION",
                _DEFAULT_RESOLUTION,
            ),
            "region": self._read_str(
                user_cfg,
                "region",
                "REMOTE_BROWSER_REGION",
                _DEFAULT_REGION,
            ),
            "recording_enabled": self._read_bool(
                user_cfg,
                "recording_enabled",
                "REMOTE_BROWSER_RECORDING",
                _DEFAULT_RECORDING_ENABLED,
            ),
            "recording_retention_days": self._read_int(
                user_cfg,
                "recording_retention_days",
                "REMOTE_BROWSER_RECORDING_RETENTION_DAYS",
                _DEFAULT_RECORDING_RETENTION_DAYS,
            ),
            "launch_arguments": self._read_launch_arguments(
                self._read_optional(user_cfg, "launch_arguments", "REMOTE_BROWSER_LAUNCH_ARGUMENTS")
            ),
            "profile_id": self._read_optional(user_cfg, "profile_id", "REMOTE_BROWSER_PROFILE_ID"),
            "profile_name": self._read_optional(user_cfg, "profile_name", "REMOTE_BROWSER_PROFILE_NAME"),
            "proxy_type": self._read_optional(user_cfg, "proxy_type", "REMOTE_BROWSER_PROXY_TYPE"),
            "proxy_url": self._read_optional(user_cfg, "proxy_url", "REMOTE_BROWSER_PROXY_URL"),
        }

    def _get_config(self) -> Dict[str, Any]:
        config = self._get_config_or_none()
        if config is None:
            raise ValueError("Remote Browser requires REMOTE_BROWSER_API_KEY.")
        return config

    def create_session(self, task_id: str) -> Dict[str, object]:
        config = self._get_config()
        payload = self._create_payload(config)
        create_url = self._url(config, config["create_path"])

        logger.info(
            "Creating Remote Browser session: task_id=%s url=%s payload=%s",
            task_id,
            self._safe_url_for_log(create_url),
            self._safe_payload_for_log(payload),
        )

        try:
            response = requests.post(
                create_url,
                headers=self._headers(config),
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Remote Browser API connection failed: {exc}") from exc

        if not response.ok:
            raise RuntimeError(
                f"Failed to create Remote Browser session: "
                f"{response.status_code} {response.text}"
            )

        session_data = response.json()
        session_id = self._read_session_id(session_data)
        session_data = self._wait_for_session_ready(session_id, config, session_data)
        cdp_url = self._read_cdp_url(session_data)
        cdp_url = self._apply_api_key_to_cdp_url(cdp_url, config["api_key"])
        session_name = f"remote_browser_{task_id}_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Remote Browser session ready: task_id=%s session_id=%s status=%s cdp_url=%s",
            task_id,
            session_id,
            session_data.get("status"),
            self._safe_url_for_log(cdp_url),
        )

        return {
            "session_name": session_name,
            "bb_session_id": session_id,
            "cdp_url": cdp_url,
            "features": {
                "remote_browser": True,
                "viewer_url": session_data.get("viewerUrl") or session_data.get("viewer_url"),
                "status": session_data.get("status"),
            },
        }

    def close_session(self, session_id: str) -> bool:
        config = self._get_config_or_none()
        if config is None:
            logger.warning(
                "Cannot close Remote Browser session %s: missing credentials",
                session_id,
            )
            return False

        terminate_path = str(config["terminate_path_template"]).format(
            session_id=session_id,
        )
        terminate_url = self._url(config, terminate_path)

        try:
            response = requests.post(
                terminate_url,
                headers=self._headers(config),
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.warning("Remote Browser terminate failed for %s: %s", session_id, exc)
            return False

        if response.status_code in {200, 201, 204}:
            logger.debug("Successfully closed Remote Browser session %s", session_id)
            return True

        logger.warning(
            "Remote Browser terminate failed for %s: HTTP %s - %s",
            session_id,
            response.status_code,
            response.text[:200],
        )
        return False

    def emergency_cleanup(self, session_id: str) -> None:
        self.close_session(session_id)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Remote Browser",
            "badge": "paid",
            "tag": "Remote Chromium sessions with CDP and live viewing",
            "env_vars": [
                {
                    "key": "REMOTE_BROWSER_API_KEY",
                    "prompt": "Remote Browser API key",
                    "url": "https://brapi.remote-browser.dev",
                },
            ],
            "post_setup": "agent_browser",
        }

    def _headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        api_key = str(config["api_key"])
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,
        }

    def _create_payload(self, config: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "resolution": config["resolution"],
            "idleTimeoutSeconds": max(60, int(config["timeout_minutes"]) * 60),
            "region": config["region"],
            "recordingEnabled": bool(config["recording_enabled"]),
            "recordingRetentionDays": int(config["recording_retention_days"]),
            "launchArguments": list(config["launch_arguments"]),
        }

        optional_config_to_payload = {
            "profile_id": "profileId",
            "profile_name": "profileName",
            "proxy_type": "proxyType",
            "proxy_url": "proxyUrl",
        }
        for config_name, payload_name in optional_config_to_payload.items():
            value = config.get(config_name)
            if value:
                payload[payload_name] = value

        return payload

    def _wait_for_session_ready(
        self,
        session_id: str,
        config: Dict[str, Any],
        initial_session_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + int(config["poll_timeout_seconds"])
        session_data = initial_session_data
        poll_count = 0

        while time.monotonic() < deadline:
            status = str(session_data.get("status") or "").lower()
            poll_count += 1

            logger.debug(
                "Remote Browser readiness poll: session_id=%s attempt=%s status=%s has_cdp_url=%s",
                session_id,
                poll_count,
                status or "unknown",
                self._has_cdp_url(session_data),
            )

            if status in _READY_STATUSES and self._has_cdp_url(session_data):
                ready_grace_seconds = int(config["ready_grace_seconds"])
                if ready_grace_seconds > 0:
                    logger.debug(
                        "Remote Browser running; waiting %s seconds for CDP readiness: session_id=%s",
                        ready_grace_seconds,
                        session_id,
                    )
                    time.sleep(ready_grace_seconds)
                return session_data

            if status in _FAILED_STATUSES:
                raise RuntimeError(
                    f"Remote Browser session {session_id} ended before CDP was ready: "
                    f"{session_data.get('providerError') or status}"
                )

            time.sleep(int(config["poll_interval_seconds"]))
            status_path = str(config["status_path_template"]).format(
                session_id=session_id,
            )
            status_url = self._url(config, status_path)

            try:
                response = requests.get(
                    status_url,
                    headers=self._headers(config),
                    timeout=15,
                )
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Remote Browser status check failed for {session_id}: {exc}"
                ) from exc

            if not response.ok:
                raise RuntimeError(
                    f"Remote Browser status check failed for {session_id}: "
                    f"{response.status_code} {response.text}"
                )

            session_data = response.json()

        raise RuntimeError(
            f"Timed out waiting for Remote Browser session {session_id} to become running."
        )

    def _read_provider_config(self) -> Dict[str, Any]:
        try:
            from hermes_cli.config import read_raw_config

            cfg = read_raw_config()
        except Exception:
            return {}

        browser_cfg = cfg.get("browser", {})
        if not isinstance(browser_cfg, dict):
            return {}
        provider_cfg = browser_cfg.get("remote_browser", {})
        return provider_cfg if isinstance(provider_cfg, dict) else {}

    def _read_session_id(self, session_data: Dict[str, Any]) -> str:
        for key in ("displayId", "id", "sessionId", "browserId"):
            value = session_data.get(key)
            if isinstance(value, str) and value:
                return value
        raise RuntimeError("Remote Browser response did not include a session id.")

    def _read_cdp_url(self, session_data: Dict[str, Any]) -> str:
        for key in ("cdpUrl", "cdp_url", "connectUrl", "connect_url"):
            value = session_data.get(key)
            if isinstance(value, str) and value:
                return value
        raise RuntimeError("Remote Browser response did not include a CDP URL.")

    def _has_cdp_url(self, session_data: Dict[str, Any]) -> bool:
        try:
            self._read_cdp_url(session_data)
            return True
        except RuntimeError:
            return False

    def _apply_api_key_to_cdp_url(self, cdp_url: str, api_key: str) -> str:
        parsed = urlsplit(cdp_url)
        path = parsed.path.rstrip("/")
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key not in {"apiKey", "token"}
        ]

        path = path.split("/api-key/", 1)[0]
        path = f"{path}/api-key/{quote(api_key, safe='')}"

        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                path,
                urlencode(query),
                parsed.fragment,
            )
        )

    def _url(self, config: Dict[str, Any], path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{config['base_url']}/{path.lstrip('/')}"

    def _safe_url_for_log(self, value: str) -> str:
        parsed = urlsplit(value)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))

        for key in ("apiKey", "token"):
            if key in query:
                query[key] = self._mask_secret(query[key])

        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )

    def _safe_payload_for_log(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        redacted = dict(payload)
        if "proxyUrl" in redacted:
            redacted["proxyUrl"] = self._mask_secret(str(redacted["proxyUrl"]))
        return redacted

    def _mask_secret(self, value: str) -> str:
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    def _read_optional(
        self,
        config: Dict[str, Any],
        key: str,
        env_name: str,
    ) -> Optional[Any]:
        if key in config:
            value = config[key]
        else:
            value = os.environ.get(env_name)
        if value in (None, ""):
            return None
        return value

    def _read_str(
        self,
        config: Dict[str, Any],
        key: str,
        env_name: str,
        fallback: str,
    ) -> str:
        value = self._read_optional(config, key, env_name)
        if value is None:
            return fallback
        return str(value)

    def _read_launch_arguments(self, value: Optional[Any]) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]

        text = str(value)
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

        return [item.strip() for item in text.split(",") if item.strip()]

    def _read_bool(
        self,
        config: Dict[str, Any],
        key: str,
        env_name: str,
        fallback: bool,
    ) -> bool:
        value = self._read_optional(config, key, env_name)
        if value is None:
            return fallback
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _read_int(
        self,
        config: Dict[str, Any],
        key: str,
        env_name: str,
        fallback: int,
    ) -> int:
        value = self._read_optional(config, key, env_name)
        if value is None:
            return fallback
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return fallback
        return parsed if parsed > 0 else fallback
