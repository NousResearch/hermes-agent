"""Opt-in mobile device registration and Firebase Cloud Messaging delivery.

The API server is the control plane for Hermes Mobile. Device tokens are kept
inside the active profile's HERMES_HOME and are never logged. Delivery is
disabled unless ``mobile_notifications.enabled`` and ``project_id`` are set in
config.yaml and Application Default Credentials can mint an FCM access token.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from hermes_cli.active_sessions import _FileLock

logger = logging.getLogger(__name__)

FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
def _registry_path() -> Path:
    return Path(get_hermes_home()) / "runtime" / "mobile_devices.json"


def _lock_path() -> Path:
    return Path(get_hermes_home()) / "runtime" / "mobile_devices.lock"


def _clean_text(value: Any, *, maximum: int) -> str:
    return " ".join(str(value or "").split())[:maximum]


@dataclass(frozen=True)
class MobileDevice:
    installation_id: str
    token: str
    host_profile_id: str
    app_version: str = ""
    notifications_enabled: bool = True
    bubbles_enabled: bool = True
    overlay_enabled: bool = False
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Optional["MobileDevice"]:
        installation_id = _clean_text(value.get("installation_id"), maximum=128)
        token = _clean_text(value.get("fid") or value.get("token"), maximum=4096)
        host_profile_id = _clean_text(value.get("host_profile_id"), maximum=128)
        if not installation_id or not token or not host_profile_id:
            return None
        capabilities = value.get("capabilities") or {}
        if not isinstance(capabilities, dict):
            capabilities = {}
        try:
            updated_at = float(value.get("updated_at") or 0)
        except (TypeError, ValueError):
            updated_at = 0.0
        return cls(
            installation_id=installation_id,
            token=token,
            host_profile_id=host_profile_id,
            app_version=_clean_text(value.get("app_version"), maximum=64),
            notifications_enabled=capabilities.get("notifications") is not False,
            bubbles_enabled=capabilities.get("bubbles") is not False,
            overlay_enabled=capabilities.get("overlay") is True,
            updated_at=updated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "installation_id": self.installation_id,
            "token": self.token,
            "host_profile_id": self.host_profile_id,
            "app_version": self.app_version,
            "capabilities": {
                "notifications": self.notifications_enabled,
                "bubbles": self.bubbles_enabled,
                "overlay": self.overlay_enabled,
            },
            "updated_at": self.updated_at,
        }


class MobileDeviceStore:
    """Small profile-scoped, cross-process-safe JSON registry."""

    def list(self) -> list[MobileDevice]:
        with _FileLock(_lock_path()):
            return self._read()

    def upsert(self, body: dict[str, Any]) -> MobileDevice:
        value = dict(body)
        value["updated_at"] = time.time()
        device = MobileDevice.from_dict(value)
        if device is None:
            raise ValueError("installation_id, fid, and host_profile_id are required")
        with _FileLock(_lock_path()):
            devices = [item for item in self._read() if item.installation_id != device.installation_id]
            devices.append(device)
            self._write(devices)
        return device

    def delete(self, installation_id: str) -> bool:
        clean_id = _clean_text(installation_id, maximum=128)
        with _FileLock(_lock_path()):
            devices = self._read()
            kept = [item for item in devices if item.installation_id != clean_id]
            if len(kept) == len(devices):
                return False
            self._write(kept)
            return True

    def delete_tokens(self, tokens: Iterable[str]) -> None:
        rejected = set(tokens)
        if not rejected:
            return
        with _FileLock(_lock_path()):
            self._write([item for item in self._read() if item.token not in rejected])

    @staticmethod
    def _read() -> list[MobileDevice]:
        try:
            raw = json.loads(_registry_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except Exception:
            logger.warning("Ignoring corrupt mobile device registry at %s", _registry_path())
            return []
        values = raw.get("devices", []) if isinstance(raw, dict) else []
        devices = [MobileDevice.from_dict(item) for item in values if isinstance(item, dict)]
        return [item for item in devices if item is not None]

    @staticmethod
    def _write(devices: Iterable[MobileDevice]) -> None:
        path = _registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps({"devices": [item.to_dict() for item in devices]}, sort_keys=True),
            encoding="utf-8",
        )
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)


def load_mobile_notification_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:
        return {}
    value = config.get("mobile_notifications") if isinstance(config, dict) else None
    return dict(value) if isinstance(value, dict) else {}


def build_fcm_message(device: MobileDevice, event: dict[str, Any]) -> dict[str, Any]:
    """Build a privacy-safe FCM data message for one registered device."""
    kind = _clean_text(event.get("event"), maximum=64) or "session.updated"
    session_id = _clean_text(event.get("session_id"), maximum=256)
    title = _clean_text(event.get("title"), maximum=120) or "Hermes session"
    state = _clean_text(event.get("state"), maximum=32) or kind.rsplit(".", 1)[-1]
    data = {
        "event": kind,
        "host_profile_id": device.host_profile_id,
        "session_id": session_id,
        "run_id": _clean_text(event.get("run_id"), maximum=128),
        "title": title,
        "state": state,
        "active_count": str(max(0, int(event.get("active_count") or 0))),
    }
    return {
        "message": {
            "fid": device.token,
            "data": data,
            "android": {
                "priority": "high",
                "ttl": "300s",
            },
        }
    }


class FCMNotifier:
    """Synchronous FCM HTTP v1 sender; callers should run it off the event loop."""

    def __init__(self, store: Optional[MobileDeviceStore] = None):
        self.store = store or MobileDeviceStore()

    def send(self, event: dict[str, Any]) -> int:
        config = load_mobile_notification_config()
        project_id = _clean_text(config.get("project_id"), maximum=256)
        if config.get("enabled") is not True or not project_id:
            return 0
        devices = [item for item in self.store.list() if item.notifications_enabled]
        if not devices:
            return 0
        try:
            import google.auth
            from google.auth.transport.requests import Request
            import requests

            credentials, _ = google.auth.default(scopes=[FCM_SCOPE])
            credentials.refresh(Request())
        except Exception as exc:
            logger.warning("Mobile notifications are configured but FCM credentials are unavailable: %s", exc)
            return 0

        endpoint = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
        sent = 0
        rejected: list[str] = []
        for device in devices:
            try:
                response = requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {credentials.token}"},
                    json=build_fcm_message(device, event),
                    timeout=10,
                )
                if 200 <= response.status_code < 300:
                    sent += 1
                elif response.status_code in {404, 410} or "UNREGISTERED" in response.text:
                    rejected.append(device.token)
                else:
                    logger.warning("FCM delivery failed with status %s", response.status_code)
            except Exception as exc:
                logger.warning("FCM delivery failed: %s", exc)
        self.store.delete_tokens(rejected)
        return sent
