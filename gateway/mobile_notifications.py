"""Opt-in mobile device registration and Firebase Cloud Messaging delivery.

The API server is the control plane for Hermes Mobile. Device tokens are kept
inside the active profile's HERMES_HOME and are never logged. Delivery is
disabled unless ``mobile_notifications.enabled`` and ``project_id`` are set in
config.yaml and Application Default Credentials can mint an FCM access token.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from hermes_cli.active_sessions import _FileLock

logger = logging.getLogger(__name__)

FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
PAIRING_GRANT_TTL_SECONDS = 300
SAFE_ERROR_CATEGORIES = frozenset({
    "authentication", "configuration", "network", "permission", "rate_limit",
    "timeout", "tool", "unknown",
})


def _registry_path() -> Path:
    return Path(get_hermes_home()) / "runtime" / "mobile_devices.json"


def _lock_path() -> Path:
    return Path(get_hermes_home()) / "runtime" / "mobile_devices.lock"


def _pairing_path() -> Path:
    return Path(get_hermes_home()) / "runtime" / "mobile_pairing.json"


def _pairing_lock_path() -> Path:
    return Path(get_hermes_home()) / "runtime" / "mobile_pairing.lock"


def _clean_text(value: Any, *, maximum: int) -> str:
    return " ".join(str(value or "").split())[:maximum]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_count(value: Any) -> str:
    try:
        return str(max(0, min(int(value or 0), 1_000_000)))
    except (TypeError, ValueError):
        return "0"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


@dataclass(frozen=True)
class PairingGrant:
    grant_id: str
    secret: str
    code: str
    expires_at: float


@dataclass(frozen=True)
class PairedMobileDevice:
    device_id: str
    installation_id: str
    device_name: str
    scope: str
    created_at: float
    token: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "installation_id": self.installation_id,
            "device_name": self.device_name,
            "scope": self.scope,
            "created_at": self.created_at,
        }


class MobilePairingStore:
    """Profile-scoped pairing grants and hashed, revocable device credentials."""

    def create_grant(self, *, now: Optional[float] = None) -> PairingGrant:
        now = time.time() if now is None else now
        secret = secrets.token_urlsafe(32)
        alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        code = "".join(secrets.choice(alphabet) for _ in range(10))
        grant = PairingGrant(uuid.uuid4().hex, secret, code, now + PAIRING_GRANT_TTL_SECONDS)
        with _FileLock(_pairing_lock_path()):
            data = self._read()
            data["grants"] = [
                item for item in data["grants"] if _number(item.get("expires_at")) > now
            ]
            data["grants"].append({
                "grant_id": grant.grant_id,
                "secret_digest": _digest(secret),
                "code_digest": _digest(code),
                "expires_at": grant.expires_at,
            })
            self._write(data)
        return grant

    def exchange(
        self,
        credential: str,
        *,
        installation_id: str,
        device_name: str = "",
        now: Optional[float] = None,
    ) -> Optional[PairedMobileDevice]:
        now = time.time() if now is None else now
        clean_credential = _clean_text(credential, maximum=256)
        clean_installation = _clean_text(installation_id, maximum=128)
        if not clean_credential or not clean_installation:
            return None
        candidate = _digest(clean_credential)
        with _FileLock(_pairing_lock_path()):
            data = self._read()
            matched = None
            kept = []
            for item in data["grants"]:
                if _number(item.get("expires_at")) <= now:
                    continue
                is_match = hmac.compare_digest(candidate, str(item.get("secret_digest") or ""))
                is_match |= hmac.compare_digest(candidate, str(item.get("code_digest") or ""))
                if matched is None and is_match:
                    matched = item
                else:
                    kept.append(item)
            data["grants"] = kept
            if matched is None:
                self._write(data)
                return None

            token = "hmob_" + secrets.token_urlsafe(32)
            device = PairedMobileDevice(
                device_id=uuid.uuid4().hex,
                installation_id=clean_installation,
                device_name=_clean_text(device_name, maximum=120) or "Mobile device",
                scope="mobile.full",
                created_at=now,
                token=token,
            )
            data["devices"] = [
                item for item in data["devices"]
                if item.get("installation_id") != clean_installation
            ]
            data["devices"].append({
                **device.public_dict(),
                "token_digest": _digest(token),
            })
            self._write(data)
            return device

    def authenticate(self, token: str) -> bool:
        candidate = _digest(token) if token else ""
        if not candidate:
            return False
        with _FileLock(_pairing_lock_path()):
            return any(
                hmac.compare_digest(candidate, str(item.get("token_digest") or ""))
                for item in self._read()["devices"]
            )

    def list_devices(self) -> list[PairedMobileDevice]:
        with _FileLock(_pairing_lock_path()):
            rows = self._read()["devices"]
        return [
            PairedMobileDevice(
                device_id=str(item.get("device_id") or ""),
                installation_id=str(item.get("installation_id") or ""),
                device_name=str(item.get("device_name") or "Mobile device"),
                scope=str(item.get("scope") or "mobile.full"),
                created_at=_number(item.get("created_at")),
            )
            for item in rows
            if item.get("device_id") and item.get("installation_id")
        ]

    def revoke(self, device_id: str) -> bool:
        clean_id = _clean_text(device_id, maximum=128)
        with _FileLock(_pairing_lock_path()):
            data = self._read()
            kept = [item for item in data["devices"] if item.get("device_id") != clean_id]
            if len(kept) == len(data["devices"]):
                return False
            data["devices"] = kept
            self._write(data)
            return True

    @staticmethod
    def _read() -> dict[str, list[dict[str, Any]]]:
        try:
            raw = json.loads(_pairing_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            raw = {}
        grants = raw.get("grants") if isinstance(raw, dict) else None
        devices = raw.get("devices") if isinstance(raw, dict) else None
        return {
            "grants": [item for item in grants if isinstance(item, dict)] if isinstance(grants, list) else [],
            "devices": [item for item in devices if isinstance(item, dict)] if isinstance(devices, list) else [],
        }

    @staticmethod
    def _write(data: dict[str, Any]) -> None:
        path = _pairing_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
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


def mobile_extension_enabled() -> bool:
    return load_mobile_notification_config().get("enabled") is True


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
        "active_count": _bounded_count(event.get("active_count")),
    }
    latest_status = _clean_text(event.get("latest_status"), maximum=180)
    if latest_status:
        data["latest_status"] = latest_status
    if event.get("timestamp") is not None:
        data["timestamp"] = _clean_text(event.get("timestamp"), maximum=64)
    for field in ("tasks_completed", "tasks_total", "active_subagents"):
        if event.get(field) is not None:
            data[field] = _bounded_count(event.get(field))
    error_category = _clean_text(event.get("error_category"), maximum=32).lower()
    if error_category in SAFE_ERROR_CATEGORIES:
        data["error_category"] = error_category
    return {
        "message": {
            "token": device.token,
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

    def send(self, event: dict[str, Any], *, installation_id: Optional[str] = None) -> int:
        config = load_mobile_notification_config()
        project_id = _clean_text(config.get("project_id"), maximum=256)
        if config.get("enabled") is not True or not project_id:
            return 0
        devices = [
            item for item in self.store.list()
            if item.notifications_enabled and (installation_id is None or item.installation_id == installation_id)
        ]
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
