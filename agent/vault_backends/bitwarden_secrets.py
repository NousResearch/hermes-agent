"""Bitwarden Secrets Manager login backend using the official Python SDK.

Website logins live in a dedicated Secrets Manager project.  The project is
never registered as a bulk ``SecretSource``: values are fetched only for an
explicit credential handle and are never injected into the agent environment.
"""

from __future__ import annotations

import base64
import json
import secrets
import subprocess
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from uuid import UUID

from agent.secret_scope import get_secret_str
from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultError, VaultItemMeta, normalize_origin, normalize_otp_secret, scrub_secret_from_text

_KEY_PREFIX = "hermes-login-v1:"
_NOTE = "Hermes browser login (opaque payload v1)"


def _b64_encode(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> str:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("utf-8")


class BitwardenSecretsLoginBackend(LoginBackend):
    name = "bitwarden_secrets"
    display_name = "Bitwarden Secrets Manager"
    prefix = "bws:"

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        self._organization_id_cache: Optional[UUID] = None

    def capabilities(self):
        return super().capabilities() | {"create_login", "update_login", "remove"}

    def _access_token(self) -> str:
        env_name = str(self.cfg.get("access_token_env") or "BWS_ACCESS_TOKEN").strip()
        token = get_secret_str(env_name).strip()
        if not token:
            token = self._keychain_token()
        if not token:
            raise VaultError(
                f"Bitwarden Secrets Manager access token is missing ({env_name}); "
                "configure the scoped machine-account token first"
            )
        return token

    def _keychain_token(self) -> str:
        service = str(self.cfg.get("keychain_service") or "").strip()
        if not service:
            return ""
        argv = ["/usr/bin/security", "find-generic-password", "-s", service]
        account = str(self.cfg.get("keychain_account") or "").strip()
        if account:
            argv.extend(["-a", account])
        argv.append("-w")
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=5, stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return (proc.stdout or "").strip() if proc.returncode == 0 else ""

    def _ids(self, client=None) -> tuple[UUID, UUID]:
        try:
            project_id = UUID(str(self.cfg.get("project_id") or ""))
        except ValueError as exc:
            raise VaultError("vault.bitwarden_secrets requires a valid project_id UUID") from exc
        configured = str(self.cfg.get("organization_id") or "").strip()
        if configured:
            try:
                return UUID(configured), project_id
            except ValueError as exc:
                raise VaultError("vault.bitwarden_secrets organization_id must be a valid UUID") from exc
        if self._organization_id_cache is None:
            client = client or self._client()
            project = self._unwrap(client.projects().get(str(project_id)), "get project")
            try:
                self._organization_id_cache = UUID(str(project.organization_id))
            except (AttributeError, ValueError) as exc:
                raise VaultError("Bitwarden project did not return a valid organization_id") from exc
        return self._organization_id_cache, project_id

    def _make_client(self, token: str):
        try:
            from bitwarden_sdk import BitwardenClient, ClientSettings
        except ImportError:
            from tools.lazy_deps import ensure
            ensure("vault.bitwarden_secrets", prompt=False)
            from bitwarden_sdk import BitwardenClient, ClientSettings

        api_url = str(self.cfg.get("api_url") or "").strip() or None
        identity_url = str(self.cfg.get("identity_url") or "").strip() or None
        settings = ClientSettings(api_url=api_url, identity_url=identity_url, user_agent="Hermes Credential Broker")
        client = BitwardenClient(settings)
        client.auth().login_access_token(token)
        return client

    def _client(self):
        token = self._access_token()
        try:
            return self._make_client(token)
        except Exception as exc:
            message = scrub_secret_from_text(str(exc), {"access_token": token})[:240]
            raise VaultError(f"Bitwarden Secrets Manager authentication failed: {message}") from exc

    @staticmethod
    def _unwrap(response, operation: str):
        if not getattr(response, "success", False) or getattr(response, "data", None) is None:
            raise VaultError(str(getattr(response, "error_message", "") or f"{operation} failed")[:240])
        return response.data

    @staticmethod
    def _origin_from_key(key: str) -> Optional[str]:
        if not key.startswith(_KEY_PREFIX):
            return None
        encoded, sep, _nonce = key[len(_KEY_PREFIX):].partition(":")
        if not sep:
            return None
        try:
            return normalize_origin(_b64_decode(encoded))
        except Exception:
            return None

    @staticmethod
    def _key(origin: str) -> str:
        return f"{_KEY_PREFIX}{_b64_encode(normalize_origin(origin))}:{secrets.token_hex(6)}"

    @staticmethod
    def _payload(*, label: str, origin: str, identifier_type: str, identifier: str,
                 password: str, otp_secret: Optional[str]) -> str:
        payload = {
            "version": 1,
            "kind": "login",
            "label": (label or "").strip(),
            "origin": normalize_origin(origin),
            "identifier_type": identifier_type,
            "identifier": (identifier or "").strip(),
            "password": password,
        }
        normalized_otp = normalize_otp_secret(str(otp_secret or ""))
        if normalized_otp:
            payload["otp_secret"] = normalized_otp
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode_payload(value: str) -> Dict[str, str]:
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise VaultError("Bitwarden credential payload is invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("kind") != "login":
            raise VaultError("Bitwarden credential payload has an unsupported format")
        required = ("label", "origin", "identifier_type", "identifier", "password")
        out = {key: str(payload.get(key) or "") for key in (*required, "otp_secret")}
        if not out["identifier"] or not out["password"]:
            raise VaultError("Bitwarden credential payload is incomplete")
        out["origin"] = normalize_origin(out["origin"])
        return out

    def _secret(self, handle: str):
        if not self.owns(handle):
            raise VaultError(f"invalid Bitwarden Secrets Manager handle {handle!r}")
        client = self._client()
        data = self._unwrap(client.secrets().get(handle[len(self.prefix):]), "get secret")
        _, project_id = self._ids(client)
        if getattr(data, "project_id", None) != project_id:
            raise VaultError("credential handle is outside the configured Bitwarden project")
        return data

    def list_items(self) -> List[VaultItemMeta]:
        client = self._client()
        org_id, project_id = self._ids(client)
        data = self._unwrap(client.secrets().list(str(org_id)), "list secrets")
        out: List[VaultItemMeta] = []
        for item in getattr(data, "data", []) or []:
            if project_id not in (getattr(item, "project_ids", []) or []):
                continue
            origin = self._origin_from_key(str(getattr(item, "key", "") or ""))
            if not origin:
                continue
            host = urlsplit(origin).hostname or origin
            label = host
            has_otp = False
            try:
                payload = self._decode_payload(
                    self._unwrap(client.secrets().get(str(item.id)), "get secret").value
                )
                label = scrub_secret_from_text(
                    payload["label"] or host,
                    {"identifier": payload["identifier"], "password": payload["password"]},
                )
                has_otp = bool(payload.get("otp_secret"))
            except VaultError:
                pass
            out.append(VaultItemMeta(
                id=f"{self.prefix}{item.id}", kind="login", label=label, origin=origin,
                created_at="", identifier_type=None, identifier=None, has_otp=has_otp,
            ))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        try:
            item = self._secret(handle)
        except VaultError as exc:
            if "not found" in str(exc).lower():
                return None
            raise
        payload = self._decode_payload(item.value)
        label = scrub_secret_from_text(
            payload["label"], {"identifier": payload["identifier"], "password": payload["password"]}
        )
        return VaultItemMeta(
            id=handle, kind="login", label=label, origin=payload["origin"],
            created_at=str(getattr(item, "creation_date", "") or ""),
            identifier_type=None, identifier=None, has_otp=bool(payload.get("otp_secret")),
        )

    def resolve_login(self, handle: str) -> Dict[str, str]:
        return self._decode_payload(self._secret(handle).value)

    def resolve_password(self, handle: str) -> str:
        return self.resolve_login(handle)["password"]

    def resolve_otp(self, handle: str) -> Optional[str]:
        from agent.vault_store import totp_now
        seed = self.resolve_login(handle).get("otp_secret", "")
        return totp_now(seed) if seed else None

    def find_login(self, origin: str, identifier: str) -> Optional[VaultItemMeta]:
        normalized = normalize_origin(origin)
        for meta in self.list_items():
            if meta.origin != normalized:
                continue
            payload = self.resolve_login(meta.id)
            if payload.get("identifier") == identifier:
                return self.get_meta(meta.id)
        return None

    def create_login(self, *, label: str, origin: str, identifier_type: str,
                     identifier: str, password: str, otp_secret: Optional[str] = None) -> VaultItemMeta:
        client = self._client()
        org_id, project_id = self._ids(client)
        value = self._payload(label=label, origin=origin, identifier_type=identifier_type,
                              identifier=identifier, password=password, otp_secret=otp_secret)
        try:
            data = self._unwrap(
                client.secrets().create(org_id, self._key(origin), value, _NOTE, [project_id]),
                "create secret",
            )
        except Exception as exc:
            message = scrub_secret_from_text(str(exc), {"identifier": identifier, "password": password,
                                                        "payload": value})[:240]
            raise VaultError(f"Bitwarden credential save failed: {message}") from exc
        return self.get_meta(f"{self.prefix}{data.id}")

    def update_login(self, handle: str, *, label: str, origin: str, identifier_type: str,
                     identifier: str, password: str, otp_secret: Optional[str] = None) -> VaultItemMeta:
        item = self._secret(handle)
        client = self._client()
        org_id, project_id = self._ids(client)
        if self._origin_from_key(str(item.key)) != normalize_origin(origin):
            raise VaultError("login updates must keep the original site origin")
        value = self._payload(label=label, origin=origin, identifier_type=identifier_type,
                              identifier=identifier, password=password, otp_secret=otp_secret)
        try:
            self._unwrap(
                client.secrets().update(str(org_id), str(item.id), str(item.key), value, _NOTE, [project_id]),
                "update secret",
            )
        except Exception as exc:
            message = scrub_secret_from_text(str(exc), {"identifier": identifier, "password": password,
                                                        "payload": value})[:240]
            raise VaultError(f"Bitwarden credential update failed: {message}") from exc
        return self.get_meta(handle)

    def remove_item(self, handle: str) -> bool:
        if self.get_meta(handle) is None:
            return False
        data = self._unwrap(self._client().secrets().delete([handle[len(self.prefix):]]), "delete secret")
        failures = [item for item in (getattr(data, "data", []) or []) if getattr(item, "error", None)]
        if failures:
            raise VaultError(str(failures[0].error)[:240])
        return True
