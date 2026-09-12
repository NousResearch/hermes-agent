"""Thin HTTP client for the Gateway's ``/v1/sync/wisdom/`` routes.

Auth is the Nous bearer from the shared sync identity; every response the plugin acts on is
integrity-checked against the hashes the server publishes, so a compromised transport cannot
hand us a different package than the one the user consented to.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any
from urllib.parse import quote

import requests

from plugins.wisdom.package import PackageError, parse_manifest, sha256_address, verify_files
from tools.skills_sync_client import resolve_identity, resolve_sync_base_url
from tools.skills_sync_client_wire import SyncClient


class WisdomError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status, self.code = status, code


class WisdomAuthError(WisdomError):
    """No entitled Nous session: the caller should send the user to ``hermes login``."""


_STATUS_MESSAGES = {
    401: "Collective Wisdom needs a Nous login (run `hermes login`)",
    403: "Your Nous account or team is not entitled to this Collective Wisdom action",
    404: "Collective Wisdom item not found",
    409: "Collective Wisdom state changed under you; refresh and retry",
    422: "Collective Wisdom rejected the content",
}


def entitlement() -> dict[str, Any]:
    """``{org_id, scopes}`` decoded (unverified) from a fresh local Nous token, else ``{}``.

    Advisory only: it decides whether to *try*, the Gateway authorizes every request. Never
    refreshes credentials, so tool availability probes stay side-effect free.
    """
    import math
    import time
    from hermes_cli.auth_constants import _decode_jwt_claims
    from hermes_cli.auth_nous import get_nous_auth_status_local
    try:
        status = get_nous_auth_status_local()
    except Exception:
        return {}
    if not status.get("logged_in") or status.get("relogin_required"):
        return {}
    claims = _decode_jwt_claims(status.get("access_token"))
    exp, nbf = claims.get("exp"), claims.get("nbf", 0)
    now = time.time()
    if any(isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) for b in (exp, nbf)):
        return {}
    if float(exp) <= now or float(nbf) > now:
        return {}
    org, scopes = claims.get("org_id"), claims.get("wisdom_scopes")
    if not (isinstance(org, str) and org.strip() and isinstance(scopes, list)
            and all(isinstance(s, str) for s in scopes)):
        return {}
    return {"org_id": org, "scopes": tuple(scopes)}


def entitled(scope: str = "wisdom:read") -> bool:
    return scope in entitlement().get("scopes", ())


class WisdomClient:
    def __init__(self, *, timeout: float = 30.0):
        try:
            identity = resolve_identity()
        except Exception as exc:
            raise WisdomAuthError(f"Collective Wisdom needs a Nous login: {exc}") from exc
        base = resolve_sync_base_url()
        if not base:
            raise WisdomError("Collective Wisdom Gateway is not configured (sync.base_url)")
        self.base = base
        self.timeout = timeout
        self.org_id = str(identity["claims"].get("org_id") or identity["claims"].get("orgId") or "") or None
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {identity['api_key']}", "Accept": "application/json"})
        self.sync = SyncClient(base, identity["api_key"], timeout=timeout)

    def _request(self, method: str, path: str, *, json_body: dict | None = None, params: dict | None = None) -> Any:
        try:
            r = self.session.request(method, f"{self.base}/v1/sync/wisdom/{path}", json=json_body,
                                     params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise WisdomError(f"Collective Wisdom Gateway is unavailable: {exc}") from exc
        body = None
        if r.content:
            try:
                body = r.json()
            except ValueError as exc:
                raise WisdomError("Collective Wisdom Gateway returned a malformed response") from exc
        if 200 <= r.status_code < 300:
            return body
        code = body.get("error") if isinstance(body, dict) else None
        message = _STATUS_MESSAGES.get(r.status_code, f"Collective Wisdom Gateway failed ({r.status_code})")
        if code:
            message = f"{message}: {code}"
        cls = WisdomAuthError if r.status_code in (401, 403) else WisdomError
        raise cls(message, status=r.status_code, code=code)

    # --- discovery ---------------------------------------------------------------------------
    def list_skills(self, *, cursor: str | None = None) -> dict:
        return self._request("GET", "skills", params={"cursor": cursor} if cursor else None)

    def skill(self, skill_id: str) -> dict:
        return self._request("GET", f"skills/{quote(skill_id, safe='')}")

    def version(self, skill_id: str, version: int) -> dict:
        return self._request("GET", f"skills/{quote(skill_id, safe='')}/versions/{int(version)}")

    def content(self, skill_id: str, version: int, *, installation_id: str,
                takedown_generation: int) -> tuple[str, list[tuple[str, str, bytes]]]:
        """Installable bytes for an exact version, verified blob-by-blob and as a whole."""
        body = self._request("GET", f"skills/{quote(skill_id, safe='')}/versions/{int(version)}/content",
                             params={"installation_id": installation_id, "takedown_generation": takedown_generation})
        files = []
        for item in body.get("files") or []:
            try:
                raw = base64.b64decode(item["content_base64"], validate=True)
            except (KeyError, TypeError, ValueError) as exc:
                raise PackageError("version content contains malformed base64") from exc
            if sha256_address(raw) != item.get("hash"):
                raise PackageError(f"blob failed integrity validation: {item.get('path', '?')}")
            files.append((str(item["path"]), str(item.get("mode", "file")), raw))
        if verify_files(files) != body.get("content_hash"):
            raise PackageError("version content hash failed integrity validation")
        try:
            parse_manifest(next(b for p, _, b in files if p == "skill.manifest.json"))
        except (StopIteration, UnicodeDecodeError, ValueError) as exc:
            raise PackageError("version package manifest is invalid") from exc
        return str(body["content_hash"]), files

    # --- installations -----------------------------------------------------------------------
    def register_identity(self, installation_id: str) -> None:
        self._request("POST", "installation-identities", json_body={"installation_id": installation_id})

    def record_install(self, *, skill_id: str, installation_id: str, version: int, takedown_generation: int) -> dict:
        return self._request("POST", "installations", json_body={
            "skill_id": skill_id, "installation_id": installation_id, "version": int(version),
            "takedown_generation": int(takedown_generation)})

    def installations(self, installation_id: str) -> list[dict]:
        return self._request("GET", f"installations/{quote(installation_id, safe='')}").get("installations") or []

    def deactivate(self, installation_id: str, skill_id: str) -> None:
        self._request("DELETE", f"installations/{quote(installation_id, safe='')}/skills/{quote(skill_id, safe='')}")

    # --- sharing -----------------------------------------------------------------------------
    def submit_draft(self, prepared, *, slug: str) -> dict:
        self.sync.put_objects(prepared.objects.objects)
        return self._request("POST", "drafts", json_body={
            "slug": slug, "draft_commit": prepared.commit, "content_hash": prepared.content_hash,
            "author_description": prepared.description})["draft"]

    def draft(self, draft_id: str) -> dict:
        return self._request("GET", f"drafts/{quote(draft_id, safe='')}")["draft"]

    def list_drafts(self) -> list[dict]:
        return self._request("GET", "drafts").get("drafts") or []

    def approve_and_publish(self, draft_id: str, *, content_hash: str, description_hash: str, manifest_hash: str) -> dict:
        did = quote(draft_id, safe="")
        self._request("POST", f"drafts/{did}/approve", json_body={
            "content_hash": content_hash, "author_description_hash": description_hash,
            "package_manifest_hash": manifest_hash})
        return self._request("POST", f"drafts/{did}/publish", json_body={"content_hash": content_hash, "base_commit": None})

    def decline(self, draft_id: str) -> None:
        self._request("POST", f"drafts/{quote(draft_id, safe='')}/decline")


def new_installation_id() -> str:
    return uuid.uuid4().hex
