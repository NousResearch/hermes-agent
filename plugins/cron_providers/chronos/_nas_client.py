"""Thin HTTP client for the agent → NAS ``agent-cron`` endpoints (Chronos).

The Chronos provider speaks ONLY to NAS — it names no scheduler vendor and
holds no scheduler credentials. NAS owns the external scheduler (an internal
implementation detail) and that scheduler's account; the agent just asks NAS to
"arm a one-shot at time T" / "cancel" / "list", authenticated with the agent's
existing Nous Portal access token (the same token it already uses to call the
portal — no new secret).

Wire contract: ``docs/chronos-managed-cron-contract.md``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cron.chronos")

# Endpoint paths under the portal base URL.
_PROVISION_PATH = "/api/agent-cron/provision"
_CANCEL_PATH = "/api/agent-cron/cancel"
_LIST_PATH = "/api/agent-cron/list"


class NasCronClientError(RuntimeError):
    """Raised when a NAS agent-cron call fails (non-2xx or transport error)."""


class NasCronClient:
    """Minimal client for the agent→NAS provision/cancel/list endpoints.

    Uses the agent's refresh-aware Nous access token for auth. No scheduler
    vendor, no scheduler creds — NAS hides all of that behind these three calls.
    """

    def __init__(self, portal_url: str, *, timeout_seconds: float = 15.0) -> None:
        self.portal_url = portal_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    # -- auth -------------------------------------------------------------

    def _access_token(self) -> str:
        """The agent's existing Nous Portal access token (refresh-aware)."""
        from hermes_cli.auth import resolve_nous_access_token
        return resolve_nous_access_token()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/json",
        }

    # -- HTTP -------------------------------------------------------------

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        import requests  # lazy: agent already depends on requests

        url = f"{self.portal_url}{path}"
        try:
            resp = requests.post(
                url, json=body, headers=self._headers(), timeout=self.timeout_seconds
            )
        except Exception as e:
            raise NasCronClientError(f"POST {path} failed: {e}") from e
        if resp.status_code // 100 != 2:
            raise NasCronClientError(
                f"POST {path} returned {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json() if resp.content else {}
        except Exception:
            return {}

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        url = f"{self.portal_url}{path}"
        try:
            resp = requests.get(
                url, params=params, headers=self._headers(), timeout=self.timeout_seconds
            )
        except Exception as e:
            raise NasCronClientError(f"GET {path} failed: {e}") from e
        if resp.status_code // 100 != 2:
            raise NasCronClientError(
                f"GET {path} returned {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json() if resp.content else {}
        except Exception:
            return {}

    # -- endpoints --------------------------------------------------------

    def provision(self, *, job_id: str, fire_at: str, agent_callback_url: str,
                  dedup_key: str) -> Dict[str, Any]:
        """Ask NAS to arm a one-shot for ``job_id`` at ``fire_at`` (ISO 8601).

        ``dedup_key`` (``{job_id}:{fire_at}``) makes re-arming the same fire
        idempotent NAS-side. Returns the NAS response (e.g. ``{schedule_id}``).
        """
        return self._post(_PROVISION_PATH, {
            "job_id": job_id,
            "fire_at": fire_at,
            "agent_callback_url": agent_callback_url,
            "dedup_key": dedup_key,
        })

    def cancel(self, *, job_id: str) -> Dict[str, Any]:
        """Ask NAS to cancel any armed one-shot for ``job_id``."""
        return self._post(_CANCEL_PATH, {"job_id": job_id})

    def list_armed(self) -> List[Dict[str, Any]]:
        """List the one-shots NAS currently has armed for this agent.

        Returns a list of ``{job_id, fire_at, schedule_id}``. Best-effort: used
        by reconcile to find orphaned arms on a cold process; on error the
        caller falls back to idempotent re-arm of all desired jobs.
        """
        data = self._get(_LIST_PATH, {})
        items = data.get("armed") if isinstance(data, dict) else None
        return items if isinstance(items, list) else []
