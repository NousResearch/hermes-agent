"""Thin HTTP client for the agent -> NAS ``agent-cron`` endpoints (Chronos): arm one-shot / cancel /
list, authenticated with the existing Nous Portal token.
Wire contract: ``docs/chronos-managed-cron-contract.md``."""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List

logger = logging.getLogger("cron.chronos")

_PROVISION_PATH = "/api/agent-cron/provision"
_CANCEL_PATH = "/api/agent-cron/cancel"
_LIST_PATH = "/api/agent-cron/list"


class NasCronClientError(RuntimeError):
    """Raised when a NAS agent-cron call fails (non-2xx or transport error)."""


class NasCronClient:
    """Minimal client for the agent->NAS provision/cancel/list endpoints."""

    def __init__(self, portal_url: str, *, timeout_seconds: float = 15.0) -> None:
        self.portal_url = portal_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> Dict[str, str]:
        """Bearer auth with the agent's existing Nous Portal access token (refresh-aware)."""
        from hermes_cli.auth import resolve_nous_access_token
        return {"Authorization": f"Bearer {resolve_nous_access_token()}",
                "Content-Type": "application/json"}

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        """Issue one request; raise NasCronClientError on transport error or non-2xx."""
        import requests  # lazy: agent already depends on requests
        try:
            resp = requests.request(method, f"{self.portal_url}{path}", headers=self._headers(),
                                    timeout=self.timeout_seconds, **kwargs)
        except Exception as e:
            raise NasCronClientError(f"{method} {path} failed: {e}") from e
        if resp.status_code // 100 != 2:
            raise NasCronClientError(f"{method} {path} returned {resp.status_code}: {resp.text[:200]}")
        with contextlib.suppress(Exception):
            return resp.json() if resp.content else {}
        return {}

    def provision(self, *, job_id: str, fire_at: str, agent_callback_url: str,
                  dedup_key: str) -> Dict[str, Any]:
        """Arm a one-shot for ``job_id`` at ``fire_at`` (ISO 8601); ``dedup_key`` makes re-arming
        idempotent NAS-side. Returns the NAS response (e.g. ``{schedule_id}``)."""
        return self._request("POST", _PROVISION_PATH, json={
            "job_id": job_id, "fire_at": fire_at, "agent_callback_url": agent_callback_url,
            "dedup_key": dedup_key})

    def cancel(self, *, job_id: str) -> Dict[str, Any]:
        """Cancel any armed one-shot for ``job_id``."""
        return self._request("POST", _CANCEL_PATH, json={"job_id": job_id})

    def list_armed(self) -> List[Dict[str, Any]]:
        """List armed one-shots (``{job_id, fire_at, schedule_id}``); best-effort, [] on odd shape."""
        data = self._request("GET", _LIST_PATH, params={})
        items = data.get("armed") if isinstance(data, dict) else None
        return items if isinstance(items, list) else []


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
