"""Nous Portal upstream adapter.

Reads the user's Nous OAuth state from ``~/.hermes/auth.json``, refreshes
the access token and mints a fresh agent key when needed, and exposes the
upstream base URL plus minted bearer for the proxy server to forward to.

The minted ``agent_key`` (not the OAuth ``access_token``) is what
``inference-api.nousresearch.com`` accepts as a bearer. The refresh helper
already handles both — see :func:`hermes_cli.auth.refresh_nous_oauth_from_state`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, FrozenSet, Optional

from hermes_cli.auth import (
    DEFAULT_NOUS_INFERENCE_URL,
    _load_auth_store,
    _save_auth_store,
    _write_shared_nous_state,
    refresh_nous_oauth_from_state,
)
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

# Endpoints inference-api.nousresearch.com actually serves. Anything else
# the proxy will reject with 404 — keeps stray clients from leaking weird
# requests to the upstream.
_ALLOWED_PATHS: FrozenSet[str] = frozenset(
    {
        "/chat/completions",
        "/completions",
        "/embeddings",
        "/models",
    }
)


class NousPortalAdapter(UpstreamAdapter):
    """Proxy upstream for the Nous Portal inference API."""

    def __init__(self) -> None:
        # Lock guards _load → refresh → _save against parallel proxy requests
        # racing to refresh expired tokens. Refresh itself is HTTP, so we
        # hold the lock across the network call (brief; OAuth refresh is fast).
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "nous"

    @property
    def display_name(self) -> str:
        return "Nous Portal"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        state = self._read_state()
        if state is None:
            return False
        # We need either a usable agent_key OR (refresh_token + access_token)
        # to recover. The refresh helper will mint/refresh as needed.
        return bool(
            state.get("agent_key")
            or (state.get("refresh_token") and state.get("access_token"))
        )

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            state = self._read_state()
            if state is None:
                raise RuntimeError(
                    "Not logged into Nous Portal. Run `hermes login nous` first."
                )

            try:
                refreshed = refresh_nous_oauth_from_state(state)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to refresh Nous Portal credentials: {exc}"
                ) from exc

            self._save_state(refreshed)

            agent_key = refreshed.get("agent_key")
            if not agent_key:
                raise RuntimeError(
                    "Nous Portal refresh did not return a usable agent_key. "
                    "Try `hermes login nous` to re-authenticate."
                )

            base_url = refreshed.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL
            base_url = base_url.rstrip("/")

            return UpstreamCredential(
                bearer=agent_key,
                base_url=base_url,
                expires_at=refreshed.get("agent_key_expires_at"),
            )

    # ------------------------------------------------------------------
    # Internal helpers — auth.json access. Kept local rather than added
    # to hermes_cli.auth to avoid expanding that module's public surface.
    # ------------------------------------------------------------------

    def _read_state(self) -> Optional[Dict[str, Any]]:
        try:
            store = _load_auth_store()
        except Exception as exc:
            logger.warning("proxy: failed to load auth store: %s", exc)
            return None
        providers = store.get("providers") or {}
        state = providers.get("nous")
        if not isinstance(state, dict):
            return None
        return dict(state)  # copy so the refresh helper can mutate freely

    def _save_state(self, state: Dict[str, Any]) -> None:
        try:
            store = _load_auth_store()
            providers = store.setdefault("providers", {})
            providers["nous"] = state
            _save_auth_store(store)
            _write_shared_nous_state(state)
        except Exception as exc:
            # Best effort — we still return the fresh credential. The next
            # request just won't see cached state, which means another refresh.
            logger.warning("proxy: failed to persist refreshed Nous state: %s", exc)


__all__ = ["NousPortalAdapter"]
