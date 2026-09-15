"""Nous Portal upstream adapter."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, FrozenSet, Optional

from hermes_cli.auth import (
    AuthError,
    DEFAULT_NOUS_INFERENCE_URL,
    _load_auth_store,
    _auth_store_lock,
    _is_terminal_nous_refresh_error,
    _nous_inference_env_override,
    _quarantine_nous_oauth_state,
    _quarantine_nous_pool_entries,
    _save_auth_store,
    _validate_nous_inference_url_from_network,
    _write_shared_nous_state,
    resolve_nous_runtime_credentials,
)
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

# Endpoints inference-api.nousresearch.com actually serves; anything else is a 404 so stray
# clients cannot leak odd requests upstream.
_ALLOWED_PATHS: FrozenSet[str] = frozenset({"/chat/completions", "/completions", "/embeddings", "/models"})


class NousPortalAdapter(UpstreamAdapter):
    """Proxy upstream for the Nous Portal inference API."""

    def __init__(self) -> None:
        # In-process serialization; cross-process refresh/persistence is resolve_nous_runtime_credentials().
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
        # Usable inference JWT, OR refresh_token + access_token to recover via the refresh helper.
        state = self._read_state() or {}
        return bool(state.get("agent_key") or (state.get("refresh_token") and state.get("access_token")))

    def get_credential(self) -> UpstreamCredential:
        return self._get_credential()

    def get_retry_credential(
        self, *, failed_credential: UpstreamCredential, status_code: int
    ) -> Optional[UpstreamCredential]:
        if status_code != 401:
            return None
        logger.info("proxy: Nous upstream rejected bearer; force-refreshing invoke JWT")
        return self._get_credential(force_refresh=True, stale_access_token=failed_credential.bearer)

    def _get_credential(
        self, *, force_refresh: bool = False, stale_access_token: Optional[str] = None
    ) -> UpstreamCredential:
        with self._lock:
            state = self._read_state()
            if state is None:
                raise RuntimeError("Not logged into Nous Portal. Run `hermes auth add nous` first.")
            try:
                refreshed = resolve_nous_runtime_credentials(
                    force_refresh=force_refresh, stale_access_token=stale_access_token or None
                )
            except Exception as exc:
                if isinstance(exc, AuthError) and _is_terminal_nous_refresh_error(exc):
                    _quarantine_nous_oauth_state(state, exc, reason="proxy_refresh_failure")
                    self._save_state(state, quarantine_error=exc, quarantine_reason="proxy_refresh_failure")
                raise RuntimeError(f"Failed to refresh Nous Portal credentials: {exc}") from exc
            runtime_key = refreshed.get("api_key")
            if not runtime_key:
                raise RuntimeError(
                    "Nous Portal refresh did not return a usable inference JWT. "
                    "Try `hermes auth add nous` to re-authenticate."
                )
            # The returned base_url already honors the NOUS_INFERENCE_BASE_URL override (documented
            # dev/staging hatch); validating it against the prod allowlist would reject a legit
            # staging URL. So: env override wins, else network-validate the returned URL, else the
            # production default (defense-in-depth against a future source-layer bypass).
            base_url = (
                _nous_inference_env_override()
                or _validate_nous_inference_url_from_network(refreshed.get("base_url"))
                or DEFAULT_NOUS_INFERENCE_URL
            ).rstrip("/")
            return UpstreamCredential(bearer=runtime_key, base_url=base_url, expires_at=refreshed.get("expires_at"))

    # auth.json access — kept local so hermes_cli.auth's public surface does not grow.

    def _read_state(self) -> Optional[Dict[str, Any]]:
        try:
            with _auth_store_lock():
                store = _load_auth_store()
        except Exception as exc:
            logger.warning("proxy: failed to load auth store: %s", exc)
            return None
        state = (store.get("providers") or {}).get("nous")
        return dict(state) if isinstance(state, dict) else None

    def _save_state(
        self,
        state: Dict[str, Any],
        *,
        quarantine_error: Optional[AuthError] = None,
        quarantine_reason: Optional[str] = None,
    ) -> None:
        try:
            with _auth_store_lock():
                store = _load_auth_store()
                if quarantine_error is not None and quarantine_reason:
                    _quarantine_nous_pool_entries(store, quarantine_error, reason=quarantine_reason)
                providers = store.setdefault("providers", {})
                providers["nous"] = state
                _save_auth_store(store)
            _write_shared_nous_state(state)
        except Exception as exc:
            logger.warning("proxy: failed to persist Nous quarantine state: %s", exc)


__all__ = ["NousPortalAdapter"]
