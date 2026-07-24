"""xAI Grok OAuth upstream adapter."""

from __future__ import annotations

import logging
import threading
from typing import FrozenSet, Optional

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "xai-oauth"

# xAI's public API is OpenAI-compatible for the endpoints Hermes commonly
# uses. The Responses endpoint is included because Hermes' native xAI runtime
# uses codex_responses mode.
_ALLOWED_PATHS: FrozenSet[str] = frozenset(
    {
        "/responses",
        "/chat/completions",
        "/completions",
        "/embeddings",
        "/models",
    }
)


class XAIGrokAdapter(UpstreamAdapter):
    """Proxy upstream for xAI Grok via Hermes-managed OAuth credentials."""

    auth_hint = "hermes auth add xai-oauth --type oauth"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI Grok OAuth"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        # C1/F2/R7: shared mode probes the canonical store (and promotable
        # local grants on profile OR root), never a surviving legacy pool row
        # after profile disable. Canonical READ errors fail closed (False)
        # with no pool fallthrough.
        try:
            from hermes_cli import auth as auth_mod

            if auth_mod._xai_shared_auth_enabled():
                try:
                    if auth_mod._profile_xai_shared_disabled():
                        return False
                    shared = auth_mod._read_shared_xai_state(raise_on_unreadable=True)
                    if auth_mod._xai_shared_state_has_usable_tokens(shared):
                        return True
                    if auth_mod._shared_xai_state_is_quarantined(shared):
                        return False
                    # F1/R7/H2: sole live across profile AND root together
                    # (no profile-first short-circuit vs promoter ambiguity).
                    return bool(
                        auth_mod._xai_sole_live_promotable_across_profile_and_root_for_probe()
                    )
                except Exception:
                    return False
        except Exception:
            pass
        pool = self._load_pool()
        return bool(pool and pool.has_available())

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            from hermes_cli import auth as auth_mod

            # C1/A6/F2: shared mode is fail-closed — never select a legacy pool row.
            if auth_mod._xai_shared_auth_enabled():
                if auth_mod._profile_xai_shared_disabled():
                    raise RuntimeError(
                        "xAI shared OAuth is disabled for this profile. "
                        "Re-enable with `hermes auth xai enable-shared` or log in again."
                    )
                shared_cred = self._credential_from_shared(raise_on_error=True)
                if shared_cred is not None:
                    return shared_cred
                raise RuntimeError(
                    "No xAI OAuth credentials in the shared store. "
                    "Select xAI Grok OAuth in `hermes model` or run "
                    "`hermes auth xai migrate-shared`."
                )

            pool = self._load_pool()
            if pool is None or not pool.has_credentials():
                raise RuntimeError(
                    "No xAI OAuth credentials found. Run "
                    "`hermes auth add xai-oauth --type oauth` first."
                )

            entry = pool.select()
            if entry is None:
                raise RuntimeError(
                    "No available xAI OAuth credentials found. Run "
                    "`hermes auth reset xai-oauth` or re-authenticate with "
                    "`hermes auth add xai-oauth --type oauth`."
                )

            self._pool = pool
            return self._credential_from_entry(entry)

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        # C3: treat auth-shaped 401 AND 403 as refresh triggers (not only 401).
        if status_code not in {401, 403, 429}:
            return None

        with self._lock:
            from hermes_cli import auth as auth_mod

            shared_on = auth_mod._xai_shared_auth_enabled()
            if status_code in {401, 403}:
                # Prefer canonical shared force-refresh with the rejected bearer.
                shared_retry = self._credential_from_shared(
                    force_refresh=True,
                    rejected_access_token=failed_credential.bearer,
                    raise_on_error=False,
                )
                if (
                    shared_retry is not None
                    and shared_retry.bearer != failed_credential.bearer
                ):
                    logger.info(
                        "proxy: xAI upstream returned %s; retrying with "
                        "canonical shared credential",
                        status_code,
                    )
                    return shared_retry
                # F2: shared mode never falls through to a legacy pool row.
                if shared_on:
                    return None

            # R9: under shared mode a 429 is sole-grant / rate-limit policy —
            # do NOT run generic OAuth-pool rotation (would mark the wrong
            # reference after another process rotated). Let 429 flow back.
            if status_code == 429 and shared_on:
                logger.info(
                    "proxy: xAI upstream returned 429 under shared mode; "
                    "canonical sole-grant policy (no pool rotation)"
                )
                return None

            # Gate-off (or non-shared 429): pool rotation path.
            pool = self._pool or self._load_pool()
            if pool is None:
                return None

            if status_code == 429:
                # Mark the rate-limited key with its 1-hour cooldown and rotate
                # to the next available credential. Pass the failed bearer so a
                # freshly loaded pool cannot mark the wrong/newer entry.
                refreshed = pool.mark_exhausted_and_rotate(
                    status_code=status_code,
                    api_key_hint=failed_credential.bearer,
                )
            else:
                refreshed = pool.try_refresh_current()
                if refreshed is None:
                    refreshed = pool.mark_exhausted_and_rotate(
                        status_code=status_code,
                        api_key_hint=failed_credential.bearer,
                    )
            if refreshed is None:
                return None

            retry_cred = self._credential_from_entry(refreshed)
            if retry_cred.bearer == failed_credential.bearer:
                return None
            logger.info(
                "proxy: xAI upstream returned %s; retrying with rotated pool credential",
                status_code,
            )
            return retry_cred

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception as exc:
            logger.warning("proxy: failed to load xAI OAuth credential pool: %s", exc)
            return None

    def _credential_from_shared(
        self,
        *,
        force_refresh: bool = False,
        rejected_access_token: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> Optional[UpstreamCredential]:
        try:
            from hermes_cli import auth as auth_mod

            if not auth_mod._xai_shared_auth_enabled():
                return None
            if auth_mod._profile_xai_shared_disabled():
                if raise_on_error:
                    raise RuntimeError(
                        "xAI shared OAuth is disabled for this profile."
                    )
                return None
            creds = auth_mod.resolve_xai_oauth_runtime_credentials(
                force_refresh=force_refresh,
                refresh_if_expiring=not force_refresh,
                rejected_access_token=rejected_access_token,
            )
            bearer = str(creds.get("api_key") or "").strip()
            if not bearer:
                return None
            base_url = str(
                creds.get("base_url") or DEFAULT_XAI_OAUTH_BASE_URL
            ).strip().rstrip("/")
            return UpstreamCredential(
                bearer=bearer,
                base_url=base_url or DEFAULT_XAI_OAUTH_BASE_URL,
                expires_at=None,
            )
        except RuntimeError:
            if raise_on_error:
                raise
            return None
        except Exception as exc:
            if raise_on_error:
                raise RuntimeError(
                    f"Shared xAI OAuth credential resolve failed: {exc}"
                ) from exc
            logger.debug("proxy: shared xAI credential resolve failed: %s", exc)
            return None

    def _credential_from_entry(self, entry: PooledCredential) -> UpstreamCredential:
        bearer = (
            getattr(entry, "runtime_api_key", None)
            or getattr(entry, "access_token", "")
            or ""
        )
        bearer = str(bearer).strip()
        if not bearer:
            raise RuntimeError(
                "xAI OAuth credential pool entry did not contain an access token. "
                "Re-authenticate with `hermes auth add xai-oauth --type oauth`."
            )

        base_url = (
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None)
            or DEFAULT_XAI_OAUTH_BASE_URL
        )
        base_url = str(base_url or DEFAULT_XAI_OAUTH_BASE_URL).strip().rstrip("/")

        return UpstreamCredential(
            bearer=bearer,
            base_url=base_url or DEFAULT_XAI_OAUTH_BASE_URL,
            expires_at=getattr(entry, "expires_at", None),
        )


__all__ = ["XAIGrokAdapter"]
