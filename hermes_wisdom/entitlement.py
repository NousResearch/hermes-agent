"""Refresh-free, profile-scoped Wisdom presentation and local-work eligibility.

Portal's OAuth issuer projects the team feature flag into ``wisdom_scopes``.
These locally decoded claims are advisory, NOT server authorization: Gateway
still validates the bearer on every request. Never refresh credentials while
rendering menus or inspecting tool availability, and never cache across profiles
or beyond logout/token replacement. Expired or malformed claims fail closed.
"""

from __future__ import annotations

import math
import time


def current_entitlement() -> dict:
    """Return only fresh local entitlement metadata, never the bearer itself."""
    try:
        from hermes_cli.auth_constants import _decode_jwt_claims
        from hermes_cli.auth_nous import get_nous_auth_status_local

        status = get_nous_auth_status_local()
        if not status.get("logged_in") or status.get("relogin_required"):
            return {}
        token = status.get("access_token")
        if not isinstance(token, str) or not token:
            return {}
        # Advisory decoding uses the auth layer's stdlib-only parser. Importing
        # PyJWT here loads native cryptography during every CLI parser build,
        # which can lock updater DLLs on Windows.
        claims = _decode_jwt_claims(token)
        expires = claims.get("exp")
        now = time.time()
        if (
            isinstance(expires, bool)
            or not isinstance(expires, (int, float))
            or not math.isfinite(expires)
            or expires <= now
        ):
            return {}
        not_before = claims.get("nbf", 0)
        if (
            isinstance(not_before, bool)
            or not isinstance(not_before, (int, float))
            or not math.isfinite(not_before)
            or not_before > now
        ):
            return {}
        org_id = claims.get("org_id")
        scopes = claims.get("wisdom_scopes")
        if not isinstance(org_id, str) or not org_id.strip() or not isinstance(scopes, list):
            return {}
        if not all(isinstance(scope, str) for scope in scopes):
            return {}
        return {"org_id": org_id, "scopes": tuple(scopes), "expires_at": expires}
    except Exception:
        return {}


def is_entitled(org_id: str | None = None, *, scope: str = "wisdom:read") -> bool:
    """Whether this profile's fresh Nous token grants the requested capability."""
    entitlement = current_entitlement()
    return bool(
        scope in entitlement.get("scopes", ())
        and (org_id is None or org_id == entitlement.get("org_id"))
    )


def require_entitlement(org_id: str | None = None, *, scope: str = "wisdom:read") -> None:
    from .package import PackagePolicyError

    if not is_entitled(org_id, scope=scope):
        raise PackagePolicyError("Collective Wisdom is unavailable for the current account and team")


def local_work_allowed(store) -> bool:
    """Opt-in and matching current entitlement are both required for local work."""
    try:
        from .service import _config

        config = _config()
        org_id = store.active_org_id()
        return bool(
            config.get("enabled") is True
            and config.get("disclosure_acknowledged_at")
            and store.existing_installation_identity()
            and org_id
            and is_entitled(org_id)
        )
    except Exception:
        return False
