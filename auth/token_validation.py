"""Local token identity and validity checks shared by credentials and OAuth."""
from __future__ import annotations
import base64
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
NOUS_INFERENCE_INVOKE_SCOPE = "inference:invoke"
NOUS_INVOKE_JWT_MIN_TTL_SECONDS = 120

def _decode_jwt_claims(token: Any) -> Dict[str, Any]:
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        claims = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}

def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_iso_timestamp(value: Any) -> Optional[float]:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _is_expiring(expires_at_iso: Any, skew_seconds: int) -> bool:
    expires_epoch = _parse_iso_timestamp(expires_at_iso)
    return expires_epoch is None or expires_epoch <= (time.time() + skew_seconds)

def _scope_values(raw_scope: Any) -> set[str]:
    # OAuth token responses return a space-separated string; collections are kept for JWT ``scp``
    # claims and older stored fixtures.
    scopes: set[str] = set()
    if isinstance(raw_scope, str):
        scopes.update(part for part in raw_scope.replace(",", " ").split() if part.strip())
    elif isinstance(raw_scope, (list, tuple, set, frozenset)):
        scopes.update(*(_scope_values(item) for item in raw_scope if isinstance(item, str)))
    return scopes


def _nous_invoke_jwt_status(
    token: Any, *, scope: Any = None, expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS) -> Optional[str]:
    """Return None when the token can be used for inference, else a reason."""
    claims = _decode_jwt_claims(token)
    if not claims:
        return "access_token_not_jwt"
    scopes = (_scope_values(scope) | _scope_values(claims.get("scope"))
              | _scope_values(claims.get("scp")))
    if NOUS_INFERENCE_INVOKE_SCOPE not in scopes:
        return "missing_inference_invoke_scope"
    exp = claims.get("exp")
    skew = max(0, int(min_ttl_seconds))
    if isinstance(exp, (int, float)):
        return "invoke_jwt_expiring" if float(exp) <= (time.time() + skew) else None
    return "invoke_jwt_expiry_unknown_or_expiring" if _is_expiring(expires_at, skew) else None


def _nous_invoke_jwt_is_usable(
    token: Any, *, scope: Any = None, expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS) -> bool:
    return _nous_invoke_jwt_status(
        token, scope=scope, expires_at=expires_at, min_ttl_seconds=min_ttl_seconds) is None
