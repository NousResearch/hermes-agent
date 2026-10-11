"""Copilot stale-JWT recovery for the auxiliary ladder (#135645).

GitHub can invalidate an exchanged Copilot JWT server-side while its stored ``expires_at`` is
still hours away (session revocation). Every chat call with it answers a bare ``403 forbidden``,
which ``_is_auth_error`` does not classify as refreshable — and even the 401 refresh path was
defeated, because popping only the in-process ``_jwt_cache`` left the disk fast-path in
``_exchange_copilot_token_locked`` free to reload the poisoned entry from
``~/.hermes/.copilot_jwt.json``. These helpers live in a sibling of ``auxiliary_client`` so that
file's line budget only ever goes down.
"""

from typing import Optional


def is_stale_copilot_jwt_error(exc: Exception, auth_refresh_provider: Optional[str]) -> bool:
    """Copilot-host 403 whose body is a bare ``forbidden`` (no billing markers): GitHub revoked
    the exchanged JWT before its stored ``expires_at``, so the cached JWT — not the raw token —
    is what died; a fresh exchange recovers (#135645). Provider-scoped on purpose: other hosts'
    403s stay permission denials, and billing-worded Copilot 403s are quota, not credentials."""
    if auth_refresh_provider != "copilot" or getattr(exc, "status_code", None) != 403:
        return False
    # The aux payment list (superset of the classifier's billing patterns) stays the single
    # source for "this 403 is quota wording", so the two lists cannot drift (#107166).
    from agent.auxiliary_client import _PAYMENT_KEYWORDS
    err_lower = str(exc).lower()
    return "forbidden" in err_lower and not any(kw in err_lower for kw in _PAYMENT_KEYWORDS)


def refresh_copilot_credentials() -> bool:
    from hermes_cli.copilot_auth import evict_cached_exchanged_token, exchange_copilot_token, resolve_copilot_token
    raw_token, _source = resolve_copilot_token()
    if not str(raw_token or "").strip():
        return False
    # Evict in-process AND on-disk: the disk fast-path in ``_exchange_copilot_token_locked`` would
    # otherwise reload the still-"fresh" (by stored expires_at) JWT this refresh is meant to drop,
    # defeating it even on a healthy raw token (#135645).
    evict_cached_exchanged_token(raw_token)
    exchange_copilot_token(raw_token)
    return True
