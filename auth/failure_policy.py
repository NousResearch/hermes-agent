"""Runtime authentication failure classification."""

from auth.errors import AuthError
from auth.constants import CODEX_RATE_LIMITED_CODE


def is_rate_limited_auth_error(error: Exception) -> bool:
    """True when an :class:`AuthError` is upstream rate-limiting / quota: transient, and
    re-authenticating cannot fix it, so callers should say "retry later", not ``hermes auth``."""
    return (isinstance(error, AuthError) and not error.relogin_required
            and error.code == CODEX_RATE_LIMITED_CODE)


def primary_failure_wording(error: Exception) -> tuple[str, str]:
    """``(log_phrase, user_phrase)`` for a primary-provider failure that triggers the fallback
    chain. A 429/quota AuthError leaves the credentials valid; labelling it "auth failed" sends
    operators hunting for an expired token (#117482), so it reads as quota at every surface."""
    if is_rate_limited_auth_error(error):
        return "rate-limited (429)", "Primary provider quota exhausted"
    return "auth failed", "Primary auth failed"
