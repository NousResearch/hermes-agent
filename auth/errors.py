"""Structured provider authentication failures."""
from __future__ import annotations
from typing import Optional

class AuthError(RuntimeError):
    """Structured auth error with UX mapping hints."""

    def __init__(
        self, message: str, *, provider: str = "", code: Optional[str] = None, relogin_required: bool = False,
        retry_after: Optional[float] = None, retryable: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.relogin_required = relogin_required
        # Optional wait hint in seconds (a server ``Retry-After`` or a client cooldown) and whether a
        # later attempt can succeed at all. None = the raiser did not say; callers treat None as
        # "retryable, no hint" for transport-shaped errors and as terminal for auth refusals.
        self.retry_after = retry_after
        self.retryable = retryable

_OAUTH_GRANT_DEAD_CODES = frozenset({"invalid_grant", "invalid_token", "refresh_token_reused"})
