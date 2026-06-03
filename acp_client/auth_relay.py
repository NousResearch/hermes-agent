"""Auth boundary for the ACP client (credential-free by construction).

Reduced mirror of ``acp_adapter/auth.py``.  The deliberate Phase-1 posture
(design §2.2, §2.9, R3): Hermes **never** forwards credentials to the external
agent.  The external CLI (Claude Code, Codex, Gemini …) manages its own auth
out-of-band.  This module exists to make that boundary explicit and to refuse
any inbound ``authenticate`` method the client does not understand.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The client forwards no auth methods at all in Phase 1.  An empty set means
# every method id is unknown → refused.
_SUPPORTED_AUTH_METHODS: frozenset[str] = frozenset()


class AuthForwardingRefused(RuntimeError):
    """Raised when an external agent asks the client to authenticate."""


def assert_no_credential_forwarding(method_id: str | None) -> None:
    """Refuse any auth method — the external CLI owns its own credentials.

    Phase 1 forwards no credentials, so every ``method_id`` is unsupported.
    Higher phases that genuinely need one specific method must add it to
    ``_SUPPORTED_AUTH_METHODS`` behind an explicit Filip-approval gate.
    """
    if method_id in _SUPPORTED_AUTH_METHODS:
        return
    logger.warning(
        "Refusing ACP authenticate(method_id=%r): the client never forwards "
        "credentials; the external agent must manage its own auth.",
        method_id,
    )
    raise AuthForwardingRefused(
        f"ACP client refuses to forward credentials for method {method_id!r}; "
        "the external agent manages its own auth."
    )
