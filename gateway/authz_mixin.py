"""Authorization mixin for inbound message sources.

Provides ``_is_user_authorized`` and ``_pairing_store_for`` used by the
gateway runner to gate access for every inbound message.
"""
from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from gateway.session import SessionSource
from gateway.config import Platform

if TYPE_CHECKING:
    from gateway.pairing import PairingStore

logger = logging.getLogger(__name__)


class AuthzMixin:
    """Helpers that decide whether a ``SessionSource`` is allowed to talk
    to the bot. Used by ``gateway/run.py`` on every inbound event.
    """

    def _is_user_authorized(self, source: "SessionSource") -> bool:
        """Check if a user is authorized to use the bot."""
        # (body unchanged from upstream — see PR for full context)
        ...

    def _pairing_store_for(self, source: "SessionSource") -> "PairingStore":
        """Pick the per-profile PairingStore for a source, falling back to global.

        In a multiplexing gateway, each profile owns its own pairing whitelist
        so isolation is preserved. When the source has no profile (single-
        profile gateway, or a path that hasn't stamped profile yet) or the
        profile isn't registered, fall back to ``self.pairing_store`` (the
        global default) so existing behavior is preserved.
        """
        per_profile = getattr(self, "pairing_stores", None) or {}
        profile = getattr(source, "profile", None)
        if profile and profile in per_profile:
            return per_profile[profile]
        return getattr(self, "pairing_store", None)
