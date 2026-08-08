"""User-authorization methods for ``TelegramAdapter``.

Extracted from ``plugins/platforms/telegram/adapter.py`` as part of the
god-file decomposition campaign. This mixin holds the Telegram authorization
cluster: whether a callback query or an unauthorized DM may proceed, plus the
per-profile ``_scoped_gate_env`` reader shared with the adapter module.

Behavior-neutral: every method is lifted verbatim from ``TelegramAdapter``.
``self.*`` calls resolve unchanged via the MRO, and
``TelegramAuthorizationMixin`` precedes ``BasePlatformAdapter`` in the bases so
resolution order is what it was when these methods sat on the class.

``logger`` is bound by explicit name rather than ``__name__``, so log records
emitted from these methods keep the adapter's logger name.
"""

import logging
import os
from typing import Any, Optional

from gateway.authz_mixin import _coerce_allow_set
from gateway.config import Platform

try:
    from telegram import Message
except ImportError:  # pragma: no cover - mirrors the adapter's import guard
    Message = Any

# Bind the adapter's logger by name so log records lifted with these methods
# are emitted under exactly the name they were before.
logger = logging.getLogger("plugins.platforms.telegram.adapter")


def _scoped_gate_env(name: str, default: str = "") -> str:
    """Read a TELEGRAM_*/GATEWAY_* authorization gate env var per-profile.

    Under gateway.multiplex_profiles the process env is first-writer-wins
    (the YAML→env bridge in ``_apply_yaml_config``), so a raw ``os.getenv``
    can return ANOTHER profile's allowlist (issue #72348, Telegram mirror).
    Reads the active profile's secret scope when installed; falls back to
    ``os.getenv`` outside multiplex — identical single-profile behavior.
    """
    try:
        from gateway.authz_mixin import _platform_gate_env

        return _platform_gate_env(name, default)
    except Exception:
        return (os.getenv(name) or default).strip()

class TelegramAuthorizationMixin:
    """Authorization cluster lifted verbatim from ``TelegramAdapter``."""

    def _should_pass_unauthorized_dm_for_pairing(self, source) -> bool:
        """Return True when an unauthorized DM must still reach gateway pairing.

        Early auth (#40863) rejects before event construction. That is correct
        when unauthorized DMs are ignored, but it must not short-circuit the
        gateway pairing handshake when ``unauthorized_dm_behavior`` resolves
        to ``pair`` — including the case where an allowlist is set and the
        operator explicitly opted back into pairing via a platform override
        (resolution rule 1 in ``_get_unauthorized_dm_behavior``).
        """
        if source.chat_type != "dm":
            return False

        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        behavior_fn = getattr(runner, "_get_unauthorized_dm_behavior", None)
        if callable(behavior_fn):
            try:
                return (
                    behavior_fn(
                        Platform.TELEGRAM,
                        profile=getattr(source, "profile", None),
                    )
                    == "pair"
                )
            except Exception:
                logger.debug(
                    "[Telegram] Failed to resolve unauthorized DM behavior; "
                    "falling back to adapter-local override",
                    exc_info=True,
                )

        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        return str(extra.get("unauthorized_dm_behavior", "")).strip().lower() == "pair"
