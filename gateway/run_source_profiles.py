"""Runtime profile resolution for inbound and restored session sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from gateway.config import GatewayConfig
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewaySourceProfilesMixin:
    config: GatewayConfig
    _profile_name_for_source: Callable[[SessionSource], Optional[str]]

    def _resolve_profile_home_for_source(self, source: SessionSource) -> Path:
        """Resolve which profile's HERMES_HOME serves this source: the pinned identity's runtime
        home, else ``source.profile``, then ``_profile_name_for_source`` (sources bypassing
        ``build_source``), then the primary profile under multiplexing or the active profile."""
        from gateway.profile_routing import ProfileRouteRejected
        from gateway.session_identity import identity_of
        from hermes_cli.profiles import get_active_profile_name, get_profile_dir, profile_exists
        from hermes_constants import get_hermes_home
        identity = identity_of(source)
        if identity is not None:
            return identity.runtime_home
        explicit_profile = None  # explicitly requested (source or routing) vs. default fallback
        try:
            name = (source.profile or "").strip() or self._profile_name_for_source(source)
            explicit_profile = name or None
            if not name:
                name = ((getattr(self, "_primary_profile_name", None) or "default")
                        if self.config.multiplex_profiles else get_active_profile_name() or "default")
            profile_dir = get_profile_dir(name)
            if explicit_profile and not profile_exists(name):
                logger.warning(
                    "Profile %r does not exist for source %s/%s (guild_id=%s), "
                    "falling back to global HERMES_HOME",
                    explicit_profile, source.platform.value, source.chat_id,
                    getattr(source, "guild_id", None))
                return get_hermes_home()
            return profile_dir
        except ProfileRouteRejected:
            raise
        except Exception:
            logger.warning(
                "Failed to resolve profile directory for source %s/%s (guild_id=%s), "
                "falling back to global HERMES_HOME: %s",
                source.platform.value, source.chat_id, getattr(source, "guild_id", None),
                explicit_profile or "(no profile)", exc_info=True)
            return get_hermes_home()
