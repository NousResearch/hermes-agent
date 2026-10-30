"""Profile-aware session information used in gateway reset notices."""

from __future__ import annotations

import logging

from agent.i18n import t


logger = logging.getLogger("gateway.run")


def reset_notice_session_info(runner, source) -> str:
    """Render reset details inside the source's runtime-profile scope."""
    from gateway.session_identity import identity_of

    with runner._profile_scope_for_source(source):
        identity = identity_of(source)
        if identity is not None:
            profile = identity.runtime_profile
        elif getattr(runner.config, "multiplex_profiles", False):
            profile = source.profile or runner._active_profile_name()
        else:
            profile = runner._active_profile_name()
        header = t("gateway.profile.header", profile=profile)
        try:
            details = runner._format_session_info()
        except Exception:
            logger.debug("reset-notice session info resolution failed for %s", profile, exc_info=True)
            return header
        return f"{header}\n{details}"
