"""External-login credential-source adoption policy."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
EXTERNAL_LOGINS_NOT_ADOPTED_NOTICE = (
    "External CLI logins (Codex CLI, Claude Code) are not adopted: auth.adopt_external_logins is false. "
    "Hermes uses only its own logins; run `hermes auth add <provider>` to add one."
)
_notice_logged = False


def adopt_external_logins_enabled(*, environment) -> bool:
    """Read the existing adoption policy and retain its once-per-process notice."""
    global _notice_logged
    environment.require_current_scope()
    try:
        auth_cfg = (environment.read_config() or {}).get("auth")
    except Exception:
        return True
    enabled = not isinstance(auth_cfg, dict) or bool(
        auth_cfg.get("adopt_external_logins", True)
    )
    if not enabled and not _notice_logged:
        _notice_logged = True
        logger.info(EXTERNAL_LOGINS_NOT_ADOPTED_NOTICE)
    return enabled


