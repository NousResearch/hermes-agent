"""Host-gateway cleanup after a confirmed stop: the rendezvous record and the memoized probe."""
from __future__ import annotations

import logging

logger = logging.getLogger("hermes_cli.gateway")


def _discard_dead_host_record() -> bool:
    """Drop the host rendezvous record once its owner is provably gone (after a confirmed stop)."""
    try:
        from gateway.host_rendezvous import ROLE_GATEWAY, discard_dead_record

        return discard_dead_record(ROLE_GATEWAY)
    except Exception:
        logger.debug("Host record retraction failed", exc_info=True)
        return False


def _forget_host_gateway_probe() -> None:
    """Drop the memoized host-gateway probe after a stop, before the restart starts a new one.

    The probe is cached for ``HOST_GATEWAY_CACHE_TTL_S`` and a stop finishes well inside it, so the
    in-process ``run_gateway`` read "PID <stopped> serves this profile", printed "nothing to start"
    and exited 0 with no gateway running.
    """
    try:
        from gateway.host_attach import invalidate_host_gateway_cache

        invalidate_host_gateway_cache()
    except Exception:
        logger.debug("Host gateway probe invalidation failed", exc_info=True)
