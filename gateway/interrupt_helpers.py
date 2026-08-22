"""Interrupt-marker helpers extracted from the ``gateway.run`` god-file.

``_stamp_hygiene_compression_provenance`` records a best-effort activity
provenance stamp for hygiene compression transitions (timeout / cooldown).
It was extracted as part of the god-file campaign (#54962) and is
re-exported from ``gateway.run`` via a module-attribute import so
``gateway.run._stamp_hygiene_compression_provenance`` keeps resolving.

``_is_fresh_gateway_interruption`` intentionally STAYS in ``gateway.run``:
it depends on ``_coerce_gateway_timestamp``, which is claimed by open PR
#77433 (display helpers extraction) and may not be moved here.  Once that
PR lands, the freshness check (and its exclusive constant
``_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT``) can follow into this module.
"""

import logging
from typing import Any

logger = logging.getLogger("gateway.run")


def _stamp_hygiene_compression_provenance(
    agent: Any,
    desc: str,
    provenance: "ActivityProvenance",
    debug_label: str,
) -> None:
    """Best-effort activity provenance stamp for hygiene compression transitions."""
    try:
        agent._touch_activity(desc, provenance=provenance)
    except Exception:
        logger.debug(debug_label, exc_info=True)
