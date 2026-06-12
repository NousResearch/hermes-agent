"""Fil d'activité Jean-Billie : signaux début/fin de job cron (fire-and-forget).

POST ``http://{JB_DRAFT_ADDR}/v1/activity`` avec
``{phase: "started"|"finished", status: "ok"|"error", department?, skill_id?, job_id?, label?}``.
``label`` = nom lisible du job (``jobs.json``). Les champs d'attribution absents sont OMIS du
JSON (pas de ``null``) — même convention que le stamp des DraftRequest.

La route daemon n'existe PAS ENCORE (vague 2), d'où le gate ``JB_ACTIVITY_EVENTS=1``
(défaut OFF). Timeout court, toute erreur avalée (log debug au plus) : un signal d'activité ne
doit JAMAIS bloquer ni faire échouer un job.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Timeout volontairement court : le daemon est sur le loopback, et un signal d'activité ne vaut
# pas la peine de retenir un job plus de 2 s.
_TIMEOUT = 2.0


def enabled() -> bool:
    """Vrai uniquement quand l'opérateur a activé le fil d'activité (``JB_ACTIVITY_EVENTS=1``)."""
    return os.getenv("JB_ACTIVITY_EVENTS", "").strip() == "1"


def emit(phase: str, status: str, ctx: Optional[Dict[str, Any]]) -> None:
    """Émet un évènement d'activité (best-effort). Silencieux si le gate est fermé ou en échec."""
    if not enabled():
        return
    try:
        from . import config, http_client

        event: Dict[str, Any] = {"phase": phase, "status": status}
        for key in ("department", "skill_id", "job_id", "label"):
            value = (ctx or {}).get(key)
            if value:
                event[key] = value
        http_client.post_json(config.activity_url(), event, timeout=_TIMEOUT)
    except Exception as exc:
        logger.debug("jb_outbound: signal d'activité non délivré (%s/%s) : %s", phase, status, exc)
