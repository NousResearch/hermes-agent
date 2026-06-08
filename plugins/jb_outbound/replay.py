"""Chemin RETOUR : rejouer l'envoi réel après approbation, puis remonter le résultat.

À réception d'une `DecisionItem` (poussée par le daemon sur le listener), on retrouve l'envoi en
attente via `payload.jb_id`, on rejoue l'appel d'outil EXACT via `registry.dispatch` (qui ne
repasse pas par le middleware → pas de ré-interception), puis on POSTe un `ResultRequest` au daemon
avec `id` = l'id de la proposition (round-trip control-plane). Idempotent sur l'état du store.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def handle_decision(decision: Dict[str, Any]) -> None:
    from . import config, http_client, store

    proposal_id = str(decision.get("id") or "")
    payload = decision.get("payload") or {}
    jb_id = str(payload.get("jb_id") or "") if isinstance(payload, dict) else ""

    rec = store.load(jb_id)
    if rec is None:
        logger.info("jb_outbound: décision sans envoi en attente connu (jb_id=%r) — ignorée", jb_id)
        return
    if rec.get("status") == "executed":
        logger.info("jb_outbound: décision déjà exécutée (jb_id=%s) — idempotent", jb_id)
        return

    status, error = "executed", ""
    try:
        from tools.registry import registry  # import paresseux : seulement dans le runtime Hermes
        registry.dispatch(rec["tool_name"], rec.get("args") or {})
    except Exception as exc:  # l'envoi réel a échoué après approbation
        status, error = "failed", str(exc)
        logger.warning("jb_outbound: exécution de l'envoi échouée (jb_id=%s) : %s", jb_id, exc)

    store.mark(jb_id, status, error)

    try:
        http_client.post_json(
            config.result_url(),
            {"id": proposal_id, "status": status, "error": error},
        )
    except Exception as exc:
        logger.warning("jb_outbound: remontée du résultat échouée (jb_id=%s) : %s", jb_id, exc)
