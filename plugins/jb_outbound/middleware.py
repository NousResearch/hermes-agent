"""Le `tool_execution` middleware : le cœur de « rien ne part sans accord ».

Contrat Hermes (hermes_cli/middleware.py) : la callback reçoit `tool_name`, `args`, `next_call`.
  - Appeler `next_call(args)` et retourner son résultat = exécution normale (pass-through).
  - NE PAS appeler `next_call` et retourner une valeur = court-circuit (l'outil ne s'exécute pas).

Pour un envoi sortant, on COURT-CIRCUITE : on enregistre l'envoi localement, on dépose une
proposition (DraftRequest) sur le daemon, et on rend au modèle un résultat synthétique. L'envoi
réel n'aura lieu qu'au RETOUR (replay.py), après approbation. Le replay passe par
`registry.dispatch` qui NE repasse PAS par ce middleware → pas de ré-interception (pas besoin de
flag). On n'intercepte jamais un appel interne (lecture, terminal, etc.).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _result(payload: dict) -> str:
    # Les outils Hermes renvoient une chaîne JSON ; on respecte ce format.
    return json.dumps(payload, ensure_ascii=False)


def make_middleware() -> Callable[..., Any]:
    def jb_outbound_tool_execution(
        *,
        tool_name: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        next_call: Callable[[Any], Any],
        **_: Any,
    ) -> Any:
        from . import classify, config, contributions, http_client, job_context, mapping, store

        # Plugin passif hors box Jean-Billie (JB_DECISION_PUSH_URL non posé) : ne rien changer.
        if not config.enabled():
            return next_call(args)

        decision = classify.classify(tool_name or "")

        if decision == classify.PASS:
            return next_call(args)

        if decision == classify.BLOCK:
            logger.warning("jb_outbound: outil d'envoi non répertorié BLOQUÉ : %s", tool_name)
            return _result(
                {
                    "status": "blocked",
                    "message": (
                        "Cette action n'est pas encore autorisée. Rien n'a été envoyé — "
                        "signalez-le à l'équipe pour l'activer."
                    ),
                }
            )

        # PROPOSE : court-circuit → proposition.
        jb_id = uuid.uuid4().hex
        draft = mapping.to_draft(tool_name or "", args or {})
        store.save(jb_id, tool_name or "", args or {}, draft["kind"], draft.get("to", ""))

        # On glisse notre identifiant local dans le payload : il nous reviendra dans la DecisionItem
        # (le control-plane round-trip le payload) → corrélation décision ↔ envoi en attente.
        body = dict(draft)
        body["payload"] = {**draft.get("payload", {}), "jb_id": jb_id}

        # Attribution : si l'interception a lieu pendant un job cron (skill → casquette), le draft
        # porte le département. Champs ADDITIFS, omis hors contexte job (chat libre) — le daemon Go
        # actuel ignore les champs inconnus (contrat répliqué côté Go en vague 2).
        ctx = job_context.current() or {}
        for key in ("department", "skill_id", "job_id"):
            value = ctx.get(key)
            if value:
                body[key] = value

        # Attribution MULTI-RÔLES (D3) : le LEAD (département du job_context parent) + les casquettes
        # DÉLÉGUÉES accumulées sur le tour (hook subagent_start) → `contributors`. Émis SEULEMENT s'il y
        # a au moins un support distinct (≥ 2 contributeurs) ; sinon omis → le portail retombe sur
        # `department` (legacy). En chat libre sans lead, on promeut le 1er contributeur en lead (Q5).
        lead = ctx.get("department")
        supports = [c for c in contributions.snapshot() if c.get("department")]
        contributors = []
        if lead:
            contributors.append({"department": lead, "role": "lead"})
            contributors.extend(c for c in supports if c["department"] != lead)
        elif supports:
            first = supports[0]
            promoted = {"department": first["department"], "role": "lead"}
            if first.get("skill_id"):
                promoted["skill_id"] = first["skill_id"]
            contributors.append(promoted)
            contributors.extend(c for c in supports[1:] if c["department"] != first["department"])
        if len(contributors) >= 2:
            body["contributors"] = contributors
        contributions.reset()  # un draft = une livraison → on repart propre (Q4)

        try:
            http_client.post_json(config.draft_url(), body)
        except Exception as exc:  # dépôt impossible → on n'a rien envoyé, on le dit franchement.
            store.mark(jb_id, "failed", str(exc))
            logger.warning("jb_outbound: dépôt de la proposition échoué (%s) : %s", tool_name, exc)
            return _result(
                {
                    "status": "error",
                    "message": "Je n'ai pas pu préparer la proposition pour l'instant. Rien n'est parti.",
                }
            )

        return _result(
            {
                "status": "queued_for_approval",
                "id": jb_id,
                "message": "C'est prêt : j'ai préparé la proposition. Rien ne part tant que vous n'avez pas validé.",
            }
        )

    return jb_outbound_tool_execution
