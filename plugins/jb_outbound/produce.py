"""Outil « creer_support » : produire un support brandé (Ma marque, Option A — box Jean-Billie).

L'agent ne dessine JAMAIS lui-même : il émet une INTENTION structurée { type, contenu } que la
plateforme rend de façon DÉTERMINISTE (gabarits fixes, charte du client appliquée) puis range dans
l'Espace Documents du client. Le POST part sur le loopback du control daemon (127.0.0.1, sans mTLS,
moindre privilège : le workload ne détient aucun cert de flotte), qui le relaie au control-plane
avec SON identité mTLS — EXACTEMENT le même chemin que les drafts et request_tool_connection.

PUREMENT INTERNE : le document est DÉPOSÉ dans l'espace du client, rien ne part vers un tiers
(« rien ne part sans accord » vise l'egress tiers). L'ENVOI ultérieur du document (email,
publication) repasse par la boucle de proposition comme tout envoi — ce module n'y touche pas.

Enregistré PAR LE PLUGIN jb_outbound (ctx.register_tool, cf. __init__.py) : zéro patch du cœur.
Gated sur JB_DECISION_PUSH_URL (posé par le bundle Jean-Billie) : hors box, l'outil est invisible.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Les 9 familles rendues par la plateforme (source de vérité : intent.ts côté portail).
SUPPORT_TYPES = (
    "presentation",
    "devis",
    "facture",
    "post",
    "carrousel",
    "story",
    "prospectus",
    "signature",
    "lettre",
)

# Le rendu (PPTX/PDF/images + rangement) traverse daemon → control-plane → portail : timeout généreux,
# LÉGÈREMENT au-dessus de celui du relais daemon (90 s) pour que l'erreur amont arrive avant la nôtre.
_TIMEOUT_S = 100.0

_MSG_UNAVAILABLE = (
    "Je n'ai pas réussi à préparer ce document pour l'instant. Rien n'est perdu, on peut réessayer dans un moment."
)


def _post(url: str, payload: dict, timeout: float = _TIMEOUT_S) -> Tuple[int, dict]:
    """POST JSON qui LIT la réponse (contrairement à http_client.post_json, statut seul).

    Loopback uniquement (le daemon). Lève en cas d'erreur réseau/HTTP — l'appelant traduit en
    message franc. Corps de réponse borné implicitement (le daemon renvoie une petite enveloppe).
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (loopback only)
        body = resp.read()
    return int(getattr(resp, "status", 200)), (json.loads(body.decode("utf-8")) if body else {})


def _result(payload: dict) -> str:
    # Les outils Hermes renvoient une chaîne JSON ; on respecte ce format (cf. middleware.py).
    return json.dumps(payload, ensure_ascii=False)


def creer_support(type_de_support: str, contenu: Any, kit_id: Optional[str] = None) -> str:
    """Demande à la plateforme de produire le support et renvoie l'URL du document.

    Renvoie un JSON `{status, url?, title?, fileName?, message}` : sur `ok`, `url` est le lien signé
    du document (déjà rangé dans les Documents du client) que l'agent partage TEL QUEL ; sur `error`,
    `message` explique ce qui manque (contenu insuffisant) ; sur `unavailable`, la production n'a pas
    pu aboutir — le dire franchement, pas de bluff.
    """
    from . import config

    t = (type_de_support or "").strip().lower()
    if t not in SUPPORT_TYPES:
        return _result(
            {
                "status": "error",
                "message": (
                    "Type de support inconnu. Types possibles : " + ", ".join(SUPPORT_TYPES) + "."
                ),
            }
        )

    # Le contenu peut arriver en objet (nominal) ou en chaîne JSON (certains modèles sérialisent) :
    # on tolère les deux, jamais de plantage sur une forme inattendue.
    if isinstance(contenu, str):
        try:
            contenu = json.loads(contenu)
        except Exception:
            contenu = {}
    if not isinstance(contenu, dict):
        contenu = {}

    payload: Dict[str, Any] = {"type": t, "content": contenu}
    kit = (kit_id or "").strip()
    if kit:
        payload["kitId"] = kit

    try:
        status, out = _post(config.produce_url(), payload)
    except Exception as exc:
        logger.warning("jb_outbound: production de support indisponible (%s): %s", t, exc)
        return _result({"status": "unavailable", "message": _MSG_UNAVAILABLE})

    if status < 200 or status >= 300:
        return _result({"status": "unavailable", "message": _MSG_UNAVAILABLE})

    # Enveloppe du daemon (ProduceResp) : status ok|error. Une erreur de CONTENU porte un message
    # déjà propre pour le client (« Donnez un titre… ») que l'agent reformule.
    if out.get("status") != "ok" or not out.get("url"):
        return _result(
            {
                "status": "error",
                "message": out.get("message") or _MSG_UNAVAILABLE,
            }
        )

    return _result(
        {
            "status": "ok",
            "url": out.get("url", ""),
            "title": out.get("title", ""),
            "fileName": out.get("fileName", ""),
            "message": (
                "Le document est prêt et rangé dans les Documents du client. Partage-lui ce lien tel quel "
                "(ne le modifie pas), puis propose-lui des ajustements."
            ),
        }
    )


def check_creer_support_requirements() -> bool:
    """Disponible UNIQUEMENT sur la box Jean-Billie (même garde que le reste du plugin).

    `JB_DECISION_PUSH_URL` est posé par le bundle Jean-Billie. En CLI/dev standard sans cette
    variable, l'outil n'apparaît pas (il n'aurait personne à qui parler) — box non-JB = passif.
    """
    from . import config

    return config.enabled()


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CREER_SUPPORT_SCHEMA = {
    "name": "creer_support",
    "description": (
        "Produit un document aux couleurs du client (sa charte est appliquée automatiquement) et le "
        "range dans ses Documents. Types : presentation (diaporama), devis, facture, prospectus, "
        "lettre (courrier à en-tête), post (visuel carré), story (visuel vertical), carrousel (série "
        "de visuels), signature (signature email). Tu fournis le CONTENU (textes, lignes, arguments) ; "
        "ne fabrique jamais le visuel toi-même. Partage ensuite le lien renvoyé TEL QUEL (n'invente "
        "jamais de lien) et propose des ajustements. Si le statut est « error » ou « unavailable », "
        "dis simplement et franchement ce qui manque ou que ça n'a pas abouti."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "type_de_support": {
                "type": "string",
                "enum": list(SUPPORT_TYPES),
                "description": "La famille de support à produire.",
            },
            "contenu": {
                "type": "object",
                "description": (
                    "Le contenu structuré, selon le type. presentation: {title, subtitle?, sections: "
                    "[{heading, bullets: [..]}]} ; devis/facture: {number?, date?, fromName?, clientName?, "
                    "clientInfo?, lines: [{description, qty, unitPrice}], vatRate?, validity?, notes?} ; "
                    "post/story: {title, text?} ; carrousel: {title, points: [une idée par page], cta?} ; "
                    "prospectus: {headline, text?, points?, contact?} ; signature: {fullName, role?, phone?, "
                    "email?, website?, fromName?} ; lettre: {date?, subject?, body?, contact?}."
                ),
            },
            "kit_id": {
                "type": "string",
                "description": (
                    "Identifiant d'une charte précise (optionnel : par défaut, la charte du client)."
                ),
            },
        },
        "required": ["type_de_support", "contenu"],
    },
}
