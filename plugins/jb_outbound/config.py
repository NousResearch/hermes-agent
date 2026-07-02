"""Configuration du plugin jb_outbound (endpoints loopback de la boucle de proposition).

Les valeurs viennent du `.env` du bundle Jean-Billie (posées par materializeBundle) et du
`daemon.env` (golden image). Toutes en LOOPBACK, non sensibles. Voir
`packages/provisioning/src/bundle/materialize.ts` côté monorepo.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

# Adresse host:port où le control daemon écoute les drafts/résultats (RouteDraft/RouteResult).
_DEFAULT_DRAFT_ADDR = "127.0.0.1:8442"
# URL loopback du listener de CE plugin, où le daemon pousse les décisions approuvées.
_DEFAULT_PUSH_URL = "http://127.0.0.1:8444/jb/decision"


def _draft_addr() -> str:
    return os.getenv("JB_DRAFT_ADDR", _DEFAULT_DRAFT_ADDR).strip() or _DEFAULT_DRAFT_ADDR


def draft_url() -> str:
    """URL où POSTer un DraftRequest (ALLER)."""
    return f"http://{_draft_addr()}/v1/draft"


def result_url() -> str:
    """URL où POSTer un ResultRequest (RETOUR)."""
    return f"http://{_draft_addr()}/v1/result"


def activity_url() -> str:
    """URL où POSTer un évènement d'activité (début/fin de job — cf. activity.py)."""
    return f"http://{_draft_addr()}/v1/activity"


def produce_url() -> str:
    """URL où POSTer une intention de PRODUCTION d'un support brandé (cf. produce.py)."""
    return f"http://{_draft_addr()}/v1/produce"


def _push_url() -> str:
    return os.getenv("JB_DECISION_PUSH_URL", _DEFAULT_PUSH_URL).strip() or _DEFAULT_PUSH_URL


def listen_addr() -> tuple[str, int]:
    """Host/port où binder le listener de décisions (dérivé de JB_DECISION_PUSH_URL)."""
    p = urlparse(_push_url())
    return (p.hostname or "127.0.0.1", int(p.port or 8444))


def decision_path() -> str:
    """Chemin HTTP attendu pour le push des décisions."""
    return urlparse(_push_url()).path or "/jb/decision"


def enabled() -> bool:
    """Vrai uniquement sur la box Jean-Billie (JB_DECISION_PUSH_URL posé par le bundle).

    En CLI/dev local sans cette variable, le plugin reste passif (pas de listener, pas
    d'interception) — il ne perturbe pas un usage standard de Hermes.
    """
    return bool(os.getenv("JB_DECISION_PUSH_URL"))
