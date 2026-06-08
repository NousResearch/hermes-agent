"""Petit client HTTP loopback (stdlib uniquement — pas de dépendance).

Sert à POSTer les DraftRequest / ResultRequest au control daemon sur 127.0.0.1, en clair
(le daemon n'expose ces routes que sur le loopback, sans mTLS — moindre privilège : le
workload ne détient aucun cert de flotte).
"""

from __future__ import annotations

import json
import urllib.request


def post_json(url: str, payload: dict, timeout: float = 10.0) -> int:
    """POST un corps JSON. Renvoie le code HTTP. Lève en cas d'erreur réseau/HTTP."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (loopback only)
        return int(getattr(resp, "status", 200))
