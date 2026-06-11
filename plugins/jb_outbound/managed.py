"""Table managée des ACTIONS des MCP additionnels (hors Composio).

Déposée par le bundle Jean-Billie à ``<HERMES_HOME>/jb_outbound/managed.json`` (cf. monorepo
``packages/config-generator/src/mcp.ts:managedTableFor`` + ``materialize.ts``). Elle associe le nom
d'outil RUNTIME (``mcp_<serveur>_<outil>``, tel qu'Hermes l'enregistre) à une action
``{label, kind}``.

Le garde-fou (``classify.py``) n'intercepte nativement QUE ``mcp_composio_*`` ; les outils des autres
MCP s'exécutent sans validation. Cette table le corrige SANS bloquer : une fonction listée ici est une
ACTION (write/egress) → elle devient une proposition « à valider » (dashboard) ; toute autre fonction
d'un MCP additionnel reste une lecture → exécution normale.

Lecture TOLÉRANTE : fichier absent/illisible → aucune action managée (comportement inchangé). Le cache
suit le ``mtime`` du fichier → un opérateur qui réécrit ``managed.json`` est pris en compte au rechargement.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Cache mémoïsé par (chemin, mtime) — évite de relire le fichier à chaque appel d'outil tout en
# captant une mise à jour de la table (réécriture par l'opérateur).
_cache: Dict[str, Any] = {"path": None, "mtime": None, "write_tools": {}}


def _home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


def _path() -> Path:
    override = os.getenv("JB_OUTBOUND_MANAGED")  # échappatoire de test / chemin custom
    return Path(override) if override else _home() / "jb_outbound" / "managed.json"


def _load() -> Dict[str, Dict[str, Any]]:
    p = _path()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        # Fichier absent → aucune action managée (réinitialise le cache pour ce chemin).
        _cache.update(path=str(p), mtime=None, write_tools={})
        return {}

    if _cache["path"] == str(p) and _cache["mtime"] == mtime:
        return _cache["write_tools"]

    write_tools: Dict[str, Dict[str, Any]] = {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data.get("writeTools") if isinstance(data, dict) else None
        if isinstance(raw, dict):
            for name, action in raw.items():
                if isinstance(action, dict) and action.get("label"):
                    write_tools[str(name)] = {
                        "label": str(action["label"]),
                        "kind": str(action.get("kind") or "action"),
                    }
    except Exception:
        write_tools = {}  # JSON cassé → fail-soft (aucune action managée)

    _cache.update(path=str(p), mtime=mtime, write_tools=write_tools)
    return write_tools


def action_for(tool_name: str) -> Optional[Dict[str, Any]]:
    """Renvoie ``{label, kind}`` si ``tool_name`` est une ACTION managée, sinon ``None`` (= lecture)."""
    if not tool_name:
        return None
    return _load().get(tool_name)
