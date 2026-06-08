"""Store local des envois en attente d'approbation (~/.hermes/jb_pending/{jb_id}.json).

Persistant (volume monté) → survit au redémarrage du conteneur. Contient les ARGUMENTS complets de
l'appel d'outil pour rejouer l'envoi à l'identique après approbation. Reste STRICTEMENT local
(jamais relayé au control-plane) — minimisation RGPD. Aucun secret n'y figure (les tokens MCP
vivent dans le `.env` / l'URL scopée, pas dans les args d'un appel d'outil).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


def _dir() -> Path:
    d = _home() / "jb_pending"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(jb_id: str) -> Path:
    return _dir() / f"{jb_id}.json"


def save(jb_id: str, tool_name: str, args: Dict[str, Any], kind: str, to: str) -> None:
    record = {
        "id": jb_id,
        "tool_name": tool_name,
        "args": args or {},
        "kind": kind,
        "to": to,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = _path(jb_id).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _path(jb_id))  # écriture atomique


def load(jb_id: str) -> Optional[Dict[str, Any]]:
    if not jb_id:
        return None
    p = _path(jb_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def mark(jb_id: str, status: str, error: str = "") -> None:
    rec = load(jb_id)
    if rec is None:
        return
    rec["status"] = status
    if error:
        rec["error"] = error
    rec["settled_at"] = datetime.now(timezone.utc).isoformat()
    tmp = _path(jb_id).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _path(jb_id))


def pending_ids() -> List[str]:
    """Ids encore en attente (pour un balayage au démarrage)."""
    out: List[str] = []
    for f in _dir().glob("*.json"):
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("status") == "pending":
                out.append(f.stem)
        except Exception:
            continue
    return out
