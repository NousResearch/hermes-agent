"""Contexte d'attribution du job cron courant (casquette / skill / job).

Posé par le scheduler cron (``cron/scheduler.py::run_job``, via le pont ``_jb_job_hooks``) au
lancement de chaque job, lu par le middleware (``middleware.py``) pour estampiller les
DraftRequest avec le « département » de la tâche : ``department`` / ``skill_id`` / ``job_id``.

Pourquoi une ``ContextVar`` et pas une variable d'environnement : les jobs cron tournent dans des
THREADS du processus gateway (pool parallèle de ``tick()``) — ``os.environ`` est global au
processus, des jobs concurrents s'écraseraient mutuellement. Hermes a déjà migré l'état de
session vers des ContextVars pour cette raison (``gateway/session_context.py``) et propage le
contexte à chaque saut de thread (``copy_context`` dans ``_run_job_impl``,
``propagate_context_to_thread`` pour les outils) : une ContextVar posée dans ``run_job`` est donc
visible du middleware pendant TOUTE l'exécution du job, sans fuite entre jobs.

La casquette vient du front-matter du skill attaché au job : champ ``casquette:`` (skills gold
Jean-Billie) ou ``department:`` (skills custom). Résolution best-effort : toute erreur → champs
absents, jamais d'exception — l'attribution ne doit JAMAIS faire échouer un job.
"""

from __future__ import annotations

import logging
import os
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Contexte du job cron courant ({job_id, skill_id, department, label}) ou None hors job.
_JOB_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar("jb_job_ctx", default=None)


def current() -> Optional[Dict[str, Any]]:
    """Contexte d'attribution du job courant, ou ``None`` hors job cron (chat libre)."""
    return _JOB_CTX.get()


def job_started(job: Optional[Dict[str, Any]]) -> Optional[object]:
    """Pose le contexte d'attribution et signale le début du job. N'échoue jamais.

    Appelé par le scheduler au lancement d'un job cron. Renvoie le token de reset à repasser
    à ``job_finished`` (ou ``None`` si le plugin est passif : ni boucle de proposition
    ``JB_DECISION_PUSH_URL``, ni fil d'activité ``JB_ACTIVITY_EVENTS``).
    """
    try:
        from . import activity, config

        if not (config.enabled() or activity.enabled()):
            return None
        ctx = _build_ctx(job)
        token = _JOB_CTX.set(ctx)
        activity.emit("started", "ok", ctx)
        return token
    except Exception:
        logger.debug("jb_outbound: job_started en échec (ignoré)", exc_info=True)
        return None


def job_finished(job: Optional[Dict[str, Any]], success: bool = True, token: Optional[object] = None) -> None:
    """Signale la fin du job et nettoie le contexte. N'échoue jamais.

    Le nettoyage est indispensable : les threads du pool cron sont RÉUTILISÉS — sans reset, un
    job suivant sans skill hériterait de l'attribution du précédent.
    """
    try:
        from . import activity

        if activity.enabled():
            ctx = _JOB_CTX.get() or _build_ctx(job)
            activity.emit("finished", "ok" if success else "error", ctx)
    except Exception:
        logger.debug("jb_outbound: job_finished en échec (ignoré)", exc_info=True)
    finally:
        try:
            if token is not None:
                _JOB_CTX.reset(token)
            else:
                _JOB_CTX.set(None)
        except Exception:
            _JOB_CTX.set(None)


# ---------------------------------------------------------------------------
# Construction du contexte (job → {job_id, skill_id, department, label})
# ---------------------------------------------------------------------------

def _build_ctx(job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    job = job or {}
    skills = _skill_names(job)
    department, skill_id = _resolve_department(skills)
    return {
        "job_id": str(job.get("id") or "").strip() or None,
        "skill_id": skill_id,
        "department": department,
        "label": str(job.get("name") or "").strip() or None,
    }


def _skill_names(job: Dict[str, Any]) -> List[str]:
    """Skills du job, dans l'ordre (champ canonique ``skills``, repli legacy ``skill``)."""
    raw = job.get("skills")
    if raw is None:
        raw = [job.get("skill")] if job.get("skill") else []
    elif isinstance(raw, str):
        raw = [raw]
    out: List[str] = []
    for item in raw if isinstance(raw, list) else []:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _resolve_department(skills: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """(department, skill_id) du job : premier skill qui déclare une casquette.

    ``casquette:`` (gold) est lu avant ``department:`` (custom). Si aucun skill ne déclare de
    département, ``skill_id`` retombe sur le premier skill du job (attribution partielle).
    """
    fallback = skills[0] if skills else None
    for name in skills:
        try:
            fm = _skill_frontmatter(name)
        except Exception:
            continue
        for key in ("casquette", "department"):
            value = fm.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), name
    return None, fallback


def _home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


def _find_skill_md(name: str) -> Optional[Path]:
    """Localise le fichier d'un skill sous ``<HERMES_HOME>/skills`` (miroir allégé de skills_tool).

    Stratégies : chemin direct (``name/SKILL.md``, couvre aussi ``catégorie/name``), fichier plat
    ``name.md``, puis recherche récursive par nom de dossier. Refuse toute forme de traversée.
    """
    if not name or ".." in name.replace("\\", "/").split("/") or Path(name).is_absolute() or Path(name).drive:
        return None
    skills_dir = _home() / "skills"
    if not skills_dir.is_dir():
        return None
    direct = skills_dir / name
    if (direct / "SKILL.md").is_file():
        return direct / "SKILL.md"
    if direct.with_suffix(".md").is_file():
        return direct.with_suffix(".md")
    leaf = name.replace("\\", "/").split("/")[-1]
    for cand in skills_dir.rglob("SKILL.md"):
        if cand.parent.name == leaf:
            return cand
    for cand in skills_dir.rglob(f"{leaf}.md"):
        if cand.name != "SKILL.md":
            return cand
    return None


def _skill_frontmatter(name: str) -> Dict[str, str]:
    path = _find_skill_md(name)
    if path is None:
        return {}
    try:
        return _parse_simple_frontmatter(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_simple_frontmatter(text: str) -> Dict[str, str]:
    """Extraction minimale du front-matter YAML : clés scalaires de premier niveau.

    Suffisant pour ``casquette:`` / ``department:`` — pas de dépendance yaml ni du cœur Hermes
    (le plugin reste autonome, même esprit que ``http_client.py``). Les lignes indentées (blocs,
    listes) sont ignorées.
    """
    if not text.startswith("---"):
        return {}
    end = re.search(r"\n---\s*(\n|$)", text[3:])
    if not end:
        return {}
    out: Dict[str, str] = {}
    for line in text[3 : end.start() + 3].splitlines():
        if not line.strip() or line[:1] in (" ", "\t") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip().lower()] = value.strip().strip("'\"")
    return out
