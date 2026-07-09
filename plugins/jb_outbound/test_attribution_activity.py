"""Tests de l'attribution (stamp department/skill_id/job_id) et du fil d'activité.

Autonomes comme test_jb_outbound.py : HTTP loopback mocké, pas d'environnement Hermes complet.
Couvre : stamp présent en contexte job / absent hors contexte, résolution de la casquette depuis
le front-matter des skills (``casquette:`` gold, ``department:`` custom), gate ``JB_ACTIVITY_EVENTS``,
innocuité des échecs réseau, et le pont scheduler → plugin (cron/scheduler.py::run_job).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Rendre le paquet `jb_outbound` importable comme paquet de premier niveau (plugins/ sur le path).
_PLUGINS_DIR = Path(__file__).resolve().parents[1]
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

import jb_outbound.activity as activity  # noqa: E402
import jb_outbound.http_client as http_client  # noqa: E402
import jb_outbound.job_context as job_context  # noqa: E402
import jb_outbound.middleware as middleware  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isole HERMES_HOME et neutralise les gates JB pour CHAQUE test (baseline passive)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("JB_DECISION_PUSH_URL", raising=False)
    monkeypatch.delenv("JB_ACTIVITY_EVENTS", raising=False)
    # Filet de sécurité : jamais de contexte résiduel d'un test précédent.
    job_context._JOB_CTX.set(None)


@pytest.fixture
def posts(monkeypatch):
    """Active la boucle de proposition et capture tous les POST loopback (drafts + activité)."""
    monkeypatch.setenv("JB_DECISION_PUSH_URL", "http://127.0.0.1:8444/jb/decision")
    monkeypatch.setenv("JB_DRAFT_ADDR", "127.0.0.1:8442")
    captured: list = []
    monkeypatch.setattr(
        http_client, "post_json",
        lambda url, payload, timeout=10.0: captured.append((url, payload)) or 200,
    )
    return captured


def _write_skill(tmp_path, name: str, body: str) -> None:
    d = tmp_path / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")


_GOLD = "---\nname: relance-devis\ncasquette: Le Commercial\n---\n\n# Relance devis\n"
_CUSTOM = "---\nname: veille-presse\ndepartment: Le Marketing\n---\n\n# Veille presse\n"
_SANS = "---\nname: tri-boite\ndescription: Trie la boîte mail.\n---\n\n# Tri\n"


def _job(**over) -> dict:
    job = {"id": "a1b2c3d4e5f6", "name": "Relances du matin", "skills": ["relance-devis"]}
    job.update(over)
    return job


def _propose(tool: str = "send_message", args: dict | None = None) -> dict:
    """Passe un appel d'envoi dans le middleware (court-circuit attendu) et rend le résultat."""
    def next_call(_a):
        raise AssertionError("l'outil d'envoi NE doit PAS s'exécuter avant validation")

    return json.loads(
        middleware.make_middleware()(
            tool_name=tool, args=args or {"chat_id": "1", "content": "Hi"}, next_call=next_call
        )
    )


def _drafts(posts) -> list:
    return [p[1] for p in posts if p[0].endswith("/v1/draft")]


def _activities(posts) -> list:
    return [p[1] for p in posts if p[0].endswith("/v1/activity")]


# ---------------------------------------------------------------------------
# Tâche 1 — stamp d'attribution sur les drafts
# ---------------------------------------------------------------------------

def test_stamp_draft_en_contexte_job(posts, tmp_path):
    _write_skill(tmp_path, "relance-devis", _GOLD)
    token = job_context.job_started(_job())

    _propose()
    draft = _drafts(posts)[-1]
    assert draft["department"] == "Le Commercial"
    assert draft["skill_id"] == "relance-devis"
    assert draft["job_id"] == "a1b2c3d4e5f6"
    # L'attribution est portée au premier niveau du DraftRequest, pas dans le payload round-trip.
    assert "department" not in draft["payload"]

    job_context.job_finished(_job(), success=True, token=token)


def test_stamp_absent_hors_contexte(posts):
    _propose()
    draft = _drafts(posts)[-1]
    for key in ("department", "skill_id", "job_id"):
        assert key not in draft  # chat libre → champs OMIS, pas de null


def test_stamp_efface_apres_job(posts, tmp_path):
    _write_skill(tmp_path, "relance-devis", _GOLD)
    token = job_context.job_started(_job())
    job_context.job_finished(_job(), success=True, token=token)

    _propose()
    assert "department" not in _drafts(posts)[-1]
    assert job_context.current() is None


def test_casquette_gold_prioritaire_sur_department(posts, tmp_path):
    # Un skill qui porte les DEUX champs : `casquette:` (gold) gagne.
    _write_skill(tmp_path, "relance-devis", "---\ncasquette: Le Commercial\ndepartment: Autre\n---\n")
    token = job_context.job_started(_job())
    assert job_context.current()["department"] == "Le Commercial"
    job_context.job_finished(_job(), token=token)


def test_department_custom(posts, tmp_path):
    _write_skill(tmp_path, "veille-presse", _CUSTOM)
    token = job_context.job_started(_job(skills=["veille-presse"]))
    ctx = job_context.current()
    assert ctx["department"] == "Le Marketing"
    assert ctx["skill_id"] == "veille-presse"
    job_context.job_finished(_job(), token=token)


def test_skill_sans_casquette_stamp_partiel(posts, tmp_path):
    _write_skill(tmp_path, "tri-boite", _SANS)
    token = job_context.job_started(_job(skills=["tri-boite"]))

    _propose()
    draft = _drafts(posts)[-1]
    assert "department" not in draft  # pas de casquette déclarée → champ omis
    assert draft["skill_id"] == "tri-boite"  # attribution partielle conservée
    assert draft["job_id"] == "a1b2c3d4e5f6"

    job_context.job_finished(_job(), token=token)


def test_resolution_skill_en_categorie(posts, tmp_path):
    # Skill rangé sous une catégorie (ex. casquettes/relance-devis), référencé par nom nu.
    _write_skill(tmp_path, "casquettes/relance-devis", _GOLD)
    token = job_context.job_started(_job())
    assert job_context.current()["department"] == "Le Commercial"
    job_context.job_finished(_job(), token=token)


def test_passif_sans_box_ni_activite():
    # Ni JB_DECISION_PUSH_URL ni JB_ACTIVITY_EVENTS → job_started est un no-op total.
    assert job_context.job_started(_job()) is None
    assert job_context.current() is None


# ---------------------------------------------------------------------------
# Tâche 2 — fil d'activité (début/fin de job), gated par JB_ACTIVITY_EVENTS
# ---------------------------------------------------------------------------

def test_activity_off_par_defaut(posts, tmp_path):
    _write_skill(tmp_path, "relance-devis", _GOLD)
    token = job_context.job_started(_job())
    job_context.job_finished(_job(), success=True, token=token)
    assert _activities(posts) == []  # gate fermé → aucun évènement


def test_activity_on_emet_started_puis_finished(posts, tmp_path, monkeypatch):
    monkeypatch.setenv("JB_ACTIVITY_EVENTS", "1")
    _write_skill(tmp_path, "relance-devis", _GOLD)

    token = job_context.job_started(_job())
    job_context.job_finished(_job(), success=True, token=token)

    events = _activities(posts)
    assert [e["phase"] for e in events] == ["started", "finished"]
    started, finished = events
    assert started["status"] == "ok" and finished["status"] == "ok"
    for e in events:
        assert e["department"] == "Le Commercial"
        assert e["skill_id"] == "relance-devis"
        assert e["job_id"] == "a1b2c3d4e5f6"
        assert e["label"] == "Relances du matin"  # nom lisible du job (jobs.json)


def test_activity_status_error_si_echec(posts, monkeypatch):
    monkeypatch.setenv("JB_ACTIVITY_EVENTS", "1")
    token = job_context.job_started(_job(skills=[]))
    job_context.job_finished(_job(skills=[]), success=False, token=token)

    finished = _activities(posts)[-1]
    assert finished["phase"] == "finished" and finished["status"] == "error"
    assert "department" not in finished and "skill_id" not in finished  # job sans skill → omis


def test_activity_echec_reseau_avale(monkeypatch, tmp_path):
    # Daemon injoignable (route /v1/activity inexistante, conteneur down…) : le job continue.
    monkeypatch.setenv("JB_ACTIVITY_EVENTS", "1")
    _write_skill(tmp_path, "relance-devis", _GOLD)

    def _boom(url, payload, timeout=10.0):
        raise ConnectionError("connexion refusée")

    monkeypatch.setattr(http_client, "post_json", _boom)
    token = job_context.job_started(_job())  # ne lève pas
    assert job_context.current()["department"] == "Le Commercial"  # le contexte reste posé
    job_context.job_finished(_job(), success=True, token=token)  # ne lève pas
    assert job_context.current() is None


def test_activity_emit_sans_contexte_ne_leve_pas(posts, monkeypatch):
    monkeypatch.setenv("JB_ACTIVITY_EVENTS", "1")
    activity.emit("started", "ok", None)
    assert _activities(posts) == [{"phase": "started", "status": "ok"}]


# ---------------------------------------------------------------------------
# Pont scheduler → plugin (cron/scheduler.py::run_job)
# ---------------------------------------------------------------------------

def test_scheduler_run_job_pose_contexte_et_signale(posts, tmp_path, monkeypatch):
    """run_job (scheduler réel) pose le contexte pendant le job et émet started/finished."""
    monkeypatch.setenv("JB_ACTIVITY_EVENTS", "1")
    _write_skill(tmp_path, "relance-devis", _GOLD)

    import cron.scheduler as scheduler

    seen = {}

    def _fake_impl(job, defer_agent_teardown=False):
        # `defer_agent_teardown` : kwarg upstream v0.18 (toujours passé par run_job) — accepté
        # pour coller à la vraie signature de _run_job_impl, sans effet sur ce test.
        seen["ctx"] = job_context.current()  # visible PENDANT l'exécution du job
        return True, "doc", "réponse", None

    monkeypatch.setattr(scheduler, "_run_job_impl", _fake_impl)
    result = scheduler.run_job(_job())

    assert result[0] is True
    assert seen["ctx"]["department"] == "Le Commercial"
    assert seen["ctx"]["job_id"] == "a1b2c3d4e5f6"
    assert job_context.current() is None  # nettoyé après le job (threads de pool réutilisés)
    assert [e["phase"] for e in _activities(posts)] == ["started", "finished"]
    assert _activities(posts)[-1]["status"] == "ok"


def test_scheduler_run_job_echec_signale_error(posts, tmp_path, monkeypatch):
    monkeypatch.setenv("JB_ACTIVITY_EVENTS", "1")
    _write_skill(tmp_path, "relance-devis", _GOLD)

    import cron.scheduler as scheduler

    monkeypatch.setattr(
        scheduler,
        "_run_job_impl",
        lambda job, defer_agent_teardown=False: (False, "doc", "", "boom"),
    )
    result = scheduler.run_job(_job())

    assert result[0] is False
    finished = _activities(posts)[-1]
    assert finished["phase"] == "finished"
    assert finished["status"] == "error"
