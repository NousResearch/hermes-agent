"""Tests autonomes de l'outil « creer_support » (Ma marque, Option A).

Même style que test_jb_outbound.py : le HTTP loopback est mocké, aucun environnement Hermes complet
requis. Couvre le gating (box JB uniquement), la validation du type, le round-trip nominal (URL du
document), la tolérance de forme du contenu, et les chemins d'erreur (relais indisponible, erreur de
contenu white-label relayée) — jamais de bluff.
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

import jb_outbound.produce as produce  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Isole HERMES_HOME pour CHAQUE test (cohérence avec les autres suites du plugin)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.fixture
def on_box(monkeypatch):
    """Simule la box Jean-Billie (JB_DECISION_PUSH_URL posé par le bundle)."""
    monkeypatch.setenv("JB_DECISION_PUSH_URL", "http://127.0.0.1:8444/jb/decision")
    monkeypatch.setenv("JB_DRAFT_ADDR", "127.0.0.1:8442")


def _ok_response():
    return (
        200,
        {
            "status": "ok",
            "url": "https://media.example/devis.pdf?sig=abc",
            "title": "Devis : Mme Bernard",
            "fileName": "devis-mme-bernard.pdf",
        },
    )


def test_gated_hors_box(monkeypatch):
    """Sans JB_DECISION_PUSH_URL (Hermes standard), l'outil est invisible (check_fn faux)."""
    monkeypatch.delenv("JB_DECISION_PUSH_URL", raising=False)
    assert produce.check_creer_support_requirements() is False


def test_disponible_sur_box(on_box):
    assert produce.check_creer_support_requirements() is True


def test_type_inconnu_refuse_sans_appel(on_box, monkeypatch):
    called = []
    monkeypatch.setattr(produce, "_post", lambda *a, **k: called.append(a) or _ok_response())
    out = json.loads(produce.creer_support("affiche-geante", {"title": "x"}))
    assert out["status"] == "error"
    assert "presentation" in out["message"]  # la liste des familles est rappelée au modèle
    assert called == []  # aucun POST pour un type inconnu


def test_production_ok_round_trip(on_box, monkeypatch):
    captured = {}

    def fake_post(url, payload, timeout=0.0):
        captured["url"] = url
        captured["payload"] = payload
        return _ok_response()

    monkeypatch.setattr(produce, "_post", fake_post)
    out = json.loads(
        produce.creer_support(
            "devis",
            {"clientName": "Mme Bernard", "lines": [{"description": "Pose", "qty": 2, "unitPrice": 120}]},
            kit_id="kit-42",
        )
    )
    # Le POST suit le chemin loopback du daemon (même transport que les drafts).
    assert captured["url"] == "http://127.0.0.1:8442/v1/produce"
    assert captured["payload"]["type"] == "devis"
    assert captured["payload"]["kitId"] == "kit-42"
    assert captured["payload"]["content"]["clientName"] == "Mme Bernard"
    # L'agent reçoit l'URL signée + la consigne de partage tel quel.
    assert out["status"] == "ok"
    assert out["url"] == "https://media.example/devis.pdf?sig=abc"
    assert out["title"] == "Devis : Mme Bernard"
    assert out["fileName"] == "devis-mme-bernard.pdf"
    assert "Documents" in out["message"]


def test_contenu_en_chaine_json_tolere(on_box, monkeypatch):
    """Certains modèles sérialisent l'objet : une chaîne JSON est parsée, jamais de plantage."""
    captured = {}
    monkeypatch.setattr(
        produce, "_post", lambda url, payload, timeout=0.0: captured.update(payload) or _ok_response()
    )
    out = json.loads(produce.creer_support("post", '{"title": "Promo d\'automne"}'))
    assert out["status"] == "ok"
    assert captured["content"] == {"title": "Promo d'automne"}


def test_type_normalise_et_kit_absent_omis(on_box, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        produce, "_post", lambda url, payload, timeout=0.0: captured.update(payload) or _ok_response()
    )
    produce.creer_support("  DEVIS  ", {"clientName": "X"})
    assert captured["type"] == "devis"
    assert "kitId" not in captured  # pas de clé vide relayée


def test_relais_indisponible_message_franc(on_box, monkeypatch):
    """Daemon/control-plane injoignable : on le dit franchement, on ne bluffe pas."""

    def boom(url, payload, timeout=0.0):
        raise OSError("connexion refusée")

    monkeypatch.setattr(produce, "_post", boom)
    out = json.loads(produce.creer_support("presentation", {"title": "Offre 2026"}))
    assert out["status"] == "unavailable"
    assert "réessayer" in out["message"]


def test_erreur_de_contenu_relayee(on_box, monkeypatch):
    """Le message white-label du portail (« Donnez un titre… ») revient tel quel à l'agent."""
    monkeypatch.setattr(
        produce,
        "_post",
        lambda url, payload, timeout=0.0: (200, {"status": "error", "message": "Donnez un titre à votre visuel."}),
    )
    out = json.loads(produce.creer_support("post", {}))
    assert out["status"] == "error"
    assert out["message"] == "Donnez un titre à votre visuel."


def test_reponse_ok_sans_url_traitee_en_erreur(on_box, monkeypatch):
    """Un `ok` sans URL (réponse malformée) ne doit JAMAIS produire un faux succès."""
    monkeypatch.setattr(produce, "_post", lambda *a, **k: (200, {"status": "ok"}))
    out = json.loads(produce.creer_support("devis", {"clientName": "X"}))
    assert out["status"] == "error"


def test_register_enregistre_l_outil(on_box):
    """Le plugin enregistre creer_support via ctx.register_tool (zéro patch du cœur)."""
    import jb_outbound

    calls = {"tools": [], "middleware": [], "hooks": [], "aux": []}

    class FakeCtx:
        def register_tool(self, **kw):
            calls["tools"].append(kw)

        def register_middleware(self, kind, cb):
            calls["middleware"].append(kind)

        def register_hook(self, name, cb):
            calls["hooks"].append(name)

        def register_auxiliary_task(self, key, **kw):
            calls["aux"].append(key)

    jb_outbound.register(FakeCtx())
    names = [t["name"] for t in calls["tools"]]
    assert "creer_support" in names
    tool = next(t for t in calls["tools"] if t["name"] == "creer_support")
    assert tool["toolset"] == "jb_studio"
    assert tool["check_fn"] is produce.check_creer_support_requirements
    # Le schéma expose les 9 familles (enum fermé : l'agent ne peut pas inventer un type).
    enum = tool["schema"]["parameters"]["properties"]["type_de_support"]["enum"]
    assert set(enum) == set(produce.SUPPORT_TYPES)
    assert len(enum) == 9
    # Le handler délègue bien à creer_support (round-trip minimal, _post mocké par ailleurs inutile :
    # type inconnu → refus local sans réseau).
    out = json.loads(tool["handler"]({"type_de_support": "inconnu", "contenu": {}}))
    assert out["status"] == "error"
