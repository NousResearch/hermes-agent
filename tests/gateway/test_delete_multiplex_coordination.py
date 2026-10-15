"""Delete einer lebenden Sekundärprofil-Sitzung: keine Koordinationsverweigerung.

🔴 Defekt (gemessen 2026-10-15, Hermes-Desktop "Chat löschen"): der Delete-Wächter
in ``gateway/session_mutations.py`` verglich ``store._routing_db.db_path`` mit dem
Store, der gelöscht wird. ``_routing_db`` ist AUSDRÜCKLICH die eine, von allen
Profilen gemeinsam genutzte Routing-Index-Datei (#66887) - für ein multiplextes
Sekundärprofil (z. B. ``recherche``) können die beiden Pfade also niemals
übereinstimmen. Jedes Löschen einer im Runner lebenden Sitzung eines
Sekundärprofils endete daher dauerhaft mit HTTP 409
``runtime_coordination_required``, egal ob Zug läuft oder nicht.

Der Runner muss den Store ERREICHEN, dem die Sitzung gehört (``_db_for_key``
löst das aus dem Sitzungsschlüssel auf), nicht die Routing-Datei teilen.
"""
import asyncio
from types import SimpleNamespace

import pytest

import gateway.session_mutations as sm
from hermes_state_runtime import RuntimeStoreError

SID = "20261010_080057_51078e"
PROFIL_PFAD = "/home/x/.hermes/profiles/recherche/state.db"
ROOT_PFAD = "/home/x/.hermes/state.db"


class FakeDb:
    def __init__(self, pfad, zeilen):
        self.db_path = pfad
        self._zeilen = zeilen

    def get_session(self, sid):
        zeile = self._zeilen.get(sid)
        # Wie die echte Zeile: die Projektion liest runtime_generation.
        return None if zeile is None else {**zeile, "runtime_generation": 1}


def _lauf(store, db, monkeypatch):
    """_mutate_session('delete') mit gefaelschten Bausteinen einmal durchlaufen."""
    import hermes_state_media as media
    import hermes_state_mutation_retirement as retirement

    monkeypatch.setattr(retirement, "has_mutation_receipt", lambda *a, **k: True)
    monkeypatch.setattr(media, "collect_retired_media", lambda *a, **k: None)

    import gateway.session_mutations as _sm
    monkeypatch.setattr(_sm, "retire_live_sessions", lambda *a, **k: None)

    def fake_mutate(db_, **kw):
        kw["_live_guard"]({SID})
        return {"execution_generation": 1, "deleted_ids": [SID]}

    monkeypatch.setattr(sm, "mutate_runtime_session", fake_mutate)

    live = SimpleNamespace(route="r1",
                           event_stream=SimpleNamespace(publish=lambda *a, **k: None))
    authority = SimpleNamespace(
        profile_id="p1", epoch="e1", db=db, sessions={SID: live},
        runner=SimpleNamespace(session_store=store),
        _require_admission_open=lambda: None,
        authorize=lambda *a, **k: None, _schedule=lambda *a, **k: None)
    actor = SimpleNamespace(profile_id="p1", subject="u1",
                            capabilities={"session:control"})
    ref = SimpleNamespace(profile_id="p1", session_id=SID)
    params = {"session_id": SID, "request_id": "req-1", "expected_revision": 1,
              "operation": "delete", "payload": {}}
    return asyncio.run(sm._mutate_session(authority, actor, ref, params))


def _profil_db():
    return FakeDb(PROFIL_PFAD, {SID: {"session_key": f"agent:recherche:local:dm:{SID}"}})


def test_loeschen_lebender_sekundaerprofil_sitzung_wird_nicht_verweigert(monkeypatch):
    routing = FakeDb(ROOT_PFAD, {})
    profil = _profil_db()
    store = SimpleNamespace(
        _routing_db=routing,
        _db_for_key=lambda key: profil if (key or "").startswith("agent:recherche") else routing)
    ergebnis = _lauf(store, profil, monkeypatch)
    assert ergebnis["execution_generation"] == 1


def test_loeschen_einer_profil_sitzung_aus_dem_root_store_bleibt_moeglich(monkeypatch):
    routing = FakeDb(ROOT_PFAD, {SID: {"session_key": f"agent:main:dm:{SID}"}})
    store = SimpleNamespace(
        _routing_db=routing,
        _db_for_key=lambda key: routing)
    ergebnis = _lauf(store, routing, monkeypatch)
    assert ergebnis["execution_generation"] == 1


def test_verweigerung_bleibt_wenn_der_runner_den_store_nicht_erreicht(monkeypatch):
    fremd = FakeDb("/home/x/.hermes/profiles/other/state.db", {})
    routing = FakeDb(ROOT_PFAD, {})
    profil = _profil_db()
    store = SimpleNamespace(_routing_db=routing, _db_for_key=lambda key: fremd)
    with pytest.raises(RuntimeStoreError) as exc:
        _lauf(store, profil, monkeypatch)
    assert exc.value.reason == "runtime_coordination_required"