"""Tests autonomes du rate-limit « burst » (Lane R).

Mock-first, horloge INJECTÉE → déterministe, AUCUN ``time.sleep`` réel. Couvre :
  * défaut OFF (``make_decision`` autorise toujours sans la variable d'env) ;
  * activation par env (``JB_RATE_LIMIT_ENABLED=1``) ;
  * le bucket respecte le RPM (refill au cours du temps simulé) ;
  * isolation PAR instance (un bucket plein n'affecte pas un autre) ;
  * le burst (×ratio) est honoré ;
  * la garde par-minute (famille ``turns``) refuse au N+1 ;
  * l'état persiste cross-process (relecture depuis le JSON) ;
  * le middleware renvoie ``rate_limited`` au dépassement et N'EXÉCUTE PAS l'outil.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Rendre `jb_outbound` importable comme paquet de premier niveau (plugins/ sur le path).
_PLUGINS_DIR = Path(__file__).resolve().parents[1]
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

import jb_outbound.middleware as middleware  # noqa: E402
import jb_outbound.rate_limit as rate_limit  # noqa: E402
import jb_outbound.store as store  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Isole HERMES_HOME → l'état du rate-limit est écrit sous tmp_path, jamais ~/.hermes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.fixture
def on(monkeypatch):
    """Active le rate-limit avec des seuils SERRÉS pour tester sans des centaines d'appels."""
    monkeypatch.setenv("JB_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("JB_RATE_LIMIT_EGRESS_RPM", "60")   # 60/min = 1 jeton/s — refill simple
    monkeypatch.setenv("JB_RATE_LIMIT_TURNS_RPM", "60")
    monkeypatch.setenv("JB_RATE_LIMIT_BURST_RATIO", "1.0")  # capacité = RPM (pas de marge burst)


class _Clock:
    """Horloge déterministe injectable : on avance le temps À LA MAIN, jamais de sleep."""

    def __init__(self, t: float = 1000.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, secs: float) -> None:
        self.t += secs


# ──────────────────────────────────────────────────────────────────────
# Kill-switch
# ──────────────────────────────────────────────────────────────────────

def test_disabled_par_defaut_autorise_toujours(monkeypatch):
    monkeypatch.delenv("JB_RATE_LIMIT_ENABLED", raising=False)
    assert rate_limit.enabled() is False
    clock = _Clock()
    # Même appelé 10 000 fois au même instant, aucune décision ne refuse quand c'est OFF.
    for _ in range(10_000):
        assert rate_limit.make_decision("egress", "inst", clock=clock) is True
    # Et rien n'est persisté (chemin OFF n'écrit pas d'état).
    assert not (Path(rate_limit._state_path())).exists()


def test_enabled_par_env(on):
    assert rate_limit.enabled() is True


# ──────────────────────────────────────────────────────────────────────
# Bucket : respect du RPM + refill au cours du temps simulé
# ──────────────────────────────────────────────────────────────────────

def test_bucket_respecte_le_rpm_puis_refill(on):
    clock = _Clock()
    # capacité = 60 (RPM 60, burst 1.0) → 60 autorisés d'affilée au même instant…
    for i in range(60):
        assert rate_limit.make_decision("egress", "inst", clock=clock) is True, i
    # …le 61e (même instant, bucket vide) est REFUSÉ.
    assert rate_limit.make_decision("egress", "inst", clock=clock) is False

    # On avance d'1 s → refill de 1 jeton (60/min) → exactement 1 nouvel appel autorisé.
    clock.advance(1.0)
    assert rate_limit.make_decision("egress", "inst", clock=clock) is True
    assert rate_limit.make_decision("egress", "inst", clock=clock) is False


def test_refill_complet_apres_une_minute(on):
    clock = _Clock()
    for _ in range(60):
        assert rate_limit.make_decision("egress", "inst", clock=clock) is True
    assert rate_limit.make_decision("egress", "inst", clock=clock) is False
    # Une minute plus tard, le bucket est plein à nouveau (borné à la capacité).
    clock.advance(60.0)
    for _ in range(60):
        assert rate_limit.make_decision("egress", "inst", clock=clock) is True
    assert rate_limit.make_decision("egress", "inst", clock=clock) is False


# ──────────────────────────────────────────────────────────────────────
# Isolation par instance
# ──────────────────────────────────────────────────────────────────────

def test_isolation_par_instance(on):
    clock = _Clock()
    # job-1 vide son bucket…
    for _ in range(60):
        assert rate_limit.make_decision("egress", "job-1", clock=clock) is True
    assert rate_limit.make_decision("egress", "job-1", clock=clock) is False
    # …job-2 garde son propre bucket plein (aucune contamination).
    assert rate_limit.make_decision("egress", "job-2", clock=clock) is True


def test_isolation_par_famille(on):
    clock = _Clock()
    # egress et turns sont des buckets distincts pour la même instance.
    for _ in range(60):
        assert rate_limit.make_decision("egress", "inst", clock=clock) is True
    assert rate_limit.make_decision("egress", "inst", clock=clock) is False
    # turns de la même instance reste plein.
    assert rate_limit.make_decision("turns", "inst", clock=clock) is True


# ──────────────────────────────────────────────────────────────────────
# Burst
# ──────────────────────────────────────────────────────────────────────

def test_burst_honore(monkeypatch):
    monkeypatch.setenv("JB_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("JB_RATE_LIMIT_EGRESS_RPM", "60")
    monkeypatch.setenv("JB_RATE_LIMIT_BURST_RATIO", "1.5")  # capacité = 90
    clock = _Clock()
    for i in range(90):
        assert rate_limit.make_decision("egress", "inst", clock=clock) is True, i
    assert rate_limit.make_decision("egress", "inst", clock=clock) is False


# ──────────────────────────────────────────────────────────────────────
# Garde par-minute (turns) : refus au N+1
# ──────────────────────────────────────────────────────────────────────

def test_turns_guard_refuse_au_n_plus_un(monkeypatch):
    monkeypatch.setenv("JB_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("JB_RATE_LIMIT_TURNS_RPM", "5")   # 5 tours/min
    monkeypatch.setenv("JB_RATE_LIMIT_BURST_RATIO", "1.0")  # capacité = 5
    clock = _Clock()
    for i in range(5):
        assert rate_limit.make_decision("turns", "mission:x", clock=clock) is True, i
    # 6e tour dans la même minute → REJET (pas de différé, l'appelant skippe).
    assert rate_limit.make_decision("turns", "mission:x", clock=clock) is False


# ──────────────────────────────────────────────────────────────────────
# Persistance cross-process : relecture depuis le JSON
# ──────────────────────────────────────────────────────────────────────

def test_etat_persiste_sur_disque(on, tmp_path):
    clock = _Clock()
    for _ in range(60):
        rate_limit.make_decision("egress", "inst", clock=clock)
    # Le fichier d'état existe et porte le bucket consommé (un autre PROCESS le relirait).
    state = json.loads(Path(rate_limit._state_path()).read_text(encoding="utf-8"))
    assert "egress:inst" in state
    assert state["egress:inst"]["tokens"] < 1.0  # quasi vide
    # Une « nouvelle décision » (simulant un autre process : aucun bucket en mémoire) lit l'état
    # persisté et refuse au même instant.
    assert rate_limit.make_decision("egress", "inst", clock=clock) is False


# ──────────────────────────────────────────────────────────────────────
# Câblage middleware : rejet propre, l'outil ne s'exécute pas
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def posts(monkeypatch):
    """Active la boucle de proposition et capture les POST (jamais de vrai réseau)."""
    from jb_outbound import http_client

    monkeypatch.setenv("JB_DECISION_PUSH_URL", "http://127.0.0.1:8444/jb/decision")
    monkeypatch.setenv("JB_DRAFT_ADDR", "127.0.0.1:8442")
    captured: list = []
    monkeypatch.setattr(
        http_client, "post_json",
        lambda url, payload, timeout=10.0: captured.append((url, payload)) or 200,
    )
    return captured


def test_middleware_rate_limited_n_execute_pas_et_ne_depose_pas(posts, monkeypatch):
    monkeypatch.setenv("JB_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("JB_RATE_LIMIT_EGRESS_RPM", "1")
    monkeypatch.setenv("JB_RATE_LIMIT_BURST_RATIO", "1.0")  # capacité = 1

    def next_call(_a):
        raise AssertionError("l'outil d'envoi NE doit PAS s'exécuter quand on est rate-limité")

    mw = middleware.make_middleware()

    # 1er envoi : autorisé → proposition déposée comme d'habitude.
    out1 = json.loads(mw(tool_name="send_message", args={"chat_id": "1", "content": "a"}, next_call=next_call))
    assert out1["status"] == "queued_for_approval"
    assert len(posts) == 1

    # 2e envoi immédiat : bucket vide → REJET PROPRE. Pas d'exécution, pas de nouveau dépôt.
    out2 = json.loads(mw(tool_name="send_message", args={"chat_id": "1", "content": "b"}, next_call=next_call))
    assert out2["status"] == "rate_limited"
    assert "id" in out2
    assert len(posts) == 1  # aucun POST supplémentaire — rien n'est parti
    # L'enregistrement local du 2e est marqué rate_limited (traçable, non envoyé).
    assert store.load(out2["id"])["status"] == "rate_limited"


def test_middleware_off_se_comporte_comme_avant(posts, monkeypatch):
    """Invariant : sans la variable, le middleware propose exactement comme avant (régression OFF)."""
    monkeypatch.delenv("JB_RATE_LIMIT_ENABLED", raising=False)
    mw = middleware.make_middleware()
    for i in range(50):  # bien au-delà de tout seuil — aucun rejet possible quand OFF
        out = json.loads(mw(tool_name="send_message", args={"chat_id": "1", "content": str(i)}, next_call=lambda a: None))
        assert out["status"] == "queued_for_approval"
    assert len(posts) == 50
