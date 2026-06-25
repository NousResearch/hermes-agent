"""Rate-limit « burst » Jean-Billie — deux verrous additifs, défaut OFF (Lane R).

Deux chemins de DÉBORDEMENT, jamais d'ouverture : le rate-limit n'autorise jamais un envoi ;
il n'ajoute qu'un **refus propre** quand une instance tape trop vite.

1. **Egress** (`make_decision("egress", key)`) : posé dans `middleware.py` JUSTE AVANT le POST
   réel d'une proposition au daemon (`http_client.post_json`). Au dépassement, le middleware rend
   un résultat synthétique ``status="rate_limited"`` (même forme que ``queued_for_approval``) et
   N'APPELLE PAS ``next_call`` → l'outil réel ne s'exécute pas, rien n'est déposé.
2. **Tours / re-drive** (`make_decision("turns", key)`) : posé au seam de re-drive autonome de la
   flotte (``gateway/run.py::GatewayRunner._watchdog_sweep`` → ``adapter.handle_message``), AVANT
   de relancer un tour pour une mission bloquée. Au dépassement, le tour n'est PAS re-drivé.

Invariant : quand ``JB_RATE_LIMIT_ENABLED`` est absent (défaut), ``enabled()`` est faux et
``make_decision`` renvoie toujours « autorisé » → comportement strictement identique à avant.

Conception
----------
* **Token-bucket par instance**, capacité = ``rpm × burst_ratio``, recharge = ``rpm/60`` jeton/s.
  Le burst encaisse une rafale ponctuelle (ex. un récap qui envoie 5 emails d'un coup) sans
  jamais débloquer un débit soutenu au-delà du RPM.
* **Horloge injectable** (paramètre ``now``) → tests déterministes SANS ``time.sleep``.
* **État JSON atomique sous ``HERMES_HOME``** : CLI / gateway / cron sont des PROCESS distincts ;
  un bucket en mémoire ne tiendrait pas à travers eux. On relit-modifie-réécrit (tmp + ``os.replace``,
  même idiome atomique que ``store.py``) à chaque décision. Une micro-course (off-by-one quand deux
  process consomment au même instant) est tolérée pour un rate-limiter — on ne sur-ingénie pas le
  verrouillage (pas de ``fcntl``, pas de ``SIGALRM`` : compatible Windows).

Défauts GÉNÉREUX (ne mordent JAMAIS l'usage normal) — cf. ``_DEFAULT_*`` plus bas.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Défauts (env-overridables). GÉNÉREUX par dessein.
# ---------------------------------------------------------------------------
# Un usage normal d'instance = quelques envois sortants par minute (un récap matinal envoie peut-être
# 3-5 emails d'un coup, puis plus rien pendant des minutes) et une dizaine de tours d'agent étalés sur
# plusieurs minutes (un tour « utile » dure des secondes à dizaines de secondes). 30 egress/min et
# 20 tours/min laissent donc TOUTE la marge à un fonctionnement sain : seul un emballement (boucle qui
# spamme, re-drive en tempête) franchit le seuil. Le burst ×1.5 absorbe en plus une rafale ponctuelle
# (jusqu'à 45 egress / 30 tours en un instant) avant de retomber au débit nominal.
_DEFAULT_EGRESS_RPM = 30.0
_DEFAULT_TURNS_RPM = 20.0
_DEFAULT_BURST_RATIO = 1.5

_STATE_DIRNAME = "jb_rate_limit"
_STATE_FILENAME = "buckets.json"


def enabled() -> bool:
    """Vrai uniquement quand l'opérateur a posé ``JB_RATE_LIMIT_ENABLED=1`` (défaut OFF).

    Tant que c'est faux, ``make_decision`` autorise systématiquement : zéro changement de
    comportement, le rate-limit est un pur ajout opt-in.
    """
    return os.getenv("JB_RATE_LIMIT_ENABLED", "").strip() == "1"


def _env_float(name: str, default: float) -> float:
    """Lit un float d'env, retombe sur ``default`` si absent / vide / illisible / non-positif."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def egress_rpm() -> float:
    return _env_float("JB_RATE_LIMIT_EGRESS_RPM", _DEFAULT_EGRESS_RPM)


def turns_rpm() -> float:
    return _env_float("JB_RATE_LIMIT_TURNS_RPM", _DEFAULT_TURNS_RPM)


def burst_ratio() -> float:
    # Le burst doit être ≥ 1 (sinon la capacité tomberait sous le RPM nominal et mordrait l'usage).
    ratio = _env_float("JB_RATE_LIMIT_BURST_RATIO", _DEFAULT_BURST_RATIO)
    return ratio if ratio >= 1.0 else 1.0


# ---------------------------------------------------------------------------
# Persistance (état cross-process, atomique)
# ---------------------------------------------------------------------------

def _home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


def _state_path() -> Path:
    d = _home() / _STATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d / _STATE_FILENAME


def _load_state() -> Dict[str, Any]:
    p = _state_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        # État corrompu (crash en plein write impossible grâce à os.replace, mais défensif) :
        # on repart d'un état vide plutôt que de faire échouer une décision.
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    p = _state_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)  # écriture atomique (même idiome que store.py) — compatible Windows


# ---------------------------------------------------------------------------
# Token bucket (pur, testable, horloge injectée)
# ---------------------------------------------------------------------------

class TokenBucket:
    """Seau à jetons : capacité ``capacity``, recharge ``refill_per_sec`` jeton/s.

    Stateless vis-à-vis de l'horloge : l'appelant passe ``now`` (epoch s). Sérialisable en dict
    (``tokens`` restants + ``updated`` = dernier instant connu) pour la persistance cross-process.
    """

    __slots__ = ("capacity", "refill_per_sec", "tokens", "updated")

    def __init__(self, capacity: float, refill_per_sec: float, tokens: Optional[float] = None, updated: float = 0.0):
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        # Démarrage plein : une instance fraîche encaisse son burst immédiatement.
        self.tokens = float(capacity) if tokens is None else float(tokens)
        self.updated = float(updated)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], capacity: float, refill_per_sec: float) -> "TokenBucket":
        """Reconstruit un bucket persisté ; ré-applique capacity/refill courants (env peut changer)."""
        if not isinstance(data, dict):
            return cls(capacity, refill_per_sec)
        tokens = data.get("tokens")
        updated = data.get("updated", 0.0)
        bucket = cls(
            capacity,
            refill_per_sec,
            tokens=tokens if isinstance(tokens, (int, float)) else None,
            updated=updated if isinstance(updated, (int, float)) else 0.0,
        )
        # Si la capacité a baissé (env resserré), on borne les jetons hérités.
        bucket.tokens = min(bucket.tokens, bucket.capacity)
        return bucket

    def to_dict(self) -> Dict[str, Any]:
        return {"tokens": self.tokens, "updated": self.updated}

    def _refill(self, now: float) -> None:
        if self.updated <= 0.0:
            self.updated = now
            return
        elapsed = now - self.updated
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            self.updated = now

    def allow(self, now: float, cost: float = 1.0) -> bool:
        """Recharge selon ``now`` puis tente de consommer ``cost``. Vrai = autorisé (jeton retiré)."""
        self._refill(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


# ---------------------------------------------------------------------------
# API publique : décision rate-limit pour une (famille, instance)
# ---------------------------------------------------------------------------

_FAMILIES: Dict[str, Callable[[], float]] = {
    "egress": egress_rpm,
    "turns": turns_rpm,
}


def _params(family: str) -> Tuple[float, float]:
    """(capacity, refill_per_sec) pour une famille, selon l'env courant."""
    rpm = _FAMILIES.get(family, egress_rpm)()
    ratio = burst_ratio()
    capacity = rpm * ratio
    refill_per_sec = rpm / 60.0
    return capacity, refill_per_sec


def make_decision(
    family: str,
    key: str,
    *,
    now: Optional[float] = None,
    clock: Callable[[], float] = time.time,
) -> bool:
    """Décide si l'action (``family`` ∈ {egress, turns}) est autorisée pour l'instance ``key``.

    * Renvoie ``True`` (autorisé) **toujours** quand ``enabled()`` est faux → zéro changement OFF.
    * Sinon consomme un jeton du bucket de l'instance et persiste l'état. ``True`` = autorisé,
      ``False`` = REFUS PROPRE (l'appelant doit court-circuiter, jamais différer).
    * ``now`` (ou ``clock``) injectable pour des tests déterministes sans ``sleep``.
    * Isolation PAR instance : le bucket de ``job-1`` est indépendant de celui de ``job-2``.

    Robustesse : toute erreur de persistance est avalée et retombe sur « autorisé » — un
    rate-limiter ne doit jamais casser un envoi par défaut (fail-open sur erreur interne ; le
    fail-CLOSED applicatif « rien ne part sans accord » reste assuré par jb_outbound en amont).
    """
    if not enabled():
        return True

    instant = clock() if now is None else now
    capacity, refill_per_sec = _params(family)
    state_key = f"{family}:{key or 'default'}"

    try:
        state = _load_state()
        bucket = TokenBucket.from_dict(state.get(state_key), capacity, refill_per_sec)
        allowed = bucket.allow(instant)
        state[state_key] = bucket.to_dict()
        _save_state(state)
        if not allowed:
            logger.warning("jb rate-limit: %s refusé pour l'instance %r (débit dépassé)", family, key)
        return allowed
    except Exception as exc:
        logger.debug("jb rate-limit: décision impossible (%s/%s), on autorise : %s", family, key, exc)
        return True
