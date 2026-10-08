"""Model-scoped rate-limit cooldowns for pooled credentials.

Anthropic enforces its API rate limits (requests / tokens per minute) per
model, so a generic 429 for one Claude model says nothing about the same
credential's standing for its sibling models. Such a 429 is recorded as a
cooldown on the requested model only, beside the credential-wide status that
auth, billing and payment failures keep benching the whole credential with.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

logger = logging.getLogger(__name__)

# A Codex ChatGPT-account model entitlement 400 is a plan property, not a window: bench the
# (credential, model) pair until a bounded cooldown expires or an explicit ``hermes auth reset`` clears model_cooldowns (#71970).
MODEL_ENTITLEMENT_BENCH_SECONDS = 2 * 24 * 60 * 60

# The pre-bound code wrote a full-year entitlement bench. No provider rate-limit window reaches
# further than a month, so a stored value past this floor is a stale sentinel to clamp on load.
LEGACY_MODEL_COOLDOWN_SENTINEL_S = 30 * 24 * 60 * 60


def bound_rehydrated_model_cooldowns(cooldowns: Any) -> Dict[str, float]:
    """Cap stale entitlement sentinels on load, leaving real provider windows intact.

    Only a value past ``LEGACY_MODEL_COOLDOWN_SENTINEL_S`` is a pre-bound bench; anything
    nearer is a genuine provider-stated window and survives untouched.
    """
    now = time.time()
    deadline = now + MODEL_ENTITLEMENT_BENCH_SECONDS
    return {
        model: min(float(until), deadline)
        if float(until) - now > LEGACY_MODEL_COOLDOWN_SENTINEL_S
        else float(until)
        for model, until in cooldowns.items()
        if isinstance(until, (int, float))
    }


def model_cooldown_until(entry: "PooledCredential", model: Optional[str]) -> Optional[float]:
    """Active cooldown blocking *entry* for *model*, or ``None``.

    Callers that do not know the model stay conservative: any active model
    cooldown blocks them, so an unscoped route cannot reuse the credential.
    """
    cooldowns = entry.model_cooldowns or {}
    values = cooldowns.values() if not model else (cooldowns.get(model),)
    now = time.time()
    active = [float(until) for until in values if isinstance(until, (int, float)) and until > now]
    return max(active) if active else None


def merge_model_cooldowns(*maps: Any) -> dict[str, float]:
    """Latest reset per model across snapshots — each writer only observed its own model."""
    merged: dict[str, float] = {}
    for cooldowns in maps:
        if not isinstance(cooldowns, dict):
            continue
        for model, until in cooldowns.items():
            if isinstance(until, (int, float)):
                merged[model] = max(float(until), merged.get(model, 0.0))
    return merged


# Per-model observation epochs let the persistence boundary order a cooldown against an
# explicit reset. The map lives in PooledCredential.extra so older rows round-trip unchanged.
MODEL_COOLDOWN_OBSERVED_AT_KEY = "model_cooldown_observed_at"


def merge_model_cooldown_observations(*maps: Any) -> Dict[str, float]:
    """Latest observation epoch per model across concurrent pool snapshots."""
    merged: Dict[str, float] = {}
    for observations in maps:
        if not isinstance(observations, dict):
            continue
        for model, observed_at in observations.items():
            if isinstance(observed_at, (int, float)):
                merged[model] = max(float(observed_at), merged.get(model, 0.0))
    return merged


def model_cooldowns_after_clear(
    cooldowns: Any, observations: Any, cleared_at: Optional[float],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Cooldown/observation maps that were recorded after *cleared_at*.

    A reset is a tombstone for every model observation that predates it. Legacy cooldown rows
    have no observation epoch, so they stay untouched until a reset exists; once an operator
    resets them, the reset wins instead of allowing a stale process to resurrect the old map.
    """
    merged_cooldowns = merge_model_cooldowns(cooldowns)
    merged_observations = merge_model_cooldown_observations(observations)
    clear_epoch = float(cleared_at or 0.0)
    if clear_epoch <= 0:
        return (
            merged_cooldowns,
            {model: at for model, at in merged_observations.items() if model in merged_cooldowns},
        )
    kept = {
        model: until
        for model, until in merged_cooldowns.items()
        if merged_observations.get(model, 0.0) > clear_epoch
    }
    return kept, {model: merged_observations[model] for model in kept if model in merged_observations}


class CredentialPoolModelCooldownMixin:
    def token_is_blocked(self, token: str, *, model: Optional[str] = None) -> bool:
        """Whether a pool cooldown blocks *token* for *model*.

        Closes the paths that hand out a native Anthropic token without
        selecting it from the pool (env / borrowed credentials). Tokens the
        pool does not know fail open: no row can attribute a cooldown to them.
        """
        with self._lock:
            return any(
                entry.runtime_api_key == token and model_cooldown_until(entry, model) is not None
                for entry in self._entries
            )

    def _resync_model_cooldown_clear(
        self, entry: "PooledCredential", disk_rows: Any = None,
    ) -> "PooledCredential":
        """Honor a reset written by another process for a healthy, model-benched entry.

        Credential-wide exhausted/dead rows already resync through CredentialPool's status path.
        A model-only cooldown leaves last_status healthy, so without this sibling path a long-lived
        gateway can keep refusing the model after another process successfully ran `hermes auth reset`.
        *disk_rows* maps entry id to persisted row when the caller already read the store.
        """
        if not entry.model_cooldowns:
            return entry
        try:
            from agent.credential_pool_cooldowns import _parse_absolute_timestamp
            if disk_rows is None:
                from hermes_cli.auth import read_credential_pool
                disk_rows = {
                    row.get("id"): row
                    for row in read_credential_pool(self.provider)
                    if isinstance(row, dict) and row.get("id")
                }
            cleared_at = _parse_absolute_timestamp(
                (disk_rows.get(entry.id) or {}).get("status_cleared_at"))
        except Exception:
            logger.debug("model cooldown resync read failed", exc_info=True)
            return entry
        if not cleared_at:
            return entry

        cooldowns, observations = model_cooldowns_after_clear(
            entry.model_cooldowns,
            entry.extra.get(MODEL_COOLDOWN_OBSERVED_AT_KEY),
            cleared_at,
        )
        if cooldowns == merge_model_cooldowns(entry.model_cooldowns):
            return entry

        updated_extra = dict(entry.extra)
        if observations:
            updated_extra[MODEL_COOLDOWN_OBSERVED_AT_KEY] = observations
        else:
            updated_extra.pop(MODEL_COOLDOWN_OBSERVED_AT_KEY, None)
        current_clear = _parse_absolute_timestamp(entry.status_cleared_at) or 0.0
        return self._adopt(
            entry,
            persist=False,
            model_cooldowns=cooldowns or None,
            status_cleared_at=max(current_clear, cleared_at),
            extra=updated_extra,
        )

    def _is_model_scoped_failure(
        self, status_code: Optional[int], model: Optional[str], failure_reason: Optional[str],
    ) -> bool:
        """Anthropic per-model 429s, and a Codex ChatGPT-account model entitlement 400: the
        account cannot use *model*, but the credential stays valid for every other model (#71970)."""
        from agent.credential_pool import FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED

        if not model:
            return False
        if failure_reason == "model_entitlement":
            return True
        return (
            self.provider == "anthropic" and status_code == 429
            and failure_reason not in (FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED)
        )

    def _cool_down_model(
        self, entry: "PooledCredential", model: str, error_context: Optional[dict[str, Any]],
        failure_reason: Optional[str] = None,
    ) -> None:
        """Record a cooldown for *model* on *entry* and every sibling sharing its key.

        Same TTL policy as a credential-wide 429 (provider ``reset_at`` wins, a
        sole credential keeps its short bench), except a ``model_entitlement``
        rejection, which expires after the bounded entitlement TTL or can be
        cleared immediately by the explicit reset path.
        Siblings matter because a ``model_config`` twin seeded from the same key
        would otherwise be re-selected for the very model that just failed.
        Caller holds the lock.
        """
        from agent.credential_pool import _normalize_error_context
        from agent.credential_pool_cooldowns import _exhausted_ttl

        observed_at = time.time()
        if failure_reason == "model_entitlement":
            until = observed_at + MODEL_ENTITLEMENT_BENCH_SECONDS
        else:
            until = _normalize_error_context(error_context).get("reset_at") or (
                observed_at + _exhausted_ttl(429, sole_credential=self._is_sole_credential())
            )
        failed_key = entry.runtime_api_key
        for scoped in list(self._entries):
            if scoped.id != entry.id and not (failed_key and scoped.runtime_api_key == failed_key):
                continue
            cooldowns = merge_model_cooldowns(scoped.model_cooldowns, {model: until})
            observations = merge_model_cooldown_observations(
                scoped.extra.get(MODEL_COOLDOWN_OBSERVED_AT_KEY),
                {model: observed_at},
            )
            updated_extra = dict(scoped.extra)
            updated_extra[MODEL_COOLDOWN_OBSERVED_AT_KEY] = observations
            self._adopt(
                scoped,
                persist=False,
                model_cooldowns=cooldowns,
                extra=updated_extra,
            )
        self._persist()

    def limit_state(self, models: Iterable[str]) -> Optional[dict[str, Any]]:
        """What a picker should say about this pool's rate limits, or ``None`` when nothing is limited.

        ``{"scope": "account", "resets_at": epoch}`` when every live entry is benched credential-wide
        (the whole login is out; another model won't help), else ``{"scope": "models", "models":
        {model: epoch}}`` for the given *models* no usable entry can serve until a model cooldown
        ends. Every active model cooldown is reported: with the bench bounded to
        ``MODEL_ENTITLEMENT_BENCH_SECONDS`` a long window is a real wait, not a plan
        property that never resets. Read-only: never clears or persists a cooldown.
        """
        from agent.credential_pool import STATUS_DEAD
        from agent.credential_pool_cooldowns import _exhausted_until

        now = time.time()
        with self._lock:
            live = [entry for entry in self._entries if entry.last_status != STATUS_DEAD]
            sole = len(live) <= 1
            benched = {entry.id: until for entry in live
                       if (until := _exhausted_until(entry, sole_credential=sole)) and until > now}
            if live and len(benched) == len(live):
                return {"scope": "account", "resets_at": min(benched.values())}
            usable = [entry for entry in live if entry.id not in benched]
            cooled: dict[str, float] = {}
            for model in models:
                waits = [model_cooldown_until(entry, model) for entry in usable]
                if waits and all(waits):
                    cooled[model] = min(waits)
        return {"scope": "models", "models": cooled} if cooled else None
