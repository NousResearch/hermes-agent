"""Model-scoped rate-limit cooldowns for pooled credentials.

Anthropic enforces its API rate limits (requests / tokens per minute) per
model, so a generic 429 for one Claude model says nothing about the same
credential's standing for its sibling models. Such a 429 is recorded as a
cooldown on the requested model only, beside the credential-wide status that
auth, billing and payment failures keep benching the whole credential with.
"""
from __future__ import annotations

import logging
import math
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


def bound_rehydrated_model_cooldowns(cooldowns: Any) -> dict[str, float]:
    """Cap stale entitlement sentinels on load, leaving real provider windows intact.

    Only a value past ``LEGACY_MODEL_COOLDOWN_SENTINEL_S`` is a pre-bound bench; anything
    nearer is a genuine provider-stated window and survives untouched.
    """
    if not isinstance(cooldowns, dict):
        return {}
    now = time.time()
    deadline = now + MODEL_ENTITLEMENT_BENCH_SECONDS
    return {
        model: min(float(until), deadline)
        if float(until) - now > LEGACY_MODEL_COOLDOWN_SENTINEL_S
        else float(until)
        for model, until in cooldowns.items()
        if isinstance(until, (int, float))
    }


def has_active_model_cooldown(entry: PooledCredential) -> bool:
    """Whether *entry* carries a model cooldown that has not yet elapsed."""
    return model_cooldown_until(entry, None) is not None


def prune_expired_model_cooldowns(entry: PooledCredential) -> tuple[dict[str, float], dict[str, float]]:
    """Cooldown/observation maps of *entry* without elapsed cooldowns."""
    now = time.time()
    kept = {m: u for m, u in (entry.model_cooldowns or {}).items()
            if isinstance(u, (int, float)) and u > now}
    observed = entry.extra.get(MODEL_COOLDOWN_OBSERVED_AT_KEY)
    observed = observed if isinstance(observed, dict) else {}
    return kept, {m: at for m, at in observed.items() if m in kept}


def model_cooldown_until(entry: PooledCredential, model: Optional[str]) -> Optional[float]:
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
    """Latest reset per model across snapshots — each writer only observed its own model.

    Elapsed cooldowns are dropped: they no longer bind, and keeping them would force a store
    read on every selection."""
    merged: dict[str, float] = {}
    now = time.time()
    for cooldowns in maps:
        if not isinstance(cooldowns, dict):
            continue
        for model, until in cooldowns.items():
            if isinstance(until, (int, float)) and until > now:
                merged[model] = max(float(until), merged.get(model, 0.0))
    return merged


# Per-model observation epochs let the persistence boundary order a cooldown against an
# explicit reset. The map lives in PooledCredential.extra so older rows round-trip unchanged.
MODEL_COOLDOWN_OBSERVED_AT_KEY = "model_cooldown_observed_at"


# Models rejected by a plan-entitlement 400, mapped to the cooldown epoch written with them. Lets a
# picker say "not entitled" instead of showing a countdown for a model the plan never includes.
MODEL_ENTITLEMENT_KEY = "model_entitlement_until"


def entitlement_rejected_models(entry: PooledCredential) -> set[str]:
    """Models whose active cooldown on *entry* is an entitlement bench, not a rate-limit window."""
    marks = entry.extra.get(MODEL_ENTITLEMENT_KEY)
    cooldowns = entry.model_cooldowns or {}
    if not isinstance(marks, dict):
        return set()
    return {model for model, until in marks.items() if cooldowns.get(model) == until}


def merge_model_cooldown_observations(*maps: Any) -> dict[str, float]:
    """Latest observation epoch per model across concurrent pool snapshots."""
    merged: dict[str, float] = {}
    for observations in maps:
        if not isinstance(observations, dict):
            continue
        for model, observed_at in observations.items():
            if isinstance(observed_at, (int, float)):
                merged[model] = max(float(observed_at), merged.get(model, 0.0))
    return merged


def model_cooldowns_after_clear(
    cooldowns: Any, observations: Any, cleared_at: Optional[float],
) -> tuple[dict[str, float], dict[str, float]]:
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
        self, entry: PooledCredential, disk_rows: Any = None,
    ) -> PooledCredential:
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
        self, entry: PooledCredential, model: str, error_context: Optional[dict[str, Any]],
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
            if failure_reason == "model_entitlement":
                updated_extra[MODEL_ENTITLEMENT_KEY] = {
                    **(updated_extra.get(MODEL_ENTITLEMENT_KEY) or {}), model: until}
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
        property that never resets. Models benched only by an entitlement rejection are also listed
        under ``not_entitled`` so a picker can say so rather than count down. Read-only: never
        clears or persists a cooldown.
        """
        from agent.credential_pool import STATUS_DEAD
        from agent.credential_pool_cooldowns import _exhausted_until, _parse_absolute_timestamp

        now = time.time()
        with self._lock:
            live = [entry for entry in self._entries if entry.last_status != STATUS_DEAD]
            sole = len(live) <= 1
            # A benched row is shown at the provider's stated reset when it has one: a lone credential
            # may re-probe sooner, but the limit the user is told about is the provider's window.
            benched = {
                entry.id: _parse_absolute_timestamp(entry.last_error_reset_at) or until
                for entry in live
                if (until := _exhausted_until(entry, sole_credential=sole)) and until > now}
            if live and len(benched) == len(live):
                return {"scope": "account", "resets_at": min(benched.values())}
            usable = [entry for entry in live if entry.id not in benched]
            cooled: dict[str, float] = {}
            not_entitled: list[str] = []
            for model in models:
                waits = [model_cooldown_until(entry, model) for entry in usable]
                if waits and all(waits):
                    cooled[model] = min(waits)
                    if all(model in entitlement_rejected_models(entry) for entry in usable):
                        not_entitled.append(model)
        if not cooled:
            return None
        state: dict[str, Any] = {"scope": "models", "models": cooled}
        if not_entitled:
            state["not_entitled"] = not_entitled
        return state

    def clear_billing_benches(self, *, sources: Optional[Iterable[str]] = None) -> int:
        """Clear ``STATUS_EXHAUSTED`` benches attributed to billing; return how many.

        *sources* limits the clear to entries seeded from those sources: a caller that confirmed
        one account's balance must not lift the bench on entries it cannot attribute to that
        account. ``None`` clears every billing bench.

        A credit exhaustion — unlike a rate-limit window — can be cured mid-session by a
        portal top-up at any moment (#126818): the full bench would keep a live session
        pinned to its (paid) fallback long after the primary is funded again. Only
        billing-attributed benches clear here; rate-limit windows, auth failures and
        unverified-billing benches keep their cooldowns."""
        from agent.credential_pool import FAILURE_REASON_BILLING, STATUS_EXHAUSTED, _MARK_OK

        allowed_sources = None if sources is None else set(sources)
        with self._lock:
            cleared_ids = []
            for entry in self._entries:
                if entry.last_status != STATUS_EXHAUSTED or entry.failure_reason != FAILURE_REASON_BILLING:
                    continue
                if allowed_sources is not None and entry.source not in allowed_sources:
                    continue
                cleared_at = time.time()
                extra = {k: v for k, v in entry.extra.items() if k != "failure_reason"}
                # ``status_cleared_at`` doubles as the tombstone for every model cooldown observed
                # before it. A billing top-up lifts only the credential-wide bench, so re-stamp the
                # model cooldowns that are still active as observed just after the clear.
                kept, _ = prune_expired_model_cooldowns(entry)
                if kept:
                    extra[MODEL_COOLDOWN_OBSERVED_AT_KEY] = {
                        model: math.nextafter(cleared_at, math.inf) for model in kept}
                self._adopt(
                    entry, persist=False, **_MARK_OK, status_cleared_at=cleared_at,
                    model_cooldowns=kept or None, extra=extra,
                )
                cleared_ids.append(entry.id)
            if cleared_ids:
                # Declare the reset so the disk-recency merge cannot restore the bench.
                self._persist(status_cleared_ids=cleared_ids)
            return len(cleared_ids)
