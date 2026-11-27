"""Model-scoped rate-limit cooldowns for pooled credentials.

Anthropic enforces its API rate limits (requests / tokens per minute) per
model, so a generic 429 for one Claude model says nothing about the same
credential's standing for its sibling models. Such a 429 is recorded as a
cooldown on the requested model only, beside the credential-wide status that
auth, billing and payment failures keep benching the whole credential with.

A Claude subscription 429 that names a spent account-wide plan window is the
exception: every model on that account is refused until the window resets, so
the credential is benched as a whole (#149112).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

from agent.credential_pool_timestamps import _parse_absolute_timestamp

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

# Claude subscription plan windows. A plan 429 carries ``anthropic-ratelimit-unified-status: rejected``
# and names the refusing window in ``anthropic-ratelimit-unified-representative-claim``. ``five_hour``
# and ``seven_day`` cap the whole account; a family claim (``seven_day_opus``) caps one model family.
ACCOUNT_WIDE_PLAN_WINDOWS = frozenset({"five_hour", "seven_day"})
# Reason codes and messages that say the account's usage allowance is spent.
_USAGE_LIMIT_REASON_TOKENS = ("usage_limit_reached", "gousagelimit")
_USAGE_LIMIT_MESSAGE_TOKENS = ("usage limit reached", "usage limit has been reached")


def record_plan_window(context: dict[str, Any], headers: Any) -> None:
    """Record the Claude plan window a rejected response names as ``subscription_window``, and the
    window's ``anthropic-ratelimit-unified-reset`` as ``reset_at`` when nothing earlier set one. For an
    account-wide window a future unified reset is authoritative and replaces any earlier value
    (``retry-after`` times the next request, not the window)."""
    if str(headers.get("anthropic-ratelimit-unified-status") or "").strip().lower() != "rejected":
        return
    claim = str(headers.get("anthropic-ratelimit-unified-representative-claim") or "").strip().lower()
    if claim:
        context["subscription_window"] = claim
    reset_at = _parse_absolute_timestamp(headers.get("anthropic-ratelimit-unified-reset"))
    if reset_at is not None and (
            "reset_at" not in context or (claim in ACCOUNT_WIDE_PLAN_WINDOWS and reset_at > time.time())):
        context["reset_at"] = reset_at


def account_window_spent(error_context: Optional[dict[str, Any]]) -> bool:
    """The 429 names a spent account-wide plan window: no model on the credential is served."""
    return str((error_context or {}).get("subscription_window") or "") in ACCOUNT_WIDE_PLAN_WINDOWS


def usage_allowance_spent(error_context: Optional[dict[str, Any]]) -> bool:
    """The error says an allowance is spent, so the caller rotates on the first 429 instead of retrying
    the same credential. Any rejected Claude plan window counts; ``account_window_spent`` decides its
    scope (whole credential or one model)."""
    reason = str((error_context or {}).get("reason") or "").lower()
    message = str((error_context or {}).get("message") or "").lower()
    return bool((error_context or {}).get("subscription_window")) or any(
        t in reason for t in _USAGE_LIMIT_REASON_TOKENS) or any(t in message for t in _USAGE_LIMIT_MESSAGE_TOKENS)


# A Codex ChatGPT-account model entitlement 400 is a plan property, not a window: bench the
# (credential, model) pair until an explicit ``hermes auth reset`` clears model_cooldowns (#71970).
MODEL_ENTITLEMENT_BENCH_SECONDS = 365 * 24 * 60 * 60
# A usage-window plan limit: this model is refused until the allowance resets; every other model
# (a Free or non-Large one) stays usable on the same credential.
FAILURE_REASON_USAGE_WINDOW = "usage_window"


def pool_failure_reason(reason: Any, error_context: Optional[dict[str, Any]], billing_unverified: bool) -> str:
    """The pool's failure reason for a rotation (*reason* is a ``FailoverReason``). A usage-window 402
    benches only the refused model; an ambiguous billing body is sized as transient, not a 1-hour bench."""
    from agent.error_classifier import FailoverReason
    if reason == FailoverReason.billing and (error_context or {}).get("usage_window"):
        return FAILURE_REASON_USAGE_WINDOW
    if reason == FailoverReason.billing and billing_unverified:
        from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
        return FAILURE_REASON_BILLING_UNVERIFIED
    return reason.value


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
    """Latest reset per model across snapshots — each writer only observed its own model."""
    merged: dict[str, float] = {}
    for cooldowns in maps:
        if not isinstance(cooldowns, dict):
            continue
        for model, until in cooldowns.items():
            if isinstance(until, (int, float)):
                merged[model] = max(float(until), merged.get(model, 0.0))
    return merged


class CredentialPoolModelCooldownMixin:
    def token_is_blocked(self, token: str, *, model: Optional[str] = None) -> bool:
        """Whether a pool cooldown blocks *token* for *model*: a cooldown on that model, or a
        credential-wide 429 bench (a spent plan window refuses every model).

        Closes the paths that hand out a native Anthropic token without
        selecting it from the pool (env / borrowed credentials). Tokens the
        pool does not know fail open: no row can attribute a cooldown to them.
        """
        from agent.credential_pool import _exhausted_until

        now = time.time()
        with self._lock:
            sole = self._is_sole_credential()
            return any(
                entry.runtime_api_key == token and (
                    model_cooldown_until(entry, model) is not None
                    or (entry.last_error_code == 429 and (_exhausted_until(entry, sole_credential=sole) or 0) > now)
                )
                for entry in self._entries
            )

    def token_account_reset(self, token: str) -> Optional[float]:
        """When the latest credential-wide 429 bench on *token* ends (a spent plan window refuses
        every model until then), or ``None`` when none is active."""
        from agent.credential_pool import _exhausted_until

        now = time.time()
        with self._lock:
            sole = self._is_sole_credential()
            resets = [until for entry in self._entries
                      if entry.runtime_api_key == token and entry.last_error_code == 429
                      and (until := _exhausted_until(entry, sole_credential=sole) or 0) > now]
        return max(resets) if resets else None

    def _is_model_scoped_failure(
        self, status_code: Optional[int], model: Optional[str], failure_reason: Optional[str],
        error_context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Anthropic per-model 429s, and a Codex ChatGPT-account model entitlement 400: the
        account cannot use *model*, but the credential stays valid for every other model (#71970).
        A spent account-wide plan window is not: it refuses every model (#149112)."""
        from agent.credential_pool import FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED

        if not model or account_window_spent(error_context):
            return False
        if failure_reason in ("model_entitlement", FAILURE_REASON_USAGE_WINDOW):
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
        rejection, which stays benched until the explicit reset path clears it.
        Siblings matter because a ``model_config`` twin seeded from the same key
        would otherwise be re-selected for the very model that just failed.
        Caller holds the lock.
        """
        from agent.credential_pool import _exhausted_ttl, _normalize_error_context

        if failure_reason == "model_entitlement":
            until = time.time() + MODEL_ENTITLEMENT_BENCH_SECONDS
        else:
            until = _normalize_error_context(error_context).get("reset_at") or (
                time.time() + _exhausted_ttl(429, sole_credential=self._is_sole_credential())
            )
        failed_key = entry.runtime_api_key
        for scoped in list(self._entries):
            if scoped.id != entry.id and not (failed_key and scoped.runtime_api_key == failed_key):
                continue
            cooldowns = merge_model_cooldowns(scoped.model_cooldowns, {model: until})
            self._adopt(scoped, persist=False, model_cooldowns=cooldowns)
        self._persist()

    def limit_state(self, models: Iterable[str]) -> Optional[dict[str, Any]]:
        """What a picker should say about this pool's rate limits, or ``None`` when nothing is limited.

        ``{"scope": "account", "resets_at": epoch}`` when every live entry is benched credential-wide
        (the whole login is out; another model won't help), else ``{"scope": "models", "models":
        {model: epoch}}`` for the given *models* no usable entry can serve until a model cooldown
        ends. Entitlement benches (a year: the plan lacks the model) aren't windows that reset, so
        they're left out. Read-only: never clears or persists a cooldown.
        """
        from agent.credential_pool import STATUS_DEAD, _exhausted_until

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
                if waits and all(waits) and min(waits) - now < MODEL_ENTITLEMENT_BENCH_SECONDS / 2:
                    cooled[model] = min(waits)
        return {"scope": "models", "models": cooled} if cooled else None
