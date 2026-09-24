"""Model-scoped rate-limit cooldowns for pooled credentials.

Anthropic enforces its API rate limits (requests / tokens per minute) per
model, so a generic 429 for one Claude model says nothing about the same
credential's standing for its sibling models. Such a 429 is recorded as a
cooldown on the requested model only, beside the credential-wide status that
auth, billing and payment failures keep benching the whole credential with.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

# A Codex ChatGPT-account model entitlement 400 is a plan property, not a window: bench the
# (credential, model) pair until an explicit ``hermes auth reset`` clears model_cooldowns (#71970).
MODEL_ENTITLEMENT_BENCH_SECONDS = 365 * 24 * 60 * 60


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


def merge_model_cooldowns(*maps: Any) -> Dict[str, float]:
    """Latest reset per model across snapshots — each writer only observed its own model."""
    merged: Dict[str, float] = {}
    for cooldowns in maps:
        if not isinstance(cooldowns, dict):
            continue
        for model, until in cooldowns.items():
            if isinstance(until, (int, float)):
                merged[model] = max(float(until), merged.get(model, 0.0))
    return merged


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
        self, entry: "PooledCredential", model: str, error_context: Optional[Dict[str, Any]],
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
