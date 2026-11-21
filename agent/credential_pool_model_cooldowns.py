"""Model-scoped rate-limit cooldowns for pooled credentials.

Anthropic enforces its API rate limits (requests / tokens per minute) per
model, so a generic 429 for one Claude model says nothing about the same
credential's standing for its sibling models. Such a 429 is recorded as a
cooldown on the requested model only, beside the credential-wide status that
auth, billing and payment failures keep benching the whole credential with.

Each per-model cooldown also records its kind in ``model_cooldown_kinds``: an
``entitlement`` bench (Codex plan lacks the model, a year until reset) or a
``window`` cooldown (429 / usage-window 402, capped at
``MODEL_WINDOW_COOLDOWN_MAX_SECONDS``). Model-specific selection honours both;
model-less selection (``peek()``, ``select()`` without a model) skips
entitlement benches, so a plan gap never hides a healthy login. A model the
provider plainly does not serve records nothing (``provider_serves_model``).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

# A Codex ChatGPT-account model entitlement 400 is a plan property, not a window: bench the
# (credential, model) pair until an explicit ``hermes auth reset`` clears model_cooldowns (#71970).
MODEL_ENTITLEMENT_BENCH_SECONDS = 365 * 24 * 60 * 60
# Ceiling for a window cooldown (429 / usage-window 402): the longest real allowance is a weekly
# window, so a provider ``reset_at`` further out than this is a parse slip, not a window.
MODEL_WINDOW_COOLDOWN_MAX_SECONDS = 8 * 24 * 60 * 60
# A usage-window plan limit: this model is refused until the allowance resets; every other model
# (a Free or non-Large one) stays usable on the same credential.
FAILURE_REASON_USAGE_WINDOW = "usage_window"
# Kind of a per-model cooldown, stored beside it in ``model_cooldown_kinds`` so selection never has
# to guess from the remaining lifetime (which shrinks: a year-long bench looks like a window later).
COOLDOWN_KIND_ENTITLEMENT = "entitlement"
COOLDOWN_KIND_WINDOW = "window"


# Rows written before kinds were recorded: older writers benched an entitlement for exactly
# ``MODEL_ENTITLEMENT_BENCH_SECONDS`` and wrote windows from ``reset_at`` with no cap, so the only
# split they support is the one their own ``limit_state`` used: more than half the bench left.
LEGACY_ENTITLEMENT_MIN_REMAINING_SECONDS = MODEL_ENTITLEMENT_BENCH_SECONDS / 2


def _legacy_kind(until: float, now: float) -> str:
    return COOLDOWN_KIND_ENTITLEMENT if until - now > LEGACY_ENTITLEMENT_MIN_REMAINING_SECONDS else COOLDOWN_KIND_WINDOW


def migrate_model_cooldowns(
    provider: Optional[str], cooldowns: Any, kinds: Any, now: Optional[float] = None,
) -> tuple[Optional[dict[str, float]], Optional[dict[str, str]], bool]:
    """One-time upgrade of a stored row's per-model cooldowns: ``(cooldowns, kinds, changed)``.

    * A cooldown for a model *provider* plainly does not serve is dropped (a misroute never
      describes the login; this clears rows older writers pinned, e.g. ``claude-*`` on
      ``openai-codex``).
    * A kindless row gets its kind fixed now, once, from its remaining lifetime
      (``_legacy_kind``), and a legacy window is clamped to ``MODEL_WINDOW_COOLDOWN_MAX_SECONDS``
      like every new one. The kind is then stored, so a legacy entitlement bench never turns
      into a window as it ages. A legacy entitlement row with under half its year left is
      classed as a window and clamped: the worst case is one retry of that model after 8 days,
      which re-benches it with a recorded kind.
    * Expired rows are left as they are (nothing reads them).
    """
    if not isinstance(cooldowns, dict) or not cooldowns:
        return (cooldowns or None) if isinstance(cooldowns, dict) else None, kinds, False
    now = time.time() if now is None else now
    kinds_in = kinds if isinstance(kinds, dict) else {}
    out_cooldowns: dict[str, float] = {}
    out_kinds: dict[str, str] = {}
    changed = False
    for model, until in cooldowns.items():
        if not isinstance(until, (int, float)):
            changed = True
            continue
        if not provider_serves_model(provider, model):
            changed = True
            continue
        until = float(until)
        kind = kinds_in.get(model)
        if kind not in (COOLDOWN_KIND_ENTITLEMENT, COOLDOWN_KIND_WINDOW) and until > now:
            kind = _legacy_kind(until, now)
            if kind == COOLDOWN_KIND_WINDOW:
                until = min(until, now + MODEL_WINDOW_COOLDOWN_MAX_SECONDS)
            changed = True
        out_cooldowns[model] = until
        if kind in (COOLDOWN_KIND_ENTITLEMENT, COOLDOWN_KIND_WINDOW):
            out_kinds[model] = kind
    if set(kinds_in) - set(out_kinds):
        changed = True
    return out_cooldowns or None, out_kinds or None, changed


def cooldown_kind(entry: PooledCredential, model: str, until: float, now: float) -> str:
    """The recorded kind of *entry*'s cooldown for *model*.

    Rows loaded from disk carry a kind (``migrate_model_cooldowns`` fixes and stores it once).
    A kindless row can still arrive through a concurrent writer's disk merge before this
    process persists again; it gets the same legacy rule, never the window cap, so a fresh
    window from such a writer is still a window.
    """
    kind = (getattr(entry, "model_cooldown_kinds", None) or {}).get(model)
    if kind in (COOLDOWN_KIND_ENTITLEMENT, COOLDOWN_KIND_WINDOW):
        return kind
    return _legacy_kind(until, now)


def provider_serves_model(provider: Optional[str], model: Optional[str]) -> bool:
    """False only when *provider* is a single-vendor first-party route and *model* is plainly
    another vendor's id (``claude-*`` on ``openai-codex``). Such a rejection is a misroute, not a
    property of the credential, so it must not bench the credential for anything. Unknown or
    multi-vendor cases answer True (keep today's behaviour)."""
    if not provider or not model:
        return True
    try:
        from hermes_cli.model_normalize import detect_vendor
        from hermes_cli.models import _AGGREGATOR_PROVIDERS, _PROVIDER_MODELS, normalize_provider

        normalized = normalize_provider(provider)
        if normalized in _AGGREGATOR_PROVIDERS:
            return True
        vendor = detect_vendor(model)
        native = {detect_vendor(mid) for mid in _PROVIDER_MODELS.get(normalized, ())}
        if not vendor or len(native) != 1 or None in native:
            return True
        return native == {vendor}
    except Exception:  # pragma: no cover - catalog import trouble must never block recovery
        return True


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

    Callers that do not know the model stay conservative: any active window
    cooldown blocks them, so an unscoped route cannot reuse the credential.
    Entitlement benches are the exception: "this plan lacks model X" says
    nothing about a model-less caller (usage reader, status, diagnostics), and
    a year-long bench would otherwise hide a healthy login from them. The
    kind is read from ``model_cooldown_kinds`` (see ``cooldown_kind``), never
    from how long the bench has left, so this holds for the bench's whole life.
    A model-specific caller stays blocked by either kind until expiry.
    """
    cooldowns = entry.model_cooldowns or {}
    items = cooldowns.items() if not model else ((model, cooldowns.get(model)),)
    now = time.time()
    active = [
        float(until) for scoped_model, until in items
        if isinstance(until, (int, float)) and until > now
        and (model or cooldown_kind(entry, scoped_model, until, now) != COOLDOWN_KIND_ENTITLEMENT)
    ]
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


def merge_model_cooldown_kinds(*snapshots: Any) -> dict[str, str]:
    """Kinds that go with ``merge_model_cooldowns`` over the same snapshots.

    Each snapshot is a ``(model_cooldowns, model_cooldown_kinds)`` pair. A model's kind follows
    the snapshot whose reset won the merge (latest ``until``; a later snapshot wins a tie), so a
    kind never ends up attached to another writer's cooldown.
    """
    best: dict[str, tuple[float, Optional[str]]] = {}
    for cooldowns, kinds in snapshots:
        if not isinstance(cooldowns, dict):
            continue
        kinds = kinds if isinstance(kinds, dict) else {}
        for model, until in cooldowns.items():
            if not isinstance(until, (int, float)):
                continue
            if model not in best or float(until) >= best[model][0]:
                best[model] = (float(until), kinds.get(model))
    return {model: kind for model, (_until, kind) in best.items() if kind}


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

        now = time.time()
        if failure_reason == "model_entitlement":
            until = now + MODEL_ENTITLEMENT_BENCH_SECONDS
            kind = COOLDOWN_KIND_ENTITLEMENT
        else:
            until = _normalize_error_context(error_context).get("reset_at") or (
                now + _exhausted_ttl(429, sole_credential=self._is_sole_credential())
            )
            until = min(until, now + MODEL_WINDOW_COOLDOWN_MAX_SECONDS)
            kind = COOLDOWN_KIND_WINDOW
        failed_key = entry.runtime_api_key
        for scoped in list(self._entries):
            if scoped.id != entry.id and not (failed_key and scoped.runtime_api_key == failed_key):
                continue
            existing = (scoped.model_cooldowns, scoped.model_cooldown_kinds)
            cooldowns = merge_model_cooldowns(existing[0], {model: until})
            kinds = merge_model_cooldown_kinds(existing, ({model: until}, {model: kind}))
            self._adopt(scoped, persist=False, model_cooldowns=cooldowns, model_cooldown_kinds=kinds or None)
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
                soonest: Optional[tuple[float, PooledCredential]] = None
                for entry in usable:
                    wait = model_cooldown_until(entry, model)
                    if wait is None:
                        soonest = None
                        break
                    if soonest is None or wait < soonest[0]:
                        soonest = (wait, entry)
                if soonest and cooldown_kind(soonest[1], model, soonest[0], now) == COOLDOWN_KIND_WINDOW:
                    cooled[model] = soonest[0]
        return {"scope": "models", "models": cooled} if cooled else None
