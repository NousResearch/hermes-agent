"""Usage-window plans: what a limit blocks, and what to offer instead.

One rule decides everywhere, so the picker's "used up" note, the limit card's suggestion and the switch
button can never disagree:

* a free model is never blocked;
* a full gauge with scope ``all`` (day / week / month / any new one) blocks every paid model;
* a full ``top_tier`` gauge blocks the models the gateway marks top tier.

Hermes never switches model on its own. When a request is refused, :func:`limit_hit` describes the
limit and names up to two models from the plan's recommended list (in order) that still
work. Each surface renders that as buttons where it already has them, otherwise as a sentence.

Everything here reads cached data only (credits headers, the portal account snapshot, the cached
catalog) and never raises: a failure means "no suggestion".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

PROVIDER = "nous"
# Leave room for the reply: a suggestion must fit the conversation with this many tokens to spare.
_CONTEXT_HEADROOM_TOKENS = 8_000
_MAX_SUGGESTIONS = 2

NOTE_FREE = "free"
NOTE_INCLUDED = "included"

# Past a limit, the credits step (:func:`credits_action`).
USE_CREDITS = "use_credits"                                # Keep going off, balance: turn it on
ADD_CREDITS = "add_credits"                                # Keep going on/unknown, no balance: top up
ADD_CREDITS_AND_KEEP_GOING = "add_credits_and_keep_going"  # Keep going off, no balance: both


@dataclass(frozen=True)
class ModelSuggestion:
    model: str      # routed id, e.g. "example/model-a"
    label: str      # display name ("Model A"), the id when the catalog has none
    note: str       # NOTE_FREE | NOTE_INCLUDED
    provider: str = PROVIDER

    def to_payload(self) -> dict[str, str]:
        return {"model": self.model, "label": self.label, "note": self.note, "provider": self.provider}


@dataclass(frozen=True)
class CreditsOption:
    """The second way on, past a limit: spend top-up credits (the action comes from :func:`credits_action`).
    Interactive surfaces render it as a button; messaging points at the Portal (a chat button could spend
    the owner's money for anyone in the room)."""
    action: str     # USE_CREDITS | ADD_CREDITS | ADD_CREDITS_AND_KEEP_GOING

    def to_payload(self) -> dict[str, str]:
        return {"action": self.action}


@dataclass(frozen=True)
class LimitHit:
    gauge_id: str                   # day | week | month | top_tier | <server id>
    scope: str                      # all | top_tier
    limit_name: str                 # "daily" / "weekly" / "monthly large-model" / the server label
    resets_at_ms: Optional[int]
    message: str                    # the limit sentence (the server's display_message wins)
    suggestions: tuple[ModelSuggestion, ...] = ()
    credits: Optional[CreditsOption] = None
    current_model: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "gauge_id": self.gauge_id, "scope": self.scope, "limit_name": self.limit_name,
            "resets_at_ms": self.resets_at_ms, "message": self.message,
            "suggestions": [s.to_payload() for s in self.suggestions],
            "credits": self.credits.to_payload() if self.credits else None,
            "current_model": self.current_model,
        }


# ── catalog reads ────────────────────────────────────────────────────────────────────────────────

def cached_pricing(base_url: str = "") -> dict:
    """The calling profile's own cached Nous catalog (never fetches); ``base_url`` is not needed for it. Only
    the profile-keyed read: a process serving several profiles holds their catalogs side by side, and a
    usage level (Large / Free) is per plan, so another profile's catalog must never answer."""
    try:
        from hermes_cli.models_pricing import get_pricing_for_provider
        return get_pricing_for_provider(PROVIDER, cached_only=True) or {}
    except Exception:
        logger.debug("Cached pricing unavailable", exc_info=True)
        return {}


def _billed_elsewhere(model: str, pricing: Optional[dict]) -> bool:
    try:
        from hermes_cli.models import _is_subscription_billed
        return _is_subscription_billed((pricing or {}).get(model))
    except Exception:
        logger.debug("Subscription billing lookup failed", exc_info=True)
        return False


def is_free(model: str, pricing: Optional[dict]) -> bool:
    """$0 to the plan: a ``:free`` id or a row the catalog lists as free."""
    if not model:
        return False
    if model.endswith(":free"):
        return True
    try:
        from hermes_cli.models import _is_model_free
        return bool(pricing) and _is_model_free(model, pricing)
    except Exception:
        logger.debug("Free-model lookup failed", exc_info=True)
        return False


def is_top_tier(model: str, pricing: Optional[dict]) -> bool:
    """The gateway marks the model top tier (``usage_level: top_tier``, or an older gateway's ``top_tier:
    true``), read by the same rule as the picker tags. Unmarked or no catalog: not top tier."""
    if not model:
        return False
    try:
        from hermes_cli.model_picker_usage import usage_levels
        return usage_levels([model], pricing).get(model) == "top_tier"
    except Exception:
        logger.debug("Usage level lookup failed", exc_info=True)
        return False


def model_label(model: str, pricing: Optional[dict]) -> str:
    entry = (pricing or {}).get(model)
    name = entry.get("name") if isinstance(entry, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else model


# ── the one rule ─────────────────────────────────────────────────────────────────────────────────

def blocking_gauge(model: str, gauges: Iterable[Any], *, free: bool, top_tier: bool) -> Optional[Any]:
    """The full gauge that stops *model*, or None while it still has included room. Free models are
    never blocked. Of several full gauges, the one that frees up last (that is when the model works)."""
    if free:
        return None
    from agent.usage_gauges import SCOPE_ALL, SCOPE_TOP_TIER
    full = [g for g in gauges
            if getattr(g, "full", False) and (g.scope == SCOPE_ALL or (g.scope == SCOPE_TOP_TIER and top_tier))]
    return max(full, key=lambda g: g.resets_at_ms or 0) if full else None


def current_model_is_top_tier(agent: Any) -> Optional[bool]:
    """*agent*'s current model is top tier, by its own profile's cached catalog (see :func:`is_top_tier`), or
    None when the catalog has no row for it (cold, or its 300 s cache lapsed): the tier is unknown."""
    model = str(getattr(agent, "model", "") or "")
    return _top_tier_mark(model, cached_pricing(str(getattr(agent, "base_url", "") or "")))


def _top_tier_mark(model: str, pricing: Optional[dict]) -> Optional[bool]:
    return is_top_tier(model, pricing) if model in (pricing or {}) or is_free(model, pricing) else None


def is_blocked(model: str, gauges: Iterable[Any], pricing: Optional[dict]) -> bool:
    return blocking_gauge(model, gauges, free=is_free(model, pricing),
                          top_tier=is_top_tier(model, pricing)) is not None


# ── suggestions ──────────────────────────────────────────────────────────────────────────────────

def _plan_usage() -> Any:
    # Plan shape only (featured models), so the last account snapshot at any age answers: the 60s
    # cache has long lapsed by the time a window runs out mid-session.
    try:
        from hermes_cli.model_picker_usage import nous_picker_usage
        return nous_picker_usage(any_age=True)
    except Exception:
        logger.debug("Plan usage unavailable", exc_info=True)
        return None


def _portal_free_models() -> list[str]:
    """The Portal's free picks, from what this profile already holds for its own Portal (in-process,
    else its last copy on disk). Never fetches: this runs mid-turn."""
    try:
        from hermes_cli import models as m
        base = m._resolve_nous_portal_url()
        payload = (m._nous_recommended_cache.get((m._pricing_profile_key(), base)) or (None,))[0]
        if not payload:
            disk = m._read_nous_recommended_disk(base)
            payload = disk[0] if disk else None
        entries = payload.get("freeRecommendedModels") if isinstance(payload, dict) else None
        return [n for e in (entries or []) if (n := m._extract_model_name(e))]
    except Exception:
        logger.debug("Portal free models unavailable", exc_info=True)
        return []


def _fits_context(model: str, needed_tokens: Optional[int], base_url: str) -> bool:
    if not needed_tokens or needed_tokens <= 0:
        return True
    try:
        from agent.model_metadata import get_model_context_length
        ctx = get_model_context_length(model, base_url=base_url, provider=PROVIDER)
        return not ctx or ctx >= needed_tokens + _CONTEXT_HEADROOM_TOKENS
    except Exception:
        logger.debug("Context length lookup failed", exc_info=True)
        return True


def suggest_models(
    current_model: str, gauges: Iterable[Any], *, pricing: Optional[dict] = None, featured: Iterable[str] = (),
    needed_tokens: Optional[int] = None, base_url: str = "", limit: int = _MAX_SUGGESTIONS,
) -> tuple[ModelSuggestion, ...]:
    """The plan's recommended models (in order) that the full gauges do not block, then the Portal's free picks.
    Never the current model, only a model whose context window holds the conversation."""
    gauges = tuple(gauges)
    pool = [m for m in featured if isinstance(m, str) and m]
    pool += [m for m in _portal_free_models() if m not in pool]
    out: list[ModelSuggestion] = []
    for model in pool:
        if model == current_model or _billed_elsewhere(model, pricing):
            continue
        if is_blocked(model, gauges, pricing) or not _fits_context(model, needed_tokens, base_url):
            continue
        out.append(ModelSuggestion(model=model, label=model_label(model, pricing),
                                   note=NOTE_FREE if is_free(model, pricing) else NOTE_INCLUDED))
        if len(out) >= limit:
            break
    return tuple(out)


def credits_action(keep_going_on_credits: Optional[bool], purchased_micros: Optional[int]) -> str:
    """The credits step past a limit; every surface's copy reads this one rule. With Keep going on credits
    off, the gateway refuses whatever the balance, so a balance alone is not enough and neither is the
    setting alone: off with no balance needs both steps. An unknown balance counts as none."""
    has_balance = int(purchased_micros or 0) > 0
    if keep_going_on_credits is False:
        return USE_CREDITS if has_balance else ADD_CREDITS_AND_KEEP_GOING
    return ADD_CREDITS


def credits_option(state: Any) -> Optional[CreditsOption]:
    """How credits could carry on past the limit (see :func:`credits_action`). None while paused (a card
    update restores included usage; the paused copy says so)."""
    if state is None or getattr(state, "paused_reason", None):
        return None
    return CreditsOption(credits_action(getattr(state, "keep_going_on_credits", None),
                                        getattr(state, "purchased_micros", 0)))


def limit_hit(agent: Any, *, message: str, needed_tokens: Optional[int] = None) -> Optional[LimitHit]:
    """After a usage-window 402: which limit stopped *agent*'s current model, when it resets, and what to
    offer. None when the state names no full gauge for this model (a paused plan, a stale snapshot)."""
    try:
        state = getattr(agent, "_credits_state", None)
        if state is None or not getattr(state, "is_usage_window", False):
            return None
        model = str(getattr(agent, "model", "") or "")
        base_url = str(getattr(agent, "base_url", "") or "")
        pricing = cached_pricing(base_url)
        gauges = tuple(getattr(state, "gauges", ()) or ())
        # A model the cached catalog has no row for may be top tier: count its full top-tier gauge too, so
        # the card never names an earlier reset than the one that frees it.
        g = blocking_gauge(model, gauges, free=is_free(model, pricing), top_tier=_top_tier_mark(model, pricing) is not False)
        if g is None:
            # The 402 says this model was refused; without a top-tier mark in the cache a full top-tier
            # gauge is still the only explanation.
            from agent.usage_gauges import top_tier_gauge
            top = top_tier_gauge(gauges)
            g = top if top is not None and top.full else None
        if g is None:
            return None
        from agent.usage_gauges import gauge_limit_name
        usage = _plan_usage()
        featured = tuple(getattr(usage, "featured_model_ids", ()) or ())
        return LimitHit(
            gauge_id=g.id, scope=g.scope, limit_name=gauge_limit_name(g), resets_at_ms=g.resets_at_ms,
            message=message, current_model=model, credits=credits_option(state),
            suggestions=suggest_models(model, gauges, pricing=pricing, featured=featured,
                                       needed_tokens=needed_tokens, base_url=base_url),
        )
    except Exception:
        logger.debug("plan limits: limit_hit failed", exc_info=True)
        return None


# ── copy ─────────────────────────────────────────────────────────────────────────────────────────

def portal_wording(text: str) -> str:
    """A refusal line for a surface without ``/topup`` (messaging): point its credits steps at the Portal."""
    return (text.replace("turn on Keep going in /topup", "turn on Keep going in the Portal")
            .replace("(turn on in /topup)", "(turn on in the Portal)")
            .replace("/topup to add credits", "add credits in the Portal")
            .replace("add credits in /topup", "add credits in the Portal"))


def chat_refusal_wording(text: str, resets_at: Any = None) -> str:
    """A refusal sent to a chat: credits steps point at the Portal (:func:`portal_wording`), and the reset
    reads as a duration ("resets in 3h"), since the gateway's clock is not the reader's time zone."""
    text = portal_wording(text)
    if isinstance(resets_at, (int, float)) and not isinstance(resets_at, bool) and resets_at > 0:
        try:
            from agent.credits_tracker import format_window_reset
            from hermes_cli.model_picker_usage import compact_duration
            ms = round(resets_at * 1000)  # the epoch seconds came from these ms: round, never truncate
            if left := compact_duration(ms):
                text = text.replace(f"resets {format_window_reset(ms)}", f"resets in {left}")
        except Exception:
            logger.debug("chat reset wording failed", exc_info=True)
    return text


def suggestion_phrase(s: ModelSuggestion) -> str:
    """``Model A (free)``."""
    return f"{s.label} ({s.note})"


def next_steps_text(hit: LimitHit, *, model_command: str = "/model", topup_command: Optional[str] = "/topup",
                    portal: bool = False) -> str:
    """One line on how to keep going, for a surface without buttons. ``model_command`` is the platform's
    own spelling (e.g. ``!model``). ``portal``: point at the Portal for credits (messaging),
    instead of a command that would spend from the chat."""
    parts: list[str] = []
    if hit.suggestions:
        parts.append(f"Switch to {suggestion_phrase(hit.suggestions[0])} with {model_command}")
    else:
        parts.append(f"Pick another model with {model_command}")
    if hit.credits is not None:
        where = "the Portal" if portal or not topup_command else topup_command
        if hit.credits.action == USE_CREDITS:
            parts.append(f"turn on Keep going in {where} to use credits")
        elif hit.credits.action == ADD_CREDITS_AND_KEEP_GOING:
            parts.append(f"add credits and turn on Keep going in {where}")
        else:
            parts.append(f"add credits {'in' if where == 'the Portal' else 'with'} {where}")
    return ", or ".join(parts) + "."


def limit_next_steps_from_payload(limit: dict, *, model_command: str = "/model", topup_command: Optional[str] = "/topup",
                                  portal: bool = False, markup: bool = False) -> str:
    """:func:`next_steps_text` for a :meth:`LimitHit.to_payload` dict (what the result carries). ``markup``
    bolds the commands for a Rich panel."""
    sug = [ModelSuggestion(model=str(s.get("model") or ""), label=str(s.get("label") or s.get("model") or ""),
                           note=str(s.get("note") or NOTE_INCLUDED))
           for s in (limit.get("suggestions") or []) if isinstance(s, dict) and s.get("model")]
    credits = limit.get("credits") if isinstance(limit.get("credits"), dict) else None
    hit = LimitHit(gauge_id=str(limit.get("gauge_id") or ""), scope=str(limit.get("scope") or ""),
                   limit_name=str(limit.get("limit_name") or ""), resets_at_ms=limit.get("resets_at_ms"),
                   message=str(limit.get("message") or ""), suggestions=tuple(sug),
                   credits=CreditsOption(str(credits.get("action"))) if credits and credits.get("action") else None)
    if markup:
        model_command, topup_command = f"[bold]{model_command}[/]", (f"[bold]{topup_command}[/]" if topup_command else None)
    return next_steps_text(hit, model_command=model_command, topup_command=topup_command, portal=portal)
