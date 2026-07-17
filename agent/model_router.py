"""Deterministic, provider-aware model router for subagent delegation.

Phase03. Instead of the parent forcing one fixed model onto every subagent,
this module selects a ``(provider, model)`` pair *dynamically* from the
providers the user actually holds credentials for, scoring candidates by
task-type capability hints and preferring paid/preferred models first, then
free/local models as a final fallback tier. It also produces a
provider-diverse fallback chain so a child can recover from rate-limit /
credential-exhaustion exactly like the top-level agent does.

Design constraints (see AGENTS.md):

  * Pure, deterministic, stdlib-only. No LLM calls; nothing hits the network
    at import time. Every external lookup (provider inventory, model catalog,
    pricing, credential health) is *injected* so tests never touch live APIs.
  * Never hardcodes which model to use for a task type (no fixed model names).
    Scoring matches generic capability words against each model's own catalog
    id/description — brand names are deliberately absent from the keyword sets.
  * Zero effect unless ``delegation.model_router.enabled`` is true, so current
    behaviour and prompt caching stay byte-stable by default.
  * Explicit per-task ``model`` / ``provider`` always wins — the router steps
    aside entirely when the caller pinned a model.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Capability-hint keywords per route. These match against a MODEL's own
# catalog id/description — they are NOT a map to fixed model names. Keeping
# brand names out of this table is what makes the router "without fixed model
# names": the parent expresses *what kind of work* the task is, never *which
# model* must run it.
ROUTE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "coding": ("code", "coder", "coding", "program", "debug", "engineer", "swe", "dev"),
    "research": ("research", "search", "analy", "reason", "think", "deep", "web"),
    "writing": ("write", "writer", "author", "doc", "summar", "report", "prose", "chat"),
    "architecture": ("architect", "design", "system", "plan", "reason", "engineer"),
    "default": ("instruct", "chat", "general", "assistant", "reason"),
}

# Goal/context intent terms that pick the route. Ordered most-specific first so
# a tie prefers the earlier (more specific) route.
_ROUTE_INTENT: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "coding",
        (
            "code", "coding", "implement", "bug", "fix", "refactor", "function",
            "script", "test", "pytest", "compile", "build", "patch", "lint",
        ),
    ),
    (
        "architecture",
        (
            "architect", "design", "system", "schema", "orchestr", "pipeline",
            "plan", "structure", "diagram",
        ),
    ),
    (
        "research",
        (
            "research", "search", "investigate", "analy", "find", "compare",
            "gather", "study", "review", "benchmark",
        ),
    ),
    (
        "writing",
        (
            "write", "draft", "summar", "report", "document", "blog", "article",
            "memo", "letter", "translate",
        ),
    ),
)

# Providers whose models are always treated as the free/local fallback tier
# regardless of pricing metadata (local runtimes have no per-token price).
_LOCAL_PROVIDERS = {"lmstudio", "ollama", "ollama-cloud", "llamacpp", "llama-cpp", "custom"}

# Generic quality adjectives (no model names) used only as a soft tie-breaker.
_STRONG_HINTS = ("pro", "max", "large", "ultra", "advanced", "opus", "plus")
_LIGHT_HINTS = ("mini", "small", "lite", "fast", "flash", "turbo", "nano", "air")


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def infer_route_key(goal: str = "", context: str = "") -> str:
    """Classify a task into a route from its goal/context text.

    Returns one of ``coding``, ``research``, ``writing``, ``architecture`` or
    ``default``. Deterministic: counts keyword hits per route and returns the
    highest, ties broken by the ``_ROUTE_INTENT`` ordering.
    """
    blob = f"{goal or ''} {context or ''}".lower()
    best = "default"
    best_hits = 0
    for route, terms in _ROUTE_INTENT:
        hits = sum(1 for t in terms if t in blob)
        if hits > best_hits:
            best_hits = hits
            best = route
    return best


def is_local_provider(provider: str) -> bool:
    return (provider or "").strip().lower() in _LOCAL_PROVIDERS


def candidate_is_free(
    provider: str,
    model: str,
    description: str = "",
    pricing: Optional[Dict[str, Any]] = None,
    free_providers: Sequence[str] = (),
) -> bool:
    """Best-effort free/local classification for the final fallback tier."""
    provider_l = (provider or "").strip().lower()
    free_set = {str(p).strip().lower() for p in (free_providers or ()) if str(p).strip()}
    if provider_l in free_set or is_local_provider(provider):
        return True
    pricing = pricing or {}
    price = pricing.get(model)
    if price is None:
        price = pricing.get((model or "").lower())
    if isinstance(price, dict):
        # NOTE: do not use ``float(x or 1)`` here — a genuine 0.0 price is
        # falsy and would collapse to the fallback, misclassifying a free
        # model as paid. Only claim "free" when BOTH sides are explicitly 0.
        prompt_raw = price.get("prompt")
        completion_raw = price.get("completion")
        try:
            if prompt_raw is not None and completion_raw is not None:
                if float(prompt_raw) == 0.0 and float(completion_raw) == 0.0:
                    return True
        except (TypeError, ValueError):
            pass
    blob = f"{provider} {model} {description}".lower()
    return ":free" in blob or "(free)" in blob or blob.rstrip().endswith(" free")


def score_candidate(candidate: Dict[str, Any], route_key: str) -> int:
    """Deterministic capability score for one candidate against a route."""
    blob = (
        f"{candidate.get('provider', '')} "
        f"{candidate.get('model', '')} "
        f"{candidate.get('description', '')}"
    ).lower()
    route = route_key if route_key in ROUTE_KEYWORDS else "default"
    score = 0
    for kw in ROUTE_KEYWORDS[route]:
        if kw in blob:
            score += 10
    for kw in ROUTE_KEYWORDS["default"]:
        if kw in blob:
            score += 2
    for kw in _STRONG_HINTS:
        if kw in blob:
            score += 3
    for kw in _LIGHT_HINTS:
        if kw in blob:
            score += 1
    # Preserve provider catalog ordering as a small, stable tie-breaker.
    try:
        score -= int(candidate.get("model_index", 0) or 0)
    except (TypeError, ValueError):
        pass
    return score


def build_candidates(
    route_key: str,
    providers: Sequence[str],
    *,
    catalog_fn: Callable[[str], Sequence[Any]],
    pricing_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    credential_ok_fn: Optional[Callable[[str], bool]] = None,
    free_providers: Sequence[str] = (),
    priority_providers: Sequence[str] = (),
    max_models_per_provider: int = 60,
    max_candidates: int = 12,
) -> List[Dict[str, Any]]:
    """Build an ordered candidate list across providers.

    Paid/preferred candidates come first (highest score first), free/local
    candidates form the tail as the final fallback tier. Providers whose
    credential pool reports no availability (``credential_ok_fn`` returns
    False) are skipped so an exhausted provider is never selected.

    ``priority_providers`` (earlier = higher priority) applies a strong
    *within-tier* score boost so a user-preferred provider surfaces first,
    without letting a preferred free/local provider jump ahead of the paid
    tier (the tier split is applied before sorting).
    """
    # Map provider -> priority bonus (earlier in the list = larger bonus).
    # The weight is large enough to dominate route/capability scoring within a
    # tier, so "prefer this provider" is honoured, but the paid/free tier split
    # below still holds because it partitions before this score is used.
    prio_rank: Dict[str, int] = {}
    plist = [str(p).strip().lower() for p in (priority_providers or ()) if str(p).strip()]
    for i, p in enumerate(plist):
        prio_rank[p] = (len(plist) - i) * 1000

    candidates: List[Dict[str, Any]] = []
    for provider in providers:
        provider = (provider or "").strip()
        if not provider:
            continue
        if credential_ok_fn is not None:
            try:
                if not credential_ok_fn(provider):
                    logger.debug("model_router: skip provider (no available credentials): %s", provider)
                    continue
            except Exception:
                logger.debug("model_router: credential check failed for %s", provider, exc_info=True)
        try:
            catalog = list(catalog_fn(provider) or [])
        except Exception:
            logger.debug("model_router: catalog lookup failed for %s", provider, exc_info=True)
            catalog = []
        if not catalog:
            continue
        pricing: Dict[str, Any] = {}
        if pricing_fn is not None:
            try:
                pricing = pricing_fn(provider) or {}
            except Exception:
                pricing = {}
        prio_bonus = prio_rank.get(provider.lower(), 0)
        for idx, item in enumerate(catalog[: max(1, max_models_per_provider)]):
            if isinstance(item, (list, tuple)):
                model = str(item[0] or "").strip()
                description = str(item[1] or "") if len(item) > 1 else ""
            else:
                model = str(item or "").strip()
                description = ""
            if not model:
                continue
            cand: Dict[str, Any] = {
                "provider": provider,
                "model": model,
                "description": description,
                "model_index": idx,
                "is_free": candidate_is_free(provider, model, description, pricing, free_providers),
                "is_local": is_local_provider(provider),
            }
            cand["score"] = score_candidate(cand, route_key) + prio_bonus
            candidates.append(cand)

    paid = [c for c in candidates if not (c["is_free"] or c["is_local"])]
    free_local = [c for c in candidates if c["is_free"] or c["is_local"]]
    paid.sort(key=lambda c: (int(c["score"]), -int(c["model_index"])), reverse=True)
    free_local.sort(key=lambda c: (int(c["score"]), -int(c["model_index"])), reverse=True)
    ordered = paid + free_local

    result: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for c in ordered:
        key = (c["provider"].lower(), c["model"].lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(c)
        if len(result) >= max(1, max_candidates):
            break
    return result


def router_enabled(router_cfg: Optional[Dict[str, Any]]) -> bool:
    """Whether the dynamic router should act. Default OFF."""
    if not isinstance(router_cfg, dict):
        return False
    return _truthy(router_cfg.get("enabled", False))


def task_has_explicit_model(task: Optional[Dict[str, Any]]) -> bool:
    """True when the caller pinned a per-task model/provider (router steps aside)."""
    if not isinstance(task, dict):
        return False
    return bool(task.get("model")) or bool(task.get("provider"))


def select_delegation_model(
    task: Optional[Dict[str, Any]],
    router_cfg: Optional[Dict[str, Any]],
    *,
    provider_inventory_fn: Callable[[], Sequence[str]],
    catalog_fn: Callable[[str], Sequence[Any]],
    pricing_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    credential_ok_fn: Optional[Callable[[str], bool]] = None,
    goal: str = "",
    context: str = "",
) -> Optional[Dict[str, Any]]:
    """Select a provider/model for one delegated task.

    Returns a dict::

        {
          "selected": {"provider": ..., "model": ...},
          "candidates": [ {provider, model, score, is_free, is_local, ...}, ... ],
          "fallback_chain": [ {provider, model}, ... ],   # excludes selected
          "route": "coding" | "research" | ...,
        }

    or ``None`` when the router is disabled, the task pinned a model, or no
    usable candidate exists (caller then keeps inherited/default credentials).
    """
    if not router_enabled(router_cfg):
        return None
    if task_has_explicit_model(task):
        return None

    assert isinstance(router_cfg, dict)  # narrowed by router_enabled

    try:
        providers = [str(p).strip() for p in (provider_inventory_fn() or []) if str(p).strip()]
    except Exception:
        logger.debug("model_router: provider inventory lookup failed", exc_info=True)
        return None
    if not providers:
        return None

    priority = router_cfg.get("provider_priority") or []
    priority_providers: List[str] = []
    if isinstance(priority, (list, tuple)) and priority:
        priority_providers = [str(p).strip() for p in priority if str(p).strip()]
        providers = priority_providers + [p for p in providers if p not in priority_providers]

    task_goal = goal or (task or {}).get("goal", "") or ""
    task_context = context or (task or {}).get("context", "") or ""
    route = infer_route_key(task_goal, task_context)

    try:
        max_per_provider = int(router_cfg.get("max_models_per_provider", 60) or 60)
    except (TypeError, ValueError):
        max_per_provider = 60
    try:
        max_candidates = int(router_cfg.get("max_candidates", 12) or 12)
    except (TypeError, ValueError):
        max_candidates = 12

    candidates = build_candidates(
        route,
        providers,
        catalog_fn=catalog_fn,
        pricing_fn=pricing_fn,
        credential_ok_fn=credential_ok_fn,
        free_providers=router_cfg.get("free_providers") or (),
        priority_providers=priority_providers,
        max_models_per_provider=max_per_provider,
        max_candidates=max_candidates,
    )
    if not candidates:
        return None

    selected = candidates[0]
    fallback_chain = [
        {"provider": c["provider"], "model": c["model"]} for c in candidates[1:]
    ]
    logger.info(
        "model_router: route=%s selected=%s/%s candidates=%d",
        route, selected["provider"], selected["model"], len(candidates),
    )
    return {
        "selected": {"provider": selected["provider"], "model": selected["model"]},
        "candidates": candidates,
        "fallback_chain": fallback_chain,
        "route": route,
    }
