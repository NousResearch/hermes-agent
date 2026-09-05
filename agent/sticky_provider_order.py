"""Pin native OpenRouter provider ``order`` to one live slug, rotate on failure.

Turn-local state lives on ``agent._sticky_provider_order``. Canonical
``agent.providers_order`` / ``agent.providers_allowed`` are the pool
baseline and are never mutated by this module. Sticky is live only on
native OpenRouter chat-completions (not Nous, not ``custom:``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from hermes_constants import (
    DEFAULT_STICKY_ORDER,
    StickyOrderConfig,
    resolve_sticky_order_config,
)

logger = logging.getLogger("run_agent")

# * Failures that mean the pinned upstream is unhealthy. rate_limit (429)
# keeps the pin — the provider is alive and the cache is still warm.
_ROTATE_REASON_VALUES = frozenset({"timeout", "overloaded", "server_error"})
# * Same marker as hermes_cli.models.validate_requested_model.
_OPENROUTER_PRESET_MARKER = "@preset/"
# * OpenRouter docs: provider.order disables :nitro/:floor tier-admission
# unless a slug names a tier endpoint (openai/priority, openai/fast,
# google-vertex/flex). fast and priority are interchangeable.
_STICKY_TIER_VARIANT_SUFFIXES = frozenset({"nitro", "floor"})
_OPENROUTER_TIER_ENDPOINT_SUFFIXES = frozenset({"priority", "fast", "flex"})
# * Process-wide latch: configured order + :nitro/:floor warns once.
_nitro_floor_order_warned = False
# * Profiles that must not pin: Nous ignores/400s provider prefs; custom
# never emits extra_body.provider on the wire.
_STICKY_EXCLUDED_PROVIDERS = frozenset({
    "nous",
    "nous-portal",
    "nousresearch",
    "custom",
})


class StickyOrderState:
    """Pinned slug, rotation budget, and last-attempt timestamp."""

    def __init__(
        self,
        config: StickyOrderConfig | None = None,
        pool: list[str] | None = None,
        order_snapshot: list[str] | None = None,
        clock: Callable[[], float] | None = None,
        bound_model: str | None = None,
    ) -> None:
        self.config = config or DEFAULT_STICKY_ORDER
        self.pool = list(pool or [])
        self.order_snapshot = list(order_snapshot or [])
        self.active_index = 0
        self.last_attempt_at: float | None = None
        self.rotations_this_request = 0
        # * Failed logical-request attempts this request (rotate-worthy
        # errors). Defer model fallback while this is < len(pool).
        self.attempts_this_request = 0
        self.clock = clock or time.monotonic
        # * Exact model id (including variant suffix) this pin was bound for.
        self.bound_model = str(bound_model or "")

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def is_active(self) -> bool:
        return bool(self.config.enabled) and bool(self.pool)

    @property
    def active_slug(self) -> str | None:
        if not self.pool:
            return None
        idx = self.active_index
        if idx < 0 or idx >= len(self.pool):
            return self.pool[0]
        return self.pool[idx]

    def note_attempt(self) -> None:
        self.last_attempt_at = self.clock()

    def maybe_reset_idle(self) -> None:
        """Return to pool[0] after a long gap between logical requests."""
        if self.last_attempt_at is None:
            return
        try:
            ttl = float(self.config.ttl_seconds)
        except (TypeError, ValueError):
            return
        if ttl <= 0:
            return
        if self.clock() - self.last_attempt_at > ttl:
            self.active_index = 0

    def rotate(self, reason: Any = None) -> bool:
        """Advance one step. Cap is ``len(pool) - 1`` per logical request."""
        if not self.is_active or len(self.pool) <= 1:
            return False
        cap = len(self.pool) - 1
        if self.rotations_this_request >= cap:
            return False
        self.active_index = (self.active_index + 1) % len(self.pool)
        self.rotations_this_request += 1
        logger.info(
            "sticky_provider_order: slug=%s reason=%s index=%s",
            self.active_slug,
            _reason_label(reason),
            self.active_index,
        )
        return True

    def begin_logical_request(self) -> None:
        # * Idle TTL is a between-request rule. Apply it here — the
        # conversation-loop hook sits outside the retry while — never
        # on a prefs rebuild that also runs for in-request retries.
        self.maybe_reset_idle()
        self.rotations_this_request = 0
        self.attempts_this_request = 0

    def record_error_attempt(self) -> None:
        """Count one rotate-worthy failure against the logical-request walk."""
        self.attempts_this_request += 1


def _reason_label(reason: Any) -> str:
    if reason is None:
        return "-"
    value = getattr(reason, "value", None)
    if isinstance(value, str) and value:
        return value
    return str(reason)


def _normalize_provider_slugs(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        slug = str(item or "").strip()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append(slug)
    return out


def _resolve_bind_config(raw_routing_config: Any) -> StickyOrderConfig:
    if isinstance(raw_routing_config, StickyOrderConfig):
        return raw_routing_config
    if not isinstance(raw_routing_config, dict):
        return resolve_sticky_order_config({})
    if "sticky_order" in raw_routing_config:
        return resolve_sticky_order_config(raw_routing_config)
    if "enabled" in raw_routing_config or "ttl_seconds" in raw_routing_config:
        return resolve_sticky_order_config({"sticky_order": raw_routing_config})
    return resolve_sticky_order_config(raw_routing_config)


def _build_pool(agent: Any) -> tuple[list[str], list[str], bool]:
    """Return ``(pool, order_snapshot, empty_intersection)``."""
    order = _normalize_provider_slugs(getattr(agent, "providers_order", None))
    only = _normalize_provider_slugs(getattr(agent, "providers_allowed", None))
    if only:
        only_set = set(only)
        pool = [slug for slug in order if slug in only_set]
        return pool, order, not pool
    return order, order, False


def bind_sticky_order(
    agent: Any,
    raw_routing_config: Any = None,
    *,
    clock: Callable[[], float] | None = None,
    model: str | None = None,
) -> StickyOrderState:
    """Attach or replace ``agent._sticky_provider_order``.

    Pool is resolved ``order ∩ only``. An empty intersection disables the
    pin (warning) even when the flag is on. A changed pool (order or
    only) keeps the previous active slug when it is still eligible,
    otherwise resets the index to 0. An unchanged pool keeps the index.

    ``model`` is the id the pool was resolved for. ``None`` (the default)
    reads ``agent.model`` so constructor bind in ``agent_init`` stays a
    two-arg call.
    """
    config = _resolve_bind_config(raw_routing_config)
    pool, order_snapshot, empty_intersection = _build_pool(agent)
    if config.enabled and empty_intersection:
        logger.warning(
            "sticky_provider_order: empty only ∩ order intersection; "
            "sticky disabled",
        )

    previous = getattr(agent, "_sticky_provider_order", None)
    prev_state = previous if isinstance(previous, StickyOrderState) else None
    if model is None:
        bound_model = str(getattr(agent, "model", None) or "")
    else:
        bound_model = str(model or "")
    state = StickyOrderState(
        config=config,
        pool=pool,
        order_snapshot=order_snapshot,
        clock=clock or (prev_state.clock if prev_state is not None else None),
        bound_model=bound_model,
    )
    if prev_state is not None:
        state.last_attempt_at = prev_state.last_attempt_at
        if prev_state.pool != pool:
            prev_slug = prev_state.active_slug
            if prev_slug and prev_slug in pool:
                state.active_index = pool.index(prev_slug)
            else:
                state.active_index = 0
        else:
            state.active_index = prev_state.active_index
    if state.pool:
        state.active_index = max(0, min(state.active_index, len(state.pool) - 1))
    else:
        state.active_index = 0
    agent._sticky_provider_order = state
    return state


def sticky_state(agent: Any) -> StickyOrderState | None:
    state = getattr(agent, "_sticky_provider_order", None)
    return state if isinstance(state, StickyOrderState) else None


def _normalized_provider(agent: Any) -> str:
    return str(getattr(agent, "provider", None) or "").strip().lower()


def _agent_base_url(agent: Any) -> Any:
    return getattr(agent, "base_url", None) or getattr(
        agent, "_base_url_lower", None,
    )


def _base_url_host_is(agent: Any, domain: str) -> bool:
    base_url = _agent_base_url(agent)
    if not base_url:
        return False
    try:
        from utils import base_url_host_matches
    except Exception:
        return domain in str(base_url).lower()
    return base_url_host_matches(base_url, domain)


def _provider_excluded_from_sticky(provider: str) -> bool:
    if provider in _STICKY_EXCLUDED_PROVIDERS:
        return True
    return provider.startswith("custom:")


def _model_uses_openrouter_preset(model_id: Any) -> bool:
    return _OPENROUTER_PRESET_MARKER in str(model_id or "")


def _model_uses_openrouter_tier_variant(model_id: Any) -> bool:
    """True when the id ends with ``:nitro`` or ``:floor`` (tier admission)."""
    raw = str(model_id or "")
    if ":" not in raw:
        return False
    suffix = raw.rsplit(":", 1)[-1].strip().lower()
    return suffix in _STICKY_TIER_VARIANT_SUFFIXES


def _slug_is_openrouter_tier_endpoint(slug: str) -> bool:
    """True for OpenRouter tier slugs such as ``openai/priority``."""
    raw = str(slug or "").strip()
    if "/" not in raw:
        return False
    suffix = raw.rsplit("/", 1)[-1].strip().lower()
    return suffix in _OPENROUTER_TIER_ENDPOINT_SUFFIXES


def maybe_warn_nitro_floor_order(agent: Any, preferences: dict) -> None:
    """Warn once when ``order`` disables ``:nitro``/``:floor`` tier-admission.

    Does not mutate *preferences*. OpenRouter replaces the variant sort
    with ``provider.order``, which drops automatic priority/flex
    admission unless a tier-suffixed slug is listed. Warning and latch
    fire only on native OpenRouter chat-completions.
    """
    global _nitro_floor_order_warned
    if _nitro_floor_order_warned:
        return
    if not _model_uses_openrouter_tier_variant(getattr(agent, "model", None)):
        return
    if not isinstance(preferences, dict):
        return
    order = _normalize_provider_slugs(preferences.get("order"))
    if not order:
        return
    if any(_slug_is_openrouter_tier_endpoint(slug) for slug in order):
        return
    if not _route_applies_provider_preferences(agent):
        return
    _nitro_floor_order_warned = True
    logger.warning(
        "provider.order disables :nitro/:floor tier-admission. "
        "To keep tier endpoints eligible, name a tier-suffixed slug "
        "in order (for example openai/priority, openai/fast, or "
        "google-vertex/flex).",
    )


def _route_applies_provider_preferences(agent: Any) -> bool:
    """True on native OpenRouter chat_completions paths that emit provider prefs."""
    # * Sticky is live only where extra_body.provider is actually sent:
    # chat_completions on native OpenRouter. Other api_modes
    # (anthropic_messages, codex_responses, bedrock_converse, …) return
    # from build_api_kwargs before _provider_preferences_for_agent.
    api_mode = str(getattr(agent, "api_mode", None) or "").strip().lower()
    if api_mode != "chat_completions":
        return False
    provider = _normalized_provider(agent)
    if _provider_excluded_from_sticky(provider):
        return False
    if _base_url_host_is(agent, "nousresearch.com"):
        return False
    if provider == "openrouter":
        return True
    # * Legacy transport emits extra_body.provider when the URL is
    # OpenRouter even without an explicit provider name.
    checker = getattr(agent, "_is_openrouter_url", None)
    if callable(checker):
        try:
            if checker():
                return True
        except Exception:
            pass
    return _base_url_host_is(agent, "openrouter.ai")


def sticky_is_live(agent: Any) -> bool:
    state = sticky_state(agent)
    if state is None or not state.is_active:
        return False
    if not _route_applies_provider_preferences(agent):
        return False
    # * A model change without re-bind silently disables sticky
    # (fail-closed). Model fallback pauses the pin until the next bind.
    current_model = str(getattr(agent, "model", None) or "")
    if current_model != state.bound_model:
        return False
    if _model_uses_openrouter_preset(current_model):
        return False
    if _model_uses_openrouter_tier_variant(current_model):
        return False
    return True


def begin_sticky_logical_request(agent: Any) -> None:
    state = sticky_state(agent)
    if state is not None:
        state.begin_logical_request()


def note_attempt(state: StickyOrderState) -> None:
    """Tick ``last_attempt_at``. Callers must already be on a live wire path."""
    if isinstance(state, StickyOrderState):
        state.note_attempt()


def note_sticky_attempt(agent: Any) -> None:
    """Tick ``last_attempt_at`` from an agent without rebuilding prefs.

    Summary retries reuse a frozen extra_body, so the prefs rebuild that
    normally records the attempt never runs. Call this on every summary
    API attempt, including empty-content retries. Non-wire modes do not
    tick — a ``/model`` switch away from chat_completions must not keep
    the pin warm.
    """
    # * TTL only tracks real pinned requests. sticky_is_live reads
    # api_mode / provider / URL, not prefs, so this cannot recurse.
    if not sticky_is_live(agent):
        return
    state = sticky_state(agent)
    if state is not None:
        note_attempt(state)


def rotate(state: StickyOrderState, reason: Any = None) -> bool:
    if not isinstance(state, StickyOrderState):
        return False
    return state.rotate(reason)


def should_rotate_for_reason(reason: Any) -> bool:
    if reason is None:
        return False
    value = getattr(reason, "value", reason)
    return value in _ROTATE_REASON_VALUES


def rotate_sticky_on_classified_error(agent: Any, reason: Any) -> bool:
    if not should_rotate_for_reason(reason):
        return False
    if not sticky_is_live(agent):
        return False
    state = sticky_state(agent)
    if state is None:
        return False
    # * Count the failed slug even when rotation is already at the cap
    # so the last pool member still consumes its attempt before fallback.
    state.record_error_attempt()
    return rotate(state, reason)


def sticky_defers_model_fallback(agent: Any) -> bool:
    """True until every pool slug has produced a rotate-worthy error."""
    if not sticky_is_live(agent):
        return False
    state = sticky_state(agent)
    if state is None or len(state.pool) <= 1:
        return False
    return state.attempts_this_request < len(state.pool)


def sticky_retry_floor(agent: Any) -> int | None:
    """Minimum retry budget needed to walk the whole pin pool."""
    if not sticky_is_live(agent):
        return None
    state = sticky_state(agent)
    if state is None or not state.pool:
        return None
    return len(state.pool)


def apply_sticky_retry_budget(agent: Any, max_retries: int) -> int:
    """Reset the per-request rotation budget and raise ``max_retries``.

    Conversation-loop entry for a logical API request. When sticky is
    live, the retry ceiling is at least ``len(pool)`` so each slug can
    be attempted. Also the idle-TTL checkpoint: a gap longer than
    ``ttl_seconds`` since the last attempt returns the pin to
    ``pool[0]``. Disabled / non-wire routes still reset counters but
    return ``max_retries``.
    """
    begin_sticky_logical_request(agent)
    floor = sticky_retry_floor(agent)
    if floor:
        return max(int(max_retries), int(floor))
    return max_retries


def should_fallback_on_transport_failure(
    agent: Any,
    *,
    is_transport_failure: bool,
    retry_count: int,
) -> bool:
    """Model-fallback gate for timeout/overloaded after classification.

    Classic rule is ``is_transport_failure and retry_count >= 2``. Sticky
    defers that while ``attempts_this_request < len(pool)`` so the last
    slug still gets a real request. When the feature is off or not live
    this is the classic condition.
    """
    return bool(
        is_transport_failure
        and retry_count >= 2
        and not sticky_defers_model_fallback(agent)
    )


def apply_sticky_order_to_preferences(agent: Any, preferences: dict) -> dict:
    """Pin ``order`` to the active slug inside the resolved pool.

    Never deletes or overwrites an existing ``only`` list. If ``only`` is
    present and does not contain the active slug, or the slug is outside
    the bound pool, this is a no-op.
    """
    if not isinstance(preferences, dict):
        return preferences
    if not sticky_is_live(agent):
        return preferences
    state = sticky_state(agent)
    if state is None:
        return preferences
    # * Do not idle-reset here: prefs rebuild on every retry of the
    # same logical request, and retry backoff can exceed a small TTL.
    slug = state.active_slug
    if not slug or slug not in state.pool:
        return preferences
    if "only" in preferences:
        only_slugs = _normalize_provider_slugs(preferences.get("only"))
        if slug not in only_slugs:
            return preferences
    note_attempt(state)
    preferences["order"] = [slug]
    preferences["allow_fallbacks"] = False
    return preferences
