"""Anthropic prompt-cache warming (port of earendil-works/pi#9668).

An Anthropic cache entry lives ``cache_ttl`` (5m or 1h) past its last *use*. A tool round that
runs longer than that, or a pause between turns, drops the entire prefix back to a full-price
cache WRITE on the next request. The warmer re-sends the last request with ``max_tokens=1``
shortly before the entry expires: one cache READ of the prompt (about 1/12 of the miss price)
keeps the whole prefix warm for another TTL.

Phases: ``streaming`` while the turn is still running (a long tool call; the next request is
near-certain), ``idle`` after the turn settled (opt-in mode ``"idle"``; the next request is a
guess, priced at :data:`IDLE_CONTINUATION_PROBABILITY`). Every refresh is gated on expected
savings ``p * miss_cost - warm_cost >= $0.05`` so a cheap or unpriced model never pays for it,
and hard age limits (60 min streaming, 30 min idle) bound the spend of an abandoned session.

Only native ``anthropic_messages`` requests that actually carry ``cache_control`` markers are
warmed; a request whose thinking is budget-based is skipped because ``max_tokens=1`` would change
``budget_tokens``, which Anthropic keys the message cache on (and which it rejects outright).
"""

from __future__ import annotations

import contextvars
import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_WARMING_MODES = ("off", "streaming", "idle")
DEFAULT_CACHE_WARMING_MODE = "off"

#: Streaming warming never continues past this long after the real request that started it.
MAX_WARMING_AGE_S = 60 * 60
#: Idle warming uses a shorter horizon: continuation estimates degrade with age.
MAX_IDLE_WARMING_AGE_S = 30 * 60
#: A refresh is sent only when it is expected to save at least this many dollars.
MINIMUM_EXPECTED_SAVINGS_USD = 0.05
#: Chance a real request arrives before the entry expires while the agent sits idle (pi's
#: measured constant; per-session estimates were not better).
IDLE_CONTINUATION_PROBABILITY = 0.15
#: Anthropic cache tiers, as ``agent._cache_ttl`` spells them.
TTL_SECONDS = {"5m": 300, "1h": 3600}
#: Replays are one HTTP round trip; never let a hung socket pin the timer thread.
WARM_REQUEST_TIMEOUT_S = 60.0


def normalize_cache_warming_mode(value: Any) -> str:
    """Config value -> one of :data:`CACHE_WARMING_MODES`; unknown/falsy values mean ``off``."""
    if isinstance(value, str) and value.strip().lower() in CACHE_WARMING_MODES:
        return value.strip().lower()
    return DEFAULT_CACHE_WARMING_MODE


def cache_warming_delay_s(ttl_s: float) -> Optional[float]:
    """Refresh at 90% of the TTL while keeping at least ten seconds of margin; ``None`` when
    the TTL is too short to leave any margin."""
    if ttl_s <= 10:
        return None
    return max(1.0, min(ttl_s * 0.9, ttl_s - 10))


def has_cache_markers(api_kwargs: Dict[str, Any]) -> bool:
    """True when the request carries at least one ``cache_control`` marker (system, tools or
    messages) — without one there is no entry to keep warm."""

    def _walk(node: Any) -> bool:
        if isinstance(node, dict):
            if "cache_control" in node:
                return True
            return any(_walk(v) for v in node.values())
        if isinstance(node, list):
            return any(_walk(v) for v in node)
        return False

    return any(_walk(api_kwargs.get(k)) for k in ("system", "tools", "messages"))


def is_replayable(api_kwargs: Dict[str, Any]) -> bool:
    """Whether replaying with ``max_tokens=1`` leaves the cache entry untouched.

    Budget-based thinking (``{"type": "enabled", "budget_tokens": N}``) requires
    ``max_tokens > budget_tokens`` and Anthropic keys the message cache on the budget, so the
    replay is both rejected and useless. Adaptive thinking and no thinking replay fine.
    """
    thinking = api_kwargs.get("thinking")
    if not isinstance(thinking, dict):
        return True
    return thinking.get("type") != "enabled" or "budget_tokens" not in thinking


@dataclass(frozen=True)
class CacheWarmingDecision:
    """Inputs and outcome of one warm-or-stop decision."""

    phase: str
    warm_cost: float
    miss_cost: float
    continuation_probability: float
    expected_savings: float
    economics_available: bool
    action: str  # "warm" | "stop"


def _per_token(rate: Optional[Decimal]) -> float:
    return float(rate) / 1_000_000 if rate is not None else 0.0


def evaluate_economics(pricing: Any, prompt_tokens: int, phase: str) -> CacheWarmingDecision:
    """Price one refresh against the miss it prevents using a ``PricingEntry``-shaped object
    (``input/output/cache_read/cache_write_cost_per_million``)."""
    read = _per_token(getattr(pricing, "cache_read_cost_per_million", None))
    write = _per_token(getattr(pricing, "cache_write_cost_per_million", None))
    inp = _per_token(getattr(pricing, "input_cost_per_million", None))
    out = _per_token(getattr(pricing, "output_cost_per_million", None))
    hit_cost = prompt_tokens * read
    miss_full = prompt_tokens * (write if write > 0 else inp)
    warm_cost = hit_cost + out
    miss_cost = max(0.0, miss_full - hit_cost)
    probability = IDLE_CONTINUATION_PROBABILITY if phase == "idle" else 1.0
    economics_available = prompt_tokens > 0 and (hit_cost > 0 or miss_full > 0)
    expected = probability * miss_cost - warm_cost
    action = "warm" if economics_available and expected >= MINIMUM_EXPECTED_SAVINGS_USD else "stop"
    return CacheWarmingDecision(
        phase=phase, warm_cost=warm_cost, miss_cost=miss_cost, continuation_probability=probability,
        expected_savings=expected, economics_available=economics_available, action=action,
    )


@dataclass
class _Run:
    api_kwargs: Dict[str, Any]
    client: Any
    model: str
    prompt_tokens: int
    ttl_s: float
    delay_s: float
    is_current: Callable[[], bool]
    started_at: float
    phase: str = "streaming"
    next_warm_at: float = 0.0
    refresh_deadline_at: float = 0.0
    timer: Optional[threading.Timer] = None
    refreshing: bool = False
    warm_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class PromptCacheWarmer:
    """Keeps one prompt-cache entry alive per agent by replaying its request with a one-token
    output cap before the entry expires. ``start`` replaces any previous run; warm requests never
    extend the fixed safety windows."""

    def __init__(self, agent: Any):
        self.agent = agent
        self._run: Optional[_Run] = None
        self._inactive_reason = "waiting for first request"
        self._last_decision: Optional[CacheWarmingDecision] = None

    # -- read side ------------------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return normalize_cache_warming_mode(getattr(self.agent, "_cache_warming_mode", None))

    @property
    def status(self) -> Dict[str, Any]:
        """``{"state": inactive|scheduled|refreshing, "reason"?, "next_warm_at"?, "decision"?}``."""
        if self.mode == "off":
            return {"state": "inactive", "reason": "cache warming disabled"}
        run = self._run
        if run is None:
            return {"state": "inactive", "reason": self._inactive_reason, "decision": self._last_decision}
        if not run.is_current():
            return {"state": "inactive", "reason": "conversation context changed"}
        return {
            "state": "refreshing" if run.refreshing else "scheduled", "phase": run.phase,
            "next_warm_at": run.next_warm_at, "warm_count": run.warm_count,
            "decision": self._evaluate(run),
        }

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self, api_kwargs: Dict[str, Any], response: Any) -> Optional[str]:
        """Arm a refresh for the request that just produced ``response``. Returns the reason
        nothing was armed, ``None`` when a timer is scheduled."""
        self._clear_run()
        agent = self.agent
        if self.mode == "off":
            return self._stop("cache warming disabled")
        if getattr(agent, "api_mode", None) != "anthropic_messages":
            return self._stop("only native Anthropic requests are warmed")
        if not getattr(agent, "_use_prompt_caching", False) or getattr(agent, "_cache_disabled", False):
            return self._stop("prompt caching disabled")
        if "_moa_prepared_request" in api_kwargs:
            return self._stop("MoA requests are not replayable")
        if getattr(agent, "_interrupt_requested", False):
            return self._stop("turn interrupted")
        ttl_s = TTL_SECONDS.get(getattr(agent, "_cache_ttl", None) or "")
        if ttl_s is None:
            return self._stop("cache lifetime unavailable")
        if not has_cache_markers(api_kwargs):
            return self._stop("request carries no cache markers")
        if not is_replayable(api_kwargs):
            return self._stop("request cannot be replayed safely (budget thinking)")
        delay_s = cache_warming_delay_s(ttl_s)
        if delay_s is None:
            return self._stop("cache lifetime unavailable")
        prompt_tokens = self._prompt_tokens(response)
        if prompt_tokens <= 0:
            return self._stop("prompt size unavailable")
        client = getattr(agent, "_anthropic_client", None)
        if client is None:
            return self._stop("no Anthropic client")
        # Economics before the deep copy: an unpriced/cheap model never pays for the copy either.
        decision = evaluate_economics(self._pricing(), prompt_tokens, "streaming")
        self._last_decision = decision
        if decision.action != "warm":
            return self._stop(
                "expected savings below threshold" if decision.economics_available
                else "cache economics unavailable"
            )
        api_calls = getattr(agent, "session_api_calls", 0)
        model = getattr(agent, "model", "")

        def _is_current() -> bool:
            return (
                getattr(agent, "session_api_calls", None) == api_calls
                and getattr(agent, "model", None) == model
                and not getattr(agent, "_interrupt_requested", False)
            )

        # The loop mutates message dicts in place between requests (cache-marker redecoration,
        # tool-result eviction); the replay must send exactly what wrote the entry.
        run = _Run(
            api_kwargs=self._replay_kwargs(api_kwargs), client=client, model=model,
            prompt_tokens=prompt_tokens, ttl_s=ttl_s, delay_s=delay_s, is_current=_is_current,
            started_at=time.time(),
        )
        self._run = run
        self._schedule(run)
        return None if self._run is run else self._inactive_reason

    def on_turn_settled(self) -> None:
        """The turn finished: ``streaming`` mode stops; ``idle`` mode keeps warming for up to
        :data:`MAX_IDLE_WARMING_AGE_S` after the request."""
        run = self._run
        if run is None:
            return
        if self.mode != "idle":
            self._stop("agent run settled")
            return
        run.phase = "idle"
        deadline = run.started_at + MAX_IDLE_WARMING_AGE_S
        if run.next_warm_at > deadline or time.time() >= deadline:
            self._stop("30-minute idle safety limit reached")

    def cancel(self, reason: str = "inactive") -> None:
        self._stop(reason)

    # -- internals ------------------------------------------------------------------------------
    def _clear_run(self) -> None:
        run, self._run = self._run, None
        if run is None:
            return
        with run.lock:
            if run.timer is not None:
                run.timer.cancel()
                run.timer = None

    def _stop(self, reason: str) -> str:
        self._clear_run()
        self._inactive_reason = reason
        return reason

    def _schedule(self, run: _Run) -> None:
        now = time.time()
        run.next_warm_at = now + run.delay_s
        # A timer can fire late after sleep or a blocked interpreter. Keep half of the planned
        # pre-expiry margin for that: a late refresh is a full-price write, not a warm.
        run.refresh_deadline_at = run.next_warm_at + (run.ttl_s - run.delay_s) / 2
        limit = MAX_IDLE_WARMING_AGE_S if run.phase == "idle" else MAX_WARMING_AGE_S
        deadline = run.started_at + limit
        if run.next_warm_at > deadline or now >= deadline:
            self._stop("30-minute idle safety limit reached" if run.phase == "idle"
                       else "one-hour safety limit reached")
            return
        # Secret scope and other ContextVars live on the loop thread; the timer thread must
        # run inside a copy or every scoped read in the SDK client raises.
        ctx = contextvars.copy_context()
        timer = threading.Timer(max(0.0, run.next_warm_at - now), ctx.run, args=(self._refresh, run))
        timer.daemon = True
        timer.name = "prompt-cache-warmer"
        with run.lock:
            run.timer = timer
        timer.start()

    def _validate(self, run: _Run) -> bool:
        if self._run is not run:
            return False
        if self.mode == "off":
            self._stop("cache warming disabled")
            return False
        if self.mode == "streaming" and run.phase == "idle":
            self._stop("agent run settled")
            return False
        if not run.is_current():
            self._stop("conversation context changed")
            return False
        if time.time() > run.refresh_deadline_at:
            self._stop("cache refresh deadline missed")
            return False
        return True

    def _refresh(self, run: _Run) -> None:
        with run.lock:
            run.timer = None
        if not self._validate(run):
            return
        decision = self._evaluate(run)
        self._last_decision = decision
        if decision.action != "warm":
            self._stop("expected savings below threshold" if decision.economics_available
                       else "cache economics unavailable")
            return
        run.refreshing = True
        try:
            message = self._send(run)
        except Exception as exc:  # best-effort: warming must never disturb the live turn
            logger.info("%sprompt cache warm failed: %s", self._prefix(), exc)
            self._stop(f"warm request failed: {type(exc).__name__}")
            return
        finally:
            run.refreshing = False
        run.warm_count += 1
        self._record(run, message, decision)
        if self._run is run:
            self._schedule(run)

    def _send(self, run: _Run) -> Any:
        from agent.anthropic_adapter import create_anthropic_message

        client = run.client
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            client = with_options(max_retries=0, timeout=WARM_REQUEST_TIMEOUT_S)
        kwargs = dict(run.api_kwargs)
        kwargs["max_tokens"] = 1
        kwargs.pop("stream", None)
        return create_anthropic_message(client, kwargs, log_prefix=self._prefix(), prefer_stream=False)

    def _record(self, run: _Run, message: Any, decision: CacheWarmingDecision) -> None:
        """Fold the refresh's tokens and dollars into the session totals (never into the
        compressor or usage anchors: a one-token replay is not a conversation turn)."""
        agent = self.agent
        usage = getattr(message, "usage", None)
        if not usage:
            return
        try:
            from agent.usage_pricing import estimate_usage_cost, normalize_usage

            canonical = normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode)
            cost = estimate_usage_cost(
                agent.model, canonical, provider=agent.provider, base_url=agent.base_url,
                api_key=getattr(agent, "api_key", ""),
            )
            # Not session_api_calls: that counter is the conversation's request sequence (and the
            # warmer's own currency token); a replay is accounted, not a turn.
            agent.session_cache_warm_calls = getattr(agent, "session_cache_warm_calls", 0) + 1
            agent.session_prompt_tokens += canonical.prompt_tokens
            agent.session_completion_tokens += canonical.output_tokens
            agent.session_total_tokens += canonical.total_tokens
            agent.session_input_tokens += canonical.input_tokens
            agent.session_output_tokens += canonical.output_tokens
            agent.session_cache_read_tokens += canonical.cache_read_tokens
            agent.session_cache_write_tokens += canonical.cache_write_tokens
            amount = float(cost.amount_usd) if cost.amount_usd is not None else None
            if amount is not None:
                agent.session_estimated_cost_usd += amount
            db = getattr(agent, "_session_db", None)
            if db is not None and getattr(agent, "session_id", None):
                db.queue_token_counts(
                    agent.session_id, source="cache_warm",
                    input_tokens=canonical.input_tokens, output_tokens=canonical.output_tokens,
                    cache_read_tokens=canonical.cache_read_tokens,
                    cache_write_tokens=canonical.cache_write_tokens,
                    reasoning_tokens=canonical.reasoning_tokens, estimated_cost_usd=amount,
                    cost_status=cost.status, cost_source=cost.source,
                    billing_provider=agent.provider, billing_base_url=agent.base_url,
                    billing_mode=None, model=agent.model, api_call_count=1,
                )
            logger.info(
                "%sprompt cache warmed (%s #%d): cache_read=%d write=%d cost=%s "
                "expected_savings=$%.3f next=%s", self._prefix(), run.phase, run.warm_count,
                canonical.cache_read_tokens, canonical.cache_write_tokens,
                f"${amount:.4f}" if amount is not None else "?", decision.expected_savings,
                time.strftime("%H:%M:%S", time.localtime(time.time() + run.delay_s)),
            )
        except Exception:
            logger.debug("prompt cache warm accounting failed", exc_info=True)

    def _evaluate(self, run: _Run) -> CacheWarmingDecision:
        return evaluate_economics(self._pricing(), run.prompt_tokens, run.phase)

    def _pricing(self) -> Any:
        agent = self.agent
        try:
            from agent.usage_pricing import get_pricing_entry

            return get_pricing_entry(
                agent.model, provider=agent.provider, base_url=agent.base_url,
                api_key=getattr(agent, "api_key", ""),
            )
        except Exception:
            return None

    def _prompt_tokens(self, response: Any) -> int:
        usage = getattr(response, "usage", None)
        if not usage:
            return 0
        try:
            from agent.usage_pricing import normalize_usage

            return int(normalize_usage(usage, provider=self.agent.provider,
                                       api_mode=self.agent.api_mode).prompt_tokens or 0)
        except Exception:
            return 0

    @staticmethod
    def _replay_kwargs(api_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = {k: v for k, v in api_kwargs.items() if k not in ("stream", "_moa_prepared_request")}
        for key in ("system", "tools", "messages", "thinking", "tool_choice", "metadata"):
            if key in kwargs:
                kwargs[key] = copy.deepcopy(kwargs[key])
        return kwargs

    def _prefix(self) -> str:
        return getattr(self.agent, "log_prefix", "") or ""


# -- agent-facing seams (each a no-op when nothing is armed) ---------------------------------------

def get_prompt_cache_warmer(agent: Any) -> PromptCacheWarmer:
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is None:
        warmer = PromptCacheWarmer(agent)
        agent._prompt_cache_warmer = warmer
    return warmer


def start_prompt_cache_warming(agent: Any, api_kwargs: Any, response: Any) -> None:
    """Called once per successful provider response with the request that produced it."""
    if normalize_cache_warming_mode(getattr(agent, "_cache_warming_mode", None)) == "off":
        return
    if not isinstance(api_kwargs, dict):
        return
    try:
        get_prompt_cache_warmer(agent).start(api_kwargs, response)
    except Exception:
        logger.debug("prompt cache warming did not arm", exc_info=True)


def cancel_prompt_cache_warming(agent: Any, reason: str = "inactive") -> None:
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is not None:
        warmer.cancel(reason)


def settle_prompt_cache_warming(agent: Any) -> None:
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is not None:
        warmer.on_turn_settled()
