"""Anthropic prompt-cache warming (port of earendil-works/pi#9668).

An Anthropic cache entry lives ``cache_ttl`` (5m or 1h) past its last *use*. A tool round that
runs longer than that drops the entire prefix back to a full-price cache WRITE on the next
request. While the turn is still running (the next request is certain), the warmer re-sends the
last request with ``max_tokens=1`` shortly before the entry expires: one cache READ of the prompt
(about 1/12 of the miss price) keeps the whole prefix warm for another TTL. It always stops when
the turn ends: warming between turns would bet money on the user replying in time.

Every refresh is gated on expected savings ``miss_cost - warm_cost >= $0.05`` so a cheap,
unpriced or subscription-included route never pays for one, the refreshes of one run never cost
more in total than the single miss they prevent, and a one-hour age limit bounds the rest. That
total is the estimated billed cost of the refreshes actually sent (from their own usage), and a
refresh whose usage shows no cache read or any cache write ends the run: the entry is gone or was
re-written, so further refreshes would pay write prices for nothing. Every stop reason is logged
at INFO (once per change), so warming never ends silently.

Requests that carry ``cache_control`` markers are warmed on the Anthropic Messages wire and on
the Chat Completions wire (Nous Portal, OpenRouter and other gateways in front of Claude). A
request whose thinking is budget-based is skipped because ``max_tokens=1`` would change
``budget_tokens``, which Anthropic keys the message cache on (and which it rejects outright).

Only a native Anthropic route is trusted to keep the ``1h`` tier: a gateway may drop or shorten
it (on Nous Portal a ``ttl: "1h"`` entry was measured gone after 7 minutes), so every other route
is warmed on the 5-minute schedule whatever TTL the request asked for.
"""

from __future__ import annotations

import contextvars
import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Warming never continues past this long after the real request that started it.
MAX_WARMING_AGE_S = 60 * 60
#: A refresh is sent only when it is expected to save at least this many dollars.
MINIMUM_EXPECTED_SAVINGS_USD = 0.05
#: Anthropic cache tiers, as ``agent._cache_ttl`` spells them.
TTL_SECONDS = {"5m": 300, "1h": 3600}
#: Wires the warmer can replay a request on.
WARMABLE_API_MODES = frozenset({"anthropic_messages", "chat_completions"})
#: Replays are one HTTP round trip; never let a hung socket pin the timer thread.
WARM_REQUEST_TIMEOUT_S = 60.0


def cache_warming_delay_s(ttl_s: float) -> Optional[float]:
    """Refresh at 90% of the TTL while keeping at least ten seconds of margin; ``None`` when
    the TTL is too short to leave any margin."""
    if ttl_s <= 10:
        return None
    return max(1.0, min(ttl_s * 0.9, ttl_s - 10))


def warm_ttl_seconds(agent: Any) -> Optional[float]:
    """Seconds the agent's cache entry is trusted to live: the requested tier on native
    Anthropic, 5 minutes on every other route (gateways may drop the ``1h`` tier).

    The warmer's only lifetime source: the shared prompt-cache clock replaces this body."""
    ttl_s = TTL_SECONDS.get(getattr(agent, "_cache_ttl", None) or "")
    if ttl_s is None:
        return None
    native = getattr(agent, "api_mode", None) == "anthropic_messages" and getattr(agent, "provider", "") == "anthropic"
    return ttl_s if native else min(ttl_s, TTL_SECONDS["5m"])


def has_cache_markers(api_kwargs: dict[str, Any]) -> bool:
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


def is_replayable(api_kwargs: dict[str, Any]) -> bool:
    """Whether replaying with ``max_tokens=1`` leaves the cache entry untouched.

    Budget-based thinking (``{"type": "enabled", "budget_tokens": N}``) requires
    ``max_tokens > budget_tokens`` and Anthropic keys the message cache on the budget, so the
    replay is both rejected and useless. Adaptive thinking and no thinking replay fine.
    """
    thinking = api_kwargs.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled" and "budget_tokens" in thinking:
        return False
    # Chat Completions gateways turn ``reasoning`` into a thinking budget for Claude models that
    # predate adaptive thinking, so a one-token replay would change (or be refused for) that budget.
    reasoning = (api_kwargs.get("extra_body") or {}).get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("enabled", True) and reasoning.get("effort") != "none":
        from agent.anthropic_adapter import _supports_adaptive_thinking

        return _supports_adaptive_thinking(str(api_kwargs.get("model") or ""))
    return True


@dataclass(frozen=True)
class CacheWarmingDecision:
    """Inputs and outcome of one warm-or-stop decision."""

    warm_cost: float
    miss_cost: float
    expected_savings: float
    economics_available: bool
    action: str  # "warm" | "stop"
    within_miss: bool = True

    @property
    def stop_reason(self) -> str:
        if not self.economics_available:
            return "cache economics unavailable"
        if not self.within_miss:
            return "refreshes would cost more than the cache write they prevent"
        return "expected savings below threshold"


def _per_token(rate: Optional[Decimal]) -> float:
    return float(rate) / 1_000_000 if rate is not None else 0.0


def evaluate_economics(
    pricing: Any, cached_tokens: int, uncached_tokens: int = 0, *,
    spent_usd: float = 0.0, observed_warm_usd: float = 0.0,
) -> CacheWarmingDecision:
    """Price one refresh against the miss it prevents using a ``PricingEntry``-shaped object
    (``input/output/cache_read/cache_write_cost_per_million``).

    ``cached_tokens`` is the marked prefix the entry holds (read on a refresh, re-written on a
    miss); ``uncached_tokens`` is the tail past the last marker, billed at the input rate either
    way, so it adds to the refresh's cost and nothing to the miss it prevents. A refresh is
    reserved at no less than ``observed_warm_usd`` (what the last one was actually billed), and
    ``spent_usd`` (what the run's refreshes were billed so far) plus that reservation may not
    exceed the miss: past it, warming is dearer than letting the entry expire once."""
    read = _per_token(getattr(pricing, "cache_read_cost_per_million", None))
    write = _per_token(getattr(pricing, "cache_write_cost_per_million", None))
    inp = _per_token(getattr(pricing, "input_cost_per_million", None))
    out = _per_token(getattr(pricing, "output_cost_per_million", None))
    tail_cost = uncached_tokens * inp
    hit_cost = cached_tokens * read
    miss_full = cached_tokens * (write if write > 0 else inp)
    warm_cost = max(hit_cost + tail_cost + out, observed_warm_usd)
    miss_cost = max(0.0, miss_full - hit_cost)
    economics_available = cached_tokens > 0 and (hit_cost > 0 or miss_full > 0)
    expected = miss_cost - warm_cost
    within_miss = spent_usd + warm_cost <= miss_cost
    action = "warm" if economics_available and within_miss and expected >= MINIMUM_EXPECTED_SAVINGS_USD else "stop"
    return CacheWarmingDecision(
        warm_cost=warm_cost, miss_cost=miss_cost, expected_savings=expected,
        economics_available=economics_available, action=action, within_miss=within_miss,
    )


@dataclass
class _Run:
    api_kwargs: dict[str, Any]
    client: Any
    model: str
    cached_tokens: int
    uncached_tokens: int
    ttl_s: float
    delay_s: float
    is_current: Callable[[], bool]
    started_at: float
    api_mode: str = "anthropic_messages"
    anchor_at: float = 0.0
    next_warm_at: float = 0.0
    refresh_deadline_at: float = 0.0
    timer: Optional[threading.Timer] = None
    refreshing: bool = False
    warm_count: int = 0
    spent_usd: float = 0.0
    last_warm_usd: float = 0.0
    #: The session and billing route that sent the request: a refresh still in flight when
    #: compression rotates the session is booked in state.db where it was sent.
    billing: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class PromptCacheWarmer:
    """Keeps one prompt-cache entry alive per agent by replaying its request with a one-token
    output cap before the entry expires. ``start`` replaces any previous run; warm requests never
    extend the fixed safety windows."""

    def __init__(self, agent: Any):
        self.agent = agent
        # Guards ``_run`` ownership: the timer thread and the loop both replace or stop it.
        self._lock = threading.RLock()
        self._run: Optional[_Run] = None
        self._inactive_reason = "waiting for first request"
        self._logged_reason: Optional[str] = None
        self._last_decision: Optional[CacheWarmingDecision] = None

    # -- read side ------------------------------------------------------------------------------
    @property
    def status(self) -> dict[str, Any]:
        """``{"state": inactive|scheduled|refreshing, "reason"?, "next_warm_at"?, "decision"?}``."""
        run = self._run
        if run is None:
            return {"state": "inactive", "reason": self._inactive_reason, "decision": self._last_decision}
        if not run.is_current():
            return {"state": "inactive", "reason": "conversation context changed"}
        return {
            "state": "refreshing" if run.refreshing else "scheduled",
            "next_warm_at": run.next_warm_at, "warm_count": run.warm_count,
            "decision": self._evaluate(run),
        }

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self, api_kwargs: dict[str, Any], response: Any, *, sent_at: Optional[float] = None) -> Optional[str]:
        """Arm a refresh for the request that just produced ``response``. Returns the reason
        nothing was armed, ``None`` when a timer is scheduled."""
        self._clear_run()
        agent = self.agent
        if getattr(agent, "_skip_prompt_cache_warming", False):
            return self._stop("the parent agent keeps this prefix warm")
        api_mode = getattr(agent, "api_mode", None)
        if api_mode not in WARMABLE_API_MODES:
            return self._stop("only Anthropic Messages and Chat Completions requests are warmed")
        if not getattr(agent, "_use_prompt_caching", False) or getattr(agent, "_cache_disabled", False):
            return self._stop("prompt caching disabled")
        if api_mode == "chat_completions" and "claude" not in str(getattr(agent, "model", "")).lower():
            return self._stop("only Claude requests are warmed on Chat Completions")
        if "_moa_prepared_request" in api_kwargs or getattr(agent, "provider", "") == "moa":
            return self._stop("MoA requests are not replayable")
        if getattr(agent, "_interrupt_requested", False):
            return self._stop("turn interrupted")
        ttl_s = warm_ttl_seconds(agent)
        if ttl_s is None:
            return self._stop("cache lifetime unavailable")
        if not has_cache_markers(api_kwargs):
            return self._stop("request carries no cache markers")
        if not is_replayable(api_kwargs):
            return self._stop("request cannot be replayed safely (budget thinking)")
        delay_s = cache_warming_delay_s(ttl_s)
        if delay_s is None:
            return self._stop("cache lifetime unavailable")
        cached_tokens, uncached_tokens = self._prompt_split(response)
        if cached_tokens + uncached_tokens <= 0:
            return self._stop("prompt size unavailable")
        client = getattr(agent, "_anthropic_client" if api_mode == "anthropic_messages" else "client", None)
        if client is None:
            return self._stop("no provider client")
        # Economics before the deep copy: an unpriced/cheap model never pays for the copy either.
        decision = evaluate_economics(self._pricing(), cached_tokens, uncached_tokens)
        self._last_decision = decision
        if decision.action != "warm":
            return self._stop(decision.stop_reason)
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
            cached_tokens=cached_tokens, uncached_tokens=uncached_tokens, ttl_s=ttl_s,
            delay_s=delay_s, is_current=_is_current, started_at=time.time(), api_mode=api_mode,
            billing=self._billing_snapshot(),
        )
        # The entry's lifetime runs from when the provider served the request, not from when
        # the (possibly long) response finished: the first refresh is due that much sooner.
        run.anchor_at = sent_at if sent_at is not None else run.started_at
        with self._lock:
            self._run = run
            self._schedule(run)
            return None if self._run is run else self._inactive_reason

    def cancel(self, reason: str = "inactive") -> None:
        self._stop(reason, lifecycle=True)

    # -- internals ------------------------------------------------------------------------------
    def _clear_run(self) -> Optional[_Run]:
        with self._lock:
            run, self._run = self._run, None
        if run is None:
            return None
        with run.lock:
            if run.timer is not None:
                run.timer.cancel()
                run.timer = None
        return run

    def _stop(self, reason: str, *, lifecycle: bool = False) -> str:
        """Log why warming stopped at INFO, once per change of reason (each request of a
        non-warmable route repeats its refusal). A ``lifecycle`` stop (the loop's own next
        request or turn end) is routine and logs at INFO only when the run sent refreshes."""
        with self._lock:
            run = self._clear_run()
            sent = run.warm_count if run else 0
            routine = lifecycle and not sent
            if not routine and reason != self._logged_reason:
                logger.info("%sprompt cache warming stopped: %s (refreshes=%d, spent=$%.4f)",
                            self._prefix(), reason, sent, run.spent_usd if run else 0.0)
                self._logged_reason = reason
            elif routine:
                logger.debug("%sprompt cache warming ended: %s", self._prefix(), reason)
            self._inactive_reason = reason
        return reason

    def _stop_if_current(self, run: _Run, reason: str) -> None:
        """Timer-thread stop: never cancels a newer run the loop armed during the warm call."""
        with self._lock:
            if self._run is run:
                self._stop(reason)

    def _schedule(self, run: _Run) -> None:
        now = time.time()
        run.next_warm_at = (run.anchor_at or now) + run.delay_s
        run.anchor_at = 0.0  # later refreshes count from the refresh itself
        if run.next_warm_at - run.delay_s + run.ttl_s <= now:
            self._stop("cache entry already expired")
            return
        # A timer can fire late after sleep or a blocked interpreter. Keep half of the planned
        # pre-expiry margin for that: a late refresh is a full-price write, not a warm.
        run.refresh_deadline_at = run.next_warm_at + (run.ttl_s - run.delay_s) / 2
        deadline = run.started_at + MAX_WARMING_AGE_S
        if run.next_warm_at > deadline or now >= deadline:
            self._stop("one-hour safety limit reached")
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
        with self._lock:  # check ownership and stop in one step: never cancel a newer run
            if self._run is not run:
                return False
            reason = None
            if not run.is_current():
                reason = "conversation context changed"
            elif time.time() > run.refresh_deadline_at:
                reason = "cache refresh deadline missed"
            if reason is not None:
                self._stop(reason)
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
            self._stop_if_current(run, decision.stop_reason)
            return
        run.refreshing = True
        sent_at = time.time()
        try:
            message = self._send(run)
        except Exception as exc:  # health: allow BLE001 -- warming must never disturb the live turn; logged
            logger.info("%sprompt cache warm failed: %s", self._prefix(), exc, exc_info=True)
            self._stop_if_current(run, f"warm request failed: {type(exc).__name__}")
            return
        finally:
            run.refreshing = False
        run.warm_count += 1
        miss = self._record(run, message, decision)
        if miss is not None:
            self._stop_if_current(run, miss)
            return
        with self._lock:
            if self._run is run:
                run.anchor_at = sent_at
                self._schedule(run)

    def _send(self, run: _Run) -> Any:
        client = run.client
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            client = with_options(max_retries=0, timeout=WARM_REQUEST_TIMEOUT_S)
        kwargs = dict(run.api_kwargs)
        kwargs.pop("stream", None)
        kwargs.pop("timeout", None)  # the warm request keeps its own short timeout
        if run.api_mode == "chat_completions":
            kwargs.pop("stream_options", None)
            kwargs["max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"] = 1
            # A byte-identical replay inside OpenRouter's response-cache TTL would be answered
            # from that cache and never reach (or refresh) the prompt cache upstream.
            kwargs["extra_headers"] = {**(kwargs.get("extra_headers") or {}), "X-OpenRouter-Cache": "false"}
            return client.chat.completions.create(**kwargs)
        from agent.anthropic_adapter import create_anthropic_message

        kwargs["max_tokens"] = 1
        return create_anthropic_message(client, kwargs, log_prefix=self._prefix(), prefer_stream=False)

    def _record(self, run: _Run, message: Any, decision: CacheWarmingDecision) -> Optional[str]:
        """Fold the refresh's tokens and dollars into the session totals (never into the
        compressor or usage anchors: a one-token replay is not a conversation turn) and into the
        run's spend. Returns why warming must stop when the refresh did not prove a cache hit at
        a known price, ``None`` when it did."""
        agent, b = self.agent, run.billing
        usage = getattr(message, "usage", None)
        if not usage:
            return "cache refresh returned no usage"
        try:
            from agent.usage_pricing import estimate_usage_cost, normalize_usage

            canonical = normalize_usage(usage, provider=b["provider"], api_mode=b["api_mode"])
            cost = estimate_usage_cost(
                b["model"], canonical, provider=b["provider"], base_url=b["base_url"], api_key=b["api_key"],
            )
            amount = float(cost.amount_usd) if cost.amount_usd is not None else None
            if amount is not None:
                run.spent_usd += amount
                run.last_warm_usd = amount
            # Compression keeps the running counters (a refresh still adds to them); only
            # ``reset_session_state`` zeroes them, and it zeroes ``session_api_calls`` with them.
            if getattr(agent, "session_api_calls", 0) >= b["api_calls"]:
                self._add_to_session_totals(canonical, amount)
            db = getattr(agent, "_session_db", None)
            if db is not None and b["session_id"]:
                db.queue_token_counts(
                    b["session_id"], source=b["source"],
                    input_tokens=canonical.input_tokens, output_tokens=canonical.output_tokens,
                    cache_read_tokens=canonical.cache_read_tokens,
                    cache_write_tokens=canonical.cache_write_tokens,
                    reasoning_tokens=canonical.reasoning_tokens, estimated_cost_usd=amount,
                    cost_status=cost.status, cost_source=cost.source,
                    billing_provider=b["provider"], billing_base_url=b["base_url"],
                    billing_mode="subscription_included" if cost.status == "included" else None,
                    model=b["model"], api_call_count=1, task="cache_warm",
                )
            logger.info(
                "%sprompt cache refresh sent (#%d): cache_read=%d write=%d cost=%s "
                "expected_savings=$%.3f next=%s", self._prefix(), run.warm_count,
                canonical.cache_read_tokens, canonical.cache_write_tokens,
                f"${amount:.4f}" if amount is not None else "?", decision.expected_savings,
                time.strftime("%H:%M:%S", time.localtime(time.time() + run.delay_s)),
            )
        except Exception:  # health: allow BLE001 -- unaccounted means stop warming; logged
            logger.debug("prompt cache warm accounting failed", exc_info=True)
            return "cache refresh usage unreadable"
        if canonical.cache_read_tokens <= 0 or canonical.cache_write_tokens > 0:
            return (f"cache refresh missed (read={canonical.cache_read_tokens}, "
                    f"write={canonical.cache_write_tokens}, cost={f'${amount:.4f}' if amount is not None else '?'})")
        if amount is None:
            return "cache refresh cost unavailable"
        return None

    def _add_to_session_totals(self, canonical: Any, amount: Optional[float]) -> None:
        # Not session_api_calls: that counter is the conversation's request sequence (and the
        # warmer's own currency token); a replay is accounted, not a turn.
        agent = self.agent
        agent.session_cache_warm_calls = getattr(agent, "session_cache_warm_calls", 0) + 1
        agent.session_prompt_tokens += canonical.prompt_tokens
        agent.session_completion_tokens += canonical.output_tokens
        agent.session_total_tokens += canonical.total_tokens
        agent.session_input_tokens += canonical.input_tokens
        agent.session_output_tokens += canonical.output_tokens
        agent.session_cache_read_tokens += canonical.cache_read_tokens
        agent.session_cache_write_tokens += canonical.cache_write_tokens
        if amount is not None:
            agent.session_estimated_cost_usd += amount

    def _evaluate(self, run: _Run) -> CacheWarmingDecision:
        return evaluate_economics(
            self._pricing(), run.cached_tokens, run.uncached_tokens,
            spent_usd=run.spent_usd, observed_warm_usd=run.last_warm_usd,
        )

    def _pricing(self) -> Any:
        agent = self.agent
        try:
            from agent.usage_pricing import get_pricing_entry

            return get_pricing_entry(
                agent.model, provider=agent.provider, base_url=agent.base_url,
                api_key=getattr(agent, "api_key", ""),
            )
        except Exception:  # health: allow BLE001 -- unpriced means never warm; logged
            logger.debug("prompt cache warm pricing unavailable", exc_info=True)
            return None

    def _prompt_split(self, response: Any) -> tuple[int, int]:
        """``(cached prefix, uncached tail)`` tokens of the request that produced ``response``:
        the prefix is what it read from or wrote to the cache, the tail its plain input."""
        usage = getattr(response, "usage", None)
        if not usage:
            return 0, 0
        try:
            from agent.usage_pricing import normalize_usage

            u = normalize_usage(usage, provider=self.agent.provider, api_mode=self.agent.api_mode)
            return int(u.cache_read_tokens + u.cache_write_tokens), int(u.input_tokens)
        except Exception:  # health: allow BLE001 -- unknown size means never warm; logged
            logger.debug("prompt cache warm size unavailable", exc_info=True)
            return 0, 0

    def _billing_snapshot(self) -> dict[str, Any]:
        agent = self.agent
        from agent.turn_usage import _agent_session_source

        return {
            "session_id": getattr(agent, "session_id", None), "source": _agent_session_source(agent),
            "provider": agent.provider, "base_url": agent.base_url, "model": agent.model,
            "api_mode": agent.api_mode, "api_key": getattr(agent, "api_key", ""),
            "api_calls": getattr(agent, "session_api_calls", 0),
        }

    @staticmethod
    def _replay_kwargs(api_kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = {k: v for k, v in api_kwargs.items() if k not in ("stream", "_moa_prepared_request")}
        for key in ("system", "tools", "messages", "thinking", "tool_choice", "metadata", "extra_body"):
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


def start_prompt_cache_warming(
    agent: Any, api_kwargs: Any, response: Any, *, sent_at: Optional[float] = None,
) -> None:
    """Called once per successful provider response with the request that produced it;
    ``sent_at`` is when that request went out (the cache entry's lifetime starts there)."""
    if not isinstance(api_kwargs, dict):
        return
    try:
        get_prompt_cache_warmer(agent).start(api_kwargs, response, sent_at=sent_at)
    except Exception:
        logger.debug("prompt cache warming did not arm", exc_info=True)


def cancel_prompt_cache_warming(agent: Any, reason: str = "inactive") -> None:
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is not None:
        warmer.cancel(reason)

