"""The prompt-cache lifecycle: one controller per chat, driven by the shared prompt-cache clock.

When a chat's cache is about to expire the controller makes ONE decision, in this order:

1. WARM: re-send the last request with a tiny output cap at 5/6 of the entry's TTL (port of
   earendil-works/pi#9668): one cache READ keeps the whole prefix warm for another TTL. Up to
   :data:`WARM_BUDGET` keep-alives per budget, while a tool runs and, for chats a person types into,
   while the chat is idle. A budget belongs to one USER message: the tool-round requests inside its
   turn do not restart it, so a four-hour tool run gets three keep-alives in total. Machine-paced
   sessions (subagents, cron, one-shots, /bg, /btw, the API server) warm only while a tool runs. A
   keep-alive that misses (comes back mostly written, not read) ends warming at once and the budget
   counts as spent: step 2 or 3 is decided right then.
2. COMPACT: once the budget is spent, an idle human chat compacts IF its window is at least
   :data:`MIN_COMPACT_WINDOW_SECONDS` (a 1h or 30m tier, never a 5m one), its context is at least half
   of the compaction threshold, and it grew since its last compaction; at most one compaction per
   idle stretch. The summary is a fresh auxiliary request, so whether the old prefix is still warm
   does not change its cost and is not checked. Nothing is sent after it until the next user message.
   The surface that owns the chat (CLI input loop, TUI ticker, gateway watcher) runs the compaction
   through the normal pipeline, polling :func:`claim_compaction`. A budget ends at whichever comes
   first: its keep-alives, or the owner releasing the agent (gateway idle/LRU eviction, TUI idle reap);
   at release the owner asks :func:`claim_compaction_at_release` and, when due, compacts first.
3. EXPIRE + OFFLOAD: otherwise the entry dies, and the owner offloads old tool output right then
   (:func:`claim_offload`, the same cold offload as a return after expiry), so the chat is already
   small when the person comes back. A release past the window offloads before it releases.
4. COLD RETURN lives in ``agent.turn_proactive_prune``: the next request after a proven or declared
   expiry (or, on a route with an undocumented cache, an idle hour) offloads old tool results first.

An unknown or assumed clock means nothing acts. A keep-alive is sent only when its read costs less
than the rewrite it prevents, the budget's keep-alives together cost no more than one rewrite, the
cache prices are known, and the cached prefix clears the smallest cacheable size. A keep-alive is
bound to the prompt and route it replays (``prompt_cache_clock.context_witness``), admitted under the
lock ``cancel`` takes, and fires the auxiliary call hooks (``aux_task="cache_warm"``). Every stop
reason is logged at INFO (once per change).

Requests are replayed on the Anthropic Messages wire and on the Chat Completions wire (Nous Portal,
OpenRouter and other gateways): Claude by its ``cache_control`` markers, every other model as an
OpenAI-spec automatic cache the clock declares at 30 minutes, renewed by a read). A request whose thinking is budget-based is skipped because a one-token cap would
change ``budget_tokens``.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import logging
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Keep-alives per budget; a new user message starts one (tool rounds inside its turn do not).
WARM_BUDGET = 3
#: No entry exists below this prefix: the smallest documented minimum (Anthropic 1,024 for Sonnet/Opus,
#: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching ; OpenAI 1,024,
#: https://developers.openai.com/api/docs/guides/prompt-caching ; Qwen 1,024,
#: https://www.alibabacloud.com/help/en/model-studio/context-cache).
MIN_CACHEABLE_PREFIX_TOKENS = 1024
#: Shortest cache window a chat is compacted before: below it, ordinary pauses outlast the window.
MIN_COMPACT_WINDOW_SECONDS = 30 * 60
#: A chat is compacted only once its context is at least this share of the compaction threshold.
COMPACT_CONTEXT_FRACTION = 0.5
#: How often the surface owners poll :func:`claim_compaction`.
CHECK_INTERVAL_SECONDS = 30.0
#: Telemetry trigger name for cache-edge compactions.
TRIGGER = "cache_expiry"
#: Wires the warmer can replay a request on.
WARMABLE_API_MODES = frozenset({"anthropic_messages", "chat_completions"})
#: Replays are one HTTP round trip; never let a hung socket pin the timer thread.
WARM_REQUEST_TIMEOUT_S = 60.0


_OPENAI_REASONING_MODEL = re.compile(r"(^|/)(gpt-[5-9]|o[1-9])")


def cache_warming_delay_s(ttl_s: float) -> Optional[float]:
    """Refresh at 5/6 of the TTL (5m -> 250 s, 30m -> 25 min, 1h -> 50 min) while keeping at least ten
    seconds of margin; ``None`` when the TTL is too short to leave any margin. The same moment decides
    compaction at the end of a budget."""
    if ttl_s <= 10:
        return None
    return max(1.0, min(ttl_s * 5 / 6, ttl_s - 10))


def guaranteed_window(agent: Any) -> Optional[tuple[float, float]]:
    """``(guaranteed-warm deadline, tier seconds)`` from the shared prompt-cache clock, or ``None``
    when the clock guarantees nothing now (no window, unknown, assumed, or already past)."""
    from agent.prompt_cache_clock import cache_certainly_warm, cache_window
    from agent.prompt_caching import TTL_SECONDS

    window = cache_window(agent)
    ttl_s = TTL_SECONDS.get(getattr(window, "tier", None) or "")
    if ttl_s is None or not cache_certainly_warm(window):
        return None
    return float(window.guaranteed_until), float(ttl_s)


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
    model = str(api_kwargs.get("model") or "")
    if ("claude" in model.lower() and isinstance(reasoning, dict) and reasoning.get("enabled", True)
            and reasoning.get("effort") != "none"):
        from agent.anthropic_adapter import _supports_adaptive_thinking

        return _supports_adaptive_thinking(model)
    return True


def replay_output_cap(model: str) -> int:
    """The smallest output cap the replay may carry: OpenAI's GPT-5 family and o-series refuse an
    output cap below 16 (``max_output_tokens`` minimum, also behind Chat Completions gateways that
    translate to Responses: Nous Portal answers 400 "Expected a value >= 16" to a cap of 1)."""
    return 16 if _OPENAI_REASONING_MODEL.search((model or "").lower()) else 1


#: ``extra_body`` keys that raise the output cap past ``max_tokens`` (Gemini's long-decode budget, or a
#: gateway's own cap field merged into the body): output-only, so the replay drops them.
_EXTRA_BODY_OUTPUT_CAPS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def bounded_chat_replay(api_kwargs: dict[str, Any], model: str) -> dict[str, Any]:
    """The Chat Completions keep-alive: the same prompt with the smallest output cap ``model`` accepts
    (:func:`replay_output_cap`), not streamed, and no ``extra_body`` field that could lift that cap."""
    kwargs = {k: v for k, v in api_kwargs.items() if k not in ("stream", "stream_options")}
    kwargs["max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"] = replay_output_cap(model)
    extra = kwargs.get("extra_body")
    if isinstance(extra, dict) and any(k in extra for k in _EXTRA_BODY_OUTPUT_CAPS):
        kwargs["extra_body"] = {k: v for k, v in extra.items() if k not in _EXTRA_BODY_OUTPUT_CAPS}
    return kwargs


def _sdk_client_types(api_mode: str) -> tuple[type, ...]:
    """The SDK clients that send a request's kwargs as given on ``api_mode``'s wire."""
    if api_mode == "chat_completions":
        from openai import OpenAI

        return (OpenAI,)
    try:
        import anthropic
    except ImportError:  # no SDK installed: no Anthropic client can exist either
        return ()
    return (anthropic.Anthropic, anthropic.AnthropicBedrock, anthropic.AnthropicVertex)


def replay_cap_honoured(client: Any, api_kwargs: dict[str, Any], model: str, api_mode: str) -> bool:
    """Whether ``client`` sends the bounded replay's output cap as given, so a keep-alive costs at most
    that many output tokens. The OpenAI and Anthropic SDKs send the kwargs as they are. Any other client
    must report the cap it would really send through ``effective_output_cap(**kwargs)``: the native
    Gemini clients translate to another wire and lift the cap to 65,535 tokens when thinking is on. A
    client that reports nothing (ACP subprocess clients, an unknown plugin client) or would raise the
    cap is never warmed."""
    replay = bounded_chat_replay(api_kwargs, model) if api_mode == "chat_completions" else {**api_kwargs, "max_tokens": 1}
    cap = replay.get("max_completion_tokens", replay.get("max_tokens"))
    effective = getattr(client, "effective_output_cap", None)
    if callable(effective):
        return effective(**replay) <= cap
    return isinstance(client, _sdk_client_types(api_mode))


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
        return "a refresh would cost more than the cache write it prevents"


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
    exceed the miss: past it, warming is dearer than letting the entry expire once. There is no
    absolute dollar floor: a cheap route warms whenever the read is cheaper than the rewrite; a prefix
    below :data:`MIN_CACHEABLE_PREFIX_TOKENS` has no entry to keep."""
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
    economics_available = economics_available and cached_tokens >= MIN_CACHEABLE_PREFIX_TOKENS
    action = "warm" if economics_available and within_miss and expected > 0 else "stop"
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
    deadline_at: float  # the clock's guaranteed-warm end for the entry this run keeps alive
    is_current: Callable[[], bool]
    started_at: float
    #: ``prompt_cache_clock.context_witness`` when armed: the prompt (session, generation, rewrites)
    #: and route the replay carries. A refresh is sent, and its reply folded into the clock, only
    #: while the session still matches it; a compaction or route switch meanwhile makes it stale.
    witness: tuple = ()
    api_mode: str = "anthropic_messages"
    next_warm_at: float = 0.0
    refresh_deadline_at: float = 0.0
    timer: Optional[threading.Timer] = None
    refreshing: bool = False
    #: Set under the warmer lock when the run is cleared; checked under the same lock right
    #: before a request is admitted, so a cancel that returned never sees a later send.
    cancelled: bool = False
    #: The user turn's budget this run draws on; a run that replaced it (the next tool round) shares it.
    budget: _Budget = field(default_factory=lambda: _Budget(None))
    #: The session and billing route that sent the request: a refresh still in flight when
    #: compression rotates the session is booked in state.db where it was sent.
    billing: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: A chat a person types into keeps warming after its turn ends; a machine session does not.
    idle_ok: bool = False

    @property
    def warm_count(self) -> int:
        return self.budget.used


@dataclass
class _Budget:
    """One USER message's keep-alives. Every run its tool rounds arm draws on the same object, so a run
    replaced mid-flight still books its keep-alive here: each is reserved (count and estimated cost) at
    admission and reconciled with its billed cost when its reply lands, whichever run is current then.
    Changed only under the warmer lock."""

    turn: Any
    used: int = 0
    spent_usd: float = 0.0
    last_warm_usd: float = 0.0
    #: Set when the budget ended (keep-alives spent, or a miss): no more keep-alives this user turn.
    ended: bool = False


@dataclass(frozen=True)
class _Spent:
    """A warm budget that ran out with the entry still warm: the moment step 2 decides."""

    due_at: float  # when the next keep-alive would have gone out
    deadline_at: float  # the clock's guaranteed-warm end
    tier_s: float
    witness: tuple


class PromptCacheWarmer:
    """The per-chat lifecycle controller: keeps one prompt-cache entry alive by replaying its
    request with a one-token output cap before the entry expires (``WARM_BUDGET`` per budget), then
    hands the spent budget to step 2 (:func:`compaction_due`). ``start`` replaces any previous run."""

    def __init__(self, agent: Any):
        self.agent = agent
        # Guards ``_run`` ownership: the timer thread and the loop both replace or stop it.
        self._lock = threading.RLock()
        self._run: Optional[_Run] = None
        self._inactive_reason = "waiting for first request"
        self._logged_reason: Optional[str] = None
        self._last_decision: Optional[CacheWarmingDecision] = None
        #: Set when a budget ended (keep-alives spent, or a miss) and carried to each later prompt of the
        #: same user turn; cleared by a new user message or a claim, never by a run ending. At most one
        #: compaction (``_compacted``) and one offload (``_offloaded``) per stretch.
        self._spent: Optional[_Spent] = None
        self._compacted = False
        self._offloaded = False
        #: The current user message's budget: tool-round requests inside one turn continue it, only a new
        #: user message (``_user_turn_count``) restarts it.
        self._budget: Optional[_Budget] = None
        #: Whether the armed request carries ``cache_control`` markers (Anthropic spec: a miss is a priced
        #: write) or not (OpenAI spec: a miss re-reads the prefix at the input rate).
        self._marked = True

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
    def start(self, api_kwargs: dict[str, Any], response: Any) -> Optional[str]:
        """Arm keep-alives for the request that just produced ``response``. Returns the reason nothing
        was armed, ``None`` when a timer is scheduled. A new user message opens a new budget and idle
        stretch; a tool-round request inside the same turn carries the budget it already used."""
        self._clear_run()
        agent = self.agent
        turn = getattr(agent, "_user_turn_count", None)
        with self._lock:
            budget = self._budget
            if budget is None or budget.turn != turn:
                budget = self._budget = _Budget(turn)
                self._spent, self._compacted, self._offloaded = None, False, False
            if budget.ended or budget.used >= WARM_BUDGET:
                self._carry_ended_budget()
                return self._stop("warm budget spent this turn")
            self._spent = None
        if getattr(agent, "_skip_prompt_cache_warming", False):
            return self._stop("the parent agent keeps this prefix warm")
        api_mode = getattr(agent, "api_mode", None)
        if api_mode not in WARMABLE_API_MODES:
            return self._stop("only Anthropic Messages and Chat Completions requests are warmed")
        # Marker caching is Anthropic-spec only; an OpenAI-spec route caches automatically and proves it
        # with the reply's cached tokens (no read or write folded = no window below).
        if getattr(agent, "_cache_disabled", False) or (
                api_mode == "anthropic_messages" and not getattr(agent, "_use_prompt_caching", False)):
            return self._stop("prompt caching disabled")
        if "_moa_prepared_request" in api_kwargs or getattr(agent, "provider", "") == "moa":
            return self._stop("MoA requests are not replayable")
        if getattr(agent, "_interrupt_requested", False):
            return self._stop("turn interrupted")
        if api_mode == "anthropic_messages" and not has_cache_markers(api_kwargs):
            return self._stop("request carries no cache markers")
        if not is_replayable(api_kwargs):
            return self._stop("request cannot be replayed safely (budget thinking)")
        # An OpenAI-spec automatic cache bills a miss at the plain input rate (no write surcharge).
        self._marked = has_cache_markers(api_kwargs)
        window = guaranteed_window(agent)
        if window is None or cache_warming_delay_s(window[1]) is None:
            return self._stop("cache window not guaranteed warm")
        cached_tokens, uncached_tokens = self._prompt_split(response)
        if cached_tokens + uncached_tokens <= 0:
            return self._stop("prompt size unavailable")
        client = getattr(agent, "_anthropic_client" if api_mode == "anthropic_messages" else "client", None)
        if client is None:
            return self._stop("no provider client")
        if not replay_cap_honoured(client, api_kwargs, getattr(agent, "model", ""), api_mode):
            return self._stop("the client cannot bound the replay's output cap")
        # Economics before the deep copy: an unpriced/cheap model never pays for the copy either.
        decision = self._decide(cached_tokens, uncached_tokens, spent_usd=budget.spent_usd,
                                observed_warm_usd=budget.last_warm_usd)
        self._last_decision = decision
        if decision.action != "warm":
            return self._stop(decision.stop_reason)
        from agent.prompt_cache_clock import context_witness

        api_calls = getattr(agent, "session_api_calls", 0)
        model = getattr(agent, "model", "")
        epoch = getattr(agent, "_session_reset_epoch", 0)

        def _is_current() -> bool:
            # The epoch, not the counter alone: a reset zeroes session_api_calls, which the new
            # session's own requests then raise back to the armed value.
            return (
                getattr(agent, "_session_reset_epoch", 0) == epoch
                and getattr(agent, "session_api_calls", None) == api_calls
                and getattr(agent, "model", None) == model
                and not getattr(agent, "_interrupt_requested", False)
            )

        # The loop mutates message dicts in place between requests (cache-marker redecoration,
        # tool-result eviction); the replay must send exactly what wrote the entry.
        run = _Run(
            api_kwargs=self._replay_kwargs(api_kwargs), client=client, model=model,
            cached_tokens=cached_tokens, uncached_tokens=uncached_tokens, ttl_s=window[1],
            deadline_at=window[0], is_current=_is_current, started_at=time.time(), api_mode=api_mode,
            witness=context_witness(agent), billing=self._billing_snapshot(),
            idle_ok=human_paced(agent), budget=budget,
        )
        with self._lock:
            self._run = run
            self._schedule(run)
            return None if self._run is run else self._inactive_reason

    def end_turn(self) -> None:
        """The turn settled: a machine session stops warming (its next request is a guess nobody
        is about to make); a chat a person types into keeps its budget and warms while idle. An ended
        budget stays ended: step 2/3 still decides on it."""
        run = self._run
        if run is None or not run.idle_ok:
            self.cancel("agent run settled")

    def cancel(self, reason: str = "inactive") -> None:
        self._stop(reason, lifecycle=True)

    def _carry_ended_budget(self) -> None:
        """The budget ended earlier in this user turn and a later reply moved the prompt on: step 2/3
        decides on the prompt the turn now holds, when its next keep-alive would have gone out.
        Caller holds ``self._lock``."""
        from agent.prompt_cache_clock import context_witness

        window = guaranteed_window(self.agent)
        delay = cache_warming_delay_s(window[1]) if window is not None else None
        if window is None or delay is None:
            return
        deadline, tier = window
        self._spent = _Spent(deadline - (tier - delay), deadline, tier, context_witness(self.agent))

    # -- internals ------------------------------------------------------------------------------
    def _clear_run(self) -> Optional[_Run]:
        with self._lock:
            run, self._run = self._run, None
            if run is not None:
                run.cancelled = True
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
                            self._prefix(), reason, sent, run.budget.spent_usd if run else 0.0)
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
        margin = run.ttl_s - (cache_warming_delay_s(run.ttl_s) or run.ttl_s)
        run.next_warm_at = run.deadline_at - margin
        if run.deadline_at <= now:
            self._stop("cache entry already expired")
            return
        # A timer can fire late after sleep or a blocked interpreter. Keep half of the planned
        # pre-expiry margin for that: a late refresh is a full-price write, not a warm.
        run.refresh_deadline_at = run.next_warm_at + margin / 2
        if run.budget.ended or run.budget.used >= WARM_BUDGET:
            # Step 2 decides at the moment the next keep-alive would have gone out.
            self._end_budget(run, "warm budget spent", due_at=run.next_warm_at)
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

    def _end_budget(self, run: _Run, reason: str, *, due_at: float) -> None:
        """The budget is over (keep-alives spent, or one missed): stop, and hand step 2/3 the moment it
        decides at. The user turn's budget ends even when ``run`` was replaced meanwhile (its successor
        is refused at admission). Caller holds ``self._lock`` or owns ``run`` on the timer thread."""
        with self._lock:
            run.budget.ended = True
            if self._run is not run:
                return
            spent = _Spent(due_at, run.deadline_at, run.ttl_s, run.witness)
            self._stop(reason)
            self._spent = spent

    def _stale_reason(self, run: _Run) -> Optional[str]:
        """Why ``run`` no longer replays the live prefix, or None. Caller holds ``self._lock``."""
        from agent.prompt_cache_clock import context_witness

        if not run.is_current() or context_witness(self.agent) != run.witness:
            return "conversation context changed"
        return None

    def _admit(self, run: _Run, decision: CacheWarmingDecision) -> Optional[float]:
        """The one send admission: ownership, cancellation, currency, deadline and the user turn's
        budget checked, the keep-alive reserved against that budget (count and estimated cost), and the
        run marked in flight, in one step under the lock ``cancel`` takes. Returns the reserved cost,
        ``None`` when refused. The network request itself runs outside the lock."""
        with self._lock:
            if self._run is not run or run.cancelled:
                return None
            reason = self._stale_reason(run)
            if reason is None and time.time() > run.refresh_deadline_at:
                reason = "cache refresh deadline missed"
            if reason is not None:
                self._stop(reason)
                return None
            budget = run.budget
            if budget.ended or budget.used >= WARM_BUDGET:
                self._end_budget(run, "warm budget spent", due_at=run.next_warm_at)
                return None
            if budget.spent_usd + decision.warm_cost > decision.miss_cost:
                self._stop("refreshes would cost more than the cache write they prevent")
                return None
            budget.used += 1
            budget.spent_usd += decision.warm_cost
            run.refreshing = True
            return decision.warm_cost

    def _refresh(self, run: _Run) -> None:
        with run.lock:
            run.timer = None
        # Pricing may be slow (a pricing fetch); it decides nothing until admission rechecks.
        decision = self._evaluate(run)
        self._last_decision = decision
        if decision.action != "warm":
            self._stop_if_current(run, decision.stop_reason)
            return
        reserved = self._admit(run, decision)
        if reserved is None:
            return
        sent_at = time.time()
        try:
            message = self._send(run)
        except Exception as exc:  # health: allow BLE001 -- warming must never disturb the live turn; logged
            logger.info("%sprompt cache warm failed: %s", self._prefix(), exc, exc_info=True)
            self._stop_if_current(run, f"warm request failed: {type(exc).__name__}")
            return
        finally:
            run.refreshing = False
        miss = self._record(run, message, decision, sent_at, reserved)
        if miss is not None:
            if miss.startswith("cache refresh missed"):
                # The entry was gone and this send re-wrote it: no more keep-alives, but the budget is
                # over, so step 2 (compact) or step 3 (expire + offload) is decided now.
                window = guaranteed_window(self.agent)
                if window is not None:
                    run.deadline_at, run.ttl_s = window
                self._end_budget(run, miss, due_at=time.time())
            else:
                self._stop_if_current(run, miss)
            return
        with self._lock:
            if self._run is not run:
                return
            stale = self._stale_reason(run)
            if stale is not None:  # its reply was not folded: nothing to schedule from
                self._stop(stale)
                return
            window = guaranteed_window(self.agent)
            if window is None:
                self._stop("cache window not guaranteed warm")
                return
            run.deadline_at, run.ttl_s = window
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
            kwargs = bounded_chat_replay(kwargs, run.model)
            # A byte-identical replay inside OpenRouter's response-cache TTL would be answered
            # from that cache and never reach (or refresh) the prompt cache upstream.
            kwargs["extra_headers"] = {**(kwargs.get("extra_headers") or {}), "X-OpenRouter-Cache": "false"}
            return self._with_aux_hooks(run, client, kwargs, lambda: client.chat.completions.create(**kwargs))
        from agent.anthropic_adapter import create_anthropic_message

        kwargs["max_tokens"] = 1
        return self._with_aux_hooks(
            run, client, kwargs,
            lambda: create_anthropic_message(client, kwargs, log_prefix=self._prefix(), prefer_stream=False),
        )

    def _with_aux_hooks(self, run: _Run, client: Any, kwargs: dict[str, Any], call: Callable[[], Any]) -> Any:
        """One keep-alive attempt between ``pre_auxiliary_call`` and ``post_auxiliary_call``
        (``aux_task="cache_warm"``), so observers that meter auxiliary spend see every refresh.
        The observers read the request; the bytes sent are unchanged."""
        from agent.auxiliary_hooks import finish_aux_call_hooks, start_aux_call_hooks

        b = run.billing
        hooks = start_aux_call_hooks(
            aux_task="cache_warm", metadata={"api_request_id": f"cache-warm-{uuid.uuid4().hex}"}, client=client,
            kwargs=kwargs, provider=str(b["provider"] or ""), model=str(run.model or ""), api_mode=run.api_mode,
            streaming=False,
        )
        try:
            message = call()
        except BaseException as exc:
            finish_aux_call_hooks(hooks, error=exc)
            raise
        if hooks is not None:
            finish_aux_call_hooks(hooks, message, normalized=self._hook_view(run, message))
        return message

    @staticmethod
    def _hook_view(run: _Run, message: Any) -> tuple[Any, Any, Optional[dict[str, Any]]]:
        """``(assistant_message, finish_reason, usage)`` on either wire, usage normalized on the
        wire it came from (an Anthropic Message has no ``choices``)."""
        from agent.usage_pricing import normalize_usage

        choice = (getattr(message, "choices", None) or [None])[0]
        finish = getattr(choice, "finish_reason", None) if choice is not None else getattr(message, "stop_reason", None)
        usage = None
        if getattr(message, "usage", None):
            cu = normalize_usage(message.usage, provider=run.billing["provider"], api_mode=run.api_mode)
            usage = {k: v for k, v in asdict(cu).items() if k != "raw_usage"}
            usage.update(prompt_tokens=cu.prompt_tokens, total_tokens=cu.total_tokens)
        return getattr(choice, "message", None), finish, usage

    def _record(self, run: _Run, message: Any, decision: CacheWarmingDecision, sent_at: float,
                reserved: float = 0.0) -> Optional[str]:
        """Fold the refresh's tokens and dollars into the session totals (never into the
        compressor or usage anchors: a one-token replay is not a conversation turn), into the user
        turn's budget (its ``reserved`` estimate replaced by the billed cost, even if ``run`` was
        replaced meanwhile), and into the prompt-cache clock (a read restarts the window). Returns why warming must stop when the refresh did not prove a cache hit at
        a known price, ``None`` when it did."""
        agent, b = self.agent, run.billing
        usage = getattr(message, "usage", None)
        if not usage:
            return "cache refresh returned no usage"
        try:
            from agent.prompt_cache_clock import request_write_tier
            from agent.turn_usage import with_write_tier
            from agent.usage_pricing import estimate_usage_cost, normalize_usage

            canonical = normalize_usage(usage, provider=b["provider"], api_mode=b["api_mode"])
            # Priced at the tier the request declared (a 1h write is 2x input), as a normal response is.
            priced = with_write_tier(canonical, request_write_tier(
                run.api_kwargs, canonical, provider=b["provider"], base_url=b["base_url"], api_mode=b["api_mode"],
                model=b["model"]))
            cost = estimate_usage_cost(
                b["model"], priced, provider=b["provider"], base_url=b["base_url"], api_key=b["api_key"],
            )
            amount = float(cost.amount_usd) if cost.amount_usd is not None else None
            if amount is not None:
                with self._lock:
                    run.budget.spent_usd += amount - reserved
                    run.budget.last_warm_usd = amount
            # Compression keeps the running counters (a refresh still adds to them); only
            # ``reset_session_state`` zeroes them, and it advances the reset epoch with them under
            # this same lock, so a reset never lands between the epoch check and the add.
            with self._lock:
                if getattr(agent, "_session_reset_epoch", 0) == b["reset_epoch"]:
                    self._add_to_session_totals(canonical, amount)
            clock = self._fold_clock(run, canonical, sent_at)
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
                    model=b["model"], api_call_count=1, task="cache_warm", **clock,
                )
            logger.info(
                "%sprompt cache refresh sent (#%d): cache_read=%d write=%d cost=%s "
                "expected_savings=$%.3f warm_until(before)=%s", self._prefix(), run.warm_count,
                canonical.cache_read_tokens, canonical.cache_write_tokens,
                f"${amount:.4f}" if amount is not None else "?", decision.expected_savings,
                time.strftime("%H:%M:%S", time.localtime(run.deadline_at)),
            )
        except Exception:  # health: allow BLE001 -- unaccounted means stop warming; logged
            logger.debug("prompt cache warm accounting failed", exc_info=True)
            return "cache refresh usage unreadable"
        # A hit reads the prefix; an OpenAI-spec hit also writes the few tail tokens past the last cached
        # block, so only a reply that wrote more than it read (or read nothing) is a miss.
        if canonical.cache_read_tokens <= 0 or canonical.cache_write_tokens > canonical.cache_read_tokens:
            return (f"cache refresh missed (read={canonical.cache_read_tokens}, "
                    f"write={canonical.cache_write_tokens}, cost={f'${amount:.4f}' if amount is not None else '?'})")
        if amount is None:
            return "cache refresh cost unavailable"
        return None

    def _fold_clock(self, run: _Run, canonical: Any, sent_at: float) -> dict[str, str]:
        """Record the refresh in the shared clock while this run still owns the live prefix (the
        loop cancels the run under the same lock before it stamps its own next request), and
        return the ``prompt_cache_clock`` column for its token row. A reply to a prefix a
        compaction or route switch has since replaced is accounted but certifies nothing: the
        event is stamped with the arm-time witness, so a rewrite committing even now leaves the
        clock unknown rather than vouching for the new prefix."""
        from agent.prompt_cache_clock import attempt_facts, record_cache_response
        from agent.turn_usage import _cache_clock_column

        agent, b = self.agent, run.billing
        with self._lock:
            if self._run is not run or self._stale_reason(run) is not None:
                return {}
            attempt = dict(attempt_facts(run.api_kwargs) or {}, sent_at=sent_at)
            event = record_cache_response(agent, canonical, model=b["model"], provider=b["provider"],
                                          base_url=b["base_url"], api_mode=b["api_mode"], attempt=attempt,
                                          witness=run.witness)
        return _cache_clock_column(event)

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
        return self._decide(run.cached_tokens, run.uncached_tokens,
                            spent_usd=run.budget.spent_usd, observed_warm_usd=run.budget.last_warm_usd)

    def _decide(self, cached_tokens: int, uncached_tokens: int, **spend: float) -> CacheWarmingDecision:
        """``evaluate_economics``, refused outright when the estimator that bills a refresh
        (``_record``) cannot price the refresh's cache read or the miss's cache write: a missing
        cache rate must not read as free and approve a refresh whose saving is unknown."""
        decision = evaluate_economics(self._pricing(), cached_tokens, uncached_tokens, **spend)
        if decision.action == "warm" and not self._cache_costs_known(cached_tokens, uncached_tokens):
            return CacheWarmingDecision(
                warm_cost=decision.warm_cost, miss_cost=decision.miss_cost,
                expected_savings=decision.expected_savings, economics_available=False, action="stop",
            )
        return decision

    def _cache_costs_known(self, cached_tokens: int, uncached_tokens: int) -> bool:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        agent = self.agent
        route = dict(provider=agent.provider, base_url=agent.base_url, api_key=getattr(agent, "api_key", ""))
        refresh = CanonicalUsage(input_tokens=uncached_tokens, output_tokens=1, cache_read_tokens=cached_tokens)
        miss = (CanonicalUsage(input_tokens=uncached_tokens, output_tokens=1, cache_write_tokens=cached_tokens)
                if self._marked else
                CanonicalUsage(input_tokens=uncached_tokens + cached_tokens, output_tokens=1))
        try:
            return all(estimate_usage_cost(agent.model, usage, **route).amount_usd is not None
                       for usage in (refresh, miss))
        except Exception:  # health: allow BLE001 -- unpriced means never warm; logged
            logger.debug("prompt cache warm cost estimate unavailable", exc_info=True)
            return False

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
            "reset_epoch": getattr(agent, "_session_reset_epoch", 0),
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


# -- step 2: the cache-edge compaction decision ---------------------------------------------------

def human_paced(agent: Any) -> bool:
    """A chat a person types into (CLI, TUI, Desktop, messaging). Subagents, side agents (/bg),
    forks (/btw) and machine-paced sources (cron, one-shots, the API server, ...) are not."""
    from agent.prompt_caching import MACHINE_PACED_SOURCES
    from agent.session_source import session_source_for

    if int(getattr(agent, "_delegate_depth", 0) or 0) > 0 or getattr(agent, "side_agent", False) is True:
        return False
    if getattr(agent, "_skip_prompt_cache_warming", False):
        return False
    return session_source_for(getattr(agent, "platform", None)) not in MACHINE_PACED_SOURCES


def _context_tokens(agent: Any) -> int:
    tokens = getattr(getattr(agent, "context_compressor", None), "last_prompt_tokens", 0)
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0


def _compressor_blocked(agent: Any) -> Optional[str]:
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return "no_compressor"
    if getattr(compressor, "awaiting_real_usage_after_compression", False) is True:
        return "awaiting_usage"
    cooldown = getattr(compressor, "get_active_compression_failure_cooldown", None)
    if callable(cooldown) and cooldown():
        return "cooldown"
    return None


def _rough_tokens(agent: Any, messages: Optional[list] = None) -> int:
    from agent.model_metadata import estimate_request_tokens_rough

    msgs = list(getattr(agent, "_session_messages", None) or ()) if messages is None else messages
    return estimate_request_tokens_rough(msgs, system_prompt=getattr(agent, "_cached_system_prompt", "") or "",
                                         tools=getattr(agent, "tools", None) or None)


def _compaction_baseline(agent: Any) -> int:
    """Rough size the last compaction left: the compressor's own figure or, on an agent rebuilt since
    (the gateway evicts after compacting), the transcript through its newest summary handoff plus
    the tail budget kept verbatim after it. 0 when the session never compacted."""
    compressor = agent.context_compressor
    last = getattr(compressor, "last_compression_rough_tokens", 0)
    if isinstance(last, int) and not isinstance(last, bool) and last > 0:
        return last
    messages = list(getattr(agent, "_session_messages", None) or ())
    is_summary = getattr(compressor, "_is_context_summary_message", None)
    handoff = next((i for i in range(len(messages) - 1, -1, -1) if callable(is_summary) and is_summary(messages[i])), None)
    if handoff is None:
        return 0
    kept_tail = getattr(compressor, "tail_token_budget", 0)
    return _rough_tokens(agent, messages[:handoff + 1]) + (kept_tail if isinstance(kept_tail, int) else 0)


def _grew_enough(agent: Any) -> bool:
    """Growth guard: a chat that stays large after compacting must gain a floor's worth of new
    content (the compressor's post-compaction target) before it is summarised again."""
    compressor = agent.context_compressor
    try:
        floor = int(compressor.threshold_tokens * compressor.summary_target_ratio)
    except (AttributeError, TypeError, ValueError):
        floor = 0
    baseline = _compaction_baseline(agent)
    return baseline <= 0 or _rough_tokens(agent) > baseline + floor


def _session_gate(agent: Any) -> Optional[str]:
    """Why *agent*'s chat is never compacted at the cache edge, or None (cheap checks only)."""
    if agent is None or not getattr(agent, "compression_enabled", False):
        return "disabled"
    if getattr(agent, "api_mode", None) == "codex_app_server":
        return "codex_app_server"
    if not human_paced(agent):
        return "machine_paced"
    threshold = getattr(getattr(agent, "context_compressor", None), "threshold_tokens", 0) or 0
    if _context_tokens(agent) < COMPACT_CONTEXT_FRACTION * threshold or threshold <= 0:
        return "small_context"
    return _compressor_blocked(agent)


def _window_gate(agent: Any, spent: _Spent, now: float) -> Optional[str]:
    """Why the spent budget's window does not compact now, or None."""
    from agent.prompt_cache_clock import DECLARED, PROVEN, cache_window, context_witness

    if spent.tier_s < MIN_COMPACT_WINDOW_SECONDS:
        return f"short_window:{int(spent.tier_s)}s"
    if now < spent.due_at:
        return "not_yet"
    window = cache_window(agent)
    if window is None or window.confidence not in (PROVEN, DECLARED):
        return "window_not_proven"
    # No warmth check: the summary is a fresh auxiliary request, not a replay of the cached prefix, so
    # a cold prefix costs it nothing extra (and the gateway's 300 s housekeeping lands past the 5/6 mark).
    if context_witness(agent) != spent.witness:
        return "context_changed"
    return None


def compaction_due(agent: Any, *, now: Optional[float] = None) -> tuple[bool, str]:
    """``(due, reason)``: step 2. Due only when this chat's warm budget is over (spent or missed) on a
    proven/declared window of at least 30 minutes, its context is at least half the compaction
    threshold and grew since the last compaction, and no compaction ran this stretch."""
    now = time.time() if now is None else now
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    spent = warmer._spent if warmer is not None else None
    if spent is None:
        return False, "budget_not_spent" if warmer is None or not warmer._compacted else "compacted_this_stretch"
    reason = _session_gate(agent) or _window_gate(agent, spent, now)
    if reason is None and not _grew_enough(agent):  # last: the estimate walks the transcript
        reason = "no_growth"
    return reason is None, reason or "due"


def claim_compaction(agent: Any, *, now: Optional[float] = None) -> bool:
    """:func:`compaction_due`, and when due, consume the spent budget so the stretch compacts once
    (an attempt that commits nothing is not retried). Nothing is sent after it until the next user
    message: a compacted chat has no warm entry to keep."""
    due, _reason = compaction_due(agent, now=now)
    if due:
        warmer = agent._prompt_cache_warmer
        with warmer._lock:
            warmer._spent, warmer._compacted = None, True
    return due


def release_compaction_due(agent: Any, *, now: Optional[float] = None) -> tuple[bool, str]:
    """``(due, reason)``: step 2 at the owner's release. A budget ends at whichever comes first, its 3
    keep-alives or the owner releasing the agent (gateway idle/LRU eviction, TUI idle reap), so a release
    while a budget is still live (or just spent) decides step 2 now: the same gates as
    :func:`compaction_due` (warm proven/declared window of 30 minutes or more, half the threshold, growth)
    with the release as the decision moment."""
    now = time.time() if now is None else now
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is None:
        return False, "budget_not_spent"
    with warmer._lock:
        run, spent, compacted = warmer._run, warmer._spent, warmer._compacted
    if spent is None and run is not None:
        spent = _Spent(now, run.deadline_at, run.ttl_s, run.witness)
    if spent is None:
        return False, "compacted_this_stretch" if compacted else "budget_not_spent"
    reason = _session_gate(agent) or _window_gate(agent, replace(spent, due_at=min(spent.due_at, now)), now)
    if reason is None and not _grew_enough(agent):
        reason = "no_growth"
    return reason is None, reason or "due"


def claim_compaction_at_release(agent: Any, *, now: Optional[float] = None) -> bool:
    """:func:`release_compaction_due`, and when due, end the budget now: no keep-alive goes out after the
    release (nor during the compaction), and the stretch compacts once. The caller compacts, then releases."""
    due, _reason = release_compaction_due(agent, now=now)
    if due:
        warmer = agent._prompt_cache_warmer
        warmer.cancel("owner releasing the agent: compacting first")
        with warmer._lock:
            warmer._spent, warmer._compacted = None, True
    return due


def offload_due(agent: Any, *, now: Optional[float] = None) -> tuple[bool, str]:
    """``(due, reason)``: step 3's offload. The chat is idle and its cache is over: past the clock's
    proven/declared end, or its budget ended without a compaction (spent past the window or missed) and
    the window has since closed. Once per stretch; the cold prune itself applies its own gates (protected
    tail, >= 16k reclaim)."""
    from agent.turn_proactive_prune import prompt_cache_is_cold

    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is None:
        return False, "no_lifecycle"
    if warmer._offloaded:
        return False, "offloaded_this_stretch"
    if warmer._run is not None:
        return False, "still_warming"
    if agent is None or not getattr(agent, "compression_enabled", False):
        return False, "disabled"
    if not prompt_cache_is_cold(agent, now=now):
        return False, "not_cold"
    return True, "due"


def claim_offload(agent: Any, *, now: Optional[float] = None) -> bool:
    """:func:`offload_due`, and when due, mark the stretch offloaded so it is tried once. The owner then
    runs :func:`agent.turn_proactive_prune.offload_cold_history` on the chat's history."""
    due, _reason = offload_due(agent, now=now)
    if due:
        with agent._prompt_cache_warmer._lock:
            agent._prompt_cache_warmer._offloaded = True
    return due


# -- agent-facing seams (each a no-op when nothing is armed) ---------------------------------------

def get_prompt_cache_warmer(agent: Any) -> PromptCacheWarmer:
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is None:
        warmer = PromptCacheWarmer(agent)
        agent._prompt_cache_warmer = warmer
    return warmer


def session_totals_lock(agent: Any) -> Any:
    """What ``reset_session_state`` holds while it zeroes the running totals and advances the
    reset epoch: the lock a keep-alive holds across its epoch check and its add. With no warmer
    there is no keep-alive in flight, so nothing to exclude."""
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    return warmer._lock if warmer is not None else contextlib.nullcontext()


def start_prompt_cache_warming(agent: Any, api_kwargs: Any, response: Any) -> None:
    """Called once per successful provider response with the request that produced it, after
    the response was folded into the prompt-cache clock."""
    if not isinstance(api_kwargs, dict):
        return
    try:
        get_prompt_cache_warmer(agent).start(api_kwargs, response)
    except Exception:
        logger.debug("prompt cache warming did not arm", exc_info=True)


def end_turn_prompt_cache_warming(agent: Any) -> None:
    """Every turn exit: a machine session stops warming; a human chat keeps its budget idle."""
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is not None:
        warmer.end_turn()


def cancel_prompt_cache_warming(agent: Any, reason: str = "inactive") -> None:
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is not None:
        warmer.cancel(reason)

