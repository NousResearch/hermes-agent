"""The per-chat controller: keep-alives on the shared prompt-cache clock (per-user-turn budget, tool-wait
warming, the first-miss stop), the cache-edge claim, and the agent-facing seams."""

from __future__ import annotations

import contextlib
import contextvars
import copy
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from agent.prompt_cache_lifecycle.economics import (
    CacheWarmingDecision,
    evaluate_economics,
)
from agent.prompt_cache_lifecycle.in_flight import in_flight_window, request_split
from agent.prompt_cache_lifecycle.edge import (
    EdgeDecision,
    _compaction_reason,
    _offload_reason,
    _Spent,
    human_paced,
)
from agent.prompt_cache_lifecycle.replay import (
    WARMABLE_API_MODES,
    bounded_chat_replay,
    has_cache_markers,
    is_replayable,
    replay_cap_honoured,
    replay_output_cap,
)

logger = logging.getLogger(__name__)

#: Idle keep-alives per user message (sent while no tool runs); a new user message starts a budget.
#: Keep-alives while a tool runs are not counted: one tool wait is bounded by the rewrite it prevents.
WARM_BUDGET = 3
#: Why a session warms no more after its keep-alives missed on its route.
WARMING_DISABLED = "prompt cache warming disabled for session: keep-alive missed"
#: Consecutive real misses (same upstream, nothing read back) that end warming for the session: the first
#: is retried, since one stray miss is routing noise more often than a cache that dies early.
REAL_MISSES_TO_STOP = 2
#: Consecutive misses of any kind that end warming too: a routing blip does not repeat back to back, a
#: cache whose conversation part dies early while a shared system prefix survives does.
ANY_MISSES_TO_STOP = 3
#: A miss that still read at least this share of the stable system + tools prefix (rough size) found the
#: route's cache alive: the conversation part was served by another upstream.
_PARTIAL_READ_SHARE = 0.5
#: Replays are one HTTP round trip; never let a hung socket pin the timer thread.
WARM_REQUEST_TIMEOUT_S = 60.0


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


def learned_window(agent: Any, window: tuple[float, float], marked: bool
                   ) -> tuple[Optional[tuple[float, float]], Optional[str]]:
    """``window`` as the route store (``agent.prompt_cache_routes``) has learned it really behaves:
    ``(None, rule)`` when keep-alives cannot pay on it (``no-cache``, ``write-on-hit``, or a learned
    lifetime too short to warm), the window shortened to a learned lifetime (never lengthened past the
    tier), or unchanged."""
    from agent.prompt_cache_routes import warming_rule

    deadline, tier_s = window
    route = (getattr(agent, "provider", ""), getattr(agent, "base_url", ""), getattr(agent, "model", ""))
    rule = warming_rule(*route, marked, tier_s=tier_s)
    if rule is None:
        return window, None
    if not rule.startswith("lifetime:"):
        return None, rule
    life = min(float(rule.split(":", 1)[1]), tier_s)
    return (deadline - (tier_s - life), life), rule


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
    #: Keep-alives sent (and their cost) while this run's tool calls execute: one tool wait, bounded
    #: only by the rewrite it prevents, never by the idle budget.
    tool_warms: int = 0
    tool_spent_usd: float = 0.0
    #: Whether the keep-alive in flight was reserved against the tool wait (else the idle budget).
    inflight_tool: bool = False
    #: Armed when the request went out (:meth:`PromptCacheWarmer.start_in_flight`): it keeps that request's
    #: prompt warm while the reply generates, booked like a tool wait (never on the idle budget).
    awaiting_reply: bool = False
    #: That request's ``prompt_cache_clock.attempt_facts`` (the agent's ``_prompt_cache_attempt``): each
    #: keep-alive moves its send time on, so the reply folds from the latest renewal, not from its send.
    attempt: Optional[dict[str, Any]] = None
    #: That request's own reply was folded (:meth:`PromptCacheWarmer.fold_reply`): a keep-alive reply
    #: landing later folds after it as a later read of the same prompt.
    settled: bool = False
    #: ``prompt_cache_clock.response_upstream`` of the reply that last cached this prefix.
    upstream: Optional[str] = None
    #: Rough size of the stable system + tools prefix: a miss that read at least half of it was a routing blip.
    stable_prefix_tokens: int = 0
    #: What the last keep-alive was billed (``None`` when unpriced).
    last_billed_usd: Optional[float] = None

    @property
    def warm_count(self) -> int:
        return self.budget.used + self.tool_warms


@dataclass
class _Budget:
    """One USER message's idle keep-alives (sent while no tool runs). Every run of the turn draws on the
    same object, so a run replaced mid-flight still books its keep-alive here: each is reserved (count
    and estimated cost) at admission and reconciled with its billed cost when its reply lands, whichever
    run is current then. Keep-alives sent while a tool runs are booked on their run instead. Changed only
    under the warmer lock."""

    turn: Any
    used: int = 0
    spent_usd: float = 0.0
    last_warm_usd: float = 0.0
    #: Set when the budget ended (keep-alives spent, or a miss): no more keep-alives this user turn.
    ended: bool = False


class PromptCacheWarmer:
    """The per-chat lifecycle controller: keeps one prompt-cache entry alive by replaying its
    request with the smallest output cap the model accepts before the entry expires (``WARM_BUDGET``
    idle keep-alives per user message), then answers the cache edge (:meth:`claim_edge`). ``start``
    replaces any previous run."""

    def __init__(self, agent: Any):
        self.agent = agent
        # Guards ``_run`` ownership: the timer thread and the loop both replace or stop it.
        self._lock = threading.RLock()
        self._run: Optional[_Run] = None
        self._inactive_reason = "waiting for first request"
        self._logged_reason: Optional[str] = None
        self._last_decision: Optional[CacheWarmingDecision] = None
        #: When the keep-alive still waiting for its reply went out (None once it failed or its reply
        #: folded into the clock): it renewed the entry, so the next message's tier check counts it.
        self.last_sent_at: Optional[float] = None
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
        #: The route ``(provider, base_url, model)`` whose keep-alive missed: the cache does not live as
        #: long as assumed there, so nothing is warmed again this session (new user messages and
        #: compactions included). A model/provider switch is a different cache and clears it.
        self._missed_route: Optional[tuple] = None
        #: Keep-alive misses since the last hit: ``(real, any)``. Upstream switches and partial reads are
        #: not real misses (:meth:`_miss_ends_warming`).
        self._miss_streak = (0, 0)

    # -- read side ------------------------------------------------------------------------------
    @property
    def warming_disabled(self) -> bool:
        """Whether a keep-alive missed on the live route, so this session warms no more."""
        from agent.prompt_cache_clock import _display_route

        return self._missed_route is not None and self._missed_route == _display_route(self.agent)

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
            if self._missed_route is not None and not self.warming_disabled:
                self._missed_route = None  # a model/provider switch: a different cache, warmed afresh
            if self.warming_disabled:
                # Steps 2/3 still decide on this prompt at the moment a keep-alive would have gone out.
                budget.ended = True
                self._carry_ended_budget()
                return self._stop(WARMING_DISABLED)
            if budget.ended or budget.used >= WARM_BUDGET:
                self._carry_ended_budget()
                return self._stop("warm budget spent this turn")
            self._spent = None
        refused = self._refusal(api_kwargs)
        if refused is not None:
            return self._stop(refused)
        window = guaranteed_window(agent)
        if window is None or cache_warming_delay_s(window[1]) is None:
            return self._stop("cache window not guaranteed warm")
        window, learned = learned_window(agent, window, self._marked)
        if window is None:
            return self._stop(f"learned: {learned}")
        cached_tokens, uncached_tokens = self._prompt_split(response)
        if cached_tokens + uncached_tokens <= 0:
            return self._stop("prompt size unavailable")
        client, refused = self._replay_client(api_kwargs)
        if refused is not None:
            return self._stop(refused)
        # Economics before the deep copy: an unpriced/cheap model never pays for the copy either. Nothing
        # is spent yet on the tool wait this run may cover; admission checks the idle budget's spend.
        decision = self._decide(cached_tokens, uncached_tokens, observed_warm_usd=budget.last_warm_usd)
        self._last_decision = decision
        if decision.action != "warm":
            return self._stop(decision.stop_reason)
        from agent.prompt_cache_clock import context_witness, response_upstream

        # The loop mutates message dicts in place between requests (cache-marker redecoration,
        # tool-result eviction); the replay must send exactly what wrote the entry.
        run = _Run(
            api_kwargs=self._replay_kwargs(api_kwargs), client=client, model=getattr(agent, "model", ""),
            cached_tokens=cached_tokens, uncached_tokens=uncached_tokens, ttl_s=window[1],
            deadline_at=window[0], is_current=self._currency(), api_mode=agent.api_mode,
            witness=context_witness(agent), billing=self._billing_snapshot(),
            idle_ok=human_paced(agent), budget=budget,
            upstream=response_upstream(response),
            stable_prefix_tokens=_stable_prefix_tokens(api_kwargs, agent.api_mode),
        )
        return self._arm(run)

    def start_in_flight(self, api_kwargs: dict[str, Any]) -> Optional[str]:
        """Arm keep-alives for the request going out now, replaying that same request while its reply
        generates: a reply longer than the entry's lifetime (a long generation, a slow upstream) would
        otherwise find its own prefix gone and the next request re-writes all of it. The entry lives
        from this send for the tier the request asks; the route must already have proven it caches (the
        clock holds an earlier reply's evidence on it). Spending is bounded like a tool wait: together
        the keep-alives cost no more than the one rewrite they prevent. The reply replaces the run
        (:meth:`start`), the turn's end cancels it. Returns the reason nothing was armed."""
        from agent.prompt_cache_clock import context_witness

        self._clear_run()
        agent = self.agent
        refused = self._refusal(api_kwargs) or self._in_flight_refusal()
        if refused is not None:
            return self._stop(refused, lifecycle=True)
        window = in_flight_window(agent, api_kwargs)
        if window is None or cache_warming_delay_s(window[1]) is None:
            return self._stop("no cache window to keep warm (no markers, nothing proven)", lifecycle=True)
        window, learned = learned_window(agent, window, self._marked)
        if window is None:
            return self._stop(f"learned: {learned}", lifecycle=True)
        client, refused = self._replay_client(api_kwargs)
        if refused is not None:
            return self._stop(refused, lifecycle=True)
        # Its reply's usage is not known yet: the prefix through the last cache marker is cached once it is
        # sent, anything after it is re-billed as plain input on every replay.
        cached_tokens, uncached_tokens = request_split(api_kwargs)
        decision = self._decide(cached_tokens, uncached_tokens)
        if decision.action != "warm":
            return self._stop(decision.stop_reason, lifecycle=True)
        turn = getattr(agent, "_user_turn_count", None)
        with self._lock:
            budget = self._budget if self._budget is not None and self._budget.turn == turn else _Budget(turn)
        run = _Run(
            api_kwargs=self._replay_kwargs(api_kwargs), client=client, model=getattr(agent, "model", ""),
            cached_tokens=cached_tokens, uncached_tokens=uncached_tokens, ttl_s=window[1], deadline_at=window[0],
            is_current=self._currency(), api_mode=agent.api_mode, witness=context_witness(agent),
            billing=self._billing_snapshot(), budget=budget, awaiting_reply=True,
            attempt=getattr(agent, "_prompt_cache_attempt", None),
            stable_prefix_tokens=_stable_prefix_tokens(api_kwargs, agent.api_mode),
        )
        return self._arm(run)

    def _arm(self, run: _Run) -> Optional[str]:
        with self._lock:
            self._run = run
            self._schedule(run)
            return None if self._run is run else self._inactive_reason

    def _currency(self) -> Callable[[], bool]:
        """Whether the prompt a run was armed on is still the live one: no reply folded since (the epoch,
        not the counter alone: a reset zeroes ``session_api_calls``, which the new session's own requests
        then raise back to the armed value), the same model, no interrupt."""
        agent = self.agent
        api_calls = getattr(agent, "session_api_calls", 0)
        model = getattr(agent, "model", "")
        epoch = getattr(agent, "_session_reset_epoch", 0)

        def _is_current() -> bool:
            return (
                getattr(agent, "_session_reset_epoch", 0) == epoch
                and getattr(agent, "session_api_calls", None) == api_calls
                and getattr(agent, "model", None) == model
                and not getattr(agent, "_interrupt_requested", False)
            )

        return _is_current

    def _refusal(self, api_kwargs: dict[str, Any]) -> Optional[str]:
        """Why this request can never be warmed (route, caching, replay safety), or ``None``. Sets
        whether it carries cache markers: an OpenAI-spec automatic cache bills a miss at the plain input
        rate (no write surcharge)."""
        agent = self.agent
        if getattr(agent, "_skip_prompt_cache_warming", False):
            return "the parent agent keeps this prefix warm"
        api_mode = getattr(agent, "api_mode", None)
        if api_mode not in WARMABLE_API_MODES:
            return "only Anthropic Messages and Chat Completions requests are warmed"
        # Marker caching is Anthropic-spec only; an OpenAI-spec route caches automatically and proves it
        # with the reply's cached tokens (no read or write folded = no window).
        if getattr(agent, "_cache_disabled", False) or (
                api_mode == "anthropic_messages" and not getattr(agent, "_use_prompt_caching", False)):
            return "prompt caching disabled"
        if "_moa_prepared_request" in api_kwargs or getattr(agent, "provider", "") == "moa":
            return "MoA requests are not replayable"
        if getattr(agent, "_interrupt_requested", False):
            return "turn interrupted"
        if api_mode == "anthropic_messages" and not has_cache_markers(api_kwargs):
            return "request carries no cache markers"
        if not is_replayable(api_kwargs):
            return "request cannot be replayed safely (budget thinking)"
        self._marked = has_cache_markers(api_kwargs)
        return None

    def _in_flight_refusal(self) -> Optional[str]:
        with self._lock:
            if self._missed_route is not None and not self.warming_disabled:
                self._missed_route = None  # a model/provider switch: a different cache, warmed afresh
            return WARMING_DISABLED if self.warming_disabled else None

    def _replay_client(self, api_kwargs: dict[str, Any]) -> tuple[Any, Optional[str]]:
        agent = self.agent
        api_mode = agent.api_mode
        client = getattr(agent, "_anthropic_client" if api_mode == "anthropic_messages" else "client", None)
        if client is None:
            return None, "no provider client"
        if not replay_cap_honoured(client, api_kwargs, getattr(agent, "model", ""), api_mode):
            return None, "the client cannot bound the replay's output cap"
        return client, None

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
        self._spent = _Spent(deadline - (tier - delay), tier, context_witness(self.agent))

    # -- the cache edge (steps 2 and 3) ----------------------------------------------------------
    def edge_decision(self, *, releasing: bool = False, now: Optional[float] = None) -> EdgeDecision:
        """The cache-edge decision without claiming it (``/context``, owners' diagnostics, tests).
        ``releasing``: the owner is releasing the agent, which ends a live budget now."""
        now = time.time() if now is None else now
        with self._lock:
            compact = _compaction_reason(self.agent, self, releasing, now)
            offload = "compaction_due" if compact == "due" else _offload_reason(self.agent, self, now)
        action = "compact" if compact == "due" else "offload" if offload == "due" else None
        return EdgeDecision(action, compact, offload)

    def claim_edge(self, *, releasing: bool = False, now: Optional[float] = None) -> Optional[str]:
        """:meth:`edge_decision`, validated and latched in one step under the warmer lock: ``"compact"``
        consumes the spent budget (the stretch compacts once; an attempt that commits nothing is not
        retried; at a release the live run is cancelled, so no keep-alive goes out after it) and
        ``"offload"`` marks the stretch offloaded. The owner runs that action (compact through the
        normal pipeline, offload through ``turn_proactive_prune.offload_cold_history``); nothing is
        sent after either until the next user message."""
        with self._lock:
            action = self.edge_decision(releasing=releasing, now=now).action
            if action == "compact":
                if releasing:
                    self.cancel("owner releasing the agent: compacting first")
                self._spent, self._compacted = None, True
            elif action == "offload":
                self._offloaded = True
            return action

    def touch_snapshot(self) -> tuple[Any, Optional[float]]:
        """``(cache_window, in-flight keep-alive send time)`` read together under the lock the reply's
        fold and the in-flight mark's clearing take, so a reader never sees the old clock with the mark
        already cleared."""
        from agent.prompt_cache_clock import cache_window

        with self._lock:
            return cache_window(self.agent), self.last_sent_at

    # -- internals ------------------------------------------------------------------------------
    def _set_in_flight(self, sent_at: Optional[float]) -> None:
        with self._lock:
            self.last_sent_at = sent_at

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
        if not self._booked_on_run(run) and (run.budget.ended or run.budget.used >= WARM_BUDGET):
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
                # A late miss after the next reply replaced its run: the user turn's budget is over all the
                # same, so decide step 2/3 now on the prompt the turn holds (its own run, or the clock).
                current = self._run
                if current is not None and current.budget is run.budget:
                    self._end_budget(current, reason, due_at=due_at)
                elif current is None and self._budget is run.budget:
                    self._carry_ended_budget()
                return
            spent = _Spent(due_at, run.ttl_s, run.witness)
            self._stop(reason)
            self._spent = spent

    def _stale_reason(self, run: _Run) -> Optional[str]:
        """Why ``run`` no longer replays the live prefix, or None. Caller holds ``self._lock``."""
        from agent.prompt_cache_clock import context_witness

        if not run.is_current() or context_witness(self.agent) != run.witness:
            return "conversation context changed"
        return None

    def _tool_running(self) -> bool:
        """A tool call is executing: the model is certain to send the next request when it returns."""
        return getattr(self.agent, "_executing_tools", False) is True

    def _booked_on_run(self, run: _Run) -> bool:
        """A keep-alive now spends this run's wait (a tool call executing, or the reply to the request
        the run was armed on still generating), not the user turn's idle budget."""
        return run.awaiting_reply or self._tool_running()

    def _admit(self, run: _Run, decision: CacheWarmingDecision) -> Optional[float]:
        """The one send admission: ownership, cancellation, currency, deadline and the spend it draws on
        checked, the keep-alive reserved (count and estimated cost), and the run marked in flight, in one
        step under the lock ``cancel`` takes. While a tool runs it draws on this tool wait (no count cap:
        its keep-alives together may cost no more than the one rewrite they prevent); otherwise on the
        user turn's idle budget (:data:`WARM_BUDGET`). Returns the reserved cost, ``None`` when refused.
        The network request itself runs outside the lock."""
        with self._lock:
            if self._run is not run or run.cancelled:
                return None
            reason = self._stale_reason(run)
            if reason is None and time.time() > run.refresh_deadline_at:
                reason = "cache refresh deadline missed"
            if reason is not None:
                self._stop(reason)
                return None
            if self.warming_disabled:
                self._stop(WARMING_DISABLED)
                return None
            budget = run.budget
            run.inflight_tool = self._booked_on_run(run)
            if run.inflight_tool:
                if run.tool_spent_usd + decision.warm_cost > decision.miss_cost:
                    self._stop("tool-wait refreshes would cost more than the cache write they prevent")
                    return None
                run.tool_warms += 1
                run.tool_spent_usd += decision.warm_cost
            else:
                if budget.ended or budget.used >= WARM_BUDGET:
                    self._end_budget(run, "warm budget spent", due_at=run.next_warm_at)
                    return None
                if budget.spent_usd + decision.warm_cost > decision.miss_cost:
                    # The budget's money is spent: the same end as its count, so step 2/3 still decides.
                    self._end_budget(run, "refreshes would cost more than the cache write they prevent",
                                     due_at=run.next_warm_at)
                    return None
                budget.used += 1
                budget.spent_usd += decision.warm_cost
            run.refreshing = True
            return decision.warm_cost

    def _refresh(self, run: _Run) -> None:
        from hermes_logging import set_session_context

        set_session_context(run.billing.get("session_id") or "")  # the timer thread's log lines name the chat
        with run.lock:
            run.timer = None
        # Pricing may be slow (a pricing fetch); it decides nothing until admission rechecks.
        decision = self._evaluate(run)
        self._last_decision = decision
        if decision.action != "warm":
            if decision.economics_available and not decision.within_miss and not self._booked_on_run(run):
                with self._lock:
                    if self._run is run:  # the idle budget's money is spent: step 2/3 decides now
                        self._end_budget(run, decision.stop_reason, due_at=run.next_warm_at)
                return
            self._stop_if_current(run, decision.stop_reason)
            return
        reserved = self._admit(run, decision)
        if reserved is None:
            return
        sent_at = time.time()
        self._set_in_flight(sent_at)
        try:
            message = self._send(run)
        except Exception as exc:  # health: allow BLE001 -- warming must never disturb the live turn; logged
            self._set_in_flight(None)  # it renewed nothing
            logger.info("%sprompt cache warm failed: %s", self._prefix(), exc, exc_info=True)
            self._stop_if_current(run, f"warm request failed: {type(exc).__name__}")
            return
        finally:
            run.refreshing = False
        miss = self._record(run, message, decision, sent_at, reserved)
        self._set_in_flight(None)  # the reply is folded: the clock is the touch again (a miss renewed nothing)
        if miss is not None and not miss.startswith("cache refresh missed"):
            self._stop_if_current(run, miss)
            return
        if miss is not None and self._miss_ends_warming(run, message):
            # The entry was gone again and this send re-wrote it: the cache does not live as long as the
            # clock assumed on this route, so no keep-alive goes out again this session; the budget is
            # over, so step 2 (compact) or step 3 (expire + offload) is decided now.
            self._disable_warming(run)
            window = guaranteed_window(self.agent)
            if window is not None:
                run.deadline_at, run.ttl_s = window
            self._end_budget(run, miss, due_at=time.time())
            return
        if miss is None:
            self._miss_streak = (0, 0)
        else:
            self._uncharge_miss(run)
        retry_from = sent_at if miss is not None else None
        # A hit, a routing blip or the one retried miss: the reply (re)wrote the prefix, warm on its window.
        with self._lock:
            if self._run is not run:
                return
            stale = self._stale_reason(run)
            if stale is not None:  # its reply was not folded: nothing to schedule from
                self._stop(stale)
                return
            window = guaranteed_window(self.agent)
            if window is not None:
                window, learned = learned_window(self.agent, window, self._marked)
                if window is None:  # the budget is over all the same: steps 2/3 decide at its next send
                    self._end_budget(run, f"learned: {learned}", due_at=run.next_warm_at)
                    return
            if retry_from is not None:
                # The miss re-sent the whole prompt at ``retry_from``, so the entry lives from then, for the lifetime
                # just learned (never the tier it replaced). An automatic cache that reports no write counter left
                # the clock where it was: never retry from that.
                life = window[1] if window else run.ttl_s
                window = (max(window[0] if window else 0.0, retry_from + life), life)
            if window is None:
                self._stop("cache window not guaranteed warm")
                return
            run.deadline_at, run.ttl_s = window
            self._schedule(run)

    def _miss_ends_warming(self, run: _Run, message: Any) -> bool:
        """Whether this keep-alive miss ends warming for the session. Not when the reply came from another
        upstream than the one that cached the prefix (``response_upstream``), nor when it still read the
        stable system prefix (a partial read: the route's cache is alive): routing, not lifetime. A real
        miss ends it only on the :data:`REAL_MISSES_TO_STOP`-th in a row (one retry), and any
        :data:`ANY_MISSES_TO_STOP` misses in a row end it too."""
        from agent.prompt_cache_clock import response_upstream, upstream_switched
        from agent.usage_pricing import normalize_usage

        upstream = response_upstream(message)
        switched = upstream_switched(run.upstream, upstream)
        run.upstream = upstream or run.upstream  # the prefix now lives where this reply wrote it
        read = normalize_usage(message.usage, provider=run.billing["provider"], api_mode=run.api_mode).cache_read_tokens
        partial = 0 < run.stable_prefix_tokens * _PARTIAL_READ_SHARE <= read
        real, anyk = self._miss_streak
        self._miss_streak = (real, anyk) = (real + (not (switched or partial)), anyk + 1)
        logger.info("%sprompt cache keep-alive missed (%s; real misses in a row %d)", self._prefix(),
                    "upstream switch" if switched else "partial read" if partial else "nothing read", real)
        return real >= REAL_MISSES_TO_STOP or anyk >= ANY_MISSES_TO_STOP

    def _uncharge_miss(self, run: _Run) -> None:
        """A miss that does not end warming re-wrote the prefix: that rewrite is what the keep-alives were
        bounded by, not one of them, so it leaves the money limit and the next reservation as they were."""
        with self._lock:
            amount, run.last_billed_usd = run.last_billed_usd or 0.0, None
            if run.inflight_tool:
                run.tool_spent_usd -= amount
            else:
                run.budget.spent_usd -= amount
            run.budget.last_warm_usd = 0.0

    def _disable_warming(self, run: _Run) -> None:
        from agent.prompt_cache_clock import _route_key

        b = run.billing
        with self._lock:
            if getattr(self.agent, "_session_reset_epoch", 0) != b["reset_epoch"]:
                return  # a miss of the session a reset ended: the new one is warmed afresh
            # The route the keep-alive was sent on: a switch since then leaves the new route warmable.
            self._missed_route = _route_key(b["provider"], b["base_url"], b["model"])
            _remember_missed_route(b["session_id"], self._missed_route)
        logger.info("%sprompt cache warming disabled for session: keep-alive missed", self._prefix())

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
        from agent.auxiliary_hooks import run_with_aux_hooks

        return run_with_aux_hooks(
            call, aux_task="cache_warm", metadata={"api_request_id": f"cache-warm-{uuid.uuid4().hex}"},
            client=client, kwargs=kwargs, provider=str(run.billing["provider"] or ""), model=str(run.model or ""),
            api_mode=run.api_mode, normalized=lambda message: self._hook_view(run, message),
        )

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
        compressor or usage anchors: a bounded replay is not a conversation turn), into the user
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
            from agent.usage_pricing import estimate_usage_cost, normalize_usage, with_served_service_tier

            # The served tier (OpenAI ``service_tier`` on the response) prices it, as for a normal turn.
            canonical = with_served_service_tier(
                normalize_usage(usage, provider=b["provider"], api_mode=b["api_mode"]), message)
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
                    if run.inflight_tool:
                        run.tool_spent_usd += amount - reserved
                    else:
                        run.budget.spent_usd += amount - reserved
                    run.budget.last_warm_usd = amount
                    run.last_billed_usd = amount
            # Compression keeps the running counters (a refresh still adds to them); only
            # ``reset_session_state`` zeroes them, and it advances the reset epoch with them under
            # this same lock, so a reset never lands between the epoch check and the add.
            with self._lock:
                if getattr(agent, "_session_reset_epoch", 0) == b["reset_epoch"]:
                    self._add_to_session_totals(canonical, amount)
            clock = self._fold_clock(run, canonical, sent_at, message)
            if getattr(agent, "_session_db", None) is not None and b["session_id"]:
                from agent.turn_usage import queue_usage
                queue_usage(
                    agent, b["session_id"], b["source"],
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

    def _fold_clock(self, run: _Run, canonical: Any, sent_at: float, message: Any = None) -> dict[str, str]:
        """Record the refresh in the shared clock while this run still owns the live prefix (the
        loop cancels the run under the same lock before it stamps its own next request), and
        return the ``prompt_cache_clock`` column for its token row. A reply to a prefix a
        compaction or route switch has since replaced is accounted but certifies nothing: the
        event is stamped with the arm-time witness, so a rewrite committing even now leaves the
        clock unknown rather than vouching for the new prefix.

        A keep-alive of a request still awaiting its reply also moves that request's attempt send time
        to its own, in the same locked step, so the reply (folded by :func:`fold_reply` under this lock)
        counts its entry from the latest renewal. One whose reply was already folded (``settled``) is
        folded after it as an ordinary later read of the same prompt, unless a rewrite intervened or a request
        sent after it has folded since (an older send never shortens newer coverage)."""
        from agent.prompt_cache_clock import attempt_facts, context_witness, record_cache_response, response_upstream
        from agent.turn_usage import cache_clock_column

        agent, b = self.agent, run.billing
        with self._lock:
            if run.settled:
                # Only as the latest send: a later request already folded counts from its own send.
                latest = getattr(agent, "_prompt_cache_clock", None)
                stale = context_witness(agent) != run.witness or float(getattr(latest, "sent_at", 0.0) or 0.0) > sent_at
            else:
                stale = self._run is not run or self._stale_reason(run) is not None
            if stale:
                return {}
            attempt = dict(attempt_facts(run.api_kwargs) or {}, sent_at=sent_at, replay=True)
            event = record_cache_response(agent, canonical, model=b["model"], provider=b["provider"],
                                          base_url=b["base_url"], api_mode=b["api_mode"], attempt=attempt,
                                          witness=run.witness, upstream=response_upstream(message))
            if (event is not None and run.awaiting_reply and not run.settled and run.attempt is not None
                    and getattr(agent, "_prompt_cache_attempt", None) is run.attempt):
                run.attempt = agent._prompt_cache_attempt = dict(run.attempt, sent_at=sent_at)
        return cache_clock_column(event)

    def fold_reply(self, fold: Callable[[], Any]) -> Any:
        """Fold the loop's own reply (``fold``, the clock write) under the lock keep-alive folds take:
        a keep-alive of the request awaiting it either renewed the attempt before (the reply counts from
        there) or folds after it (``settled``), never in between."""
        with self._lock:
            run = self._run
            if run is not None and run.awaiting_reply:
                run.settled = True
            return fold()

    def _add_to_session_totals(self, canonical: Any, amount: Optional[float]) -> None:
        # Not session_api_calls: that counter is the conversation's request sequence (and the
        # warmer's own currency token); a replay is accounted, not a turn.
        from agent.turn_usage import add_session_usage

        agent = self.agent
        agent.session_cache_warm_calls = getattr(agent, "session_cache_warm_calls", 0) + 1
        add_session_usage(agent, canonical, prompt_tokens=canonical.prompt_tokens,
                          completion_tokens=canonical.output_tokens, total_tokens=canonical.total_tokens)
        if amount is not None:
            agent.session_estimated_cost_usd += amount

    def _evaluate(self, run: _Run) -> CacheWarmingDecision:
        spent = run.tool_spent_usd if self._booked_on_run(run) else run.budget.spent_usd
        return self._decide(run.cached_tokens, run.uncached_tokens,
                            spent_usd=spent, observed_warm_usd=run.budget.last_warm_usd)

    def _decide(self, cached_tokens: int, uncached_tokens: int, **spend: float) -> CacheWarmingDecision:
        """``evaluate_economics``, refused outright when the estimator that bills a refresh
        (``_record``) cannot price the refresh's cache read or the miss's cache write: a missing
        cache rate must not read as free and approve a refresh whose saving is unknown."""
        decision = evaluate_economics(self._pricing(), cached_tokens, uncached_tokens, **spend,
                                      output_tokens=replay_output_cap(self.agent.model))
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
        cap = replay_output_cap(agent.model)
        refresh = CanonicalUsage(input_tokens=uncached_tokens, output_tokens=cap, cache_read_tokens=cached_tokens)
        miss = (CanonicalUsage(input_tokens=uncached_tokens, output_tokens=cap, cache_write_tokens=cached_tokens)
                if self._marked else
                CanonicalUsage(input_tokens=uncached_tokens + cached_tokens, output_tokens=cap))
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
        from agent.turn_usage import agent_session_source

        return {
            "session_id": getattr(agent, "session_id", None), "source": agent_session_source(agent),
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


def _stable_prefix_tokens(api_kwargs: dict[str, Any], api_mode: str) -> int:
    """Rough token size of the part of the prompt every request of the session shares: the system prompt
    and the tools (on Chat Completions the system message leads ``messages``)."""
    import json

    from agent.model_metadata import estimate_tokens_rough

    stable: list[Any] = [api_kwargs.get("tools") or []]
    if api_mode == "anthropic_messages":
        stable.append(api_kwargs.get("system") or [])
    else:
        messages = api_kwargs.get("messages") or []
        if messages and isinstance(messages[0], dict) and messages[0].get("role") in ("system", "developer"):
            stable.append(messages[0])
    return estimate_tokens_rough(json.dumps(stable, default=str))


# -- agent-facing seams (each a no-op when nothing is armed) ---------------------------------------

#: Session id -> the route whose keep-alive missed. A miss ends warming for the SESSION, which outlives
#: one agent: the gateway evicts and rebuilds agents (cache-edge compaction, idle eviction), and a
#: compaction moves the conversation to a child session id (``carry_missed_route``). Bounded LRU.
_MISSED_ROUTES: OrderedDict[str, tuple] = OrderedDict()
_MISSED_ROUTES_MAX = 1000
_missed_routes_lock = threading.Lock()


def _remember_missed_route(session_id: Optional[str], route: Optional[tuple]) -> None:
    if not session_id:
        return
    with _missed_routes_lock:
        if route is None:
            _MISSED_ROUTES.pop(session_id, None)
            return
        _MISSED_ROUTES[session_id] = route
        _MISSED_ROUTES.move_to_end(session_id)
        while len(_MISSED_ROUTES) > _MISSED_ROUTES_MAX:
            _MISSED_ROUTES.popitem(last=False)


def carry_missed_route(old_session_id: Optional[str], new_session_id: Optional[str]) -> None:
    """A compaction continued the conversation in a child session: its keep-alive miss goes with it."""
    with _missed_routes_lock:
        route = _MISSED_ROUTES.get(old_session_id or "")
    if route is not None:
        _remember_missed_route(new_session_id, route)


def get_prompt_cache_warmer(agent: Any) -> PromptCacheWarmer:
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is None:
        warmer = PromptCacheWarmer(agent)
        with _missed_routes_lock:  # an agent rebuilt for the same session keeps its miss
            warmer._missed_route = _MISSED_ROUTES.get(getattr(agent, "session_id", None) or "")
        agent._prompt_cache_warmer = warmer
    return warmer


def cache_touch_snapshot(agent: Any) -> tuple[Any, Optional[float]]:
    """:meth:`PromptCacheWarmer.touch_snapshot` for any agent; one that never armed a warmer has no
    keep-alive in flight."""
    from agent.prompt_cache_clock import cache_window

    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if not isinstance(warmer, PromptCacheWarmer):
        return cache_window(agent), None
    return warmer.touch_snapshot()


def session_totals_lock(agent: Any) -> Any:
    """What ``reset_session_state`` holds while it zeroes the running totals and advances the
    reset epoch: the lock a keep-alive holds across its epoch check and its add. With no warmer
    there is no keep-alive in flight, so nothing to exclude."""
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    return warmer._lock if warmer is not None else contextlib.nullcontext()


def reset_prompt_cache_session(agent: Any) -> None:
    """A fresh session (``/new``, a session switch): the old one's keep-alive miss, budget and edge
    latches no longer apply (its turn counter restarts, so a kept budget would be mistaken for the new
    chat's), and an adaptive ``auto`` tier starts on 5m again. Compaction keeps the session, the miss and
    the tier with it."""
    from agent.prompt_cache_lifecycle.tier import reset_cache_tier

    reset_cache_tier(agent)
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is not None:
        warmer.cancel("session reset")
        with warmer._lock:
            warmer._missed_route = None  # a new session id: nothing remembered for it
            warmer._miss_streak = (0, 0)
            warmer._budget, warmer._spent, warmer._compacted, warmer._offloaded = None, None, False, False


def prompt_cache_warming_disabled(agent: Any) -> bool:
    """Whether a keep-alive missed on the live route, so this session sends no more (``/context``)."""
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    return warmer is not None and warmer.warming_disabled


def start_prompt_cache_warming(agent: Any, api_kwargs: Any, response: Any) -> None:
    """Called once per successful provider response with the request that produced it, after
    the response was folded into the prompt-cache clock."""
    if not isinstance(api_kwargs, dict):
        return
    try:
        get_prompt_cache_warmer(agent).start(api_kwargs, response)
    except Exception:
        logger.debug("prompt cache warming did not arm", exc_info=True)


def fold_reply(agent: Any, fold: Callable[[], Any]) -> Any:
    """:meth:`PromptCacheWarmer.fold_reply` for any agent (no warmer: nothing to order against)."""
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    return warmer.fold_reply(fold) if isinstance(warmer, PromptCacheWarmer) else fold()


def start_in_flight_prompt_cache_warming(agent: Any, api_kwargs: Any) -> None:
    """Called as a turn-loop request goes out (after its attempt is stamped on the clock)."""
    if not isinstance(api_kwargs, dict):
        return
    try:
        get_prompt_cache_warmer(agent).start_in_flight(api_kwargs)
    except Exception:
        logger.debug("prompt cache warming did not arm for the request in flight", exc_info=True)


def end_turn_prompt_cache_warming(agent: Any) -> None:
    """Every turn exit: a machine session stops warming; a human chat keeps its budget idle."""
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is not None:
        warmer.end_turn()


def cancel_prompt_cache_warming(agent: Any, reason: str = "inactive") -> None:
    warmer = getattr(agent, "_prompt_cache_warmer", None)
    if warmer is not None:
        warmer.cancel(reason)
