"""The prompt-cache lifecycle: one controller per chat, driven by the shared prompt-cache clock.

Modules: ``replay`` (which requests replay, with which output cap, through which client), ``economics``
(a keep-alive's cost against the rewrite it prevents), ``edge`` (the cache-edge gates: compact or offload),
``warmer`` (``PromptCacheWarmer``: keep-alives, budget, first-miss stop, ``claim_edge``; the agent-facing
seams), ``in_flight`` (the window a request still awaiting its reply keeps warm). This facade is the surface; code imports topic modules directly inside the package.

When a chat's cache is about to expire the controller makes ONE decision, in this order:

1. WARM: re-send the last request with a tiny output cap at 5/6 of the entry's TTL (port of
   earendil-works/pi#9668): one cache READ keeps the whole prefix warm for another TTL. While a tool
   call runs, keep-alives continue until it returns, uncounted, bounded only by money: one tool wait's
   keep-alives together cost no more than the rewrite they prevent. While a chat a person types into
   is idle, up to :data:`WARM_BUDGET` per USER message. Machine-paced sessions (subagents, cron,
   one-shots, /bg, /btw, the API server) warm only while a tool runs. The first keep-alive that misses
   (comes back mostly written, not read) ends warming for the rest of the session on that route, and
   step 2 or 3 is decided right then.
2. COMPACT: once the budget is spent, an idle human chat compacts IF its window is proven or
   declared (any tier: on 5m that is once its keep-alives stop covering the pause), its context is at
   least half of the compaction threshold, and it grew since its last compaction; at most one
   compaction per idle stretch. An ``auto`` chat still on 5m moves to 1h on the next message: its
   5m entry is gone by then. The summary is a fresh auxiliary request, so whether the old prefix is still warm
   does not change its cost and is not checked. Nothing is sent after it until the next user message.
   The surface that owns the chat (TUI ticker, gateway watcher: the CLI's turns run in the gateway) polls
   ``PromptCacheWarmer.claim_edge`` and runs the action it returns through the normal pipeline. A
   budget ends at whichever comes first: its keep-alives, or the owner releasing the agent (gateway
   idle/LRU eviction, TUI idle reap); at release the owner asks ``claim_edge(releasing=True)`` and acts
   on the answer first.
3. EXPIRE + OFFLOAD: otherwise the entry dies, and ``claim_edge`` answers ``"offload"``: the owner
   offloads old tool output right then (the same cold offload as a return after expiry), so the chat is
   already small when the person comes back. A release past the window offloads before it releases.
4. COLD RETURN lives in ``agent.turn_proactive_prune``: the next request after a proven or declared
   expiry, or on a new route, offloads old tool results first.

An unknown or assumed clock means nothing acts. A keep-alive is sent only when its read costs less
than the rewrite it prevents, the budget's keep-alives together cost no more than one rewrite, the
cache prices are known, and the cached prefix clears the smallest cacheable size. A keep-alive is
bound to the prompt and route it replays (``prompt_cache_clock.context_witness``), admitted under the
lock ``cancel`` takes, and fires the auxiliary call hooks (``aux_task="cache_warm"``). Every stop
reason is logged at INFO (once per change).

Requests are replayed on the Anthropic Messages wire and on the Chat Completions wire (Nous Portal,
OpenRouter and other gateways): Claude by its ``cache_control`` markers, every other model as an
OpenAI-spec automatic cache the clock declares at 30 minutes, renewed by a read. A request whose
thinking is budget-based is skipped because the replay's small output cap would change ``budget_tokens``.
"""

from agent.prompt_cache_lifecycle.economics import (
    MIN_CACHEABLE_PREFIX_TOKENS,
    CacheWarmingDecision,
    evaluate_economics,
)
from agent.prompt_cache_lifecycle.edge import (
    CHECK_INTERVAL_SECONDS,
    COMPACT_CONTEXT_FRACTION,
    TRIGGER,
    EdgeDecision,
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
from agent.prompt_cache_lifecycle.warmer import (
    WARM_BUDGET,
    WARM_REQUEST_TIMEOUT_S,
    WARMING_DISABLED,
    PromptCacheWarmer,
    cache_warming_delay_s,
    cancel_prompt_cache_warming,
    carry_missed_route,
    end_turn_prompt_cache_warming,
    get_prompt_cache_warmer,
    guaranteed_window,
    learned_window,
    prompt_cache_warming_disabled,
    reset_prompt_cache_session,
    session_totals_lock,
    start_in_flight_prompt_cache_warming,
    start_prompt_cache_warming,
)

__all__ = [
    "CHECK_INTERVAL_SECONDS",
    "COMPACT_CONTEXT_FRACTION",
    "MIN_CACHEABLE_PREFIX_TOKENS",
    "TRIGGER",
    "WARMABLE_API_MODES",
    "WARMING_DISABLED",
    "WARM_BUDGET",
    "WARM_REQUEST_TIMEOUT_S",
    "CacheWarmingDecision",
    "EdgeDecision",
    "PromptCacheWarmer",
    "bounded_chat_replay",
    "cache_warming_delay_s",
    "cancel_prompt_cache_warming",
    "carry_missed_route",
    "end_turn_prompt_cache_warming",
    "evaluate_economics",
    "get_prompt_cache_warmer",
    "guaranteed_window",
    "has_cache_markers",
    "human_paced",
    "is_replayable",
    "learned_window",
    "prompt_cache_warming_disabled",
    "replay_cap_honoured",
    "replay_output_cap",
    "reset_prompt_cache_session",
    "session_totals_lock",
    "start_in_flight_prompt_cache_warming",
    "start_prompt_cache_warming",
]
