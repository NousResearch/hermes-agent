"""Working + episodic + strategic context for the reasoning layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from hermes_trader.config import TraderConfig, load_trader_config
from hermes_trader.market_state import MarketState

if TYPE_CHECKING:
    from hermes_trader.memory.episodes import EpisodeStore


def retrieve_context(
    state: MarketState,
    *,
    limit: int = 3,
    config: Optional[TraderConfig] = None,
    episode_store: Optional["EpisodeStore"] = None,
    strategy_tag: Optional[str] = None,
    liquidity_usd: Optional[float] = None,
    token_address: str = "",
) -> List[dict[str, Any]]:
    """Retrieve strategic rules + top-K similar episodes (untrusted)."""
    from hermes_trader.memory.episodes import EpisodeStore
    from hermes_trader.memory.retrieval import retrieve_similar_episodes
    from hermes_trader.memory.strategic_rules import load_strategic_rules

    cfg = config or load_trader_config()
    retrieval_limit = max(1, getattr(cfg, "memory_retrieval_limit", limit) or limit)

    context: list[dict[str, Any]] = []
    context.extend(load_strategic_rules().to_context_snippets())

    store = episode_store or EpisodeStore()
    episodes = retrieve_similar_episodes(
        state,
        store,
        limit=retrieval_limit,
        strategy_tag=strategy_tag,
        liquidity_usd=liquidity_usd,
        token_address=token_address,
    )
    context.extend(episodes)

    context.append(
        {
            "kind": "working_memory",
            "trust": "trusted",
            "chain": state.chain,
            "portfolio_tokens": len(state.portfolio_tokens),
            "trending_pools": len(state.trending_pools),
            "new_pools": len(state.new_pools),
        }
    )
    return context