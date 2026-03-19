"""
Integration tests for cosmos Financial Scanners.
"""

import pytest
import asyncio
from cosmos.integration.financial.dexscreener import dex_screener
from cosmos.integration.financial.market_sentiment import market_sentiment
from cosmos.integration.financial.memecoin_tracker import memecoin_tracker

@pytest.mark.asyncio
async def test_dexscreener_search():
    """Test searching for a well-known token (SOL)."""
    pairs = await dex_screener.search_pairs("SOL")
    assert isinstance(pairs, list)
    if pairs:
        assert "pairAddress" in pairs[0]

@pytest.mark.asyncio
async def test_market_sentiment():
    """Test fetching market sentiment."""
    fng = await market_sentiment.get_fear_and_greed()
    assert "value" in fng
    assert "value_classification" in fng

@pytest.mark.asyncio
async def test_pump_tracking():
    """Test fetching recent tokens from pump.fun."""
    tokens = await memecoin_tracker.get_pump_new_tokens(limit=5)
    assert isinstance(tokens, list)
    # Note: This might fail if the API is rate-limited or down
    if tokens:
        assert "mint" in tokens[0]
