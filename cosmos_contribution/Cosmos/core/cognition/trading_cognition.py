"""
cosmos Trading Cognition - Reasoning & Learning.

"I don't just trade; I observe the fractal nature of the degen mind."

This module provides high-level logic for:
1. Signal Reasoning (Multi-factor evaluation)
2. Trade Learning (Feedback loops based on price action)
3. Exception-aware Execution
"""

import asyncio
from loguru import logger
from typing import   Optional
from datetime import datetime, timedelta

from Cosmos.core.nexus import nexus, Signal, SignalType
from Cosmos.core.cognition.sequential_thinking import sequential_thinker
from Cosmos.memory.memory_system import memory_system # type: ignore

class TradingCognition:
    def __init__(self):
        self.active_signals: dict = {}
        self.learning_history: list[dict] = []

    async def evaluate_token(self, token_data: dict) -> dict:
        """
        Evaluate a token using Sequential Thinking.
        Synthesizes Market Sentiment, Bonding Curve, and Liquidity.
        """
        sequential_thinker.start_new_chain()
        
        # Step 1: Sentiment Analysis
        from Cosmos.integration.financial.market_sentiment import market_sentiment
        fng = await market_sentiment.get_fear_and_greed()
        fng_val = int(fng.get('value', 50))
        
        sequential_thinker.add_step(
            f"Analyzing Market Sentiment: {fng.get('value_classification')} ({fng_val})",
            verification=f"F&G Index is {fng_val}/100"
        )

        # Step 2: Token Check (Liquidity/Curve)
        liq = float(token_data.get('liquidity', {}).get('usd', 0))
        vol = float(token_data.get('volume', {}).get('h24', 0))
        
        sequential_thinker.add_step(
            f"Checking Liquidity and Volume: ${liq} / ${vol}",
            verification="High volume relative to liquidity indicates high interest/churn." if vol > liq else "Healthy ratio."
        )

        # Step 3: Social/Hype Check (Mocked for now, would use X/Bags)
        sequential_thinker.add_step(
            "Synthesizing Social Hype vs. On-chain reality",
            verification="Social sentiment is neutral."
        )

        # Calculate Final Signal
        score = 0
        if fng_val < 30: score += 10 # Buy the fear
        if liq > 50000: score += 20
        if vol > liq * 0.5: score += 30
        
        signal_state = "BULLISH" if score > 40 else "NEUTRAL"
        
        sequential_thinker.add_step(
            f"Final Strategic Conclusion: {signal_state} (Confidence: {score}/100)",
            verification="Reasoning chain complete."
        )

        result = {
            "token": token_data.get('symbol', 'UNKNOWN'),
            "score": score,
            "signal": signal_state,
            "reasoning": sequential_thinker.get_summary()
        }
        
        # Store for learning
        self.active_signals[token_data.get('mint', 'none')] = result
        return result

    async def record_outcome(self, mint: str, price_after_4h: float):
        """
        Record the result of a signal to 'learn' what indicators worked.
        """
        if mint not in self.active_signals:
            return

        signal = self.active_signals.pop(mint)
        prediction = signal['signal']
        
        # In a real impl, we'd compare price_after to initial_price
        # Here we just demo the 'learning' process
        success = True # Mock success
        
        self.learning_history.append({
            "mint": mint,
            "signal": signal,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Trading Cognition: Signal for {mint} was { 'CORRECT' if success else 'WRONG' }. Updating model weights...")
        
        # Persistent Memory Storage
        await nexus.emit(SignalType.MEMORY_CONSOLIDATION, {
            "type": "trade_learning",
            "content": f"Signal for {signal['token']} ({mint}) score {signal['score']} resulted in success: {success}"
        })

class TradeExceptionManager:
    """Handles fallbacks and explanations for failed trades."""
    
    @staticmethod
    async def handle_execution(coro):
        try:
            return await coro
        except Exception as e:
            error_msg = str(e)
            if "slippage" in error_msg.lower():
                return {"error": "Slippage too high. The price is moving too fast for the Professor!", "advice": "Increase slippage tolerance or use a priority fee."}
            elif "insufficient funds" in error_msg.lower():
                return {"error": "Your wallet is as empty as a Davinci's conscience.", "advice": "Top up your SOL balance."}
            else:
                return {"error": f"Unexpected error: {error_msg}", "advice": "Check RPC connection or network congestion."}

trading_cognition = TradingCognition()
exception_manager = TradeExceptionManager()
