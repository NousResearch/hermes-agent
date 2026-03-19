"""
cosmos TradFi Module - Stocks & Forex
-----------------------------------------

"Stocks, bonds, golden parachutes... I want it all!"

Integrates with Alpha Vantage / Yahoo Finance / OANDA for traditional market data.
"""

import aiohttp
import os
import json
from typing import Dict, List, Optional
from loguru import logger

class TradFiAgent:
    def __init__(self, alpha_vantage_key: str = None):
        self.av_key = alpha_vantage_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        self.yfinance_available = False
        try:
            import yfinance as yf
            self.yfinance_available = True
        except ImportError:
            logger.warning("yfinance not installed. Falling back to API/Mock.")

    async def get_stock_quote(self, symbol: str) -> Dict:
        """Get real-time stock quote."""
        logger.info(f"TradFi: Fetching quote for {symbol}")
        
        # 1. Try yfinance (free, robust)
        if self.yfinance_available:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                return {
                    "symbol": symbol,
                    "price": info.last_price,
                    "prev_close": info.previous_close,
                    "change_pct": ((info.last_price - info.previous_close) / info.previous_close) * 100,
                    "source": "yfinance"
                }
            except Exception as e:
                logger.error(f"yfinance failed: {e}")
        
        # 2. Fallback to Alpha Vantage
        if self.av_key:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.av_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    quote = data.get("Global Quote", {})
                    return {
                        "symbol": symbol,
                        "price": float(quote.get("05. price", 0)),
                        "change_pct": quote.get("10. change percent", "0%"),
                        "source": "AlphaVantage"
                    }
                    
        return {"error": "No data source available"}

    async def get_forex_rate(self, from_currency: str, to_currency: str) -> Dict:
        """Get FX rate."""
        symbol = f"{from_currency}{to_currency}=X"
        logger.info(f"TradFi: FX Rate {from_currency}/{to_currency}")
        
        if self.yfinance_available:
            import yfinance as yf
            data = yf.Ticker(symbol)
            return {
                "pair": f"{from_currency}/{to_currency}",
                "rate": data.fast_info.last_price,
                "source": "yfinance"
            }
        return {"error": "yfinance required for FX"}

    async def analyze_sentiment(self, symbol: str) -> str:
        """Basic news sentiment analysis (Mock)."""
        # In a real impl, we'd scrape news and feed to LLM
        return "BULLISH (Based on recent positive earnings surprise)"

tradfi = TradFiAgent()
